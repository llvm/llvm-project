//===- CtxInstrProfiling.cpp - contextual instrumented PGO ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CtxInstrProfiling.h"
#include "RootAutoDetector.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_atomic_clang.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_dense_map.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_thread_safety.h"
#include "sanitizer_common/sanitizer_vector.h"

#include <assert.h>

using namespace __ctx_profile;

namespace {
// Keep track of all the context roots we actually saw, so we can then traverse
// them when the user asks for the profile in __llvm_ctx_profile_fetch
__sanitizer::SpinMutex AllContextsMutex;
SANITIZER_GUARDED_BY(AllContextsMutex)
__sanitizer::Vector<ContextRoot *> AllContextRoots;

__sanitizer::atomic_uintptr_t AllFunctionsData = {};

// Keep all the functions for which we collect a flat profile in a linked list.
__sanitizer::SpinMutex FlatCtxArenaMutex;
SANITIZER_GUARDED_BY(FlatCtxArenaMutex)
Arena *FlatCtxArenaHead = nullptr;
SANITIZER_GUARDED_BY(FlatCtxArenaMutex)
Arena *FlatCtxArena = nullptr;

// Set to true when we enter a root, and false when we exit - regardless if this
// thread collects a contextual profile for that root.
__thread bool IsUnderContext = false;
__sanitizer::atomic_uint8_t ProfilingStarted = {};

__sanitizer::atomic_uintptr_t RootDetector = {};
RootAutoDetector *getRootDetector() {
  return reinterpret_cast<RootAutoDetector *>(
      __sanitizer::atomic_load_relaxed(&RootDetector));
}

// utility to taint a pointer by setting the LSB. There is an assumption
// throughout that the addresses of contexts are even (really, they should be
// align(8), but "even"-ness is the minimum assumption)
// "scratch contexts" are buffers that we return in certain cases - they are
// large enough to allow for memory safe counter access, but they don't link
// subcontexts below them (the runtime recognizes them and enforces that)
ContextNode *markAsScratch(const ContextNode *Ctx) {
  return reinterpret_cast<ContextNode *>(reinterpret_cast<uint64_t>(Ctx) | 1);
}

// Used when getting the data from TLS. We don't *really* need to reset, but
// it's a simpler system if we do.
template <typename T> inline T consume(T &V) {
  auto R = V;
  V = {0};
  return R;
}

// We allocate at least kBuffSize Arena pages. The scratch buffer is also that
// large.
constexpr size_t kPower = 20;
constexpr size_t kBuffSize = 1 << kPower;

// Highly unlikely we need more than kBuffSize for a context.
size_t getArenaAllocSize(size_t Needed) {
  if (Needed >= kBuffSize)
    return 2 * Needed;
  return kBuffSize;
}

// verify the structural integrity of the context
bool validate(const ContextRoot *Root) {
  // all contexts should be laid out in some arena page. Go over each arena
  // allocated for this Root, and jump over contained contexts based on
  // self-reported sizes.
  __sanitizer::DenseMap<uint64_t, bool> ContextStartAddrs;
  for (const auto *Mem = Root->FirstMemBlock; Mem; Mem = Mem->next()) {
    const auto *Pos = Mem->start();
    while (Pos < Mem->pos()) {
      const auto *Ctx = reinterpret_cast<const ContextNode *>(Pos);
      if (!ContextStartAddrs.insert({reinterpret_cast<uint64_t>(Ctx), true})
               .second)
        return false;
      Pos += Ctx->size();
    }
  }

  // Now traverse the contexts again the same way, but validate all nonull
  // subcontext addresses appear in the set computed above.
  for (const auto *Mem = Root->FirstMemBlock; Mem; Mem = Mem->next()) {
    const auto *Pos = Mem->start();
    while (Pos < Mem->pos()) {
      const auto *Ctx = reinterpret_cast<const ContextNode *>(Pos);
      for (uint32_t I = 0; I < Ctx->callsites_size(); ++I)
        for (auto *Sub = Ctx->subContexts()[I]; Sub; Sub = Sub->next())
          if (!ContextStartAddrs.find(reinterpret_cast<uint64_t>(Sub)))
            return false;

      Pos += Ctx->size();
    }
  }
  return true;
}

inline ContextNode *allocContextNode(char *Place, GUID Guid,
                                     uint32_t NumCounters,
                                     uint32_t NumCallsites,
                                     ContextNode *Next = nullptr) {
  assert(reinterpret_cast<uint64_t>(Place) % ExpectedAlignment == 0);
  return new (Place) ContextNode(Guid, NumCounters, NumCallsites, Next);
}

void resetContextNode(ContextNode &Node) {
  // FIXME(mtrofin): this is std::memset, which we can probably use if we
  // drop/reduce the dependency on sanitizer_common.
  for (uint32_t I = 0; I < Node.counters_size(); ++I)
    Node.counters()[I] = 0;
  for (uint32_t I = 0; I < Node.callsites_size(); ++I)
    for (auto *Next = Node.subContexts()[I]; Next; Next = Next->next())
      resetContextNode(*Next);
}

ContextNode *onContextEnter(ContextNode &Node) {
  ++Node.counters()[0];
  return &Node;
}

} // namespace

// the scratch buffer - what we give when we can't produce a real context (the
// scratch isn't "real" in that it's expected to be clobbered carelessly - we
// don't read it). The other important thing is that the callees from a scratch
// context also get a scratch context.
// Eventually this can be replaced with per-function buffers, a'la the typical
// (flat) instrumented FDO buffers. The clobbering aspect won't apply there, but
// the part about determining the nature of the subcontexts does.
__thread char __Buffer[kBuffSize] = {0};

#define TheScratchContext                                                      \
  markAsScratch(reinterpret_cast<ContextNode *>(__Buffer))

// init the TLSes
__thread void *volatile __llvm_ctx_profile_expected_callee[2] = {nullptr,
                                                                 nullptr};
__thread ContextNode **volatile __llvm_ctx_profile_callsite[2] = {0, 0};

__thread ContextRoot *volatile __llvm_ctx_profile_current_context_root =
    nullptr;

Arena::Arena(uint32_t Size) : Size(Size) {
  __sanitizer::internal_memset(start(), 0, Size);
}

// FIXME(mtrofin): use malloc / mmap instead of sanitizer common APIs to reduce
// the dependency on the latter.
Arena *Arena::allocateNewArena(size_t Size, Arena *Prev) {
  assert(!Prev || Prev->Next == nullptr);
  Arena *NewArena = new (__sanitizer::InternalAlloc(
      Size + sizeof(Arena), /*cache=*/nullptr, /*alignment=*/ExpectedAlignment))
      Arena(Size);
  if (Prev)
    Prev->Next = NewArena;
  return NewArena;
}

void Arena::freeArenaList(Arena *&A) {
  assert(A);
  for (auto *I = A; I != nullptr;) {
    auto *Current = I;
    I = I->Next;
    __sanitizer::InternalFree(Current);
  }
  A = nullptr;
}

// If this is the first time we hit a callsite with this (Guid) particular
// callee, we need to allocate.
ContextNode *getCallsiteSlow(GUID Guid, ContextNode **InsertionPoint,
                             uint32_t NumCounters, uint32_t NumCallsites) {
  auto AllocSize = ContextNode::getAllocSize(NumCounters, NumCallsites);
  auto *Mem = __llvm_ctx_profile_current_context_root->CurrentMem;
  char *AllocPlace = Mem->tryBumpAllocate(AllocSize);
  if (!AllocPlace) {
    // if we failed to allocate on the current arena, allocate a new arena,
    // and place it on __llvm_ctx_profile_current_context_root->CurrentMem so we
    // find it from now on for other cases when we need to getCallsiteSlow.
    // Note that allocateNewArena will link the allocated memory in the list of
    // Arenas.
    __llvm_ctx_profile_current_context_root->CurrentMem = Mem =
        Mem->allocateNewArena(getArenaAllocSize(AllocSize), Mem);
    AllocPlace = Mem->tryBumpAllocate(AllocSize);
  }
  auto *Ret = allocContextNode(AllocPlace, Guid, NumCounters, NumCallsites,
                               *InsertionPoint);
  *InsertionPoint = Ret;
  return Ret;
}

ContextNode *getFlatProfile(FunctionData &Data, void *Callee, GUID Guid,
                            uint32_t NumCounters) {
  if (ContextNode *Existing = Data.FlatCtx)
    return Existing;
  {
    // We could instead try to take the lock and, if that fails, return
    // TheScratchContext. But that could leave message pump loops more sparsely
    // profiled than everything else. Maybe that doesn't matter, and we can
    // optimize this later.
    __sanitizer::GenericScopedLock<__sanitizer::StaticSpinMutex> L(&Data.Mutex);
    if (ContextNode *Existing = Data.FlatCtx)
      return Existing;

    auto NeededSize = ContextNode::getAllocSize(NumCounters, 0);
    char *AllocBuff = nullptr;
    {
      __sanitizer::GenericScopedLock<__sanitizer::SpinMutex> FL(
          &FlatCtxArenaMutex);
      if (FlatCtxArena)
        AllocBuff = FlatCtxArena->tryBumpAllocate(NeededSize);
      if (!AllocBuff) {
        FlatCtxArena = Arena::allocateNewArena(getArenaAllocSize(NeededSize),
                                               FlatCtxArena);
        AllocBuff = FlatCtxArena->tryBumpAllocate(NeededSize);
      }
      if (!FlatCtxArenaHead)
        FlatCtxArenaHead = FlatCtxArena;
    }
    auto *Ret = allocContextNode(AllocBuff, Guid, NumCounters, 0);
    Data.FlatCtx = Ret;

    Data.EntryAddress = Callee;
    Data.Next = reinterpret_cast<FunctionData *>(
        __sanitizer::atomic_load_relaxed(&AllFunctionsData));
    while (!__sanitizer::atomic_compare_exchange_strong(
        &AllFunctionsData, reinterpret_cast<uintptr_t *>(&Data.Next),
        reinterpret_cast<uintptr_t>(&Data),
        __sanitizer::memory_order_release)) {
    }
  }

  return Data.FlatCtx;
}

// This should be called once for a Root. Allocate the first arena, set up the
// first context.
void setupContext(ContextRoot *Root, GUID Guid, uint32_t NumCounters,
                  uint32_t NumCallsites) {
  __sanitizer::GenericScopedLock<__sanitizer::SpinMutex> Lock(
      &AllContextsMutex);
  // Re-check - we got here without having had taken a lock.
  if (Root->FirstMemBlock)
    return;
  const auto Needed = ContextNode::getAllocSize(NumCounters, NumCallsites);
  auto *M = Arena::allocateNewArena(getArenaAllocSize(Needed));
  Root->FirstMemBlock = M;
  Root->CurrentMem = M;
  Root->FirstNode = allocContextNode(M->tryBumpAllocate(Needed), Guid,
                                     NumCounters, NumCallsites);
  AllContextRoots.PushBack(Root);
}

ContextRoot *FunctionData::getOrAllocateContextRoot() {
  auto *Root = CtxRoot;
  if (Root)
    return Root;
  __sanitizer::GenericScopedLock<__sanitizer::StaticSpinMutex> L(&Mutex);
  Root = CtxRoot;
  if (!Root) {
    Root = new (__sanitizer::InternalAlloc(sizeof(ContextRoot))) ContextRoot();
    CtxRoot = Root;
  }

  assert(Root);
  return Root;
}

ContextNode *tryStartContextGivenRoot(ContextRoot *Root, GUID Guid,
                                      uint32_t Counters, uint32_t Callsites)
    SANITIZER_NO_THREAD_SAFETY_ANALYSIS {
  IsUnderContext = true;
  __sanitizer::atomic_fetch_add(&Root->TotalEntries, 1,
                                __sanitizer::memory_order_relaxed);
  if (!Root->FirstMemBlock) {
    setupContext(Root, Guid, Counters, Callsites);
  }
  if (Root->Taken.TryLock()) {
    __llvm_ctx_profile_current_context_root = Root;
    onContextEnter(*Root->FirstNode);
    return Root->FirstNode;
  }
  // If this thread couldn't take the lock, return scratch context.
  __llvm_ctx_profile_current_context_root = nullptr;
  return TheScratchContext;
}

ContextNode *getUnhandledContext(FunctionData &Data, void *Callee, GUID Guid,
                                 uint32_t NumCounters, uint32_t NumCallsites,
                                 ContextRoot *CtxRoot) {

  // 1) if we are currently collecting a contextual profile, fetch a ContextNode
  // in the `Unhandled` set. We want to do this regardless of `ProfilingStarted`
  // to (hopefully) offset the penalty of creating these contexts to before
  // profiling.
  //
  // 2) if we are under a root (regardless if this thread is collecting or not a
  // contextual profile for that root), do not collect a flat profile. We want
  // to keep flat profiles only for activations that can't happen under a root,
  // to avoid confusing profiles. We can, for example, combine flattened and
  // flat profiles meaningfully, as we wouldn't double-count anything.
  //
  // 3) to avoid lengthy startup, don't bother with flat profiles until the
  // profiling has started. We would reset them anyway when profiling starts.
  // HOWEVER. This does lose profiling for message pumps: those functions are
  // entered once and never exit. They should be assumed to be entered before
  // profiling starts - because profiling should start after the server is up
  // and running (which is equivalent to "message pumps are set up").
  if (!CtxRoot) {
    if (auto *RAD = getRootDetector())
      RAD->sample();
    else if (auto *CR = Data.CtxRoot)
      return tryStartContextGivenRoot(CR, Guid, NumCounters, NumCallsites);
    if (IsUnderContext || !__sanitizer::atomic_load_relaxed(&ProfilingStarted))
      return TheScratchContext;
    else
      return markAsScratch(
          onContextEnter(*getFlatProfile(Data, Callee, Guid, NumCounters)));
  }
  auto [Iter, Ins] = CtxRoot->Unhandled.insert({Guid, nullptr});
  if (Ins)
    Iter->second = getCallsiteSlow(Guid, &CtxRoot->FirstUnhandledCalleeNode,
                                   NumCounters, 0);
  return markAsScratch(onContextEnter(*Iter->second));
}

ContextNode *__llvm_ctx_profile_get_context(FunctionData *Data, void *Callee,
                                            GUID Guid, uint32_t NumCounters,
                                            uint32_t NumCallsites) {
  auto *CtxRoot = __llvm_ctx_profile_current_context_root;
  // fast "out" if we're not even doing contextual collection.
  if (!CtxRoot)
    return getUnhandledContext(*Data, Callee, Guid, NumCounters, NumCallsites,
                               nullptr);

  // also fast "out" if the caller is scratch. We can see if it's scratch by
  // looking at the interior pointer into the subcontexts vector that the caller
  // provided, which, if the context is scratch, so is that interior pointer
  // (because all the address calculations are using even values. Or more
  // precisely, aligned - 8 values)
  auto **CallsiteContext = consume(__llvm_ctx_profile_callsite[0]);
  if (!CallsiteContext || isScratch(CallsiteContext))
    return getUnhandledContext(*Data, Callee, Guid, NumCounters, NumCallsites,
                               CtxRoot);

  // if the callee isn't the expected one, return scratch.
  // Signal handler(s) could have been invoked at any point in the execution.
  // Should that have happened, and had it (the handler) be built with
  // instrumentation, its __llvm_ctx_profile_get_context would have failed here.
  // Its sub call graph would have then populated
  // __llvm_ctx_profile_{expected_callee | callsite} at index 1.
  // The normal call graph may be impacted in that, if the signal handler
  // happened somewhere before we read the TLS here, we'd see the TLS reset and
  // we'd also fail here. That would just mean we would loose counter values for
  // the normal subgraph, this time around. That should be very unlikely, but if
  // it happens too frequently, we should be able to detect discrepancies in
  // entry counts (caller-callee). At the moment, the design goes on the
  // assumption that is so unfrequent, though, that it's not worth doing more
  // for that case.
  auto *ExpectedCallee = consume(__llvm_ctx_profile_expected_callee[0]);
  if (ExpectedCallee != Callee)
    return getUnhandledContext(*Data, Callee, Guid, NumCounters, NumCallsites,
                               CtxRoot);

  auto *Callsite = *CallsiteContext;
  // in the case of indirect calls, we will have all seen targets forming a
  // linked list here. Find the one corresponding to this callee.
  while (Callsite && Callsite->guid() != Guid) {
    Callsite = Callsite->next();
  }
  auto *Ret = Callsite ? Callsite
                       : getCallsiteSlow(Guid, CallsiteContext, NumCounters,
                                         NumCallsites);
  if (Ret->callsites_size() != NumCallsites ||
      Ret->counters_size() != NumCounters)
    __sanitizer::Printf("[ctxprof] Returned ctx differs from what's asked: "
                        "Context: %p, Asked: %lu %u %u, Got: %lu %u %u \n",
                        reinterpret_cast<void *>(Ret), Guid, NumCallsites,
                        NumCounters, Ret->guid(), Ret->callsites_size(),
                        Ret->counters_size());
  onContextEnter(*Ret);
  return Ret;
}

ContextNode *__llvm_ctx_profile_start_context(FunctionData *FData, GUID Guid,
                                              uint32_t Counters,
                                              uint32_t Callsites) {

  return tryStartContextGivenRoot(FData->getOrAllocateContextRoot(), Guid,
                                  Counters, Callsites);
}

void __llvm_ctx_profile_release_context(FunctionData *FData)
    SANITIZER_NO_THREAD_SAFETY_ANALYSIS {
  const auto *CurrentRoot = __llvm_ctx_profile_current_context_root;
  if (!CurrentRoot || FData->CtxRoot != CurrentRoot)
    return;
  IsUnderContext = false;
  assert(FData->CtxRoot);
  __llvm_ctx_profile_current_context_root = nullptr;
  FData->CtxRoot->Taken.Unlock();
}

void __llvm_ctx_profile_start_collection(unsigned AutodetectDuration) {
  size_t NumMemUnits = 0;
  __sanitizer::GenericScopedLock<__sanitizer::SpinMutex> Lock(
      &AllContextsMutex);
  for (uint32_t I = 0; I < AllContextRoots.Size(); ++I) {
    auto *Root = AllContextRoots[I];
    __sanitizer::GenericScopedLock<__sanitizer::StaticSpinMutex> Lock(
        &Root->Taken);
    for (auto *Mem = Root->FirstMemBlock; Mem; Mem = Mem->next())
      ++NumMemUnits;

    resetContextNode(*Root->FirstNode);
    if (Root->FirstUnhandledCalleeNode)
      resetContextNode(*Root->FirstUnhandledCalleeNode);
    __sanitizer::atomic_store_relaxed(&Root->TotalEntries, 0);
  }
  if (AutodetectDuration) {
    // we leak RD intentionally. Knowing when to free it is tricky, there's a
    // race condition with functions observing the `RootDectector` as non-null.
    // This can be addressed but the alternatives have some added complexity and
    // it's not (yet) worth it.
    auto *RD = new (__sanitizer::InternalAlloc(sizeof(RootAutoDetector)))
        RootAutoDetector(AllFunctionsData, RootDetector, AutodetectDuration);
    RD->start();
  } else {
    __sanitizer::Printf("[ctxprof] Initial NumMemUnits: %zu \n", NumMemUnits);
  }
  __sanitizer::atomic_store_relaxed(&ProfilingStarted, true);
}

bool __llvm_ctx_profile_fetch(ProfileWriter &Writer) {
  __sanitizer::atomic_store_relaxed(&ProfilingStarted, false);
  if (auto *RD = getRootDetector()) {
    __sanitizer::Printf("[ctxprof] Expected the root autodetector to have "
                        "finished well before attempting to fetch a context");
    RD->join();
  }

  __sanitizer::GenericScopedLock<__sanitizer::SpinMutex> Lock(
      &AllContextsMutex);

  Writer.startContextSection();
  for (int I = 0, E = AllContextRoots.Size(); I < E; ++I) {
    auto *Root = AllContextRoots[I];
    __sanitizer::GenericScopedLock<__sanitizer::StaticSpinMutex> TakenLock(
        &Root->Taken);
    if (!validate(Root)) {
      __sanitizer::Printf("[ctxprof] Contextual Profile is %s\n", "invalid");
      return false;
    }
    Writer.writeContextual(
        *Root->FirstNode, Root->FirstUnhandledCalleeNode,
        __sanitizer::atomic_load_relaxed(&Root->TotalEntries));
  }
  Writer.endContextSection();
  Writer.startFlatSection();
  // The list progresses behind the head, so taking this snapshot allows the
  // list to grow concurrently without causing a race condition with our
  // traversing it.
  const auto *Pos = reinterpret_cast<const FunctionData *>(
      __sanitizer::atomic_load_relaxed(&AllFunctionsData));
  for (; Pos; Pos = Pos->Next)
    if (!Pos->CtxRoot)
      Writer.writeFlat(Pos->FlatCtx->guid(), Pos->FlatCtx->counters(),
                       Pos->FlatCtx->counters_size());
  Writer.endFlatSection();
  return true;
}

void __llvm_ctx_profile_free() {
  __sanitizer::atomic_store_relaxed(&ProfilingStarted, false);
  {
    __sanitizer::GenericScopedLock<__sanitizer::SpinMutex> Lock(
        &AllContextsMutex);
    for (int I = 0, E = AllContextRoots.Size(); I < E; ++I)
      for (auto *A = AllContextRoots[I]->FirstMemBlock; A;) {
        auto *C = A;
        A = A->next();
        __sanitizer::InternalFree(C);
      }
    AllContextRoots.Reset();
  }
  __sanitizer::atomic_store_relaxed(&AllFunctionsData, 0U);
  {
    __sanitizer::GenericScopedLock<__sanitizer::SpinMutex> Lock(
        &FlatCtxArenaMutex);
    FlatCtxArena = nullptr;
    for (auto *A = FlatCtxArenaHead; A;) {
      auto *C = A;
      A = C->next();
      __sanitizer::InternalFree(C);
    }

    FlatCtxArenaHead = nullptr;
  }
}
