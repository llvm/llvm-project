//===- CtxInstrProfiling.cpp - contextual instrumented PGO ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CtxInstrProfiling.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_dense_map.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_thread_safety.h"
#include "sanitizer_common/sanitizer_vector.h"

#include <assert.h>

using namespace __ctx_profile;

namespace {
__sanitizer::SpinMutex AllContextsMutex;
SANITIZER_GUARDED_BY(AllContextsMutex)
__sanitizer::Vector<ContextRoot *> AllContextRoots;

ContextNode *markAsScratch(const ContextNode *Ctx) {
  return reinterpret_cast<ContextNode *>(reinterpret_cast<uint64_t>(Ctx) | 1);
}

template <typename T> T consume(T &V) {
  auto R = V;
  V = {0};
  return R;
}

constexpr size_t kPower = 20;
constexpr size_t kBuffSize = 1 << kPower;

size_t getArenaAllocSize(size_t Needed) {
  if (Needed >= kBuffSize)
    return 2 * Needed;
  return kBuffSize;
}

bool validate(const ContextRoot *Root) {
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
} // namespace

__thread char __Buffer[kBuffSize] = {0};

#define TheScratchContext                                                      \
  markAsScratch(reinterpret_cast<ContextNode *>(__Buffer))
__thread void *volatile __llvm_ctx_profile_expected_callee[2] = {nullptr,
                                                                 nullptr};
__thread ContextNode **volatile __llvm_ctx_profile_callsite[2] = {0, 0};

__thread ContextRoot *volatile __llvm_ctx_profile_current_context_root =
    nullptr;

// FIXME(mtrofin): use malloc / mmap instead of sanitizer common APIs to reduce
// the dependency on the latter.
Arena *Arena::allocateNewArena(size_t Size, Arena *Prev) {
  assert(!Prev || Prev->Next == nullptr);
  Arena *NewArena =
      new (__sanitizer::InternalAlloc(Size + sizeof(Arena))) Arena(Size);
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

inline ContextNode *ContextNode::alloc(char *Place, GUID Guid,
                                       uint32_t NrCounters,
                                       uint32_t NrCallsites,
                                       ContextNode *Next) {
  return new (Place) ContextNode(Guid, NrCounters, NrCallsites, Next);
}

void ContextNode::reset() {
  for (uint32_t I = 0; I < NrCounters; ++I)
    counters()[I] = 0;
  for (uint32_t I = 0; I < NrCallsites; ++I)
    for (auto *Next = subContexts()[I]; Next; Next = Next->Next)
      Next->reset();
}

ContextNode *getCallsiteSlow(uint64_t Guid, ContextNode **InsertionPoint,
                             uint32_t NrCounters, uint32_t NrCallsites) {
  auto AllocSize = ContextNode::getAllocSize(NrCounters, NrCallsites);
  auto *Mem = __llvm_ctx_profile_current_context_root->CurrentMem;
  char *AllocPlace = Mem->tryBumpAllocate(AllocSize);
  if (!AllocPlace) {
    __llvm_ctx_profile_current_context_root->CurrentMem = Mem =
        Mem->allocateNewArena(getArenaAllocSize(AllocSize), Mem);
  }
  auto *Ret = ContextNode::alloc(AllocPlace, Guid, NrCounters, NrCallsites,
                                 *InsertionPoint);
  *InsertionPoint = Ret;
  return Ret;
}

ContextNode *__llvm_ctx_profile_get_context(void *Callee, GUID Guid,
                                            uint32_t NrCounters,
                                            uint32_t NrCallsites) {
  if (!__llvm_ctx_profile_current_context_root) {
    return TheScratchContext;
  }
  auto **CallsiteContext = consume(__llvm_ctx_profile_callsite[0]);
  if (!CallsiteContext || isScratch(*CallsiteContext))
    return TheScratchContext;

  auto *ExpectedCallee = consume(__llvm_ctx_profile_expected_callee[0]);
  if (ExpectedCallee != Callee)
    return TheScratchContext;

  auto *Callsite = *CallsiteContext;
  while (Callsite && Callsite->guid() != Guid) {
    Callsite = Callsite->next();
  }
  auto *Ret = Callsite ? Callsite
                       : getCallsiteSlow(Guid, CallsiteContext, NrCounters,
                                         NrCallsites);
  if (Ret->callsites_size() != NrCallsites ||
      Ret->counters_size() != NrCounters)
    __sanitizer::Printf("[ctxprof] Returned ctx differs from what's asked: "
                        "Context: %p, Asked: %lu %u %u, Got: %lu %u %u \n",
                        Ret, Guid, NrCallsites, NrCounters, Ret->guid(),
                        Ret->callsites_size(), Ret->counters_size());
  Ret->onEntry();
  return Ret;
}

void setupContext(ContextRoot *Root, GUID Guid, uint32_t NrCounters,
                  uint32_t NrCallsites) {
  __sanitizer::GenericScopedLock<__sanitizer::SpinMutex> Lock(
      &AllContextsMutex);
  // Re-check - we got here without having had taken a lock.
  if (Root->FirstMemBlock)
    return;
  const auto Needed = ContextNode::getAllocSize(NrCounters, NrCallsites);
  auto *M = Arena::allocateNewArena(getArenaAllocSize(Needed));
  Root->FirstMemBlock = M;
  Root->CurrentMem = M;
  Root->FirstNode = ContextNode::alloc(M->tryBumpAllocate(Needed), Guid,
                                       NrCounters, NrCallsites);
  AllContextRoots.PushBack(Root);
}

ContextNode *__llvm_ctx_profile_start_context(
    ContextRoot *Root, GUID Guid, uint32_t Counters,
    uint32_t Callsites) SANITIZER_NO_THREAD_SAFETY_ANALYSIS {
  if (!Root->FirstMemBlock) {
    setupContext(Root, Guid, Counters, Callsites);
  }
  if (Root->Taken.TryLock()) {
    __llvm_ctx_profile_current_context_root = Root;
    Root->FirstNode->onEntry();
    return Root->FirstNode;
  }
  __llvm_ctx_profile_current_context_root = nullptr;
  return TheScratchContext;
}

void __llvm_ctx_profile_release_context(ContextRoot *Root)
    SANITIZER_NO_THREAD_SAFETY_ANALYSIS {
  if (__llvm_ctx_profile_current_context_root) {
    __llvm_ctx_profile_current_context_root = nullptr;
    Root->Taken.Unlock();
  }
}

void __llvm_ctx_profile_start_collection() {
  size_t NrMemUnits = 0;
  __sanitizer::GenericScopedLock<__sanitizer::SpinMutex> Lock(
      &AllContextsMutex);
  for (uint32_t I = 0; I < AllContextRoots.Size(); ++I) {
    auto *Root = AllContextRoots[I];
    __sanitizer::GenericScopedLock<__sanitizer::StaticSpinMutex> Lock(
        &Root->Taken);
    for (auto *Mem = Root->FirstMemBlock; Mem; Mem = Mem->next())
      ++NrMemUnits;

    Root->FirstNode->reset();
  }
  __sanitizer::Printf("[ctxprof] Initial NrMemUnits: %zu \n", NrMemUnits);
}

bool __llvm_ctx_profile_fetch(
    void *Data, bool (*Writer)(void *W, const __ctx_profile::ContextNode &)) {
  assert(Writer);
  __sanitizer::GenericScopedLock<__sanitizer::SpinMutex> Lock(
      &AllContextsMutex);

  for (int I = 0, E = AllContextRoots.Size(); I < E; ++I) {
    auto *Root = AllContextRoots[I];
    __sanitizer::GenericScopedLock<__sanitizer::StaticSpinMutex> TakenLock(
        &Root->Taken);
    if (!validate(Root)) {
      __sanitizer::Printf("[ctxprof] Contextual Profile is %s\n", "invalid");
      return false;
    }
    if (!Writer(Data, *Root->FirstNode))
      return false;
  }
  return true;
}

void __llvm_ctx_profile_free() {
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
