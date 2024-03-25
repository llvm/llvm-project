//===- InstrProfilingContextual.cpp - PGO runtime initialization ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstrProfiling.h"

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_atomic_clang.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_dense_map.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_thread_safety.h"
#include "sanitizer_common/sanitizer_vector.h"

#include "InstrProfilingContextual.h"

using namespace __profile;

Arena *Arena::allocate(size_t Size, Arena *Prev) {
  Arena *NewArena =
      new (__sanitizer::InternalAlloc(Size + sizeof(Arena))) Arena(Size);
  if (Prev)
    Prev->Next = NewArena;
  return NewArena;
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

namespace {
__sanitizer::SpinMutex AllContextsMutex;
SANITIZER_GUARDED_BY(AllContextsMutex)
__sanitizer::Vector<ContextRoot *> AllContextRoots;

ContextNode * markAsScratch(const ContextNode* Ctx) {
  return reinterpret_cast<ContextNode*>(reinterpret_cast<uint64_t>(Ctx) | 1);
}

template<typename T>
T consume(T& V) {
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

extern "C" {

__thread char __Buffer[kBuffSize] = {0};

#define TheScratchContext                                                      \
  markAsScratch(reinterpret_cast<ContextNode *>(__Buffer))
__thread void *volatile __llvm_instrprof_expected_callee[2] = {nullptr, nullptr};
__thread ContextNode **volatile __llvm_instrprof_callsite[2] = {0, 0};

COMPILER_RT_VISIBILITY __thread ContextRoot
    *volatile __llvm_instrprof_current_context_root = nullptr;

COMPILER_RT_VISIBILITY ContextNode *
__llvm_instprof_slow_get_callsite(uint64_t Guid, ContextNode **InsertionPoint,
                                  uint32_t NrCounters, uint32_t NrCallsites) {
  auto AllocSize = ContextNode::getAllocSize(NrCounters, NrCallsites);
  auto *Mem = __llvm_instrprof_current_context_root->CurrentMem;
  char* AllocPlace = Mem->tryAllocate(AllocSize);
  if (!AllocPlace) {
    __llvm_instrprof_current_context_root->CurrentMem = Mem =
        Mem->allocate(getArenaAllocSize(AllocSize), Mem);
  }
  auto *Ret = ContextNode::alloc(AllocPlace, Guid, NrCounters, NrCallsites,
                                 *InsertionPoint);
  *InsertionPoint = Ret;
  return Ret;
}

COMPILER_RT_VISIBILITY ContextNode *
__llvm_instrprof_get_context(void *Callee, GUID Guid, uint32_t NrCounters,
                            uint32_t NrCallsites) {
  if (!__llvm_instrprof_current_context_root) {
    return TheScratchContext;
  }
  auto **CallsiteContext = consume(__llvm_instrprof_callsite[0]);
  if (!CallsiteContext || isScratch(*CallsiteContext))
    return TheScratchContext;

  auto *ExpectedCallee = consume(__llvm_instrprof_expected_callee[0]);
  if (ExpectedCallee != Callee)
    return TheScratchContext;

  auto *Callsite = *CallsiteContext;
  while (Callsite && Callsite->guid() != Guid) {
    Callsite = Callsite->next();
  }
  auto *Ret = Callsite ? Callsite
                       : __llvm_instprof_slow_get_callsite(
                             Guid, CallsiteContext, NrCounters, NrCallsites);
  if (Ret->callsites_size() != NrCallsites || Ret->counters_size() != NrCounters)
    __sanitizer::Printf("[ctxprof] Returned ctx differs from what's asked: "
                        "Context: %p, Asked: %lu %u %u, Got: %lu %u %u \n",
                        Ret, Guid, NrCallsites, NrCounters, Ret->guid(),
                        Ret->callsites_size(), Ret->counters_size());
  Ret->onEntry();
  return Ret;
}

COMPILER_RT_VISIBILITY void
__llvm_instprof_setup_context(ContextRoot *Root, GUID Guid, uint32_t NrCounters,
                              uint32_t NrCallsites) {
  __sanitizer::GenericScopedLock<__sanitizer::SpinMutex> Lock(
      &AllContextsMutex);
  // Re-check - we got here without having had taken a lock.
  if (Root->FirstMemBlock)
    return;
  const auto Needed = ContextNode::getAllocSize(NrCounters, NrCallsites);
  auto *M = Arena::allocate(getArenaAllocSize(Needed));
  Root->FirstMemBlock = M;
  Root->CurrentMem = M;
  Root->FirstNode =
      ContextNode::alloc(M->tryAllocate(Needed), Guid, NrCounters, NrCallsites);
  AllContextRoots.PushBack(Root);
}

COMPILER_RT_VISIBILITY ContextNode *__llvm_instrprof_start_context(
    ContextRoot *Root, GUID Guid, uint32_t Counters,
    uint32_t Callsites) SANITIZER_NO_THREAD_SAFETY_ANALYSIS {
  if (!Root->FirstMemBlock) {
    __llvm_instprof_setup_context(Root, Guid, Counters, Callsites);
  }
  if (Root->Taken.TryLock()) {
    __llvm_instrprof_current_context_root = Root;
    Root->FirstNode->onEntry();
    return Root->FirstNode;
  }
  __llvm_instrprof_current_context_root = nullptr;
  return TheScratchContext;
}

COMPILER_RT_VISIBILITY void __llvm_instrprof_release_context(ContextRoot *Root)
    SANITIZER_NO_THREAD_SAFETY_ANALYSIS {
  if (__llvm_instrprof_current_context_root) {
    __llvm_instrprof_current_context_root = nullptr;
    Root->Taken.Unlock();
  }
}

COMPILER_RT_VISIBILITY void __llvm_profile_reset_ctx_counters(void) {
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

COMPILER_RT_VISIBILITY
int __llvm_ctx_profile_dump(const char* Filename) {
  __sanitizer::Printf("[ctxprof] Start Dump\n");
  __sanitizer::GenericScopedLock<__sanitizer::SpinMutex> Lock(
      &AllContextsMutex);

  for (int I = 0, E = AllContextRoots.Size(); I < E; ++I) {
    auto *Root = AllContextRoots[I];
    __sanitizer::GenericScopedLock<__sanitizer::StaticSpinMutex> TakenLock(
        &Root->Taken);
    if (!validate(Root)) {
      PROF_ERR("Contextual Profile is %s\n", "invalid");
      return 1;
    }
  }

  if (!Filename) {
    PROF_ERR("Failed to write file : %s\n", "Filename not set");
    return -1;
  }
  FILE *F = fopen(Filename, "w");
  if (!F) {
    PROF_ERR("Failed to open file : %s\n", Filename);
    return -1;
  }

  for (int I = 0, E = AllContextRoots.Size(); I < E; ++I) {
    const auto *Root = AllContextRoots[I];
    for (const auto *Mem = Root->FirstMemBlock; Mem; Mem = Mem->next()) {
      const uint64_t ContextStartAddr =
          reinterpret_cast<const uint64_t>(Mem->start());
      if (fwrite(reinterpret_cast<const char *>(&ContextStartAddr),
                 sizeof(uint64_t), 1, F) != 1)
        return -1;
      const uint64_t Size = Mem->size();
      if (fwrite(reinterpret_cast<const char *>(&Size), sizeof(uint64_t), 1,
                 F) != 1)
        return -1;
      if (fwrite(reinterpret_cast<const char *>(Mem->start()), sizeof(char),
                 Size, F) != Size)
        return -1;
    }
  }
  __sanitizer::Printf("[ctxprof] End Dump. Closing file.\n");
  return fclose(F);
}
}
