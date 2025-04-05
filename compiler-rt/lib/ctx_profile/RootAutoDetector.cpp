//===- RootAutodetector.cpp - detect contextual profiling roots -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RootAutoDetector.h"

#include "CtxInstrProfiling.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_placement_new.h" // IWYU pragma: keep (DenseMap)
#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>

using namespace __ctx_profile;
template <typename T> using Set = DenseMap<T, bool>;

namespace __sanitizer {
void BufferedStackTrace::UnwindImpl(uptr pc, uptr bp, void *context,
                                    bool request_fast, u32 max_depth) {
  // We can't implement the fast variant. The fast variant ends up invoking an
  // external allocator, because of pthread_attr_getstack. If this happens
  // during an allocation of the program being instrumented, a non-reentrant
  // lock may be taken (this was observed). The allocator called by
  // pthread_attr_getstack will also try to take that lock.
  UnwindSlow(pc, max_depth);
}
} // namespace __sanitizer

RootAutoDetector::PerThreadSamples::PerThreadSamples(RootAutoDetector &Parent) {
  GenericScopedLock<SpinMutex> L(&Parent.AllSamplesMutex);
  Parent.AllSamples.PushBack(this);
}

void RootAutoDetector::start() {
  atomic_store_relaxed(&Self, reinterpret_cast<uintptr_t>(this));
  pthread_create(
      &WorkerThread, nullptr,
      +[](void *Ctx) -> void * {
        RootAutoDetector *RAD = reinterpret_cast<RootAutoDetector *>(Ctx);
        SleepForSeconds(RAD->WaitSeconds);
        // To avoid holding the AllSamplesMutex, make a snapshot of all the
        // thread samples collected so far
        Vector<PerThreadSamples *> SamplesSnapshot;
        {
          GenericScopedLock<SpinMutex> M(&RAD->AllSamplesMutex);
          SamplesSnapshot.Resize(RAD->AllSamples.Size());
          for (uptr I = 0; I < RAD->AllSamples.Size(); ++I)
            SamplesSnapshot[I] = RAD->AllSamples[I];
        }
        DenseMap<uptr, uint64_t> AllRoots;
        for (uptr I = 0; I < SamplesSnapshot.Size(); ++I) {
          GenericScopedLock<SpinMutex>(&SamplesSnapshot[I]->M);
          SamplesSnapshot[I]->TrieRoot.determineRoots().forEach([&](auto &KVP) {
            auto [FAddr, Count] = KVP;
            AllRoots[FAddr] += Count;
            return true;
          });
        }
        // FIXME: as a next step, establish a minimum relative nr of samples
        // per root that would qualify it as a root.
        for (auto *FD = reinterpret_cast<FunctionData *>(
                 atomic_load_relaxed(&RAD->FunctionDataListHead));
             FD; FD = FD->Next) {
          if (AllRoots.contains(reinterpret_cast<uptr>(FD->EntryAddress))) {
            FD->getOrAllocateContextRoot();
          }
        }
        atomic_store_relaxed(&RAD->Self, 0);
        return nullptr;
      },
      this);
}

void RootAutoDetector::join() { pthread_join(WorkerThread, nullptr); }

void RootAutoDetector::sample() {
  // tracking reentry in case we want to re-explore fast stack unwind - which
  // does potentially re-enter the runtime because it calls the instrumented
  // allocator because of pthread_attr_getstack. See the notes also on
  // UnwindImpl above.
  static thread_local bool Entered = false;
  static thread_local uint64_t Entries = 0;
  if (Entered || (++Entries % SampleRate))
    return;
  Entered = true;
  collectStack();
  Entered = false;
}

void RootAutoDetector::collectStack() {
  GET_CALLER_PC_BP;
  BufferedStackTrace CurrentStack;
  CurrentStack.Unwind(pc, bp, nullptr, false);
  // 2 stack frames would be very unlikely to mean anything, since at least the
  // compiler-rt frame - which can't be inlined - should be observable, which
  // counts as 1; we can be even more aggressive with this number.
  if (CurrentStack.size <= 2)
    return;
  static thread_local PerThreadSamples *ThisThreadSamples =
      new (__sanitizer::InternalAlloc(sizeof(PerThreadSamples)))
          PerThreadSamples(*this);

  if (!ThisThreadSamples->M.TryLock())
    return;

  ThisThreadSamples->TrieRoot.insertStack(CurrentStack);
  ThisThreadSamples->M.Unlock();
}

uptr PerThreadCallsiteTrie::getFctStartAddr(uptr CallsiteAddress) const {
  // this requires --linkopt=-Wl,--export-dynamic
  Dl_info Info;
  if (dladdr(reinterpret_cast<const void *>(CallsiteAddress), &Info) != 0)
    return reinterpret_cast<uptr>(Info.dli_saddr);
  return 0;
}

void PerThreadCallsiteTrie::insertStack(const StackTrace &ST) {
  ++TheTrie.Count;
  auto *Current = &TheTrie;
  // the stack is backwards - the first callsite is at the top.
  for (int I = ST.size - 1; I >= 0; --I) {
    uptr ChildAddr = ST.trace[I];
    auto [Iter, _] = Current->Children.insert({ChildAddr, Trie(ChildAddr)});
    ++Iter->second.Count;
    Current = &Iter->second;
  }
}

DenseMap<uptr, uint64_t> PerThreadCallsiteTrie::determineRoots() const {
  // Assuming a message pump design, roots are those functions called by the
  // message pump. The message pump is an infinite loop (for all practical
  // considerations) fetching data from a queue. The root functions return -
  // otherwise the message pump doesn't work. This function detects roots as the
  // first place in the trie (starting from the root) where a function calls 2
  // or more functions.
  //
  // We start with a callsite trie - the nodes are callsites. Different child
  // nodes may actually correspond to the same function.
  //
  // For example: using function(callsite)
  // f1(csf1_1) -> f2(csf2_1) -> f3
  //            -> f2(csf2_2) -> f4
  //
  // would be represented in our trie as:
  // csf1_1 -> csf2_1 -> f3
  //        -> csf2_2 -> f4
  //
  // While we can assert the control flow returns to f2, we don't know if it
  // ever returns to f1. f2 could be the message pump.
  //
  // We need to convert our callsite tree into a function tree. We can also,
  // more economically, just see how many distinct functions there are at a
  // certain depth. When that count is greater than 1, we got to potential roots
  // and everything above should be considered as non-roots.
  DenseMap<uptr, uint64_t> Result;
  Set<const Trie *> Worklist;
  Worklist.insert({&TheTrie, {}});

  while (!Worklist.empty()) {
    Set<const Trie *> NextWorklist;
    DenseMap<uptr, uint64_t> Candidates;
    Worklist.forEach([&](const auto &KVP) {
      auto [Node, _] = KVP;
      auto SA = getFctStartAddr(Node->CallsiteAddress);
      Candidates[SA] += Node->Count;
      Node->Children.forEach([&](auto &ChildKVP) {
        NextWorklist.insert({&ChildKVP.second, true});
        return true;
      });
      return true;
    });
    if (Candidates.size() > 1) {
      Result.swap(Candidates);
      break;
    }
    Worklist.swap(NextWorklist);
  }
  return Result;
}
