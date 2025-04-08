//===- RootAutodetector.cpp - detect contextual profiling roots -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RootAutoDetector.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_placement_new.h" // IWYU pragma: keep (DenseMap)
#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>

using namespace __ctx_profile;
template <typename T> using Set = DenseMap<T, bool>;

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
