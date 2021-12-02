//===-- sanitizer_stackdepot.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_stackdepot.h"

#include "sanitizer_common.h"
#include "sanitizer_hash.h"
#include "sanitizer_stack_store.h"
#include "sanitizer_stackdepotbase.h"

namespace __sanitizer {

struct StackDepotNode {
  using hash_type = u64;
  hash_type stack_hash;
  u32 link;
  StackStore::Id store_id;

  static const u32 kTabSizeLog = SANITIZER_ANDROID ? 16 : 20;

  typedef StackTrace args_type;
  bool eq(hash_type hash, const args_type &args) const {
    return hash == stack_hash;
  }
  static uptr allocated();
  static hash_type hash(const args_type &args) {
    MurMur2Hash64Builder H(args.size * sizeof(uptr));
    for (uptr i = 0; i < args.size; i++) H.add(args.trace[i]);
    H.add(args.tag);
    return H.get();
  }
  static bool is_valid(const args_type &args) {
    return args.size > 0 && args.trace;
  }
  void store(u32 id, const args_type &args, hash_type hash);
  args_type load(u32 id) const;
  static StackDepotHandle get_handle(u32 id);

  typedef StackDepotHandle handle_type;
};

static StackStore stackStore;

// FIXME(dvyukov): this single reserved bit is used in TSan.
typedef StackDepotBase<StackDepotNode, 1, StackDepotNode::kTabSizeLog>
    StackDepot;
static StackDepot theDepot;
// Keep mutable data out of frequently access nodes to improve caching
// efficiency.
static TwoLevelMap<atomic_uint32_t, StackDepot::kNodesSize1,
                   StackDepot::kNodesSize2>
    useCounts;

int StackDepotHandle::use_count() const {
  return atomic_load_relaxed(&useCounts[id_]);
}

void StackDepotHandle::inc_use_count_unsafe() {
  atomic_fetch_add(&useCounts[id_], 1, memory_order_relaxed);
}

uptr StackDepotNode::allocated() {
  return stackStore.Allocated() + useCounts.MemoryUsage();
}

void StackDepotNode::store(u32 id, const args_type &args, hash_type hash) {
  stack_hash = hash;
  uptr pack = 0;
  store_id = stackStore.Store(args, &pack);
  if (pack)
    stackStore.Pack(StackStore::Compression::None);
}

StackDepotNode::args_type StackDepotNode::load(u32 id) const {
  if (!store_id)
    return {};
  return stackStore.Load(store_id);
}

StackDepotStats StackDepotGetStats() { return theDepot.GetStats(); }

u32 StackDepotPut(StackTrace stack) { return theDepot.Put(stack); }

StackDepotHandle StackDepotPut_WithHandle(StackTrace stack) {
  return StackDepotNode::get_handle(theDepot.Put(stack));
}

StackTrace StackDepotGet(u32 id) {
  return theDepot.Get(id);
}

void StackDepotLockAll() {
  theDepot.LockAll();
}

void StackDepotUnlockAll() {
  theDepot.UnlockAll();
}

void StackDepotPrintAll() {
#if !SANITIZER_GO
  theDepot.PrintAll();
#endif
}

StackDepotHandle StackDepotNode::get_handle(u32 id) {
  return StackDepotHandle(&theDepot.nodes[id], id);
}

void StackDepotTestOnlyUnmap() {
  theDepot.TestOnlyUnmap();
  stackStore.TestOnlyUnmap();
}

} // namespace __sanitizer
