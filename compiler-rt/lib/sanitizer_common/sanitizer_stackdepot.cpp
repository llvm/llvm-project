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
#include "sanitizer_persistent_allocator.h"
#include "sanitizer_stackdepotbase.h"

namespace __sanitizer {

static PersistentAllocator<uptr> traceAllocator;

struct StackDepotNode {
  using hash_type = u64;
  hash_type stack_hash;
  uptr *stack_trace;
  u32 link;
  atomic_uint32_t tag_and_use_count;  // tag : 12 high bits; use_count : 20;

  static const u32 kTabSizeLog = SANITIZER_ANDROID ? 16 : 20;
  static const u32 kUseCountBits = 20;
  static const u32 kMaxUseCount = 1 << kUseCountBits;
  static const u32 kUseCountMask = (1 << kUseCountBits) - 1;

  typedef StackTrace args_type;
  bool eq(hash_type hash, const args_type &args) const {
    return hash == stack_hash;
  }
  static uptr allocated() { return traceAllocator.allocated(); }
  static hash_type hash(const args_type &args) {
    MurMur2Hash64Builder H(args.size * sizeof(uptr));
    for (uptr i = 0; i < args.size; i++) H.add(args.trace[i]);
    H.add(args.tag);
    return H.get();
  }
  static bool is_valid(const args_type &args) {
    return args.size > 0 && args.trace;
  }
  void store(const args_type &args, hash_type hash) {
    CHECK_EQ(args.tag & (~kUseCountMask >> kUseCountBits), args.tag);
    atomic_store(&tag_and_use_count, args.tag << kUseCountBits,
                 memory_order_relaxed);
    stack_hash = hash;
    stack_trace = traceAllocator.alloc(args.size + 1);
    *stack_trace = args.size;
    internal_memcpy(stack_trace + 1, args.trace, args.size * sizeof(uptr));
  }
  args_type load() const {
    if (!stack_trace)
      return {};
    u32 tag =
        atomic_load(&tag_and_use_count, memory_order_relaxed) >> kUseCountBits;
    return args_type(stack_trace + 1, *stack_trace, tag);
  }
  static StackDepotHandle get_handle(u32 id);

  typedef StackDepotHandle handle_type;
};

COMPILER_CHECK(StackDepotNode::kMaxUseCount >= (u32)kStackDepotMaxUseCount);

int StackDepotHandle::use_count() const {
  return atomic_load(&node_->tag_and_use_count, memory_order_relaxed) &
         StackDepotNode::kUseCountMask;
}
void StackDepotHandle::inc_use_count_unsafe() {
  u32 prev =
      atomic_fetch_add(&node_->tag_and_use_count, 1, memory_order_relaxed) &
      StackDepotNode::kUseCountMask;
  CHECK_LT(prev + 1, StackDepotNode::kMaxUseCount);
}

// FIXME(dvyukov): this single reserved bit is used in TSan.
typedef StackDepotBase<StackDepotNode, 1, StackDepotNode::kTabSizeLog>
    StackDepot;
static StackDepot theDepot;

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

} // namespace __sanitizer
