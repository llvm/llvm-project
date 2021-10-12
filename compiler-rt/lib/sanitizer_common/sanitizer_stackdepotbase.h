//===-- sanitizer_stackdepotbase.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of a mapping from arbitrary values to unique 32-bit
// identifiers.
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_STACKDEPOTBASE_H
#define SANITIZER_STACKDEPOTBASE_H

#include <stdio.h>

#include "sanitizer_atomic.h"
#include "sanitizer_flat_map.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_mutex.h"

namespace __sanitizer {

template <class Node, int kReservedBits, int kTabSizeLog>
class StackDepotBase {
  static const u32 kIdSizeLog = sizeof(u32) * 8 - kReservedBits;
  static const u32 kNodesSize1Log = kIdSizeLog / 2;
  static const u32 kNodesSize2Log = kIdSizeLog - kNodesSize1Log;

 public:
  typedef typename Node::args_type args_type;
  typedef typename Node::handle_type handle_type;
  typedef typename Node::hash_type hash_type;

  static const u64 kNodesSize1 = 1ull << kNodesSize1Log;
  static const u64 kNodesSize2 = 1ull << kNodesSize2Log;

  // Maps stack trace to an unique id.
  handle_type Put(args_type args, bool *inserted = nullptr);
  // Retrieves a stored stack trace by the id.
  args_type Get(u32 id);

  StackDepotStats GetStats() const {
    return {
        atomic_load_relaxed(&n_uniq_ids),
        nodes.MemoryUsage() + Node::allocated(),
    };
  }

  void LockAll();
  void UnlockAll();
  void PrintAll();

 private:
  static Node *find(Node *s, args_type args, hash_type hash);
  static Node *lock(atomic_uintptr_t *p);
  static void unlock(atomic_uintptr_t *p, Node *s);

  static const int kTabSize = 1 << kTabSizeLog;  // Hash table size.

  atomic_uintptr_t tab[kTabSize];  // Hash table of Node's.

  atomic_uint32_t n_uniq_ids;

  TwoLevelMap<Node, kNodesSize1, kNodesSize2> nodes;

  friend class StackDepotReverseMap;
};

template <class Node, int kReservedBits, int kTabSizeLog>
Node *StackDepotBase<Node, kReservedBits, kTabSizeLog>::find(Node *s,
                                                             args_type args,
                                                             hash_type hash) {
  // Searches linked list s for the stack, returns its id.
  for (; s; s = s->link) {
    if (s->eq(hash, args)) {
      return s;
    }
  }
  return nullptr;
}

template <class Node, int kReservedBits, int kTabSizeLog>
Node *StackDepotBase<Node, kReservedBits, kTabSizeLog>::lock(
    atomic_uintptr_t *p) {
  // Uses the pointer lsb as mutex.
  for (int i = 0;; i++) {
    uptr cmp = atomic_load(p, memory_order_relaxed);
    if ((cmp & 1) == 0 &&
        atomic_compare_exchange_weak(p, &cmp, cmp | 1, memory_order_acquire))
      return (Node *)cmp;
    if (i < 10)
      proc_yield(10);
    else
      internal_sched_yield();
  }
}

template <class Node, int kReservedBits, int kTabSizeLog>
void StackDepotBase<Node, kReservedBits, kTabSizeLog>::unlock(
    atomic_uintptr_t *p, Node *s) {
  DCHECK_EQ((uptr)s & 1, 0);
  atomic_store(p, (uptr)s, memory_order_release);
}

template <class Node, int kReservedBits, int kTabSizeLog>
typename StackDepotBase<Node, kReservedBits, kTabSizeLog>::handle_type
StackDepotBase<Node, kReservedBits, kTabSizeLog>::Put(args_type args,
                                                      bool *inserted) {
  if (inserted)
    *inserted = false;
  if (!LIKELY(Node::is_valid(args)))
    return handle_type();
  hash_type h = Node::hash(args);
  atomic_uintptr_t *p = &tab[h % kTabSize];
  uptr v = atomic_load(p, memory_order_consume);
  Node *s = (Node *)(v & ~uptr(1));
  // First, try to find the existing stack.
  Node *node = find(s, args, h);
  if (LIKELY(node))
    return node->get_handle();
  // If failed, lock, retry and insert new.
  Node *s2 = lock(p);
  if (s2 != s) {
    node = find(s2, args, h);
    if (node) {
      unlock(p, s2);
      return node->get_handle();
    }
  }
  u32 id = atomic_fetch_add(&n_uniq_ids, 1, memory_order_relaxed) + 1;
  CHECK_NE(id, 0);
  CHECK_EQ(id & (((u32)-1) >> kReservedBits), id);
  s = &nodes[id];
  s->id = id;
  s->store(args, h);
  s->link = s2;
  unlock(p, s);
  if (inserted) *inserted = true;
  return s->get_handle();
}

template <class Node, int kReservedBits, int kTabSizeLog>
typename StackDepotBase<Node, kReservedBits, kTabSizeLog>::args_type
StackDepotBase<Node, kReservedBits, kTabSizeLog>::Get(u32 id) {
  if (id == 0)
    return args_type();
  CHECK_EQ(id & (((u32)-1) >> kReservedBits), id);
  if (!nodes.contains(id))
    return args_type();
  const Node &node = nodes[id];
  if (node.id != id)
    return args_type();
  return node.load();
}

template <class Node, int kReservedBits, int kTabSizeLog>
void StackDepotBase<Node, kReservedBits, kTabSizeLog>::LockAll() {
  for (int i = 0; i < kTabSize; ++i) {
    lock(&tab[i]);
  }
}

template <class Node, int kReservedBits, int kTabSizeLog>
void StackDepotBase<Node, kReservedBits, kTabSizeLog>::UnlockAll() {
  for (int i = 0; i < kTabSize; ++i) {
    atomic_uintptr_t *p = &tab[i];
    uptr s = atomic_load(p, memory_order_relaxed);
    unlock(p, (Node *)(s & ~uptr(1)));
  }
}

template <class Node, int kReservedBits, int kTabSizeLog>
void StackDepotBase<Node, kReservedBits, kTabSizeLog>::PrintAll() {
  for (int i = 0; i < kTabSize; ++i) {
    atomic_uintptr_t *p = &tab[i];
    uptr v = atomic_load(p, memory_order_consume);
    Node *s = (Node *)(v & ~uptr(1));
    for (; s; s = s->link) {
      Printf("Stack for id %u:\n", s->id);
      s->load().Print();
    }
  }
}

} // namespace __sanitizer

#endif // SANITIZER_STACKDEPOTBASE_H
