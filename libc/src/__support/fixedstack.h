//===-- A lock-free data structure for a fixed capacity stack ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FIXEDSTACK_H
#define LLVM_LIBC_SRC___SUPPORT_FIXEDSTACK_H

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/threads/sleep.h"

#include <stdint.h>

namespace LIBC_NAMESPACE {

// A lock-free fixed size stack backed by an underlying cpp::array data
// structure. It supports push and pop operations in a thread safe manner.
template <typename T, uint32_t CAPACITY> class alignas(16) FixedStack {
  // The index is stored as a 20-bit value and cannot index into any more.
  static_assert(CAPACITY < 1024 * 1024, "Invalid buffer size");

  // The head of the free and used stacks. Represents as a 20-bit index combined
  // with a 44-bit ABA tag that is updated in a single atomic operation.
  uint64_t free;
  uint64_t used;

  // The stack is a linked list of indices into the underlying data
  cpp::array<uint32_t, CAPACITY> next;
  cpp::array<T, CAPACITY> data;

  // Get the 20-bit index into the underlying array from the head.
  static constexpr uint32_t get_node(uint64_t head) {
    return static_cast<uint32_t>(head & 0xffff);
  }

  // Increment the old ABA tag and merge it into the new index.
  static constexpr uint64_t make_new_head(uint64_t orig, uint32_t node) {
    return static_cast<uint64_t>(node) | (((orig >> 20ul) + 1ul) << 20ul);
  }

  // Helper macros for the atomic operations. We cannot use the standard
  // cpp::atomic helpers because the initializer will no longer be constexpr and
  // the NVPTX backend cannot currently support all of the atomics.
#define atomic_load(val, mem_order) __atomic_load_n(val, (int)mem_order)
#define atomic_cas(val, expected, desired, success_order, failure_order)       \
  __atomic_compare_exchange_n(val, expected, desired, /*weak=*/true,           \
                              (int)success_order, (int)failure_order)

  // Attempts to pop data from the given stack by making it point to the next
  // node. We repeatedly attempt to write to the head using compare-and-swap,
  // expecting that it has not been changed by any other thread.
  uint32_t pop_impl(uint64_t *head) {
    uint64_t orig = atomic_load(head, cpp::MemoryOrder::RELAXED);

    for (;;) {
      if (get_node(orig) == CAPACITY)
        return CAPACITY;

      uint32_t node =
          atomic_load(&next[get_node(orig)], cpp::MemoryOrder::RELAXED);
      if (atomic_cas(head, &orig, make_new_head(orig, node),
                     cpp::MemoryOrder::ACQUIRE, cpp::MemoryOrder::RELAXED))
        break;
      sleep_briefly();
    }
    return get_node(orig);
  }

  // Attempts to push data to the given stack by making it point to the new
  // node. We repeatedly attempt to write to the head using compare-and-swap,
  // expecting that it has not been changed by any other thread.
  uint32_t push_impl(uint64_t *head, uint32_t node) {
    uint64_t orig = atomic_load(head, cpp::MemoryOrder::RELAXED);
    for (;;) {
      next[node] = get_node(orig);
      if (atomic_cas(head, &orig, make_new_head(orig, node),
                     cpp::MemoryOrder::RELEASE, cpp::MemoryOrder::RELAXED))
        break;
      sleep_briefly();
    }
    return get_node(*head);
  }

public:
  // Initialize the free stack to be full and the used stack to be empty. We use
  // the capacity of the stack as a sentinel value.
  constexpr FixedStack() : free(0), used(CAPACITY), data{} {
    for (uint32_t i = 0; i < CAPACITY; ++i)
      next[i] = i + 1;
  }

  bool push(const T &val) {
    uint32_t node = pop_impl(&free);
    if (node == CAPACITY)
      return false;

    data[node] = val;
    push_impl(&used, node);
    return true;
  }

  bool pop(T &val) {
    uint32_t node = pop_impl(&used);
    if (node == CAPACITY)
      return false;

    val = data[node];
    push_impl(&free, node);
    return true;
  }

  bool empty() const {
    return get_node(atomic_load(&used, cpp::MemoryOrder::RELAXED)) == CAPACITY;
  }

  bool full() const {
    return get_node(atomic_load(&free, cpp::MemoryOrder::RELAXED)) == CAPACITY;
  }

#undef atomic_load
#undef atomic_cas
};

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FIXEDSTACK_H
