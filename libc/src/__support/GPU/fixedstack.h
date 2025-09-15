//===-- A lock-free data structure for a fixed capacity stack ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_GPU_FIXEDSTACK_H
#define LLVM_LIBC_SRC___SUPPORT_GPU_FIXEDSTACK_H

#include "src/__support/CPP/atomic.h"
#include "src/__support/threads/sleep.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {

// A lock-free fixed size stack backed by an underlying array of data. It
// supports push and pop operations in a completely lock-free manner.
template <typename T, uint32_t CAPACITY> struct alignas(16) FixedStack {
  // The index is stored as a 20-bit value and cannot index into any more.
  static_assert(CAPACITY < 1024 * 1024, "Invalid buffer size");

  // The head of the free and used stacks. Represents as a 20-bit index combined
  // with a 44-bit ABA tag that is updated in a single atomic operation.
  uint64_t free;
  uint64_t used;

  // The stack is a linked list of indices into the underlying data
  uint32_t next[CAPACITY];
  T data[CAPACITY];

  // Get the 20-bit index into the underlying array from the head.
  LIBC_INLINE static constexpr uint32_t get_node(uint64_t head) {
    return static_cast<uint32_t>(head & 0xfffff);
  }

  // Increment the old ABA tag and merge it into the new index.
  LIBC_INLINE static constexpr uint64_t make_head(uint64_t orig,
                                                  uint32_t node) {
    return static_cast<uint64_t>(node) | (((orig >> 20ul) + 1ul) << 20ul);
  }

  // Attempts to pop data from the given stack by making it point to the next
  // node. We repeatedly attempt to write to the head using compare-and-swap,
  // expecting that it has not been changed by any other thread.
  LIBC_INLINE uint32_t pop_impl(cpp::AtomicRef<uint64_t> head) {
    uint64_t orig = head.load(cpp::MemoryOrder::RELAXED);

    for (;;) {
      if (get_node(orig) == CAPACITY)
        return CAPACITY;

      uint32_t node =
          cpp::AtomicRef(next[get_node(orig)]).load(cpp::MemoryOrder::RELAXED);
      if (head.compare_exchange_strong(orig, make_head(orig, node),
                                       cpp::MemoryOrder::ACQUIRE,
                                       cpp::MemoryOrder::RELAXED))
        break;
    }
    return get_node(orig);
  }

  // Attempts to push data to the given stack by making it point to the new
  // node. We repeatedly attempt to write to the head using compare-and-swap,
  // expecting that it has not been changed by any other thread.
  LIBC_INLINE uint32_t push_impl(cpp::AtomicRef<uint64_t> head, uint32_t node) {
    uint64_t orig = head.load(cpp::MemoryOrder::RELAXED);
    for (;;) {
      next[node] = get_node(orig);
      if (head.compare_exchange_strong(orig, make_head(orig, node),
                                       cpp::MemoryOrder::RELEASE,
                                       cpp::MemoryOrder::RELAXED))
        break;
    }
    return get_node(head.load(cpp::MemoryOrder::RELAXED));
  }

public:
  // Initialize the free stack to be full and the used stack to be empty. We use
  // the capacity of the stack as a sentinel value.
  LIBC_INLINE constexpr FixedStack() : free(0), used(CAPACITY), data{} {
    for (uint32_t i = 0; i < CAPACITY; ++i)
      next[i] = i + 1;
  }

  LIBC_INLINE bool push(const T &val) {
    uint32_t node = pop_impl(cpp::AtomicRef(free));
    if (node == CAPACITY)
      return false;

    data[node] = val;
    push_impl(cpp::AtomicRef(used), node);
    return true;
  }

  LIBC_INLINE bool pop(T &val) {
    uint32_t node = pop_impl(cpp::AtomicRef(used));
    if (node == CAPACITY)
      return false;

    val = data[node];
    push_impl(cpp::AtomicRef(free), node);
    return true;
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_GPU_FIXEDSTACK_H
