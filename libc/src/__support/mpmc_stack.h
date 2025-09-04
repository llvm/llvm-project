//===-- Simple Lock-free MPMC Stack -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MPMC_STACK_H
#define LLVM_LIBC_SRC___SUPPORT_MPMC_STACK_H

#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/aba_ptr.h"

namespace LIBC_NAMESPACE_DECL {
template <class T> class MPMCStack {
  struct Node {
    cpp::Atomic<size_t> visitor;
    Node *next;
    T value;

    LIBC_INLINE Node(T val) : visitor(0), next(nullptr), value(val) {}
  };
  AbaPtr<Node> head;

public:
  static_assert(cpp::is_copy_constructible<T>::value,
                "T must be copy constructible");
  LIBC_INLINE constexpr MPMCStack() : head(nullptr) {}
  LIBC_INLINE bool push(T value) {
    AllocChecker ac;
    Node *new_node = new (ac) Node(value);
    if (!ac)
      return false;
    head.transaction([new_node](Node *old_head) {
      new_node->next = old_head;
      return new_node;
    });
    return true;
  }
  LIBC_INLINE bool push_all(T values[], size_t count) {
    struct Guard {
      size_t count;
      Node **allocated;
      LIBC_INLINE Guard(Node *allocated[]) : count(0), allocated(allocated) {}
      LIBC_INLINE ~Guard() {
        for (size_t i = 0; i < count; ++i)
          delete allocated[i];
      }
      LIBC_INLINE void add(Node *node) { allocated[count++] = node; }
      LIBC_INLINE void clear() { count = 0; }
    };
    // Variable sized array is a GNU extension.
    __extension__ Node *allocated[count];
    {
      Guard guard(allocated);
      for (size_t i = 0; i < count; ++i) {
        AllocChecker ac;
        Node *new_node = new (ac) Node(values[i]);
        if (!ac)
          return false;
        guard.add(new_node);
        if (i != 0)
          new_node->next = allocated[i - 1];
      }
      guard.clear();
    }
    head.transaction([&allocated, count](Node *old_head) {
      allocated[0]->next = old_head;
      return allocated[count - 1];
    });
    return true;
  }
  LIBC_INLINE cpp::optional<T> pop() {
    cpp::optional<T> res = cpp::nullopt;
    Node *node = nullptr;
    head.transaction([&](Node *current_head) -> Node * {
      if (!current_head) {
        res = cpp::nullopt;
        return nullptr;
      }
      node = current_head;
      node->visitor.fetch_add(1);
      res = cpp::optional<T>{node->value};
      Node *next = node->next;
      node->visitor.fetch_sub(1);
      return next;
    });
    // On a successful transaction, a node is popped by us. So we must delete
    // it. When we are at here, no one else can acquire
    // new reference to the node, but we still need to wait until other threads
    // inside the transaction who may potentially be holding a reference to the
    // node.
    if (res) {
      // Spin until the node is no longer in use.
      while (node->visitor.load() != 0)
        LIBC_NAMESPACE::sleep_briefly();
      delete node;
    }
    return res;
  }
};
} // namespace LIBC_NAMESPACE_DECL

#endif
