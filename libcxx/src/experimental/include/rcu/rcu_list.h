// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RCU_RCU_LIST_H
#define _LIBCPP___RCU_RCU_LIST_H

#include <__config>
#include <__functional/function.h>
#include <__rcu/rcu_domain.h>
#include <atomic>
#include <type_traits>

#include "thread_local_container.h"

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

class rcu_atomic_list_view;

class rcu_singly_list_view {
private:
  __rcu_node* head_ = nullptr;
  __rcu_node* tail_ = nullptr;

public:
  void splice_back(rcu_singly_list_view& other) noexcept {
    if (other.head_ == nullptr) {
      return;
    }
    if (head_ == nullptr) {
      head_ = other.head_;
      tail_ = other.tail_;
    } else {
      tail_->__next_ = other.head_;
      tail_          = other.tail_;
    }
    other.head_ = nullptr;
    other.tail_ = nullptr;
  }

  void splice_back(rcu_atomic_list_view& other) noexcept;

  template <class Func>
  void for_each(Func&& f) noexcept {
    __rcu_node* current = head_;
    while (current != nullptr) {
      // __f could delete __current, so we need to get the next pointer first
      auto __next = current->__next_;
      f(current);
      current = __next;
    }
  }
};

class rcu_atomic_list_view {
  std::atomic<__rcu_node*> head_{};

  friend class rcu_singly_list_view;

public:
  void push_front(__rcu_node* node) noexcept {
    auto expected_head = head_.load(std::memory_order_relaxed);
    while (true) {
      node->__next_ = expected_head;
      if (expected_head == nullptr) {
        node->__tail_if_atomic_head = node;
      } else {
        node->__tail_if_atomic_head = expected_head->__tail_if_atomic_head;
      }
      if (head_.compare_exchange_weak(expected_head, node, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        break;
      }
    }
  }
};

void rcu_singly_list_view::splice_back(rcu_atomic_list_view& other) noexcept {
  if (other.head_.load(std::memory_order_relaxed) == nullptr) {
    return;
  }
  auto head = other.head_.exchange(nullptr, std::memory_order_acq_rel);
  rcu_singly_list_view tmp;
  tmp.head_ = head;
  tmp.tail_ = head->__tail_if_atomic_head;
  this->splice_back(tmp);
}

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RCU_RCU_LIST_H
