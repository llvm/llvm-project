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

#include "thread_local_container.h"

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

class rcu_singly_list_view {
private:
  __rcu_node* head_ = nullptr;
  __rcu_node* tail_ = nullptr;

public:
  rcu_singly_list_view() = default;
  rcu_singly_list_view(__rcu_node* head, __rcu_node* tail) : head_(head), tail_(tail) {}

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

struct alignas(2 * sizeof(void*)) rcu_atomic_list_view {
  __rcu_node* head_ = nullptr;
  __rcu_node* tail_ = nullptr;

  static void push_front(atomic_ref<rcu_atomic_list_view> view_ref, __rcu_node* node) noexcept {
    auto expected_view = view_ref.load(std::memory_order_relaxed);
    auto original_next = node->__next_;
    while (true) {
      auto new_view = [&] {
        if (expected_view.head_ == nullptr) {
          return rcu_atomic_list_view{node, node};
        } else {
          node->__next_ = expected_view.head_;
          return rcu_atomic_list_view{node, expected_view.tail_};
        }
      }();
      if (view_ref.compare_exchange_weak(
              expected_view, new_view, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        break;
      } else {
        node->__next_ = original_next;
      }
    }
  }

  static void splice_back(atomic_ref<rcu_atomic_list_view> from_ref, rcu_singly_list_view& to) noexcept {
    if (from_ref.load(std::memory_order_relaxed).head_ == nullptr) {
      return;
    }
    auto from = from_ref.exchange(rcu_atomic_list_view{nullptr, nullptr}, std::memory_order_acq_rel);
    rcu_singly_list_view tmp(from.head_, from.tail_);
    to.splice_back(tmp);
  }
};

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RCU_RCU_LIST_H
