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

class rcu_thread_local_list_view;

class rcu_singly_list_view {
private:
  __rcu_node* __head_ = nullptr;
  __rcu_node* __tail_ = nullptr;

public:
  void __splice_back(rcu_singly_list_view& __other) noexcept {
    if (__other.__head_ == nullptr) {
      return;
    }
    if (__head_ == nullptr) {
      __head_ = __other.__head_;
      __tail_ = __other.__tail_;
    } else {
      __tail_->__next_ = __other.__head_;
      __tail_          = __other.__tail_;
    }
    __other.__head_ = nullptr;
    __other.__tail_ = nullptr;
  }

  void __splice_back(rcu_thread_local_list_view& __other) noexcept;


  template <class _Func>
  void __for_each(_Func&& __f) noexcept {
    __rcu_node* __current = __head_;
    while (__current != nullptr) {
      // __f could delete __current, so we need to get the next pointer first
      auto __next = __current->__next_;
      __f(__current);
      __current = __next;
    }
  }
};

class rcu_thread_local_list_view {
  struct alignas(2*sizeof(void*)) thread_entry {
    __rcu_node* __head_ = nullptr;
    __rcu_node* __tail_ = nullptr;
  };

  using per_thread_entries = thread_local_container<thread_entry>;

  friend class rcu_singly_list_view;

public:
  void __push_front(__rcu_node* __node) noexcept {
    atomic_ref<thread_entry> entry_ref = per_thread_entries::get_current_thread_instance();
    auto expected_entry                = entry_ref.load(std::memory_order_relaxed);
    auto original_next = __node->__next_;
    while (true) {
      auto new_entry = [&] {
        if (expected_entry.__head_ == nullptr) {
          return thread_entry{__node, __node};
        } else {
          __node->__next_ = expected_entry.__head_;
          return thread_entry{__node, expected_entry.__tail_};
        }
      }();
      if (entry_ref.compare_exchange_weak(
              expected_entry, new_entry, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        break;
      } else {
        __node->__next_ = original_next;
      }
    }
  }
};

void rcu_singly_list_view::__splice_back(rcu_thread_local_list_view& __other) noexcept {
  using thread_entry             = rcu_thread_local_list_view::thread_entry;
  const auto splice_single_entry = [this](atomic_ref<thread_entry> entry_ref) noexcept {
    if (entry_ref.load(std::memory_order_relaxed).__head_ == nullptr) {
      return;
    }
    auto entry = entry_ref.exchange(thread_entry{nullptr, nullptr}, std::memory_order_acq_rel);
    rcu_singly_list_view tmp;
    tmp.__head_ = entry.__head_;
    tmp.__tail_ = entry.__tail_;
    this->__splice_back(tmp);
  };
  rcu_thread_local_list_view::per_thread_entries::for_each(splice_single_entry);
}

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RCU_RCU_LIST_H
