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

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

struct __rcu_node {
  function<void()> __callback_{};
  __rcu_node* __next_ = nullptr;
};

class __rcu_singly_list_view {
private:
  __rcu_node* __head_ = nullptr;
  __rcu_node* __tail_ = nullptr;

public:
  void __splice_back(__rcu_singly_list_view& __other) noexcept {
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

  void __push_back(__rcu_node* __node) noexcept {
    // assert(__node->__next_ == nullptr);
    if (__head_ == nullptr) {
      __head_ = __node;
      __tail_ = __node;
    } else {
      __tail_->__next_ = __node;
      __tail_          = __node;
    }
  }

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


#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RCU_RCU_LIST_H
