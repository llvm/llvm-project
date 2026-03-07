// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RCU_RCU_DOMAIN_H
#define _LIBCPP___RCU_RCU_DOMAIN_H

#include <__config>
#include <__functional/function.h>
#include <__memory/unique_ptr.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

struct __rcu_node {
  function<void()> __callback_{};
  __rcu_node* __next_ = nullptr;

  _LIBCPP_HIDE_FROM_ABI __rcu_node() {}
  _LIBCPP_HIDE_FROM_ABI ~__rcu_node() {}
};

class _LIBCPP_EXPORTED_FROM_ABI rcu_domain {
  class __impl;
  unique_ptr<__impl> __pimpl_;

  template <class, class>
  friend class rcu_obj_base;

  friend rcu_domain& rcu_default_domain() noexcept;
  friend void rcu_synchronize(rcu_domain&) noexcept;

  static rcu_domain& __rcu_default_domain() noexcept;

  template <class _Tp, class _Dp = default_delete<_Tp>>
  _LIBCPP_HIDE_FROM_ABI void __rcu_retire_hidden_friend(_Tp* __tp, _Dp __deleter, rcu_domain& __dom) {
    auto* __node        = new __rcu_node();
    __node->__callback_ = [__tp, __deleter = std::move(__deleter)]() mutable { __deleter(__tp); };
    __dom.__retire(__node);
  }

  rcu_domain();

  void __retire(__rcu_node*) noexcept;

public:
  rcu_domain(const rcu_domain&)            = delete;
  rcu_domain& operator=(const rcu_domain&) = delete;
  ~rcu_domain();

  void printAllReaderStatesInHex();

  void lock() noexcept;

  _LIBCPP_HIDE_FROM_ABI bool try_lock() noexcept {
    lock();
    return true;
  }

  void unlock() noexcept;
};

_LIBCPP_EXPORTED_FROM_ABI rcu_domain& rcu_default_domain() noexcept;

_LIBCPP_EXPORTED_FROM_ABI void rcu_synchronize(rcu_domain& __dom = rcu_default_domain()) noexcept;

_LIBCPP_EXPORTED_FROM_ABI void rcu_barrier(rcu_domain& __dom = rcu_default_domain()) noexcept;

template <class _Tp, class _Dp = default_delete<_Tp>>
_LIBCPP_HIDE_FROM_ABI void rcu_retire(_Tp* __tp, _Dp __deleter = _Dp(), rcu_domain& __dom = rcu_default_domain()) {
  __rcu_retire_hidden_friend(__tp, std::move(__deleter), __dom);
}

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RCU_RCU_DOMAIN_H
