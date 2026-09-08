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
#include <__functional/function_ref.h>
#include <__memory/unique_ptr.h>
#include <__type_traits/is_constructible.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

struct __rcu_node {
  function_ref<void()> __callback_ = std::cw<[] {}>;
  __rcu_node* __next_              = nullptr;
};

template <class _Tp, class _Deleter>
struct __rcu_node_with_deleter : __rcu_node {
  _Tp* __obj_;
  _LIBCPP_NO_UNIQUE_ADDRESS _Deleter __deleter_ = _Deleter();

  _LIBCPP_HIDE_FROM_ABI __rcu_node_with_deleter(_Tp* __obj, _Deleter __deleter) : __obj_(__obj), __deleter_(__deleter) {
    __callback_ = function_ref<void()>(std::cw<&__rcu_node_with_deleter::__destroy>, this);
  }

  _LIBCPP_HIDE_FROM_ABI void __destroy() const {
    __deleter_(__obj_);
    delete this;
  }
};

class _LIBCPP_EXPORTED_FROM_ABI rcu_domain {
  class __impl;
  unique_ptr<__impl> __pimpl_;

  template <class, class>
  friend class rcu_obj_base;

  friend rcu_domain& rcu_default_domain() noexcept;
  friend void rcu_synchronize(rcu_domain&) noexcept;
  friend void rcu_barrier(rcu_domain&) noexcept;

  static rcu_domain& __rcu_default_domain() noexcept;

  template <class _Tp, class _Dp>
  friend _LIBCPP_HIDE_FROM_ABI void __rcu_retire_hidden_friend(_Tp* __tp, _Dp __deleter, rcu_domain& __dom) {
    auto* __node = new __rcu_node_with_deleter<_Tp, _Dp>(__tp, std::move(__deleter));
    __dom.__retire(__node);
  }

  rcu_domain();

  void __retire(__rcu_node*) noexcept;

public:
  rcu_domain(const rcu_domain&)            = delete;
  rcu_domain& operator=(const rcu_domain&) = delete;
  ~rcu_domain();

  void debugPrintAllReaderStatesInHex();

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
  static_assert(std::is_move_constructible_v<_Dp>);
  static_assert(requires(_Dp __dp, _Tp* __ptr) { __dp(__ptr); }, "Deleter must be callable with a pointer");
  __rcu_retire_hidden_friend(__tp, std::move(__deleter), __dom);
}

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RCU_RCU_DOMAIN_H
