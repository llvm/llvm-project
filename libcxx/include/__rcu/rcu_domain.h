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
#include <__memory/unique_ptr.h>
#include <__type_traits/is_constructible.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

struct __rcu_node;

class rcu_domain {
  class __impl;
  unique_ptr<__impl> __pimpl_;

  friend struct __rcu_domain_access;

  template <class, class>
  friend class rcu_obj_base;

  _LIBCPP_EXPORTED_FROM_ABI static rcu_domain& __rcu_default_domain() noexcept;

  _LIBCPP_EXPORTED_FROM_ABI void __retire(__rcu_node*) noexcept;

  _LIBCPP_EXPORTED_FROM_ABI void __lock() noexcept;

  _LIBCPP_EXPORTED_FROM_ABI void __unlock() noexcept;

  _LIBCPP_EXPORTED_FROM_ABI rcu_domain();

public:
  rcu_domain(const rcu_domain&)            = delete;
  rcu_domain& operator=(const rcu_domain&) = delete;

  _LIBCPP_EXPORTED_FROM_ABI ~rcu_domain();

  _LIBCPP_HIDE_FROM_ABI void lock() noexcept { __lock(); }

  _LIBCPP_HIDE_FROM_ABI bool try_lock() noexcept {
    lock();
    return true;
  }

  _LIBCPP_HIDE_FROM_ABI void unlock() noexcept { __unlock(); }
};

_LIBCPP_EXPORTED_FROM_ABI void __rcu_synchronize(rcu_domain& __dom) noexcept;

_LIBCPP_EXPORTED_FROM_ABI void __rcu_barrier(rcu_domain& __dom) noexcept;

struct __rcu_domain_access {
  _LIBCPP_HIDE_FROM_ABI static void __retire(rcu_domain& __dom, __rcu_node* __node) noexcept { __dom.__retire(__node); }

  _LIBCPP_HIDE_FROM_ABI static rcu_domain& __rcu_default_domain() noexcept {
    return rcu_domain::__rcu_default_domain();
  }

  _LIBCPP_HIDE_FROM_ABI static unique_ptr<rcu_domain::__impl>& __get_impl(rcu_domain& __dom) noexcept {
    return __dom.__pimpl_;
  }
};

struct __rcu_node {
  using __cb_type _LIBCPP_NODEBUG = void(__rcu_node*);
  __cb_type* __callback_          = [](__rcu_node*) {};
  __rcu_node* __next_             = nullptr;
};

template <class _Tp, class _Deleter>
struct __rcu_node_with_deleter : __rcu_node {
  _Tp* __obj_;
  _LIBCPP_NO_UNIQUE_ADDRESS _Deleter __deleter_ = _Deleter();

  _LIBCPP_HIDE_FROM_ABI __rcu_node_with_deleter(_Tp* __obj, _Deleter __deleter) : __obj_(__obj), __deleter_(__deleter) {
    __callback_ = &__rcu_node_with_deleter::__destroy;
  }

  _LIBCPP_HIDE_FROM_ABI static void __destroy(__rcu_node* __node) {
    auto __self = static_cast<__rcu_node_with_deleter*>(__node);
    __self->__deleter_(__self->__obj_);
    delete __self;
  }
};

inline _LIBCPP_HIDE_FROM_ABI rcu_domain& rcu_default_domain() noexcept {
  return __rcu_domain_access::__rcu_default_domain();
}

inline _LIBCPP_HIDE_FROM_ABI void rcu_synchronize(rcu_domain& __dom = rcu_default_domain()) noexcept {
  std::__rcu_synchronize(__dom);
}

inline _LIBCPP_HIDE_FROM_ABI void rcu_barrier(rcu_domain& __dom = rcu_default_domain()) noexcept {
  std::__rcu_barrier(__dom);
}

template <class _Tp, class _Dp = default_delete<_Tp>>
_LIBCPP_HIDE_FROM_ABI void rcu_retire(_Tp* __tp, _Dp __deleter = _Dp(), rcu_domain& __dom = rcu_default_domain()) {
  static_assert(std::is_move_constructible_v<_Dp>);
  static_assert(requires(_Dp __dp, _Tp* __ptr) { __dp(__ptr); }, "Deleter must be callable with a pointer");

  auto* __node = new __rcu_node_with_deleter<_Tp, _Dp>(__tp, std::move(__deleter));
  __rcu_domain_access::__retire(__dom, __node);
}

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RCU_RCU_DOMAIN_H
