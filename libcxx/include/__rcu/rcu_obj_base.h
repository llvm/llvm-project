// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RCU_RCU_OBJ_BASE_H
#define _LIBCPP___RCU_RCU_OBJ_BASE_H

#include <__config>
#include <__memory/unique_ptr.h> // for default_delete
#include <__rcu/rcu_domain.h>
#include <__type_traits/is_assignable.h>
#include <__type_traits/is_constructible.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

template <class _Tp, class _Dp = default_delete<_Tp>>
class rcu_obj_base : private __rcu_node {
  static_assert(std::is_default_constructible_v<_Dp>);
  static_assert(std::is_move_assignable_v<_Dp>);
  static_assert(requires(_Tp* __ptr, _Dp __d) { __d(__ptr); }, "Deleter must be callable with an object pointer.");

public:
  _LIBCPP_HIDE_FROM_ABI void retire(_Dp __deleter = _Dp(), rcu_domain& __dom = rcu_default_domain()) noexcept {
    static_assert(std::is_base_of_v<rcu_obj_base, _Tp>, "T must be an rcu-protectable type.");
    __deleter_  = std::move(__deleter);
    __callback_ = &rcu_obj_base::__destroy;
    __dom.__retire(this);
  }

protected:
  _LIBCPP_HIDE_FROM_ABI rcu_obj_base()                               = default;
  _LIBCPP_HIDE_FROM_ABI rcu_obj_base(const rcu_obj_base&)            = default;
  _LIBCPP_HIDE_FROM_ABI rcu_obj_base(rcu_obj_base&&)                 = default;
  _LIBCPP_HIDE_FROM_ABI rcu_obj_base& operator=(const rcu_obj_base&) = default;
  _LIBCPP_HIDE_FROM_ABI rcu_obj_base& operator=(rcu_obj_base&&)      = default;
  _LIBCPP_HIDE_FROM_ABI ~rcu_obj_base()                              = default;

private:
  _LIBCPP_HIDE_FROM_ABI static void __destroy(__rcu_node* __node) {
    auto __self = static_cast<_Tp*>(__node);
    __self->__deleter_(__self);
  }

  _LIBCPP_NO_UNIQUE_ADDRESS _Dp __deleter_ = _Dp();
};

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RCU_RCU_OBJ_BASE_H
