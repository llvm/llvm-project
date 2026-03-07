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
#include <__functional/function.h>
#include <__memory/unique_ptr.h> // for default_delete
#include <__rcu/rcu_domain.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

template <class _Tp, class _Dp = default_delete<_Tp>>
class rcu_obj_base : private __rcu_node {
public:
  _LIBCPP_HIDE_FROM_ABI void retire(_Dp __deleter = _Dp(), rcu_domain& __dom = rcu_default_domain()) noexcept {
    auto __ptr = static_cast<_Tp*>(this);

    // todo: std::function can throw on the assignment. Perhaps we can store
    //       the deleter in the class and either use virtual function or use
    //       function_ref here.
    __callback_ = [__ptr, __deleter = std::move(__deleter)]() mutable { __deleter(__ptr); };
    __dom.__retire(this);
  }

protected:
  _LIBCPP_HIDE_FROM_ABI rcu_obj_base()                               = default;
  _LIBCPP_HIDE_FROM_ABI rcu_obj_base(const rcu_obj_base&)            = default;
  _LIBCPP_HIDE_FROM_ABI rcu_obj_base(rcu_obj_base&&)                 = default;
  _LIBCPP_HIDE_FROM_ABI rcu_obj_base& operator=(const rcu_obj_base&) = default;
  _LIBCPP_HIDE_FROM_ABI rcu_obj_base& operator=(rcu_obj_base&&)      = default;
  _LIBCPP_HIDE_FROM_ABI ~rcu_obj_base()                              = default;
};

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RCU_RCU_OBJ_BASE_H
