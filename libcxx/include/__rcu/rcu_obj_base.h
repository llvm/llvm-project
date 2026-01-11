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
#include <__rcu/rcu_list.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

template <class T, class D = default_delete<T>>
class rcu_obj_base : private __rcu_node {
public:
  void retire(D d = D(), rcu_domain& dom = rcu_default_domain()) noexcept {
    auto ptr = static_cast<T*>(this);
    dom.__retire_callback(
        [ptr, d = std::move(d)]() mutable { d(ptr); });
  }

protected:
  rcu_obj_base()                               = default;
  rcu_obj_base(const rcu_obj_base&)            = default;
  rcu_obj_base(rcu_obj_base&&)                 = default;
  rcu_obj_base& operator=(const rcu_obj_base&) = default;
  rcu_obj_base& operator=(rcu_obj_base&&)      = default;
  ~rcu_obj_base()                              = default;

};

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RCU_RCU_OBJ_BASE_H
