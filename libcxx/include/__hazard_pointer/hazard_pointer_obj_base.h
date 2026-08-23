// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___HAZARD_POINTER_HAZARD_POINTER_OBJ_BASE_H
#define _LIBCPP___HAZARD_POINTER_HAZARD_POINTER_OBJ_BASE_H

#include <__assert>
#include <__concepts/same_as.h>
#include <__config>
#include <__hazard_pointer/domain.h>
#include <__memory/unique_ptr.h>
#include <__type_traits/is_assignable.h>
#include <__type_traits/is_class.h>
#include <__type_traits/is_constructible.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS

_LIBCPP_BEGIN_NAMESPACE_STD

class hazard_pointer;

template <class _Tp, class _Dp = default_delete<_Tp>>
class _LIBCPP_NO_SPECIALIZATIONS hazard_pointer_obj_base;

// Deduction-only helper (declared, never defined): given a T*, find *the* hazard_pointer_obj_base base
// through a derived-to-base conversion. Deduction fails when there is no such base or more than one
// (ambiguous), the return type is ill-formed when the base is virtual, and the conversion is inaccessible
// when the base is not public.
template <class _Tp, class _Dp>
auto __hazard_pointer_protectable_check(hazard_pointer_obj_base<_Tp, _Dp>* __base)
    -> decltype(static_cast<_Tp*>(__base));

// [saferecl.hp.general]/2: a class type T is hazard-protectable if it has exactly one base class of type
// hazard_pointer_obj_base<T, D> for some D, that base is public and non-virtual, and it has no base classes
// of type hazard_pointer_obj_base<T2, D2> for any other combination T2, D2.
template <class _Tp>
concept __hazard_protectable = is_class_v<_Tp> && requires(_Tp* __p) {
  { std::__hazard_pointer_protectable_check(__p) } -> same_as<_Tp*>;
};

template <class _Tp, class _Dp>
class _LIBCPP_AVAILABILITY_HAZARD_POINTER _LIBCPP_NO_SPECIALIZATIONS hazard_pointer_obj_base
    : private __hazard_pointer_obj_node {
  static_assert(is_default_constructible_v<_Dp> && is_move_assignable_v<_Dp>,
                "hazard_pointer_obj_base<T, D>: D must be Cpp17DefaultConstructible and Cpp17MoveAssignable");

  // hazard_pointer::__node_of performs the derived-to-private-base conversion.
  friend class hazard_pointer;

  _LIBCPP_NO_UNIQUE_ADDRESS _Dp __deleter_ = _Dp();

public:
  _LIBCPP_HIDE_FROM_ABI void retire(_Dp __d = _Dp()) noexcept {
    static_assert(__hazard_protectable<_Tp>,
                  "hazard_pointer_obj_base<T, D>::retire(): T must be a hazard-protectable type (a class with exactly "
                  "one public, non-virtual hazard_pointer_obj_base<T, D> base)");
    _LIBCPP_ASSERT_VALID_DEALLOCATION(
        this->__next_ == this, "hazard_pointer_obj_base::retire(): object has already been retired");
    __deleter_       = std::move(__d);
    this->__reclaim_ = [](__hazard_pointer_obj_node* __node) noexcept {
      hazard_pointer_obj_base* __base = static_cast<hazard_pointer_obj_base*>(__node);
      __base->__deleter_(static_cast<_Tp*>(__base));
    };
    std::__hazard_pointer_retire(this);
  }

protected:
  _LIBCPP_HIDE_FROM_ABI hazard_pointer_obj_base()                                          = default;
  _LIBCPP_HIDE_FROM_ABI hazard_pointer_obj_base(const hazard_pointer_obj_base&)            = default;
  _LIBCPP_HIDE_FROM_ABI hazard_pointer_obj_base(hazard_pointer_obj_base&&)                 = default;
  _LIBCPP_HIDE_FROM_ABI hazard_pointer_obj_base& operator=(const hazard_pointer_obj_base&) = default;
  _LIBCPP_HIDE_FROM_ABI hazard_pointer_obj_base& operator=(hazard_pointer_obj_base&&)      = default;
  _LIBCPP_HIDE_FROM_ABI ~hazard_pointer_obj_base()                                         = default;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS

_LIBCPP_POP_MACROS

#endif // _LIBCPP___HAZARD_POINTER_HAZARD_POINTER_OBJ_BASE_H
