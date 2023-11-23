//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_NO_DESTROY_H
#define _LIBCPP___UTILITY_NO_DESTROY_H

#include <__config>
#include <__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

struct __uninitialized_tag {};
struct __zero_initialized_tag {};

// This class stores an object of type T but never destroys it.
//
// This is akin to using __attribute__((no_destroy)), except that it is possible
// to control the lifetime of the object with more flexibility by deciding e.g.
// whether to initialize the object at construction or to defer to a later
// initialization using __emplace.
template <class _Tp>
struct __no_destroy {
  _LIBCPP_HIDE_FROM_ABI explicit __no_destroy(__uninitialized_tag) {}
  _LIBCPP_HIDE_FROM_ABI explicit constexpr __no_destroy(__zero_initialized_tag) : __buf{} {}

  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI explicit __no_destroy(_Args&&... __args) {
    new (&__buf) _Tp(std::forward<_Args>(__args)...);
  }

  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI _Tp& __emplace(_Args&&... __args) {
    return *new (&__buf) _Tp(std::forward<_Args>(__args)...);
  }

  _LIBCPP_HIDE_FROM_ABI _Tp& __get() { return *reinterpret_cast<_Tp*>(&__buf); }
  _LIBCPP_HIDE_FROM_ABI _Tp const& __get() const { return *reinterpret_cast<_Tp const*>(&__buf); }

private:
  _ALIGNAS_TYPE(_Tp) char __buf[sizeof(_Tp)];
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___UTILITY_NO_DESTROY_H
