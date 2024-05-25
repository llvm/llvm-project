// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___OPTIONAL_COOPERATE_H
#define _LIBCPP___OPTIONAL_COOPERATE_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 17

namespace __optional {

struct __disengaged_construct {
  constexpr explicit __disengaged_construct() = default;
#  if _LIBCPP_STD_VER < 20

private:
  _LIBCPP_HIDE_FROM_ABI explicit __disengaged_construct(int, int) {} // This is not an aggregate
#  endif
};

/// To cooperate with std::optional<T>, specialize std::__optional::__cooperate and implement the member functions as
/// documented. Must be constructible from _Tp(__disengaged_construct, __disengaged_construct) to be in a "disengaged"
/// state. An object in a disengaged state never has its destructor called.
template <class _Tp>
struct __cooperate {
  // Return "true" if the cooperated layout should be used
  _LIBCPP_HIDE_FROM_ABI static constexpr bool __do_cooperate() { return false; }

  // Return if __v is not in a disengaged state
  _LIBCPP_HIDE_FROM_ABI static constexpr bool __is_engaged(const _Tp& /*__v*/) { return false; }

  // Given a __v where !__is_engaged(__v), act as if destroy_at(addressof(__v)); construct_at(addressof(__v), __args...)
  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI static constexpr void __construct_over(_Tp& /*__v*/, _Args&&... /*__args*/) {}

  // Given a __v, make it so !__is_engaged(__v). __v may already be disengaged
  template <class _Up>
  _LIBCPP_HIDE_FROM_ABI static constexpr void __disengage(_Tp& /*__v*/) {}
};

} // namespace __optional

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___OPTIONAL_COOPERATE_H
