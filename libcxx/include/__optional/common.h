// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_OPTIONAL_COMMON_H
#define _LIBCPP_OPTIONAL_COMMON_H

#include <__assert>
#include <__config>
#include <__exception/exception.h>
#include <__fwd/format.h>
#include <__ranges/enable_borrowed_range.h>
#include <__ranges/enable_view.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_object.h>
#include <__type_traits/is_reference.h>

#include <__fwd/optional.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

namespace std // purposefully not using versioning namespace
{

class _LIBCPP_EXPORTED_FROM_ABI bad_optional_access : public exception {
public:
  _LIBCPP_HIDE_FROM_ABI bad_optional_access() _NOEXCEPT                                      = default;
  _LIBCPP_HIDE_FROM_ABI bad_optional_access(const bad_optional_access&) _NOEXCEPT            = default;
  _LIBCPP_HIDE_FROM_ABI bad_optional_access& operator=(const bad_optional_access&) _NOEXCEPT = default;
  // Get the key function ~bad_optional_access() into the dylib
  ~bad_optional_access() _NOEXCEPT override;
  [[__nodiscard__]] const char* what() const _NOEXCEPT override;
};

} // namespace std

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

[[noreturn]] inline void __throw_bad_optional_access() {
#  if _LIBCPP_HAS_EXCEPTIONS
  throw bad_optional_access();
#  else
  _LIBCPP_VERBOSE_ABORT("bad_optional_access was thrown in -fno-exceptions mode");
#  endif
}

struct __optional_construct_from_invoke_tag {};

template <class _Tp>
struct __is_std_optional : false_type {};

template <class _Tp>
struct __is_std_optional<optional<_Tp>> : true_type {};

#  if _LIBCPP_STD_VER < 26
template <class _Tp>
inline constexpr bool __is_valid_optional_contained_type = is_object_v<_Tp>;
#  else
template <class _Tp>
inline constexpr bool __is_valid_optional_contained_type = is_object_v<_Tp> || is_lvalue_reference_v<_Tp>;
#  endif

#  if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_OPTIONAL_ITERATOR

template <class _Tp>
constexpr bool ranges::enable_view<optional<_Tp>> = true;

template <class _Tp>
constexpr range_format format_kind<optional<_Tp>> = range_format::disabled;

template <class _Tp>
constexpr bool ranges::enable_borrowed_range<optional<_Tp&>> = true;

#  endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_OPTIONAL_ITERATOR

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif
