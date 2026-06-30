//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_REFERENCE_CONSTRUCTS_FROM_TEMPORARY_H
#define _LIBCPP___TYPE_TRAITS_REFERENCE_CONSTRUCTS_FROM_TEMPORARY_H

#include <__config>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_reference.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if defined(_LIBCPP_CLANG_VER) && _LIBCPP_CLANG_VER < 2300

// A non-reference type can never bind to a temporary, so the result is always `false` for such a
// `_Tp`. We short-circuit before reaching the builtin because Clang's `__reference_constructs_from_temporary`
// eagerly instantiates the construction of `_Up` (including the element's constructor exception
// specification) even when `_Tp` is not a reference, which can hard-error on misbehaved types.
//
// https://godbolt.org/z/4xz1ozKev
//
// TODO: Clang 23 fix this builtin, remove this guard once all supported clang versions include this fix.
template <class _Tp, class _Up, bool = is_reference<_Tp>::value>
inline const bool __reference_constructs_from_temporary_v = false;

template <class _Tp, class _Up>
inline const bool __reference_constructs_from_temporary_v<_Tp, _Up, true> =
    __reference_constructs_from_temporary(_Tp, _Up);

#else

template <class _Tp, class _Up>
inline const bool __reference_constructs_from_temporary_v = __reference_constructs_from_temporary(_Tp, _Up);

#endif // defined(_LIBCPP_CLANG_VER) && _LIBCPP_CLANG_VER >= 2300

#if _LIBCPP_STD_VER >= 23

template <class _Tp, class _Up>
struct _LIBCPP_NO_SPECIALIZATIONS reference_constructs_from_temporary
    : public bool_constant<__reference_constructs_from_temporary_v<_Tp, _Up>> {};

template <class _Tp, class _Up>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool reference_constructs_from_temporary_v =
    __reference_constructs_from_temporary_v<_Tp, _Up>;

#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_REFERENCE_CONSTRUCTS_FROM_TEMPORARY_H
