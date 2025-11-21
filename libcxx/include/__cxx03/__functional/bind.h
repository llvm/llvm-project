// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___FUNCTIONAL_BIND_H
#define _LIBCPP___CXX03___FUNCTIONAL_BIND_H

#include <__cxx03/__config>
#include <__cxx03/__functional/weak_result_type.h>
#include <__cxx03/__fwd/functional.h>
#include <__cxx03/__type_traits/decay.h>
#include <__cxx03/__type_traits/invoke.h>
#include <__cxx03/__type_traits/is_reference_wrapper.h>
#include <__cxx03/__type_traits/is_void.h>
#include <__cxx03/__type_traits/remove_cvref.h>
#include <__cxx03/cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct is_bind_expression
    : _If< _IsSame<_Tp, __remove_cvref_t<_Tp> >::value, false_type, is_bind_expression<__remove_cvref_t<_Tp> > > {};

template <class _Tp>
struct is_placeholder
    : _If< _IsSame<_Tp, __remove_cvref_t<_Tp> >::value,
           integral_constant<int, 0>,
           is_placeholder<__remove_cvref_t<_Tp> > > {};

namespace placeholders {

template <int _Np>
struct __ph {};

// C++17 recommends that we implement placeholders as `inline constexpr`, but allows
// implementing them as `extern <implementation-defined>`. Libc++ implements them as
// `extern const` in all standard modes to avoid an ABI break in C++03: making them
// `inline constexpr` requires removing their definition in the shared library to
// avoid ODR violations, which is an ABI break.
//
// In practice, since placeholders are empty, `extern const` is almost impossible
// to distinguish from `inline constexpr` from a usage stand point.
_LIBCPP_EXPORTED_FROM_ABI extern const __ph<1> _1;
_LIBCPP_EXPORTED_FROM_ABI extern const __ph<2> _2;
_LIBCPP_EXPORTED_FROM_ABI extern const __ph<3> _3;
_LIBCPP_EXPORTED_FROM_ABI extern const __ph<4> _4;
_LIBCPP_EXPORTED_FROM_ABI extern const __ph<5> _5;
_LIBCPP_EXPORTED_FROM_ABI extern const __ph<6> _6;
_LIBCPP_EXPORTED_FROM_ABI extern const __ph<7> _7;
_LIBCPP_EXPORTED_FROM_ABI extern const __ph<8> _8;
_LIBCPP_EXPORTED_FROM_ABI extern const __ph<9> _9;
_LIBCPP_EXPORTED_FROM_ABI extern const __ph<10> _10;

} // namespace placeholders

template <int _Np>
struct is_placeholder<placeholders::__ph<_Np> > : public integral_constant<int, _Np> {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___FUNCTIONAL_BIND_H
