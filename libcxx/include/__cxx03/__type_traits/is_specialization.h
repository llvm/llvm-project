// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___TYPE_TRAITS_IS_SPECIALIZATION
#define _LIBCPP___CXX03___TYPE_TRAITS_IS_SPECIALIZATION

// This contains parts of P2098R1 but is based on MSVC STL's implementation.
//
// The paper has been rejected
//   We will not pursue P2098R0 (std::is_specialization_of) at this time; we'd
//   like to see a solution to this problem, but it requires language evolution
//   too.
//
// Since it is expected a real solution will be provided in the future only the
// minimal part is implemented.
//
// Note a cvref qualified _Tp is never considered a specialization.

#include <__cxx03/__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___TYPE_TRAITS_IS_SPECIALIZATION
