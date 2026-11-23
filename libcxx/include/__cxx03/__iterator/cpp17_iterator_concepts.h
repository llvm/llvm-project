//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ITERATOR_CPP17_ITERATOR_CONCEPTS_H
#define _LIBCPP___CXX03___ITERATOR_CPP17_ITERATOR_CONCEPTS_H

#include <__cxx03/__concepts/boolean_testable.h>
#include <__cxx03/__concepts/convertible_to.h>
#include <__cxx03/__concepts/same_as.h>
#include <__cxx03/__config>
#include <__cxx03/__iterator/iterator_traits.h>
#include <__cxx03/__type_traits/is_constructible.h>
#include <__cxx03/__type_traits/is_convertible.h>
#include <__cxx03/__type_traits/is_signed.h>
#include <__cxx03/__type_traits/is_void.h>
#include <__cxx03/__utility/as_const.h>
#include <__cxx03/__utility/forward.h>
#include <__cxx03/__utility/move.h>
#include <__cxx03/__utility/swap.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

#  define _LIBCPP_REQUIRE_CPP17_INPUT_ITERATOR(iter_t, message) static_assert(true)
#  define _LIBCPP_REQUIRE_CPP17_OUTPUT_ITERATOR(iter_t, write_t, message) static_assert(true)
#  define _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(iter_t, message) static_assert(true)
#  define _LIBCPP_REQUIRE_CPP17_BIDIRECTIONAL_ITERATOR(iter_t, message) static_assert(true)
#  define _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(iter_t, message) static_assert(true)

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___ITERATOR_CPP17_ITERATOR_CONCEPTS_H
