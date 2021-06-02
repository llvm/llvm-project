// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCPP___ITERATOR_INDIRECT_CONCEPTS_H
#define _LIBCPP___ITERATOR_INDIRECT_CONCEPTS_H

#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/readable_traits.h>
#include <concepts>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if !defined(_LIBCPP_HAS_NO_RANGES)

template<class _Fp, class _It>
concept indirectly_unary_invocable =
  indirectly_readable<_It> &&
  copy_constructible<_Fp> &&
  invocable<_Fp&, iter_value_t<_It>&> &&
  invocable<_Fp&, iter_reference_t<_It>> &&
  invocable<_Fp&, iter_common_reference_t<_It>> &&
  common_reference_with<
    invoke_result_t<_Fp&, iter_value_t<_It>&>,
    invoke_result_t<_Fp&, iter_reference_t<_It>>>;

template<class _Fp, class _It>
concept indirectly_regular_unary_invocable =
  indirectly_readable<_It> &&
  copy_constructible<_Fp> &&
  regular_invocable<_Fp&, iter_value_t<_It>&> &&
  regular_invocable<_Fp&, iter_reference_t<_It>> &&
  regular_invocable<_Fp&, iter_common_reference_t<_It>> &&
  common_reference_with<
    invoke_result_t<_Fp&, iter_value_t<_It>&>,
    invoke_result_t<_Fp&, iter_reference_t<_It>>>;

template<class _Fp, class _It>
concept indirect_unary_predicate =
  indirectly_readable<_It> &&
  copy_constructible<_Fp> &&
  predicate<_Fp&, iter_value_t<_It>&> &&
  predicate<_Fp&, iter_reference_t<_It>> &&
  predicate<_Fp&, iter_common_reference_t<_It>>;

template<class _Fp, class _It1, class _It2>
concept indirect_binary_predicate =
  indirectly_readable<_It1> && indirectly_readable<_It2> &&
  copy_constructible<_Fp> &&
  predicate<_Fp&, iter_value_t<_It1>&, iter_value_t<_It2>&> &&
  predicate<_Fp&, iter_value_t<_It1>&, iter_reference_t<_It2>> &&
  predicate<_Fp&, iter_reference_t<_It1>, iter_value_t<_It2>&> &&
  predicate<_Fp&, iter_reference_t<_It1>, iter_reference_t<_It2>> &&
  predicate<_Fp&, iter_common_reference_t<_It1>, iter_common_reference_t<_It2>>;

template<class _Fp, class _It1, class _It2 = _It1>
concept indirect_equivalence_relation =
  indirectly_readable<_It1> && indirectly_readable<_It2> &&
  copy_constructible<_Fp> &&
  equivalence_relation<_Fp&, iter_value_t<_It1>&, iter_value_t<_It2>&> &&
  equivalence_relation<_Fp&, iter_value_t<_It1>&, iter_reference_t<_It2>> &&
  equivalence_relation<_Fp&, iter_reference_t<_It1>, iter_value_t<_It2>&> &&
  equivalence_relation<_Fp&, iter_reference_t<_It1>, iter_reference_t<_It2>> &&
  equivalence_relation<_Fp&, iter_common_reference_t<_It1>, iter_common_reference_t<_It2>>;

template<class _Fp, class _It1, class _It2 = _It1>
concept indirect_strict_weak_order =
  indirectly_readable<_It1> && indirectly_readable<_It2> &&
  copy_constructible<_Fp> &&
  strict_weak_order<_Fp&, iter_value_t<_It1>&, iter_value_t<_It2>&> &&
  strict_weak_order<_Fp&, iter_value_t<_It1>&, iter_reference_t<_It2>> &&
  strict_weak_order<_Fp&, iter_reference_t<_It1>, iter_value_t<_It2>&> &&
  strict_weak_order<_Fp&, iter_reference_t<_It1>, iter_reference_t<_It2>> &&
  strict_weak_order<_Fp&, iter_common_reference_t<_It1>, iter_common_reference_t<_It2>>;

template<class _Fp, class... _Its>
  requires (indirectly_readable<_Its> && ...) && invocable<_Fp, iter_reference_t<_Its>...>
using indirect_result_t = invoke_result_t<_Fp, iter_reference_t<_Its>...>;

#endif // !defined(_LIBCPP_HAS_NO_RANGES)

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ITERATOR_INDIRECT_CONCEPTS_H
