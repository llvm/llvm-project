// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_CONCEPTS_H
#define _LIBCPP___ITERATOR_CONCEPTS_H

#include <__concepts/arithmetic.h>
#include <__concepts/assignable.h>
#include <__concepts/common_reference_with.h>
#include <__concepts/constructible.h>
#include <__concepts/copyable.h>
#include <__concepts/derived_from.h>
#include <__concepts/equality_comparable.h>
#include <__concepts/invocable.h>
#include <__concepts/movable.h>
#include <__concepts/predicate.h>
#include <__concepts/regular.h>
#include <__concepts/relation.h>
#include <__concepts/same_as.h>
#include <__concepts/semiregular.h>
#include <__concepts/totally_ordered.h>
#include <__config>
#include <__functional/invoke.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iter_move.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/readable_traits.h>
#include <__memory/pointer_traits.h>
#include <__type_traits/add_pointer.h>
#include <__type_traits/common_reference.h>
#include <__type_traits/is_pointer.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

// [iterator.concept.readable]
template <class _In>
concept __indirectly_readable_impl =
    requires(const _In __i) {
      typename iter_value_t<_In>;
      typename iter_reference_t<_In>;
      typename iter_rvalue_reference_t<_In>;
      { *__i } -> same_as<iter_reference_t<_In>>;
      { ranges::iter_move(__i) } -> same_as<iter_rvalue_reference_t<_In>>;
    } && common_reference_with<iter_reference_t<_In>&&, iter_value_t<_In>&> &&
    common_reference_with<iter_reference_t<_In>&&, iter_rvalue_reference_t<_In>&&> &&
    common_reference_with<iter_rvalue_reference_t<_In>&&, const iter_value_t<_In>&>;

template <class _In>
concept indirectly_readable = __indirectly_readable_impl<remove_cvref_t<_In>>;

template <indirectly_readable _Tp>
using iter_common_reference_t = common_reference_t<iter_reference_t<_Tp>, iter_value_t<_Tp>&>;

// [iterator.concept.writable]
template <class _Out, class _Tp>
concept indirectly_writable = requires(_Out&& __o, _Tp&& __t) {
  *__o                                             = std::forward<_Tp>(__t); // not required to be equality-preserving
  *std::forward<_Out>(__o)                         = std::forward<_Tp>(__t); // not required to be equality-preserving
  const_cast<const iter_reference_t<_Out>&&>(*__o) = std::forward<_Tp>(__t); // not required to be equality-preserving
  const_cast<const iter_reference_t<_Out>&&>(*std::forward<_Out>(__o)) =
      std::forward<_Tp>(__t); // not required to be equality-preserving
};

// [iterator.concept.winc]
template <class _Tp>
concept __integer_like = integral<_Tp> && !same_as<_Tp, bool>;

template <class _Tp>
concept __signed_integer_like = signed_integral<_Tp>;

template <class _Iter>
concept weakly_incrementable =
    // TODO: remove this once the clang bug is fixed (bugs.llvm.org/PR48173).
    !same_as<_Iter, bool> && // Currently, clang does not handle bool correctly.
    movable<_Iter> && requires(_Iter __i) {
      typename iter_difference_t<_Iter>;
      requires __signed_integer_like<iter_difference_t<_Iter>>;
      { ++__i } -> same_as<_Iter&>; // not required to be equality-preserving
      __i++;                      // not required to be equality-preserving
    };

// [iterator.concept.inc]
template <class _Iter>
concept incrementable = regular<_Iter> && weakly_incrementable<_Iter> && requires(_Iter __i) {
  { __i++ } -> same_as<_Iter>;
};

// [iterator.concept.iterator]
template <class _Iter>
concept input_or_output_iterator = requires(_Iter __i) {
  { *__i } -> __can_reference;
} && weakly_incrementable<_Iter>;

// [iterator.concept.sentinel]
template <class _Sent, class _Iter>
concept sentinel_for =
    semiregular<_Sent> && input_or_output_iterator<_Iter> && __weakly_equality_comparable_with<_Sent, _Iter>;

template <class, class>
inline constexpr bool disable_sized_sentinel_for = false;

template <class _Sent, class _Iter>
concept sized_sentinel_for =
    sentinel_for<_Sent, _Iter> && !disable_sized_sentinel_for<remove_cv_t<_Sent>, remove_cv_t<_Iter>> &&
    requires(const _Iter& __i, const _Sent& __s) {
      { __s - __i } -> same_as<iter_difference_t<_Iter>>;
      { __i - __s } -> same_as<iter_difference_t<_Iter>>;
    };

// [iterator.concept.input]
template <class _Iter>
concept input_iterator = input_or_output_iterator<_Iter> && indirectly_readable<_Iter> && requires {
  typename _ITER_CONCEPT<_Iter>;
} && derived_from<_ITER_CONCEPT<_Iter>, input_iterator_tag>;

// [iterator.concept.output]
template <class _Iter, class _Tp>
concept output_iterator =
    input_or_output_iterator<_Iter> && indirectly_writable<_Iter, _Tp> && requires(_Iter __it, _Tp&& __t) {
      *__it++ = std::forward<_Tp>(__t); // not required to be equality-preserving
    };

// [iterator.concept.forward]
template <class _Iter>
concept forward_iterator =
    input_iterator<_Iter> && derived_from<_ITER_CONCEPT<_Iter>, forward_iterator_tag> && incrementable<_Iter> &&
    sentinel_for<_Iter, _Iter>;

// [iterator.concept.bidir]
template <class _Iter>
concept bidirectional_iterator =
    forward_iterator<_Iter> && derived_from<_ITER_CONCEPT<_Iter>, bidirectional_iterator_tag> && requires(_Iter __i) {
      { --__i } -> same_as<_Iter&>;
      { __i-- } -> same_as<_Iter>;
    };

template <class _Iter>
concept random_access_iterator =
    bidirectional_iterator<_Iter> && derived_from<_ITER_CONCEPT<_Iter>, random_access_iterator_tag> &&
    totally_ordered<_Iter> && sized_sentinel_for<_Iter, _Iter> &&
    requires(_Iter __i, const _Iter __j, const iter_difference_t<_Iter> __n) {
      { __i += __n } -> same_as<_Iter&>;
      { __j + __n } -> same_as<_Iter>;
      { __n + __j } -> same_as<_Iter>;
      { __i -= __n } -> same_as<_Iter&>;
      { __j - __n } -> same_as<_Iter>;
      { __j[__n] } -> same_as<iter_reference_t<_Iter>>;
    };

template <class _Iter>
concept contiguous_iterator =
    random_access_iterator<_Iter> && derived_from<_ITER_CONCEPT<_Iter>, contiguous_iterator_tag> &&
    is_lvalue_reference_v<iter_reference_t<_Iter>> &&
    same_as<iter_value_t<_Iter>, remove_cvref_t<iter_reference_t<_Iter>>> && requires(const _Iter& __i) {
      { std::to_address(__i) } -> same_as<add_pointer_t<iter_reference_t<_Iter>>>;
    };

template <class _Iter>
concept __has_arrow = input_iterator<_Iter> && (is_pointer_v<_Iter> || requires(_Iter __i) { __i.operator->(); });

// [indirectcallable.indirectinvocable]
template <class _Func, class _It>
concept indirectly_unary_invocable =
    indirectly_readable<_It> && copy_constructible<_Func> && invocable<_Func&, iter_value_t<_It>&> &&
    invocable<_Func&, iter_reference_t<_It>> && invocable<_Func&, iter_common_reference_t<_It>> &&
    common_reference_with< invoke_result_t<_Func&, iter_value_t<_It>&>, invoke_result_t<_Func&, iter_reference_t<_It>>>;

template <class _Func, class _It>
concept indirectly_regular_unary_invocable =
    indirectly_readable<_It> && copy_constructible<_Func> && regular_invocable<_Func&, iter_value_t<_It>&> &&
    regular_invocable<_Func&, iter_reference_t<_It>> && regular_invocable<_Func&, iter_common_reference_t<_It>> &&
    common_reference_with< invoke_result_t<_Func&, iter_value_t<_It>&>, invoke_result_t<_Func&, iter_reference_t<_It>>>;

template <class _Func, class _It>
concept indirect_unary_predicate =
    indirectly_readable<_It> && copy_constructible<_Func> && predicate<_Func&, iter_value_t<_It>&> &&
    predicate<_Func&, iter_reference_t<_It>> && predicate<_Func&, iter_common_reference_t<_It>>;

template <class _Func, class _Iter1, class _Iter2>
concept indirect_binary_predicate =
    indirectly_readable<_Iter1> && indirectly_readable<_Iter2> && copy_constructible<_Func> &&
    predicate<_Func&, iter_value_t<_Iter1>&, iter_value_t<_Iter2>&> &&
    predicate<_Func&, iter_value_t<_Iter1>&, iter_reference_t<_Iter2>> &&
    predicate<_Func&, iter_reference_t<_Iter1>, iter_value_t<_Iter2>&> &&
    predicate<_Func&, iter_reference_t<_Iter1>, iter_reference_t<_Iter2>> &&
    predicate<_Func&, iter_common_reference_t<_Iter1>, iter_common_reference_t<_Iter2>>;

template <class _Func, class _Iter1, class _Iter2 = _Iter1>
concept indirect_equivalence_relation =
    indirectly_readable<_Iter1> && indirectly_readable<_Iter2> && copy_constructible<_Func> &&
    equivalence_relation<_Func&, iter_value_t<_Iter1>&, iter_value_t<_Iter2>&> &&
    equivalence_relation<_Func&, iter_value_t<_Iter1>&, iter_reference_t<_Iter2>> &&
    equivalence_relation<_Func&, iter_reference_t<_Iter1>, iter_value_t<_Iter2>&> &&
    equivalence_relation<_Func&, iter_reference_t<_Iter1>, iter_reference_t<_Iter2>> &&
    equivalence_relation<_Func&, iter_common_reference_t<_Iter1>, iter_common_reference_t<_Iter2>>;

template <class _Func, class _Iter1, class _Iter2 = _Iter1>
concept indirect_strict_weak_order =
    indirectly_readable<_Iter1> && indirectly_readable<_Iter2> && copy_constructible<_Func> &&
    strict_weak_order<_Func&, iter_value_t<_Iter1>&, iter_value_t<_Iter2>&> &&
    strict_weak_order<_Func&, iter_value_t<_Iter1>&, iter_reference_t<_Iter2>> &&
    strict_weak_order<_Func&, iter_reference_t<_Iter1>, iter_value_t<_Iter2>&> &&
    strict_weak_order<_Func&, iter_reference_t<_Iter1>, iter_reference_t<_Iter2>> &&
    strict_weak_order<_Func&, iter_common_reference_t<_Iter1>, iter_common_reference_t<_Iter2>>;

template <class _Func, class... _Its>
  requires(indirectly_readable<_Its> && ...) && invocable<_Func, iter_reference_t<_Its>...>
using indirect_result_t = invoke_result_t<_Func, iter_reference_t<_Its>...>;

template <class _In, class _Out>
concept indirectly_movable = indirectly_readable<_In> && indirectly_writable<_Out, iter_rvalue_reference_t<_In>>;

template <class _In, class _Out>
concept indirectly_movable_storable =
    indirectly_movable<_In, _Out> && indirectly_writable<_Out, iter_value_t<_In>> && movable<iter_value_t<_In>> &&
    constructible_from<iter_value_t<_In>, iter_rvalue_reference_t<_In>> &&
    assignable_from<iter_value_t<_In>&, iter_rvalue_reference_t<_In>>;

template <class _In, class _Out>
concept indirectly_copyable = indirectly_readable<_In> && indirectly_writable<_Out, iter_reference_t<_In>>;

template <class _In, class _Out>
concept indirectly_copyable_storable =
    indirectly_copyable<_In, _Out> && indirectly_writable<_Out, iter_value_t<_In>&> &&
    indirectly_writable<_Out, const iter_value_t<_In>&> && indirectly_writable<_Out, iter_value_t<_In>&&> &&
    indirectly_writable<_Out, const iter_value_t<_In>&&> && copyable<iter_value_t<_In>> &&
    constructible_from<iter_value_t<_In>, iter_reference_t<_In>> &&
    assignable_from<iter_value_t<_In>&, iter_reference_t<_In>>;

// Note: indirectly_swappable is located in iter_swap.h to prevent a dependency cycle
// (both iter_swap and indirectly_swappable require indirectly_readable).

#endif // _LIBCPP_STD_VER >= 20

template <class _Tp>
using __has_random_access_iterator_category_or_concept
#if _LIBCPP_STD_VER >= 20
    = integral_constant<bool, random_access_iterator<_Tp>>;
#else  // _LIBCPP_STD_VER < 20
    = __has_random_access_iterator_category<_Tp>;
#endif // _LIBCPP_STD_VER

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ITERATOR_CONCEPTS_H
