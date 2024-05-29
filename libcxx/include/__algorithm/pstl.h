//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_H
#define _LIBCPP___ALGORITHM_PSTL_H

#include <__algorithm/copy_n.h>
#include <__algorithm/count.h>
#include <__algorithm/equal.h>
#include <__algorithm/fill_n.h>
#include <__algorithm/for_each.h>
#include <__algorithm/for_each_n.h>
#include <__algorithm/pstl_frontend_dispatch.h>
#include <__atomic/atomic.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/cpp17_iterator_concepts.h>
#include <__iterator/iterator_traits.h>
#include <__numeric/pstl.h>
#include <__pstl/configuration.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_execution_policy.h>
#include <__type_traits/is_trivially_copyable.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/empty.h>
#include <__utility/move.h>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Predicate,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__remove_cvref_t<_ForwardIterator>>
__find_if(_ExecutionPolicy&&, _ForwardIterator&& __first, _ForwardIterator&& __last, _Predicate&& __pred) noexcept {
  using _Backend = typename __select_backend<_RawPolicy>::type;
  return std::__pstl_find_if<_RawPolicy>(_Backend{}, std::move(__first), std::move(__last), std::move(__pred));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Predicate,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _ForwardIterator
find_if(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "find_if requires ForwardIterators");
  auto __res = std::__find_if(__policy, std::move(__first), std::move(__last), std::move(__pred));
  if (!__res)
    std::__throw_bad_alloc();
  return *std::move(__res);
}

template <class>
void __pstl_any_of(); // declaration needed for the frontend dispatch below

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Predicate,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<bool> __any_of(
    _ExecutionPolicy&& __policy, _ForwardIterator&& __first, _ForwardIterator&& __last, _Predicate&& __pred) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_any_of, _RawPolicy),
      [&](_ForwardIterator __g_first, _ForwardIterator __g_last, _Predicate __g_pred) -> optional<bool> {
        auto __res = std::__find_if(__policy, __g_first, __g_last, __g_pred);
        if (!__res)
          return nullopt;
        return *__res != __g_last;
      },
      std::move(__first),
      std::move(__last),
      std::move(__pred));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Predicate,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI bool
any_of(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "any_of requires a ForwardIterator");
  auto __res = std::__any_of(__policy, std::move(__first), std::move(__last), std::move(__pred));
  if (!__res)
    std::__throw_bad_alloc();
  return *std::move(__res);
}

template <class>
void __pstl_all_of(); // declaration needed for the frontend dispatch below

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Pred,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<bool>
__all_of(_ExecutionPolicy&& __policy, _ForwardIterator&& __first, _ForwardIterator&& __last, _Pred&& __pred) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_all_of, _RawPolicy),
      [&](_ForwardIterator __g_first, _ForwardIterator __g_last, _Pred __g_pred) -> optional<bool> {
        auto __res = std::__any_of(__policy, __g_first, __g_last, [&](__iter_reference<_ForwardIterator> __value) {
          return !__g_pred(__value);
        });
        if (!__res)
          return nullopt;
        return !*__res;
      },
      std::move(__first),
      std::move(__last),
      std::move(__pred));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Pred,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI bool
all_of(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Pred __pred) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "all_of requires a ForwardIterator");
  auto __res = std::__all_of(__policy, std::move(__first), std::move(__last), std::move(__pred));
  if (!__res)
    std::__throw_bad_alloc();
  return *std::move(__res);
}

template <class>
void __pstl_none_of(); // declaration needed for the frontend dispatch below

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Pred,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<bool>
__none_of(_ExecutionPolicy&& __policy, _ForwardIterator&& __first, _ForwardIterator&& __last, _Pred&& __pred) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_none_of, _RawPolicy),
      [&](_ForwardIterator __g_first, _ForwardIterator __g_last, _Pred __g_pred) -> optional<bool> {
        auto __res = std::__any_of(__policy, __g_first, __g_last, __g_pred);
        if (!__res)
          return nullopt;
        return !*__res;
      },
      std::move(__first),
      std::move(__last),
      std::move(__pred));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Pred,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI bool
none_of(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Pred __pred) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "none_of requires a ForwardIterator");
  auto __res = std::__none_of(__policy, std::move(__first), std::move(__last), std::move(__pred));
  if (!__res)
    std::__throw_bad_alloc();
  return *std::move(__res);
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _UnaryOperation,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__remove_cvref_t<_ForwardOutIterator>>
__transform(_ExecutionPolicy&&,
            _ForwardIterator&& __first,
            _ForwardIterator&& __last,
            _ForwardOutIterator&& __result,
            _UnaryOperation&& __op) noexcept {
  using _Backend = typename __select_backend<_RawPolicy>::type;
  return std::__pstl_transform<_RawPolicy>(
      _Backend{}, std::move(__first), std::move(__last), std::move(__result), std::move(__op));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _UnaryOperation,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _ForwardOutIterator transform(
    _ExecutionPolicy&& __policy,
    _ForwardIterator __first,
    _ForwardIterator __last,
    _ForwardOutIterator __result,
    _UnaryOperation __op) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "transform requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardOutIterator, "transform requires an OutputIterator");
  _LIBCPP_REQUIRE_CPP17_OUTPUT_ITERATOR(
      _ForwardOutIterator, decltype(__op(*__first)), "transform requires an OutputIterator");
  auto __res = std::__transform(__policy, std::move(__first), std::move(__last), std::move(__result), std::move(__op));
  if (!__res)
    std::__throw_bad_alloc();
  return *std::move(__res);
}

template <class _ExecutionPolicy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          class _ForwardOutIterator,
          class _BinaryOperation,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI optional<__remove_cvref_t<_ForwardOutIterator>>
__transform(_ExecutionPolicy&&,
            _ForwardIterator1&& __first1,
            _ForwardIterator1&& __last1,
            _ForwardIterator2&& __first2,
            _ForwardOutIterator&& __result,
            _BinaryOperation&& __op) noexcept {
  using _Backend = typename __select_backend<_RawPolicy>::type;
  return std::__pstl_transform<_RawPolicy>(
      _Backend{}, std::move(__first1), std::move(__last1), std::move(__first2), std::move(__result), std::move(__op));
}

template <class _ExecutionPolicy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          class _ForwardOutIterator,
          class _BinaryOperation,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _ForwardOutIterator transform(
    _ExecutionPolicy&& __policy,
    _ForwardIterator1 __first1,
    _ForwardIterator1 __last1,
    _ForwardIterator2 __first2,
    _ForwardOutIterator __result,
    _BinaryOperation __op) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator1, "transform requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator2, "transform requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardOutIterator, "transform requires an OutputIterator");
  _LIBCPP_REQUIRE_CPP17_OUTPUT_ITERATOR(
      _ForwardOutIterator, decltype(__op(*__first1, *__first2)), "transform requires an OutputIterator");
  auto __res = std::__transform(
      __policy, std::move(__first1), std::move(__last1), std::move(__first2), std::move(__result), std::move(__op));
  if (!__res)
    std::__throw_bad_alloc();
  return *std::move(__res);
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Function,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
__for_each(_ExecutionPolicy&&, _ForwardIterator&& __first, _ForwardIterator&& __last, _Function&& __func) noexcept {
  using _Backend = typename __select_backend<_RawPolicy>::type;
  return std::__pstl_for_each<_RawPolicy>(_Backend{}, std::move(__first), std::move(__last), std::move(__func));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Function,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void
for_each(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Function __func) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "for_each requires ForwardIterators");
  if (!std::__for_each(__policy, std::move(__first), std::move(__last), std::move(__func)))
    std::__throw_bad_alloc();
}

// TODO: Use the std::copy/move shenanigans to forward to std::memmove

template <class>
void __pstl_copy();

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
__copy(_ExecutionPolicy&& __policy,
       _ForwardIterator&& __first,
       _ForwardIterator&& __last,
       _ForwardOutIterator&& __result) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_copy, _RawPolicy),
      [&__policy](_ForwardIterator __g_first, _ForwardIterator __g_last, _ForwardOutIterator __g_result) {
        return std::__transform(__policy, __g_first, __g_last, __g_result, __identity());
      },
      std::move(__first),
      std::move(__last),
      std::move(__result));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _ForwardOutIterator
copy(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _ForwardOutIterator __result) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(
      _ForwardIterator, "copy(first, last, result) requires [first, last) to be ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(
      _ForwardOutIterator, "copy(first, last, result) requires result to be a ForwardIterator");
  _LIBCPP_REQUIRE_CPP17_OUTPUT_ITERATOR(
      _ForwardOutIterator, decltype(*__first), "copy(first, last, result) requires result to be an OutputIterator");
  auto __res = std::__copy(__policy, std::move(__first), std::move(__last), std::move(__result));
  if (!__res)
    std::__throw_bad_alloc();
  return *std::move(__res);
}

template <class>
void __pstl_copy_n();

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _Size,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator> __copy_n(
    _ExecutionPolicy&& __policy, _ForwardIterator&& __first, _Size&& __n, _ForwardOutIterator&& __result) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_copy_n, _RawPolicy),
      [&__policy](
          _ForwardIterator __g_first, _Size __g_n, _ForwardOutIterator __g_result) -> optional<_ForwardIterator> {
        if constexpr (__has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
          return std::__copy(__policy, std::move(__g_first), std::move(__g_first + __g_n), std::move(__g_result));
        } else {
          (void)__policy;
          return std::copy_n(__g_first, __g_n, __g_result);
        }
      },
      std::move(__first),
      std::move(__n),
      std::move(__result));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _Size,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _ForwardOutIterator
copy_n(_ExecutionPolicy&& __policy, _ForwardIterator __first, _Size __n, _ForwardOutIterator __result) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(
      _ForwardIterator, "copy_n(first, n, result) requires first to be a ForwardIterator");
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(
      _ForwardOutIterator, "copy_n(first, n, result) requires result to be a ForwardIterator");
  _LIBCPP_REQUIRE_CPP17_OUTPUT_ITERATOR(
      _ForwardOutIterator, decltype(*__first), "copy_n(first, n, result) requires result to be an OutputIterator");
  auto __res = std::__copy_n(__policy, std::move(__first), std::move(__n), std::move(__result));
  if (!__res)
    std::__throw_bad_alloc();
  return *std::move(__res);
}

template <class>
void __pstl_count_if(); // declaration needed for the frontend dispatch below

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Predicate,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__iter_diff_t<_ForwardIterator>> __count_if(
    _ExecutionPolicy&& __policy, _ForwardIterator&& __first, _ForwardIterator&& __last, _Predicate&& __pred) noexcept {
  using __diff_t = __iter_diff_t<_ForwardIterator>;
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_count_if, _RawPolicy),
      [&](_ForwardIterator __g_first, _ForwardIterator __g_last, _Predicate __g_pred) -> optional<__diff_t> {
        return std::__transform_reduce(
            __policy,
            std::move(__g_first),
            std::move(__g_last),
            __diff_t(),
            std::plus{},
            [&](__iter_reference<_ForwardIterator> __element) -> bool { return __g_pred(__element); });
      },
      std::move(__first),
      std::move(__last),
      std::move(__pred));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Predicate,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI __iter_diff_t<_ForwardIterator>
count_if(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(
      _ForwardIterator, "count_if(first, last, pred) requires [first, last) to be ForwardIterators");
  auto __res = std::__count_if(__policy, std::move(__first), std::move(__last), std::move(__pred));
  if (!__res)
    std::__throw_bad_alloc();
  return *std::move(__res);
}

template <class>
void __pstl_count(); // declaration needed for the frontend dispatch below

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__iter_diff_t<_ForwardIterator>> __count(
    _ExecutionPolicy&& __policy, _ForwardIterator&& __first, _ForwardIterator&& __last, const _Tp& __value) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_count, _RawPolicy),
      [&](_ForwardIterator __g_first, _ForwardIterator __g_last, const _Tp& __g_value)
          -> optional<__iter_diff_t<_ForwardIterator>> {
        return std::count_if(__policy, __g_first, __g_last, [&](__iter_reference<_ForwardIterator> __v) {
          return __v == __g_value;
        });
      },
      std::forward<_ForwardIterator>(__first),
      std::forward<_ForwardIterator>(__last),
      __value);
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI __iter_diff_t<_ForwardIterator>
count(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(
      _ForwardIterator, "count(first, last, val) requires [first, last) to be ForwardIterators");
  auto __res = std::__count(__policy, std::move(__first), std::move(__last), __value);
  if (!__res)
    std::__throw_bad_alloc();
  return *__res;
}

template <class>
void __pstl_equal();

template <class _ExecutionPolicy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          class _Pred,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<bool>
__equal(_ExecutionPolicy&& __policy,
        _ForwardIterator1&& __first1,
        _ForwardIterator1&& __last1,
        _ForwardIterator2&& __first2,
        _Pred&& __pred) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_equal, _RawPolicy),
      [&__policy](
          _ForwardIterator1 __g_first1, _ForwardIterator1 __g_last1, _ForwardIterator2 __g_first2, _Pred __g_pred) {
        return std::__transform_reduce(
            __policy,
            std::move(__g_first1),
            std::move(__g_last1),
            std::move(__g_first2),
            true,
            std::logical_and{},
            std::move(__g_pred));
      },
      std::move(__first1),
      std::move(__last1),
      std::move(__first2),
      std::move(__pred));
}

template <class _ExecutionPolicy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          class _Pred,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI bool
equal(_ExecutionPolicy&& __policy,
      _ForwardIterator1 __first1,
      _ForwardIterator1 __last1,
      _ForwardIterator2 __first2,
      _Pred __pred) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator1, "equal requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator2, "equal requires ForwardIterators");
  auto __res = std::__equal(__policy, std::move(__first1), std::move(__last1), std::move(__first2), std::move(__pred));
  if (!__res)
    std::__throw_bad_alloc();
  return *__res;
}

template <class _ExecutionPolicy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          enable_if_t<is_execution_policy_v<__remove_cvref_t<_ExecutionPolicy>>, int> = 0>
_LIBCPP_HIDE_FROM_ABI bool
equal(_ExecutionPolicy&& __policy, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator1, "equal requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator2, "equal requires ForwardIterators");
  auto __res = std::__equal(__policy, std::move(__first1), std::move(__last1), std::move(__first2), std::equal_to{});
  if (!__res)
    std::__throw_bad_alloc();
  return *__res;
}

template <class _ExecutionPolicy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          class _Pred,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<bool>
__equal(_ExecutionPolicy&& __policy,
        _ForwardIterator1&& __first1,
        _ForwardIterator1&& __last1,
        _ForwardIterator2&& __first2,
        _ForwardIterator2&& __last2,
        _Pred&& __pred) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_equal, _RawPolicy),
      [&__policy](_ForwardIterator1 __g_first1,
                  _ForwardIterator1 __g_last1,
                  _ForwardIterator2 __g_first2,
                  _ForwardIterator2 __g_last2,
                  _Pred __g_pred) -> optional<bool> {
        if constexpr (__has_random_access_iterator_category<_ForwardIterator1>::value &&
                      __has_random_access_iterator_category<_ForwardIterator2>::value) {
          if (__g_last1 - __g_first1 != __g_last2 - __g_first2)
            return false;
          return std::__equal(
              __policy, std::move(__g_first1), std::move(__g_last1), std::move(__g_first2), std::move(__g_pred));
        } else {
          (void)__policy; // Avoid unused lambda capture warning
          return std::equal(
              std::move(__g_first1),
              std::move(__g_last1),
              std::move(__g_first2),
              std::move(__g_last2),
              std::move(__g_pred));
        }
      },
      std::move(__first1),
      std::move(__last1),
      std::move(__first2),
      std::move(__last2),
      std::move(__pred));
}

template <class _ExecutionPolicy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          class _Pred,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI bool
equal(_ExecutionPolicy&& __policy,
      _ForwardIterator1 __first1,
      _ForwardIterator1 __last1,
      _ForwardIterator2 __first2,
      _ForwardIterator2 __last2,
      _Pred __pred) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator1, "equal requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator2, "equal requires ForwardIterators");
  auto __res = std::__equal(
      __policy, std::move(__first1), std::move(__last1), std::move(__first2), std::move(__last2), std::move(__pred));
  if (!__res)
    std::__throw_bad_alloc();
  return *__res;
}

template <class _ExecutionPolicy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          enable_if_t<is_execution_policy_v<__remove_cvref_t<_ExecutionPolicy>>, int> = 0>
_LIBCPP_HIDE_FROM_ABI bool
equal(_ExecutionPolicy&& __policy,
      _ForwardIterator1 __first1,
      _ForwardIterator1 __last1,
      _ForwardIterator2 __first2,
      _ForwardIterator2 __last2) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator1, "equal requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator2, "equal requires ForwardIterators");
  auto __res = std::__equal(
      __policy, std::move(__first1), std::move(__last1), std::move(__first2), std::move(__last2), std::equal_to{});
  if (!__res)
    std::__throw_bad_alloc();
  return *__res;
}

template <class>
void __pstl_fill(); // declaration needed for the frontend dispatch below

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI optional<__empty> __fill(
    _ExecutionPolicy&& __policy, _ForwardIterator&& __first, _ForwardIterator&& __last, const _Tp& __value) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_fill, _RawPolicy),
      [&](_ForwardIterator __g_first, _ForwardIterator __g_last, const _Tp& __g_value) {
        return std::__for_each(__policy, __g_first, __g_last, [&](__iter_reference<_ForwardIterator> __element) {
          __element = __g_value;
        });
      },
      std::forward<_ForwardIterator>(__first),
      std::forward<_ForwardIterator>(__last),
      __value);
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void
fill(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "fill requires ForwardIterators");
  if (!std::__fill(__policy, std::move(__first), std::move(__last), __value))
    std::__throw_bad_alloc();
}

template <class>
void __pstl_fill_n(); // declaration needed for the frontend dispatch below

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _SizeT,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
__fill_n(_ExecutionPolicy&& __policy, _ForwardIterator&& __first, _SizeT&& __n, const _Tp& __value) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_fill_n, _RawPolicy),
      [&](_ForwardIterator __g_first, _SizeT __g_n, const _Tp& __g_value) {
        if constexpr (__has_random_access_iterator_category_or_concept<_ForwardIterator>::value)
          std::fill(__policy, __g_first, __g_first + __g_n, __g_value);
        else
          std::fill_n(__g_first, __g_n, __g_value);
        return optional<__empty>{__empty{}};
      },
      std::move(__first),
      std::move(__n),
      __value);
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _SizeT,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void
fill_n(_ExecutionPolicy&& __policy, _ForwardIterator __first, _SizeT __n, const _Tp& __value) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "fill_n requires ForwardIterators");
  if (!std::__fill_n(__policy, std::move(__first), std::move(__n), __value))
    std::__throw_bad_alloc();
}

template <class>
void __pstl_find_if_not();

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Predicate,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__remove_cvref_t<_ForwardIterator>> __find_if_not(
    _ExecutionPolicy&& __policy, _ForwardIterator&& __first, _ForwardIterator&& __last, _Predicate&& __pred) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_find_if_not, _RawPolicy),
      [&](_ForwardIterator&& __g_first, _ForwardIterator&& __g_last, _Predicate&& __g_pred)
          -> optional<__remove_cvref_t<_ForwardIterator>> {
        return std::__find_if(
            __policy, __g_first, __g_last, [&](__iter_reference<__remove_cvref_t<_ForwardIterator>> __value) {
              return !__g_pred(__value);
            });
      },
      std::forward<_ForwardIterator>(__first),
      std::forward<_ForwardIterator>(__last),
      std::forward<_Predicate>(__pred));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Predicate,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _ForwardIterator
find_if_not(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "find_if_not requires ForwardIterators");
  auto __res = std::__find_if_not(__policy, std::move(__first), std::move(__last), std::move(__pred));
  if (!__res)
    std::__throw_bad_alloc();
  return *std::move(__res);
}

template <class>
void __pstl_find();

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__remove_cvref_t<_ForwardIterator>> __find(
    _ExecutionPolicy&& __policy, _ForwardIterator&& __first, _ForwardIterator&& __last, const _Tp& __value) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_find, _RawPolicy),
      [&](_ForwardIterator __g_first, _ForwardIterator __g_last, const _Tp& __g_value) -> optional<_ForwardIterator> {
        return std::find_if(
            __policy, __g_first, __g_last, [&](__iter_reference<__remove_cvref_t<_ForwardIterator>> __element) {
              return __element == __g_value;
            });
      },
      std::forward<_ForwardIterator>(__first),
      std::forward<_ForwardIterator>(__last),
      __value);
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _ForwardIterator
find(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "find requires ForwardIterators");
  auto __res = std::__find(__policy, std::move(__first), std::move(__last), __value);
  if (!__res)
    std::__throw_bad_alloc();
  return *std::move(__res);
}

template <class>
void __pstl_for_each_n(); // declaration needed for the frontend dispatch below

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Size,
          class _Function,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
__for_each_n(_ExecutionPolicy&& __policy, _ForwardIterator&& __first, _Size&& __size, _Function&& __func) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_for_each_n, _RawPolicy),
      [&](_ForwardIterator __g_first, _Size __g_size, _Function __g_func) -> optional<__empty> {
        if constexpr (__has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
          std::for_each(__policy, std::move(__g_first), __g_first + __g_size, std::move(__g_func));
          return __empty{};
        } else {
          std::for_each_n(std::move(__g_first), __g_size, std::move(__g_func));
          return __empty{};
        }
      },
      std::move(__first),
      std::move(__size),
      std::move(__func));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Size,
          class _Function,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void
for_each_n(_ExecutionPolicy&& __policy, _ForwardIterator __first, _Size __size, _Function __func) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "for_each_n requires a ForwardIterator");
  auto __res = std::__for_each_n(__policy, std::move(__first), std::move(__size), std::move(__func));
  if (!__res)
    std::__throw_bad_alloc();
}

template <class>
void __pstl_generate();

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Generator,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty> __generate(
    _ExecutionPolicy&& __policy, _ForwardIterator&& __first, _ForwardIterator&& __last, _Generator&& __gen) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_generate, _RawPolicy),
      [&__policy](_ForwardIterator __g_first, _ForwardIterator __g_last, _Generator __g_gen) {
        return std::__for_each(
            __policy, std::move(__g_first), std::move(__g_last), [&](__iter_reference<_ForwardIterator> __element) {
              __element = __g_gen();
            });
      },
      std::move(__first),
      std::move(__last),
      std::move(__gen));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Generator,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void
generate(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Generator __gen) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "generate requires ForwardIterators");
  if (!std::__generate(__policy, std::move(__first), std::move(__last), std::move(__gen)))
    std::__throw_bad_alloc();
}

template <class>
void __pstl_generate_n();

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Size,
          class _Generator,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
__generate_n(_ExecutionPolicy&& __policy, _ForwardIterator&& __first, _Size&& __n, _Generator&& __gen) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_generate_n, _RawPolicy),
      [&__policy](_ForwardIterator __g_first, _Size __g_n, _Generator __g_gen) {
        return std::__for_each_n(
            __policy, std::move(__g_first), std::move(__g_n), [&](__iter_reference<_ForwardIterator> __element) {
              __element = __g_gen();
            });
      },
      std::move(__first),
      __n,
      std::move(__gen));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Size,
          class _Generator,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void
generate_n(_ExecutionPolicy&& __policy, _ForwardIterator __first, _Size __n, _Generator __gen) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "generate_n requires a ForwardIterator");
  if (!std::__generate_n(__policy, std::move(__first), std::move(__n), std::move(__gen)))
    std::__throw_bad_alloc();
}

template <class>
void __pstl_is_partitioned();

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Predicate,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<bool> __is_partitioned(
    _ExecutionPolicy&& __policy, _ForwardIterator&& __first, _ForwardIterator&& __last, _Predicate&& __pred) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_is_partitioned, _RawPolicy),
      [&__policy](_ForwardIterator __g_first, _ForwardIterator __g_last, _Predicate __g_pred) {
        __g_first = std::find_if_not(__policy, __g_first, __g_last, __g_pred);
        if (__g_first == __g_last)
          return true;
        ++__g_first;
        return std::none_of(__policy, __g_first, __g_last, __g_pred);
      },
      std::move(__first),
      std::move(__last),
      std::move(__pred));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Predicate,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI bool
is_partitioned(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "is_partitioned requires ForwardIterators");
  auto __res = std::__is_partitioned(__policy, std::move(__first), std::move(__last), std::move(__pred));
  if (!__res)
    std::__throw_bad_alloc();
  return *std::move(__res);
}

template <class _ExecutionPolicy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          class _ForwardOutIterator,
          class _Comp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
__merge(_ExecutionPolicy&&,
        _ForwardIterator1&& __first1,
        _ForwardIterator1&& __last1,
        _ForwardIterator2&& __first2,
        _ForwardIterator2&& __last2,
        _ForwardOutIterator&& __result,
        _Comp&& __comp) noexcept {
  using _Backend = typename __select_backend<_RawPolicy>::type;
  return std::__pstl_merge<_RawPolicy>(
      _Backend{},
      std::forward<_ForwardIterator1>(__first1),
      std::forward<_ForwardIterator1>(__last1),
      std::forward<_ForwardIterator2>(__first2),
      std::forward<_ForwardIterator2>(__last2),
      std::forward<_ForwardOutIterator>(__result),
      std::forward<_Comp>(__comp));
}

template <class _ExecutionPolicy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          class _ForwardOutIterator,
          class _Comp                                         = std::less<>,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _ForwardOutIterator
merge(_ExecutionPolicy&& __policy,
      _ForwardIterator1 __first1,
      _ForwardIterator1 __last1,
      _ForwardIterator2 __first2,
      _ForwardIterator2 __last2,
      _ForwardOutIterator __result,
      _Comp __comp = {}) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator1, "merge requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator2, "merge requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_OUTPUT_ITERATOR(_ForwardOutIterator, decltype(*__first1), "merge requires an OutputIterator");
  _LIBCPP_REQUIRE_CPP17_OUTPUT_ITERATOR(_ForwardOutIterator, decltype(*__first2), "merge requires an OutputIterator");
  auto __res = std::__merge(
      __policy,
      std::move(__first1),
      std::move(__last1),
      std::move(__first2),
      std::move(__last2),
      std::move(__result),
      std::move(__comp));
  if (!__res)
    std::__throw_bad_alloc();
  return *std::move(__res);
}

// TODO: Use the std::copy/move shenanigans to forward to std::memmove
//       Investigate whether we want to still forward to std::transform(policy)
//       in that case for the execution::par part, or whether we actually want
//       to run everything serially in that case.

template <class>
void __pstl_move();

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
__move(_ExecutionPolicy&& __policy,
       _ForwardIterator&& __first,
       _ForwardIterator&& __last,
       _ForwardOutIterator&& __result) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_move, _RawPolicy),
      [&__policy](_ForwardIterator __g_first, _ForwardIterator __g_last, _ForwardOutIterator __g_result) {
        return std::__transform(__policy, __g_first, __g_last, __g_result, [](auto&& __v) { return std::move(__v); });
      },
      std::move(__first),
      std::move(__last),
      std::move(__result));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _ForwardOutIterator
move(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _ForwardOutIterator __result) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "move requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardOutIterator, "move requires an OutputIterator");
  _LIBCPP_REQUIRE_CPP17_OUTPUT_ITERATOR(
      _ForwardOutIterator, decltype(std::move(*__first)), "move requires an OutputIterator");
  auto __res = std::__move(__policy, std::move(__first), std::move(__last), std::move(__result));
  if (!__res)
    std::__throw_bad_alloc();
  return *__res;
}

template <class>
void __pstl_replace_if();

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Pred,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
__replace_if(_ExecutionPolicy&& __policy,
             _ForwardIterator&& __first,
             _ForwardIterator&& __last,
             _Pred&& __pred,
             const _Tp& __new_value) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_replace_if, _RawPolicy),
      [&__policy](
          _ForwardIterator&& __g_first, _ForwardIterator&& __g_last, _Pred&& __g_pred, const _Tp& __g_new_value) {
        std::for_each(__policy, __g_first, __g_last, [&](__iter_reference<_ForwardIterator> __element) {
          if (__g_pred(__element))
            __element = __g_new_value;
        });
        return optional<__empty>{__empty{}};
      },
      std::move(__first),
      std::move(__last),
      std::move(__pred),
      __new_value);
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Pred,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void
replace_if(_ExecutionPolicy&& __policy,
           _ForwardIterator __first,
           _ForwardIterator __last,
           _Pred __pred,
           const _Tp& __new_value) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "replace_if requires ForwardIterators");
  auto __res = std::__replace_if(__policy, std::move(__first), std::move(__last), std::move(__pred), __new_value);
  if (!__res)
    std::__throw_bad_alloc();
}

template <class>
void __pstl_replace();

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
__replace(_ExecutionPolicy&& __policy,
          _ForwardIterator&& __first,
          _ForwardIterator&& __last,
          const _Tp& __old_value,
          const _Tp& __new_value) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_replace, _RawPolicy),
      [&__policy](
          _ForwardIterator __g_first, _ForwardIterator __g_last, const _Tp& __g_old_value, const _Tp& __g_new_value) {
        return std::__replace_if(
            __policy,
            std::move(__g_first),
            std::move(__g_last),
            [&](__iter_reference<_ForwardIterator> __element) { return __element == __g_old_value; },
            __g_new_value);
      },
      std::forward<_ForwardIterator>(__first),
      std::forward<_ForwardIterator>(__last),
      __old_value,
      __new_value);
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void
replace(_ExecutionPolicy&& __policy,
        _ForwardIterator __first,
        _ForwardIterator __last,
        const _Tp& __old_value,
        const _Tp& __new_value) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "replace requires ForwardIterators");
  if (!std::__replace(__policy, std::move(__first), std::move(__last), __old_value, __new_value))
    std::__throw_bad_alloc();
}

template <class>
void __pstl_replace_copy_if();

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _Pred,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty> __replace_copy_if(
    _ExecutionPolicy&& __policy,
    _ForwardIterator&& __first,
    _ForwardIterator&& __last,
    _ForwardOutIterator&& __result,
    _Pred&& __pred,
    const _Tp& __new_value) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_replace_copy_if, _RawPolicy),
      [&__policy](_ForwardIterator __g_first,
                  _ForwardIterator __g_last,
                  _ForwardOutIterator __g_result,
                  _Pred __g_pred,
                  const _Tp& __g_new_value) -> optional<__empty> {
        if (!std::__transform(
                __policy, __g_first, __g_last, __g_result, [&](__iter_reference<_ForwardIterator> __element) {
                  return __g_pred(__element) ? __g_new_value : __element;
                }))
          return nullopt;
        return __empty{};
      },
      std::move(__first),
      std::move(__last),
      std::move(__result),
      std::move(__pred),
      __new_value);
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _Pred,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void replace_copy_if(
    _ExecutionPolicy&& __policy,
    _ForwardIterator __first,
    _ForwardIterator __last,
    _ForwardOutIterator __result,
    _Pred __pred,
    const _Tp& __new_value) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "replace_copy_if requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardOutIterator, "replace_copy_if requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_OUTPUT_ITERATOR(
      _ForwardOutIterator, decltype(*__first), "replace_copy_if requires an OutputIterator");
  _LIBCPP_REQUIRE_CPP17_OUTPUT_ITERATOR(_ForwardOutIterator, const _Tp&, "replace_copy requires an OutputIterator");
  if (!std::__replace_copy_if(
          __policy, std::move(__first), std::move(__last), std::move(__result), std::move(__pred), __new_value))
    std::__throw_bad_alloc();
}

template <class>
void __pstl_replace_copy();

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty> __replace_copy(
    _ExecutionPolicy&& __policy,
    _ForwardIterator&& __first,
    _ForwardIterator&& __last,
    _ForwardOutIterator&& __result,
    const _Tp& __old_value,
    const _Tp& __new_value) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_replace_copy, _RawPolicy),
      [&__policy](_ForwardIterator __g_first,
                  _ForwardIterator __g_last,
                  _ForwardOutIterator __g_result,
                  const _Tp& __g_old_value,
                  const _Tp& __g_new_value) {
        return std::__replace_copy_if(
            __policy,
            std::move(__g_first),
            std::move(__g_last),
            std::move(__g_result),
            [&](__iter_reference<_ForwardIterator> __element) { return __element == __g_old_value; },
            __g_new_value);
      },
      std::move(__first),
      std::move(__last),
      std::move(__result),
      __old_value,
      __new_value);
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _Tp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void replace_copy(
    _ExecutionPolicy&& __policy,
    _ForwardIterator __first,
    _ForwardIterator __last,
    _ForwardOutIterator __result,
    const _Tp& __old_value,
    const _Tp& __new_value) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "replace_copy requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardOutIterator, "replace_copy requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_OUTPUT_ITERATOR(
      _ForwardOutIterator, decltype(*__first), "replace_copy requires an OutputIterator");
  _LIBCPP_REQUIRE_CPP17_OUTPUT_ITERATOR(_ForwardOutIterator, const _Tp&, "replace_copy requires an OutputIterator");
  if (!std::__replace_copy(
          __policy, std::move(__first), std::move(__last), std::move(__result), __old_value, __new_value))
    std::__throw_bad_alloc();
}

template <class>
void __pstl_rotate_copy();

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
__rotate_copy(_ExecutionPolicy&& __policy,
              _ForwardIterator&& __first,
              _ForwardIterator&& __middle,
              _ForwardIterator&& __last,
              _ForwardOutIterator&& __result) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_rotate_copy, _RawPolicy),
      [&__policy](_ForwardIterator __g_first,
                  _ForwardIterator __g_middle,
                  _ForwardIterator __g_last,
                  _ForwardOutIterator __g_result) -> optional<_ForwardOutIterator> {
        auto __result_mid =
            std::__copy(__policy, _ForwardIterator(__g_middle), std::move(__g_last), std::move(__g_result));
        if (!__result_mid)
          return nullopt;
        return std::__copy(__policy, std::move(__g_first), std::move(__g_middle), *std::move(__result_mid));
      },
      std::move(__first),
      std::move(__middle),
      std::move(__last),
      std::move(__result));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _ForwardOutIterator rotate_copy(
    _ExecutionPolicy&& __policy,
    _ForwardIterator __first,
    _ForwardIterator __middle,
    _ForwardIterator __last,
    _ForwardOutIterator __result) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "rotate_copy requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardOutIterator, "rotate_copy requires ForwardIterators");
  _LIBCPP_REQUIRE_CPP17_OUTPUT_ITERATOR(
      _ForwardOutIterator, decltype(*__first), "rotate_copy requires an OutputIterator");
  auto __res =
      std::__rotate_copy(__policy, std::move(__first), std::move(__middle), std::move(__last), std::move(__result));
  if (!__res)
    std::__throw_bad_alloc();
  return *__res;
}

template <class _ExecutionPolicy,
          class _RandomAccessIterator,
          class _Comp                                         = less<>,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty> __stable_sort(
    _ExecutionPolicy&&, _RandomAccessIterator&& __first, _RandomAccessIterator&& __last, _Comp&& __comp = {}) noexcept {
  using _Backend = typename __select_backend<_RawPolicy>::type;
  return std::__pstl_stable_sort<_RawPolicy>(_Backend{}, std::move(__first), std::move(__last), std::move(__comp));
}

template <class _ExecutionPolicy,
          class _RandomAccessIterator,
          class _Comp                                         = less<>,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void stable_sort(
    _ExecutionPolicy&& __policy, _RandomAccessIterator __first, _RandomAccessIterator __last, _Comp __comp = {}) {
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(_RandomAccessIterator, "stable_sort requires RandomAccessIterators");
  if (!std::__stable_sort(__policy, std::move(__first), std::move(__last), std::move(__comp)))
    std::__throw_bad_alloc();
}

template <class>
void __pstl_sort();

template <class _ExecutionPolicy,
          class _RandomAccessIterator,
          class _Comp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
__sort(_ExecutionPolicy&& __policy,
       _RandomAccessIterator&& __first,
       _RandomAccessIterator&& __last,
       _Comp&& __comp) noexcept {
  return std::__pstl_frontend_dispatch(
      _LIBCPP_PSTL_CUSTOMIZATION_POINT(__pstl_sort, _RawPolicy),
      [&__policy](_RandomAccessIterator __g_first, _RandomAccessIterator __g_last, _Comp __g_comp) {
        std::stable_sort(__policy, std::move(__g_first), std::move(__g_last), std::move(__g_comp));
        return optional<__empty>{__empty{}};
      },
      std::forward<_RandomAccessIterator>(__first),
      std::forward<_RandomAccessIterator>(__last),
      std::forward<_Comp>(__comp));
}

template <class _ExecutionPolicy,
          class _RandomAccessIterator,
          class _Comp,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void
sort(_ExecutionPolicy&& __policy, _RandomAccessIterator __first, _RandomAccessIterator __last, _Comp __comp) {
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(_RandomAccessIterator, "sort requires RandomAccessIterators");
  if (!std::__sort(__policy, std::move(__first), std::move(__last), std::move(__comp)))
    std::__throw_bad_alloc();
}

template <class _ExecutionPolicy,
          class _RandomAccessIterator,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void
sort(_ExecutionPolicy&& __policy, _RandomAccessIterator __first, _RandomAccessIterator __last) {
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(_RandomAccessIterator, "sort requires RandomAccessIterators");
  if (!std::__sort(__policy, std::move(__first), std::move(__last), less{}))
    std::__throw_bad_alloc();
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_PSTL_H
