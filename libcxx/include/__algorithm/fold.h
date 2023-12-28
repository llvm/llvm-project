// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_FOLD_H
#define _LIBCPP___ALGORITHM_FOLD_H

#include <__concepts/assignable.h>
#include <__concepts/convertible_to.h>
#include <__concepts/invocable.h>
#include <__concepts/movable.h>
#include <__config>
#include <__functional/invoke.h>
#include <__functional/reference_wrapper.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/next.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/dangling.h>
#include <__type_traits/decay.h>
#include <__type_traits/invoke.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace ranges {
template <class _Iter, class _Tp>
struct in_value_result {
  _LIBCPP_NO_UNIQUE_ADDRESS _Iter in;
  _LIBCPP_NO_UNIQUE_ADDRESS _Tp value;

  template <class _I2, class _T2>
    requires convertible_to<const _Iter&, _I2> && convertible_to<const _Tp&, _T2>
  _LIBCPP_HIDE_FROM_ABI constexpr operator in_value_result<_I2, _T2>() const& {
    return {in, value};
  }

  template <class _I2, class _T2>
    requires convertible_to<_Iter, _I2> && convertible_to<_Tp, _T2>
  _LIBCPP_HIDE_FROM_ABI constexpr operator in_value_result<_I2, _T2>() && {
    return {std::move(in), std::move(value)};
  }
};

template <class _Iter, class _Tp>
using fold_left_with_iter_result = in_value_result<_Iter, _Tp>;

template <class _Func, class _Tp, class _Iter, class _Rp, class _Up = decay_t<_Rp>>
concept __indirectly_binary_left_foldable_impl =
    convertible_to<_Rp, _Up> &&                        //
    movable<_Tp> &&                                    //
    movable<_Up> &&                                    //
    convertible_to<_Tp, _Up> &&                        //
    invocable<_Func&, _Up, iter_reference_t<_Iter>> && //
    assignable_from<_Up&, invoke_result_t<_Func&, _Up, iter_reference_t<_Iter>>>;

template <class _Func, class _Tp, class _Iter>
concept __indirectly_binary_left_foldable =
    copy_constructible<_Func> &&                       //
    invocable<_Func&, _Tp, iter_reference_t<_Iter>> && //
    __indirectly_binary_left_foldable_impl<_Func, _Tp, _Iter, invoke_result_t<_Func&, _Tp, iter_reference_t<_Iter>>>;

struct __fold_left_with_iter {
  template <input_iterator _Iter,
            sentinel_for<_Iter> _Sent,
            class _Tp,
            __indirectly_binary_left_foldable<_Tp, _Iter> _Func>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Iter __first, _Sent __last, _Tp __init, _Func __f) {
    using _Up = decay_t<invoke_result_t<_Func&, _Tp, iter_reference_t<_Iter>>>;

    if (__first == __last) {
      return fold_left_with_iter_result<_Iter, _Up>{std::move(__first), _Up(std::move(__init))};
    }

    _Up __result = std::invoke(__f, std::move(__init), *__first);
    for (++__first; __first != __last; ++__first) {
      __result = std::invoke(__f, std::move(__result), *__first);
    }

    return fold_left_with_iter_result<_Iter, _Up>{std::move(__first), std::move(__result)};
  }

  template <input_range _Rp, class _Tp, __indirectly_binary_left_foldable<_Tp, iterator_t<_Rp>> _Func>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Rp&& __r, _Tp __init, _Func __f) {
    auto __result = operator()(ranges::begin(__r), ranges::end(__r), std::move(__init), std::ref(__f));

    using _Up = decay_t<invoke_result_t<_Func&, _Tp, range_reference_t<_Rp>>>;
    return fold_left_with_iter_result<borrowed_iterator_t<_Rp>, _Up>{std::move(__result.in), std::move(__result.value)};
  }
};

inline constexpr auto fold_left_with_iter = __fold_left_with_iter();

struct __fold_left {
  template <input_iterator _Iter,
            sentinel_for<_Iter> _Sent,
            class _Tp,
            __indirectly_binary_left_foldable<_Tp, _Iter> _Func>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Iter __first, _Sent __last, _Tp __init, _Func __f) {
    return fold_left_with_iter(std::move(__first), std::move(__last), std::move(__init), std::ref(__f)).value;
  }

  template <input_range _Rp, class _Tp, __indirectly_binary_left_foldable<_Tp, iterator_t<_Rp>> _Func>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Rp&& __r, _Tp __init, _Func __f) {
    return fold_left_with_iter(ranges::begin(__r), ranges::end(__r), std::move(__init), std::ref(__f)).value;
  }
};

inline constexpr auto fold_left = __fold_left();
} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_FOLD_H
