// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_FOLD_H
#define _LIBCPP___ALGORITHM_RANGES_FOLD_H

#include <__concepts/assignable.h>
#include <__concepts/constructible.h>
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
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace ranges {
template <class _Ip, class _Tp>
struct in_value_result {
  _LIBCPP_NO_UNIQUE_ADDRESS _Ip in;
  _LIBCPP_NO_UNIQUE_ADDRESS _Tp value;

  template <class _I2, class _T2>
    requires convertible_to<const _Ip&, _I2> && convertible_to<const _Tp&, _T2>
  _LIBCPP_HIDE_FROM_ABI constexpr operator in_value_result<_I2, _T2>() const& {
    return {in, value};
  }

  template <class _I2, class _T2>
    requires convertible_to<_Ip, _I2> && convertible_to<_Tp, _T2>
  _LIBCPP_HIDE_FROM_ABI constexpr operator in_value_result<_I2, _T2>() && {
    return {std::move(in), std::move(value)};
  }
};

template <class _Ip, class _Tp>
using fold_left_with_iter_result = in_value_result<_Ip, _Tp>;

template <class _Ip, class _Tp>
using fold_left_first_with_iter_result = in_value_result<_Ip, _Tp>;

template <class _Fp, class _Tp, class _Ip, class _Rp, class _Up = decay_t<_Rp>>
concept __indirectly_binary_left_foldable_impl =
    convertible_to<_Rp, _Up> &&                    //
    movable<_Tp> &&                                //
    movable<_Up> &&                                //
    convertible_to<_Tp, _Up> &&                    //
    invocable<_Fp&, _Up, iter_reference_t<_Ip>> && //
    assignable_from<_Up&, invoke_result_t<_Fp&, _Up, iter_reference_t<_Ip>>>;

template <class _Fp, class _Tp, class _Ip>
concept __indirectly_binary_left_foldable =
    copy_constructible<_Fp> &&                     //
    invocable<_Fp&, _Tp, iter_reference_t<_Ip>> && //
    __indirectly_binary_left_foldable_impl<_Fp, _Tp, _Ip, invoke_result_t<_Fp&, _Tp, iter_reference_t<_Ip>>>;

struct __fold_left_with_iter {
  template <input_iterator _Ip, sentinel_for<_Ip> _Sp, class _Tp, __indirectly_binary_left_foldable<_Tp, _Ip> _Fp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Ip __first, _Sp __last, _Tp __init, _Fp __f) {
    using _Up = decay_t<invoke_result_t<_Fp&, _Tp, iter_reference_t<_Ip>>>;

    if (__first == __last) {
      return fold_left_with_iter_result<_Ip, _Up>{std::move(__first), _Up(std::move(__init))};
    }

    _Up __result = std::invoke(__f, std::move(__init), *__first);
    for (++__first; __first != __last; ++__first) {
      __result = std::invoke(__f, std::move(__result), *__first);
    }

    return fold_left_with_iter_result<_Ip, _Up>{std::move(__first), std::move(__result)};
  }

  template <input_range _Rp, class _Tp, __indirectly_binary_left_foldable<_Tp, iterator_t<_Rp>> _Fp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Rp&& __r, _Tp __init, _Fp __f) {
    auto __result = operator()(ranges::begin(__r), ranges::end(__r), std::move(__init), std::ref(__f));

    using _Up = decay_t<invoke_result_t<_Fp&, _Tp, range_reference_t<_Rp>>>;
    return fold_left_with_iter_result<borrowed_iterator_t<_Rp>, _Up>{std::move(__result.in), std::move(__result.value)};
  }
};

inline constexpr auto fold_left_with_iter = __fold_left_with_iter();

struct __fold_left {
  template <input_iterator _Ip, sentinel_for<_Ip> _Sp, class _Tp, __indirectly_binary_left_foldable<_Tp, _Ip> _Fp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Ip __first, _Sp __last, _Tp __init, _Fp __f) {
    return fold_left_with_iter(std::move(__first), std::move(__last), std::move(__init), std::ref(__f)).value;
  }

  template <input_range _Rp, class _Tp, __indirectly_binary_left_foldable<_Tp, iterator_t<_Rp>> _Fp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Rp&& __r, _Tp __init, _Fp __f) {
    return fold_left_with_iter(ranges::begin(__r), ranges::end(__r), std::move(__init), std::ref(__f)).value;
  }
};

inline constexpr auto fold_left = __fold_left();

struct __fold_left_first_with_iter {
  template <input_iterator _Ip, sentinel_for<_Ip> _Sp, __indirectly_binary_left_foldable<iter_value_t<_Ip>, _Ip> _Fp>
    requires constructible_from<iter_value_t<_Ip>, iter_reference_t<_Ip>>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Ip __first, _Sp __last, _Fp __f) {
    using _Up = decltype(fold_left(std::move(__first), __last, iter_value_t<_Ip>(*__first), __f));

    // case of empty range
    if (__first == __last)
      return fold_left_first_with_iter_result<_Ip, optional<_Up>>{std::move(__first), optional<_Up>()};

    _Up __result(*__first);
    for (++__first; __first != __last; ++__first)
      __result = std::invoke(__f, std::move(__result), *__first);

    return fold_left_first_with_iter_result<_Ip, optional<_Up>>{std::move(__first), optional<_Up>(std::move(__result))};
  }

  template <input_range _Rp, __indirectly_binary_left_foldable<range_value_t<_Rp>, iterator_t<_Rp>> _Fp>
    requires constructible_from<range_value_t<_Rp>, range_reference_t<_Rp>>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Rp&& __r, _Fp __f) {
    auto __result = operator()(ranges::begin(__r), ranges::end(__r), std::ref(__f));

    using _Up = decltype(fold_left(ranges::begin(__r), ranges::end(__r), range_value_t<_Rp>(*ranges::begin(__r)), __f));
    return fold_left_first_with_iter_result<borrowed_iterator_t<_Rp>, optional<_Up>>{
        std::move(__result.in), std::move(__result.value)};
  }
};

inline constexpr auto fold_left_first_with_iter = __fold_left_first_with_iter();

struct __fold_left_first {
  template <input_iterator _Ip, sentinel_for<_Ip> _Sp, __indirectly_binary_left_foldable<iter_value_t<_Ip>, _Ip> _Fp>
    requires constructible_from<iter_value_t<_Ip>, iter_reference_t<_Ip>>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Ip __first, _Sp __last, _Fp __f) {
    return fold_left_first_with_iter(std::move(__first), std::move(__last), std::ref(__f)).value;
  }

  template <input_range _Rp, __indirectly_binary_left_foldable<range_value_t<_Rp>, iterator_t<_Rp>> _Fp>
    requires constructible_from<range_value_t<_Rp>, range_reference_t<_Rp>>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Rp&& __r, _Fp __f) {
    return fold_left_first_with_iter(ranges::begin(__r), ranges::end(__r), std::ref(__f)).value;
  }
};

inline constexpr auto fold_left_first = __fold_left_first();

} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_RANGES_FOLD_H
