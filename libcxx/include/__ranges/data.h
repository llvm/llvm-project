// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_DATA_H
#define _LIBCPP___RANGES_DATA_H

#include <__config>
#include <__iterator/concepts.h>
#include <__memory/pointer_traits.h>
#include <__ranges/access.h>
#include <__ranges/enable_borrowed_range.h>
#include <__type_traits/is_array.h>
#include <__type_traits/is_object.h>
#include <__type_traits/is_pointer.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/remove_all_extents.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_pointer.h>
#include <__type_traits/remove_reference.h>
#include <__utility/auto_cast.h>
#include <__utility/cpo.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

// [range.prim.data]

namespace ranges {
namespace __data {
template <class _Tp>
concept __ptr_to_object = is_pointer_v<_Tp> && is_object_v<remove_pointer_t<_Tp>>;

template <class _Tp>
concept __member_data = __can_borrow<_Tp> && requires(_Tp&& __t) {
  { _LIBCPP_AUTO_CAST(__t.data()) } -> __ptr_to_object;
};

template <class _Tp>
concept __ranges_begin_invocable = !__member_data<_Tp> && __can_borrow<_Tp> && requires(_Tp&& __t) {
  { ranges::begin(__t) } -> contiguous_iterator;
};

struct __fn : _CPO<[]<class _Ep> consteval noexcept {
  // [range.prim.data]

  // Given a subexpression E with type T, let t be an lvalue that denotes the reified object for E. Then:
  using _Tp = remove_reference_t<_Ep>;

  // If E is an rvalue and enable_borrowed_range<remove_cv_t<T>> is false, ranges​::​data(E) is ill-formed.
  if constexpr (is_rvalue_reference_v<_Ep> && !enable_borrowed_range<remove_cv_t<_Tp>>) {
    return;

    // Otherwise, if T is an array type ([dcl.array]) and remove_all_extents_t<T> is an incomplete type,
    // ranges​::​data(E) is ill-formed with no diagnostic required.
  } else if constexpr (is_array_v<_Tp>) {
    if constexpr (!requires { sizeof(remove_all_extents_t<_Tp>); }) {
      return;
    } else {
      // This is inlined from ranges::begin
      return [](_Ep __v) noexcept { return __v + 0; };
    }

    // Otherwise, if auto(t.data()) is a valid expression of pointer to object type,
    // ranges​::​data(E) is expression-equivalent to auto(t.data()).
  } else if constexpr (requires(_Tp& __t) {
                         { _LIBCPP_AUTO_CAST(__t.data()) } -> __ptr_to_object;
                       }) {
    return [](_Ep __v) noexcept(noexcept(_LIBCPP_AUTO_CAST(__v.data()))) { return _LIBCPP_AUTO_CAST(__v.data()); };

    // Otherwise, if ranges​::​begin(t) is a valid expression whose type models contiguous_iterator,
    // ranges​::​data(E) is expression-equivalent to to_address(ranges​::​begin(t)).
  } else if constexpr (requires(_Tp& __t) {
                         { ranges::begin(__t) } -> contiguous_iterator;
                       }) {
    return [](_Ep __v) noexcept(noexcept(std::to_address(ranges::begin(__v)))) {
      return std::to_address(ranges::begin(__v));
    };

    // Otherwise, ranges​::​data(E) is ill-formed.
  } else {
    return;
  }
}> {};
} // namespace __data

inline namespace __cpo {
inline constexpr auto data = __data::__fn{};
} // namespace __cpo
} // namespace ranges

// [range.prim.cdata]

namespace ranges {
namespace __cdata {
struct __fn {
  template <class _Tp>
    requires is_lvalue_reference_v<_Tp&&>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(ranges::data(static_cast<const remove_reference_t<_Tp>&>(__t))))
          -> decltype(ranges::data(static_cast<const remove_reference_t<_Tp>&>(__t))) {
    return ranges::data(static_cast<const remove_reference_t<_Tp>&>(__t));
  }

  template <class _Tp>
    requires is_rvalue_reference_v<_Tp&&>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(ranges::data(static_cast<const _Tp&&>(__t))))
          -> decltype(ranges::data(static_cast<const _Tp&&>(__t))) {
    return ranges::data(static_cast<const _Tp&&>(__t));
  }
};
} // namespace __cdata

inline namespace __cpo {
inline constexpr auto cdata = __cdata::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

#endif // _LIBCPP___RANGES_DATA_H
