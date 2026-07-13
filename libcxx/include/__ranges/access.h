// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_ACCESS_H
#define _LIBCPP___RANGES_ACCESS_H

#include <__concepts/class_or_enum.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__ranges/enable_borrowed_range.h>
#include <__type_traits/extent.h>
#include <__type_traits/is_array.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/remove_all_extents.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_cvref.h>
#include <__type_traits/remove_reference.h>
#include <__utility/auto_cast.h>
#include <__utility/cpo.h>
#include <__utility/declval.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {
template <class _Tp>
concept __can_borrow = is_lvalue_reference_v<_Tp> || enable_borrowed_range<remove_cvref_t<_Tp>>;
} // namespace ranges

// [range.access.begin]

namespace ranges {
namespace __begin {
void begin() = delete;

struct __fn : _CPO<[]<class _Ep> consteval noexcept {
  // [range.access.begin]

  // Given a subexpression E with type T, let t be an lvalue that denotes the reified object for E. Then:
  using _Tp = remove_reference_t<_Ep>;

  // If E is an rvalue and enable_borrowed_range<remove_cv_t<T>> is false, ranges​::​begin(E) is ill-formed.
  if constexpr (is_rvalue_reference_v<_Ep> && !enable_borrowed_range<remove_cv_t<_Tp>>) {
    return __diagnostic<"calling ranges::begin on an rvalue of a non-borrowed range">{};

    // Otherwise, if T is an array type ([dcl.array]) and remove_all_extents_t<T> is an incomplete type,
    // ranges​::​begin(E) is ill-formed with no diagnostic required.
  } else if constexpr (is_array_v<_Tp>) {
    if constexpr (!requires { sizeof(remove_all_extents_t<_Tp>); })
      return __diagnostic<"calling ranges::begin on an array of incomplete type">{};
    else
      // Otherwise, if T is an array type, ranges​::​begin(E) is expression-equivalent to t + 0.
      return [](_Ep __v) noexcept { return __v + 0; };

    // Otherwise, if auto(t.begin()) is a valid expression whose type models input_or_output_iterator,
    // ranges​::​begin(E) is expression-equivalent to auto(t.begin()).
  } else if constexpr (requires(_Tp& __t) {
                         { _LIBCPP_AUTO_CAST(__t.begin()) } -> input_or_output_iterator;
                       }) {
    return [](_Ep __v) noexcept(noexcept(_LIBCPP_AUTO_CAST(__v.begin()))) { return _LIBCPP_AUTO_CAST(__v.begin()); };

    // Otherwise, if T is a class or enumeration type and auto(begin(t)) is a valid expression whose type models
    // input_or_output_iterator where the meaning of begin is established as-if by performing argument-dependent lookup
    // only ([basic.lookup.argdep]), then ranges​::​begin(E) is expression-equivalent to that expression.
  } else if constexpr (__class_or_enum<_Tp>) {
    if constexpr (requires(_Tp& __t) {
                    { _LIBCPP_AUTO_CAST(begin(__t)) } -> input_or_output_iterator;
                  })
      return [](_Ep __v) noexcept(noexcept(_LIBCPP_AUTO_CAST(begin(__v)))) { return _LIBCPP_AUTO_CAST(begin(__v)); };
    else
      return __diagnostic<"neither v.begin() nor begin(v) are valid calls">{};

    // Otherwise, ranges​::​begin(E) is ill-formed.
  } else {
    return __diagnostic<"neither v.begin() nor begin(v) are valid calls">{};
  }
}> {};
} // namespace __begin

inline namespace __cpo {
inline constexpr auto begin = __begin::__fn{};
} // namespace __cpo
} // namespace ranges

// [range.range]

namespace ranges {
template <class _Tp>
using iterator_t = decltype(ranges::begin(std::declval<_Tp&>()));
} // namespace ranges

// [range.access.end]

namespace ranges {
namespace __end {
struct __fn : _CPO<[]<class _Ep> consteval noexcept {
  // [range.access.end]

  // Given a subexpression E with type T, let t be an lvalue that denotes the reified object for E. Then:
  using _Tp = remove_reference_t<_Ep>;

  // If E is an rvalue and enable_borrowed_range<remove_cv_t<T>> is false, ranges​::​end(E) is ill-formed.
  if constexpr (is_rvalue_reference_v<_Ep> && !enable_borrowed_range<remove_cv_t<_Tp>>) {
    return;

    // Otherwise, if T is an array type ([dcl.array]) and remove_all_extents_t<T> is an incomplete type,
    // ranges​::​end(E) is ill-formed with no diagnostic required.
  } else if constexpr (is_array_v<_Tp>) {
    if constexpr (!requires { sizeof(remove_all_extents_t<_Tp>); }) {
      return;

      // Otherwise, if T is an array of unknown bound, ranges​::​end(E) is ill-formed.
    } else if constexpr (is_unbounded_array_v<_Tp>) {
      return;

      // Otherwise, if T is an array, ranges​::​end(E) is expression-equivalent to t + extent_v<T>.
    } else {
      return [](_Ep __v) noexcept { return __v + extent_v<_Tp>; };
    }

    // Otherwise, if auto(t.end()) is a valid expression whose type models sentinel_for<iterator_t<T>> then
    // ranges​::​end(E) is expression-equivalent to auto(t.end()).
  } else if constexpr (requires(_Tp& __t) {
                         { _LIBCPP_AUTO_CAST(__t.end()) } -> sentinel_for<iterator_t<_Tp>>;
                       }) {
    return [](_Ep __v) noexcept(noexcept(_LIBCPP_AUTO_CAST(__v.end()))) { return _LIBCPP_AUTO_CAST(__v.end()); };

    // Otherwise, if T is a class or enumeration type and auto(end(t)) is a valid expression whose type models
    // sentinel_for<iterator_t<T>> where the meaning of end is established as-if by performing argument-dependent lookup
    // only ([basic.lookup.argdep]), then ranges​::​end(E) is expression-equivalent to that expression.
  } else if constexpr (__class_or_enum<_Tp>) {
    if constexpr (requires(_Tp& __t) {
                    { _LIBCPP_AUTO_CAST(end(__t)) } -> sentinel_for<iterator_t<_Tp>>;
                  }) {
      return [](_Ep __v) noexcept(noexcept(_LIBCPP_AUTO_CAST(end(__v)))) { return _LIBCPP_AUTO_CAST(end(__v)); };
    } else {
      return;
    }
  } else {
    return;
  }
}> {};
} // namespace __end

inline namespace __cpo {
inline constexpr auto end = __end::__fn{};
} // namespace __cpo
} // namespace ranges

// [range.access.cbegin]

namespace ranges {
namespace __cbegin {
struct __fn {
  template <class _Tp>
    requires is_lvalue_reference_v<_Tp&&>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(ranges::begin(static_cast<const remove_reference_t<_Tp>&>(__t))))
          -> decltype(ranges::begin(static_cast<const remove_reference_t<_Tp>&>(__t))) {
    return ranges::begin(static_cast<const remove_reference_t<_Tp>&>(__t));
  }

  template <class _Tp>
    requires is_rvalue_reference_v<_Tp&&>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(ranges::begin(static_cast<const _Tp&&>(__t))))
          -> decltype(ranges::begin(static_cast<const _Tp&&>(__t))) {
    return ranges::begin(static_cast<const _Tp&&>(__t));
  }
};
} // namespace __cbegin

inline namespace __cpo {
inline constexpr auto cbegin = __cbegin::__fn{};
} // namespace __cpo
} // namespace ranges

// [range.access.cend]

namespace ranges {
namespace __cend {
struct __fn {
  template <class _Tp>
    requires is_lvalue_reference_v<_Tp&&>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(ranges::end(static_cast<const remove_reference_t<_Tp>&>(__t))))
          -> decltype(ranges::end(static_cast<const remove_reference_t<_Tp>&>(__t))) {
    return ranges::end(static_cast<const remove_reference_t<_Tp>&>(__t));
  }

  template <class _Tp>
    requires is_rvalue_reference_v<_Tp&&>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(ranges::end(static_cast<const _Tp&&>(__t))))
          -> decltype(ranges::end(static_cast<const _Tp&&>(__t))) {
    return ranges::end(static_cast<const _Tp&&>(__t));
  }
};
} // namespace __cend

inline namespace __cpo {
inline constexpr auto cend = __cend::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

#endif // _LIBCPP___RANGES_ACCESS_H
