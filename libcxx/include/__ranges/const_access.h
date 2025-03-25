// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_CONST_ACCESS_H
#define _LIBCPP___RANGES_CONST_ACCESS_H

#include <__iterator/const_iterator.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/enable_borrowed_range.h>
#include <__ranges/rbegin.h>
#include <__ranges/rend.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_reference.h>
#include <__utility/declval.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

namespace ranges {
// [range.const]
#  if _LIBCPP_STD_VER >= 23
template <input_range _Rp>
_LIBCPP_HIDE_FROM_ABI constexpr auto& __possibly_const_range(_Rp& __rng) noexcept {
  if constexpr (input_range<const _Rp>) {
    return const_cast<const _Rp&>(__rng);
  } else {
    return __rng;
  }
}
#  endif // _LIBCPP_STD_VER >= 23

// [range.access.cbegin]

namespace __cbegin {
struct __fn {
#  if _LIBCPP_STD_VER >= 23
  template <class _Rng>
  using _UType _LIBCPP_NODEBUG = decltype(ranges::begin(ranges::__possibly_const_range(std::declval<_Rng&>())));

  template <__can_borrow _Rng>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr static auto operator()(_Rng&& __rng) noexcept(
      noexcept(const_iterator<_UType<_Rng>>(ranges::begin(ranges::__possibly_const_range(__rng)))))
      -> const_iterator<_UType<_Rng>> {
    return const_iterator<_UType<_Rng>>(ranges::begin(ranges::__possibly_const_range(__rng)));
  }
#  else  // ^^^ _LIBCPP_STD_VER >= 23 / _LIBCPP_STD_VER < 23 vvv
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
#  endif // ^^^ _LIBCPP_STD_VER < 23
};
} // namespace __cbegin

inline namespace __cpo {
inline constexpr auto cbegin = __cbegin::__fn{};
} // namespace __cpo

#  if _LIBCPP_STD_VER >= 23
// [range.range]
template <class _Rp>
using const_iterator_t = decltype(ranges::cbegin(std::declval<_Rp&>()));
#  endif // _LIBCPP_STD_VER >= 23

// [range.access.cend]

namespace __cend {
struct __fn {
#  if _LIBCPP_STD_VER >= 23
  template <class _Rng>
  using _UType _LIBCPP_NODEBUG = decltype(ranges::end(ranges::__possibly_const_range(std::declval<_Rng&>())));

  template <__can_borrow _Rng>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr static auto operator()(_Rng&& __rng) noexcept(
      noexcept(const_sentinel<_UType<_Rng>>(ranges::end(ranges::__possibly_const_range(__rng)))))
      -> const_sentinel<_UType<_Rng>> {
    return const_sentinel<_UType<_Rng>>(ranges::end(ranges::__possibly_const_range(__rng)));
  }
#  else // ^^^ _LIBCPP_STD_VER >= 23 / _LIBCPP_STD_VER < 23 vvv
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
#  endif
};
} // namespace __cend

inline namespace __cpo {
inline constexpr auto cend = __cend::__fn{};
} // namespace __cpo

#  if _LIBCPP_STD_VER >= 23
// [range.range]
template <class _Rp>
using const_sentinel_t = decltype(ranges::cend(std::declval<_Rp&>()));
#  endif

// [range.access.crbegin]
namespace __crbegin {
struct __fn {
#  if _LIBCPP_STD_VER >= 23
  template <class _Rng>
  using _UType _LIBCPP_NODEBUG = decltype(ranges::rbegin(ranges::__possibly_const_range(std::declval<_Rng&>())));

  template <__can_borrow _Rng>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr static auto operator()(_Rng&& __rng) noexcept(
      noexcept(const_iterator<_UType<_Rng>>(ranges::rbegin(ranges::__possibly_const_range(__rng)))))
      -> const_iterator<_UType<_Rng>> {
    return const_iterator<_UType<_Rng>>(ranges::rbegin(ranges::__possibly_const_range(__rng)));
  }
#  else  // ^^^ _LIBCPP_STD_VER >= 23 / _LIBCPP_STD_VER < 23
  template <class _Tp>
    requires is_lvalue_reference_v<_Tp&&>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(ranges::rbegin(static_cast<const remove_reference_t<_Tp>&>(__t))))
          -> decltype(ranges::rbegin(static_cast<const remove_reference_t<_Tp>&>(__t))) {
    return ranges::rbegin(static_cast<const remove_reference_t<_Tp>&>(__t));
  }

  template <class _Tp>
    requires is_rvalue_reference_v<_Tp&&>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(ranges::rbegin(static_cast<const _Tp&&>(__t))))
          -> decltype(ranges::rbegin(static_cast<const _Tp&&>(__t))) {
    return ranges::rbegin(static_cast<const _Tp&&>(__t));
  }
#  endif // ^^^ _LIBCPP_STD_VER < 23
};
} // namespace __crbegin

inline namespace __cpo {
inline constexpr auto crbegin = __crbegin::__fn{};
} // namespace __cpo

// [range.access.crend]
namespace __crend {
struct __fn {
#  if _LIBCPP_STD_VER >= 23
  template <class _Rng>
  using _UType _LIBCPP_NODEBUG = decltype(ranges::rend(ranges::__possibly_const_range(std::declval<_Rng&>())));

  template <__can_borrow _Rng>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr static auto operator()(_Rng&& __rng) noexcept(
      noexcept(const_sentinel<_UType<_Rng>>(ranges::rend(ranges::__possibly_const_range(__rng)))))
      -> const_sentinel<_UType<_Rng>> {
    return const_sentinel<_UType<_Rng>>(ranges::rend(ranges::__possibly_const_range(__rng)));
  }
#  else  // ^^^ _LIBCPP_STD_VER >= 23 / _LIBCPP_STD_VER < 23 vvv
  template <class _Tp>
    requires is_lvalue_reference_v<_Tp&&>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(ranges::rend(static_cast<const remove_reference_t<_Tp>&>(__t))))
          -> decltype(ranges::rend(static_cast<const remove_reference_t<_Tp>&>(__t))) {
    return ranges::rend(static_cast<const remove_reference_t<_Tp>&>(__t));
  }

  template <class _Tp>
    requires is_rvalue_reference_v<_Tp&&>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(ranges::rend(static_cast<const _Tp&&>(__t))))
          -> decltype(ranges::rend(static_cast<const _Tp&&>(__t))) {
    return ranges::rend(static_cast<const _Tp&&>(__t));
  }
#  endif // ^^^ _LIBCPP_STD_VER < 23
};
} // namespace __crend

inline namespace __cpo {
inline constexpr auto crend = __crend::__fn{};
} // namespace __cpo

// [range.prim.cdata]

namespace __cdata {
struct __fn {
#  if _LIBCPP_STD_VER >= 23
  template <class _Tp>
  _LIBCPP_HIDE_FROM_ABI constexpr static auto __as_const_pointer(const _Tp* __ptr) noexcept {
    return __ptr;
  }

  template <__can_borrow _Rng>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr static auto
  operator()(_Rng&& __rng) noexcept(noexcept(__as_const_pointer(ranges::data(ranges::__possibly_const_range(__rng)))))
      -> decltype(__as_const_pointer(ranges::data(ranges::__possibly_const_range(__rng)))) {
    return __as_const_pointer(ranges::data(ranges::__possibly_const_range(__rng)));
  }

#  else  // ^^^ _LIBCPP_STD_VER >= 23 / _LIBCPP_STD_VER < 23 vvv
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
#  endif // ^^^ _LIBCPP_STD_VER < 23
};
} // namespace __cdata

inline namespace __cpo {
inline constexpr auto cdata = __cdata::__fn{};
} // namespace __cpo
} // namespace ranges

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RANGES_CONST_ACCESS_H
