//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___SIMD_SIMD_FLAGS_H
#define _LIBCPP___SIMD_SIMD_FLAGS_H

#include <__algorithm/max.h>
#include <__bit/has_single_bit.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__type_traits/pack_utils.h>

#if _LIBCPP_STD_VER >= 26

_LIBCPP_BEGIN_NAMESPACE_STD
namespace datapar {

template <class>
inline constexpr bool __is_flag_v = false;

struct __aligned_flag {};

template <>
inline constexpr bool __is_flag_v<__aligned_flag> = true;

struct __convert_flag {};

template <>
inline constexpr bool __is_flag_v<__convert_flag> = true;

template <size_t _Np>
struct __overaligned_flag {};

template <size_t _Np>
inline constexpr bool __is_flag_v<__overaligned_flag<_Np>> = true;

template <class... _Args>
inline constexpr size_t __get_max_overaligned = 0;

template <size_t _Np, class... _Args>
inline constexpr size_t __get_max_overaligned<__overaligned_flag<_Np>, _Args...> =
    std::max(_Np, __get_max_overaligned<_Args...>);

template <class _Tp, class... _Args>
inline constexpr size_t __get_max_overaligned<_Tp, _Args...> = __get_max_overaligned<_Args...>;

// TODO: Also add alignment when __aligned_flag is in _Args
template <class _Tp, class... _Args>
inline constexpr size_t __get_align_for = std::max(1uz, __get_max_overaligned<_Args...>);

template <class... _Flags>
  requires(__is_flag_v<_Flags> && ...)
struct simd_flags {
  template <class... _Args, class... _Result>
  static consteval auto __copy_aligned(simd_flags<_Args...>, simd_flags<_Result...>) {
    if constexpr (__contains_type_v<__type_list<_Args...>, __aligned_flag>) {
      return simd_flags<__aligned_flag, _Result...>{};
    } else {
      return simd_flags<_Result...>{};
    }
  }

  template <class... _Args, class... _Result>
  static consteval auto __copy_convert(simd_flags<_Args...>, simd_flags<_Result...>) {
    if constexpr (__contains_type_v<__type_list<_Args...>, __convert_flag>) {
      return simd_flags<__convert_flag, _Result...>{};
    } else {
      return simd_flags<_Result...>{};
    }
  }

  template <class... _Args, class... _Result>
  static consteval auto __copy_overaligned(simd_flags<_Args...>, simd_flags<_Result...>) {
    if constexpr (constexpr auto __max_align = __get_max_overaligned<_Args...>; __max_align > 0) {
      return simd_flags<__overaligned_flag<__max_align>, _Result...>{};
    } else {
      return simd_flags<_Result...>{};
    }
  }

  template <class... _Other>
  friend consteval auto operator|(simd_flags, simd_flags<_Other...>) {
    using _Combined = simd_flags<_Flags..., _Other...>;
    return __copy_overaligned(_Combined{}, __copy_convert(_Combined{}, __copy_aligned(_Combined{}, {})));
  }
};

inline constexpr simd_flags<> simd_flag_default{};
inline constexpr simd_flags<__convert_flag> simd_flag_convert{};
inline constexpr simd_flags<__aligned_flag> simd_flag_aligned{};

template <size_t _Np>
  requires(std::has_single_bit(_Np))
inline constexpr simd_flags<__overaligned_flag<_Np>> simd_flag_overaligned{};

} // namespace datapar
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

#endif // _LIBCPP___SIMD_SIMD_FLAGS_H
