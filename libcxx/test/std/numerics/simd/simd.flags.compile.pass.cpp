//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <simd>

#include <simd>
#include <type_traits>

namespace dp = std::datapar;

template <class T>
struct unwrap_flag;

template <class T>
struct unwrap_flag<dp::simd_flags<T>> {
  using type = T;
};

using aligned_flag_t = unwrap_flag<std::remove_cvref_t<decltype(dp::simd_flag_convert)>>::type;
using convert_flag_t = unwrap_flag<std::remove_cvref_t<decltype(dp::simd_flag_aligned)>>::type;
template <std::size_t N>
using overaligned_flag_t = unwrap_flag<std::remove_cvref_t<decltype(dp::simd_flag_overaligned<N>)>>::type;

static_assert(std::is_same_v<dp::simd_flags<>, decltype(dp::simd_flags<>{} | dp::simd_flags<>{})>);

template <class, class>
constexpr bool contains_type_v = false;

template <class... Args, class T>
constexpr bool contains_type_v<dp::simd_flags<Args...>, T> = (std::is_same_v<Args, T> || ...);

template <class RequiredT, class... LHSArgs, class... RHSArgs>
consteval bool test(dp::simd_flags<LHSArgs...> lhs, dp::simd_flags<RHSArgs...> rhs) {
  return contains_type_v<decltype(lhs | rhs), RequiredT>;
}

static_assert(test<aligned_flag_t>(dp::simd_flags<aligned_flag_t>{}, dp::simd_flags<aligned_flag_t>{}));
static_assert(test<aligned_flag_t>(dp::simd_flags<>{}, dp::simd_flags<aligned_flag_t>{}));
static_assert(test<aligned_flag_t>(dp::simd_flags<aligned_flag_t>{}, dp::simd_flags<>{}));
static_assert(!test<aligned_flag_t>(dp::simd_flags<convert_flag_t>{}, dp::simd_flags<>{}));
static_assert(test<convert_flag_t>(dp::simd_flags<convert_flag_t>{}, dp::simd_flags<>{}));
static_assert(test<convert_flag_t>(dp::simd_flags<>{}, dp::simd_flags<convert_flag_t>{}));
static_assert(test<convert_flag_t>(dp::simd_flags<convert_flag_t>{}, dp::simd_flags<convert_flag_t>{}));
static_assert(test<overaligned_flag_t<1>>(dp::simd_flags<overaligned_flag_t<1>>{}, dp::simd_flags<>{}));
static_assert(test<overaligned_flag_t<1>>(dp::simd_flags<>{}, dp::simd_flags<overaligned_flag_t<1>>{}));
static_assert(test<overaligned_flag_t<1>>(dp::simd_flags<overaligned_flag_t<1>>{}, dp::simd_flags<overaligned_flag_t<1>>{}));
static_assert(test<overaligned_flag_t<16>>(dp::simd_flags<overaligned_flag_t<16>>{}, dp::simd_flags<overaligned_flag_t<1>>{}));
