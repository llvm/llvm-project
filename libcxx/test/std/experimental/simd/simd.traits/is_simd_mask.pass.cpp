//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <experimental/simd>
//
// [simd.traits]
// template <class T> struct ex::is_simd_mask;
// template <class T> inline constexpr bool ex::is_simd_mask_v =
// ex::is_simd_mask<T>::value;

#include <cstdint>
#include <experimental/simd>
#include "test_macros.h"

namespace ex = std::experimental::parallelism_v2;

struct UserType {};

static_assert(ex::is_simd_mask<ex::native_simd_mask<std::int8_t>>::value, "");
static_assert(ex::is_simd_mask<ex::native_simd_mask<std::int16_t>>::value, "");
static_assert(ex::is_simd_mask<ex::native_simd_mask<std::int32_t>>::value, "");
static_assert(ex::is_simd_mask<ex::native_simd_mask<std::int64_t>>::value, "");
static_assert(ex::is_simd_mask<ex::native_simd_mask<std::uint8_t>>::value, "");
static_assert(ex::is_simd_mask<ex::native_simd_mask<std::uint16_t>>::value, "");
static_assert(ex::is_simd_mask<ex::native_simd_mask<std::uint32_t>>::value, "");
static_assert(ex::is_simd_mask<ex::native_simd_mask<std::uint64_t>>::value, "");
static_assert(ex::is_simd_mask<ex::native_simd_mask<float>>::value, "");
static_assert(ex::is_simd_mask<ex::native_simd_mask<double>>::value, "");

static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::int8_t, 1>>::value, "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::int16_t, 1>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::int32_t, 1>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::int64_t, 1>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::uint8_t, 1>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::uint16_t, 1>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::uint32_t, 1>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::uint64_t, 1>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<float, 1>>::value, "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<double, 1>>::value, "");

static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::int8_t, 3>>::value, "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::int16_t, 3>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::int32_t, 3>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::int64_t, 3>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::uint8_t, 3>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::uint16_t, 3>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::uint32_t, 3>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::uint64_t, 3>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<float, 3>>::value, "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<double, 3>>::value, "");

static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::int8_t, 32>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::int16_t, 32>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::int32_t, 32>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::int64_t, 32>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::uint8_t, 32>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::uint16_t, 32>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::uint32_t, 32>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<std::uint64_t, 32>>::value,
              "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<float, 32>>::value, "");
static_assert(ex::is_simd_mask<ex::fixed_size_simd_mask<double, 32>>::value,
              "");

static_assert(!ex::is_simd_mask<void>::value, "");
static_assert(!ex::is_simd_mask<int>::value, "");
static_assert(!ex::is_simd_mask<float>::value, "");
static_assert(!ex::is_simd_mask<ex::simd<int>>::value, "");
static_assert(!ex::is_simd_mask<ex::simd<float>>::value, "");
static_assert(!ex::is_simd_mask<UserType>::value, "");

static_assert(ex::is_simd_mask_v<ex::native_simd_mask<std::int8_t>>, "");
static_assert(ex::is_simd_mask_v<ex::native_simd_mask<std::int16_t>>, "");
static_assert(ex::is_simd_mask_v<ex::native_simd_mask<std::int32_t>>, "");
static_assert(ex::is_simd_mask_v<ex::native_simd_mask<std::int64_t>>, "");
static_assert(ex::is_simd_mask_v<ex::native_simd_mask<std::uint8_t>>, "");
static_assert(ex::is_simd_mask_v<ex::native_simd_mask<std::uint16_t>>, "");
static_assert(ex::is_simd_mask_v<ex::native_simd_mask<std::uint32_t>>, "");
static_assert(ex::is_simd_mask_v<ex::native_simd_mask<std::uint64_t>>, "");
static_assert(ex::is_simd_mask_v<ex::native_simd_mask<float>>, "");
static_assert(ex::is_simd_mask_v<ex::native_simd_mask<double>>, "");

static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::int8_t, 1>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::int16_t, 1>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::int32_t, 1>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::int64_t, 1>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::uint8_t, 1>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::uint16_t, 1>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::uint32_t, 1>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::uint64_t, 1>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<float, 1>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<double, 1>>, "");

static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::int8_t, 3>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::int16_t, 3>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::int32_t, 3>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::int64_t, 3>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::uint8_t, 3>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::uint16_t, 3>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::uint32_t, 3>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::uint64_t, 3>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<float, 3>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<double, 3>>, "");

static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::int8_t, 32>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::int16_t, 32>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::int32_t, 32>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::int64_t, 32>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::uint8_t, 32>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::uint16_t, 32>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::uint32_t, 32>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<std::uint64_t, 32>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<float, 32>>, "");
static_assert(ex::is_simd_mask_v<ex::fixed_size_simd_mask<double, 32>>, "");

static_assert(!ex::is_simd_mask_v<void>, "");
static_assert(!ex::is_simd_mask_v<int>, "");
static_assert(!ex::is_simd_mask_v<float>, "");
static_assert(!ex::is_simd_mask_v<ex::simd<int>>, "");
static_assert(!ex::is_simd_mask_v<ex::simd<float>>, "");
static_assert(!ex::is_simd_mask_v<UserType>, "");

int main(int, char**) {
  return 0;
}
