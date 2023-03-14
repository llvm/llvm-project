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
// [simd.class]
// template <class U> simd(U&& value);

#include <experimental/simd>
#include <cstdint>
#include <cassert>

#include "test_macros.h"

namespace ex = std::experimental::parallelism_v2;

template <class T, class... Args>
auto not_supported_native_simd_ctor(Args&&... args)
    -> decltype(ex::native_simd<T>(std::forward<Args>(args)...),
                void()) = delete;

template <class T>
void not_supported_native_simd_ctor(...) {}

template <class T, class... Args>
auto supported_native_simd_ctor(Args&&... args)
    -> decltype(ex::native_simd<T>(std::forward<Args>(args)...), void()) {}

template <class T>
void supported_native_simd_ctor(...) = delete;

void compile_narrowing_conversion() {
  supported_native_simd_ctor<std::int8_t>(3);
  supported_native_simd_ctor<std::int16_t>(3);
  supported_native_simd_ctor<std::int32_t>(3);
  supported_native_simd_ctor<std::int64_t>(3);
  supported_native_simd_ctor<std::uint8_t>(3);
  supported_native_simd_ctor<std::uint16_t>(3);
  supported_native_simd_ctor<std::uint32_t>(3);
  supported_native_simd_ctor<std::uint64_t>(3);
  supported_native_simd_ctor<float>(3.f);
  supported_native_simd_ctor<double>(3.);
  supported_native_simd_ctor<long double>(3.);

  not_supported_native_simd_ctor<float>(3.);
  not_supported_native_simd_ctor<std::int8_t>(long(3));
  not_supported_native_simd_ctor<float>(long(3));
  not_supported_native_simd_ctor<int>(3.);
}

void compile_convertible() {
  struct ConvertibleToInt {
    operator std::int64_t() const;
  };
  supported_native_simd_ctor<std::int64_t>(ConvertibleToInt());

  struct NotConvertibleToInt {};
  not_supported_native_simd_ctor<std::int64_t>(NotConvertibleToInt());
}

void compile_unsigned() {
  not_supported_native_simd_ctor<int>(3u);
  supported_native_simd_ctor<std::uint16_t>(3u);
}

template <typename SimdType>
void test_broadcast() {
  SimdType a(3);
  for (std::size_t i = 0; i < a.size(); i++) {
    assert(a[i] == 3);
  }
}

int main(int, char**) {
  test_broadcast<ex::native_simd<int>>();
  test_broadcast<ex::fixed_size_simd<int, 4>>();
  test_broadcast<ex::fixed_size_simd<int, 15>>();

  return 0;
}
