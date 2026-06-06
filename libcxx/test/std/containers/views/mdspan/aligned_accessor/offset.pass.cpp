//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <mdspan>

// constexpr typename offset_policy::data_handle_type offset(data_handle_type p, size_t i) const noexcept;
//
// Effects: Equivalent to: return assume_aligned<byte_alignment>(p) + i;

#include <mdspan>
#include <cassert>
#include <cstddef>
#include <concepts>
#include <type_traits>

#include "test_macros.h"

// We are not using MinimalElementType.h because MinimalElementType is not
// default constructible and uninitialized storage does not work in constexpr.

// Same as MinimalElementType but with a defaulted default constructor
struct MyMinimalElementType {
  int val;
  constexpr MyMinimalElementType()                            = default;
  constexpr MyMinimalElementType(const MyMinimalElementType&) = delete;
  constexpr explicit MyMinimalElementType(int v) noexcept : val(v) {}
  constexpr MyMinimalElementType& operator=(const MyMinimalElementType&) = delete;
};

template <class T, std::size_t N>
constexpr void test_offset() {
  constexpr std::size_t Sz = 10;
  alignas(N) T data[Sz]{};
  T* ptr = &data[0];
  std::aligned_accessor<T, N> acc;
  for (std::size_t i = 0; i < Sz; ++i) {
    std::same_as<typename std::default_accessor<T>::data_handle_type> decltype(auto) x = acc.offset(ptr, i);
    ASSERT_NOEXCEPT(acc.offset(ptr, i));
    assert(x == ptr + i);
  }
}

template <class T>
constexpr void test_it() {
  constexpr std::size_t N = alignof(T);
  test_offset<T, N>();
  test_offset<T, 2 * N>();
  test_offset<T, 4 * N>();
  test_offset<T, 8 * N>();
  test_offset<T, 16 * N>();
}

constexpr bool test() {
  test_it<int>();
  test_it<const int>();
  test_it<MyMinimalElementType>();
  test_it<const MyMinimalElementType>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
