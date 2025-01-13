//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <mdspan>

// constexpr reference access(data_handle_type p, size_t i) const noexcept;
//
// Effects: Equivalent to: return assume_aligned<byte_alignment>(p)[i];

#include <mdspan>
#include <cassert>
#include <cstdint>
#include <concepts>
#include <type_traits>

#include "test_macros.h"

#include "../MinimalElementType.h"

template <class T, size_t N>
constexpr void test_access() {
  ElementPool<std::remove_const_t<T>, 10 + N> data;
  T* ptr = data.get_ptr();
  // align ptr
  for (size_t i = 0; i < N; ++i) {
    if (reinterpret_cast<std::uintptr_t>(ptr + i) % N == 0) {
      ptr += i;
      break;
    }
  }
  std::aligned_accessor<T, N> acc;
  for (size_t i = 0; i < 10; ++i) {
    std::same_as<typename std::aligned_accessor<T, N>::reference> decltype(auto) x = acc.access(ptr, i);
    ASSERT_NOEXCEPT(acc.access(ptr, i));
    assert(&x == ptr + i);
  }
}

template <class T>
constexpr void test_it() {
  constexpr size_t N = alignof(T);
  test_access<T, N>();
  test_access<T, 2 * N>();
  test_access<T, 4 * N>();
  test_access<T, 8 * N>();
  test_access<T, 16 * N>();
}

constexpr bool test() {
  test_it<int>();
  test_it<const int>();
  test_it<MinimalElementType>();
  test_it<const MinimalElementType>();
  return true;
}

int main(int, char**) {
  test();
  //static_assert(test());
  return 0;
}
