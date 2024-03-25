//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept;
//
// Effects: Equivalent to: return p+i;

#include <mdspan>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

#include "../MinimalElementType.h"

template <class T>
constexpr void test_offset() {
  ElementPool<std::remove_const_t<T>, 10> data;
  T* ptr = data.get_ptr();
  std::default_accessor<T> acc;
  for(int i = 0; i < 10; i++) {
    static_assert(std::is_same_v<decltype(acc.offset(ptr, i)), typename std::default_accessor<T>::data_handle_type>);
    ASSERT_NOEXCEPT(acc.offset(ptr, i));
    assert(acc.offset(ptr, i) == ptr + i);
  }
}

constexpr bool test() {
  test_offset<int>();
  test_offset<const int>();
  test_offset<MinimalElementType>();
  test_offset<const MinimalElementType>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
