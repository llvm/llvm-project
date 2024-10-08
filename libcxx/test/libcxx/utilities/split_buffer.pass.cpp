//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__split_buffer>
#include <__config>
#include <cassert>
#include <memory>
#include <type_traits>
#include <string>

#include "test_macros.h"

struct simple_test_type {
  int value;
};

struct complex_test_type {
  std::string value;
  complex_test_type(const std::string& v) : value(v) {}
};

template <class TEST_TYPE>
_LIBCPP_CONSTEXPR_SINCE_CXX20 int capacity_after_initialization() {
  std::__split_buffer<TEST_TYPE> sb;
  return sb.capacity();
}

template <class TEST_TYPE>
_LIBCPP_CONSTEXPR_SINCE_CXX20 int capacity_after_reserve(const std::size_t n) {
  std::__split_buffer<TEST_TYPE> sb;
  sb.reserve(n);
  return sb.capacity();
}

int main() {
  { // check test_types' features
#if _LIBCPP_STD_VER >= 17
    static_assert(std::is_trivially_copyable_v<simple_test_type>);
    static_assert(not std::is_trivially_copyable_v<complex_test_type>);
#endif
  }

  { // test simple_test_type at run-time
    assert(0 == capacity_after_initialization<simple_test_type>());
    assert(42 == capacity_after_reserve<simple_test_type>(42));
  }

  { // test complex_test_type at run-time
    assert(0 == capacity_after_initialization<complex_test_type>());
    assert(42 == capacity_after_reserve<complex_test_type>(42));
  }

#if _LIBCPP_STD_VER >= 20
  { // test simple_test_type at compile-time
    static_assert(0 == capacity_after_initialization<simple_test_type>());
    static_assert(42 == capacity_after_reserve<simple_test_type>(42));
  }

  { // test complex_test_type at compile-time
    static_assert(0 == capacity_after_initialization<complex_test_type>());
    static_assert(42 == capacity_after_reserve<complex_test_type>(42));
  }
#endif
}
