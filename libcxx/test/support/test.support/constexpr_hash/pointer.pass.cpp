//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// "support/constexpr_hash.h"

// Not very portable

#include "constexpr_hash.h"

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "test_macros.h"

template <class T>
TEST_CONSTEXPR_CXX26 void test() {
  typedef support::constexpr_hash<T> H;
#if TEST_STD_VER <= 17
  static_assert((std::is_same<typename H::argument_type, T>::value), "");
  static_assert((std::is_same<typename H::result_type, std::size_t>::value), "");
#endif
  ASSERT_NOEXCEPT(H()(T()));
  H h;

  typedef typename std::remove_pointer<T>::type type;
  type i;
  type j;
  assert(h(&i) != h(&j));
}

TEST_CONSTEXPR_CXX26 void test_nullptr() {
  typedef std::nullptr_t T;
  typedef support::constexpr_hash<T> H;
#if TEST_STD_VER <= 17
  static_assert((std::is_same<typename H::argument_type, T>::value), "");
  static_assert((std::is_same<typename H::result_type, std::size_t>::value), "");
#endif
  ASSERT_NOEXCEPT(H()(T()));
}

TEST_CONSTEXPR_CXX26 bool test_with_hash() {
  test_nullptr();
  return true;
}

int main(int, char**) {
  assert(test_with_hash());
  // test<int*>();

#if TEST_STD_VER >= 26
  static_assert(test_with_hash());
#endif

  return 0;
}
