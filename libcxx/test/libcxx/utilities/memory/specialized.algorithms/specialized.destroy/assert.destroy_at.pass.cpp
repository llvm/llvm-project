//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T>
// constexpr void destroy_at(T* location);

// REQUIRES: can-test-hardening-assertions-extensive
// UNSUPPORTED: c++03, c++11, c++14

#include <memory>

#include "check_assertion.h"
#include "test_macros.h"

struct NonTrivial {
  ~NonTrivial() {}
};

int main(int, char**) {
  int* null_int           = nullptr;
  NonTrivial* null_struct = nullptr;

  TEST_LIBCPP_ASSERT_FAILURE(std::destroy_at(null_int), "null pointer given to destroy_at");
  TEST_LIBCPP_ASSERT_FAILURE(std::destroy_at(null_struct), "null pointer given to destroy_at");

#if TEST_STD_VER >= 20
  int (*null_arr)[3] = nullptr;
  TEST_LIBCPP_ASSERT_FAILURE(std::destroy_at(null_arr), "null pointer given to destroy_at");
  TEST_LIBCPP_ASSERT_FAILURE(std::ranges::destroy_at(null_int), "null pointer given to destroy_at");
  TEST_LIBCPP_ASSERT_FAILURE(std::ranges::destroy_at(null_struct), "null pointer given to destroy_at");
#endif

  return 0;
}
