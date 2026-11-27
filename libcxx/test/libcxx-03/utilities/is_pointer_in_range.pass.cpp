//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__cxx03/__utility/is_pointer_in_range.h>
#include <cassert>

#include "test_macros.h"

template <class T, class U>
TEST_CONSTEXPR_CXX14 void test_cv_quals() {
  T i = 0;
  U j = 0;
  assert(!std::__is_pointer_in_range(&i, &i, &i));
  assert(std::__is_pointer_in_range(&i, &i + 1, &i));
  assert(!std::__is_pointer_in_range(&i, &i + 1, &j));
}

TEST_CONSTEXPR_CXX14 bool test() {
  test_cv_quals<int, int>();
  test_cv_quals<const int, int>();
  test_cv_quals<int, const int>();
  test_cv_quals<const int, const int>();
  test_cv_quals<volatile int, int>();
  test_cv_quals<int, volatile int>();
  test_cv_quals<volatile int, volatile int>();

  return true;
}

int main(int, char**) {
  test();

  return 0;
}
