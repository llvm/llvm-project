//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__utility/is_pointer_in_range.h>
#include <cassert>

#include "test_macros.h"

template <class T, class U>
TEST_CONSTEXPR_CXX14 void test_cv_quals() {
  T i = 0;
  U j = 0;
  assert(!std::__is_pointer_in_range(&i, &i, &i));
  assert(std::__is_pointer_in_range(&i, &i + 1, &i));
  assert(!std::__is_pointer_in_range(&i, &i + 1, &j));

#if TEST_STD_VER >= 20
  {
    T* arr1 = new int[4]{1, 2, 3, 4};
    U* arr2 = new int[4]{5, 6, 7, 8};

    assert(!std::__is_pointer_in_range(arr1, arr1 + 4, arr2));
    assert(std::__is_pointer_in_range(arr1, arr1 + 4, arr1 + 3));
    assert(!std::__is_pointer_in_range(arr1, arr1, arr1 + 3));

    delete[] arr1;
    delete[] arr2;
  }
#endif
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
#if TEST_STD_VER >= 14
  static_assert(test(), "");
#endif

  return 0;
}
