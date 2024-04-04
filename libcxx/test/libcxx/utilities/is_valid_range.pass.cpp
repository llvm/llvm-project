//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__utility/is_valid_range.h>
#include <cassert>

#include "test_macros.h"

template <class T>
TEST_CONSTEXPR_CXX14 void check_type() {
  {
    T i = 0;
    T j = 0;
    assert(std::__is_valid_range(&i, &i));

    assert(!std::__is_valid_range(&i, &j));
    assert(!std::__is_valid_range(&i + 1, &i));

    // We detect this one as being a valid range.
    // Ideally we would detect it as an invalid range, but this may not be implementable.
    assert(std::__is_valid_range(&i, &i + 1));
  }

  {
    T arr[3] = {1, 2, 3};
    assert(std::__is_valid_range(&arr[0], &arr[0]));
    assert(std::__is_valid_range(&arr[0], &arr[1]));
    assert(std::__is_valid_range(&arr[0], &arr[2]));

    assert(!std::__is_valid_range(&arr[1], &arr[0]));
    assert(!std::__is_valid_range(&arr[2], &arr[0]));
  }

#if TEST_STD_VER >= 20
  {
    T* arr = new int[4]{1, 2, 3, 4};
    assert(std::__is_valid_range(arr, arr + 4));
    delete[] arr;
  }
#endif
}

TEST_CONSTEXPR_CXX14 bool test() {
  check_type<int>();
  check_type<const int>();
  check_type<volatile int>();
  check_type<const volatile int>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 14
  static_assert(test(), "");
#endif

  return 0;
}
