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

template <class T, class TQualified>
TEST_CONSTEXPR_CXX14 void check_type() {
  {
    // We need to ensure that the addresses of i and j are ordered as &i < &j for
    // the test below to work portably, so we define them in a struct.
    struct {
      T i = 0;
      T j = 0;
    } storage;
    assert(std::__is_valid_range(static_cast<TQualified*>(&storage.i), static_cast<TQualified*>(&storage.i)));
    assert(std::__is_valid_range(static_cast<TQualified*>(&storage.i), static_cast<TQualified*>(&storage.i + 1)));

    assert(!std::__is_valid_range(static_cast<TQualified*>(&storage.j), static_cast<TQualified*>(&storage.i)));
    assert(!std::__is_valid_range(static_cast<TQualified*>(&storage.i + 1), static_cast<TQualified*>(&storage.i)));

    // We detect this as being a valid range even though it is not really valid.
    assert(std::__is_valid_range(static_cast<TQualified*>(&storage.i), static_cast<TQualified*>(&storage.j)));
  }

  {
    T arr[3] = {1, 2, 3};
    assert(std::__is_valid_range(static_cast<TQualified*>(&arr[0]), static_cast<TQualified*>(&arr[0])));
    assert(std::__is_valid_range(static_cast<TQualified*>(&arr[0]), static_cast<TQualified*>(&arr[1])));
    assert(std::__is_valid_range(static_cast<TQualified*>(&arr[0]), static_cast<TQualified*>(&arr[2])));

    assert(!std::__is_valid_range(static_cast<TQualified*>(&arr[1]), static_cast<TQualified*>(&arr[0])));
    assert(!std::__is_valid_range(static_cast<TQualified*>(&arr[2]), static_cast<TQualified*>(&arr[0])));
  }

#if TEST_STD_VER >= 20
  {
    T* arr = new int[4]{1, 2, 3, 4};
    assert(std::__is_valid_range(static_cast<TQualified*>(arr), static_cast<TQualified*>(arr + 4)));
    delete[] arr;
  }
#endif
}

TEST_CONSTEXPR_CXX14 bool test() {
  check_type<int, int>();
  check_type<int, int const>();
  check_type<int, int volatile>();
  check_type<int, int const volatile>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 14
  static_assert(test(), "");
#endif

  return 0;
}
