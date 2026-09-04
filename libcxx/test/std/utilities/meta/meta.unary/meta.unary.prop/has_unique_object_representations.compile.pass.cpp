//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// type_traits

// has_unique_object_representations

#include <type_traits>

template <bool ExpectedValue, class T>
void test() {
  static_assert(std::has_unique_object_representations<T>::value == ExpectedValue);
  static_assert(std::has_unique_object_representations<const T>::value == ExpectedValue);
  static_assert(std::has_unique_object_representations<volatile T>::value == ExpectedValue);
  static_assert(std::has_unique_object_representations<const volatile T>::value == ExpectedValue);

  static_assert(std::has_unique_object_representations_v<T> == ExpectedValue);
  static_assert(std::has_unique_object_representations_v<const T> == ExpectedValue);
  static_assert(std::has_unique_object_representations_v<volatile T> == ExpectedValue);
  static_assert(std::has_unique_object_representations_v<const volatile T> == ExpectedValue);
}

class Empty {};

union EmptyUnion {};

struct NonEmptyUnion {
  int x;
  unsigned y;
};

struct ZeroWidthBitfield {
  int : 0;
};

class Virtual {
  virtual ~Virtual();
};

class Abstract {
  virtual ~Abstract() = 0;
};

struct UnsignedInt {
  unsigned foo;
};

struct WithoutPadding {
  int x;
  int y;
};

struct WithPadding {
  char bar;
  int foo;
};

template <int>
class NTTP_ClassType_WithoutPadding {
  int x;
};

void test() {
  test<false, void>();
  test<false, Empty>();
  test<false, EmptyUnion>();
  test<false, Virtual>();
  test<false, ZeroWidthBitfield>();
  test<false, Abstract>();
  test<false, WithPadding>();
  test<false, WithPadding[]>();
  test<false, WithPadding[][3]>();

  // I would also expect that there are systems where they do not.
  // I would expect all three of these to have unique representations.
  //   test<false, int&>();
  //   test<false, int *>();
  //   test<false, double>();

  test<true, unsigned>();
  test<true, UnsignedInt>();
  test<true, WithoutPadding>();
  test<true, NonEmptyUnion>();
  test<true, char[3]>();
  test<true, char[3][4]>();
  test<true, char[3][4][5]>();
  test<true, char[]>();
  test<true, char[][2]>();
  test<true, char[][2][3]>();

  // Important test case for https://llvm.org/PR95311.
  // Note that the order is important here, we want to instantiate the array
  // variants before the non-array ones, otherwise we don't trigger the bug.
  {
    test<true, NTTP_ClassType_WithoutPadding<0>[]>();
    test<true, NTTP_ClassType_WithoutPadding<0>[][3]>();
    test<true, NTTP_ClassType_WithoutPadding<0>>();
  }
}
