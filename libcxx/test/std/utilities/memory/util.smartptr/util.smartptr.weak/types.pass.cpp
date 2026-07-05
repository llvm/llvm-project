//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template<class T> class weak_ptr
// {
// public:
//     typedef T element_type; // until C++17
//     typedef remove_extent_t<T> element_type; // since C++17
//     ...
// };

#include <memory>
#include <type_traits>

#include "test_macros.h"

struct A;  // purposefully incomplete
struct B {
  int x;
  B() = default;
};

template <class T>
void test() {
  ASSERT_SAME_TYPE(typename std::weak_ptr<T>::element_type, T);
#if TEST_STD_VER > 14
  ASSERT_SAME_TYPE(typename std::weak_ptr<T[]>::element_type, T);
  ASSERT_SAME_TYPE(typename std::weak_ptr<T[8]>::element_type, T);
#endif
}

int main(int, char**)
{
  test<A>();
  test<B>();
  test<int>();
  test<char*>();

#if TEST_STD_VER > 14
  ASSERT_SAME_TYPE(typename std::weak_ptr<int[][2]>::element_type, int[2]);
#endif

  return 0;
}
