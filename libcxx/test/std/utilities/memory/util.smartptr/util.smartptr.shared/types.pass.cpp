//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template<class T> class shared_ptr
// {
// public:
//     typedef T element_type;
//     typedef weak_ptr<T> weak_type; // C++17
//     ...
// };

#include <memory>

#include "test_macros.h"

#if TEST_STD_VER > 14
template <typename T, typename = std::void_t<> >
struct has_less : std::false_type {};

template <typename T>
struct has_less<T,
                std::void_t<decltype(std::declval<T>() < std::declval<T>())> >
    : std::true_type {};
#endif

struct A;  // purposefully incomplete
struct B {
  int x;
  B() = default;
};

template <class T>
void test() {
  ASSERT_SAME_TYPE(typename std::shared_ptr<T>::element_type, T);
#if TEST_STD_VER > 14
  ASSERT_SAME_TYPE(typename std::shared_ptr<T>::weak_type, std::weak_ptr<T>);
  static_assert(std::is_copy_constructible<std::shared_ptr<T> >::value, "");
  static_assert(std::is_copy_assignable<std::shared_ptr<T> >::value, "");
  static_assert(has_less<std::shared_ptr<T> >::value);
  static_assert(
      std::is_same<typename std::shared_ptr<T[]>::element_type, T>::value, "");
  static_assert(
      std::is_same<typename std::shared_ptr<T[8]>::element_type, T>::value, "");
#endif
}

int main(int, char**) {
  test<A>();
  test<B>();
  test<int>();
  test<char*>();

  return 0;
}
