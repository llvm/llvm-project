//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class Ptr>
// struct pointer_traits
// {
//     static pointer pointer_to(<details>);
//     ...
// };

#include <memory>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <class T>
struct A {
private:
  struct nat {};

public:
  typedef T element_type;
  element_type* t_;

  A(element_type* t) : t_(t) {}

  static A pointer_to(typename std::conditional<std::is_void<element_type>::value, nat, element_type>::type& et) {
    return A(&et);
  }
};

template <class Pointer>
void test() {
  typename Pointer::element_type obj;
  static_assert(std::is_same<Pointer, decltype(std::pointer_traits<Pointer>::pointer_to(obj))>::value, "");
  Pointer p = std::pointer_traits<Pointer>::pointer_to(obj);
  assert(p.t_ == &obj);
}

int main(int, char**) {
  test<A<int> >();
  test<A<long> >();
  { (std::pointer_traits<A<void> >::element_type)0; }

  return 0;
}
