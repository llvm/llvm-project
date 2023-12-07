//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// tuple_element<I, pair<T1, T2> >::type

#include <utility>

void f() {
  typedef std::pair<int, short> T;
  std::tuple_element<2, T>::type foo; // expected-error@*:* {{Index out of bounds in std::tuple_element<std::pair<T1, T2>>}}
}
