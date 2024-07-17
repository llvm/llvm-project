//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <utility>
#include <cassert>

struct A {
  int data_ = 0;
};

inline bool operator==(const A& x, const A& y) { return x.data_ == y.data_; }

inline bool operator<(const A& x, const A& y) { return x.data_ < y.data_; }

void test() {
  using namespace std::rel_ops;
  A a1{1};
  A a2{2};
  (void)(a1 == a1);
  (void)(a1 != a2);                 // note not deprecated message, due to compiler generated operator.
  std::rel_ops::operator!=(a1, a2); // expected-warning {{is deprecated}}
  (void)(a1 < a2);
  (void)(a1 > a2);  // expected-warning 2 {{is deprecated}}
  (void)(a1 <= a2); // expected-warning 2 {{is deprecated}}
  (void)(a1 >= a2); // expected-warning 2 {{is deprecated}}
}
