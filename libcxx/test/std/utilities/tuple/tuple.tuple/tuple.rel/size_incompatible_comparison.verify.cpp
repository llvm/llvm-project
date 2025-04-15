//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template<class... TTypes, class... UTypes>
//   bool
//   operator==(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
// template<class... TTypes, class... UTypes>
//   bool
//   operator<(const tuple<TTypes...>& t, const tuple<UTypes...>& u);

// UNSUPPORTED: c++03

#include <tuple>

#include "test_macros.h"

void f(std::tuple<int> t1, std::tuple<int, long> t2) {
  // We test only the core comparison operators and trust that the others
  // fall back on the same implementations prior to C++20.
#if TEST_STD_VER >= 26
  static_cast<void>(t1 == t2);
#else
  static_cast<void>(t1 == t2); // expected-error@*:* {{}}
#endif
  static_cast<void>(t1 < t2); // expected-error@*:* {{}}
}
