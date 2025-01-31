//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <utility>

// template <class T1, class T2> struct pair

// template<size_t I, class T1, class T2>
//     typename tuple_element<I, std::pair<T1, T2> >::type&&
//     get(pair<T1, T2>&&);

#include <cassert>
#include <utility>

#include "test_macros.h"

TEST_CONSTEXPR_CXX14 bool test() {
  int i = 1;
  int j = 2;

  {
    std::pair<int&, int&&> p(i, std::move(j));
    assert(&std::get<int&>(p) == &i);
    assert(&std::get<int&&>(p) == &j);

    assert(&std::get<int&>(std::move(p)) == &i);
    assert(std::get<int&&>(std::move(p)) == 2);

    const std::pair<int&, int&&> cp(i, std::move(j));
    assert(&std::get<int&>(cp) == &i);
    assert(&std::get<int&&>(cp) == &j);

    assert(&std::get<int&>(std::move(cp)) == &i);
    assert(std::get<int&&>(std::move(cp)) == 2);
  }

  {
    std::pair<int&&, int&> p(std::move(i), j);
    assert(&std::get<int&>(p) == &j);
    assert(&std::get<int&&>(p) == &i);

    assert(&std::get<int&>(std::move(p)) == &j);
    assert(std::get<int&&>(std::move(p)) == 1);

    const std::pair<int&&, int&> cp(std::move(i), j);
    assert(&std::get<int&>(cp) == &j);
    assert(&std::get<int&&>(cp) == &i);

    assert(&std::get<int&>(std::move(cp)) == &j);
    assert(std::get<int&&>(std::move(cp)) == 1);
  }

  return true;
}

int main() {
  test();
#if TEST_STD_VER >= 14
  static_assert(test(), "");
#endif
}
