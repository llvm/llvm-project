//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>

// move_sentinel

// constexpr explicit move_sentinel(S s);

#include <iterator>
#include <cassert>

constexpr bool test()
{
  {
    static_assert(!std::is_convertible_v<int, std::move_sentinel<int>>);
    std::move_sentinel<int> m(42);
    assert(m.base() == 42);
  }
  {
    static_assert(!std::is_convertible_v<int*, std::move_sentinel<int*>>);
    int i = 42;
    std::move_sentinel<int*> m(&i);
    assert(m.base() == &i);
  }
  {
    struct S {
      explicit S() = default;
      constexpr explicit S(int j) : i(j) {}
      int i = 3;
    };
    static_assert(!std::is_convertible_v<S, std::move_sentinel<S>>);
    std::move_sentinel<S> m(S(42));
    assert(m.base().i == 42);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
