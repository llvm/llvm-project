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

// constexpr move_sentinel();

#include <iterator>
#include <cassert>

constexpr bool test()
{
  {
    std::move_sentinel<int> m;
    assert(m.base() == 0);
  }
  {
    std::move_sentinel<int*> m;
    assert(m.base() == nullptr);
  }
  {
    struct S {
      explicit S() = default;
      int i = 3;
    };
    std::move_sentinel<S> m;
    assert(m.base().i == 3);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
