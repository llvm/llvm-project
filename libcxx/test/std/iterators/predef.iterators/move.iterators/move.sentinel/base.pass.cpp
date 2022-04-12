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

// constexpr S base() const;

#include <iterator>
#include <cassert>

#include "test_macros.h"

constexpr bool test()
{
  {
    auto m = std::move_sentinel<int>(42);
    const auto& cm = m;
    assert(m.base() == 42);
    assert(cm.base() == 42);
    assert(std::move(m).base() == 42);
    assert(std::move(cm).base() == 42);
    ASSERT_SAME_TYPE(decltype(m.base()), int);
    ASSERT_SAME_TYPE(decltype(cm.base()), int);
    ASSERT_SAME_TYPE(decltype(std::move(m).base()), int);
    ASSERT_SAME_TYPE(decltype(std::move(cm).base()), int);
  }
  {
    int a[] = {1, 2, 3};
    auto m = std::move_sentinel<const int*>(a);
    const auto& cm = m;
    assert(m.base() == a);
    assert(cm.base() == a);
    assert(std::move(m).base() == a);
    assert(std::move(cm).base() == a);
    ASSERT_SAME_TYPE(decltype(m.base()), const int*);
    ASSERT_SAME_TYPE(decltype(cm.base()), const int*);
    ASSERT_SAME_TYPE(decltype(std::move(m).base()), const int*);
    ASSERT_SAME_TYPE(decltype(std::move(cm).base()), const int*);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
