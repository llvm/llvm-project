//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// template <class T, class U, class... Args>
//   constexpr optional<T> make_optional(initializer_list<U> il, Args&&... args);

#include <cassert>
#include <memory>
#include <optional>
#include <string>

#include "test_macros.h"

struct TestT {
  int x;
  int size;
  int *ptr;
  constexpr TestT(std::initializer_list<int> il)
    : x(*il.begin()), size(static_cast<int>(il.size())), ptr(nullptr) {}
  constexpr TestT(std::initializer_list<int> il, int *p)
    : x(*il.begin()), size(static_cast<int>(il.size())), ptr(p) {}
};

constexpr bool test()
{
  {
    auto opt = std::make_optional<TestT>({42, 2, 3});
    ASSERT_SAME_TYPE(decltype(opt), std::optional<TestT>);
    assert(opt->x == 42);
    assert(opt->size == 3);
    assert(opt->ptr == nullptr);
  }
  {
    int i = 42;
    auto opt = std::make_optional<TestT>({42, 2, 3}, &i);
    ASSERT_SAME_TYPE(decltype(opt), std::optional<TestT>);
    assert(opt->x == 42);
    assert(opt->size == 3);
    assert(opt->ptr == &i);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  {
    auto opt = std::make_optional<std::string>({'1', '2', '3'});
    assert(*opt == "123");
  }
  {
    auto opt = std::make_optional<std::string>({'a', 'b', 'c'}, std::allocator<char>{});
    assert(*opt == "abc");
  }
  return 0;
}
