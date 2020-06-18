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

#include <optional>
#include <string>
#include <memory>
#include <cassert>

#include "test_macros.h"

struct TestT {
  int x;
  int size;
  constexpr TestT(std::initializer_list<int> il) : x(*il.begin()), size(static_cast<int>(il.size())) {}
  constexpr TestT(std::initializer_list<int> il, const int*)
      : x(*il.begin()), size(static_cast<int>(il.size())) {}
};

int main(int, char**)
{
    using std::make_optional;
    {
        constexpr auto opt = make_optional<TestT>({42, 2, 3});
        ASSERT_SAME_TYPE(decltype(opt), const std::optional<TestT>);
        static_assert(opt->x == 42, "");
        static_assert(opt->size == 3, "");
    }
    {
        constexpr auto opt = make_optional<TestT>({42, 2, 3}, nullptr);
        static_assert(opt->x == 42, "");
        static_assert(opt->size == 3, "");
    }
    {
        auto opt = make_optional<std::string>({'1', '2', '3'});
        assert(*opt == "123");
    }
    {
        auto opt = make_optional<std::string>({'a', 'b', 'c'}, std::allocator<char>{});
        assert(*opt == "abc");
    }

  return 0;
}
