//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <optional>
// UNSUPPORTED: c++03, c++11, c++14

// template<class T>
//   optional(T) -> optional<T>;

#include <optional>
#include <cassert>

#include "test_macros.h"

struct A {};

int main(int, char**)
{
//  Test the explicit deduction guides
    {
//  optional(T)
    std::optional opt(5);
    ASSERT_SAME_TYPE(decltype(opt), std::optional<int>);
    assert(static_cast<bool>(opt));
    assert(*opt == 5);
    }

    {
//  optional(T)
    std::optional opt(A{});
    ASSERT_SAME_TYPE(decltype(opt), std::optional<A>);
    assert(static_cast<bool>(opt));
    }

    {
//  optional(const T&);
    const int& source = 5;
    std::optional opt(source);
    ASSERT_SAME_TYPE(decltype(opt), std::optional<int>);
    assert(static_cast<bool>(opt));
    assert(*opt == 5);
    }

    {
//  optional(T*);
    const int* source = nullptr;
    std::optional opt(source);
    ASSERT_SAME_TYPE(decltype(opt), std::optional<const int*>);
    assert(static_cast<bool>(opt));
    assert(*opt == nullptr);
    }

    {
//  optional(T[]);
    int source[] = {1, 2, 3};
    std::optional opt(source);
    ASSERT_SAME_TYPE(decltype(opt), std::optional<int*>);
    assert(static_cast<bool>(opt));
    assert((*opt)[0] == 1);
    }

//  Test the implicit deduction guides
    {
//  optional(optional);
    std::optional<char> source('A');
    std::optional opt(source);
    ASSERT_SAME_TYPE(decltype(opt), std::optional<char>);
    assert(static_cast<bool>(opt) == static_cast<bool>(source));
    assert(*opt == *source);
    }

  return 0;
}
