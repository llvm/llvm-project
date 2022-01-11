//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{cxx} %{flags} %{compile_flags} -c %s -o %t.tu1.o -DTU1
// RUN: %{cxx} %{flags} %{compile_flags} -c %s -o %t.tu2.o -DTU2
// RUN: %{cxx} %t.tu1.o %t.tu2.o %{flags} %{link_flags} -o %t.exe
// RUN: %{exec} %t.exe

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// Test the libc++-specific behavior that we handle the IFNDR case for ranges::begin
// by returning the beginning of the array-of-incomplete-type.
// Use two translation units so that `Incomplete` really is never completed
// at any point within TU2, but the array `bounded` is still given a definition
// (in TU1) to avoid an "undefined reference" error from the linker.
// All of the actually interesting stuff takes place within TU2.

#include <ranges>
#include <cassert>

#include "test_macros.h"

#if defined(TU1)

struct Incomplete {};
Incomplete bounded[10];
Incomplete unbounded[10];

#else // defined(TU1)

struct Incomplete;

constexpr bool test()
{
    {
    extern Incomplete bounded[10];
    assert(std::ranges::begin(bounded) == bounded);
    assert(std::ranges::cbegin(bounded) == bounded);
    assert(std::ranges::begin(std::as_const(bounded)) == bounded);
    assert(std::ranges::cbegin(std::as_const(bounded)) == bounded);
    ASSERT_SAME_TYPE(decltype(std::ranges::begin(bounded)), Incomplete*);
    ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(bounded)), const Incomplete*);
    ASSERT_SAME_TYPE(decltype(std::ranges::begin(std::as_const(bounded))), const Incomplete*);
    ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(std::as_const(bounded))), const Incomplete*);
    }
    {
    extern Incomplete unbounded[];
    assert(std::ranges::begin(unbounded) == unbounded);
    assert(std::ranges::cbegin(unbounded) == unbounded);
    assert(std::ranges::begin(std::as_const(unbounded)) == unbounded);
    assert(std::ranges::cbegin(std::as_const(unbounded)) == unbounded);
    ASSERT_SAME_TYPE(decltype(std::ranges::begin(unbounded)), Incomplete*);
    ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(unbounded)), const Incomplete*);
    ASSERT_SAME_TYPE(decltype(std::ranges::begin(std::as_const(unbounded))), const Incomplete*);
    ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(std::as_const(unbounded))), const Incomplete*);
    }

    return true;
}

int main(int, char**)
{
    test();
    static_assert(test());
    return 0;
}

#endif // defined(TU1)
