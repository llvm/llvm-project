//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// Test that we properly provide the trivial copy operations by default.

// FreeBSD still provides the old ABI for std::pair.
// XFAIL: freebsd
// ADDITIONAL_COMPILE_FLAGS: -Wno-invalid-offsetof

#include <utility>
#include <type_traits>
#include <cstdlib>
#include <cstddef>
#include <cassert>

#include "test_macros.h"

template <class T>
struct HasTrivialABI : std::integral_constant<bool,
    std::is_trivially_destructible<T>::value
    && (!std::is_copy_constructible<T>::value || std::is_trivially_copy_constructible<T>::value)
> {};

struct TrivialNoAssignment {
  int arr[4];
  TrivialNoAssignment& operator=(const TrivialNoAssignment&) = delete;
};

struct TrivialNoConstruction {
  int arr[4];
  TrivialNoConstruction()                                        = default;
  TrivialNoConstruction(const TrivialNoConstruction&)            = delete;
  TrivialNoConstruction& operator=(const TrivialNoConstruction&) = default;
};

void test_trivial()
{
    {
        typedef std::pair<int, short> P;
        static_assert(std::is_copy_constructible<P>::value, "");
        static_assert(HasTrivialABI<P>::value, "");
    }
    {
        using P = std::pair<TrivialNoAssignment, int>;
        static_assert(std::is_trivially_copy_constructible<P>::value, "");
        static_assert(std::is_trivially_move_constructible<P>::value, "");
        static_assert(std::is_trivially_destructible<P>::value, "");
    }
    {
        using P = std::pair<TrivialNoConstruction, int>;
        static_assert(!std::is_trivially_copy_assignable<P>::value, "");
        static_assert(!std::is_trivially_move_assignable<P>::value, "");
        static_assert(std::is_trivially_destructible<P>::value, "");
    }
}

void test_layout() {
    typedef std::pair<std::pair<char, char>, char> PairT;
    static_assert(sizeof(PairT) == 3, "");
    static_assert(TEST_ALIGNOF(PairT) == TEST_ALIGNOF(char), "");
    static_assert(offsetof(PairT, first) == 0, "");
}

int main(int, char**) {
    test_trivial();
    test_layout();
    return 0;
}
