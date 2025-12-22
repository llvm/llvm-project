//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The test suite needs to define the ABI macros on the command line when
// modules are enabled.
// UNSUPPORTED: clang-modules-build

// <utility>

// template <class T1, class T2> struct pair

// Test that we provide the non-trivial copy operations when _LIBCPP_DEPRECATED_ABI_DISABLE_PAIR_TRIVIAL_COPY_CTOR
// is specified.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEPRECATED_ABI_DISABLE_PAIR_TRIVIAL_COPY_CTOR
// ADDITIONAL_COMPILE_FLAGS: -Wno-macro-redefined -Wno-invalid-offsetof

#include <utility>
#include <type_traits>
#include <cstdlib>
#include <cstddef>
#include <cassert>

#include "test_macros.h"

template <class T>
struct HasNonTrivialABI : std::integral_constant<bool,
    !std::is_trivially_destructible<T>::value
    || (std::is_copy_constructible<T>::value && !std::is_trivially_copy_constructible<T>::value)
> {};


void test_trivial()
{
    {
        typedef std::pair<int, short> P;
        static_assert(std::is_copy_constructible<P>::value, "");
        static_assert(HasNonTrivialABI<P>::value, "");
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
