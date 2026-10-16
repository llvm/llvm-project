//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <type_traits>

// __libcpp_is_constant_evaluated()

// returns false when there's no constant evaluation support from the compiler.
//  as well as when called not in a constexpr context

#include <__cxx03/__type_traits/is_constant_evaluated.h>
#include <cassert>

#include "test_macros.h"

int main (int, char**) {
    ASSERT_SAME_TYPE(decltype(std::__libcpp_is_constant_evaluated()), bool);
    ASSERT_NOEXCEPT(std::__libcpp_is_constant_evaluated());

    bool p = std::__libcpp_is_constant_evaluated();
    assert(!p);

    return 0;
    }
