//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <utility>

// template <class T1, class T2> struct pair

// constexpr pair();

#include <utility>
#include <type_traits>

#include "test_macros.h"


struct ThrowingDefault {
  ThrowingDefault() { }
};

struct NonThrowingDefault {
  NonThrowingDefault() noexcept { }
};

int main(int, char**) {

    static_assert(!std::is_nothrow_default_constructible<std::pair<ThrowingDefault, ThrowingDefault>>::value, "");
    static_assert(!std::is_nothrow_default_constructible<std::pair<NonThrowingDefault, ThrowingDefault>>::value, "");
    static_assert(!std::is_nothrow_default_constructible<std::pair<ThrowingDefault, NonThrowingDefault>>::value, "");
    static_assert( std::is_nothrow_default_constructible<std::pair<NonThrowingDefault, NonThrowingDefault>>::value, "");

  return 0;
}
