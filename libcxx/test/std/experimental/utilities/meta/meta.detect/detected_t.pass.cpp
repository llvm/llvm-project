//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <experimental/type_traits>

#include <experimental/type_traits>
#include <string>

#include "test_macros.h"

namespace ex = std::experimental;

template <typename T>
  using callFoo = decltype(std::declval<T&>().Foo());

struct yesFoo {
    int Foo() { return 0; }
};

struct noFoo {
};

struct wrongFoo {
    std::string Foo() { return ""; }
};

struct convertibleFoo {
    long Foo() { return 0; }
};


template <typename T, typename Res>
void test() {
    static_assert( std::is_same<Res, typename ex::detected_t<callFoo, T>>::value, "" );
}

int main(int, char**) {
    test<yesFoo, int>();
    test<noFoo, ex::nonesuch>();  // lookup failure returns nonesuch
    test<wrongFoo, std::string>();

  return 0;
}
