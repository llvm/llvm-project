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
  using hasFoo = typename T::Foo;

struct yesFoo {
    using Foo = int;
};

struct noFoo {
};


template <typename T, typename Res>
void test() {
    static_assert( std::is_same<Res, typename ex::detected_or  <double, hasFoo, T>::type>::value, "" );
    static_assert( std::is_same<Res, typename ex::detected_or_t<double, hasFoo, T>      >::value, "" );
}

int main(int, char**) {
    test<yesFoo, int>();
    test<noFoo, double>();

  return 0;
}
