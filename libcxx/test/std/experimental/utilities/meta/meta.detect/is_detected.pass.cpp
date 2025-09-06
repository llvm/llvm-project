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
#include <utility>

#include "test_macros.h"

namespace ex = std::experimental;

template <typename T>
  using copy_assign_t = decltype(std::declval<T&>() = std::declval<T const &>());

struct not_assignable {
    not_assignable & operator=(const not_assignable&) = delete;
};

template <typename T, bool b>
void test() {
    static_assert( b == ex::is_detected  <copy_assign_t, T>::value, "" );
    static_assert( b == ex::is_detected_v<copy_assign_t, T>, "" );
}

int main(int, char**) {
    test<int, true>();
    test<std::string, true>();
    test<not_assignable, false>();

  return 0;
}
