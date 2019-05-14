//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <experimental/type_traits>
//
//  struct nonesuch

#include <experimental/type_traits>
#include <string>

#include "test_macros.h"

namespace ex = std::experimental;

struct such {};
void foo(const such &) {}
void foo(const ex::nonesuch &) {}

int main(int, char**) {
    foo({});  // nonesuch is not an aggregate

    return 0;
}
