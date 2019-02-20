// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <variant>

// template <class ...Types> class variant;


#include <variant>
#include <type_traits>
#include <string>
#include <cassert>

#include "test_macros.h"
#include "variant_test_helpers.hpp"
#include "test_convertible.hpp"

int main(int, char**)
{
    // expected-error@variant:* 3 {{static_assert failed}}
    std::variant<int, int[]> v; // expected-note {{requested here}}
    std::variant<int, int[42]> v2; // expected-note {{requested here}}
    std::variant<int, int[][42]> v3; // expected-note {{requested here}}

  return 0;
}
