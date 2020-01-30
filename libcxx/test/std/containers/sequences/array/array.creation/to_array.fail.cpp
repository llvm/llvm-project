//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

#include <array>

#include "test_macros.h"
#include "MoveOnly.h"

int main(int, char**) {
  {
    char source[3][6] = {"hi", "world"};
    std::to_array(source); // expected-error {{here}}
  }

  {
    MoveOnly mo[] = {MoveOnly{3}};
    std::to_array(mo); // expected-error {{here}}
  }

  {
    const MoveOnly cmo[] = {MoveOnly{3}};
    std::to_array(std::move(cmo)); // expected-error {{here}}
  }

  return 0;
}
