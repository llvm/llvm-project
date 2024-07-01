//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <contracts>

// TODO

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_CONTRACTS_EVALUATION_SEMANTIC_IGNORE

#include <contracts>

#include "check_assertion.h"

int main(int, char**) {
  __pre__(false, "some message");             // nothing happening
  __post__(false, "some message");            // nothing happening
  __contract_assert__(false, "some message"); // nothing happening

  return 0;
}
