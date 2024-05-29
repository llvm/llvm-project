//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// UNSUPPORTED: no-localization

// check that <regex> functions are marked [[nodiscard]]

#include <regex>

void test() {
  std::cmatch match_result;
  match_result.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
