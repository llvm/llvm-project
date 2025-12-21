//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <filesystem>

// path operator/(path const&, path const&);

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
namespace fs = std::filesystem;

// This is mainly tested via the member append functions.
int main(int, char**) {
  using namespace fs;
  path p1("abc");
  path p2("def");
  path p3 = p1 / p2;
  assert(p3 == "abc/def");

  path p4 = p1 / "def";
  assert(p4 == "abc/def");

  return 0;
}
