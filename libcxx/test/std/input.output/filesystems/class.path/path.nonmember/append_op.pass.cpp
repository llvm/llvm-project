//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <filesystem>

// path operator/(path const&, path const&);

#include "filesystem_include.hpp"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.hpp"

// This is mainly tested via the member append functions.
int main()
{
  using namespace fs;
  path p1("abc");
  path p2("def");
  path p3 = p1 / p2;
  assert(p3 == "abc/def");

  path p4 = p1 / "def";
  assert(p4 == "abc/def");
}
