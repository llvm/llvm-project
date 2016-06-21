//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/filesystem>

// path operator/(path const&, path const&);

#include <experimental/filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.hpp"

namespace fs = std::experimental::filesystem;

// This is mainly tested via the member append functions.
int main()
{
  using namespace fs;
  path p1("abc");
  path p2("def");
  path p3 = p1 / p2;
  assert(p3 == "abc/def");
}
