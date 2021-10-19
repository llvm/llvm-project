//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class path

// path& operator=(path const&);

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"


int main(int, char**) {
  using namespace fs;
  static_assert(std::is_copy_assignable<path>::value, "");
  static_assert(!std::is_nothrow_copy_assignable<path>::value, "should not be noexcept");
  const std::string s("foo");
  const path p(s);
  path p2;
  path& pref = (p2 = p);
  assert(p.string() == s);
  assert(p2.string() == s);
  assert(&pref == &p2);

  return 0;
}
