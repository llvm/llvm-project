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

// path& operator=(path&&) noexcept

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "count_new.h"


int main(int, char**) {
  using namespace fs;
  static_assert(std::is_nothrow_move_assignable<path>::value, "");
  assert(globalMemCounter.checkOutstandingNewEq(0));
  const std::string s("we really really really really really really really "
                      "really really long string so that we allocate");
  assert(globalMemCounter.checkOutstandingNewEq(1));
  const fs::path::string_type ps(s.begin(), s.end());
  path p(s);
  {
    DisableAllocationGuard g;
    path p2;
    path& pref = (p2 = std::move(p));
    assert(p2.native() == ps);
    assert(p.native() != ps); // Testing moved from state
    assert(&pref == &p2);
  }

  return 0;
}
