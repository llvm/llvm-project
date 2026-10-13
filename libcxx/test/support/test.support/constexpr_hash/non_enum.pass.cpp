//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// "support/constexpr_hash.h"

//  Hashing a struct w/o a defined hash should *not* fail, but it should
// create a type that is not constructible and not callable.
// See also: https://cplusplus.github.io/LWG/lwg-defects.html#2543

#include "constexpr_hash.h"
#include <cassert>
#include <type_traits>

#include "test_macros.h"

struct X {};

TEST_CONSTEXPR_CXX26 bool test_with_hash() {
  using H = support::constexpr_hash<X>;

  static_assert(!std::is_default_constructible<H>::value, "");
  static_assert(!std::is_copy_constructible<H>::value, "");
  static_assert(!std::is_move_constructible<H>::value, "");
  static_assert(!std::is_copy_assignable<H>::value, "");
  static_assert(!std::is_move_assignable<H>::value, "");
#if TEST_STD_VER > 14
  static_assert(!std::is_invocable<H, X&>::value, "");
  static_assert(!std::is_invocable<H, X const&>::value, "");
#endif

  return true;
}

int main(int, char**) {
  assert(test_with_hash());

#if TEST_STD_VER >= 26
  static_assert(test_with_hash());
#endif

  return 0;
}
