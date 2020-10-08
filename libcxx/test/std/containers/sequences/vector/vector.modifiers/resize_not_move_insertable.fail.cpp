//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// GCC 5 does not evaluate static assertions dependent on a template parameter.
// UNSUPPORTED: gcc-5

// <vector>

// Test that vector produces a decent diagnostic for user types that explicitly
// delete their move constructor. Such types don't meet the Cpp17CopyInsertible
// requirements.

#include <vector>

template <int>
class BadUserNoCookie {
public:
  BadUserNoCookie() { }

  BadUserNoCookie(BadUserNoCookie&&) = delete;
  BadUserNoCookie& operator=(BadUserNoCookie&&) = delete;

  BadUserNoCookie(const BadUserNoCookie&) = default;
  BadUserNoCookie& operator=(const BadUserNoCookie&) = default;
};

int main(int, char**) {
  // expected-error@memory:* 2 {{"The specified type does not meet the requirements of Cpp17MoveInsertable"}}

  // Other diagnostics that might be seen as Clang tries to continue compiling:
  // expected-error@memory:* 0-2 {{call to deleted constructor}}
  // expected-error@memory:* 0-2 {{no matching function for call to 'construct_at'}}
  {

    std::vector<BadUserNoCookie<1> > x;
    x.emplace_back();
  }
  {
    std::vector<BadUserNoCookie<2>> x;
    BadUserNoCookie<2> c;
    x.push_back(c);
  }
  return 0;
}
