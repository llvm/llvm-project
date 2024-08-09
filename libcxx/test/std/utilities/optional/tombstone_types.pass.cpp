//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

#include <cassert>
#include <optional>
#include <string>

constexpr bool test() {
  {
    std::optional<bool> opt;
    assert(!opt);
    opt = true;
    assert(opt);
    assert(*opt);
    opt = false;
    assert(opt);
    assert(!*opt);
  }

  {
    std::optional<std::string> opt;
    assert(!opt);
    opt = "";
    assert(opt);
    assert(*opt == "");
    opt = "23 letter string to SSO";
    assert(opt);
    assert(*opt == "23 letter string to SSO");
  }

  {
    std::optional<std::string> opt;
    assert(!opt);
    opt = "";
    assert(opt);
    assert(*opt == "");
    opt = "23 letter string to SSO";
    assert(opt);
    assert(*opt == "23 letter string to SSO");
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
}
