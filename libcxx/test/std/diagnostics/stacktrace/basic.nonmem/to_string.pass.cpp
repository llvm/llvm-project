//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: no-localization
// XFAIL: availability-stacktrace-missing

/*
  (19.6.4.6) Non-member functions

  string to_string(const stacktrace_entry& f);

  template<class Allocator>
    string to_string(const basic_stacktrace<Allocator>& st);
*/

#include <__config_site>
#if _LIBCPP_HAS_LOCALIZATION

#  include <cassert>
#  include <iostream>
#  include <stacktrace>

int main(int, char**) {
  auto a    = std::stacktrace::current();
  auto astr = std::to_string(a);
  std::cerr << astr << '\n';

  // assert(std::to_string(a[0]).contains("main"));
  assert(std::to_string(a[0]).contains("to_string.pass"));

  // assert(std::to_string(a).contains("main"));
  assert(std::to_string(a).contains("to_string.pass"));

  return 0;
}

#else
int main() { return 0; }
#endif // _LIBCPP_HAS_LOCALIZATION
