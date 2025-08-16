//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -g

/*
  (19.6.4.6) Non-member functions

  ostream& operator<<(ostream& os, const stacktrace_entry& f);
  template<class Allocator>
    ostream& operator<<(ostream& os, const basic_stacktrace<Allocator>& st);
*/

#include <cassert>
#include <sstream>
#include <stacktrace>

int main(int, char**) {
  auto a = std::stacktrace::current();

  std::stringstream entry_os;
  entry_os << a[0];
  assert(entry_os.str() == std::to_string(a[0]));

  std::stringstream trace_os;
  trace_os << a;
  assert(trace_os.str() == std::to_string(a));

  return 0;
}
