//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test checks that we retain extern template instantiation declarations
// for members of <locale> even when the debug mode is enabled, which is
// necessary for correctness. See https://llvm.org/D94718 for details.

// UNSUPPORTED: !libcpp-has-debug-mode
// UNSUPPORTED: no-localization
// UNSUPPORTED: cant-build-shared-library

// This test relies on linking a shared library and then passing that shared
// library as input when linking an executable; this is generally not supported
// on Windows (GNU ld supports it, but MS link.exe and LLD don't) - one has to
// use an import library instead. (Additionally, the test uses the -fPIC
// option which clang doesn't accept on Windows.)
// UNSUPPORTED: windows

// RUN: %{cxx} %{flags} %{compile_flags} %s %{link_flags} -fPIC -DTU1 -fvisibility=hidden -shared -o %t.lib
// RUN: cd %T && %{cxx} %{flags} %{compile_flags} %s ./%basename_t.tmp.lib %{link_flags} -DTU2 -fvisibility=hidden -o %t.exe
// RUN: %{exec} %t.exe

#include <cassert>
#include <cstdio>
#include <sstream>
#include <string>

std::string __attribute__((visibility("default"))) numToString(int x);

#if defined(TU1)

std::string numToString(int x) {
  std::stringstream ss;
  ss << x;
  return ss.str();
}

#elif defined(TU2)

int main(int, char**) {
  std::string s = numToString(42);
  assert(s == "42");

  return 0;
}

#endif
