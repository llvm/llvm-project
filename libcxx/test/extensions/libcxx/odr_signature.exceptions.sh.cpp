//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ABI tags have no effect in MSVC mode.
// XFAIL: msvc

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// Test that we encode whether exceptions are supported in an ABI tag to avoid
// ODR violations when linking TUs that have different values for it.

// RUN: %{cxx} %s %{flags} %{compile_flags} -c -DTU1  -fno-exceptions -o %t.tu1.o
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -DTU2  -fexceptions    -o %t.tu2.o
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -DMAIN                 -o %t.main.o
// RUN: %{cxx} %t.tu1.o %t.tu2.o %t.main.o %{flags} %{link_flags} -o %t.exe
// RUN: %{exec} %t.exe

#include "test_macros.h"

// -fno-exceptions
#ifdef TU1
#  include <__config>
_LIBCPP_HIDE_FROM_ABI TEST_NOINLINE inline int f() { return 1; }
int tu1() { return f(); }
#endif // TU1

// -fexceptions
#ifdef TU2
#  include <__config>
_LIBCPP_HIDE_FROM_ABI TEST_NOINLINE inline int f() { return 2; }
int tu2() { return f(); }
#endif // TU2

#ifdef MAIN
#  include <cassert>

int tu1();
int tu2();

int main(int, char**) {
  assert(tu1() == 1);
  assert(tu2() == 2);
  return 0;
}
#endif // MAIN
