//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ABI tags have no effect in MSVC mode.
// XFAIL: msvc

// Assertion semantics are not supported in C++03 mode and currently are experimental.
// UNSUPPORTED: c++03, libcpp-has-no-experimental-hardening-observe-semantic

// Test that we encode the assertion semantic in an ABI tag to avoid ODR violations when linking TUs that have different
// values for it.

// RUN: %{cxx} %s %{flags} %{compile_flags} -c -DTU1  -U_LIBCPP_ASSERTION_SEMANTIC -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_IGNORE        -o %t.tu1.o
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -DTU2  -U_LIBCPP_ASSERTION_SEMANTIC -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_OBSERVE       -o %t.tu2.o
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -DTU3  -U_LIBCPP_ASSERTION_SEMANTIC -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_QUICK_ENFORCE -o %t.tu3.o
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -DTU4  -U_LIBCPP_ASSERTION_SEMANTIC -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_ENFORCE       -o %t.tu4.o
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -DMAIN                                                                                                    -o %t.main.o
// RUN: %{cxx} %t.tu1.o %t.tu2.o %t.tu3.o %t.tu4.o %t.main.o %{flags} %{link_flags} -o %t.exe
// RUN: %{exec} %t.exe

#include "test_macros.h"

// `ignore` assertion semantic.
#ifdef TU1
#  include <__config>
_LIBCPP_HIDE_FROM_ABI TEST_NOINLINE inline int f() { return 1; }
int tu1() { return f(); }
#endif // TU1

// `observe` assertion semantic.
#ifdef TU2
#  include <__config>
_LIBCPP_HIDE_FROM_ABI TEST_NOINLINE inline int f() { return 2; }
int tu2() { return f(); }
#endif // TU2

// `quick-enforce` assertion semantic.
#ifdef TU3
#  include <__config>
_LIBCPP_HIDE_FROM_ABI TEST_NOINLINE inline int f() { return 3; }
int tu3() { return f(); }
#endif // TU3

// `enforce` assertion semantic.
#ifdef TU4
#  include <__config>
_LIBCPP_HIDE_FROM_ABI TEST_NOINLINE inline int f() { return 4; }
int tu4() { return f(); }
#endif // TU4

#ifdef MAIN
#  include <cassert>

int tu1();
int tu2();
int tu3();
int tu4();

int main(int, char**) {
  assert(tu1() == 1);
  assert(tu2() == 2);
  assert(tu3() == 3);
  assert(tu4() == 4);
  return 0;
}
#endif // MAIN
