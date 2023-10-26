//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: Remove these UNSUPPORTED lines once we change how hardening is enabled to avoid
//       mutually exclusive modes being enabled at the same time.
// UNSUPPORTED: libcpp-has-hardened-mode
// UNSUPPORTED: libcpp-has-debug-mode
// UNSUPPORTED: libcpp-has-assertions

// TODO: Investigate
// XFAIL: msvc

// Test that we encode the hardening mode in an ABI tag to avoid ODR violations
// when linking TUs that have different values for it.

// RUN: %{cxx} %s %{flags} %{compile_flags} -c -DTU1  -D_LIBCPP_ENABLE_HARDENED_MODE -o %t.tu1.o
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -DTU2  -D_LIBCPP_ENABLE_ASSERTIONS    -o %t.tu2.o
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -DTU3  -D_LIBCPP_ENABLE_DEBUG_MODE    -o %t.tu3.o
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -DTU4                                 -o %t.tu4.o
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -DMAIN                                -o %t.main.o
// RUN: %{cxx} %t.tu1.o %t.tu2.o %t.tu3.o %t.tu4.o %t.main.o %{flags} %{link_flags} -o %t.exe
// RUN: %{exec} %t.exe

// hardened mode
#ifdef TU1
#  include <__config>
_LIBCPP_HIDE_FROM_ABI inline int f() { return 1; }
int tu1() { return f(); }
#endif // TU1

// safe mode
#ifdef TU2
#  include <__config>
_LIBCPP_HIDE_FROM_ABI inline int f() { return 2; }
int tu2() { return f(); }
#endif // TU2

// debug mode
#ifdef TU3
#  include <__config>
_LIBCPP_HIDE_FROM_ABI inline int f() { return 3; }
int tu3() { return f(); }
#endif // TU3

// unchecked mode
#ifdef TU4
#  include <__config>
_LIBCPP_HIDE_FROM_ABI inline int f() { return 4; }
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
