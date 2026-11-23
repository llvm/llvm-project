//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// REQUIRES: can-test-hardening-assertions-extensive

// Ensure that passing std::functions across -fno-rtti/-frtti boundaries is asserted on.

#include <cassert>
#include <functional>
#include <typeinfo>

// RUN: %{cxx} %s %{flags} %{compile_flags} -c -frtti -o %t.tu1.o
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -DNO_RTTI -fno-rtti -o %t.tu2.o
// RUN: %{cxx} %t.tu1.o %t.tu2.o %{flags} %{link_flags} -o %t.exe
// RUN: %{exec} %t.exe

std::function<void()> get_func();

#ifdef NO_RTTI
std::function<void()> get_func() { return get_func; }
#else

// This can only be included once.
#  include "check_assertion.h"

int main(int, char**) {
  TEST_LIBCPP_ASSERT_FAILURE(
      get_func().target<int>(), "Trying to access type_info of std::function created in -fno-rtti mode!");
  TEST_LIBCPP_ASSERT_FAILURE(
      get_func().target_type(), "Trying to access type_info of std::function created in -fno-rtti mode!");

  return 0;
}
#endif
