// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that _LIBCPP_NODISCARD_AFTER_CXX17 is disabled whenever
// _LIBCPP_DISABLE_NODISCARD_AFTER_CXX17 is defined by the user.

// MODULES_DEFINES: _LIBCPP_DISABLE_NODISCARD_AFTER_CXX17
#define _LIBCPP_DISABLE_NODISCARD_AFTER_CXX17
#include <__config>

#include "test_macros.h"

_LIBCPP_NODISCARD_AFTER_CXX17 int foo() { return 6; }

int main(int, char**)
{
	foo();	// no error here!

  return 0;
}
