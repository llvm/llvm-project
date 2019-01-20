// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions
// MODULES_DEFINES: _LIBCPP_DEBUG=1
// MODULES_DEFINES: _LIBCPP_DEBUG_USE_EXCEPTIONS

// Can't test the system lib because this test enables debug mode
// UNSUPPORTED: with_system_cxx_lib

// Test that defining _LIBCPP_DEBUG_USE_EXCEPTIONS causes _LIBCPP_ASSERT
// to throw on failure.

#define _LIBCPP_DEBUG 1
#define _LIBCPP_DEBUG_USE_EXCEPTIONS

#include <cstdlib>
#include <exception>
#include <type_traits>
#include <__debug>
#include <cassert>

int main()
{
  try {
    _LIBCPP_ASSERT(false, "foo");
    assert(false);
  } catch (...) {}
}
