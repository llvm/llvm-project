// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions
// MODULES_DEFINES: _LIBCPP_DEBUG=0

// Can't test the system lib because this test enables debug mode
// UNSUPPORTED: with_system_cxx_lib

// Test that the default debug handler can be overridden and test the
// throwing debug handler.

#define _LIBCPP_DEBUG 0

#include <cstdlib>
#include <exception>
#include <type_traits>
#include <__debug>

int main()
{
  {
    std::__libcpp_debug_function = std::__libcpp_throw_debug_function;
    try {
      _LIBCPP_ASSERT(false, "foo");
    } catch (std::__libcpp_debug_exception const&) {}
  }
  {
    // test that the libc++ exception type derives from std::exception
    static_assert((std::is_base_of<std::exception,
                                  std::__libcpp_debug_exception
                  >::value), "must be an exception");
  }
}
