//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1
// UNSUPPORTED: libcxx-no-debug-mode

#include <cstdlib>
#include <string>
#include <type_traits>
#include <__debug>
#include <cassert>

#include "test_macros.h"

void my_debug_function(std::__libcpp_debug_info const& info) {
  assert(info.__msg_ == std::string("foo"));
  std::exit(0);
}

int main(int, char**)
{
  std::__libcpp_set_debug_function(&my_debug_function);
  _LIBCPP_ASSERT(false, "foo");
  return 1;
}
