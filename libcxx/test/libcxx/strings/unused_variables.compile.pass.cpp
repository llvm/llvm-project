//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that we don't introduce new warnings on unused variables of libc++ classes if
// _LIBCPP_DISABLE_UNUSED_STRUCT_WARNINGS is set.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_UNUSED_STRUCT_WARNINGS

#include <string>

#include "test_macros.h"

void strings() {
  std::string l;

#if TEST_STD_VER <= 11
  std::string::iterator l;
  std::string::const_iterator l;
#endif
}
