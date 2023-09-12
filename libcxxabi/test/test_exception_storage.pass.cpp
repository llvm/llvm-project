//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include "assert_macros.h"
#include "concat_macros.h"
#include "../src/cxa_exception.h"

int main(int, char**) {
  void* globals = __cxxabiv1::__cxa_get_globals();
  TEST_REQUIRE(globals != nullptr, TEST_WRITE_CONCATENATED("Got null result from __cxa_get_globals"));

  void* fast_globals = __cxxabiv1::__cxa_get_globals_fast();
  TEST_REQUIRE(globals == fast_globals, TEST_WRITE_CONCATENATED("__cxa_get_globals returned ", globals,
                                                                " but __cxa_get_globals_fast returned ", fast_globals));

  return 0;
}
