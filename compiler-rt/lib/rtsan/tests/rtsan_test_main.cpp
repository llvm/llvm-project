//===--- rtsan_test_main.cpp - Realtime Sanitizer ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#include "sanitizer_test_utils.h"

// Default RTSAN_OPTIONS for the unit tests.
extern "C" const char *__rtsan_default_options() {
#if SANITIZER_APPLE
  // On Darwin, we default to `abort_on_error=1`, which would make tests run
  // much slower. Let's override this and run lit tests with 'abort_on_error=0'
  // and make sure we do not overwhelm the syslog while testing. Also, let's
  // turn symbolization off to speed up testing, especially when not running
  // with llvm-symbolizer but with atos.
  return "symbolize=false:"
         "abort_on_error=0:"
         "log_to_syslog=0:"
         "verify_interceptors=0:"; // some of our tests don't need interceptors
#else
  // Let's turn symbolization off to speed up testing (more than 3 times speedup
  // observed).
  return "symbolize=false";
#endif
}

int main(int argc, char **argv) {
  testing::GTEST_FLAG(death_test_style) = "threadsafe";
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
