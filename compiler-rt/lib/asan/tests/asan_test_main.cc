//===-- asan_test_main.cc -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
//===----------------------------------------------------------------------===//
#include "asan_test_utils.h"
#include "sanitizer_common/sanitizer_platform.h"

// Default ASAN_OPTIONS for the unit tests. Let's turn symbolication off to
// speed up testing (unit tests don't use it anyway).
extern "C" const char* __asan_default_options() {
#if SANITIZER_MAC
  // On Darwin, we default to `abort_on_error=1`, which would make tests run
  // much slower. Let's override this and run lit tests with 'abort_on_error=0'.
  // Also, make sure we do not overwhelm the syslog while testing.
  return "symbolize=false:abort_on_error=0:log_to_syslog=0";
#else
  return "symbolize=false";
#endif
}

int main(int argc, char **argv) {
  testing::GTEST_FLAG(death_test_style) = "threadsafe";
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
