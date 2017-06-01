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

namespace __sanitizer {
bool ReexecDisabled() {
#if __has_feature(address_sanitizer) && SANITIZER_MAC
  // Allow re-exec in instrumented unit tests on Darwin.  Technically, we only
  // need this for 10.10 and below, where re-exec is required for the
  // interceptors to work, but to avoid duplicating the version detection logic,
  // let's just allow re-exec for all Darwin versions.  On newer OS versions,
  // returning 'false' doesn't do anything anyway, because we don't re-exec.
  return false;
#else
  return true;
#endif
}
}  // namespace __sanitizer

int main(int argc, char **argv) {
  testing::GTEST_FLAG(death_test_style) = "threadsafe";
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
