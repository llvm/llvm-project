//===-- sanitizer_test_main.cc --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//
#include "gtest/gtest.h"
#include "sanitizer_common/sanitizer_flags.h"

const char *argv0;

int main(int argc, char **argv) {
  argv0 = argv[0];
  testing::GTEST_FLAG(death_test_style) = "threadsafe";
  testing::InitGoogleTest(&argc, argv);
  SetCommonFlagsDefaults();
  return RUN_ALL_TESTS();
}
