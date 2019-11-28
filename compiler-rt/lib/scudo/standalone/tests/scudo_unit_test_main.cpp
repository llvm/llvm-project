//===-- scudo_unit_test_main.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

// This allows us to turn on/off a Quarantine for specific tests. The Quarantine
// parameters are on the low end, to avoid having to loop excessively in some
// tests.
bool UseQuarantine = true;
extern "C" __attribute__((visibility("default"))) const char *
__scudo_default_options() {
  if (!UseQuarantine)
    return "dealloc_type_mismatch=true";
  return "quarantine_size_kb=256:thread_local_quarantine_size_kb=128:"
         "quarantine_max_chunk_size=512:dealloc_type_mismatch=true";
}

int main(int argc, char **argv) {
#if !SCUDO_FUCHSIA
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#else
  return RUN_ALL_TESTS(argc, argv);
#endif
}
