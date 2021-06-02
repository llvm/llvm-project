//===-- scudo_unit_test_main.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memtag.h"
#include "tests/scudo_unit_test.h"

// Match Android's default configuration, which disables Scudo's mismatch
// allocation check, as it is being triggered by some third party code.
#if SCUDO_ANDROID
#define DEALLOC_TYPE_MISMATCH "false"
#else
#define DEALLOC_TYPE_MISMATCH "true"
#endif

// This allows us to turn on/off a Quarantine for specific tests. The Quarantine
// parameters are on the low end, to avoid having to loop excessively in some
// tests.
bool UseQuarantine = true;
extern "C" __attribute__((visibility("default"))) const char *
__scudo_default_options() {
  if (!UseQuarantine)
    return "dealloc_type_mismatch=" DEALLOC_TYPE_MISMATCH;
  return "quarantine_size_kb=256:thread_local_quarantine_size_kb=128:"
         "quarantine_max_chunk_size=512:"
         "dealloc_type_mismatch=" DEALLOC_TYPE_MISMATCH;
}

// The zxtest library provides a default main function that does the same thing
// for Fuchsia builds.
#if !SCUDO_FUCHSIA
int main(int argc, char **argv) {
  if (scudo::archSupportsMemoryTagging())
    scudo::enableSystemMemoryTaggingTestOnly();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
