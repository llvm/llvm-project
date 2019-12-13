//===------------------------- fuzz_test.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fuzzing/fuzzing.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

#ifndef TEST_FUNCTION
#error TEST_FUNCTION must be defined
#endif

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  int result = fuzzing::TEST_FUNCTION(data, size);
  assert(result == 0); return 0;
}
