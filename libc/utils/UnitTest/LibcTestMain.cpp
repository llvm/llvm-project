//===-- Main function for implementation of base class for libc unittests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcTest.h"

static const char *getTestFilter(int argc, char *argv[]) {
  return argc > 1 ? argv[1] : nullptr;
}

int main(int argc, char *argv[]) {
  const char *TestFilter = getTestFilter(argc, argv);
  return __llvm_libc::testing::Test::runTests(TestFilter);
}
