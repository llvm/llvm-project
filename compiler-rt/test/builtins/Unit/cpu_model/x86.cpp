//===-- x86.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for the x86 cpuid builtins.
//
//===----------------------------------------------------------------------===//

#include "cpuid.h"
#include "gtest/gtest.h"

// TODO(boomanaiden154): This file currently only contains a single test to
// ensure that the build system components for this test work as expected. The
// set of tests needs to be expanded once the build system components are
// validated as working on the buildbots.

TEST(BuiltsinCPUModelTest, TestTrue) {
  OverrideCPUID(1, 0, 0, 4294967295);
  int SupportsCmov = __builtin_cpu_supports("cmov");
  ASSERT_TRUE(SupportsCmov);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
