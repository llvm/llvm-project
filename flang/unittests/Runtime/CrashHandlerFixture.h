//===-- flang/unittests/RuntimeGTest/CrashHandlerFixture.h ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Test fixture registers a custom crash handler to ensure death tests fail
/// with expected message.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_FLANG_UNITTESTS_RUNTIMEGTEST_CRASHHANDLERFIXTURE_H
#define LLVM_FLANG_UNITTESTS_RUNTIMEGTEST_CRASHHANDLERFIXTURE_H
#include <gtest/gtest.h>

struct CrashHandlerFixture : testing::Test {
  void SetUp();
};

#endif
