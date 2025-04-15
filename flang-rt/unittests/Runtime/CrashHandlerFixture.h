//===-- unittests/Runtime/CrashHandlerFixture.h -----------------*- C++ -*-===//
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

#ifndef FLANG_RT_UNITTESTS_RUNTIME_CRASHHANDLERFIXTURE_H_
#define FLANG_RT_UNITTESTS_RUNTIME_CRASHHANDLERFIXTURE_H_
#include <gtest/gtest.h>

struct CrashHandlerFixture : testing::Test {
  void SetUp();
};

#endif /* FLANG_RT_UNITTESTS_RUNTIME_CRASHHANDLERFIXTURE_H_ */
