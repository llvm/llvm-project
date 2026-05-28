//===-- Unittests for scope_exit ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/utility/move.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::scope_exit;

TEST(LlvmLibcScopeExitTest, Basic) {
  bool called = false;
  {
    scope_exit cleanup([&called] { called = true; });
    ASSERT_FALSE(called);
  }
  ASSERT_TRUE(called);
}

TEST(LlvmLibcScopeExitTest, Release) {
  bool called = false;
  {
    scope_exit cleanup([&called] { called = true; });
    ASSERT_FALSE(called);
    cleanup.release();
  }
  ASSERT_FALSE(called);
}

TEST(LlvmLibcScopeExitTest, Move) {
  bool called = false;
  {
    scope_exit cleanup1([&called] { called = true; });
    {
      scope_exit cleanup2(LIBC_NAMESPACE::cpp::move(cleanup1));
      ASSERT_FALSE(called);
    }
    // cleanup2 goes out of scope here, should call the function.
    ASSERT_TRUE(called);
    called = false; // reset
  }
  // cleanup1 goes out of scope here, but it was moved from, so it shouldn't
  // call.
  ASSERT_FALSE(called);
}
