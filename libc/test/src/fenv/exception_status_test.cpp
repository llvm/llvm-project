//===-- Unittests for feclearexcept, feraiseexcept, fetestexpect ----------===//
//===-- and fesetexcept ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/feclearexcept.h"
#include "src/fenv/feraiseexcept.h"
#include "src/fenv/fesetexcept.h"
#include "src/fenv/fetestexcept.h"

#include "src/__support/FPUtil/FEnvImpl.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/Test.h"

#include "hdr/fenv_macros.h"

#include "excepts.h"

using LlvmLibcExceptionStatusTest = LIBC_NAMESPACE::testing::FEnvSafeTest;

TEST_F(LlvmLibcExceptionStatusTest, RaiseAndTest) {
  // This test raises a set of exceptions and checks that the exception
  // status flags are updated. The intention is really not to invoke the
  // exception handler. Hence, we will disable all exceptions at the
  // beginning.
  LIBC_NAMESPACE::fputil::disable_except(FE_ALL_EXCEPT);

  for (int e : EXCEPTS) {
    int r = LIBC_NAMESPACE::feraiseexcept(e);
    ASSERT_EQ(r, 0);
    int s = LIBC_NAMESPACE::fetestexcept(e);
    ASSERT_EQ(s, e);

    r = LIBC_NAMESPACE::feclearexcept(e);
    ASSERT_EQ(r, 0);
    s = LIBC_NAMESPACE::fetestexcept(e);
    ASSERT_EQ(s, 0);

    r = LIBC_NAMESPACE::fesetexcept(e);
    ASSERT_EQ(r, 0);
    s = LIBC_NAMESPACE::fetestexcept(e);
    ASSERT_EQ(s, e);
  }

  for (int e1 : EXCEPTS) {
    for (int e2 : EXCEPTS) {
      int e = e1 | e2;
      int r = LIBC_NAMESPACE::feraiseexcept(e);
      ASSERT_EQ(r, 0);
      int s = LIBC_NAMESPACE::fetestexcept(e);
      ASSERT_EQ(s, e);

      r = LIBC_NAMESPACE::feclearexcept(e);
      ASSERT_EQ(r, 0);
      s = LIBC_NAMESPACE::fetestexcept(e);
      ASSERT_EQ(s, 0);

      r = LIBC_NAMESPACE::fesetexcept(e);
      ASSERT_EQ(r, 0);
      s = LIBC_NAMESPACE::fetestexcept(e);
      ASSERT_EQ(s, e);
    }
  }

  for (int e1 : EXCEPTS) {
    for (int e2 : EXCEPTS) {
      for (int e3 : EXCEPTS) {
        int e = e1 | e2 | e3;
        int r = LIBC_NAMESPACE::feraiseexcept(e);
        ASSERT_EQ(r, 0);
        int s = LIBC_NAMESPACE::fetestexcept(e);
        ASSERT_EQ(s, e);

        r = LIBC_NAMESPACE::feclearexcept(e);
        ASSERT_EQ(r, 0);
        s = LIBC_NAMESPACE::fetestexcept(e);
        ASSERT_EQ(s, 0);

        r = LIBC_NAMESPACE::fesetexcept(e);
        ASSERT_EQ(r, 0);
        s = LIBC_NAMESPACE::fetestexcept(e);
        ASSERT_EQ(s, e);
      }
    }
  }

  for (int e1 : EXCEPTS) {
    for (int e2 : EXCEPTS) {
      for (int e3 : EXCEPTS) {
        for (int e4 : EXCEPTS) {
          int e = e1 | e2 | e3 | e4;
          int r = LIBC_NAMESPACE::feraiseexcept(e);
          ASSERT_EQ(r, 0);
          int s = LIBC_NAMESPACE::fetestexcept(e);
          ASSERT_EQ(s, e);

          r = LIBC_NAMESPACE::feclearexcept(e);
          ASSERT_EQ(r, 0);
          s = LIBC_NAMESPACE::fetestexcept(e);
          ASSERT_EQ(s, 0);

          r = LIBC_NAMESPACE::fesetexcept(e);
          ASSERT_EQ(r, 0);
          s = LIBC_NAMESPACE::fetestexcept(e);
          ASSERT_EQ(s, e);
        }
      }
    }
  }

  for (int e1 : EXCEPTS) {
    for (int e2 : EXCEPTS) {
      for (int e3 : EXCEPTS) {
        for (int e4 : EXCEPTS) {
          for (int e5 : EXCEPTS) {
            int e = e1 | e2 | e3 | e4 | e5;
            int r = LIBC_NAMESPACE::feraiseexcept(e);
            ASSERT_EQ(r, 0);
            int s = LIBC_NAMESPACE::fetestexcept(e);
            ASSERT_EQ(s, e);

            r = LIBC_NAMESPACE::feclearexcept(e);
            ASSERT_EQ(r, 0);
            s = LIBC_NAMESPACE::fetestexcept(e);
            ASSERT_EQ(s, 0);

            r = LIBC_NAMESPACE::fesetexcept(e);
            ASSERT_EQ(r, 0);
            s = LIBC_NAMESPACE::fetestexcept(e);
            ASSERT_EQ(s, e);
          }
        }
      }
    }
  }

  int r = LIBC_NAMESPACE::feraiseexcept(ALL_EXCEPTS);
  ASSERT_EQ(r, 0);
  int s = LIBC_NAMESPACE::fetestexcept(ALL_EXCEPTS);
  ASSERT_EQ(s, ALL_EXCEPTS);

  r = LIBC_NAMESPACE::fesetexcept(ALL_EXCEPTS);
  ASSERT_EQ(r, 0);
  s = LIBC_NAMESPACE::fetestexcept(ALL_EXCEPTS);
  ASSERT_EQ(s, ALL_EXCEPTS);
}
