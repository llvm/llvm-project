//===-- Unittests for feclearexcept, feraiseexcept and fetestexpect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/feclearexcept.h"
#include "src/fenv/feraiseexcept.h"
#include "src/fenv/fetestexcept.h"

#include "src/__support/FPUtil/FEnvImpl.h"
#include "test/UnitTest/Test.h"

#include <fenv.h>

TEST(LlvmLibcExceptionStatusTest, RaiseAndTest) {
  // This test raises a set of exceptions and checks that the exception
  // status flags are updated. The intention is really not to invoke the
  // exception handler. Hence, we will disable all exceptions at the
  // beginning.
  LIBC_NAMESPACE::fputil::disable_except(FE_ALL_EXCEPT);

  int excepts[] = {FE_DIVBYZERO, FE_INVALID, FE_INEXACT, FE_OVERFLOW,
                   FE_UNDERFLOW};

  constexpr int ALL_EXCEPTS =
      FE_DIVBYZERO | FE_INVALID | FE_INEXACT | FE_OVERFLOW | FE_UNDERFLOW;

  for (int e : excepts) {
    int r = LIBC_NAMESPACE::feraiseexcept(e);
    ASSERT_EQ(r, 0);
    int s = LIBC_NAMESPACE::fetestexcept(e);
    ASSERT_EQ(s, e);

    r = LIBC_NAMESPACE::feclearexcept(e);
    ASSERT_EQ(r, 0);
    s = LIBC_NAMESPACE::fetestexcept(e);
    ASSERT_EQ(s, 0);
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      int e = e1 | e2;
      int r = LIBC_NAMESPACE::feraiseexcept(e);
      ASSERT_EQ(r, 0);
      int s = LIBC_NAMESPACE::fetestexcept(e);
      ASSERT_EQ(s, e);

      r = LIBC_NAMESPACE::feclearexcept(e);
      ASSERT_EQ(r, 0);
      s = LIBC_NAMESPACE::fetestexcept(e);
      ASSERT_EQ(s, 0);
    }
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      for (int e3 : excepts) {
        int e = e1 | e2 | e3;
        int r = LIBC_NAMESPACE::feraiseexcept(e);
        ASSERT_EQ(r, 0);
        int s = LIBC_NAMESPACE::fetestexcept(e);
        ASSERT_EQ(s, e);

        r = LIBC_NAMESPACE::feclearexcept(e);
        ASSERT_EQ(r, 0);
        s = LIBC_NAMESPACE::fetestexcept(e);
        ASSERT_EQ(s, 0);
      }
    }
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      for (int e3 : excepts) {
        for (int e4 : excepts) {
          int e = e1 | e2 | e3 | e4;
          int r = LIBC_NAMESPACE::feraiseexcept(e);
          ASSERT_EQ(r, 0);
          int s = LIBC_NAMESPACE::fetestexcept(e);
          ASSERT_EQ(s, e);

          r = LIBC_NAMESPACE::feclearexcept(e);
          ASSERT_EQ(r, 0);
          s = LIBC_NAMESPACE::fetestexcept(e);
          ASSERT_EQ(s, 0);
        }
      }
    }
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      for (int e3 : excepts) {
        for (int e4 : excepts) {
          for (int e5 : excepts) {
            int e = e1 | e2 | e3 | e4 | e5;
            int r = LIBC_NAMESPACE::feraiseexcept(e);
            ASSERT_EQ(r, 0);
            int s = LIBC_NAMESPACE::fetestexcept(e);
            ASSERT_EQ(s, e);

            r = LIBC_NAMESPACE::feclearexcept(e);
            ASSERT_EQ(r, 0);
            s = LIBC_NAMESPACE::fetestexcept(e);
            ASSERT_EQ(s, 0);
          }
        }
      }
    }
  }

  int r = LIBC_NAMESPACE::feraiseexcept(ALL_EXCEPTS);
  ASSERT_EQ(r, 0);
  int s = LIBC_NAMESPACE::fetestexcept(ALL_EXCEPTS);
  ASSERT_EQ(s, ALL_EXCEPTS);
}
