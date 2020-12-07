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

#include "utils/UnitTest/Test.h"

#include <fenv.h>

TEST(ExceptionStatusTest, RaiseAndTest) {
  int excepts[] = {FE_DIVBYZERO, FE_INVALID, FE_INEXACT, FE_OVERFLOW,
                   FE_UNDERFLOW};
  for (int e : excepts) {
    int r = __llvm_libc::feraiseexcept(e);
    EXPECT_EQ(r, 0);
    int s = __llvm_libc::fetestexcept(e);
    EXPECT_EQ(s, e);

    r = __llvm_libc::feclearexcept(e);
    EXPECT_EQ(r, 0);
    s = __llvm_libc::fetestexcept(e);
    EXPECT_EQ(s, 0);
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      int e = e1 | e2;
      int r = __llvm_libc::feraiseexcept(e);
      EXPECT_EQ(r, 0);
      int s = __llvm_libc::fetestexcept(e);
      EXPECT_EQ(s, e);

      r = __llvm_libc::feclearexcept(e);
      EXPECT_EQ(r, 0);
      s = __llvm_libc::fetestexcept(e);
      EXPECT_EQ(s, 0);
    }
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      for (int e3 : excepts) {
        int e = e1 | e2 | e3;
        int r = __llvm_libc::feraiseexcept(e);
        EXPECT_EQ(r, 0);
        int s = __llvm_libc::fetestexcept(e);
        EXPECT_EQ(s, e);

        r = __llvm_libc::feclearexcept(e);
        EXPECT_EQ(r, 0);
        s = __llvm_libc::fetestexcept(e);
        EXPECT_EQ(s, 0);
      }
    }
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      for (int e3 : excepts) {
        for (int e4 : excepts) {
          int e = e1 | e2 | e3 | e4;
          int r = __llvm_libc::feraiseexcept(e);
          EXPECT_EQ(r, 0);
          int s = __llvm_libc::fetestexcept(e);
          EXPECT_EQ(s, e);

          r = __llvm_libc::feclearexcept(e);
          EXPECT_EQ(r, 0);
          s = __llvm_libc::fetestexcept(e);
          EXPECT_EQ(s, 0);
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
            int r = __llvm_libc::feraiseexcept(e);
            EXPECT_EQ(r, 0);
            int s = __llvm_libc::fetestexcept(e);
            EXPECT_EQ(s, e);

            r = __llvm_libc::feclearexcept(e);
            EXPECT_EQ(r, 0);
            s = __llvm_libc::fetestexcept(e);
            EXPECT_EQ(s, 0);
          }
        }
      }
    }
  }

  int r = __llvm_libc::feraiseexcept(FE_ALL_EXCEPT);
  EXPECT_EQ(r, 0);
  int s = __llvm_libc::fetestexcept(FE_ALL_EXCEPT);
  EXPECT_EQ(s, FE_ALL_EXCEPT);
}
