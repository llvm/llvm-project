//===-- Unittests for feenableexcept  -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/architectures.h"
#include "src/fenv/fedisableexcept.h"
#include "src/fenv/feenableexcept.h"
#include "src/fenv/fegetexcept.h"

#include "test/UnitTest/Test.h"

#include <fenv.h>

TEST(LlvmLibcFEnvTest, EnableTest) {
#if defined(LIBC_TARGET_IS_AARCH64)
  // Few aarch64 HW implementations do not trap exceptions. We skip this test
  // completely on such HW.
  //
  // Whether HW supports trapping exceptions or not is deduced by enabling an
  // exception and reading back to see if the exception got enabled. If the
  // exception did not get enabled, then it means that the HW does not support
  // trapping exceptions.
  __llvm_libc::fedisableexcept(FE_ALL_EXCEPT);
  __llvm_libc::feenableexcept(FE_DIVBYZERO);
  if (__llvm_libc::fegetexcept() == 0)
    return;
#endif // defined(LIBC_TARGET_IS_AARCH64)

  int excepts[] = {FE_DIVBYZERO, FE_INVALID, FE_INEXACT, FE_OVERFLOW,
                   FE_UNDERFLOW};
  __llvm_libc::fedisableexcept(FE_ALL_EXCEPT);
  ASSERT_EQ(0, __llvm_libc::fegetexcept());

  for (int e : excepts) {
    __llvm_libc::feenableexcept(e);
    ASSERT_EQ(e, __llvm_libc::fegetexcept());
    __llvm_libc::fedisableexcept(e);
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      __llvm_libc::feenableexcept(e1 | e2);
      ASSERT_EQ(e1 | e2, __llvm_libc::fegetexcept());
      __llvm_libc::fedisableexcept(e1 | e2);
    }
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      for (int e3 : excepts) {
        __llvm_libc::feenableexcept(e1 | e2 | e3);
        ASSERT_EQ(e1 | e2 | e3, __llvm_libc::fegetexcept());
        __llvm_libc::fedisableexcept(e1 | e2 | e3);
      }
    }
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      for (int e3 : excepts) {
        for (int e4 : excepts) {
          __llvm_libc::feenableexcept(e1 | e2 | e3 | e4);
          ASSERT_EQ(e1 | e2 | e3 | e4, __llvm_libc::fegetexcept());
          __llvm_libc::fedisableexcept(e1 | e2 | e3 | e4);
        }
      }
    }
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      for (int e3 : excepts) {
        for (int e4 : excepts) {
          for (int e5 : excepts) {
            __llvm_libc::feenableexcept(e1 | e2 | e3 | e4 | e5);
            ASSERT_EQ(e1 | e2 | e3 | e4 | e5, __llvm_libc::fegetexcept());
            __llvm_libc::fedisableexcept(e1 | e2 | e3 | e4 | e5);
          }
        }
      }
    }
  }
}
