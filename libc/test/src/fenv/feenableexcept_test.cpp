//===-- Unittests for feenableexcept  -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/properties/architectures.h"
#include "src/fenv/fedisableexcept.h"
#include "src/fenv/feenableexcept.h"
#include "src/fenv/fegetexcept.h"

#include "test/UnitTest/Test.h"

#include <fenv.h>

TEST(LlvmLibcFEnvTest, EnableTest) {
#if defined(LIBC_TARGET_ARCH_IS_ANY_ARM) || defined(LIBC_TARGET_ARCH_IS_RISCV64)
  // Few Arm HW implementations do not trap exceptions. We skip this test
  // completely on such HW.
  //
  // Whether HW supports trapping exceptions or not is deduced by enabling an
  // exception and reading back to see if the exception got enabled. If the
  // exception did not get enabled, then it means that the HW does not support
  // trapping exceptions.
  LIBC_NAMESPACE::fedisableexcept(FE_ALL_EXCEPT);
  LIBC_NAMESPACE::feenableexcept(FE_DIVBYZERO);
  if (LIBC_NAMESPACE::fegetexcept() == 0)
    return;
#endif // Architectures where exception trapping is not supported

  int excepts[] = {FE_DIVBYZERO, FE_INVALID, FE_INEXACT, FE_OVERFLOW,
                   FE_UNDERFLOW};
  LIBC_NAMESPACE::fedisableexcept(FE_ALL_EXCEPT);
  ASSERT_EQ(0, LIBC_NAMESPACE::fegetexcept());

  for (int e : excepts) {
    LIBC_NAMESPACE::feenableexcept(e);
    ASSERT_EQ(e, LIBC_NAMESPACE::fegetexcept());
    LIBC_NAMESPACE::fedisableexcept(e);
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      LIBC_NAMESPACE::feenableexcept(e1 | e2);
      ASSERT_EQ(e1 | e2, LIBC_NAMESPACE::fegetexcept());
      LIBC_NAMESPACE::fedisableexcept(e1 | e2);
    }
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      for (int e3 : excepts) {
        LIBC_NAMESPACE::feenableexcept(e1 | e2 | e3);
        ASSERT_EQ(e1 | e2 | e3, LIBC_NAMESPACE::fegetexcept());
        LIBC_NAMESPACE::fedisableexcept(e1 | e2 | e3);
      }
    }
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      for (int e3 : excepts) {
        for (int e4 : excepts) {
          LIBC_NAMESPACE::feenableexcept(e1 | e2 | e3 | e4);
          ASSERT_EQ(e1 | e2 | e3 | e4, LIBC_NAMESPACE::fegetexcept());
          LIBC_NAMESPACE::fedisableexcept(e1 | e2 | e3 | e4);
        }
      }
    }
  }

  for (int e1 : excepts) {
    for (int e2 : excepts) {
      for (int e3 : excepts) {
        for (int e4 : excepts) {
          for (int e5 : excepts) {
            LIBC_NAMESPACE::feenableexcept(e1 | e2 | e3 | e4 | e5);
            ASSERT_EQ(e1 | e2 | e3 | e4 | e5, LIBC_NAMESPACE::fegetexcept());
            LIBC_NAMESPACE::fedisableexcept(e1 | e2 | e3 | e4 | e5);
          }
        }
      }
    }
  }
}
