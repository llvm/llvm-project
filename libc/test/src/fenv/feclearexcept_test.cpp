//===-- Unittests for feclearexcept with exceptions enabled ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/feclearexcept.h"

#include "src/__support/FPUtil/FEnvUtils.h"
#include "utils/UnitTest/Test.h"

#include <fenv.h>
#include <stdint.h>

TEST(LlvmLibcFEnvTest, ClearTest) {
  uint16_t excepts[] = {FE_DIVBYZERO, FE_INVALID, FE_INEXACT, FE_OVERFLOW,
                        FE_UNDERFLOW};
  __llvm_libc::fputil::disable_except(FE_ALL_EXCEPT);
  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);

  for (uint16_t e : excepts)
    ASSERT_EQ(__llvm_libc::fputil::test_except(e), 0);

  __llvm_libc::fputil::raise_except(FE_ALL_EXCEPT);
  for (uint16_t e : excepts) {
    // We clear one exception and test to verify that it was cleared.
    __llvm_libc::feclearexcept(e);
    ASSERT_EQ(uint16_t(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT)),
              uint16_t(FE_ALL_EXCEPT & ~e));
    // After clearing, we raise the exception again.
    __llvm_libc::fputil::raise_except(e);
  }

  for (uint16_t e1 : excepts) {
    for (uint16_t e2 : excepts) {
      __llvm_libc::feclearexcept(e1 | e2);
      ASSERT_EQ(uint16_t(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT)),
                uint16_t(FE_ALL_EXCEPT & ~(e1 | e2)));
      __llvm_libc::fputil::raise_except(e1 | e2);
    }
  }

  for (uint16_t e1 : excepts) {
    for (uint16_t e2 : excepts) {
      for (uint16_t e3 : excepts) {
        __llvm_libc::feclearexcept(e1 | e2 | e3);
        ASSERT_EQ(uint16_t(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT)),
                  uint16_t(FE_ALL_EXCEPT & ~(e1 | e2 | e3)));
        __llvm_libc::fputil::raise_except(e1 | e2 | e3);
      }
    }
  }

  for (uint16_t e1 : excepts) {
    for (uint16_t e2 : excepts) {
      for (uint16_t e3 : excepts) {
        for (uint16_t e4 : excepts) {
          __llvm_libc::feclearexcept(e1 | e2 | e3 | e4);
          ASSERT_EQ(uint16_t(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT)),
                    uint16_t(FE_ALL_EXCEPT & ~(e1 | e2 | e3 | e4)));
          __llvm_libc::fputil::raise_except(e1 | e2 | e3 | e4);
        }
      }
    }
  }

  for (uint16_t e1 : excepts) {
    for (uint16_t e2 : excepts) {
      for (uint16_t e3 : excepts) {
        for (uint16_t e4 : excepts) {
          for (uint16_t e5 : excepts) {
            __llvm_libc::feclearexcept(e1 | e2 | e3 | e4 | e5);
            ASSERT_EQ(uint16_t(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT)),
                      uint16_t(FE_ALL_EXCEPT & ~(e1 | e2 | e3 | e4 | e5)));
            __llvm_libc::fputil::raise_except(e1 | e2 | e3 | e4 | e5);
          }
        }
      }
    }
  }
}
