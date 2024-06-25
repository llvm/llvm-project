//===-- Unittests for uksqrtui --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ISqrtTest.h"

#include "src/__support/fixed_point/sqrt.h"
#include "src/stdfix/uksqrtui.h"

unsigned accum uksqrtui_fast(unsigned int x) {
  return LIBC_NAMESPACE::fixed_point::isqrt_fast(x);
}

LIST_ISQRT_TESTS(UI, unsigned int, LIBC_NAMESPACE::uksqrtui);

TEST_F(LlvmLibcISqrtUITest, LargeInteger) {
  testSpecificInput(65529u, LIBC_NAMESPACE::uksqrtui(65529u), 0x1.fff8fep7,
                    0x3.0p-16);
}

LIST_ISQRT_TESTS(UIFast, unsigned int, uksqrtui_fast);

TEST_F(LlvmLibcISqrtUIFastTest, LargeInteger) {
  testSpecificInput(65529u, uksqrtui_fast(65529u), 0x1.fff8fep7, 0x3.0p-16);
}
