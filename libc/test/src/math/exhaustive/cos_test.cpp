//===-- Worst case test for cos -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/cos.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "worst_case_test.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

using LlvmLibcCosWorstCaseTest =
    LlvmLibcUnaryOpWorstCaseMathTest<double, mpfr::Operation::Cos,
                                     LIBC_NAMESPACE::cos>;

TEST_F(LlvmLibcCosWorstCaseTest, WorstCases) {
  test_file_all_roundings("cos", /*test_symmetric=*/true);
}
