//===-- Worst case test for sin -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sin.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "worst_case_test.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

using LlvmLibcSinWorstCaseTest =
    LlvmLibcUnaryOpWorstCaseMathTest<double, mpfr::Operation::Sin,
                                     LIBC_NAMESPACE::sin>;

TEST_F(LlvmLibcSinWorstCaseTest, WorstCases) {
  test_file_all_roundings("sin", /*test_symmetric=*/true);
}
