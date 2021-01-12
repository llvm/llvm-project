//===-- Unittests for fmaf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FmaTest.h"

#include "src/math/fmaf.h"

using FmaTest = FmaTestTemplate<float>;

TEST_F(FmaTest, SpecialNumbers) { testSpecialNumbers(&__llvm_libc::fmaf); }

TEST_F(FmaTest, SubnormalRange) { testSubnormalRange(&__llvm_libc::fmaf); }

TEST_F(FmaTest, NormalRange) { testNormalRange(&__llvm_libc::fmaf); }
