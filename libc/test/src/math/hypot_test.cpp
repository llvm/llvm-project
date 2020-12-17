//===-- Unittests for hypot -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HypotTest.h"

#include "src/math/hypot.h"

using HypotTest = HypotTestTemplate<double>;

TEST_F(HypotTest, SpecialNumbers) { testSpecialNumbers(&__llvm_libc::hypot); }

TEST_F(HypotTest, SubnormalRange) { testSubnormalRange(&__llvm_libc::hypot); }

TEST_F(HypotTest, NormalRange) { testNormalRange(&__llvm_libc::hypot); }
