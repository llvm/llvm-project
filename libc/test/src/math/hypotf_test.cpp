//===-- Unittests for hypotf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HypotTest.h"

#include "include/math.h"
#include "src/math/hypotf.h"

using HypotfTest = HypotTestTemplate<float>;

TEST_F(HypotfTest, SpecialNumbers) { testSpecialNumbers(&__llvm_libc::hypotf); }

TEST_F(HypotfTest, SubnormalRange) { testSubnormalRange(&__llvm_libc::hypotf); }

TEST_F(HypotfTest, NormalRange) { testNormalRange(&__llvm_libc::hypotf); }
