//===- llvm/unittest/MC/MCInstPrinter.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/DynoStats.h"
#include "bolt/Core/BinaryFunction.h"
#include "gtest/gtest.h"
#include <map>

using namespace llvm::bolt;

TEST(DynoStatsTest, emptyFuncs) {
  std::map<uint64_t, BinaryFunction> BinaryFunctions;
  DynoStats DynoStatsAArch64 =
      getDynoStats(BinaryFunctions, /* BC.isAArch64() = */ true);
  DynoStats DynoStatsNonAArch64 =
      getDynoStats(BinaryFunctions, /* BC.isAArch64() = */ false);
  // Both should be null
  ASSERT_EQ(DynoStatsAArch64, DynoStatsNonAArch64);
}
