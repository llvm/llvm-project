//===- bolt/unittests/Profile/DataAggregator.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Profile/DataAggregator.h"
#include "llvm/Support/CommandLine.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::bolt;

namespace opts {
extern cl::opt<bool> ReadPreAggregated;
}

TEST(DataAggregatorTest, buildID) {
  // Avoid looking for perf tool.
  opts::ReadPreAggregated = true;

  DataAggregator DA("<pseudo input>");
  Optional<StringRef> FileName;

  DA.setParsingBuffer("");
  ASSERT_FALSE(DA.hasAllBuildIDs());
  FileName = DA.getFileNameForBuildID("1234");
  ASSERT_FALSE(FileName);

  StringRef PartialValidBuildIDs = "     File0\n"
                                   "1111 File1\n"
                                   "     File2\n";
  DA.setParsingBuffer(PartialValidBuildIDs);
  ASSERT_FALSE(DA.hasAllBuildIDs());
  FileName = DA.getFileNameForBuildID("0000");
  ASSERT_FALSE(FileName);
  FileName = DA.getFileNameForBuildID("1111");
  ASSERT_EQ(*FileName, "File1");

  StringRef AllValidBuildIDs = "0000 File0\n"
                               "1111 File1\n"
                               "2222 File2\n"
                               "333  File3\n";
  DA.setParsingBuffer(AllValidBuildIDs);
  ASSERT_TRUE(DA.hasAllBuildIDs());
  FileName = DA.getFileNameForBuildID("1234");
  ASSERT_FALSE(FileName);
  FileName = DA.getFileNameForBuildID("2222");
  ASSERT_EQ(*FileName, "File2");
  FileName = DA.getFileNameForBuildID("333");
  ASSERT_EQ(*FileName, "File3");
}
