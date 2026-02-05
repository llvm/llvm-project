//===- SummaryExtractorRegistryTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Serialization/SerializationFormatRegistry.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ssaf;

namespace {

TEST(SerializationFormatRegistryTest, isFormatRegistered) {
  EXPECT_FALSE(isFormatRegistered("Non-existent-format"));
  EXPECT_TRUE(isFormatRegistered("MockSerializationFormat"));
}

TEST(SerializationFormatRegistryTest, EnumeratingRegistryEntries) {
  auto Formats = SerializationFormatRegistry::entries();
  ASSERT_EQ(std::distance(Formats.begin(), Formats.end()), 1U);
  EXPECT_EQ(Formats.begin()->getName(), "MockSerializationFormat");
}

} // namespace
