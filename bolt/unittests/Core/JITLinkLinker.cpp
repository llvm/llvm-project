//===- bolt/unittest/Core/JITLinkLinker.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/JITLinkLinker.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::bolt;
using namespace llvm::jitlink;

TEST(JITLinkLinkerTest, AssignBlockAddressesRespectsAlignment) {
  LinkGraph G("test", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-unknown-linux"), SubtargetFeatures(),
              getGenericEdgeKindName);
  auto &Section =
      G.createSection(".data", orc::MemProt::Read | orc::MemProt::Write);

  auto &First = G.createContentBlock(Section, ArrayRef<char>("aaaaa", 5),
                                     orc::ExecutorAddr(0), 16, 0);
  auto &Second = G.createContentBlock(Section, ArrayRef<char>("bbbb", 4),
                                      orc::ExecutorAddr(20), 16, 4);

  JITLinkLinker::assignBlockAddresses(Section, 0);
  EXPECT_EQ(First.getAddress().getValue() % First.getAlignment(),
            First.getAlignmentOffset());
  EXPECT_EQ(Second.getAddress().getValue() % Second.getAlignment(),
            Second.getAlignmentOffset());
}
