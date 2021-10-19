//===-- DraftStoreTests.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "DraftStore.h"
#include "SourceCode.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(DraftStore, Versions) {
  DraftStore DS;
  Path File = "foo.cpp";

  EXPECT_EQ("25", DS.addDraft(File, "25", ""));
  EXPECT_EQ("25", DS.getDraft(File)->Version);
  EXPECT_EQ("", *DS.getDraft(File)->Contents);

  EXPECT_EQ("26", DS.addDraft(File, "", "x"));
  EXPECT_EQ("26", DS.getDraft(File)->Version);
  EXPECT_EQ("x", *DS.getDraft(File)->Contents);

  EXPECT_EQ("27", DS.addDraft(File, "", "x")) << "no-op change";
  EXPECT_EQ("27", DS.getDraft(File)->Version);
  EXPECT_EQ("x", *DS.getDraft(File)->Contents);

  // We allow versions to go backwards.
  EXPECT_EQ("7", DS.addDraft(File, "7", "y"));
  EXPECT_EQ("7", DS.getDraft(File)->Version);
  EXPECT_EQ("y", *DS.getDraft(File)->Contents);
}

} // namespace
} // namespace clangd
} // namespace clang
