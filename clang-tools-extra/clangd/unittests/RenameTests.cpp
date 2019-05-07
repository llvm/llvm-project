//===-- RenameTests.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "TestFS.h"
#include "TestTU.h"
#include "refactor/Rename.h"
#include "clang/Tooling/Core/Replacement.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(RenameTest, SingleFile) {
  Annotations Code(R"cpp(
    void foo() {
      fo^o();
    }
  )cpp");
  auto TU = TestTU::withCode(Code.code());
  TU.HeaderCode = "void foo();"; // outside main file, will not be touched.

  auto AST = TU.build();
  auto RenameResult =
      renameWithinFile(AST, testPath(TU.Filename), Code.point(), "abcde");
  ASSERT_TRUE(bool(RenameResult)) << RenameResult.takeError();
  auto ApplyResult = tooling::applyAllReplacements(Code.code(), *RenameResult);
  ASSERT_TRUE(bool(ApplyResult)) << ApplyResult.takeError();

  const char *Want = R"cpp(
    void abcde() {
      abcde();
    }
  )cpp";
  EXPECT_EQ(Want, *ApplyResult);
}

} // namespace
} // namespace clangd
} // namespace clang
