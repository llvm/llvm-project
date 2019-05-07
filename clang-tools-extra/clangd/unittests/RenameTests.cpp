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
  struct Test {
    const char* Before;
    const char* After;
  } Tests[] = {
      // Rename function.
      {
          R"cpp(
            void foo() {
              fo^o();
            }
          )cpp",
          R"cpp(
            void abcde() {
              abcde();
            }
          )cpp",
      },
      // Rename type.
      {
          R"cpp(
            struct foo{};
            foo test() {
               f^oo x;
               return x;
            }
          )cpp",
          R"cpp(
            struct abcde{};
            abcde test() {
               abcde x;
               return x;
            }
          )cpp",
      },
      // Rename variable.
      {
          R"cpp(
            void bar() {
              if (auto ^foo = 5) {
                foo = 3;
              }
            }
          )cpp",
          R"cpp(
            void bar() {
              if (auto abcde = 5) {
                abcde = 3;
              }
            }
          )cpp",
      },
  };
  for (const Test &T : Tests) {
    Annotations Code(T.Before);
    auto TU = TestTU::withCode(Code.code());
    TU.HeaderCode = "void foo();"; // outside main file, will not be touched.
    auto AST = TU.build();
    auto RenameResult =
        renameWithinFile(AST, testPath(TU.Filename), Code.point(), "abcde");
    ASSERT_TRUE(bool(RenameResult)) << RenameResult.takeError();
    auto ApplyResult =
        tooling::applyAllReplacements(Code.code(), *RenameResult);
    ASSERT_TRUE(bool(ApplyResult)) << ApplyResult.takeError();

    EXPECT_EQ(T.After, *ApplyResult) << T.Before;
  }
}

} // namespace
} // namespace clangd
} // namespace clang
