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

MATCHER_P2(RenameRange, Code, Range, "") {
  return replacementToEdit(Code, arg).range == Range;
}

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

TEST(RenameTest, Renameable) {
  struct Case {
    const char *Code;
    const char* ErrorMessage; // null if no error
    bool IsHeaderFile;
  };
  const bool HeaderFile = true;
  Case Cases[] = {
      {R"cpp(// allow -- function-local
        void f(int [[Lo^cal]]) {
          [[Local]] = 2;
        }
      )cpp",
       nullptr, HeaderFile},

      {R"cpp(// allow -- symbol is indexable and has no refs in index.
        void [[On^lyInThisFile]]();
      )cpp",
       nullptr, HeaderFile},

      {R"cpp(// disallow -- symbol is indexable and has other refs in index.
        void f() {
          Out^side s;
        }
      )cpp",
       "used outside main file", HeaderFile},

      {R"cpp(// disallow -- symbol is not indexable.
        namespace {
        class Unin^dexable {};
        }
      )cpp",
       "not eligible for indexing", HeaderFile},

      {R"cpp(// disallow -- namespace symbol isn't supported
        namespace fo^o {}
      )cpp",
       "not a supported kind", HeaderFile},

      {
          R"cpp(
         #define MACRO 1
         int s = MAC^RO;
       )cpp",
          "not a supported kind", HeaderFile},

      {R"cpp(// foo is declared outside the file.
        void fo^o() {}
      )cpp", "used outside main file", !HeaderFile/*cc file*/},
  };
  const char *CommonHeader = R"cpp(
    class Outside {};
    void foo();
  )cpp";
  TestTU OtherFile = TestTU::withCode("Outside s; auto ss = &foo;");
  OtherFile.HeaderCode = CommonHeader;
  OtherFile.Filename = "other.cc";
  // The index has a "Outside" reference and a "foo" reference.
  auto OtherFileIndex = OtherFile.index();

  for (const auto& Case : Cases) {
    Annotations T(Case.Code);
    TestTU TU = TestTU::withCode(T.code());
    TU.HeaderCode = CommonHeader;
    if (Case.IsHeaderFile) {
      // We open the .h file as the main file.
      TU.Filename = "test.h";
      // Parsing the .h file as C++ include.
      TU.ExtraArgs.push_back("-xobjective-c++-header");
    }
    auto AST = TU.build();

    auto Results = renameWithinFile(AST, testPath(TU.Filename), T.point(),
                                    "dummyNewName", OtherFileIndex.get());
    bool WantRename = true;
    if (T.ranges().empty())
      WantRename = false;
    if (!WantRename) {
      assert(Case.ErrorMessage && "Error message must be set!");
      EXPECT_FALSE(Results) << "expected renameWithinFile returned an error: "
                            << T.code();
      auto ActualMessage = llvm::toString(Results.takeError());
      EXPECT_THAT(ActualMessage, testing::HasSubstr(Case.ErrorMessage));
    } else {
      EXPECT_TRUE(bool(Results)) << "renameWithinFile returned an error: "
                                 << llvm::toString(Results.takeError());
      std::vector<testing::Matcher<tooling::Replacement>> Expected;
      for (const auto &R : T.ranges())
        Expected.push_back(RenameRange(TU.Code, R));
      EXPECT_THAT(*Results, UnorderedElementsAreArray(Expected));
    }
  }
}

} // namespace
} // namespace clangd
} // namespace clang
