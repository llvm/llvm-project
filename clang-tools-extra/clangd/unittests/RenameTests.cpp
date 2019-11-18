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

// Generates an expected rename result by replacing all ranges in the given
// annotation with the NewName.
std::string expectedResult(Annotations Test, llvm::StringRef NewName) {
  std::string Result;
  unsigned NextChar = 0;
  llvm::StringRef Code = Test.code();
  for (const auto &R : Test.llvm::Annotations::ranges()) {
    assert(R.Begin <= R.End && NextChar <= R.Begin);
    Result += Code.substr(NextChar, R.Begin - NextChar);
    Result += NewName;
    NextChar = R.End;
  }
  Result += Code.substr(NextChar);
  return Result;
}

TEST(RenameTest, SingleFile) {
  // "^" points to the position of the rename, and "[[]]" ranges point to the
  // identifier that is being renamed.
  llvm::StringRef Tests[] = {
      // Rename function.
      R"cpp(
        void [[foo]]() {
          [[fo^o]]();
        }
      )cpp",

      // Rename type.
      R"cpp(
        struct [[foo]]{};
        [[foo]] test() {
           [[f^oo]] x;
           return x;
        }
      )cpp",

      // Rename variable.
      R"cpp(
        void bar() {
          if (auto [[^foo]] = 5) {
            [[foo]] = 3;
          }
        }
      )cpp",
  };
  for (const auto T : Tests) {
    Annotations Code(T);
    auto TU = TestTU::withCode(Code.code());
    auto AST = TU.build();
    llvm::StringRef NewName = "abcde";
    auto RenameResult =
        renameWithinFile(AST, testPath(TU.Filename), Code.point(), NewName);
    ASSERT_TRUE(bool(RenameResult)) << RenameResult.takeError();
    auto ApplyResult = llvm::cantFail(
        tooling::applyAllReplacements(Code.code(), *RenameResult));
    EXPECT_EQ(expectedResult(Code, NewName), ApplyResult);
  }
}

TEST(RenameTest, Renameable) {
  struct Case {
    const char *Code;
    const char* ErrorMessage; // null if no error
    bool IsHeaderFile;
    const SymbolIndex *Index;
  };
  TestTU OtherFile = TestTU::withCode("Outside s; auto ss = &foo;");
  const char *CommonHeader = R"cpp(
    class Outside {};
    void foo();
  )cpp";
  OtherFile.HeaderCode = CommonHeader;
  OtherFile.Filename = "other.cc";
  // The index has a "Outside" reference and a "foo" reference.
  auto OtherFileIndex = OtherFile.index();
  const SymbolIndex *Index = OtherFileIndex.get();

  const bool HeaderFile = true;
  Case Cases[] = {
      {R"cpp(// allow -- function-local
        void f(int [[Lo^cal]]) {
          [[Local]] = 2;
        }
      )cpp",
       nullptr, HeaderFile, Index},

      {R"cpp(// allow -- symbol is indexable and has no refs in index.
        void [[On^lyInThisFile]]();
      )cpp",
       nullptr, HeaderFile, Index},

      {R"cpp(// disallow -- symbol is indexable and has other refs in index.
        void f() {
          Out^side s;
        }
      )cpp",
       "used outside main file", HeaderFile, Index},

      {R"cpp(// disallow -- symbol is not indexable.
        namespace {
        class Unin^dexable {};
        }
      )cpp",
       "not eligible for indexing", HeaderFile, Index},

      {R"cpp(// disallow -- namespace symbol isn't supported
        namespace n^s {}
      )cpp",
       "not a supported kind", HeaderFile, Index},

      {
          R"cpp(
         #define MACRO 1
         int s = MAC^RO;
       )cpp",
          "not a supported kind", HeaderFile, Index},

      {

          R"cpp(
        struct X { X operator++(int); };
        void f(X x) {x+^+;})cpp",
          "not a supported kind", HeaderFile, Index},

      {R"cpp(// foo is declared outside the file.
        void fo^o() {}
      )cpp", "used outside main file", !HeaderFile /*cc file*/, Index},

      {R"cpp(
         // We should detect the symbol is used outside the file from the AST.
         void fo^o() {})cpp",
       "used outside main file", !HeaderFile, nullptr /*no index*/},
  };

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
    EXPECT_TRUE(AST.getDiagnostics().empty())
      << AST.getDiagnostics().front() << T.code();
    llvm::StringRef NewName = "dummyNewName";
    auto Results = renameWithinFile(AST, testPath(TU.Filename), T.point(),
                                    NewName, Case.Index);
    bool WantRename = true;
    if (T.ranges().empty())
      WantRename = false;
    if (!WantRename) {
      assert(Case.ErrorMessage && "Error message must be set!");
      EXPECT_FALSE(Results)
          << "expected renameWithinFile returned an error: " << T.code();
      auto ActualMessage = llvm::toString(Results.takeError());
      EXPECT_THAT(ActualMessage, testing::HasSubstr(Case.ErrorMessage));
    } else {
      EXPECT_TRUE(bool(Results)) << "renameWithinFile returned an error: "
                                 << llvm::toString(Results.takeError());
      auto ApplyResult =
          llvm::cantFail(tooling::applyAllReplacements(T.code(), *Results));
      EXPECT_EQ(expectedResult(T, NewName), ApplyResult);
    }
  }
}

} // namespace
} // namespace clangd
} // namespace clang
