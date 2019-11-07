//===-- CollectMacrosTests.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "CollectMacros.h"
#include "Matchers.h"
#include "SourceCode.h"
#include "TestTU.h"
#include "index/SymbolID.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using testing::UnorderedElementsAreArray;

TEST(CollectMainFileMacros, SelectedMacros) {
  // References of the same symbol must have the ranges with the same
  // name(integer). If there are N different symbols then they must be named
  // from 1 to N. Macros for which SymbolID cannot be computed must be named
  // "Unknown".
  const char *Tests[] = {
      R"cpp(// Macros: Cursor on definition.
        #define $1[[FOO]](x,y) (x + y)
        int main() { int x = $1[[FOO]]($1[[FOO]](3, 4), $1[[FOO]](5, 6)); }
      )cpp",
      R"cpp(
        #define $1[[M]](X) X;
        #define $2[[abc]] 123
        int s = $1[[M]]($2[[abc]]);
      )cpp",
      // FIXME: Locating macro in duplicate definitions doesn't work. Enable
      // this once LocateMacro is fixed.
      // R"cpp(// Multiple definitions.
      //   #define $1[[abc]] 1
      //   int func1() { int a = $1[[abc]];}
      //   #undef $1[[abc]]

      //   #define $2[[abc]] 2
      //   int func2() { int a = $2[[abc]];}
      //   #undef $2[[abc]]
      // )cpp",
      R"cpp(
        #ifdef $Unknown[[UNDEFINED]]
        #endif
      )cpp",
      R"cpp(
        #ifndef $Unknown[[abc]]
        #define $1[[abc]]
        #ifdef $1[[abc]]
        #endif
        #endif
      )cpp",
      R"cpp(
        // Macros from token concatenations not included.
        #define $1[[CONCAT]](X) X##A()
        #define $2[[PREPEND]](X) MACRO##X()
        #define $3[[MACROA]]() 123
        int B = $1[[CONCAT]](MACRO);
        int D = $2[[PREPEND]](A)
      )cpp",
      R"cpp(
        // FIXME: Macro names in a definition are not detected.
        #define $1[[MACRO_ARGS2]](X, Y) X Y
        #define $2[[FOO]] BAR
        #define $3[[BAR]] 1
        int A = $2[[FOO]];
      )cpp"};
  for (const char *Test : Tests) {
    Annotations T(Test);
    auto AST = TestTU::withCode(T.code()).build();
    auto ActualMacroRefs = AST.getMacros();
    auto &SM = AST.getSourceManager();
    auto &PP = AST.getPreprocessor();

    // Known macros.
    for (int I = 1;; I++) {
      const auto ExpectedRefs = T.ranges(llvm::to_string(I));
      if (ExpectedRefs.empty())
        break;

      auto Loc = getBeginningOfIdentifier(ExpectedRefs.begin()->start, SM,
                                          AST.getASTContext().getLangOpts());
      auto Macro = locateMacroAt(Loc, PP);
      assert(Macro);
      auto SID = getSymbolID(Macro->Name, Macro->Info, SM);

      EXPECT_THAT(ExpectedRefs,
                  UnorderedElementsAreArray(ActualMacroRefs.MacroRefs[*SID]))
          << "Annotation=" << I << ", MacroName=" << Macro->Name
          << ", Test = " << Test;
    }
    // Unkown macros.
    EXPECT_THAT(AST.getMacros().UnknownMacros,
                UnorderedElementsAreArray(T.ranges("Unknown")))
        << "Unknown macros doesn't match in " << Test;
  }
}
} // namespace
} // namespace clangd
} // namespace clang
