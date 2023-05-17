//===-- CollectMacrosTests.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "AST.h"
#include "Annotations.h"
#include "CollectMacros.h"
#include "Matchers.h"
#include "SourceCode.h"
#include "TestTU.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>

namespace clang {
namespace clangd {
namespace {

using testing::UnorderedElementsAreArray;

MATCHER_P(rangeIs, R, "") { return arg.Rng == R; }
MATCHER(isDef, "") { return arg.IsDefinition; }
MATCHER(inConditionalDirective, "") { return arg.InConditionalDirective; }

TEST(CollectMainFileMacros, SelectedMacros) {
  // References of the same symbol must have the ranges with the same
  // name(integer). If there are N different symbols then they must be named
  // from 1 to N. Macros for which SymbolID cannot be computed must be named
  // "Unknown". The payload of the annotation describes the extra bit
  // information of the MacroOccurrence (e.g. $1(def) => IsDefinition).
  const char *Tests[] = {
      R"cpp(// Macros: Cursor on definition.
        #define $1(def)[[FOO]](x,y) (x + y)
        int main() { int x = $1[[FOO]]($1[[FOO]](3, 4), $1[[FOO]](5, 6)); }
      )cpp",
      R"cpp(
        #define $1(def)[[M]](X) X;
        #define $2(def)[[abc]] 123
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
        #ifdef $Unknown(condit)[[UNDEFINED]]
        #elifdef $Unknown(condit)[[UNDEFINED]]
        #endif

        #ifdef $Unknown(condit)[[UNDEFINED]]
        #elifndef $Unknown(condit)[[UNDEFINED]]
        #endif

        #ifndef $Unknown(condit)[[UNDEFINED]]
        #endif

        #if defined($Unknown(condit)[[UNDEFINED]])
        #endif
      )cpp",
      R"cpp(
        #ifndef $Unknown(condit)[[abc]]
        #define $1(def)[[abc]]
        #ifdef $1(condit)[[abc]]
        #endif
        #endif
      )cpp",
      R"cpp(
        // Macros from token concatenations not included.
        #define $1(def)[[CONCAT]](X) X##A()
        #define $2(def)[[PREPEND]](X) MACRO##X()
        #define $3(def)[[MACROA]]() 123
        int B = $1[[CONCAT]](MACRO);
        int D = $2[[PREPEND]](A);
      )cpp",
      R"cpp(
        #define $1(def)[[MACRO_ARGS2]](X, Y) X Y
        #define $3(def)[[BAR]] 1
        #define $2(def)[[FOO]] $3[[BAR]]
        int A = $2[[FOO]];
      )cpp"};
  auto ExpectedResults = [](const Annotations &T, StringRef Name) {
    std::vector<Matcher<MacroOccurrence>> ExpectedLocations;
    for (const auto &[R, Bits] : T.rangesWithPayload(Name)) {
      if (Bits == "def")
        ExpectedLocations.push_back(testing::AllOf(rangeIs(R), isDef()));
      else if (Bits == "condit")
        ExpectedLocations.push_back(
            testing::AllOf(rangeIs(R), inConditionalDirective()));
      else
        ExpectedLocations.push_back(testing::AllOf(rangeIs(R)));
    }
    return ExpectedLocations;
  };

  for (const char *Test : Tests) {
    Annotations T(Test);
    auto Inputs = TestTU::withCode(T.code());
    Inputs.ExtraArgs.push_back("-std=c++2b");
    auto AST = Inputs.build();
    auto ActualMacroRefs = AST.getMacros();
    auto &SM = AST.getSourceManager();
    auto &PP = AST.getPreprocessor();
    for (const auto &[Name, Ranges] : T.all_ranges()) {
      if (Name == "Unknown") {
        EXPECT_THAT(ActualMacroRefs.UnknownMacros,
                    UnorderedElementsAreArray(ExpectedResults(T, "Unknown")))
            << "Unknown macros doesn't match in " << Test;
        continue;
      }

      auto Loc = sourceLocationInMainFile(
          SM, offsetToPosition(T.code(), Ranges.front().Begin));
      ASSERT_TRUE(bool(Loc));
      const auto *Id = syntax::spelledIdentifierTouching(*Loc, AST.getTokens());
      ASSERT_TRUE(Id);
      auto Macro = locateMacroAt(*Id, PP);
      assert(Macro);
      auto SID = getSymbolID(Macro->Name, Macro->Info, SM);

      EXPECT_THAT(ActualMacroRefs.MacroRefs[SID],
                  UnorderedElementsAreArray(ExpectedResults(T, Name)))
          << "Annotation=" << Name << ", MacroName=" << Macro->Name
          << ", Test = " << Test;
    }
  }
}
} // namespace
} // namespace clangd
} // namespace clang
