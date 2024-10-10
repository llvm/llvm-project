//=== unittests/Sema/GetCountedByAttrSourceRange.cpp =========================//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// FIXME: audit these header files
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Sema/Sema.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::ast_matchers;

using CallBackTy =
    std::function<void(CompilerInstance &, const CountAttributedType *)>;

bool runOnToolAndCheckAttrSourceRange(llvm::Annotations &Code,
                                      bool TestForFallback = false) {
  class RunFnAfterSema : public clang::ASTFrontendAction {
    llvm::Annotations &Code;
    bool TestForFallback;

  public:
    RunFnAfterSema(llvm::Annotations &Code, bool TestForFallback)
        : Code(Code), TestForFallback(TestForFallback) {}
    void EndSourceFile() override {
      auto &S = getCompilerInstance().getSema();
      // Find `TestStruct::ptr`
      auto Matched = match(
          recordDecl(hasName("TestStruct"),
                     hasDescendant(fieldDecl(hasName("ptr")).bind("ptr"))),
          S.getASTContext());
      EXPECT_EQ(Matched.size(), 1u);
      auto *FD = Matched[0].getNodeAs<FieldDecl>("ptr");
      ASSERT_NE(FD, nullptr);

      const auto *CATy = FD->getType()->getAs<CountAttributedType>();

      auto SR = S.BoundsSafetySourceRangeFor(CATy);

      if (TestForFallback) {
        // Make sure for these types of test cases that there are no
        // annotations. The presence of annotations for this type of test would
        // be very misleading because they aren't being checked.
        const auto &Ranges = Code.all_ranges();
        ASSERT_EQ(Ranges.size(), 0U);

        // The fallback is using the SourceRange of the CountExpr.
        ASSERT_EQ(SR, CATy->getCountExpr()->getSourceRange());
        // Don't test for precise column position in this case.
        // The code below doesn't really work correctly when the
        // count expression is itself a macro.
        return;
      }

      // Check Begin
      auto ExpectedBegin = Code.range("attr").Begin;
      auto Begin = SR.getBegin();
      auto BeginSpellingLoc = S.getSourceManager().getSpellingLoc(Begin);
      auto BeginFileOffset =
          S.getSourceManager().getFileOffset(BeginSpellingLoc);
      ASSERT_EQ(BeginFileOffset, ExpectedBegin);
      // Check End
      auto ExpectedEnd = Code.range("attr").End - 1;
      auto End = SR.getEnd();
      auto EndSpellingLoc = S.getSourceManager().getSpellingLoc(End);
      auto EndFileOffset = S.getSourceManager().getFileOffset(EndSpellingLoc);
      ASSERT_EQ(EndFileOffset, ExpectedEnd);

      ASTFrontendAction::EndSourceFile();
    }

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef InFile) override {
      return std::make_unique<ASTConsumer>();
    }
  };
  auto Action = std::make_unique<RunFnAfterSema>(Code, TestForFallback);
  return clang::tooling::runToolOnCodeWithArgs(std::move(Action), Code.code(),
                                               {"-std=c11"}, "test.c");
}

//==============================================================================
// counted_by
//==============================================================================

TEST(GetCountedByAttrSourceRange, GNUAttrNoAffix) {
  llvm::Annotations Code(
      R"C(
#define __counted_by(X) __attribute__((counted_by(X)))
struct TestStruct {
  int count;
  char* __attribute__(($attr[[counted_by(count)]])) ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetCountedByAttrSourceRange, GNUAttrNoAffixArgIsMacro) {
  // For this code no special SourceRange is computed
  // FIXME: This shouldn't use the fallback path.
  llvm::Annotations Code(
      R"C(
#define COUNT count
#define __counted_by(X) __attribute__((counted_by(X)))
struct TestStruct {
  int count;
  char* __attribute__((counted_by(COUNT))) ptr;
};
)C");
  bool result =
      runOnToolAndCheckAttrSourceRange(Code, /*TestForFallback=*/true);
  ASSERT_TRUE(result);
}

TEST(GetCountedByAttrSourceRange, GNUAttrAffix) {
  llvm::Annotations Code(
      R"C(
#define __counted_by(X) __attribute__((__counted_by__(X)))
struct TestStruct {
  int count;
  char* __attribute__(($attr[[counted_by(count)]])) ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetCountedByAttrSourceRange, GNUAttrAffixArgIsMacro) {
  // For this code no special SourceRange is computed
  // FIXME: This shouldn't use the fallback path.
  llvm::Annotations Code(
      R"C(
#define COUNT count
#define __counted_by(X) __attribute__((__counted_by__(X)))
struct TestStruct {
  int count;
  char* __attribute__((counted_by(COUNT))) ptr;
};
)C");
  bool result =
      runOnToolAndCheckAttrSourceRange(Code, /*TestForFallback=*/true);
  ASSERT_TRUE(result);
}

TEST(GetCountedByAttrSourceRange, MacroNoAffixImpl) {
  llvm::Annotations Code(
      R"C(
#define __counted_by(X) __attribute__((counted_by(X)))
struct TestStruct {
  int count;
  char* $attr[[__counted_by(count)]] ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetCountedByAttrSourceRange, MacroAffixImpl) {
  llvm::Annotations Code(
      R"C(
#define __counted_by(X) __attribute__((__counted_by__(X)))
struct TestStruct {
  int count;
  char* $attr[[__counted_by(count)]] ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetCountedByAttrSourceRange, CustomMacro) {
  // For this code no special SourceRange is computed.
  llvm::Annotations Code(
      R"C(
#define custom_cb(X) __attribute__((__counted_by__(X)))
struct TestStruct {
  int count;
  char* custom_cb(count) ptr;
};
)C");
  bool result =
      runOnToolAndCheckAttrSourceRange(Code, /*TestForFallback=*/true);
  ASSERT_TRUE(result);
}

TEST(GetCountedByAttrSourceRange, MacroArgIsMacro) {
  // For this code no special SourceRange is computed
  // FIXME: This shouldn't use the fallback path.
  llvm::Annotations Code(
      R"C(
#define COUNT count
#define __counted_by(X) __attribute__((__counted_by__(X)))
struct TestStruct {
  int count;
  char* __counted_by(COUNT) ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code, true);
  ASSERT_TRUE(result);
}

//==============================================================================
// counted_by_or_null
//==============================================================================

TEST(GetCountedByOrNullAttrSourceRange, GNUAttrNoAffix) {
  llvm::Annotations Code(
      R"C(
#define __counted_by_or_null(X) __attribute__((counted_by_or_null(X)))
struct TestStruct {
  int count;
  char* __attribute__(($attr[[counted_by_or_null(count)]])) ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetCountedByOrNullAttrSourceRange, GNUAttrNoAffixArgIsMacro) {
  // For this code no special SourceRange is computed
  // FIXME: This shouldn't use the fallback path.
  llvm::Annotations Code(
      R"C(
#define COUNT count
#define __counted_by_or_null(X) __attribute__((counted_by_or_null(X)))
struct TestStruct {
  int count;
  char* __attribute__((counted_by_or_null(COUNT))) ptr;
};
)C");
  bool result =
      runOnToolAndCheckAttrSourceRange(Code, /*TestForFallback=*/true);
  ASSERT_TRUE(result);
}

TEST(GetCountedByOrNullAttrSourceRange, GNUAttrAffix) {
  llvm::Annotations Code(
      R"C(
#define __counted_by_or_null(X) __attribute__((__counted_by_or_null__(X)))
struct TestStruct {
  int count;
  char* __attribute__(($attr[[counted_by_or_null(count)]])) ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetCountedByOrNullAttrSourceRange, GNUAttrAffixArgIsMacro) {
  // For this code no special SourceRange is computed
  // FIXME: This shouldn't use the fallback path.
  llvm::Annotations Code(
      R"C(
#define COUNT count
#define __counted_by_or_null(X) __attribute__((__counted_by_or_null__(X)))
struct TestStruct {
  int count;
  char* __attribute__((counted_by_or_null(COUNT))) ptr;
};
)C");
  bool result =
      runOnToolAndCheckAttrSourceRange(Code, /*TestForFallback=*/true);
  ASSERT_TRUE(result);
}

TEST(GetCountedByOrNullAttrSourceRange, MacroNoAffixImpl) {
  llvm::Annotations Code(
      R"C(
#define __counted_by_or_null(X) __attribute__((counted_by_or_null(X)))
struct TestStruct {
  int count;
  char* $attr[[__counted_by_or_null(count)]] ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetCountedByOrNullAttrSourceRange, MacroAffixImpl) {
  llvm::Annotations Code(
      R"C(
#define __counted_by_or_null(X) __attribute__((__counted_by_or_null__(X)))
struct TestStruct {
  int count;
  char* $attr[[__counted_by_or_null(count)]] ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetCountedByOrNullAttrSourceRange, CustomMacro) {
  // For this code no special SourceRange is computed.
  llvm::Annotations Code(
      R"C(
#define custom_cbon(X) __attribute__((__counted_by_or_null__(X)))
struct TestStruct {
  int count;
  char* custom_cbon(count) ptr;
};
)C");
  bool result =
      runOnToolAndCheckAttrSourceRange(Code, /*TestForFallback=*/true);
  ASSERT_TRUE(result);
}

TEST(GetCountedByOrNullAttrSourceRange, MacroArgIsMacro) {
  // For this code no special SourceRange is computed
  // FIXME: This shouldn't use the fallback path.
  llvm::Annotations Code(
      R"C(
#define COUNT count
#define __counted_by_or_null(X) __attribute__((__counted_by_or_null__(X)))
struct TestStruct {
  int count;
  char* __counted_by_or_null(COUNT) ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code, true);
  ASSERT_TRUE(result);
}

//==============================================================================
// sized_by
//==============================================================================

TEST(GetSizedByAttrSourceRange, GNUAttrNoAffix) {
  llvm::Annotations Code(
      R"C(
#define __sized_by(X) __attribute__((sized_by(X)))
struct TestStruct {
  int count;
  char* __attribute__(($attr[[sized_by(count)]])) ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetSizedByAttrSourceRange, GNUAttrNoAffixArgIsMacro) {
  // For this code no special SourceRange is computed
  // FIXME: This shouldn't use the fallback path.
  llvm::Annotations Code(
      R"C(
#define COUNT count
#define __sized_by(X) __attribute__((sized_by(X)))
struct TestStruct {
  int count;
  char* __attribute__((sized_by(COUNT))) ptr;
};
)C");
  bool result =
      runOnToolAndCheckAttrSourceRange(Code, /*TestForFallback=*/true);
  ASSERT_TRUE(result);
}

TEST(GetSizedByAttrSourceRange, GNUAttrAffix) {
  llvm::Annotations Code(
      R"C(
#define __sized_by(X) __attribute__((__sized_by__(X)))
struct TestStruct {
  int count;
  char* __attribute__(($attr[[sized_by(count)]])) ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetSizedByAttrSourceRange, GNUAttrAffixArgIsMacro) {
  // For this code no special SourceRange is computed
  // FIXME: This shouldn't use the fallback path.
  llvm::Annotations Code(
      R"C(
#define COUNT count
#define __sized_by(X) __attribute__((__sized_by__(X)))
struct TestStruct {
  int count;
  char* __attribute__((sized_by(COUNT))) ptr;
};
)C");
  bool result =
      runOnToolAndCheckAttrSourceRange(Code, /*TestForFallback=*/true);
  ASSERT_TRUE(result);
}

TEST(GetSizedByAttrSourceRange, MacroNoAffixImpl) {
  llvm::Annotations Code(
      R"C(
#define __sized_by(X) __attribute__((sized_by(X)))
struct TestStruct {
  int count;
  char* $attr[[__sized_by(count)]] ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetSizedByAttrSourceRange, MacroAffixImpl) {
  llvm::Annotations Code(
      R"C(
#define __sized_by(X) __attribute__((__sized_by__(X)))
struct TestStruct {
  int count;
  char* $attr[[__sized_by(count)]] ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetSizedByAttrSourceRange, CustomMacro) {
  // For this code no special SourceRange is computed.
  llvm::Annotations Code(
      R"C(
#define custom_sb(X) __attribute__((__sized_by__(X)))
struct TestStruct {
  int count;
  char* custom_sb(count) ptr;
};
)C");
  bool result =
      runOnToolAndCheckAttrSourceRange(Code, /*TestForFallback=*/true);
  ASSERT_TRUE(result);
}

TEST(GetSizedByAttrSourceRange, MacroArgIsMacro) {
  // For this code no special SourceRange is computed
  // FIXME: This shouldn't use the fallback path.
  llvm::Annotations Code(
      R"C(
#define COUNT count
#define __sized_by(X) __attribute__((__sized_by__(X)))
struct TestStruct {
  int count;
  char* __sized_by(COUNT) ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code, true);
  ASSERT_TRUE(result);
}

//==============================================================================
// sized_by_or_null
//==============================================================================

TEST(GetSizedByOrNullAttrSourceRange, GNUAttrNoAffix) {
  llvm::Annotations Code(
      R"C(
#define __sized_by_or_null(X) __attribute__((sized_by_or_null(X)))
struct TestStruct {
  int count;
  char* __attribute__(($attr[[sized_by_or_null(count)]])) ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetSizedByOrNullAttrSourceRange, GNUAttrNoAffixArgIsMacro) {
  // For this code no special SourceRange is computed
  // FIXME: This shouldn't use the fallback path.
  llvm::Annotations Code(
      R"C(
#define COUNT count
#define __sized_by_or_null(X) __attribute__((sized_by_or_null(X)))
struct TestStruct {
  int count;
  char* __attribute__((sized_by_or_null(COUNT))) ptr;
};
)C");
  bool result =
      runOnToolAndCheckAttrSourceRange(Code, /*TestForFallback=*/true);
  ASSERT_TRUE(result);
}

TEST(GetSizedByOrNullAttrSourceRange, GNUAttrAffix) {
  llvm::Annotations Code(
      R"C(
#define __sized_by_or_null(X) __attribute__((__sized_by_or_null__(X)))
struct TestStruct {
  int count;
  char* __attribute__(($attr[[sized_by_or_null(count)]])) ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetSizedByOrNullAttrSourceRange, GNUAttrAffixArgIsMacro) {
  // For this code no special SourceRange is computed
  // FIXME: This shouldn't use the fallback path.
  llvm::Annotations Code(
      R"C(
#define COUNT count
#define __sized_by_or_null(X) __attribute__((__sized_by_or_null__(X)))
struct TestStruct {
  int count;
  char* __attribute__((sized_by_or_null(COUNT))) ptr;
};
)C");
  bool result =
      runOnToolAndCheckAttrSourceRange(Code, /*TestForFallback=*/true);
  ASSERT_TRUE(result);
}

TEST(GetSizedByOrNullAttrSourceRange, MacroNoAffixImpl) {
  llvm::Annotations Code(
      R"C(
#define __sized_by_or_null(X) __attribute__((sized_by_or_null(X)))
struct TestStruct {
  int count;
  char* $attr[[__sized_by_or_null(count)]] ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetSizedByOrNullAttrSourceRange, MacroAffixImpl) {
  llvm::Annotations Code(
      R"C(
#define __sized_by_or_null(X) __attribute__((__sized_by_or_null__(X)))
struct TestStruct {
  int count;
  char* $attr[[__sized_by_or_null(count)]] ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code);
  ASSERT_TRUE(result);
}

TEST(GetSizedByOrNullAttrSourceRange, CustomMacro) {
  // For this code no special SourceRange is computed.
  llvm::Annotations Code(
      R"C(
#define custom_sbon(X) __attribute__((__sized_by_or_null__(X)))
struct TestStruct {
  int count;
  char* custom_sbon(count) ptr;
};
)C");
  bool result =
      runOnToolAndCheckAttrSourceRange(Code, /*TestForFallback=*/true);
  ASSERT_TRUE(result);
}

TEST(GetSizedByOrNullAttrSourceRange, MacroArgIsMacro) {
  // For this code no special SourceRange is computed
  // FIXME: This shouldn't use the fallback path.
  llvm::Annotations Code(
      R"C(
#define COUNT count
#define __sized_by_or_null(X) __attribute__((__sized_by_or_null__(X)))
struct TestStruct {
  int count;
  char* __sized_by_or_null(COUNT) ptr;
};
)C");
  bool result = runOnToolAndCheckAttrSourceRange(Code, true);
  ASSERT_TRUE(result);
}
