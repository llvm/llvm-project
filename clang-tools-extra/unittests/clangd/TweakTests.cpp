//===-- TweakTests.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "SourceCode.h"
#include "TestTU.h"
#include "refactor/Tweak.h"
#include "clang/AST/Expr.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>

using llvm::Failed;
using llvm::HasValue;
using llvm::Succeeded;

namespace clang {
namespace clangd {
namespace {

std::string markRange(llvm::StringRef Code, Range R) {
  size_t Begin = llvm::cantFail(positionToOffset(Code, R.start));
  size_t End = llvm::cantFail(positionToOffset(Code, R.end));
  assert(Begin <= End);
  if (Begin == End) // Mark a single point.
    return (Code.substr(0, Begin) + "^" + Code.substr(Begin)).str();
  // Mark a range.
  return (Code.substr(0, Begin) + "[[" + Code.substr(Begin, End - Begin) +
          "]]" + Code.substr(End))
      .str();
}

void checkAvailable(StringRef ID, llvm::StringRef Input, bool Available) {
  Annotations Code(Input);
  ASSERT_TRUE(0 < Code.points().size() || 0 < Code.ranges().size())
      << "no points of interest specified";
  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.Code = Code.code();

  ParsedAST AST = TU.build();

  auto CheckOver = [&](Range Selection) {
    auto CursorLoc = llvm::cantFail(sourceLocationInMainFile(
        AST.getASTContext().getSourceManager(), Selection.start));
    auto T = prepareTweak(ID, Tweak::Selection{Code.code(), AST, CursorLoc});
    if (Available)
      EXPECT_THAT_EXPECTED(T, Succeeded())
          << "code is " << markRange(Code.code(), Selection);
    else
      EXPECT_THAT_EXPECTED(T, Failed())
          << "code is " << markRange(Code.code(), Selection);
  };
  for (auto P : Code.points())
    CheckOver(Range{P, P});
  for (auto R : Code.ranges())
    CheckOver(R);
}

/// Checks action is available at every point and range marked in \p Input.
void checkAvailable(StringRef ID, llvm::StringRef Input) {
  return checkAvailable(ID, Input, /*Available=*/true);
}

/// Same as checkAvailable, but checks the action is not available.
void checkNotAvailable(StringRef ID, llvm::StringRef Input) {
  return checkAvailable(ID, Input, /*Available=*/false);
}
llvm::Expected<std::string> apply(StringRef ID, llvm::StringRef Input) {
  Annotations Code(Input);
  Range SelectionRng;
  if (Code.points().size() != 0) {
    assert(Code.ranges().size() == 0 &&
           "both a cursor point and a selection range were specified");
    SelectionRng = Range{Code.point(), Code.point()};
  } else {
    SelectionRng = Code.range();
  }
  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.Code = Code.code();

  ParsedAST AST = TU.build();
  auto CursorLoc = llvm::cantFail(sourceLocationInMainFile(
      AST.getASTContext().getSourceManager(), SelectionRng.start));
  Tweak::Selection S = {Code.code(), AST, CursorLoc};

  auto T = prepareTweak(ID, S);
  if (!T)
    return T.takeError();
  auto Replacements = (*T)->apply(S);
  if (!Replacements)
    return Replacements.takeError();
  return applyAllReplacements(Code.code(), *Replacements);
}

void checkTransform(llvm::StringRef ID, llvm::StringRef Input,
                    llvm::StringRef Output) {
  EXPECT_THAT_EXPECTED(apply(ID, Input), HasValue(Output))
      << "action id is" << ID;
}

TEST(TweakTest, SwapIfBranches) {
  llvm::StringLiteral ID = "SwapIfBranches";

  checkAvailable(ID, R"cpp(
    void test() {
      ^i^f^^(^t^r^u^e^) { return 100; } ^e^l^s^e^ { continue; }
    }
  )cpp");

  checkNotAvailable(ID, R"cpp(
    void test() {
      if (true) {^return ^100;^ } else { ^continue^;^ }
    }
  )cpp");

  llvm::StringLiteral Input = R"cpp(
    void test() {
      ^if (true) { return 100; } else { continue; }
    }
  )cpp";
  llvm::StringLiteral Output = R"cpp(
    void test() {
      if (true) { continue; } else { return 100; }
    }
  )cpp";
  checkTransform(ID, Input, Output);

  Input = R"cpp(
    void test() {
      ^if () { return 100; } else { continue; }
    }
  )cpp";
  Output = R"cpp(
    void test() {
      if () { continue; } else { return 100; }
    }
  )cpp";
  checkTransform(ID, Input, Output);
}

} // namespace
} // namespace clangd
} // namespace clang
