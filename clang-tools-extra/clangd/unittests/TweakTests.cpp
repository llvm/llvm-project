//===-- TweakTests.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    unsigned Begin = cantFail(positionToOffset(Code.code(), Selection.start));
    unsigned End = cantFail(positionToOffset(Code.code(), Selection.end));
    auto T = prepareTweak(ID, Tweak::Selection(AST, Begin, End));
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
  unsigned Begin = cantFail(positionToOffset(Code.code(), SelectionRng.start));
  unsigned End = cantFail(positionToOffset(Code.code(), SelectionRng.end));
  Tweak::Selection S(AST, Begin, End);

  auto T = prepareTweak(ID, S);
  if (!T)
    return T.takeError();
  auto Replacements = (*T)->apply(S);
  if (!Replacements)
    return Replacements.takeError();
  return applyAllReplacements(Code.code(), *Replacements);
}

void checkTransform(llvm::StringRef ID, llvm::StringRef Input,
                    std::string Output) {
  auto Result = apply(ID, Input);
  ASSERT_TRUE(bool(Result)) << llvm::toString(Result.takeError()) << Input;
  EXPECT_EQ(Output, std::string(*Result)) << Input;
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

  // Available in subexpressions of the condition.
  checkAvailable(ID, R"cpp(
    void test() {
      if(2 + [[2]] + 2) { return 2 + 2 + 2; } else { continue; }
    }
  )cpp");
  // But not as part of the branches.
  checkNotAvailable(ID, R"cpp(
    void test() {
      if(2 + 2 + 2) { return 2 + [[2]] + 2; } else { continue; }
    }
  )cpp");
  // Range covers the "else" token, so available.
  checkAvailable(ID, R"cpp(
    void test() {
      if(2 + 2 + 2) { return 2 + [[2 + 2; } else { continue;]] }
    }
  )cpp");
  // Not available in compound statements in condition.
  checkNotAvailable(ID, R"cpp(
    void test() {
      if([]{return [[true]];}()) { return 2 + 2 + 2; } else { continue; }
    }
  )cpp");
  // Not available if both sides aren't braced.
  checkNotAvailable(ID, R"cpp(
    void test() {
      ^if (1) return; else { return; }
    }
  )cpp");
  // Only one if statement is supported!
  checkNotAvailable(ID, R"cpp(
    [[if(1){}else{}if(2){}else{}]]
  )cpp");
}

TEST(TweakTest, RawStringLiteral) {
  llvm::StringLiteral ID = "RawStringLiteral";

  checkAvailable(ID, R"cpp(
    const char *A = ^"^f^o^o^\^n^";
    const char *B = R"(multi )" ^"token " "str\ning";
  )cpp");

  checkNotAvailable(ID, R"cpp(
    const char *A = ^"f^o^o^o^"; // no chars need escaping
    const char *B = R"(multi )" ^"token " u8"str\ning"; // not all ascii
    const char *C = ^R^"^(^multi )" "token " "str\ning"; // chunk is raw
    const char *D = ^"token\n" __FILE__; // chunk is macro expansion
    const char *E = ^"a\r\n"; // contains forbidden escape character
  )cpp");

  const char *Input = R"cpp(
    const char *X = R"(multi
token)" "\nst^ring\n" "literal";
    }
  )cpp";
  const char *Output = R"cpp(
    const char *X = R"(multi
token
string
literal)";
    }
  )cpp";
  checkTransform(ID, Input, Output);
}

} // namespace
} // namespace clangd
} // namespace clang
