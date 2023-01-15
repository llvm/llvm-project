//===--- TweakTesting.h - Test helpers for refactoring actions ---*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_UNITTESTS_TWEAKS_TWEAKTESTING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_UNITTESTS_TWEAKS_TWEAKTESTING_H

#include "ParsedAST.h"
#include "index/Index.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>

namespace clang {
namespace clangd {

// Fixture base for testing tweaks. Intended to be subclassed for each tweak.
//
// Usage:
// TWEAK_TEST(ExpandDeducedType);
//
// TEST_F(ExpandDeducedTypeTest, ShortensTypes) {
//   Header = R"cpp(
//     namespace foo { template<typename> class X{}; }
//     using namespace foo;
//   )cpp";
//   Context = Function;
//   EXPECT_THAT(apply("[[auto]] X = foo<int>();"),
//               "foo<int> X = foo<int();");
//   EXPECT_AVAILABLE("^a^u^t^o^ X = foo<int>();");
//   EXPECT_UNAVAILABLE("auto ^X^ = ^foo<int>();");
// }
class TweakTest : public ::testing::Test {
  const char *TweakID;

public:
  // Inputs are wrapped in file boilerplate before attempting to apply a tweak.
  // Context describes the type of boilerplate.
  enum CodeContext {
    // Code snippet is placed directly into the source file. e.g. a declaration.
    File,
    // Snippet will appear within a function body. e.g. a statement.
    Function,
    // Snippet is an expression.
    Expression,
  };

  // Mapping from file name to contents.
  llvm::StringMap<std::string> ExtraFiles;

protected:
  TweakTest(const char *TweakID) : TweakID(TweakID) {}

  // Contents of a header file to be implicitly included.
  // This typically contains declarations that will be used for a set of related
  // testcases.
  std::string Header;

  llvm::StringRef FileName = "TestTU.cpp";

  // Extra flags passed to the compilation in apply().
  std::vector<std::string> ExtraArgs;

  // Context in which snippets of code should be placed to run tweaks.
  CodeContext Context = File;

  // Index to be passed into Tweak::Selection.
  std::unique_ptr<const SymbolIndex> Index = nullptr;

  // Apply the current tweak to the range (or point) in MarkedCode.
  // MarkedCode will be wrapped according to the Context.
  //  - if the tweak produces edits, returns the edited code (without markings)
  //    for the main file.
  //    Populates \p EditedFiles if there were changes to other files whenever
  //    it is non-null. It is a mapping from absolute path of the edited file to
  //    its new contents. Passing a nullptr to \p EditedFiles when there are
  //    changes, will result in a failure.
  //    The context added to MarkedCode will be stripped away before returning,
  //    unless the tweak edited it.
  //  - if the tweak produces a message, returns "message:\n<message>"
  //  - if prepare() returns false, returns "unavailable"
  //  - if apply() returns an error, returns "fail: <message>"
  std::string apply(llvm::StringRef MarkedCode,
                    llvm::StringMap<std::string> *EditedFiles = nullptr) const;

  // Helpers for EXPECT_AVAILABLE/EXPECT_UNAVAILABLE macros.
  using WrappedAST = std::pair<ParsedAST, /*WrappingOffset*/ unsigned>;
  WrappedAST build(llvm::StringRef) const;
  bool isAvailable(WrappedAST &, llvm::Annotations::Range) const;
  // Return code re-decorated with a single point/range.
  static std::string decorate(llvm::StringRef, unsigned);
  static std::string decorate(llvm::StringRef, llvm::Annotations::Range);
};

MATCHER_P2(FileWithContents, FileName, Contents, "") {
  return arg.first() == FileName && arg.second == Contents;
}

#define TWEAK_TEST(TweakID)                                                    \
  class TweakID##Test : public ::clang::clangd::TweakTest {                    \
  protected:                                                                   \
    TweakID##Test() : TweakTest(#TweakID) {}                                   \
  }

#define EXPECT_AVAILABLE_(MarkedCode, Available)                               \
  do {                                                                         \
    llvm::Annotations A{llvm::StringRef(MarkedCode)};                          \
    auto AST = build(A.code());                                                \
    assert(!A.points().empty() || !A.ranges().empty());                        \
    for (const auto &P : A.points())                                           \
      EXPECT_EQ(Available, isAvailable(AST, {P, P})) << decorate(A.code(), P); \
    for (const auto &R : A.ranges())                                           \
      EXPECT_EQ(Available, isAvailable(AST, R)) << decorate(A.code(), R);      \
  } while (0)
#define EXPECT_AVAILABLE(MarkedCode) EXPECT_AVAILABLE_(MarkedCode, true)
#define EXPECT_UNAVAILABLE(MarkedCode) EXPECT_AVAILABLE_(MarkedCode, false)

} // namespace clangd
} // namespace clang

#endif
