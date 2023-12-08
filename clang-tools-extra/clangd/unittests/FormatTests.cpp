//===-- FormatTests.cpp - Automatic code formatting tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Format.h"
#include "Annotations.h"
#include "SourceCode.h"
#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

std::string afterTyped(llvm::StringRef CodeWithCursor, llvm::StringRef Typed,
                       clang::format::FormatStyle Style) {
  Annotations Code(CodeWithCursor);
  unsigned Cursor = llvm::cantFail(positionToOffset(Code.code(), Code.point()));
  auto Changes = formatIncremental(Code.code(), Cursor, Typed, Style);
  tooling::Replacements Merged;
  for (const auto& R : Changes)
    if (llvm::Error E = Merged.add(R))
      ADD_FAILURE() << llvm::toString(std::move(E));
  auto NewCode = tooling::applyAllReplacements(Code.code(), Merged);
  EXPECT_TRUE(bool(NewCode))
      << "Bad replacements: " << llvm::toString(NewCode.takeError());
  NewCode->insert(transformCursorPosition(Cursor, Changes), "^");
  return *NewCode;
}

// We can't pass raw strings directly to EXPECT_EQ because of gcc bugs.
void expectAfterNewline(const char *Before, const char *After,
                        format::FormatStyle Style = format::getGoogleStyle(
                            format::FormatStyle::LK_Cpp)) {
  EXPECT_EQ(After, afterTyped(Before, "\n", Style)) << Before;
}
void expectAfter(const char *Typed, const char *Before, const char *After,
                 format::FormatStyle Style =
                     format::getGoogleStyle(format::FormatStyle::LK_Cpp)) {
  EXPECT_EQ(After, afterTyped(Before, Typed, Style)) << Before;
}

TEST(FormatIncremental, SplitComment) {
  expectAfterNewline(R"cpp(
// this comment was
^split
)cpp",
   R"cpp(
// this comment was
// ^split
)cpp");

  expectAfterNewline(R"cpp(
// trailing whitespace is not a split
^   
)cpp",
   R"cpp(
// trailing whitespace is not a split
^
)cpp");

  expectAfterNewline(R"cpp(
// splitting a
^
// multiline comment
)cpp",
                     R"cpp(
// splitting a
// ^
// multiline comment
)cpp");

  expectAfterNewline(R"cpp(
// extra   
    ^     whitespace
)cpp",
   R"cpp(
// extra
// ^whitespace
)cpp");

  expectAfterNewline(R"cpp(
/// triple
^slash
)cpp",
   R"cpp(
/// triple
/// ^slash
)cpp");

  expectAfterNewline(R"cpp(
/// editor continuation
//^
)cpp",
   R"cpp(
/// editor continuation
/// ^
)cpp");

  expectAfterNewline(R"cpp(
// break before
^ // slashes
)cpp",
   R"cpp(
// break before
^// slashes
)cpp");


  expectAfterNewline(R"cpp(
int x;  // aligned
^comment
)cpp",
   R"cpp(
int x;  // aligned
        // ^comment
)cpp");

  // Fixed bug: the second line of the aligned comment shouldn't be "attached"
  // to the cursor and outdented.
  expectAfterNewline(R"cpp(
void foo() {
  if (x)
    return; // All spelled tokens are accounted for.
            // that takes two lines
            ^
}
)cpp",
                     R"cpp(
void foo() {
  if (x)
    return;  // All spelled tokens are accounted for.
             // that takes two lines
  ^
}
)cpp");

  // Handle tab character in leading indentation
  format::FormatStyle TabStyle =
      format::getGoogleStyle(format::FormatStyle::LK_Cpp);
  TabStyle.UseTab = format::FormatStyle::UT_Always;
  TabStyle.TabWidth = 4;
  TabStyle.IndentWidth = 4;
  // Do not use raw strings, otherwise '\t' will be interpreted literally.
  expectAfterNewline("void foo() {\n\t// this comment was\n^split\n}\n",
                     "void foo() {\n\t// this comment was\n\t// ^split\n}\n",
                     TabStyle);
}

TEST(FormatIncremental, Indentation) {
  expectAfterNewline(R"cpp(
void foo() {
  if (bar)
^
)cpp",
   R"cpp(
void foo() {
  if (bar)
    ^
)cpp");

  expectAfterNewline(R"cpp(
void foo() {
  bar(baz(
^
)cpp",
   R"cpp(
void foo() {
  bar(baz(
      ^
)cpp");

  expectAfterNewline(R"cpp(
void foo() {
^}
)cpp",
   R"cpp(
void foo() {
  ^
}
)cpp");

  expectAfterNewline(R"cpp(
class X {
protected:
^
)cpp",
   R"cpp(
class X {
 protected:
  ^
)cpp");

// Mismatched brackets (1)
  expectAfterNewline(R"cpp(
void foo() {
  foo{bar(
^}
}
)cpp",
   R"cpp(
void foo() {
  foo {
    bar(
        ^}
}
)cpp");
// Mismatched brackets (2)
  expectAfterNewline(R"cpp(
void foo() {
  foo{bar(
^text}
}
)cpp",
   R"cpp(
void foo() {
  foo {
    bar(
        ^text}
}
)cpp");
// Matched brackets
  expectAfterNewline(R"cpp(
void foo() {
  foo{bar(
^)
}
)cpp",
   R"cpp(
void foo() {
  foo {
    bar(
        ^)
}
)cpp");
}

TEST(FormatIncremental, FormatPreviousLine) {
  expectAfterNewline(R"cpp(
void foo() {
   untouched( );
int x=2;
^
)cpp",
                     R"cpp(
void foo() {
   untouched( );
   int x = 2;
   ^
)cpp");

  expectAfterNewline(R"cpp(
int x=untouched( );
auto L = []{return;return;};
^
)cpp",
   R"cpp(
int x=untouched( );
auto L = [] {
  return;
  return;
};
^
)cpp");
}

TEST(FormatIncremental, Annoyances) {
  // Don't remove newlines the user typed!
  expectAfterNewline(R"cpp(
int x(){


^
}
)cpp",
   R"cpp(
int x(){


  ^
}
)cpp");
  // FIXME: we should not remove newlines here, either.
  expectAfterNewline(R"cpp(
class x{
 public:

^
}
)cpp",
   R"cpp(
class x{
 public:
  ^
}
)cpp");
}

TEST(FormatIncremental, FormatBrace) {
  expectAfter("}", R"cpp(
vector<int> x= {
  1,
  2,
  3}^
)cpp",
              R"cpp(
vector<int> x = {1, 2, 3}^
)cpp");
}

} // namespace
} // namespace clangd
} // namespace clang
