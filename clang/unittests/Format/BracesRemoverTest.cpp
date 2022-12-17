//===- unittest/Format/BracesRemoverTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"

#include "../Tooling/ReplacementTest.h"
#include "FormatTestUtils.h"

#define DEBUG_TYPE "braces-remover-test"

namespace clang {
namespace format {
namespace {

// TODO:
// Refactor the class declaration, which is copied from BracesInserterTest.cpp.
class BracesRemoverTest : public ::testing::Test {
protected:
  std::string format(llvm::StringRef Code, const FormatStyle &Style,
                     const std::vector<tooling::Range> &Ranges) {
    LLVM_DEBUG(llvm::errs() << "---\n");
    LLVM_DEBUG(llvm::errs() << Code << "\n\n");
    auto NonEmptyRanges = Ranges;
    if (Ranges.empty())
      NonEmptyRanges = {1, tooling::Range(0, Code.size())};
    FormattingAttemptStatus Status;
    tooling::Replacements Replaces =
        reformat(Style, Code, NonEmptyRanges, "<stdin>", &Status);
    EXPECT_EQ(true, Status.FormatComplete) << Code << "\n\n";
    ReplacementCount = Replaces.size();
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    LLVM_DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  void _verifyFormat(const char *File, int Line, llvm::StringRef Expected,
                     llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle(),
                     const std::vector<tooling::Range> &Ranges = {}) {
    testing::ScopedTrace t(File, Line, ::testing::Message() << Code.str());
    EXPECT_EQ(Expected.str(), format(Expected, Style, Ranges))
        << "Expected code is not stable";
    EXPECT_EQ(Expected.str(), format(Code, Style, Ranges));
    if (Style.Language == FormatStyle::LK_Cpp && Ranges.empty()) {
      // Objective-C++ is a superset of C++, so everything checked for C++
      // needs to be checked for Objective-C++ as well.
      FormatStyle ObjCStyle = Style;
      ObjCStyle.Language = FormatStyle::LK_ObjC;
      EXPECT_EQ(Expected.str(), format(test::messUp(Code), ObjCStyle, Ranges));
    }
  }

  void _verifyFormat(const char *File, int Line, llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle(),
                     const std::vector<tooling::Range> &Ranges = {}) {
    _verifyFormat(File, Line, Code, Code, Style, Ranges);
  }

  int ReplacementCount;
};

#define verifyFormat(...) _verifyFormat(__FILE__, __LINE__, __VA_ARGS__)

TEST_F(BracesRemoverTest, RemoveBraces) {
  FormatStyle Style = getLLVMStyle();
  Style.RemoveBracesLLVM = true;

  // The following test cases are fully-braced versions of the examples at
  // "llvm.org/docs/CodingStandards.html#don-t-use-braces-on-simple-single-
  // statement-bodies-of-if-else-loop-statements".

  // Omit the braces since the body is simple and clearly associated with the
  // `if`.
  verifyFormat("if (isa<FunctionDecl>(D))\n"
               "  handleFunctionDecl(D);\n"
               "else if (isa<VarDecl>(D))\n"
               "  handleVarDecl(D);",
               "if (isa<FunctionDecl>(D)) {\n"
               "  handleFunctionDecl(D);\n"
               "} else if (isa<VarDecl>(D)) {\n"
               "  handleVarDecl(D);\n"
               "}",
               Style);

  // Here we document the condition itself and not the body.
  verifyFormat("if (isa<VarDecl>(D)) {\n"
               "  // It is necessary that we explain the situation with this\n"
               "  // surprisingly long comment, so it would be unclear\n"
               "  // without the braces whether the following statement is in\n"
               "  // the scope of the `if`.\n"
               "  // Because the condition is documented, we can't really\n"
               "  // hoist this comment that applies to the body above the\n"
               "  // `if`.\n"
               "  handleOtherDecl(D);\n"
               "}",
               Style);

  // Use braces on the outer `if` to avoid a potential dangling `else`
  // situation.
  verifyFormat("if (isa<VarDecl>(D)) {\n"
               "  if (shouldProcessAttr(A))\n"
               "    handleAttr(A);\n"
               "}",
               "if (isa<VarDecl>(D)) {\n"
               "  if (shouldProcessAttr(A)) {\n"
               "    handleAttr(A);\n"
               "  }\n"
               "}",
               Style);

  // Use braces for the `if` block to keep it uniform with the `else` block.
  verifyFormat("if (isa<FunctionDecl>(D)) {\n"
               "  handleFunctionDecl(D);\n"
               "} else {\n"
               "  // In this `else` case, it is necessary that we explain the\n"
               "  // situation with this surprisingly long comment, so it\n"
               "  // would be unclear without the braces whether the\n"
               "  // following statement is in the scope of the `if`.\n"
               "  handleOtherDecl(D);\n"
               "}",
               Style);

  // This should also omit braces. The `for` loop contains only a single
  // statement, so it shouldn't have braces.  The `if` also only contains a
  // single simple statement (the `for` loop), so it also should omit braces.
  verifyFormat("if (isa<FunctionDecl>(D))\n"
               "  for (auto *A : D.attrs())\n"
               "    handleAttr(A);",
               "if (isa<FunctionDecl>(D)) {\n"
               "  for (auto *A : D.attrs()) {\n"
               "    handleAttr(A);\n"
               "  }\n"
               "}",
               Style);

  // Use braces for a `do-while` loop and its enclosing statement.
  verifyFormat("if (Tok->is(tok::l_brace)) {\n"
               "  do {\n"
               "    Tok = Tok->Next;\n"
               "  } while (Tok);\n"
               "}",
               Style);

  // Use braces for the outer `if` since the nested `for` is braced.
  verifyFormat("if (isa<FunctionDecl>(D)) {\n"
               "  for (auto *A : D.attrs()) {\n"
               "    // In this `for` loop body, it is necessary that we\n"
               "    // explain the situation with this surprisingly long\n"
               "    // comment, forcing braces on the `for` block.\n"
               "    handleAttr(A);\n"
               "  }\n"
               "}",
               Style);

  // Use braces on the outer block because there are more than two levels of
  // nesting.
  verifyFormat("if (isa<FunctionDecl>(D)) {\n"
               "  for (auto *A : D.attrs())\n"
               "    for (ssize_t i : llvm::seq<ssize_t>(count))\n"
               "      handleAttrOnDecl(D, A, i);\n"
               "}",
               "if (isa<FunctionDecl>(D)) {\n"
               "  for (auto *A : D.attrs()) {\n"
               "    for (ssize_t i : llvm::seq<ssize_t>(count)) {\n"
               "      handleAttrOnDecl(D, A, i);\n"
               "    }\n"
               "  }\n"
               "}",
               Style);

  // Use braces on the outer block because of a nested `if`; otherwise the
  // compiler would warn: `add explicit braces to avoid dangling else`
  verifyFormat("if (auto *D = dyn_cast<FunctionDecl>(D)) {\n"
               "  if (shouldProcess(D))\n"
               "    handleVarDecl(D);\n"
               "  else\n"
               "    markAsIgnored(D);\n"
               "}",
               "if (auto *D = dyn_cast<FunctionDecl>(D)) {\n"
               "  if (shouldProcess(D)) {\n"
               "    handleVarDecl(D);\n"
               "  } else {\n"
               "    markAsIgnored(D);\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("// clang-format off\n"
               "// comment\n"
               "while (i > 0) { --i; }\n"
               "// clang-format on\n"
               "while (j < 0)\n"
               "  ++j;",
               "// clang-format off\n"
               "// comment\n"
               "while (i > 0) { --i; }\n"
               "// clang-format on\n"
               "while (j < 0) { ++j; }",
               Style);

  verifyFormat("for (;;) {\n"
               "  for (;;)\n"
               "    for (;;)\n"
               "      a;\n"
               "}",
               "for (;;) {\n"
               "  for (;;) {\n"
               "    for (;;) {\n"
               "      a;\n"
               "    }\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "  b; // comment\n"
               "else if (c)\n"
               "  d; /* comment */\n"
               "else\n"
               "  e;",
               "if (a) {\n"
               "  b; // comment\n"
               "} else if (c) {\n"
               "  d; /* comment */\n"
               "} else {\n"
               "  e;\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  b;\n"
               "  c;\n"
               "} else if (d) {\n"
               "  e;\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "#undef NDEBUG\n"
               "  b;\n"
               "} else {\n"
               "  c;\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  // comment\n"
               "} else if (b) {\n"
               "  c;\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  b;\n"
               "} else {\n"
               "  { c; }\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  if (b) // comment\n"
               "    c;\n"
               "} else if (d) {\n"
               "  e;\n"
               "}",
               "if (a) {\n"
               "  if (b) { // comment\n"
               "    c;\n"
               "  }\n"
               "} else if (d) {\n"
               "  e;\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  if (b) {\n"
               "    c;\n"
               "    // comment\n"
               "  } else if (d) {\n"
               "    e;\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  if (b)\n"
               "    c;\n"
               "}",
               "if (a) {\n"
               "  if (b) {\n"
               "    c;\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "  if (b)\n"
               "    c;\n"
               "  else\n"
               "    d;\n"
               "else\n"
               "  e;",
               "if (a) {\n"
               "  if (b) {\n"
               "    c;\n"
               "  } else {\n"
               "    d;\n"
               "  }\n"
               "} else {\n"
               "  e;\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  // comment\n"
               "  if (b)\n"
               "    c;\n"
               "  else if (d)\n"
               "    e;\n"
               "} else {\n"
               "  g;\n"
               "}",
               "if (a) {\n"
               "  // comment\n"
               "  if (b) {\n"
               "    c;\n"
               "  } else if (d) {\n"
               "    e;\n"
               "  }\n"
               "} else {\n"
               "  g;\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "  b;\n"
               "else if (c)\n"
               "  d;\n"
               "else\n"
               "  e;",
               "if (a) {\n"
               "  b;\n"
               "} else {\n"
               "  if (c) {\n"
               "    d;\n"
               "  } else {\n"
               "    e;\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  if (b)\n"
               "    c;\n"
               "  else if (d)\n"
               "    e;\n"
               "} else {\n"
               "  g;\n"
               "}",
               "if (a) {\n"
               "  if (b)\n"
               "    c;\n"
               "  else {\n"
               "    if (d)\n"
               "      e;\n"
               "  }\n"
               "} else {\n"
               "  g;\n"
               "}",
               Style);

  verifyFormat("if (isa<VarDecl>(D)) {\n"
               "  for (auto *A : D.attrs())\n"
               "    if (shouldProcessAttr(A))\n"
               "      handleAttr(A);\n"
               "}",
               "if (isa<VarDecl>(D)) {\n"
               "  for (auto *A : D.attrs()) {\n"
               "    if (shouldProcessAttr(A)) {\n"
               "      handleAttr(A);\n"
               "    }\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("do {\n"
               "  ++I;\n"
               "} while (hasMore() && Filter(*I));",
               "do { ++I; } while (hasMore() && Filter(*I));", Style);

  verifyFormat("if (a)\n"
               "  if (b)\n"
               "    c;\n"
               "  else {\n"
               "    if (d)\n"
               "      e;\n"
               "  }\n"
               "else\n"
               "  f;",
               Style);

  verifyFormat("if (a)\n"
               "  if (b)\n"
               "    c;\n"
               "  else {\n"
               "    if (d)\n"
               "      e;\n"
               "    else if (f)\n"
               "      g;\n"
               "  }\n"
               "else\n"
               "  h;",
               Style);

  verifyFormat("if (a) {\n"
               "  b;\n"
               "} else if (c) {\n"
               "  d;\n"
               "  e;\n"
               "}",
               "if (a) {\n"
               "  b;\n"
               "} else {\n"
               "  if (c) {\n"
               "    d;\n"
               "    e;\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  b;\n"
               "  c;\n"
               "} else if (d) {\n"
               "  e;\n"
               "  f;\n"
               "}",
               "if (a) {\n"
               "  b;\n"
               "  c;\n"
               "} else {\n"
               "  if (d) {\n"
               "    e;\n"
               "    f;\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  b;\n"
               "} else if (c) {\n"
               "  d;\n"
               "} else {\n"
               "  e;\n"
               "  f;\n"
               "}",
               "if (a) {\n"
               "  b;\n"
               "} else {\n"
               "  if (c) {\n"
               "    d;\n"
               "  } else {\n"
               "    e;\n"
               "    f;\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  b;\n"
               "} else if (c) {\n"
               "  d;\n"
               "} else if (e) {\n"
               "  f;\n"
               "  g;\n"
               "}",
               "if (a) {\n"
               "  b;\n"
               "} else {\n"
               "  if (c) {\n"
               "    d;\n"
               "  } else if (e) {\n"
               "    f;\n"
               "    g;\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  if (b)\n"
               "    c;\n"
               "  else if (d) {\n"
               "    e;\n"
               "    f;\n"
               "  }\n"
               "} else {\n"
               "  g;\n"
               "}",
               "if (a) {\n"
               "  if (b)\n"
               "    c;\n"
               "  else {\n"
               "    if (d) {\n"
               "      e;\n"
               "      f;\n"
               "    }\n"
               "  }\n"
               "} else {\n"
               "  g;\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "  if (b)\n"
               "    c;\n"
               "  else {\n"
               "    if (d) {\n"
               "      e;\n"
               "      f;\n"
               "    }\n"
               "  }\n"
               "else\n"
               "  g;",
               Style);

  verifyFormat("if (a) {\n"
               "  b;\n"
               "  c;\n"
               "} else { // comment\n"
               "  if (d) {\n"
               "    e;\n"
               "    f;\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "  b;\n"
               "else if (c)\n"
               "  while (d)\n"
               "    e;\n"
               "// comment",
               "if (a)\n"
               "{\n"
               "  b;\n"
               "} else if (c) {\n"
               "  while (d) {\n"
               "    e;\n"
               "  }\n"
               "}\n"
               "// comment",
               Style);

  verifyFormat("if (a) {\n"
               "  b;\n"
               "} else if (c) {\n"
               "  d;\n"
               "} else {\n"
               "  e;\n"
               "  g;\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  b;\n"
               "} else if (c) {\n"
               "  d;\n"
               "} else {\n"
               "  e;\n"
               "} // comment",
               Style);

  verifyFormat("int abs = [](int i) {\n"
               "  if (i >= 0)\n"
               "    return i;\n"
               "  return -i;\n"
               "};",
               "int abs = [](int i) {\n"
               "  if (i >= 0) {\n"
               "    return i;\n"
               "  }\n"
               "  return -i;\n"
               "};",
               Style);

  verifyFormat("if (a)\n"
               "  foo();\n"
               "else\n"
               "  bar();",
               "if (a)\n"
               "{\n"
               "  foo();\n"
               "}\n"
               "else\n"
               "{\n"
               "  bar();\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "  foo();\n"
               "// comment\n"
               "else\n"
               "  bar();",
               "if (a) {\n"
               "  foo();\n"
               "}\n"
               "// comment\n"
               "else {\n"
               "  bar();\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  if (b)\n"
               "    c = 1; // comment\n"
               "}",
               "if (a) {\n"
               "  if (b) {\n"
               "    c = 1; // comment\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("if (a) // comment\n"
               "  b = 1;",
               "if (a) // comment\n"
               "{\n"
               "  b = 1;\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "Label:\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "Label:\n"
               "  f();\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  f();\n"
               "Label:\n"
               "}",
               Style);

  verifyFormat("if consteval {\n"
               "  f();\n"
               "} else {\n"
               "  g();\n"
               "}",
               Style);

  verifyFormat("if not consteval {\n"
               "  f();\n"
               "} else if (a) {\n"
               "  g();\n"
               "}",
               Style);

  verifyFormat("if !consteval {\n"
               "  g();\n"
               "}",
               Style);

  verifyFormat("while (0)\n"
               "  if (a)\n"
               "    return b;\n"
               "return a;",
               "while (0) {\n"
               "  if (a) {\n"
               "    return b;\n"
               "}}\n"
               "return a;",
               Style);

  Style.ColumnLimit = 65;
  verifyFormat("if (condition) {\n"
               "  ff(Indices,\n"
               "     [&](unsigned LHSI, unsigned RHSI) { return true; });\n"
               "} else {\n"
               "  ff(Indices,\n"
               "     [&](unsigned LHSI, unsigned RHSI) { return true; });\n"
               "}",
               Style);

  Style.ColumnLimit = 20;

  verifyFormat("int i;\n"
               "#define FOO(a, b)  \\\n"
               "  while (a) {      \\\n"
               "    b;             \\\n"
               "  }",
               Style);

  verifyFormat("int ab = [](int i) {\n"
               "  if (i > 0) {\n"
               "    i = 12345678 -\n"
               "        i;\n"
               "  }\n"
               "  return i;\n"
               "};",
               Style);

  verifyFormat("if (a) {\n"
               "  b = c + // 1 -\n"
               "      d;\n"
               "}",
               Style);

  verifyFormat("if (a) {\n"
               "  b = c >= 0 ? d\n"
               "             : e;\n"
               "}",
               "if (a) {\n"
               "  b = c >= 0 ? d : e;\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "  b = c > 0 ? d : e;",
               "if (a) {\n"
               "  b = c > 0 ? d : e;\n"
               "}",
               Style);

  verifyFormat("if (-b >=\n"
               "    c) { // Keep.\n"
               "  foo();\n"
               "} else {\n"
               "  bar();\n"
               "}",
               "if (-b >= c) { // Keep.\n"
               "  foo();\n"
               "} else {\n"
               "  bar();\n"
               "}",
               Style);

  verifyFormat("if (a) /* Remove. */\n"
               "  f();\n"
               "else\n"
               "  g();",
               "if (a) <% /* Remove. */\n"
               "  f();\n"
               "%> else <%\n"
               "  g();\n"
               "%>",
               Style);

  verifyFormat("while (\n"
               "    !i--) <% // Keep.\n"
               "  foo();\n"
               "%>",
               "while (!i--) <% // Keep.\n"
               "  foo();\n"
               "%>",
               Style);

  verifyFormat("for (int &i : chars)\n"
               "  ++i;",
               "for (int &i :\n"
               "     chars) {\n"
               "  ++i;\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "  b;\n"
               "else if (c) {\n"
               "  d;\n"
               "  e;\n"
               "} else\n"
               "  f = g(foo, bar,\n"
               "        baz);",
               "if (a)\n"
               "  b;\n"
               "else {\n"
               "  if (c) {\n"
               "    d;\n"
               "    e;\n"
               "  } else\n"
               "    f = g(foo, bar, baz);\n"
               "}",
               Style);

  Style.ColumnLimit = 0;
  verifyFormat("if (a)\n"
               "  b234567890223456789032345678904234567890 = "
               "c234567890223456789032345678904234567890;",
               "if (a) {\n"
               "  b234567890223456789032345678904234567890 = "
               "c234567890223456789032345678904234567890;\n"
               "}",
               Style);

  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_Always;
  Style.BraceWrapping.BeforeElse = true;

  Style.ColumnLimit = 65;

  verifyFormat("if (condition)\n"
               "{\n"
               "  ff(Indices,\n"
               "     [&](unsigned LHSI, unsigned RHSI) { return true; });\n"
               "}\n"
               "else\n"
               "{\n"
               "  ff(Indices,\n"
               "     [&](unsigned LHSI, unsigned RHSI) { return true; });\n"
               "}",
               "if (condition) {\n"
               "  ff(Indices,\n"
               "     [&](unsigned LHSI, unsigned RHSI) { return true; });\n"
               "} else {\n"
               "  ff(Indices,\n"
               "     [&](unsigned LHSI, unsigned RHSI) { return true; });\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "{ //\n"
               "  foo();\n"
               "}",
               "if (a) { //\n"
               "  foo();\n"
               "}",
               Style);

  verifyFormat("if (a) // comment\n"
               "  b = 1;",
               "if (a) // comment\n"
               "{\n"
               "  b = 1;\n"
               "}",
               Style);

  Style.ColumnLimit = 20;

  verifyFormat("int ab = [](int i) {\n"
               "  if (i > 0)\n"
               "  {\n"
               "    i = 12345678 -\n"
               "        i;\n"
               "  }\n"
               "  return i;\n"
               "};",
               "int ab = [](int i) {\n"
               "  if (i > 0) {\n"
               "    i = 12345678 -\n"
               "        i;\n"
               "  }\n"
               "  return i;\n"
               "};",
               Style);

  verifyFormat("if (a)\n"
               "{\n"
               "  b = c + // 1 -\n"
               "      d;\n"
               "}",
               "if (a) {\n"
               "  b = c + // 1 -\n"
               "      d;\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "{\n"
               "  b = c >= 0 ? d\n"
               "             : e;\n"
               "}",
               "if (a) {\n"
               "  b = c >= 0 ? d : e;\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "  b = c > 0 ? d : e;",
               "if (a)\n"
               "{\n"
               "  b = c > 0 ? d : e;\n"
               "}",
               Style);

  verifyFormat("if (foo + bar <=\n"
               "    baz)\n"
               "{\n"
               "  func(arg1, arg2);\n"
               "}",
               "if (foo + bar <= baz) {\n"
               "  func(arg1, arg2);\n"
               "}",
               Style);

  verifyFormat("if (foo + bar < baz)\n"
               "  func(arg1, arg2);\n"
               "else\n"
               "  func();",
               "if (foo + bar < baz)\n"
               "<%\n"
               "  func(arg1, arg2);\n"
               "%>\n"
               "else\n"
               "<%\n"
               "  func();\n"
               "%>",
               Style);

  verifyFormat("while (i--)\n"
               "<% // Keep.\n"
               "  foo();\n"
               "%>",
               "while (i--) <% // Keep.\n"
               "  foo();\n"
               "%>",
               Style);

  verifyFormat("for (int &i : chars)\n"
               "  ++i;",
               "for (int &i : chars)\n"
               "{\n"
               "  ++i;\n"
               "}",
               Style);
}

} // namespace
} // namespace format
} // namespace clang
