//===- unittest/Format/BracesInserterTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestBase.h"

#define DEBUG_TYPE "braces-inserter-test"

namespace clang {
namespace format {
namespace test {
namespace {

class BracesInserterTest : public FormatTestBase {};

TEST_F(BracesInserterTest, InsertBraces) {
  FormatStyle Style = getLLVMStyle();
  Style.InsertBraces = true;

  verifyFormat("// clang-format off\n"
               "// comment\n"
               "if (a) f();\n"
               "// clang-format on\n"
               "if (b) {\n"
               "  g();\n"
               "}",
               "// clang-format off\n"
               "// comment\n"
               "if (a) f();\n"
               "// clang-format on\n"
               "if (b) g();",
               Style);

  verifyFormat("if (a) {\n"
               "  switch (b) {\n"
               "  case 1:\n"
               "    c = 0;\n"
               "    break;\n"
               "  default:\n"
               "    c = 1;\n"
               "  }\n"
               "}",
               "if (a)\n"
               "  switch (b) {\n"
               "  case 1:\n"
               "    c = 0;\n"
               "    break;\n"
               "  default:\n"
               "    c = 1;\n"
               "  }",
               Style);

  verifyFormat("for (auto node : nodes) {\n"
               "  if (node) {\n"
               "    break;\n"
               "  }\n"
               "}",
               "for (auto node : nodes)\n"
               "  if (node)\n"
               "    break;",
               Style);

  verifyFormat("for (auto node : nodes) {\n"
               "  if (node)\n"
               "}",
               "for (auto node : nodes)\n"
               "  if (node)",
               Style);

  verifyFormat("do {\n"
               "  --a;\n"
               "} while (a);",
               "do\n"
               "  --a;\n"
               "while (a);",
               Style);

  verifyFormat("if (i) {\n"
               "  ++i;\n"
               "} else {\n"
               "  --i;\n"
               "}",
               "if (i)\n"
               "  ++i;\n"
               "else {\n"
               "  --i;\n"
               "}",
               Style);

  verifyFormat("void f() {\n"
               "  while (j--) {\n"
               "    while (i) {\n"
               "      --i;\n"
               "    }\n"
               "  }\n"
               "}",
               "void f() {\n"
               "  while (j--)\n"
               "    while (i)\n"
               "      --i;\n"
               "}",
               Style);

  verifyFormat("f({\n"
               "  if (a) {\n"
               "    g();\n"
               "  }\n"
               "});",
               "f({\n"
               "  if (a)\n"
               "    g();\n"
               "});",
               Style);

  verifyFormat("if (a) {\n"
               "  f();\n"
               "} else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  h();\n"
               "}",
               "if (a)\n"
               "  f();\n"
               "else if (b)\n"
               "  g();\n"
               "else\n"
               "  h();",
               Style);

  verifyFormat("if (a) {\n"
               "  f();\n"
               "}\n"
               "// comment\n"
               "/* comment */",
               "if (a)\n"
               "  f();\n"
               "// comment\n"
               "/* comment */",
               Style);

  verifyFormat("if (a) {\n"
               "  // foo\n"
               "  // bar\n"
               "  f();\n"
               "}",
               "if (a)\n"
               "  // foo\n"
               "  // bar\n"
               "  f();",
               Style);

  verifyFormat("if (a) { //\n"
               "  b = 1;\n"
               "}",
               "if (a) //\n"
               "  b = 1;",
               Style);

  verifyFormat("if (a) { // comment\n"
               "  // comment\n"
               "  f();\n"
               "}",
               "if (a) // comment\n"
               "  // comment\n"
               "  f();",
               Style);

  verifyFormat("if (a) {\n"
               "  f(); // comment\n"
               "}",
               "if (a)\n"
               "  f(); // comment",
               Style);

  verifyFormat("if (a) {\n"
               "  f();\n"
               "}\n"
               "#undef A\n"
               "#undef B",
               "if (a)\n"
               "  f();\n"
               "#undef A\n"
               "#undef B",
               Style);

  verifyFormat("if (a)\n"
               "#ifdef A\n"
               "  f();\n"
               "#else\n"
               "  g();\n"
               "#endif",
               Style);

  verifyFormat("if (a) {\n"
               "  if (b)\n"
               "#ifdef A\n"
               "    f();\n"
               "#else\n"
               "    g();\n"
               "#endif\n"
               "}",
               "if (a)\n"
               "  if (b)\n"
               "#ifdef A\n"
               "    f();\n"
               "#else\n"
               "    g();\n"
               "#endif",
               Style);

  verifyFormat("if (a) {\n"
               "  if (b)\n"
               "#if A\n"
               "    f();\n"
               "#elif B\n"
               "    g();\n"
               "#else\n"
               "    h();\n"
               "#endif // A\n"
               "}",
               "if (a)\n"
               "  if (b)\n"
               "#if A\n"
               "    f();\n"
               "#elif B\n"
               "    g();\n"
               "#else\n"
               "    h();\n"
               "#endif // A",
               Style);

  verifyFormat("if (a) {\n"
               "  if (b) {\n"
               "#ifdef A\n"
               "    f();\n"
               "#endif\n"
               "  }\n"
               "}",
               "if (a)\n"
               "  if (b) {\n"
               "#ifdef A\n"
               "    f();\n"
               "#endif\n"
               "  }",
               Style);

  verifyFormat("do {\n"
               "  if (b)\n"
               "#ifdef A\n"
               "    f();\n"
               "#else\n"
               "    g();\n"
               "#endif\n"
               "} while (a);",
               "do\n"
               "  if (b)\n"
               "#ifdef A\n"
               "    f();\n"
               "#else\n"
               "    g();\n"
               "#endif\n"
               "while (a);",
               Style);

  verifyFormat("while (a) {\n"
               "  if (b) {\n"
               "    f(\n"
               "#ifdef A\n"
               "        1);\n"
               "#else\n"
               "        2);\n"
               "#endif\n"
               "  }\n"
               "}",
               "while (a)\n"
               "  if (b)\n"
               "    f(\n"
               "#ifdef A\n"
               "        1);\n"
               "#else\n"
               "        2);\n"
               "#endif",
               Style);

  verifyFormat("if (a) {\n"
               "  f(\n"
               "#ifdef A\n"
               "      1\n"
               "#else\n"
               "      2\n"
               "#endif\n"
               "  );\n"
               "}",
               "if (a)\n"
               "  f(\n"
               "#ifdef A\n"
               "      1\n"
               "#else\n"
               "      2\n"
               "#endif\n"
               "  );",
               Style);

  verifyFormat("if (a)\n"
               "  if (b)\n"
               "#ifdef A\n"
               "    f();\n"
               "x();\n"
               "#else\n"
               "    g();\n"
               "#endif",
               Style);

  verifyFormat("if (a)\n"
               "  if (b)\n"
               "#ifdef A\n"
               "    f();\n"
               "#endif\n"
               "g();",
               Style);

  verifyFormat("if (a)\n"
               "  if (b)\n"
               "#if A\n"
               "    f();\n"
               "#elif B\n"
               "#else\n"
               "    g();\n"
               "#endif\n"
               "h();",
               Style);

  verifyFormat("if (a)\n"
               "  if (b)\n"
               "#ifdef A\n"
               "#else\n"
               "    f();\n"
               "#endif\n"
               "    g();",
               Style);

  verifyFormat("if (a)\n"
               "  if (b)\n"
               "#ifdef A\n"
               "    f();\n"
               "#else\n"
               "    g();\n"
               "h();\n"
               "#endif",
               Style);

  verifyFormat("if (a)\n"
               "  f(\n"
               "#if A\n"
               "      1\n"
               "#else\n"
               "      2)\n"
               "#if B\n"
               "      ;\n"
               "h();\n"
               "#else\n"
               "      ;\n"
               "g();\n"
               "#endif\n"
               "f(3\n"
               "#endif\n"
               "  );",
               Style);

  verifyFormat("if (a) {\n"
               "  f(\n"
               "#if A\n"
               "      1\n"
               "#else\n"
               "#if B\n"
               "      2\n"
               "#else\n"
               "      3\n"
               "#endif\n"
               "#endif\n"
               "  );\n"
               "}",
               "if (a)\n"
               "  f(\n"
               "#if A\n"
               "      1\n"
               "#else\n"
               "#if B\n"
               "      2\n"
               "#else\n"
               "      3\n"
               "#endif\n"
               "#endif\n"
               "  );",
               Style);

  verifyFormat("if (a) {\n"
               "  if (b)\n"
               "/* comment */ #ifdef A\n"
               "    f();\n"
               "#else\n"
               "    g();\n"
               "#endif\n"
               "}",
               "if (a)\n"
               "  if (b)\n"
               "/* comment */ #ifdef A\n"
               "    f();\n"
               "#else\n"
               "    g();\n"
               "#endif",
               Style);

  verifyFormat("#ifdef A\n"
               "if (a)\n"
               "  f(\n"
               "#else\n"
               "if (b)\n"
               "  g(\n"
               "#endif\n"
               "      1);",
               Style);

  verifyFormat("if (a) {\n"
               "  f();\n"
               "}\n"
               "#ifdef A\n"
               "g();\n"
               "#endif",
               "if (a)\n"
               "  f();\n"
               "#ifdef A\n"
               "g();\n"
               "#endif",
               Style);

  verifyFormat("#if A\n"
               "if (a) {\n"
               "  f();\n"
               "}\n"
               "#else\n"
               "if (b) {\n"
               "  g();\n"
               "}\n"
               "#endif",
               "#if A\n"
               "if (a) f();\n"
               "#else\n"
               "if (b) g();\n"
               "#endif",
               Style);

  verifyFormat("void f() {\n"
               "#if A\n"
               "  a();\n"
               "#else\n"
               "  b();\n"
               "#endif\n"
               "  if (c) {\n"
               "    h();\n"
               "  }\n"
               "}",
               "void f() {\n"
               "#if A\n"
               "  a();\n"
               "#else\n"
               "  b();\n"
               "#endif\n"
               "  if (c) h();\n"
               "}",
               Style);

  verifyFormat("void f() {\n"
               "#if A\n"
               "  a();\n"
               "#else\n"
               "  b();\n"
               "#endif\n"
               "  if (c) {\n"
               "    if (d) {\n"
               "      h();\n"
               "    }\n"
               "  }\n"
               "}",
               "void f() {\n"
               "#if A\n"
               "  a();\n"
               "#else\n"
               "  b();\n"
               "#endif\n"
               "  if (c)\n"
               "    if (d) h();\n"
               "}",
               Style);

  verifyFormat("#if 0\n"
               "#elif 1\n"
               "#endif\n"
               "void f() {\n"
               "  if (a) {\n"
               "    g();\n"
               "  }\n"
               "}",
               "#if 0\n"
               "#elif 1\n"
               "#endif\n"
               "void f() {\n"
               "  if (a) g();\n"
               "}",
               Style);

  verifyFormat("do {\n"
               "#if 0\n"
               "#else\n"
               "  if (b) {\n"
               "#endif\n"
               "  }\n"
               "} while (0);",
               Style);

  Style.RemoveBracesLLVM = true;
  verifyFormat("if (a) //\n"
               "  return b;",
               Style);
  Style.RemoveBracesLLVM = false;

  Style.ColumnLimit = 15;

  verifyFormat("#define A     \\\n"
               "  if (a)      \\\n"
               "    f();",
               Style);

  verifyFormat("if (a + b >\n"
               "    c) {\n"
               "  f();\n"
               "}",
               "if (a + b > c)\n"
               "  f();",
               Style);

  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_Always;

  verifyFormat("if (a) //\n"
               "{\n"
               "  b = 1;\n"
               "}",
               "if (a) //\n"
               "  b = 1;",
               Style);
}

TEST_F(BracesInserterTest, InsertBracesRange) {
  FormatStyle Style = getLLVMStyle();
  Style.InsertBraces = true;

  constexpr StringRef Code("while (a)\n"
                           "  if (b)\n"
                           "    return;");

  verifyFormat("while (a) {\n"
               "  if (b)\n"
               "    return;\n"
               "}",
               Code, Style, {tooling::Range(0, 9)}); // line 1

  verifyFormat("while (a) {\n"
               "  if (b) {\n"
               "    return;\n"
               "  }\n"
               "}",
               Code, Style, {tooling::Range(0, 18)}); // lines 1-2

  verifyFormat("while (a)\n"
               "  if (b) {\n"
               "    return;\n"
               "  }",
               Code, Style, {tooling::Range(10, 8)}); // line 2

  verifyFormat(Code, Code, Style, {tooling::Range(19, 11)}); // line 3
}

} // namespace
} // namespace test
} // namespace format
} // namespace clang
