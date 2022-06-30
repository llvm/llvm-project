//===- unittest/Format/FormatTestVerilog.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestUtils.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {

class FormatTestVerilog : public ::testing::Test {
protected:
  static std::string format(llvm::StringRef Code, unsigned Offset,
                            unsigned Length, const FormatStyle &Style) {
    LLVM_DEBUG(llvm::errs() << "---\n");
    LLVM_DEBUG(llvm::errs() << Code << "\n\n");
    std::vector<tooling::Range> Ranges(1, tooling::Range(Offset, Length));
    tooling::Replacements Replaces = reformat(Style, Code, Ranges);
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    LLVM_DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  static std::string
  format(llvm::StringRef Code,
         const FormatStyle &Style = getLLVMStyle(FormatStyle::LK_Verilog)) {
    return format(Code, 0, Code.size(), Style);
  }

  static void verifyFormat(
      llvm::StringRef Code,
      const FormatStyle &Style = getLLVMStyle(FormatStyle::LK_Verilog)) {
    EXPECT_EQ(Code.str(), format(Code, Style)) << "Expected code is not stable";
    EXPECT_EQ(Code.str(),
              format(test::messUp(Code, /*HandleHash=*/false), Style));
  }
};

TEST_F(FormatTestVerilog, Delay) {
  // Delay by the default unit.
  verifyFormat("#0;");
  verifyFormat("#1;");
  verifyFormat("#10;");
  verifyFormat("#1.5;");
  // Explicit unit.
  verifyFormat("#1fs;");
  verifyFormat("#1.5fs;");
  verifyFormat("#1ns;");
  verifyFormat("#1.5ns;");
  verifyFormat("#1us;");
  verifyFormat("#1.5us;");
  verifyFormat("#1ms;");
  verifyFormat("#1.5ms;");
  verifyFormat("#1s;");
  verifyFormat("#1.5s;");
  // The following expression should be on the same line.
  verifyFormat("#1 x = x;");
  EXPECT_EQ("#1 x = x;", format("#1\n"
                                "x = x;"));
}

TEST_F(FormatTestVerilog, If) {
  verifyFormat("if (x)\n"
               "  x = x;");
  verifyFormat("if (x)\n"
               "  x = x;\n"
               "x = x;");

  // Test else
  verifyFormat("if (x)\n"
               "  x = x;\n"
               "else if (x)\n"
               "  x = x;\n"
               "else\n"
               "  x = x;");
  verifyFormat("if (x) begin\n"
               "  x = x;\n"
               "end else if (x) begin\n"
               "  x = x;\n"
               "end else begin\n"
               "  x = x;\n"
               "end");
  verifyFormat("if (x) begin : x\n"
               "  x = x;\n"
               "end : x else if (x) begin : x\n"
               "  x = x;\n"
               "end : x else begin : x\n"
               "  x = x;\n"
               "end : x");

  // Test block keywords.
  verifyFormat("if (x) begin\n"
               "  x = x;\n"
               "end");
  verifyFormat("if (x) begin : x\n"
               "  x = x;\n"
               "end : x");
  verifyFormat("if (x) begin\n"
               "  x = x;\n"
               "  x = x;\n"
               "end");
  verifyFormat("disable fork;\n"
               "x = x;");
  verifyFormat("rand join x x;\n"
               "x = x;");
  verifyFormat("if (x) fork\n"
               "  x = x;\n"
               "join");
  verifyFormat("if (x) fork\n"
               "  x = x;\n"
               "join_any");
  verifyFormat("if (x) fork\n"
               "  x = x;\n"
               "join_none");
  verifyFormat("if (x) generate\n"
               "  x = x;\n"
               "endgenerate");
  verifyFormat("if (x) generate : x\n"
               "  x = x;\n"
               "endgenerate : x");

  // Test that concatenation braces don't get regarded as blocks.
  verifyFormat("if (x)\n"
               "  {x} = x;");
  verifyFormat("if (x)\n"
               "  x = {x};");
  verifyFormat("if (x)\n"
               "  x = {x};\n"
               "else\n"
               "  {x} = {x};");
}

TEST_F(FormatTestVerilog, Preprocessor) {
  auto Style = getLLVMStyle(FormatStyle::LK_Verilog);
  Style.ColumnLimit = 20;

  // Macro definitions.
  EXPECT_EQ("`define X          \\\n"
            "  if (x)           \\\n"
            "    x = x;",
            format("`define X if(x)x=x;", Style));
  EXPECT_EQ("`define X(x)       \\\n"
            "  if (x)           \\\n"
            "    x = x;",
            format("`define X(x) if(x)x=x;", Style));
  EXPECT_EQ("`define X          \\\n"
            "  x = x;           \\\n"
            "  x = x;",
            format("`define X x=x;x=x;", Style));
  // Macro definitions with invocations inside.
  EXPECT_EQ("`define LIST       \\\n"
            "  `ENTRY           \\\n"
            "  `ENTRY",
            format("`define LIST \\\n"
                   "`ENTRY \\\n"
                   "`ENTRY",
                   Style));
  EXPECT_EQ("`define LIST       \\\n"
            "  `x = `x;         \\\n"
            "  `x = `x;",
            format("`define LIST \\\n"
                   "`x = `x; \\\n"
                   "`x = `x;",
                   Style));
  EXPECT_EQ("`define LIST       \\\n"
            "  `x = `x;         \\\n"
            "  `x = `x;",
            format("`define LIST `x=`x;`x=`x;", Style));
  // Macro invocations.
  verifyFormat("`x = (`x1 + `x2 + x);");
  // Lines starting with a preprocessor directive should not be indented.
  std::string Directives[] = {
      "begin_keywords",
      "celldefine",
      "default_nettype",
      "define",
      "else",
      "elsif",
      "end_keywords",
      "endcelldefine",
      "endif",
      "ifdef",
      "ifndef",
      "include",
      "line",
      "nounconnected_drive",
      "pragma",
      "resetall",
      "timescale",
      "unconnected_drive",
      "undef",
      "undefineall",
  };
  for (auto &Name : Directives) {
    EXPECT_EQ("if (x)\n"
              "`" +
                  Name +
                  "\n"
                  "  ;",
              format("if (x)\n"
                     "`" +
                         Name +
                         "\n"
                         ";",
                     Style));
  }
  // Lines starting with a regular macro invocation should be indented as a
  // normal line.
  EXPECT_EQ("if (x)\n"
            "  `x = `x;\n"
            "`timescale 1ns / 1ps",
            format("if (x)\n"
                   "`x = `x;\n"
                   "`timescale 1ns / 1ps",
                   Style));
  EXPECT_EQ("if (x)\n"
            "`timescale 1ns / 1ps\n"
            "  `x = `x;",
            format("if (x)\n"
                   "`timescale 1ns / 1ps\n"
                   "`x = `x;",
                   Style));
  std::string NonDirectives[] = {
      // For `__FILE__` and `__LINE__`, although the standard classifies them as
      // preprocessor directives, they are used like regular macros.
      "__FILE__", "__LINE__", "elif", "foo", "x",
  };
  for (auto &Name : NonDirectives) {
    EXPECT_EQ("if (x)\n"
              "  `" +
                  Name + ";",
              format("if (x)\n"
                     "`" +
                         Name +
                         "\n"
                         ";",
                     Style));
  }
}

} // namespace format
} // end namespace clang
