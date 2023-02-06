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

TEST_F(FormatTestVerilog, Align) {
  FormatStyle Style = getLLVMStyle(FormatStyle::LK_Verilog);
  Style.AlignConsecutiveAssignments.Enabled = true;
  verifyFormat("x            <= x;\n"
               "sfdbddfbdfbb <= x;\n"
               "x             = x;",
               Style);
  verifyFormat("x            = x;\n"
               "sfdbddfbdfbb = x;\n"
               "x            = x;",
               Style);
  // Compound assignments are not aligned by default. '<=' is not a compound
  // assignment.
  verifyFormat("x            <= x;\n"
               "sfdbddfbdfbb <= x;",
               Style);
  verifyFormat("x += x;\n"
               "sfdbddfbdfbb <= x;",
               Style);
  verifyFormat("x <<= x;\n"
               "sfdbddfbdfbb <= x;",
               Style);
  verifyFormat("x <<<= x;\n"
               "sfdbddfbdfbb <= x;",
               Style);
  verifyFormat("x >>= x;\n"
               "sfdbddfbdfbb <= x;",
               Style);
  verifyFormat("x >>>= x;\n"
               "sfdbddfbdfbb <= x;",
               Style);
  Style.AlignConsecutiveAssignments.AlignCompound = true;
  verifyFormat("x            <= x;\n"
               "sfdbddfbdfbb <= x;",
               Style);
  verifyFormat("x            += x;\n"
               "sfdbddfbdfbb <= x;",
               Style);
  verifyFormat("x            <<= x;\n"
               "sfdbddfbdfbb  <= x;",
               Style);
  verifyFormat("x            <<<= x;\n"
               "sfdbddfbdfbb   <= x;",
               Style);
  verifyFormat("x            >>= x;\n"
               "sfdbddfbdfbb  <= x;",
               Style);
  verifyFormat("x            >>>= x;\n"
               "sfdbddfbdfbb   <= x;",
               Style);
}

TEST_F(FormatTestVerilog, BasedLiteral) {
  verifyFormat("x = '0;");
  verifyFormat("x = '1;");
  verifyFormat("x = 'X;");
  verifyFormat("x = 'x;");
  verifyFormat("x = 'Z;");
  verifyFormat("x = 'z;");
  verifyFormat("x = 659;");
  verifyFormat("x = 'h837ff;");
  verifyFormat("x = 'o7460;");
  verifyFormat("x = 4'b1001;");
  verifyFormat("x = 5'D3;");
  verifyFormat("x = 3'b01x;");
  verifyFormat("x = 12'hx;");
  verifyFormat("x = 16'hz;");
  verifyFormat("x = -8'd6;");
  verifyFormat("x = 4'shf;");
  verifyFormat("x = -4'sd15;");
  verifyFormat("x = 16'sd?;");
}

TEST_F(FormatTestVerilog, Block) {
  verifyFormat("begin\n"
               "  x = x;\n"
               "end");
  verifyFormat("begin : x\n"
               "  x = x;\n"
               "end : x");
  verifyFormat("begin\n"
               "  x = x;\n"
               "  x = x;\n"
               "end");
  verifyFormat("fork\n"
               "  x = x;\n"
               "join");
  verifyFormat("fork\n"
               "  x = x;\n"
               "join_any");
  verifyFormat("fork\n"
               "  x = x;\n"
               "join_none");
  verifyFormat("generate\n"
               "  x = x;\n"
               "endgenerate");
  verifyFormat("generate : x\n"
               "  x = x;\n"
               "endgenerate : x");
  // Nested blocks.
  verifyFormat("begin\n"
               "  begin\n"
               "  end\n"
               "end");
  verifyFormat("begin : x\n"
               "  begin\n"
               "  end\n"
               "end : x");
  verifyFormat("begin : x\n"
               "  begin : x\n"
               "  end : x\n"
               "end : x");
  verifyFormat("begin\n"
               "  begin : x\n"
               "  end : x\n"
               "end");
  // Test that 'disable fork' and 'rand join' don't get mistaken as blocks.
  verifyFormat("disable fork;\n"
               "x = x;");
  verifyFormat("rand join x x;\n"
               "x = x;");
}

TEST_F(FormatTestVerilog, Case) {
  verifyFormat("case (data)\n"
               "endcase");
  verifyFormat("casex (data)\n"
               "endcase");
  verifyFormat("casez (data)\n"
               "endcase");
  verifyFormat("case (data) inside\n"
               "endcase");
  verifyFormat("case (data)\n"
               "  16'd0:\n"
               "    result = 10'b0111111111;\n"
               "endcase");
  verifyFormat("case (data)\n"
               "  xxxxxxxx:\n"
               "    result = 10'b0111111111;\n"
               "endcase");
  // Test labels with multiple options.
  verifyFormat("case (data)\n"
               "  16'd0, 16'd1:\n"
               "    result = 10'b0111111111;\n"
               "endcase");
  verifyFormat("case (data)\n"
               "  16'd0, //\n"
               "      16'd1:\n"
               "    result = 10'b0111111111;\n"
               "endcase");
  // Test that blocks following labels are indented.
  verifyFormat("case (data)\n"
               "  16'd1: fork\n"
               "    result = 10'b1011111111;\n"
               "  join\n"
               "endcase\n");
  verifyFormat("case (data)\n"
               "  16'd1: fork : x\n"
               "    result = 10'b1011111111;\n"
               "  join : x\n"
               "endcase\n");
  // Test default.
  verifyFormat("case (data)\n"
               "  default\n"
               "    result = 10'b1011111111;\n"
               "endcase");
  verifyFormat("case (data)\n"
               "  default:\n"
               "    result = 10'b1011111111;\n"
               "endcase");
  // Test that question marks and colons don't get mistaken as labels.
  verifyFormat("case (data)\n"
               "  8'b1???????:\n"
               "    instruction1(ir);\n"
               "endcase");
  verifyFormat("case (data)\n"
               "  x ? 8'b1??????? : 1:\n"
               "    instruction3(ir);\n"
               "endcase");
  // Test indention options.
  auto Style = getLLVMStyle(FormatStyle::LK_Verilog);
  Style.IndentCaseLabels = false;
  verifyFormat("case (data)\n"
               "16'd0:\n"
               "  result = 10'b0111111111;\n"
               "endcase",
               Style);
  verifyFormat("case (data)\n"
               "16'd0: begin\n"
               "  result = 10'b0111111111;\n"
               "end\n"
               "endcase",
               Style);
  Style.IndentCaseLabels = true;
  verifyFormat("case (data)\n"
               "  16'd0:\n"
               "    result = 10'b0111111111;\n"
               "endcase",
               Style);
  verifyFormat("case (data)\n"
               "  16'd0: begin\n"
               "    result = 10'b0111111111;\n"
               "  end\n"
               "endcase",
               Style);
}

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

TEST_F(FormatTestVerilog, Hierarchy) {
  verifyFormat("module x;\n"
               "endmodule");
  // Test that the end label is on the same line as the end keyword.
  verifyFormat("module x;\n"
               "endmodule : x");
  // Test that things inside are indented.
  verifyFormat("module x;\n"
               "  generate\n"
               "  endgenerate\n"
               "endmodule");
  verifyFormat("program x;\n"
               "  generate\n"
               "  endgenerate\n"
               "endprogram");
  verifyFormat("interface x;\n"
               "  generate\n"
               "  endgenerate\n"
               "endinterface");
  verifyFormat("task x;\n"
               "  generate\n"
               "  endgenerate\n"
               "endtask");
  verifyFormat("function x;\n"
               "  generate\n"
               "  endgenerate\n"
               "endfunction");
  verifyFormat("class x;\n"
               "  generate\n"
               "  endgenerate\n"
               "endclass");
  // Test that they nest.
  verifyFormat("module x;\n"
               "  program x;\n"
               "    program x;\n"
               "    endprogram\n"
               "  endprogram\n"
               "endmodule");
  // Test that an extern declaration doesn't change the indentation.
  verifyFormat("extern module x;\n"
               "x = x;");
  // Test complex headers
  verifyFormat("extern module x\n"
               "    import x.x::x::*;\n"
               "    import x;\n"
               "    #(parameter x)\n"
               "    (output x);");
  verifyFormat("module x\n"
               "    import x.x::x::*;\n"
               "    import x;\n"
               "    #(parameter x)\n"
               "    (output x);\n"
               "  generate\n"
               "  endgenerate\n"
               "endmodule : x");
  verifyFormat("virtual class x\n"
               "    (x)\n"
               "    extends x(x)\n"
               "    implements x, x, x;\n"
               "  generate\n"
               "  endgenerate\n"
               "endclass : x\n");
  verifyFormat("function automatic logic [1 : 0] x\n"
               "    (input x);\n"
               "  generate\n"
               "  endgenerate\n"
               "endfunction : x");
}

TEST_F(FormatTestVerilog, If) {
  verifyFormat("if (x)\n"
               "  x = x;");
  verifyFormat("unique if (x)\n"
               "  x = x;");
  verifyFormat("unique0 if (x)\n"
               "  x = x;");
  verifyFormat("priority if (x)\n"
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

  // With attributes.
  verifyFormat("(* x *) if (x)\n"
               "  x = x;");
  verifyFormat("(* x = \"x\" *) if (x)\n"
               "  x = x;");
  verifyFormat("(* x, x = \"x\" *) if (x)\n"
               "  x = x;");
}

TEST_F(FormatTestVerilog, Operators) {
  // Test that unary operators are not followed by space.
  verifyFormat("x = +x;");
  verifyFormat("x = -x;");
  verifyFormat("x = !x;");
  verifyFormat("x = ~x;");
  verifyFormat("x = &x;");
  verifyFormat("x = ~&x;");
  verifyFormat("x = |x;");
  verifyFormat("x = ~|x;");
  verifyFormat("x = ^x;");
  verifyFormat("x = ~^x;");
  verifyFormat("x = ^~x;");
  verifyFormat("x = ++x;");
  verifyFormat("x = --x;");

  // Test that operators don't get split.
  verifyFormat("x = x++;");
  verifyFormat("x = x--;");
  verifyFormat("x = x ** x;");
  verifyFormat("x = x << x;");
  verifyFormat("x = x >> x;");
  verifyFormat("x = x <<< x;");
  verifyFormat("x = x >>> x;");
  verifyFormat("x = x <= x;");
  verifyFormat("x = x >= x;");
  verifyFormat("x = x == x;");
  verifyFormat("x = x != x;");
  verifyFormat("x = x === x;");
  verifyFormat("x = x !== x;");
  verifyFormat("x = x ==? x;");
  verifyFormat("x = x !=? x;");
  verifyFormat("x = x ~^ x;");
  verifyFormat("x = x ^~ x;");
  verifyFormat("x = x && x;");
  verifyFormat("x = x || x;");
  verifyFormat("x = x->x;");
  verifyFormat("x = x <-> x;");
  verifyFormat("x += x;");
  verifyFormat("x -= x;");
  verifyFormat("x *= x;");
  verifyFormat("x /= x;");
  verifyFormat("x %= x;");
  verifyFormat("x &= x;");
  verifyFormat("x ^= x;");
  verifyFormat("x |= x;");
  verifyFormat("x <<= x;");
  verifyFormat("x >>= x;");
  verifyFormat("x <<<= x;");
  verifyFormat("x >>>= x;");
  verifyFormat("x <= x;");

  // Test that space is added between operators.
  EXPECT_EQ("x = x < -x;", format("x=x<-x;"));
  EXPECT_EQ("x = x << -x;", format("x=x<<-x;"));
  EXPECT_EQ("x = x <<< -x;", format("x=x<<<-x;"));
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

TEST_F(FormatTestVerilog, Primitive) {
  verifyFormat("primitive multiplexer\n"
               "    (mux, control, dataA, dataB);\n"
               "  output mux;\n"
               "  input control, dataA, dataB;\n"
               "  table\n"
               "    0 1 ? : 1;\n"
               "    0 0 ? : 0;\n"
               "    1 ? 1 : 1;\n"
               "    1 ? 0 : 0;\n"
               "    x 0 0 : 0;\n"
               "    x 1 1 : 1;\n"
               "  endtable\n"
               "endprimitive");
  verifyFormat("primitive latch\n"
               "    (q, ena_, data);\n"
               "  output q;\n"
               "  reg q;\n"
               "  input ena_, data;\n"
               "  table\n"
               "    0 1 : ? : 1;\n"
               "    0 0 : ? : 0;\n"
               "    1 ? : ? : -;\n"
               "    ? * : ? : -;\n"
               "  endtable\n"
               "endprimitive");
  verifyFormat("primitive d\n"
               "    (q, clock, data);\n"
               "  output q;\n"
               "  reg q;\n"
               "  input clock, data;\n"
               "  table\n"
               "    (01) 0 : ? : 0;\n"
               "    (01) 1 : ? : 1;\n"
               "    (0?) 1 : 1 : 1;\n"
               "    (0?) 0 : 0 : 0;\n"
               "    (?0) ? : ? : -;\n"
               "    (?\?) ? : ? : -;\n"
               "  endtable\n"
               "endprimitive");
}
} // namespace format
} // end namespace clang
