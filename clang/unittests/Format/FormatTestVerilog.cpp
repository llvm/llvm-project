//===- unittest/Format/FormatTestVerilog.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestBase.h"

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {
namespace test {
namespace {
class FormatTestVerilog : public test::FormatTestBase {
protected:
  FormatStyle getDefaultStyle() const override {
    return getLLVMStyle(FormatStyle::LK_Verilog);
  }
  std::string messUp(StringRef Code) const override {
    return test::messUp(Code, /*HandleHash=*/false);
  }
};

TEST_F(FormatTestVerilog, Align) {
  FormatStyle Style = getDefaultStyle();
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

TEST_F(FormatTestVerilog, Assign) {
  verifyFormat("assign mynet = enable;");
  verifyFormat("assign (strong1, pull0) #1 mynet = enable;");
  verifyFormat("assign #1 mynet = enable;");
  verifyFormat("assign mynet = enable;");
  // Test that assignments are on separate lines.
  verifyFormat("assign mynet = enable,\n"
               "       mynet1 = enable1;");
  // Test that `<=` and `,` don't confuse it.
  verifyFormat("assign mynet = enable1 <= enable2;");
  verifyFormat("assign mynet = enable1 <= enable2,\n"
               "       mynet1 = enable3;");
  verifyFormat("assign mynet = enable,\n"
               "       mynet1 = enable2 <= enable3;");
  verifyFormat("assign mynet = enable(enable1, enable2);");
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
  // The begin keyword should not be indented if it is too long to fit on the
  // same line.
  verifyFormat("while (true) //\n"
               "begin\n"
               "  while (true) //\n"
               "  begin\n"
               "  end\n"
               "end");
  verifyFormat("while (true) //\n"
               "begin : x\n"
               "  while (true) //\n"
               "  begin : x\n"
               "  end : x\n"
               "end : x");
  verifyFormat("while (true) //\n"
               "fork\n"
               "  while (true) //\n"
               "  fork\n"
               "  join\n"
               "join");
  auto Style = getDefaultStyle();
  Style.ColumnLimit = 17;
  verifyFormat("while (true)\n"
               "begin\n"
               "  while (true)\n"
               "  begin\n"
               "  end\n"
               "end",
               "while (true) begin\n"
               "  while (true) begin"
               "  end\n"
               "end",
               Style);
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
               "endcase");
  verifyFormat("case (data)\n"
               "  16'd1: fork : x\n"
               "    result = 10'b1011111111;\n"
               "  join : x\n"
               "endcase");
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
  auto Style = getDefaultStyle();
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
  // Other colons should not be mistaken as case colons.
  Style = getDefaultStyle();
  Style.BitFieldColonSpacing = FormatStyle::BFCS_None;
  verifyFormat("case (x[1:0])\n"
               "endcase",
               Style);
  verifyFormat("default:\n"
               "  x[1:0] = x[1:0];",
               Style);
  Style.BitFieldColonSpacing = FormatStyle::BFCS_Both;
  verifyFormat("case (x[1 : 0])\n"
               "endcase",
               Style);
  verifyFormat("default:\n"
               "  x[1 : 0] = x[1 : 0];",
               Style);
  Style = getDefaultStyle();
  Style.SpacesInContainerLiterals = true;
  verifyFormat("case ('{x : x, default : 9})\n"
               "endcase",
               Style);
  verifyFormat("x = '{x : x, default : 9};", Style);
  verifyFormat("default:\n"
               "  x = '{x : x, default : 9};",
               Style);
  Style.SpacesInContainerLiterals = false;
  verifyFormat("case ('{x: x, default: 9})\n"
               "endcase",
               Style);
  verifyFormat("x = '{x: x, default: 9};", Style);
  verifyFormat("default:\n"
               "  x = '{x: x, default: 9};",
               Style);
  // When the line following the case label needs to be broken, the continuation
  // should be indented correctly.
  verifyFormat("case (data)\n"
               "  16'd0:\n"
               "    result = //\n"
               "        10'b0111111111;\n"
               "endcase");
  verifyFormat("case (data)\n"
               "  16'd0, //\n"
               "      16'd1:\n"
               "    result = //\n"
               "        10'b0111111111;\n"
               "endcase");
  verifyFormat("case (data)\n"
               "  16'd0:\n"
               "    result = (10'b0111111111 + //\n"
               "              10'b0111111111 + //\n"
               "              10'b0111111111);\n"
               "endcase");
  verifyFormat("case (data)\n"
               "  16'd0:\n"
               "    result =              //\n"
               "        (10'b0111111111 + //\n"
               "         10'b0111111111 + //\n"
               "         10'b0111111111);\n"
               "endcase");
  verifyFormat("case (data)\n"
               "  16'd0:\n"
               "    result =          //\n"
               "        longfunction( //\n"
               "            arg);\n"
               "endcase");
  verifyFormat("case (data)\n"
               "  16'd0:\n"
               "    //\n"
               "    result = //\n"
               "        10'b0111111111;\n"
               "endcase");
  verifyFormat("case (data)\n"
               "  16'd0:\n"
               "    //\n"
               "\n"
               "    //\n"
               "    result = //\n"
               "        10'b0111111111;\n"
               "endcase");
  Style = getDefaultStyle();
  Style.ContinuationIndentWidth = 1;
  verifyFormat("case (data)\n"
               "  16'd0:\n"
               "    result = //\n"
               "     10'b0111111111;\n"
               "endcase",
               Style);
  verifyFormat("case (data)\n"
               "  16'd0:\n"
               "    result =       //\n"
               "     longfunction( //\n"
               "      arg);\n"
               "endcase",
               Style);

  verifyFormat("case (v) matches\n"
               "  tagged Valid .n:\n"
               "    ;\n"
               "endcase");
}

TEST_F(FormatTestVerilog, Coverage) {
  verifyFormat("covergroup x\n"
               "    @@(begin x);\n"
               "endgroup");
}

TEST_F(FormatTestVerilog, Declaration) {
  verifyFormat("wire mynet;");
  verifyFormat("wire mynet, mynet1;");
  verifyFormat("wire mynet, //\n"
               "     mynet1;");
  verifyFormat("wire mynet = enable;");
  verifyFormat("wire mynet = enable, mynet1;");
  verifyFormat("wire mynet = enable, //\n"
               "     mynet1;");
  verifyFormat("wire mynet, mynet1 = enable;");
  verifyFormat("wire mynet, //\n"
               "     mynet1 = enable;");
  verifyFormat("wire mynet = enable, mynet1 = enable;");
  verifyFormat("wire mynet = enable, //\n"
               "     mynet1 = enable;");
  verifyFormat("wire (strong1, pull0) mynet;");
  verifyFormat("wire (strong1, pull0) mynet, mynet1;");
  verifyFormat("wire (strong1, pull0) mynet, //\n"
               "                      mynet1;");
  verifyFormat("wire (strong1, pull0) mynet = enable;");
  verifyFormat("wire (strong1, pull0) mynet = enable, mynet1;");
  verifyFormat("wire (strong1, pull0) mynet = enable, //\n"
               "                      mynet1;");
  verifyFormat("wire (strong1, pull0) mynet, mynet1 = enable;");
  verifyFormat("wire (strong1, pull0) mynet, //\n"
               "                      mynet1 = enable;");
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
  verifyFormat("#1 x = x;", "#1\n"
                            "x = x;");
}

TEST_F(FormatTestVerilog, Enum) {
  verifyFormat("enum { x } x;");
  verifyFormat("typedef enum { x } x;");
  verifyFormat("enum { red, yellow, green } x;");
  verifyFormat("typedef enum { red, yellow, green } x;");
  verifyFormat("enum integer { x } x;");
  verifyFormat("typedef enum { x = 0 } x;");
  verifyFormat("typedef enum { red = 0, yellow = 1, green = 2 } x;");
  verifyFormat("typedef enum integer { x } x;");
  verifyFormat("typedef enum bit [0 : 1] { x } x;");
  verifyFormat("typedef enum { add = 10, sub[5], jmp[6 : 8] } E1;");
  verifyFormat("typedef enum { add = 10, sub[5] = 0, jmp[6 : 8] = 1 } E1;");
}

TEST_F(FormatTestVerilog, Headers) {
  // Test headers with multiple ports.
  verifyFormat("module mh1\n"
               "    (input var int in1,\n"
               "     input var shortreal in2,\n"
               "     output tagged_st out);\n"
               "endmodule");
  // There should be a space following the type but not the variable name.
  verifyFormat("module test\n"
               "    (input wire [7 : 0] a,\n"
               "     input wire b[7 : 0],\n"
               "     input wire [7 : 0] c[7 : 0]);\n"
               "endmodule");
  // Ports should be grouped by types.
  verifyFormat("module test\n"
               "    (input [7 : 0] a,\n"
               "     input signed [7 : 0] b, c, d);\n"
               "endmodule");
  verifyFormat("module test\n"
               "    (input [7 : 0] a,\n"
               "     (* x = x *) input signed [7 : 0] b, c, d);\n"
               "endmodule");
  verifyFormat("module test\n"
               "    (input [7 : 0] a = 0,\n"
               "     input signed [7 : 0] b = 0, c = 0, d = 0);\n"
               "endmodule");
  verifyFormat("module test\n"
               "    #(parameter x)\n"
               "    (input [7 : 0] a,\n"
               "     input signed [7 : 0] b, c, d);\n"
               "endmodule");
  // When a line needs to be broken, ports of the same type should be aligned to
  // the same column.
  verifyFormat("module test\n"
               "    (input signed [7 : 0] b, c, //\n"
               "                          d);\n"
               "endmodule");
  verifyFormat("module test\n"
               "    ((* x = x *) input signed [7 : 0] b, c, //\n"
               "                                      d);\n"
               "endmodule");
  verifyFormat("module test\n"
               "    (input signed [7 : 0] b = 0, c, //\n"
               "                          d);\n"
               "endmodule");
  verifyFormat("module test\n"
               "    (input signed [7 : 0] b, c = 0, //\n"
               "                          d);\n"
               "endmodule");
  verifyFormat("module test\n"
               "    (input signed [7 : 0] b, c, //\n"
               "                          d = 0);\n"
               "endmodule");
  verifyFormat("module test\n"
               "    (input wire logic signed [7 : 0][0 : 1] b, c, //\n"
               "                                            d);\n"
               "endmodule");
  verifyFormat("module test\n"
               "    (input signed [7 : 0] b, //\n"
               "                          c, //\n"
               "                          d);\n"
               "endmodule");
  verifyFormat("module test\n"
               "    (input [7 : 0] a,\n"
               "     input signed [7 : 0] b, //\n"
               "                          c, //\n"
               "                          d);\n"
               "endmodule");
  verifyFormat("module test\n"
               "    (input signed [7 : 0] b, //\n"
               "                          c, //\n"
               "                          d,\n"
               "     output signed [7 : 0] h);\n"
               "endmodule");
  // With a modport.
  verifyFormat("module m\n"
               "    (i2.master i);\n"
               "endmodule");
  verifyFormat("module m\n"
               "    (i2.master i, ii);\n"
               "endmodule");
  verifyFormat("module m\n"
               "    (i2.master i, //\n"
               "               ii);\n"
               "endmodule");
  verifyFormat("module m\n"
               "    (i2.master i,\n"
               "     input ii);\n"
               "endmodule");
  verifyFormat("module m\n"
               "    (i2::i2.master i);\n"
               "endmodule");
  verifyFormat("module m\n"
               "    (i2::i2.master i, ii);\n"
               "endmodule");
  verifyFormat("module m\n"
               "    (i2::i2.master i, //\n"
               "                   ii);\n"
               "endmodule");
  verifyFormat("module m\n"
               "    (i2::i2.master i,\n"
               "     input ii);\n"
               "endmodule");
  verifyFormat("module m\n"
               "    (i2::i2 i);\n"
               "endmodule");
  verifyFormat("module m\n"
               "    (i2::i2 i, ii);\n"
               "endmodule");
  verifyFormat("module m\n"
               "    (i2::i2 i, //\n"
               "            ii);\n"
               "endmodule");
  verifyFormat("module m\n"
               "    (i2::i2 i,\n"
               "     input ii);\n"
               "endmodule");
  // With a macro in the names.
  verifyFormat("module m\n"
               "    (input var `x a, b);\n"
               "endmodule");
  verifyFormat("module m\n"
               "    (input var `x a, //\n"
               "                  b);\n"
               "endmodule");
  verifyFormat("module m\n"
               "    (input var x `a, b);\n"
               "endmodule");
  verifyFormat("module m\n"
               "    (input var x `a, //\n"
               "                 b);\n"
               "endmodule");
  // A line comment shouldn't disrupt the indentation of the port list.
  verifyFormat("extern module x\n"
               "    (//\n"
               "     output y);");
  verifyFormat("extern module x\n"
               "    #(//\n"
               "      parameter x)\n"
               "    (//\n"
               "     output y);");
  // With a concatenation in the names.
  auto Style = getDefaultStyle();
  Style.ColumnLimit = 40;
  verifyFormat("`define X(x)                           \\\n"
               "  module test                          \\\n"
               "      (input var x``x a, b);",
               Style);
  verifyFormat("`define X(x)                           \\\n"
               "  module test                          \\\n"
               "      (input var x``x aaaaaaaaaaaaaaa, \\\n"
               "                      b);",
               Style);
  verifyFormat("`define X(x)                           \\\n"
               "  module test                          \\\n"
               "      (input var x a``x, b);",
               Style);
  verifyFormat("`define X(x)                           \\\n"
               "  module test                          \\\n"
               "      (input var x aaaaaaaaaaaaaaa``x, \\\n"
               "                   b);",
               Style);
  // When the ports line is not to be formatted, following lines should not take
  // on its indentation.
  verifyFormat("module x\n"
               "    (output x);\n"
               "  assign x = 0;\n"
               "endmodule",
               "module x\n"
               "    (output x);\n"
               "    assign x = 0;\n"
               "endmodule",
               getDefaultStyle(), {tooling::Range(25, 18)});
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
               "endclass : x");
  verifyFormat("function automatic logic [1 : 0] x\n"
               "    (input x);\n"
               "  generate\n"
               "  endgenerate\n"
               "endfunction : x");
}

TEST_F(FormatTestVerilog, Identifiers) {
  // Escaped identifiers should not be split.
  verifyFormat("\\busa+index");
  verifyFormat("\\-clock");
  verifyFormat("\\***error-condition***");
  verifyFormat("\\net1\\/net2");
  verifyFormat("\\{a,b}");
  verifyFormat("\\a*(b+c)");
  // Escaped identifiers can't be joined with the next token.  Extra space
  // should be removed.
  verifyFormat("\\busa+index ;", "\\busa+index\n"
                                 ";");
  verifyFormat("\\busa+index ;", "\\busa+index\r\n"
                                 ";");
  verifyFormat("\\busa+index ;", "\\busa+index  ;");
  verifyFormat("\\busa+index ;", "\\busa+index\n"
                                 " ;");
  verifyFormat("\\busa+index ;");
  verifyFormat("(\\busa+index );");
  verifyFormat("\\busa+index \\busa+index ;");
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

  // Assert are treated similar to if.  But the else parts should not be
  // chained.
  verifyFormat("assert (x);");
  verifyFormat("assert (x)\n"
               "  $info();");
  verifyFormat("assert (x)\n"
               "  $info();\n"
               "else\n"
               "  $error();");
  verifyFormat("assert (x)\n"
               "else\n"
               "  $error();");
  verifyFormat("assert (x)\n"
               "else begin\n"
               "end");
  verifyFormat("assert (x)\n"
               "else\n"
               "  if (x)\n"
               "    $error();");
  verifyFormat("assert (x)\n"
               "  $info();\n"
               "else\n"
               "  if (x)\n"
               "    $error();");
  verifyFormat("assert (x)\n"
               "  $info();\n"
               "else\n"
               "  if (x)\n"
               "    $error();\n"
               "  else\n"
               "    $error();");
  verifyFormat("assert (x)\n"
               "  $info();\n"
               "else\n"
               "  if (x)\n"
               "    $error();\n"
               "  else if (x)\n"
               "    $error();\n"
               "  else\n"
               "    $error();");
  // The body is optional for asserts.  The next line should not be indented if
  // the statement already ended with a semicolon.
  verifyFormat("assert (x);\n"
               "x = x;");
  verifyFormat("if (x)\n"
               "  assert (x);\n"
               "else if (x) begin\n"
               "end else begin\n"
               "end");
  verifyFormat("if (x)\n"
               "  assert (x);\n"
               "else begin\n"
               "end");
  verifyFormat("if (x)\n"
               "  assert (x)\n"
               "  else begin\n"
               "  end");
  // Other keywords.
  verifyFormat("assume (x)\n"
               "  $info();");
  verifyFormat("cover (x)\n"
               "  $info();");
  verifyFormat("restrict (x)\n"
               "  $info();");
  verifyFormat("assert #0 (x)\n"
               "  $info();");
  verifyFormat("assert final (x)\n"
               "  $info();");
  verifyFormat("cover #0 (x)\n"
               "  $info();");
  verifyFormat("cover final (x)\n"
               "  $info();");

  // The space around parentheses options should work.
  auto Style = getDefaultStyle();
  verifyFormat("if (x)\n"
               "  x = x;\n"
               "else if (x)\n"
               "  x = x;",
               Style);
  verifyFormat("assert (x);", Style);
  verifyFormat("assert #0 (x);", Style);
  verifyFormat("assert (x)\n"
               "else\n"
               "  if (x)\n"
               "    x = x;",
               Style);
  Style.SpacesInParens = FormatStyle::SIPO_Custom;
  Style.SpacesInParensOptions.InConditionalStatements = true;
  verifyFormat("if ( x )\n"
               "  x = x;\n"
               "else if ( x )\n"
               "  x = x;",
               Style);
  verifyFormat("assert ( x );", Style);
  verifyFormat("assert #0 ( x );", Style);
  verifyFormat("assert ( x )\n"
               "else\n"
               "  if ( x )\n"
               "    x = x;",
               Style);
}

TEST_F(FormatTestVerilog, Instantiation) {
  // Without ports.
  verifyFormat("ffnand ff1;");
  // With named ports.
  verifyFormat("ffnand ff1(.qbar(out1),\n"
               "           .clear(in1),\n"
               "           .preset(in2));");
  // With wildcard.
  verifyFormat("ffnand ff1(.qbar(out1),\n"
               "           .clear(in1),\n"
               "           .preset(in2),\n"
               "           .*);");
  verifyFormat("ffnand ff1(.*,\n"
               "           .qbar(out1),\n"
               "           .clear(in1),\n"
               "           .preset(in2));");
  // With unconnected ports.
  verifyFormat("ffnand ff1(.q(),\n"
               "           .qbar(out1),\n"
               "           .clear(in1),\n"
               "           .preset(in2));");
  verifyFormat("ffnand ff1(.q(),\n"
               "           .qbar(),\n"
               "           .clear(),\n"
               "           .preset());");
  verifyFormat("ffnand ff1(,\n"
               "           .qbar(out1),\n"
               "           .clear(in1),\n"
               "           .preset(in2));");
  // With positional ports.
  verifyFormat("ffnand ff1(out1,\n"
               "           in1,\n"
               "           in2);");
  verifyFormat("ffnand ff1(,\n"
               "           out1,\n"
               "           in1,\n"
               "           in2);");
  // Multiple instantiations.
  verifyFormat("ffnand ff1(.q(),\n"
               "           .qbar(out1),\n"
               "           .clear(in1),\n"
               "           .preset(in2)),\n"
               "       ff1(.q(),\n"
               "           .qbar(out1),\n"
               "           .clear(in1),\n"
               "           .preset(in2));");
  verifyFormat("ffnand //\n"
               "    ff1(.q(),\n"
               "        .qbar(out1),\n"
               "        .clear(in1),\n"
               "        .preset(in2)),\n"
               "    ff1(.q(),\n"
               "        .qbar(out1),\n"
               "        .clear(in1),\n"
               "        .preset(in2));");
  // With breaking between instance ports disabled.
  auto Style = getDefaultStyle();
  Style.VerilogBreakBetweenInstancePorts = false;
  verifyFormat("ffnand ff1;", Style);
  verifyFormat("ffnand ff1(.qbar(out1), .clear(in1), .preset(in2), .*);",
               Style);
  verifyFormat("ffnand ff1(out1, in1, in2);", Style);
  verifyFormat("ffnand ff1(.q(), .qbar(out1), .clear(in1), .preset(in2)),\n"
               "       ff1(.q(), .qbar(out1), .clear(in1), .preset(in2));",
               Style);
  verifyFormat("ffnand //\n"
               "    ff1(.q(), .qbar(out1), .clear(in1), .preset(in2)),\n"
               "    ff1(.q(), .qbar(out1), .clear(in1), .preset(in2));",
               Style);
}

TEST_F(FormatTestVerilog, Loop) {
  verifyFormat("foreach (x[x])\n"
               "  x = x;");
  verifyFormat("repeat (x)\n"
               "  x = x;");
  verifyFormat("foreach (x[x]) begin\n"
               "end");
  verifyFormat("repeat (x) begin\n"
               "end");
  auto Style = getDefaultStyle();
  Style.SpacesInParens = FormatStyle::SIPO_Custom;
  Style.SpacesInParensOptions.InConditionalStatements = true;
  verifyFormat("foreach ( x[x] )\n"
               "  x = x;",
               Style);
  verifyFormat("repeat ( x )\n"
               "  x = x;",
               Style);
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

  // Test that `*` and `*>` are binary.
  verifyFormat("x = x * x;");
  verifyFormat("x = (x * x);");
  verifyFormat("(opcode *> o1) = 6.1;");
  verifyFormat("(C, D *> Q) = 18;");
  // The wildcard import is not a binary operator.
  verifyFormat("import p::*;");

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
  verifyFormat("x = x -> x;");
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
  verifyFormat("x = x < -x;", "x=x<-x;");
  verifyFormat("x = x << -x;", "x=x<<-x;");
  verifyFormat("x = x <<< -x;", "x=x<<<-x;");

  // Test that operators that are C++ identifiers get treated as operators.
  verifyFormat("solve s before d;");                       // before
  verifyFormat("binsof(i) intersect {0};");                // intersect
  verifyFormat("req dist {1};");                           // dist
  verifyFormat("a inside {b, c};");                        // inside
  verifyFormat("bus.randomize() with { atype == low; };"); // with
}

TEST_F(FormatTestVerilog, Preprocessor) {
  auto Style = getDefaultStyle();
  Style.ColumnLimit = 20;

  // Macro definitions.
  verifyFormat("`define X          \\\n"
               "  if (x)           \\\n"
               "    x = x;",
               "`define X if(x)x=x;", Style);
  verifyFormat("`define X(x)       \\\n"
               "  if (x)           \\\n"
               "    x = x;",
               "`define X(x) if(x)x=x;", Style);
  verifyFormat("`define X          \\\n"
               "  x = x;           \\\n"
               "  x = x;",
               "`define X x=x;x=x;", Style);
  // Macro definitions with invocations inside.
  verifyFormat("`define LIST       \\\n"
               "  `ENTRY           \\\n"
               "  `ENTRY",
               "`define LIST \\\n"
               "`ENTRY \\\n"
               "`ENTRY",
               Style);
  verifyFormat("`define LIST       \\\n"
               "  `x = `x;         \\\n"
               "  `x = `x;",
               "`define LIST \\\n"
               "`x = `x; \\\n"
               "`x = `x;",
               Style);
  verifyFormat("`define LIST       \\\n"
               "  `x = `x;         \\\n"
               "  `x = `x;",
               "`define LIST `x=`x;`x=`x;", Style);
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
    verifyFormat("if (x)\n"
                 "`" +
                     Name +
                     "\n"
                     "  ;",
                 "if (x)\n"
                 "`" +
                     Name +
                     "\n"
                     ";",
                 Style);
  }
  // Lines starting with a regular macro invocation should be indented as a
  // normal line.
  verifyFormat("if (x)\n"
               "  `x = `x;\n"
               "`timescale 1ns / 1ps",
               "if (x)\n"
               "`x = `x;\n"
               "`timescale 1ns / 1ps",
               Style);
  verifyFormat("if (x)\n"
               "`timescale 1ns / 1ps\n"
               "  `x = `x;",
               "if (x)\n"
               "`timescale 1ns / 1ps\n"
               "`x = `x;",
               Style);
  std::string NonDirectives[] = {
      // For `__FILE__` and `__LINE__`, although the standard classifies them as
      // preprocessor directives, they are used like regular macros.
      "__FILE__", "__LINE__", "elif", "foo", "x",
  };
  for (auto &Name : NonDirectives) {
    verifyFormat("if (x)\n"
                 "  `" +
                     Name + ";",
                 "if (x)\n"
                 "`" +
                     Name +
                     "\n"
                     ";",
                 Style);
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

TEST_F(FormatTestVerilog, Streaming) {
  verifyFormat("x = {>>{j}};");
  verifyFormat("x = {>>byte{j}};");
  verifyFormat("x = {<<{j}};");
  verifyFormat("x = {<<byte{j}};");
  verifyFormat("x = {<<16{j}};");
  verifyFormat("x = {<<{8'b0011_0101}};");
  verifyFormat("x = {<<4{6'b11_0101}};");
  verifyFormat("x = {>>4{6'b11_0101}};");
  verifyFormat("x = {<<2{{<<{4'b1101}}}};");
  verifyFormat("bit [96 : 1] y = {>>{a, b, c}};");
  verifyFormat("int j = {>>{a, b, c}};");
  verifyFormat("{>>{a, b, c}} = 23'b1;");
  verifyFormat("{>>{a, b, c}} = x;");
  verifyFormat("{>>{j}} = x;");
  verifyFormat("{>>byte{j}} = x;");
  verifyFormat("{<<{j}} = x;");
  verifyFormat("{<<byte{j}} = x;");
}

TEST_F(FormatTestVerilog, StringLiteral) {
  // Long strings should be broken.
  verifyFormat(R"(x({"xxxxxxxxxxxxxxxx ",
   "xxxx"});)",
               R"(x({"xxxxxxxxxxxxxxxx xxxx"});)",
               getStyleWithColumns(getDefaultStyle(), 23));
  verifyFormat(R"(x({"xxxxxxxxxxxxxxxx",
   " xxxx"});)",
               R"(x({"xxxxxxxxxxxxxxxx xxxx"});)",
               getStyleWithColumns(getDefaultStyle(), 22));
  // Braces should be added when they don't already exist.
  verifyFormat(R"(x({"xxxxxxxxxxxxxxxx ",
   "xxxx"});)",
               R"(x("xxxxxxxxxxxxxxxx xxxx");)",
               getStyleWithColumns(getDefaultStyle(), 23));
  verifyFormat(R"(x({"xxxxxxxxxxxxxxxx",
   " xxxx"});)",
               R"(x("xxxxxxxxxxxxxxxx xxxx");)",
               getStyleWithColumns(getDefaultStyle(), 22));
  verifyFormat(R"(x({{"xxxxxxxxxxxxxxxx ",
    "xxxx"} == x});)",
               R"(x({"xxxxxxxxxxxxxxxx xxxx" == x});)",
               getStyleWithColumns(getDefaultStyle(), 24));
  verifyFormat(R"(string x = {"xxxxxxxxxxxxxxxx ",
            "xxxxxxxx"};)",
               R"(string x = "xxxxxxxxxxxxxxxx xxxxxxxx";)",
               getStyleWithColumns(getDefaultStyle(), 32));
  // Space around braces should be correct.
  auto Style = getStyleWithColumns(getDefaultStyle(), 24);
  Style.Cpp11BracedListStyle = false;
  verifyFormat(R"(x({ "xxxxxxxxxxxxxxxx ",
    "xxxx" });)",
               R"(x("xxxxxxxxxxxxxxxx xxxx");)", Style);
  // Braces should not be added twice.
  verifyFormat(R"(x({"xxxxxxxx",
   "xxxxxxxx",
   "xxxxxx"});)",
               R"(x("xxxxxxxxxxxxxxxxxxxxxx");)",
               getStyleWithColumns(getDefaultStyle(), 14));
  verifyFormat(R"(x({"xxxxxxxxxxxxxxxx ",
   "xxxxxxxxxxxxxxxx ",
   "xxxx"});)",
               R"(x("xxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxx xxxx");)",
               getStyleWithColumns(getDefaultStyle(), 23));
  verifyFormat(R"(x({"xxxxxxxxxxxxxxxx ",
   "xxxxxxxxxxxxxxxx ",
   "xxxx"});)",
               R"(x({"xxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxx ", "xxxx"});)",
               getStyleWithColumns(getDefaultStyle(), 23));
  // import/export "DPI"/"DPI-C" cannot be split.
  verifyFormat(R"(import
    "DPI-C" function void foo
    ();)",
               R"(import "DPI-C" function void foo();)",
               getStyleWithColumns(getDefaultStyle(), 23));
  verifyFormat(R"(export "DPI-C" function foo;)",
               R"(export "DPI-C" function foo;)",
               getStyleWithColumns(getDefaultStyle(), 23));
  // These kinds of strings don't exist in Verilog.
  verifyNoCrash(R"(x(@"xxxxxxxxxxxxxxxx xxxx");)",
                getStyleWithColumns(getDefaultStyle(), 23));
  verifyNoCrash(R"(x(u"xxxxxxxxxxxxxxxx xxxx");)",
                getStyleWithColumns(getDefaultStyle(), 23));
  verifyNoCrash(R"(x(u8"xxxxxxxxxxxxxxxx xxxx");)",
                getStyleWithColumns(getDefaultStyle(), 23));
  verifyNoCrash(R"(x(_T("xxxxxxxxxxxxxxxx xxxx"));)",
                getStyleWithColumns(getDefaultStyle(), 23));
}

TEST_F(FormatTestVerilog, StructLiteral) {
  verifyFormat("c = '{0, 0.0};");
  verifyFormat("c = '{'{1, 1.0}, '{2, 2.0}};");
  verifyFormat("c = '{a: 0, b: 0.0};");
  verifyFormat("c = '{a: 0, b: 0.0, default: 0};");
  verifyFormat("d = {int: 1, shortreal: 1.0};");
  verifyFormat("c = '{default: 0};");

  // The identifier before the quote can be either a tag or a type case.  There
  // should be a space between the tag and the quote.
  verifyFormat("c = ab'{a: 0, b: 0.0};");
  verifyFormat("c = ab'{cd: cd'{1, 1.0}, ef: ef'{2, 2.0}};");
  verifyFormat("c = ab'{cd'{1, 1.0}, ef'{2, 2.0}};");
  verifyFormat("d = ab'{int: 1, shortreal: 1.0};");
  verifyFormat("x = tagged Add '{e1, 4, ed};");

  auto Style = getDefaultStyle();
  Style.SpacesInContainerLiterals = true;
  verifyFormat("c = '{a : 0, b : 0.0};", Style);
  verifyFormat("c = '{a : 0, b : 0.0, default : 0};", Style);
  verifyFormat("c = ab'{a : 0, b : 0.0};", Style);
  verifyFormat("c = ab'{cd : cd'{1, 1.0}, ef : ef'{2, 2.0}};", Style);

  // It should be indented correctly when the line has to break.
  verifyFormat("c = //\n"
               "    '{default: 0};");
  Style = getDefaultStyle();
  Style.ContinuationIndentWidth = 2;
  verifyFormat("c = //\n"
               "  '{default: 0};",
               Style);
}

TEST_F(FormatTestVerilog, StructuredProcedure) {
  // Blocks should be indented correctly.
  verifyFormat("initial begin\n"
               "end");
  verifyFormat("initial begin\n"
               "  x <= x;\n"
               "  x <= x;\n"
               "end");
  verifyFormat("initial\n"
               "  x <= x;\n"
               "x <= x;");
  verifyFormat("always @(x) begin\n"
               "end");
  verifyFormat("always @(x) begin\n"
               "  x <= x;\n"
               "  x <= x;\n"
               "end");
  verifyFormat("always @(x)\n"
               "  x <= x;\n"
               "x <= x;");
  // Various keywords.
  verifyFormat("always @(x)\n"
               "  x <= x;");
  verifyFormat("always @(posedge x)\n"
               "  x <= x;");
  verifyFormat("always @(posedge x or posedge y)\n"
               "  x <= x;");
  verifyFormat("always @(posedge x, posedge y)\n"
               "  x <= x;");
  verifyFormat("always @(negedge x, negedge y)\n"
               "  x <= x;");
  verifyFormat("always @(edge x, edge y)\n"
               "  x <= x;");
  verifyFormat("always\n"
               "  x <= x;");
  verifyFormat("always @*\n"
               "  x <= x;");
  verifyFormat("always @(*)\n"
               "  x <= x;");
  verifyFormat("always_comb\n"
               "  x <= x;");
  verifyFormat("always_latch @(x)\n"
               "  x <= x;");
  verifyFormat("always_ff @(posedge x)\n"
               "  x <= x;");
  verifyFormat("initial\n"
               "  x <= x;");
  verifyFormat("final\n"
               "  x <= x;");
  verifyFormat("forever\n"
               "  x <= x;");
}
} // namespace
} // namespace test
} // namespace format
} // namespace clang
