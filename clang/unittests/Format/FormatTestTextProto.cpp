//===- unittest/Format/FormatTestProto.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FormatTestUtils.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {

class FormatTestTextProto : public ::testing::Test {
protected:
  static std::string format(llvm::StringRef Code, unsigned Offset,
                            unsigned Length, const FormatStyle &Style) {
    DEBUG(llvm::errs() << "---\n");
    DEBUG(llvm::errs() << Code << "\n\n");
    std::vector<tooling::Range> Ranges(1, tooling::Range(Offset, Length));
    tooling::Replacements Replaces = reformat(Style, Code, Ranges);
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  static std::string format(llvm::StringRef Code) {
    FormatStyle Style = getGoogleStyle(FormatStyle::LK_TextProto);
    Style.ColumnLimit = 60; // To make writing tests easier.
    return format(Code, 0, Code.size(), Style);
  }

  static void verifyFormat(llvm::StringRef Code) {
    EXPECT_EQ(Code.str(), format(test::messUp(Code)));
  }
};

TEST_F(FormatTestTextProto, KeepsTopLevelEntriesFittingALine) {
  verifyFormat("field_a: OK field_b: OK field_c: OK field_d: OK field_e: OK");
}

TEST_F(FormatTestTextProto, SupportsMessageFields) {
  verifyFormat("msg_field: {}");

  verifyFormat("msg_field: { field_a: A }");

  verifyFormat("msg_field: { field_a: \"OK\" field_b: 123 }");

  verifyFormat("msg_field: {\n"
               "  field_a: 1\n"
               "  field_b: OK\n"
               "  field_c: \"OK\"\n"
               "  field_d: 123\n"
               "  field_e: 23\n"
               "}");

  verifyFormat("msg_field {}");

  verifyFormat("msg_field { field_a: A }");

  verifyFormat("msg_field { field_a: \"OK\" field_b: 123 }");

  verifyFormat("msg_field {\n"
               "  field_a: 1\n"
               "  field_b: OK\n"
               "  field_c: \"OK\"\n"
               "  field_d: 123\n"
               "  field_e: 23.0\n"
               "  field_f: false\n"
               "  field_g: 'lala'\n"
               "  field_h: 1234.567e-89\n"
               "}");

  verifyFormat("msg_field: { msg_field { field_a: 1 } }");

  verifyFormat("id: \"ala.bala\"\n"
               "item { type: ITEM_A rank: 1 score: 90.0 }\n"
               "item { type: ITEM_B rank: 2 score: 70.5 }\n"
               "item {\n"
               "  type: ITEM_A\n"
               "  rank: 3\n"
               "  score: 20.0\n"
               "  description: \"the third item has a description\"\n"
               "}");
}

TEST_F(FormatTestTextProto, AvoidsTopLevelBinPacking) {
  verifyFormat("field_a: OK\n"
               "field_b: OK\n"
               "field_c: OK\n"
               "field_d: OK\n"
               "field_e: OK\n"
               "field_f: OK");

  verifyFormat("field_a: OK\n"
               "field_b: \"OK\"\n"
               "field_c: \"OK\"\n"
               "msg_field: { field_d: 123 }\n"
               "field_e: OK\n"
               "field_f: OK");

  verifyFormat("field_a: OK\n"
               "field_b: \"OK\"\n"
               "field_c: \"OK\"\n"
               "msg_field: { field_d: 123 field_e: OK }");

  verifyFormat("a: {\n"
               "  field_a: OK\n"
               "  field_b { field_c: OK }\n"
               "  field_d: OKOKOK\n"
               "  field_e: OK\n"
               "}");

  verifyFormat("field_a: OK,\n"
               "field_b { field_c: OK },\n"
               "field_d: OKOKOK,\n"
               "field_e: OK");
}

TEST_F(FormatTestTextProto, AddsNewlinesAfterTrailingComments) {
  verifyFormat("field_a: OK  // Comment\n"
               "field_b: 1");

  verifyFormat("field_a: OK\n"
               "msg_field: {\n"
               "  field_b: OK  // Comment\n"
               "}");

  verifyFormat("field_a: OK\n"
               "msg_field {\n"
               "  field_b: OK  // Comment\n"
               "}");
}

TEST_F(FormatTestTextProto, SupportsAngleBracketMessageFields) {
  // Single-line tests
  verifyFormat("msg_field <>");
  verifyFormat("msg_field: <>");
  verifyFormat("msg_field < field_a: OK >");
  verifyFormat("msg_field: < field_a: 123 >");
  verifyFormat("msg_field < field_a <> >");
  verifyFormat("msg_field < field_a < field_b <> > >");
  verifyFormat("msg_field: < field_a < field_b: <> > >");
  verifyFormat("msg_field < field_a: OK, field_b: \"OK\" >");
  verifyFormat("msg_field < field_a: OK field_b: <>, field_c: OK >");
  verifyFormat("msg_field < field_a { field_b: 1 }, field_c: < f_d: 2 > >");
  verifyFormat("msg_field: < field_a: OK, field_b: \"OK\" >");
  verifyFormat("msg_field: < field_a: OK field_b: <>, field_c: OK >");
  verifyFormat("msg_field: < field_a { field_b: 1 }, field_c: < fd_d: 2 > >");
  verifyFormat("field_a: \"OK\", msg_field: < field_b: 123 >, field_c: {}");
  verifyFormat("field_a < field_b: 1 >, msg_fid: < fiel_b: 123 >, field_c <>");
  verifyFormat("field_a < field_b: 1 > msg_fied: < field_b: 123 > field_c <>");
  verifyFormat("field < field < field: <> >, field <> > field: < field: 1 >");

  // Multiple lines tests
  verifyFormat("msg_field <\n"
               "  field_a: OK\n"
               "  field_b: \"OK\"\n"
               "  field_c: 1\n"
               "  field_d: 12.5\n"
               "  field_e: OK\n"
               ">");

  verifyFormat("msg_field: <>\n"
               "field_c: \"OK\",\n"
               "msg_field: < field_d: 123 >\n"
               "field_e: OK\n"
               "msg_field: < field_d: 12 >");

  verifyFormat("field_a: OK,\n"
               "field_b < field_c: OK >,\n"
               "field_d: < 12.5 >,\n"
               "field_e: OK");

  verifyFormat("field_a: OK\n"
               "field_b < field_c: OK >\n"
               "field_d: < 12.5 >\n"
               "field_e: OKOKOK");

  verifyFormat("msg_field <\n"
               "  field_a: OK,\n"
               "  field_b < field_c: OK >,\n"
               "  field_d: < 12.5 >,\n"
               "  field_e: OK\n"
               ">");

  verifyFormat("msg_field <\n"
               "  field_a: < field: OK >,\n"
               "  field_b < field_c: OK >,\n"
               "  field_d: < 12.5 >,\n"
               "  field_e: OK,\n"
               ">");

  verifyFormat("msg_field: <\n"
               "  field_a: \"OK\"\n"
               "  msg_field: { field_b: OK }\n"
               "  field_g: OK\n"
               "  field_g: OK\n"
               "  field_g: OK\n"
               ">");

  verifyFormat("field_a {\n"
               "  field_d: ok\n"
               "  field_b: < field_c: 1 >\n"
               "  field_d: ok\n"
               "  field_d: ok\n"
               "}");

  verifyFormat("field_a: {\n"
               "  field_d: ok\n"
               "  field_b: < field_c: 1 >\n"
               "  field_d: ok\n"
               "  field_d: ok\n"
               "}");

  verifyFormat("field_a: < f1: 1, f2: <> >\n"
               "field_b <\n"
               "  field_b1: <>\n"
               "  field_b2: ok,\n"
               "  field_b3: <\n"
               "    field_x {}  // Comment\n"
               "    field_y: { field_z: 1 }\n"
               "    field_w: ok\n"
               "  >\n"
               "  field {\n"
               "    field_x <>  // Comment\n"
               "    field_y: < field_z: 1 >\n"
               "    field_w: ok\n"
               "    msg_field: <\n"
               "      field: <>\n"
               "      field: < field: 1 >\n"
               "      field: < field: 2 >\n"
               "      field: < field: 3 >\n"
               "      field: < field: 4 >\n"
               "      field: ok\n"
               "    >\n"
               "  }\n"
               ">\n"
               "field: OK,\n"
               "field_c < field < field <> > >");

  verifyFormat("app_id: 'com.javax.swing.salsa.latino'\n"
               "head_id: 1\n"
               "data < key: value >");

  verifyFormat("app_id: 'com.javax.swing.salsa.latino'\n"
               "head_id: 1\n"
               "data < key: value >\n"
               "tail_id: 2");

  verifyFormat("app_id: 'com.javax.swing.salsa.latino'\n"
               "head_id: 1\n"
               "data < key: value >\n"
               "data { key: value }");

  verifyFormat("app {\n"
               "  app_id: 'com.javax.swing.salsa.latino'\n"
               "  head_id: 1\n"
               "  data < key: value >\n"
               "}");

  verifyFormat("app: {\n"
               "  app_id: 'com.javax.swing.salsa.latino'\n"
               "  head_id: 1\n"
               "  data < key: value >\n"
               "}");

  verifyFormat("app_id: 'com.javax.swing.salsa.latino'\n"
               "headheadheadheadheadhead_id: 1\n"
               "product_data { product { 1 } }");

  verifyFormat("app_id: 'com.javax.swing.salsa.latino'\n"
               "headheadheadheadheadhead_id: 1\n"
               "product_data < product { 1 } >");

  verifyFormat("app_id: 'com.javax.swing.salsa.latino'\n"
               "headheadheadheadheadhead_id: 1\n"
               "product_data < product < 1 > >");

  verifyFormat("app <\n"
               "  app_id: 'com.javax.swing.salsa.latino'\n"
               "  headheadheadheadheadhead_id: 1\n"
               "  product_data < product { 1 } >\n"
               ">");

  verifyFormat("dcccwrnfioeruvginerurneitinfo {\n"
               "  exte3nsionrnfvui { key: value }\n"
               "}");
}

TEST_F(FormatTestTextProto, DiscardsUnbreakableTailIfCanBreakAfter) {
  // The two closing braces count towards the string UnbreakableTailLength, but
  // since we have broken after the corresponding opening braces, we don't
  // consider that length for string breaking.
  verifyFormat(
      "foo: {\n"
      "  bar: {\n"
      "    text: \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\n"
      "  }\n"
      "}");
}

TEST_F(FormatTestTextProto, KeepsLongStringLiteralsOnSameLine) {
  verifyFormat(
      "foo: {\n"
      "  text: \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaasaaaaaaaaaa\"\n"
      "}");
}
} // end namespace tooling
} // end namespace clang
