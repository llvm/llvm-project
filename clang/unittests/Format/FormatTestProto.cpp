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

class FormatTestProto : public ::testing::Test {
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
    FormatStyle Style = getGoogleStyle(FormatStyle::LK_Proto);
    Style.ColumnLimit = 60; // To make writing tests easier.
    return format(Code, 0, Code.size(), Style);
  }

  static void verifyFormat(llvm::StringRef Code) {
    EXPECT_EQ(Code.str(), format(test::messUp(Code)));
  }
};

TEST_F(FormatTestProto, FormatsMessages) {
  verifyFormat("message SomeMessage {\n"
               "  required int32 field1 = 1;\n"
               "}");
  verifyFormat("message SomeMessage {\n"
               "  required .absolute.Reference field1 = 1;\n"
               "}");
  verifyFormat("message SomeMessage {\n"
               "  required int32 field1 = 1;\n"
               "  optional string field2 = 2 [default = \"2\"]\n"
               "}");

  verifyFormat("message SomeMessage {\n"
               "  optional really.really.long.qualified.type.aaa.aaaaaaa\n"
               "      fiiiiiiiiiiiiiiiiiiiiiiiiield = 1;\n"
               "  optional\n"
               "      really.really.long.qualified.type.aaa.aaaaaaa.aaaaaaaa\n"
               "          another_fiiiiiiiiiiiiiiiiiiiiield = 2;\n"
               "}");
  verifyFormat("message SomeMessage {\n"
               "  map<string, Project> projects = 1;\n"
               "  optional map<string, int32> size_projects = 2;\n"
               "  map<int, really.really.really.long.qualified.type.nameeee>\n"
               "      projects = 3;\n"
               "  map<int, really.really.really.really.long.qualified.type\n"
               "               .nameeee> projects = 4;\n"
               "  map<int,\n"
               "      reallyreallyreallyreallyreallyreallyreallylongname>\n"
               "      projects = 5;\n"
               "  map<int, Project>\n"
               "      longlonglonglonglonglonglonglonglonglongonglon = 6;\n"
               "  map<releleallyreallyreallyreallyreallyreallyreallylongname,\n"
               "      int> projects = 7;\n"
               "  map<releleallyreallyreallyreallyreallyreallyreallylongname,\n"
               "      releleallyreallyreallyreallyreallyreallyreallylongname>\n"
               "      releleallyreallyreallyreallyreallyreallyreallylongnam =\n"
               "          8;\n"
               "  map<relele.llyreal.yreallyr.allyreally.eallyreal\n"
               "          .sauenirylongname,\n"
               "      really.really.really.really.long.qualified.type\n"
               "          .nameeee> projects = 9;\n"
               "}");
}

TEST_F(FormatTestProto, KeywordsInOtherLanguages) {
  verifyFormat("optional string operator = 1;");
}

TEST_F(FormatTestProto, FormatsEnums) {
  verifyFormat("enum Type {\n"
               "  UNKNOWN = 0;\n"
               "  TYPE_A = 1;\n"
               "  TYPE_B = 2;\n"
               "};");
  verifyFormat("enum Type {\n"
               "  UNKNOWN = 0 [(some_options) = { a: aa, b: bb }];\n"
               "};");
  verifyFormat("enum Type {\n"
               "  UNKNOWN = 0 [(some_options) = {\n"
               "    a: aa,  // wrap\n"
               "    b: bb\n"
               "  }];\n"
               "};");
}

TEST_F(FormatTestProto, UnderstandsReturns) {
  verifyFormat("rpc Search(SearchRequest) returns (SearchResponse);");
}

TEST_F(FormatTestProto, MessageFieldAttributes) {
  verifyFormat("optional string test = 1 [default = \"test\"];");
  verifyFormat("optional bool a = 1 [default = true, deprecated = true];");
  verifyFormat("optional LongMessageType long_proto_field = 1 [\n"
               "  default = REALLY_REALLY_LONG_CONSTANT_VALUE,\n"
               "  deprecated = true\n"
               "];");
  verifyFormat("optional LongMessageType long_proto_field = 1\n"
               "    [default = REALLY_REALLY_LONG_CONSTANT_VALUE];");
  verifyFormat("repeated double value = 1\n"
               "    [(aaaaaaa.aaaaaaaaa) = { aaaaaaaaaaaaaaaaa: AAAAAAAA }];");
  verifyFormat("repeated double value = 1 [(aaaaaaa.aaaaaaaaa) = {\n"
               "  aaaaaaaaaaaaaaaa: AAAAAAAAAA,\n"
               "  bbbbbbbbbbbbbbbb: BBBBBBBBBB\n"
               "}];");
  verifyFormat("repeated double value = 1 [(aaaaaaa.aaaaaaaaa) = {\n"
               "  aaaaaaaaaaaaaaaa: AAAAAAAAAA\n"
               "  bbbbbbbbbbbbbbbb: BBBBBBBBBB\n"
               "}];");
  verifyFormat("repeated double value = 1 [\n"
               "  (aaaaaaa.aaaaaaaaa) = {\n"
               "    aaaaaaaaaaaaaaaa: AAAAAAAAAA\n"
               "    bbbbbbbbbbbbbbbb: BBBBBBBBBB\n"
               "  },\n"
               "  (bbbbbbb.bbbbbbbbb) = {\n"
               "    aaaaaaaaaaaaaaaa: AAAAAAAAAA\n"
               "    bbbbbbbbbbbbbbbb: BBBBBBBBBB\n"
               "  }\n"
               "];");
  verifyFormat("repeated double value = 1 [(aaaaaaa.aaaaaaaaa) = {\n"
               "  type: \"AAAAAAAAAA\"\n"
               "  is: \"AAAAAAAAAA\"\n"
               "  or: \"BBBBBBBBBB\"\n"
               "}];");
  verifyFormat("repeated double value = 1 [(aaaaaaa.aaaaaaaaa) = {\n"
               "  aaaaaaaaaaaaaaaa: AAAAAAAAAA,\n"
               "  bbbbbbb: BBBB,\n"
               "  bbbb: BBB\n"
               "}];");
  verifyFormat("optional AAA aaa = 1 [\n"
               "  foo = {\n"
               "    key: 'a'  //\n"
               "  },\n"
               "  bar = {\n"
               "    key: 'a'  //\n"
               "  }\n"
               "];");
  verifyFormat("optional string test = 1 [default =\n"
               "                              \"test\"\n"
               "                              \"test\"];");
  verifyFormat("optional Aaaaaaaa aaaaaaaa = 12 [\n"
               "  (aaa) = aaaa,\n"
               "  (bbbbbbbbbbbbbbbbbbbbbbbbbb) = {\n"
               "    aaaaaaaaaaaaaaaaa: true,\n"
               "    aaaaaaaaaaaaaaaa: true\n"
               "  }\n"
               "];");
}

TEST_F(FormatTestProto, DoesntWrapFileOptions) {
  EXPECT_EQ(
      "option java_package = "
      "\"some.really.long.package.that.exceeds.the.column.limit\";",
      format("option    java_package   =    "
             "\"some.really.long.package.that.exceeds.the.column.limit\";"));
}

TEST_F(FormatTestProto, FormatsOptions) {
  verifyFormat("option (MyProto.options) = {\n"
               "  field_a: OK\n"
               "  field_b: \"OK\"\n"
               "  field_c: \"OK\"\n"
               "  msg_field: { field_d: 123 }\n"
               "};");
  verifyFormat("option (MyProto.options) = {\n"
               "  field_a: OK\n"
               "  field_b: \"OK\"\n"
               "  field_c: \"OK\"\n"
               "  msg_field: { field_d: 123 field_e: OK }\n"
               "};");
  verifyFormat("option (MyProto.options) = {\n"
               "  field_a: OK  // Comment\n"
               "  field_b: \"OK\"\n"
               "  field_c: \"OK\"\n"
               "  msg_field: { field_d: 123 }\n"
               "};");
  verifyFormat("option (MyProto.options) = {\n"
               "  field_c: \"OK\"\n"
               "  msg_field { field_d: 123 }\n"
               "};");
  verifyFormat("option (MyProto.options) = {\n"
               "  field_a: OK\n"
               "  field_b { field_c: OK }\n"
               "  field_d: OKOKOK\n"
               "  field_e: OK\n"
               "}");

  // Support syntax with <> instead of {}.
  verifyFormat("option (MyProto.options) = {\n"
               "  field_c: \"OK\",\n"
               "  msg_field: < field_d: 123 >\n"
               "  empty: <>\n"
               "  empty <>\n"
               "};");

  verifyFormat("option (MyProto.options) = {\n"
               "  field_a: OK\n"
               "  field_b < field_c: OK >\n"
               "  field_d: OKOKOK\n"
               "  field_e: OK\n"
               "}");

  verifyFormat("option (MyProto.options) = {\n"
               "  msg_field: <>\n"
               "  field_c: \"OK\",\n"
               "  msg_field: < field_d: 123 >\n"
               "  field_e: OK\n"
               "  msg_field: < field_d: 12 >\n"
               "};");

  verifyFormat("option (MyProto.options) = <\n"
               "  field_a: OK\n"
               "  field_b: \"OK\"\n"
               "  field_c: 1\n"
               "  field_d: 12.5\n"
               "  field_e: OK\n"
               ">;");

  verifyFormat("option (MyProto.options) = <\n"
               "  field_a: OK,\n"
               "  field_b: \"OK\",\n"
               "  field_c: 1,\n"
               "  field_d: 12.5,\n"
               "  field_e: OK,\n"
               ">;");

  verifyFormat("option (MyProto.options) = <\n"
               "  field_a: \"OK\"\n"
               "  msg_field: { field_b: OK }\n"
               "  field_g: OK\n"
               "  field_g: OK\n"
               "  field_g: OK\n"
               ">;");

  verifyFormat("option (MyProto.options) = <\n"
               "  field_a: \"OK\"\n"
               "  msg_field <\n"
               "    field_b: OK\n"
               "    field_c: OK\n"
               "    field_d: OK\n"
               "    field_e: OK\n"
               "    field_f: OK\n"
               "  >\n"
               "  field_g: OK\n"
               ">;");

  verifyFormat("option (MyProto.options) = <\n"
               "  field_a: \"OK\"\n"
               "  msg_field <\n"
               "    field_b: OK,\n"
               "    field_c: OK,\n"
               "    field_d: OK,\n"
               "    field_e: OK,\n"
               "    field_f: OK\n"
               "  >\n"
               "  field_g: OK\n"
               ">;");

  verifyFormat("option (MyProto.options) = <\n"
               "  field_a: \"OK\"\n"
               "  msg_field: <\n"
               "    field_b: OK\n"
               "    field_c: OK\n"
               "    field_d: OK\n"
               "    field_e: OK\n"
               "    field_f: OK\n"
               "  >\n"
               "  field_g: OK\n"
               ">;");

  verifyFormat("option (MyProto.options) = <\n"
               "  field_a: \"OK\"\n"
               "  msg_field: {\n"
               "    field_b: OK\n"
               "    field_c: OK\n"
               "    field_d: OK\n"
               "    field_e: OK\n"
               "    field_f: OK\n"
               "  }\n"
               "  field_g: OK\n"
               ">;");

  verifyFormat("option (MyProto.options) = <\n"
               "  field_a: \"OK\"\n"
               "  msg_field {\n"
               "    field_b: OK\n"
               "    field_c: OK\n"
               "    field_d: OK\n"
               "    field_e: OK\n"
               "    field_f: OK\n"
               "  }\n"
               "  field_g: OK\n"
               ">;");

  verifyFormat("option (MyProto.options) = {\n"
               "  field_a: \"OK\"\n"
               "  msg_field <\n"
               "    field_b: OK\n"
               "    field_c: OK\n"
               "    field_d: OK\n"
               "    field_e: OK\n"
               "    field_f: OK\n"
               "  >\n"
               "  field_g: OK\n"
               "};");

  verifyFormat("option (MyProto.options) = {\n"
               "  field_a: \"OK\"\n"
               "  msg_field: <\n"
               "    field_b: OK\n"
               "    field_c: OK\n"
               "    field_d: OK\n"
               "    field_e: OK\n"
               "    field_f: OK\n"
               "  >\n"
               "  field_g: OK\n"
               "};");

  verifyFormat("option (MyProto.options) = <\n"
               "  field_a: \"OK\"\n"
               "  msg_field {\n"
               "    field_b: OK\n"
               "    field_c: OK\n"
               "    field_d: OK\n"
               "    msg_field <\n"
               "      field_A: 1\n"
               "      field_B: 2\n"
               "      field_C: 3\n"
               "      field_D: 4\n"
               "      field_E: 5\n"
               "    >\n"
               "    msg_field < field_A: 1 field_B: 2 field_C: 3 f_D: 4 >\n"
               "    field_e: OK\n"
               "    field_f: OK\n"
               "  }\n"
               "  field_g: OK\n"
               ">;");

  verifyFormat("option (MyProto.options) = <\n"
               "  data1 < key1: value1 >\n"
               "  data2 { key2: value2 }\n"
               ">;");

  verifyFormat("option (MyProto.options) = <\n"
               "  app_id: 'com.javax.swing.salsa.latino'\n"
               "  head_id: 1\n"
               "  data < key: value >\n"
               ">;");

  verifyFormat("option (MyProto.options) = {\n"
               "  app_id: 'com.javax.swing.salsa.latino'\n"
               "  head_id: 1\n"
               "  headheadheadheadheadhead_id: 1\n"
               "  product_data { product { 1 } }\n"
               "};");
}

TEST_F(FormatTestProto, FormatsService) {
  verifyFormat("service SearchService {\n"
               "  rpc Search(SearchRequest) returns (SearchResponse) {\n"
               "    option foo = true;\n"
               "  }\n"
               "};");
}

TEST_F(FormatTestProto, ExtendingMessage) {
  verifyFormat("extend .foo.Bar {\n"
               "}");
}

TEST_F(FormatTestProto, FormatsImports) {
  verifyFormat("import \"a.proto\";\n"
               "import \"b.proto\";\n"
               "// comment\n"
               "message A {\n"
               "}");

  verifyFormat("import public \"a.proto\";\n"
               "import \"b.proto\";\n"
               "// comment\n"
               "message A {\n"
               "}");

  // Missing semicolons should not confuse clang-format.
  verifyFormat("import \"a.proto\"\n"
               "import \"b.proto\"\n"
               "// comment\n"
               "message A {\n"
               "}");
}

TEST_F(FormatTestProto, KeepsLongStringLiteralsOnSameLine) {
  verifyFormat(
      "option (MyProto.options) = {\n"
      "  foo: {\n"
      "    text: \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaasaaaaaaaa\"\n"
      "  }\n"
      "}");
}

} // end namespace tooling
} // end namespace clang
