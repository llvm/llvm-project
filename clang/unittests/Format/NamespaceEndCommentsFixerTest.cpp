//===- NamespaceEndCommentsFixerTest.cpp - Formatting unit tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"

#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "namespace-end-comments-fixer-test"

namespace clang {
namespace format {
namespace {

class NamespaceEndCommentsFixerTest : public testing::Test {
protected:
  std::string
  fixNamespaceEndComments(StringRef Code,
                          const std::vector<tooling::Range> &Ranges,
                          const FormatStyle &Style = getLLVMStyle()) {
    LLVM_DEBUG(llvm::errs() << "---\n");
    LLVM_DEBUG(llvm::errs() << Code << "\n\n");
    tooling::Replacements Replaces =
        format::fixNamespaceEndComments(Style, Code, Ranges, "<stdin>");
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    LLVM_DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  std::string
  fixNamespaceEndComments(StringRef Code,
                          const FormatStyle &Style = getLLVMStyle()) {
    return fixNamespaceEndComments(
        Code,
        /*Ranges=*/{1, tooling::Range(0, Code.size())}, Style);
  }

  bool isFormatted(StringRef Code, const std::vector<tooling::Range> &Ranges,
                   const FormatStyle &Style = getLLVMStyle()) const {
    return format::fixNamespaceEndComments(Style, Code, Ranges, "<stdin>")
        .empty();
  }

  bool isFormatted(StringRef Code,
                   const FormatStyle &Style = getLLVMStyle()) const {
    return isFormatted(Code, {1, tooling::Range(0, Code.size())}, Style);
  }
};

TEST_F(NamespaceEndCommentsFixerTest, AddsEndComment) {
  EXPECT_EQ("namespace {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace",
            fixNamespaceEndComments("namespace {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));

  EXPECT_EQ("namespace {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace",
            fixNamespaceEndComments("namespace {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
  EXPECT_EQ("#define M(x) x##x\n"
            "namespace M(x) {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace M(x)",
            fixNamespaceEndComments("#define M(x) x##x\n"
                                    "namespace M(x) {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
  EXPECT_EQ("#define M(x) x##x\n"
            "namespace A::M(x) {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A::M(x)",
            fixNamespaceEndComments("#define M(x) x##x\n"
                                    "namespace A::M(x) {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
  EXPECT_EQ("#define M(x) x##x\n"
            "namespace M(x)::A {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace M(x)::A",
            fixNamespaceEndComments("#define M(x) x##x\n"
                                    "namespace M(x)::A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
  EXPECT_EQ("#define M(x) x##x\n"
            "namespace A::inline M(x)::B {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A::inline M(x)::B",
            fixNamespaceEndComments("#define M(x) x##x\n"
                                    "namespace A::inline M(x)::B {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
  EXPECT_EQ("#define M(x) x##x\n"
            "namespace [[deprecated(\"foo\")]] A::inline M(x)::A {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A::inline M(x)::A",
            fixNamespaceEndComments(
                "#define M(x) x##x\n"
                "namespace [[deprecated(\"foo\")]] A::inline M(x)::A {\n"
                "int i;\n"
                "int j;\n"
                "}"));
  EXPECT_EQ(
      "namespace /* comment */ [[deprecated(\"foo\")]] /* comment */ A {\n"
      "int i;\n"
      "int j;\n"
      "} // namespace A",
      fixNamespaceEndComments(
          "namespace /* comment */ [[deprecated(\"foo\")]] /* comment */ A {\n"
          "int i;\n"
          "int j;\n"
          "}"));
  EXPECT_EQ("namespace /* comment */ [[deprecated(\"foo\")]] A {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A",
            fixNamespaceEndComments(
                "namespace /* comment */ [[deprecated(\"foo\")]] A {\n"
                "int i;\n"
                "int j;\n"
                "}"));
  EXPECT_EQ(
      "#define M(x) x##x\n"
      "namespace /* comment */ [[deprecated(\"foo\")]] /* comment */ M(x) {\n"
      "int i;\n"
      "int j;\n"
      "} // namespace M(x)",
      fixNamespaceEndComments("#define M(x) x##x\n"
                              "namespace /* comment */ "
                              "[[deprecated(\"foo\")]] /* comment */ M(x) {\n"
                              "int i;\n"
                              "int j;\n"
                              "}"));
  EXPECT_EQ("#define M(x) x##x\n"
            "namespace /* comment */ [[deprecated(\"foo\")]] /* comment */ "
            "A::M(x) {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A::M(x)",
            fixNamespaceEndComments(
                "#define M(x) x##x\n"
                "namespace /* comment */ "
                "[[deprecated(\"foo\")]] /* comment */ A::M(x) {\n"
                "int i;\n"
                "int j;\n"
                "}"));
  EXPECT_EQ("#define M(x) x##x\n"
            "namespace /* comment */ [[deprecated(\"foo\")]] /* comment */ "
            "M(x) /* comment */ {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace M(x)",
            fixNamespaceEndComments(
                "#define M(x) x##x\n"
                "namespace /* comment */ [[deprecated(\"foo\")]] /* comment "
                "*/ M(x) /* comment */ {\n"
                "int i;\n"
                "int j;\n"
                "}"));
  EXPECT_EQ("#define M(x) x##x\n"
            "namespace /* comment */ [[deprecated(\"foo\")]] /* comment */ "
            "A::M(x) /* comment */ {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A::M(x)",
            fixNamespaceEndComments(
                "#define M(x) x##x\n"
                "namespace /* comment */ [[deprecated(\"foo\")]] /* comment "
                "*/ A::M(x) /* comment */ {\n"
                "int i;\n"
                "int j;\n"
                "}"));
  EXPECT_EQ("#define M(x) x##x\n"
            "namespace A M(x) {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A M(x)",
            fixNamespaceEndComments("#define M(x) x##x\n"
                                    "namespace A M(x) {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
  EXPECT_EQ(
      "#define B __attribute__((availability(macos, introduced=10.15)))\n"
      "namespace A B {\n"
      "int i;\n"
      "int j;\n"
      "} // namespace A B",
      fixNamespaceEndComments(
          "#define B __attribute__((availability(macos, introduced=10.15)))\n"
          "namespace A B {\n"
          "int i;\n"
          "int j;\n"
          "}"));
  EXPECT_EQ("#define M(x) x##x\n"
            "namespace A::B M(x) {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A::B",
            fixNamespaceEndComments("#define M(x) x##x\n"
                                    "namespace A::B M(x) {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
  EXPECT_EQ(
      "namespace A __attribute__((availability(macos, introduced=10.15))) {\n"
      "int i;\n"
      "int j;\n"
      "} // namespace A",
      fixNamespaceEndComments(
          "namespace A __attribute__((availability(macos, introduced=10.15))) "
          "{\n"
          "int i;\n"
          "int j;\n"
          "}"));
  EXPECT_EQ("inline namespace A {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A",
            fixNamespaceEndComments("inline namespace A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));

  EXPECT_EQ("namespace [[deprecated(\"foo\")]] A::B {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A::B",
            fixNamespaceEndComments("namespace [[deprecated(\"foo\")]] A::B {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));

  EXPECT_EQ("namespace [[deprecated(\"foo\")]] A::inline B::inline C {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A::inline B::inline C",
            fixNamespaceEndComments(
                "namespace [[deprecated(\"foo\")]] A::inline B::inline C {\n"
                "int i;\n"
                "int j;\n"
                "}"));

  EXPECT_EQ("namespace DEPRECATED A::B {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A::B",
            fixNamespaceEndComments("namespace DEPRECATED A::B {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));

  EXPECT_EQ("inline namespace [[deprecated]] A {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A",
            fixNamespaceEndComments("inline namespace [[deprecated]] A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));

  EXPECT_EQ("namespace ::A {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace ::A",
            fixNamespaceEndComments("namespace ::A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
  EXPECT_EQ("namespace ::A::B {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace ::A::B",
            fixNamespaceEndComments("namespace ::A::B {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
  EXPECT_EQ("namespace /**/::/**/A/**/::/**/B/**/ {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace ::A::B",
            fixNamespaceEndComments("namespace /**/::/**/A/**/::/**/B/**/ {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
  EXPECT_EQ("namespace A {\n"
            "namespace B {\n"
            "int i;\n"
            "}\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "namespace B {\n"
                                    "int i;\n"
                                    "}\n"
                                    "}"));
  EXPECT_EQ("namespace A {\n"
            "namespace B {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace B\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "namespace B {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}\n"
                                    "}"));
  EXPECT_EQ("namespace A {\n"
            "int a;\n"
            "int b;\n"
            "} // namespace A\n"
            "namespace B {\n"
            "int b;\n"
            "int a;\n"
            "} // namespace B",
            fixNamespaceEndComments("namespace A {\n"
                                    "int a;\n"
                                    "int b;\n"
                                    "}\n"
                                    "namespace B {\n"
                                    "int b;\n"
                                    "int a;\n"
                                    "}"));
  EXPECT_EQ("namespace A {\n"
            "int a1;\n"
            "int a2;\n"
            "} // namespace A\n"
            "namespace A {\n"
            "int a2;\n"
            "int a1;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "int a1;\n"
                                    "int a2;\n"
                                    "}\n"
                                    "namespace A {\n"
                                    "int a2;\n"
                                    "int a1;\n"
                                    "}"));
  EXPECT_EQ("namespace A {\n"
            "int a;\n"
            "int b;\n"
            "} // namespace A\n"
            "// comment about b\n"
            "int b;",
            fixNamespaceEndComments("namespace A {\n"
                                    "int a;\n"
                                    "int b;\n"
                                    "}\n"
                                    "// comment about b\n"
                                    "int b;"));

  EXPECT_EQ("namespace A {\n"
            "namespace B {\n"
            "namespace C {\n"
            "namespace D {\n"
            "}\n"
            "} // namespace C\n"
            "} // namespace B\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "namespace B {\n"
                                    "namespace C {\n"
                                    "namespace D {\n"
                                    "}\n"
                                    "}\n"
                                    "}\n"
                                    "}"));

  // Add comment for namespaces which will be 'compacted'
  FormatStyle CompactNamespacesStyle = getLLVMStyle();
  CompactNamespacesStyle.CompactNamespaces = true;
  EXPECT_EQ("namespace out { namespace in {\n"
            "int i;\n"
            "int j;\n"
            "}} // namespace out::in",
            fixNamespaceEndComments("namespace out { namespace in {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}}",
                                    CompactNamespacesStyle));
  EXPECT_EQ("namespace out {\n"
            "namespace in {\n"
            "int i;\n"
            "int j;\n"
            "}\n"
            "} // namespace out::in",
            fixNamespaceEndComments("namespace out {\n"
                                    "namespace in {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}\n"
                                    "}",
                                    CompactNamespacesStyle));
  EXPECT_EQ("namespace out { namespace in {\n"
            "int i;\n"
            "int j;\n"
            "};} // namespace out::in",
            fixNamespaceEndComments("namespace out { namespace in {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "};}",
                                    CompactNamespacesStyle));
  EXPECT_EQ("namespace out { namespace {\n"
            "int i;\n"
            "int j;\n"
            "}} // namespace out",
            fixNamespaceEndComments("namespace out { namespace {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}}",
                                    CompactNamespacesStyle));

  // Adds an end comment after a semicolon.
  EXPECT_EQ("namespace {\n"
            "int i;\n"
            "int j;\n"
            "}; // namespace",
            fixNamespaceEndComments("namespace {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "};"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "int j;\n"
            "}; // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "};"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "int j;\n"
            "}; // namespace A\n"
            "// unrelated",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "};\n"
                                    "// unrelated"));
}

TEST_F(NamespaceEndCommentsFixerTest, WorksForObjCpp) {
  FormatStyle ObjCppStyle = getLLVMStyle();
  ObjCppStyle.Language = FormatStyle::LK_ObjC;
  EXPECT_EQ("namespace {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace",
            fixNamespaceEndComments("namespace {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}",
                                    ObjCppStyle));
}

TEST_F(NamespaceEndCommentsFixerTest, AddsMacroEndComment) {
  FormatStyle Style = getLLVMStyle();
  Style.NamespaceMacros.push_back("TESTSUITE");

  EXPECT_EQ("TESTSUITE() {\n"
            "int i;\n"
            "int j;\n"
            "} // TESTSUITE()",
            fixNamespaceEndComments("TESTSUITE() {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}",
                                    Style));

  EXPECT_EQ("TESTSUITE(A) {\n"
            "int i;\n"
            "int j;\n"
            "} // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}",
                                    Style));
  EXPECT_EQ("inline TESTSUITE(A) {\n"
            "int i;\n"
            "int j;\n"
            "} // TESTSUITE(A)",
            fixNamespaceEndComments("inline TESTSUITE(A) {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}",
                                    Style));
  EXPECT_EQ("TESTSUITE(::A) {\n"
            "int i;\n"
            "int j;\n"
            "} // TESTSUITE(::A)",
            fixNamespaceEndComments("TESTSUITE(::A) {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}",
                                    Style));
  EXPECT_EQ("TESTSUITE(::A::B) {\n"
            "int i;\n"
            "int j;\n"
            "} // TESTSUITE(::A::B)",
            fixNamespaceEndComments("TESTSUITE(::A::B) {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}",
                                    Style));
  EXPECT_EQ("TESTSUITE(/**/::/**/A/**/::/**/B/**/) {\n"
            "int i;\n"
            "int j;\n"
            "} // TESTSUITE(::A::B)",
            fixNamespaceEndComments("TESTSUITE(/**/::/**/A/**/::/**/B/**/) {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}",
                                    Style));
  EXPECT_EQ("TESTSUITE(A, B) {\n"
            "int i;\n"
            "int j;\n"
            "} // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A, B) {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}",
                                    Style));
  EXPECT_EQ("TESTSUITE(\"Test1\") {\n"
            "int i;\n"
            "int j;\n"
            "} // TESTSUITE(\"Test1\")",
            fixNamespaceEndComments("TESTSUITE(\"Test1\") {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}",
                                    Style));
}

TEST_F(NamespaceEndCommentsFixerTest, AddsNewlineIfNeeded) {
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A\n"
            " int k;",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "} int k;"));
  EXPECT_EQ("namespace {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace\n"
            " int k;",
            fixNamespaceEndComments("namespace {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "} int k;"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A\n"
            " namespace B {\n"
            "int j;\n"
            "int k;\n"
            "} // namespace B",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "} namespace B {\n"
                                    "int j;\n"
                                    "int k;\n"
                                    "}"));
  EXPECT_EQ("namespace {\n"
            "int i;\n"
            "int j;\n"
            "}; // namespace\n"
            "int k;",
            fixNamespaceEndComments("namespace {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "};int k;"));
  EXPECT_EQ("namespace {\n"
            "int i;\n"
            "int j;\n"
            "}; // namespace\n"
            ";",
            fixNamespaceEndComments("namespace {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "};;"));
}

TEST_F(NamespaceEndCommentsFixerTest, DoesNotAddEndCommentForShortNamespace) {
  EXPECT_EQ("namespace {}", fixNamespaceEndComments("namespace {}"));
  EXPECT_EQ("namespace A {}", fixNamespaceEndComments("namespace A {}"));
  EXPECT_EQ("namespace A { a }", fixNamespaceEndComments("namespace A { a }"));
  EXPECT_EQ("namespace A { a };",
            fixNamespaceEndComments("namespace A { a };"));
}

TEST_F(NamespaceEndCommentsFixerTest, DoesNotAddCommentAfterUnaffectedRBrace) {
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "}",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "}",
                                    // The range (16, 3) spans the 'int' above.
                                    /*Ranges=*/{1, tooling::Range(16, 3)}));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "};",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "};",
                                    // The range (16, 3) spans the 'int' above.
                                    /*Ranges=*/{1, tooling::Range(16, 3)}));
}

TEST_F(NamespaceEndCommentsFixerTest,
       DoesNotAddCommentAfterRBraceInPPDirective) {
  EXPECT_EQ("#define SAD \\\n"
            "namespace A { \\\n"
            "int i; \\\n"
            "}",
            fixNamespaceEndComments("#define SAD \\\n"
                                    "namespace A { \\\n"
                                    "int i; \\\n"
                                    "}"));
}

TEST_F(NamespaceEndCommentsFixerTest, KeepsValidEndComment) {
  EXPECT_TRUE(isFormatted("namespace {\n"
                          "int i;\n"
                          "} // end anonymous namespace"));
  EXPECT_TRUE(isFormatted("namespace A {\n"
                          "int i;\n"
                          "} /* end of namespace A */"));
  EXPECT_TRUE(isFormatted("namespace A {\n"
                          "int i;\n"
                          "}   //   namespace A"));
  EXPECT_TRUE(isFormatted("namespace A::B {\n"
                          "int i;\n"
                          "} // end namespace A::B"));
  EXPECT_TRUE(isFormatted("namespace A {\n"
                          "int i;\n"
                          "}; // end namespace A"));
  EXPECT_TRUE(isFormatted("namespace {\n"
                          "int i;\n"
                          "}; /* unnamed namespace */"));
  EXPECT_TRUE(isFormatted("namespace a::inline b {\n"
                          "int c;\n"
                          "}; // namespace a::inline b"));
}

TEST_F(NamespaceEndCommentsFixerTest, KeepsValidMacroEndComment) {
  FormatStyle Style = getLLVMStyle();
  Style.NamespaceMacros.push_back("TESTSUITE");

  EXPECT_TRUE(isFormatted("TESTSUITE() {\n"
                          "int i;\n"
                          "} // end anonymous TESTSUITE()",
                          Style));
  EXPECT_TRUE(isFormatted("TESTSUITE(A) {\n"
                          "int i;\n"
                          "} /* end of TESTSUITE(A) */",
                          Style));
  EXPECT_TRUE(isFormatted("TESTSUITE(A) {\n"
                          "int i;\n"
                          "}   //   TESTSUITE(A)",
                          Style));
  EXPECT_TRUE(isFormatted("TESTSUITE(A::B) {\n"
                          "int i;\n"
                          "} // end TESTSUITE(A::B)",
                          Style));
  EXPECT_TRUE(isFormatted("TESTSUITE(A) {\n"
                          "int i;\n"
                          "}; // end TESTSUITE(A)",
                          Style));
  EXPECT_TRUE(isFormatted("TESTSUITE() {\n"
                          "int i;\n"
                          "}; /* unnamed TESTSUITE() */",
                          Style));
  EXPECT_TRUE(isFormatted("TESTSUITE(\"foo\") {\n"
                          "int i;\n"
                          "} // TESTSUITE(\"foo\")",
                          Style));
}

TEST_F(NamespaceEndCommentsFixerTest, UpdatesInvalidEndLineComment) {
  EXPECT_EQ("namespace {\n"
            "int i;\n"
            "} // namespace",
            fixNamespaceEndComments("namespace {\n"
                                    "int i;\n"
                                    "} // namespace A"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "} // namespace"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "} //"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "}; // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "}; //"));

  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "} // banamespace A"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "}; // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "}; // banamespace A"));
  // Updates invalid line comments even for short namespaces.
  EXPECT_EQ("namespace A {} // namespace A",
            fixNamespaceEndComments("namespace A {} // namespace"));
  EXPECT_EQ("namespace A {}; // namespace A",
            fixNamespaceEndComments("namespace A {}; // namespace"));

  // Update invalid comments for compacted namespaces.
  FormatStyle CompactNamespacesStyle = getLLVMStyle();
  CompactNamespacesStyle.CompactNamespaces = true;
  EXPECT_EQ("namespace out { namespace in {\n"
            "}} // namespace out::in",
            fixNamespaceEndComments("namespace out { namespace in {\n"
                                    "}} // namespace out",
                                    CompactNamespacesStyle));
  EXPECT_EQ("namespace out { namespace in {\n"
            "}} // namespace out::in",
            fixNamespaceEndComments("namespace out { namespace in {\n"
                                    "}} // namespace in",
                                    CompactNamespacesStyle));
  EXPECT_EQ("namespace out { namespace in {\n"
            "}\n"
            "} // namespace out::in",
            fixNamespaceEndComments("namespace out { namespace in {\n"
                                    "}// banamespace in\n"
                                    "} // namespace out",
                                    CompactNamespacesStyle));
  EXPECT_EQ("namespace out { namespace {\n"
            "}} // namespace out",
            fixNamespaceEndComments("namespace out { namespace {\n"
                                    "}} // namespace out::",
                                    CompactNamespacesStyle));
}

TEST_F(NamespaceEndCommentsFixerTest, UpdatesInvalidMacroEndLineComment) {
  FormatStyle Style = getLLVMStyle();
  Style.NamespaceMacros.push_back("TESTSUITE");

  EXPECT_EQ("TESTSUITE() {\n"
            "int i;\n"
            "} // TESTSUITE()",
            fixNamespaceEndComments("TESTSUITE() {\n"
                                    "int i;\n"
                                    "} // TESTSUITE(A)",
                                    Style));
  EXPECT_EQ("TESTSUITE(A) {\n"
            "int i;\n"
            "} // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {\n"
                                    "int i;\n"
                                    "} // TESTSUITE()",
                                    Style));
  EXPECT_EQ("TESTSUITE(A) {\n"
            "int i;\n"
            "} // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {\n"
                                    "int i;\n"
                                    "} //",
                                    Style));
  EXPECT_EQ("TESTSUITE(A) {\n"
            "int i;\n"
            "}; // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {\n"
                                    "int i;\n"
                                    "}; //",
                                    Style));
  EXPECT_EQ("TESTSUITE(A) {\n"
            "int i;\n"
            "} // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {\n"
                                    "int i;\n"
                                    "} // TESTSUITE A",
                                    Style));
  EXPECT_EQ("TESTSUITE() {\n"
            "int i;\n"
            "} // TESTSUITE()",
            fixNamespaceEndComments("TESTSUITE() {\n"
                                    "int i;\n"
                                    "} // TESTSUITE",
                                    Style));
  EXPECT_EQ("TESTSUITE(A) {\n"
            "int i;\n"
            "} // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {\n"
                                    "int i;\n"
                                    "} // TOASTSUITE(A)",
                                    Style));
  EXPECT_EQ("TESTSUITE(A) {\n"
            "int i;\n"
            "}; // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {\n"
                                    "int i;\n"
                                    "}; // TOASTSUITE(A)",
                                    Style));
  // Updates invalid line comments even for short namespaces.
  EXPECT_EQ("TESTSUITE(A) {} // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {} // TESTSUITE()", Style));
  EXPECT_EQ("TESTSUITE(A) {}; // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {}; // TESTSUITE()", Style));

  // Update invalid comments for compacted namespaces.
  FormatStyle CompactNamespacesStyle = getLLVMStyle();
  CompactNamespacesStyle.CompactNamespaces = true;
  CompactNamespacesStyle.NamespaceMacros.push_back("TESTSUITE");

  EXPECT_EQ("TESTSUITE(out) { TESTSUITE(in) {\n"
            "}} // TESTSUITE(out::in)",
            fixNamespaceEndComments("TESTSUITE(out) { TESTSUITE(in) {\n"
                                    "}} // TESTSUITE(out)",
                                    CompactNamespacesStyle));
  EXPECT_EQ("TESTSUITE(out) { TESTSUITE(in) {\n"
            "}} // TESTSUITE(out::in)",
            fixNamespaceEndComments("TESTSUITE(out) { TESTSUITE(in) {\n"
                                    "}} // TESTSUITE(in)",
                                    CompactNamespacesStyle));
  EXPECT_EQ("TESTSUITE(out) { TESTSUITE(in) {\n"
            "}\n"
            "} // TESTSUITE(out::in)",
            fixNamespaceEndComments("TESTSUITE(out) { TESTSUITE(in) {\n"
                                    "}// TAOSTSUITE(in)\n"
                                    "} // TESTSUITE(out)",
                                    CompactNamespacesStyle));
}

TEST_F(NamespaceEndCommentsFixerTest, UpdatesInvalidEndBlockComment) {
  EXPECT_EQ("namespace {\n"
            "int i;\n"
            "} // namespace",
            fixNamespaceEndComments("namespace {\n"
                                    "int i;\n"
                                    "} /* namespace A */"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "}  // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "}  /* end namespace */"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "} /**/"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "} /* end unnamed namespace */"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "} /* banamespace A */"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "}; // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "}; /* banamespace A */"));
  EXPECT_EQ("namespace A {} // namespace A",
            fixNamespaceEndComments("namespace A {} /**/"));
  EXPECT_EQ("namespace A {}; // namespace A",
            fixNamespaceEndComments("namespace A {}; /**/"));
}

TEST_F(NamespaceEndCommentsFixerTest, UpdatesInvalidMacroEndBlockComment) {
  FormatStyle Style = getLLVMStyle();
  Style.NamespaceMacros.push_back("TESTSUITE");

  EXPECT_EQ("TESTSUITE() {\n"
            "int i;\n"
            "} // TESTSUITE()",
            fixNamespaceEndComments("TESTSUITE() {\n"
                                    "int i;\n"
                                    "} /* TESTSUITE(A) */",
                                    Style));
  EXPECT_EQ("TESTSUITE(A) {\n"
            "int i;\n"
            "}  // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {\n"
                                    "int i;\n"
                                    "}  /* end TESTSUITE() */",
                                    Style));
  EXPECT_EQ("TESTSUITE(A) {\n"
            "int i;\n"
            "} // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {\n"
                                    "int i;\n"
                                    "} /**/",
                                    Style));
  EXPECT_EQ("TESTSUITE(A) {\n"
            "int i;\n"
            "} // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {\n"
                                    "int i;\n"
                                    "} /* end unnamed TESTSUITE() */",
                                    Style));
  EXPECT_EQ("TESTSUITE(A) {\n"
            "int i;\n"
            "} // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {\n"
                                    "int i;\n"
                                    "} /* TOASTSUITE(A) */",
                                    Style));
  EXPECT_EQ("TESTSUITE(A) {\n"
            "int i;\n"
            "}; // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {\n"
                                    "int i;\n"
                                    "}; /* TAOSTSUITE(A) */",
                                    Style));
  EXPECT_EQ("TESTSUITE(A) {} // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {} /**/", Style));
  EXPECT_EQ("TESTSUITE(A) {}; // TESTSUITE(A)",
            fixNamespaceEndComments("TESTSUITE(A) {}; /**/", Style));
}

TEST_F(NamespaceEndCommentsFixerTest,
       DoesNotAddEndCommentForNamespacesControlledByMacros) {
  EXPECT_EQ("#ifdef 1\n"
            "namespace A {\n"
            "#elseif\n"
            "namespace B {\n"
            "#endif\n"
            "int i;\n"
            "}\n"
            "}",
            fixNamespaceEndComments("#ifdef 1\n"
                                    "namespace A {\n"
                                    "#elseif\n"
                                    "namespace B {\n"
                                    "#endif\n"
                                    "int i;\n"
                                    "}\n"
                                    "}"));
}

TEST_F(NamespaceEndCommentsFixerTest, AddsEndCommentForNamespacesAroundMacros) {
  // Conditional blocks around are fine
  EXPECT_EQ("namespace A {\n"
            "#if 1\n"
            "int i;\n"
            "#endif\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "#if 1\n"
                                    "int i;\n"
                                    "#endif\n"
                                    "}"));
  EXPECT_EQ("#if 1\n"
            "#endif\n"
            "namespace A {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A",
            fixNamespaceEndComments("#if 1\n"
                                    "#endif\n"
                                    "namespace A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A\n"
            "#if 1\n"
            "#endif",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}\n"
                                    "#if 1\n"
                                    "#endif"));
  EXPECT_EQ("#if 1\n"
            "namespace A {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A\n"
            "#endif",
            fixNamespaceEndComments("#if 1\n"
                                    "namespace A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}\n"
                                    "#endif"));

  // Macro definition has no impact
  EXPECT_EQ("namespace A {\n"
            "#define FOO\n"
            "int i;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "#define FOO\n"
                                    "int i;\n"
                                    "}"));
  EXPECT_EQ("#define FOO\n"
            "namespace A {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A",
            fixNamespaceEndComments("#define FOO\n"
                                    "namespace A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
  EXPECT_EQ("namespace A {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A\n"
            "#define FOO",
            fixNamespaceEndComments("namespace A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}\n"
                                    "#define FOO"));

  // No replacement if open & close in different conditional blocks
  EXPECT_EQ("#if 1\n"
            "namespace A {\n"
            "#endif\n"
            "int i;\n"
            "int j;\n"
            "#if 1\n"
            "}\n"
            "#endif",
            fixNamespaceEndComments("#if 1\n"
                                    "namespace A {\n"
                                    "#endif\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "#if 1\n"
                                    "}\n"
                                    "#endif"));
  EXPECT_EQ("#ifdef A\n"
            "namespace A {\n"
            "#endif\n"
            "int i;\n"
            "int j;\n"
            "#ifdef B\n"
            "}\n"
            "#endif",
            fixNamespaceEndComments("#ifdef A\n"
                                    "namespace A {\n"
                                    "#endif\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "#ifdef B\n"
                                    "}\n"
                                    "#endif"));

  // No replacement inside unreachable conditional block
  EXPECT_EQ("#if 0\n"
            "namespace A {\n"
            "int i;\n"
            "int j;\n"
            "}\n"
            "#endif",
            fixNamespaceEndComments("#if 0\n"
                                    "namespace A {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}\n"
                                    "#endif"));
}

TEST_F(NamespaceEndCommentsFixerTest,
       DoesNotAddEndCommentForNamespacesInMacroDeclarations) {
  EXPECT_EQ("#ifdef 1\n"
            "namespace A {\n"
            "#elseif\n"
            "namespace B {\n"
            "#endif\n"
            "int i;\n"
            "}\n"
            "}",
            fixNamespaceEndComments("#ifdef 1\n"
                                    "namespace A {\n"
                                    "#elseif\n"
                                    "namespace B {\n"
                                    "#endif\n"
                                    "int i;\n"
                                    "}\n"
                                    "}"));
  EXPECT_EQ("namespace {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace\n"
            "#if A\n"
            "int i;\n"
            "#else\n"
            "int j;\n"
            "#endif",
            fixNamespaceEndComments("namespace {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}\n"
                                    "#if A\n"
                                    "int i;\n"
                                    "#else\n"
                                    "int j;\n"
                                    "#endif"));
  EXPECT_EQ("#if A\n"
            "namespace A {\n"
            "#else\n"
            "namespace B {\n"
            "#endif\n"
            "int i;\n"
            "int j;\n"
            "}",
            fixNamespaceEndComments("#if A\n"
                                    "namespace A {\n"
                                    "#else\n"
                                    "namespace B {\n"
                                    "#endif\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
  EXPECT_EQ("#if A\n"
            "namespace A {\n"
            "#else\n"
            "namespace B {\n"
            "#endif\n"
            "int i;\n"
            "int j;\n"
            "} // namespace A",
            fixNamespaceEndComments("#if A\n"
                                    "namespace A {\n"
                                    "#else\n"
                                    "namespace B {\n"
                                    "#endif\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "} // namespace A"));
  EXPECT_EQ("#if A\n"
            "namespace A {\n"
            "#else\n"
            "namespace B {\n"
            "#endif\n"
            "int i;\n"
            "int j;\n"
            "} // namespace B",
            fixNamespaceEndComments("#if A\n"
                                    "namespace A {\n"
                                    "#else\n"
                                    "namespace B {\n"
                                    "#endif\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "} // namespace B"));
  EXPECT_EQ("namespace A\n"
            "int i;\n"
            "int j;\n"
            "#if A\n"
            "}\n"
            "#else\n"
            "}\n"
            "#endif",
            fixNamespaceEndComments("namespace A\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "#if A\n"
                                    "}\n"
                                    "#else\n"
                                    "}\n"
                                    "#endif"));
  EXPECT_EQ("namespace A\n"
            "int i;\n"
            "int j;\n"
            "#if A\n"
            "} // namespace A\n"
            "#else\n"
            "} // namespace A\n"
            "#endif",
            fixNamespaceEndComments("namespace A\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "#if A\n"
                                    "} // namespace A\n"
                                    "#else\n"
                                    "} // namespace A\n"
                                    "#endif"));
}

TEST_F(NamespaceEndCommentsFixerTest,
       DoesNotAddEndCommentForUnbalancedRBracesAfterNamespaceEnd) {
  EXPECT_EQ("namespace {\n"
            "int i;\n"
            "} // namespace\n"
            "}",
            fixNamespaceEndComments("namespace {\n"
                                    "int i;\n"
                                    "} // namespace\n"
                                    "}"));
}

TEST_F(NamespaceEndCommentsFixerTest, HandlesInlineAtEndOfLine_PR32438) {
  EXPECT_EQ("template <int> struct a {};\n"
            "struct a<bool{}> b() {\n"
            "}\n"
            "#define c inline\n"
            "void d() {\n"
            "}",
            fixNamespaceEndComments("template <int> struct a {};\n"
                                    "struct a<bool{}> b() {\n"
                                    "}\n"
                                    "#define c inline\n"
                                    "void d() {\n"
                                    "}"));
}

TEST_F(NamespaceEndCommentsFixerTest, IgnoreUnbalanced) {
  EXPECT_EQ("namespace A {\n"
            "class Foo {\n"
            "}\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "class Foo {\n"
                                    "}\n"
                                    "}"));
  EXPECT_EQ("namespace A {\n"
            "class Foo {\n"
            "}",
            fixNamespaceEndComments("namespace A {\n"
                                    "class Foo {\n"
                                    "}"));

  EXPECT_EQ("namespace A {\n"
            "class Foo {\n"
            "}\n"
            "}\n"
            "}",
            fixNamespaceEndComments("namespace A {\n"
                                    "class Foo {\n"
                                    "}\n"
                                    "}\n"
                                    "}"));
}

using ShortNamespaceLinesTest = NamespaceEndCommentsFixerTest;

TEST_F(ShortNamespaceLinesTest, ZeroUnwrappedLines) {
  auto Style = getLLVMStyle();
  Style.ShortNamespaceLines = 0u;

  EXPECT_EQ("namespace OneLinerNamespace {}",
            fixNamespaceEndComments("namespace OneLinerNamespace {}", Style));
  EXPECT_EQ("namespace ShortNamespace {\n"
            "}",
            fixNamespaceEndComments("namespace ShortNamespace {\n"
                                    "}",
                                    Style));
  EXPECT_EQ("namespace LongNamespace {\n"
            "int i;\n"
            "} // namespace LongNamespace",
            fixNamespaceEndComments("namespace LongNamespace {\n"
                                    "int i;\n"
                                    "}",
                                    Style));
}

TEST_F(ShortNamespaceLinesTest, OneUnwrappedLine) {
  constexpr auto DefaultUnwrappedLines = 1u;
  auto const Style = getLLVMStyle();

  EXPECT_EQ(DefaultUnwrappedLines, Style.ShortNamespaceLines);
  EXPECT_EQ("namespace ShortNamespace {\n"
            "int i;\n"
            "}",
            fixNamespaceEndComments("namespace ShortNamespace {\n"
                                    "int i;\n"
                                    "}"));
  EXPECT_EQ("namespace LongNamespace {\n"
            "int i;\n"
            "int j;\n"
            "} // namespace LongNamespace",
            fixNamespaceEndComments("namespace LongNamespace {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}"));
}

TEST_F(ShortNamespaceLinesTest, MultipleUnwrappedLine) {
  auto Style = getLLVMStyle();
  Style.ShortNamespaceLines = 2u;

  EXPECT_EQ("namespace ShortNamespace {\n"
            "int i;\n"
            "int j;\n"
            "}",
            fixNamespaceEndComments("namespace ShortNamespace {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "}",
                                    Style));
  EXPECT_EQ("namespace LongNamespace {\n"
            "int i;\n"
            "int j;\n"
            "int k;\n"
            "} // namespace LongNamespace",
            fixNamespaceEndComments("namespace LongNamespace {\n"
                                    "int i;\n"
                                    "int j;\n"
                                    "int k;\n"
                                    "}",
                                    Style));

  // The namespace body has 5 unwrapped/annotated lines.
  const std::string NestedLambdas{"namespace foo {\n"
                                  "auto bar = [] {\n" // line 1
                                  "  int i;\n"        // line 2
                                  "  return [] {\n"   // line 3
                                  "      int j;"      // line 4
                                  "      return 0;\n" // line 5
                                  "  };\n"            // part of line 3
                                  "};\n"              // part of line 1
                                  "}"};
  Style.ShortNamespaceLines = 4;
  EXPECT_EQ(NestedLambdas + " // namespace foo",
            fixNamespaceEndComments(NestedLambdas, Style));
  ++Style.ShortNamespaceLines;
  EXPECT_EQ(NestedLambdas, fixNamespaceEndComments(NestedLambdas, Style));
}

TEST_F(ShortNamespaceLinesTest, NamespaceAlias) {
  auto Style = getLLVMStyle();

  EXPECT_EQ("namespace n = nn;\n"
            "{\n"
            "  int i;\n"
            "  int j;\n"
            "}",
            fixNamespaceEndComments("namespace n = nn;\n"
                                    "{\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "}",
                                    Style));

  EXPECT_EQ("namespace n = nn; // comment\n"
            "{\n"
            "  int i;\n"
            "  int j;\n"
            "}",
            fixNamespaceEndComments("namespace n = nn; // comment\n"
                                    "{\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "}",
                                    Style));

  EXPECT_EQ("namespace n = nn; /* comment */\n"
            "{\n"
            "  int i;\n"
            "  int j;\n"
            "}",
            fixNamespaceEndComments("namespace n = nn; /* comment */\n"
                                    "{\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "}",
                                    Style));

  EXPECT_EQ(
      "namespace n = nn; /* comment */ /* comment2 */\n"
      "{\n"
      "  int i;\n"
      "  int j;\n"
      "}",
      fixNamespaceEndComments("namespace n = nn; /* comment */ /* comment2 */\n"
                              "{\n"
                              "  int i;\n"
                              "  int j;\n"
                              "}",
                              Style));

  EXPECT_EQ("namespace n = nn; {\n"
            "  int i;\n"
            "  int j;\n"
            "}",
            fixNamespaceEndComments("namespace n = nn; {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "}",
                                    Style));
  EXPECT_EQ("int foo;\n"
            "namespace n\n"
            "{\n"
            "  int i;\n"
            "  int j;\n"
            "} // namespace n",
            fixNamespaceEndComments("int foo;\n"
                                    "namespace n\n"
                                    "{\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "}",
                                    Style));
}
} // end namespace
} // end namespace format
} // end namespace clang
