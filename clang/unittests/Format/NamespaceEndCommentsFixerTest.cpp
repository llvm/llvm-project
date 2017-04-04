//===- NamespaceEndCommentsFixerTest.cpp - Formatting unit tests ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"

#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "namespace-end-comments-fixer-test"

namespace clang {
namespace format {
namespace {

class NamespaceEndCommentsFixerTest : public ::testing::Test {
protected:
  std::string
  fixNamespaceEndComments(llvm::StringRef Code,
                          std::vector<tooling::Range> Ranges,
                          const FormatStyle &Style = getLLVMStyle()) {
    DEBUG(llvm::errs() << "---\n");
    DEBUG(llvm::errs() << Code << "\n\n");
    tooling::Replacements Replaces =
        clang::format::fixNamespaceEndComments(Style, Code, Ranges, "<stdin>");
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  std::string
  fixNamespaceEndComments(llvm::StringRef Code,
                          const FormatStyle &Style = getLLVMStyle()) {
    return fixNamespaceEndComments(
        Code,
        /*Ranges=*/{1, tooling::Range(0, Code.size())}, Style);
  }
};

TEST_F(NamespaceEndCommentsFixerTest, AddsEndComment) {
  EXPECT_EQ("namespace {\n"
            "  int i;\n"
            "  int j;\n"
            "}// namespace",
            fixNamespaceEndComments("namespace {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "}"));
  EXPECT_EQ("namespace {\n"
            "  int i;\n"
            "  int j;\n"
            "}// namespace\n",
            fixNamespaceEndComments("namespace {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "}\n"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "  int j;\n"
            "}// namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "}"));
  EXPECT_EQ("inline namespace A {\n"
            "  int i;\n"
            "  int j;\n"
            "}// namespace A",
            fixNamespaceEndComments("inline namespace A {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "}"));
  EXPECT_EQ("namespace ::A {\n"
            "  int i;\n"
            "  int j;\n"
            "}// namespace ::A",
            fixNamespaceEndComments("namespace ::A {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "}"));
  EXPECT_EQ("namespace ::A::B {\n"
            "  int i;\n"
            "  int j;\n"
            "}// namespace ::A::B",
            fixNamespaceEndComments("namespace ::A::B {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "}"));
  EXPECT_EQ("namespace /**/::/**/A/**/::/**/B/**/ {\n"
            "  int i;\n"
            "  int j;\n"
            "}// namespace ::A::B",
            fixNamespaceEndComments("namespace /**/::/**/A/**/::/**/B/**/ {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "}"));
  EXPECT_EQ("namespace A {\n"
            "namespace B {\n"
            "  int i;\n"
            "}\n"
            "}// namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "namespace B {\n"
                                    "  int i;\n"
                                    "}\n"
                                    "}"));
  EXPECT_EQ("namespace A {\n"
            "namespace B {\n"
            "  int i;\n"
            "  int j;\n"
            "}// namespace B\n"
            "}// namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "namespace B {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "}\n"
                                    "}"));
  EXPECT_EQ("namespace A {\n"
            "  int a;\n"
            "  int b;\n"
            "}// namespace A\n"
            "namespace B {\n"
            "  int b;\n"
            "  int a;\n"
            "}// namespace B",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int a;\n"
                                    "  int b;\n"
                                    "}\n"
                                    "namespace B {\n"
                                    "  int b;\n"
                                    "  int a;\n"
                                    "}"));
  EXPECT_EQ("namespace A {\n"
            "  int a1;\n"
            "  int a2;\n"
            "}// namespace A\n"
            "namespace A {\n"
            "  int a2;\n"
            "  int a1;\n"
            "}// namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int a1;\n"
                                    "  int a2;\n"
                                    "}\n"
                                    "namespace A {\n"
                                    "  int a2;\n"
                                    "  int a1;\n"
                                    "}"));
  EXPECT_EQ("namespace A {\n"
            "  int a;\n"
            "  int b;\n"
            "}// namespace A\n"
            "// comment about b\n"
            "int b;",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int a;\n"
                                    "  int b;\n"
                                    "}\n"
                                    "// comment about b\n"
                                    "int b;"));

  EXPECT_EQ("namespace A {\n"
            "namespace B {\n"
            "namespace C {\n"
            "namespace D {\n"
            "}\n"
            "}// namespace C\n"
            "}// namespace B\n"
            "}// namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "namespace B {\n"
                                    "namespace C {\n"
                                    "namespace D {\n"
                                    "}\n"
                                    "}\n"
                                    "}\n"
                                    "}"));

  // Adds an end comment after a semicolon.
  EXPECT_EQ("namespace {\n"
            "  int i;\n"
            "  int j;\n"
            "};// namespace",
            fixNamespaceEndComments("namespace {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "};"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "  int j;\n"
            "};// namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "};"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "  int j;\n"
            "};// namespace A\n"
            "// unrelated",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "};\n"
                                    "// unrelated"));
}

TEST_F(NamespaceEndCommentsFixerTest, AddsNewlineIfNeeded) {
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "  int j;\n"
            "}// namespace A\n"
            " int k;",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "} int k;"));
  EXPECT_EQ("namespace {\n"
            "  int i;\n"
            "  int j;\n"
            "}// namespace\n"
            " int k;",
            fixNamespaceEndComments("namespace {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "} int k;"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "  int j;\n"
            "}// namespace A\n"
            " namespace B {\n"
            "  int j;\n"
            "  int k;\n"
            "}// namespace B",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "} namespace B {\n"
                                    "  int j;\n"
                                    "  int k;\n"
                                    "}"));
  EXPECT_EQ("namespace {\n"
            "  int i;\n"
            "  int j;\n"
            "};// namespace\n"
            "int k;",
            fixNamespaceEndComments("namespace {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "};int k;"));
  EXPECT_EQ("namespace {\n"
            "  int i;\n"
            "  int j;\n"
            "};// namespace\n"
            ";",
            fixNamespaceEndComments("namespace {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "};;"));
}

TEST_F(NamespaceEndCommentsFixerTest, DoesNotAddEndCommentForShortNamespace) {
  EXPECT_EQ("namespace {}", fixNamespaceEndComments("namespace {}"));
  EXPECT_EQ("namespace A {}", fixNamespaceEndComments("namespace A {}"));
  EXPECT_EQ("namespace A { a }",
            fixNamespaceEndComments("namespace A { a }"));
  EXPECT_EQ("namespace A { a };",
            fixNamespaceEndComments("namespace A { a };"));
}

TEST_F(NamespaceEndCommentsFixerTest, DoesNotAddCommentAfterUnaffectedRBrace) {
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "}",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "}",
                                    // The range (16, 3) spans the 'int' above.
                                    /*Ranges=*/{1, tooling::Range(16, 3)}));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "};",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "};",
                                    // The range (16, 3) spans the 'int' above.
                                    /*Ranges=*/{1, tooling::Range(16, 3)}));
}

TEST_F(NamespaceEndCommentsFixerTest, DoesNotAddCommentAfterRBraceInPPDirective) {
  EXPECT_EQ("#define SAD \\\n"
            "namespace A { \\\n"
            "  int i; \\\n"
            "}",
            fixNamespaceEndComments("#define SAD \\\n"
                                    "namespace A { \\\n"
                                    "  int i; \\\n"
                                    "}"));
}

TEST_F(NamespaceEndCommentsFixerTest, KeepsValidEndComment) {
  EXPECT_EQ("namespace {\n"
            "  int i;\n"
            "} // end anonymous namespace",
            fixNamespaceEndComments("namespace {\n"
                                    "  int i;\n"
                                    "} // end anonymous namespace"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "} /* end of namespace A */",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "} /* end of namespace A */"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "}   //   namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "}   //   namespace A"));
  EXPECT_EQ("namespace A::B {\n"
            "  int i;\n"
            "} // end namespace A::B",
            fixNamespaceEndComments("namespace A::B {\n"
                                    "  int i;\n"
                                    "} // end namespace A::B"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "}; // end namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "}; // end namespace A"));
  EXPECT_EQ("namespace {\n"
            "  int i;\n"
            "}; /* unnamed namespace */",
            fixNamespaceEndComments("namespace {\n"
                                    "  int i;\n"
                                    "}; /* unnamed namespace */"));
}

TEST_F(NamespaceEndCommentsFixerTest, UpdatesInvalidEndLineComment) {
  EXPECT_EQ("namespace {\n"
            "  int i;\n"
            "} // namespace",
            fixNamespaceEndComments("namespace {\n"
                                    "  int i;\n"
                                    "} // namespace A"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "} // namespace"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "} //"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "} //"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "} // banamespace A"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "}; // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "}; // banamespace A"));
  // Updates invalid line comments even for short namespaces.
  EXPECT_EQ("namespace A {} // namespace A",
            fixNamespaceEndComments("namespace A {} // namespace"));
  EXPECT_EQ("namespace A {}; // namespace A",
            fixNamespaceEndComments("namespace A {}; // namespace"));
}

TEST_F(NamespaceEndCommentsFixerTest, UpdatesInvalidEndBlockComment) {
  EXPECT_EQ("namespace {\n"
            "  int i;\n"
            "} // namespace",
            fixNamespaceEndComments("namespace {\n"
                                    "  int i;\n"
                                    "} /* namespace A */"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "}  // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "}  /* end namespace */"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "} /**/"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "} /* end unnamed namespace */"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "} // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "} /* banamespace A */"));
  EXPECT_EQ("namespace A {\n"
            "  int i;\n"
            "}; // namespace A",
            fixNamespaceEndComments("namespace A {\n"
                                    "  int i;\n"
                                    "}; /* banamespace A */"));
  EXPECT_EQ("namespace A {} // namespace A",
            fixNamespaceEndComments("namespace A {} /**/"));
  EXPECT_EQ("namespace A {}; // namespace A",
            fixNamespaceEndComments("namespace A {}; /**/"));
}

TEST_F(NamespaceEndCommentsFixerTest,
       DoesNotAddEndCommentForNamespacesControlledByMacros) {
  EXPECT_EQ("#ifdef 1\n"
            "namespace A {\n"
            "#elseif\n"
            "namespace B {\n"
            "#endif\n"
            "  int i;\n"
            "}\n"
            "}\n",
            fixNamespaceEndComments("#ifdef 1\n"
                                    "namespace A {\n"
                                    "#elseif\n"
                                    "namespace B {\n"
                                    "#endif\n"
                                    "  int i;\n"
                                    "}\n"
                                    "}\n"));
}

TEST_F(NamespaceEndCommentsFixerTest,
       DoesNotAddEndCommentForNamespacesInMacroDeclarations) {
  EXPECT_EQ("#ifdef 1\n"
            "namespace A {\n"
            "#elseif\n"
            "namespace B {\n"
            "#endif\n"
            "  int i;\n"
            "}\n"
            "}\n",
            fixNamespaceEndComments("#ifdef 1\n"
                                    "namespace A {\n"
                                    "#elseif\n"
                                    "namespace B {\n"
                                    "#endif\n"
                                    "  int i;\n"
                                    "}\n"
                                    "}\n"));
  EXPECT_EQ("namespace {\n"
            "  int i;\n"
            "  int j;\n"
            "}// namespace\n"
            "#if A\n"
            "  int i;\n"
            "#else\n"
            "  int j;\n"
            "#endif",
            fixNamespaceEndComments("namespace {\n"
                                    "  int i;\n"
                                    "  int j;\n"
                                    "}\n"
                                    "#if A\n"
                                    "  int i;\n"
                                    "#else\n"
                                    "  int j;\n"
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
            "  int i;\n"
            "} // namespace\n"
            "}",
            fixNamespaceEndComments("namespace {\n"
                                    "  int i;\n"
                                    "} // namespace\n"
                                    "}"));
}

TEST_F(NamespaceEndCommentsFixerTest, HandlesInlineAtEndOfLine_PR32438) {
  EXPECT_EQ("template <int> struct a {};\n"
            "struct a<bool{}> b() {\n"
            "}\n"
            "#define c inline\n"
            "void d() {\n"
            "}\n",
            fixNamespaceEndComments("template <int> struct a {};\n"
                                    "struct a<bool{}> b() {\n"
                                    "}\n"
                                    "#define c inline\n"
                                    "void d() {\n"
                                    "}\n"));
}
} // end namespace
} // end namespace format
} // end namespace clang
