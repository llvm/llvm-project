//===-- SelectionTests.cpp - ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "Selection.h"
#include "SourceCode.h"
#include "TestTU.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {
using ::testing::UnorderedElementsAreArray;

SelectionTree makeSelectionTree(const StringRef MarkedCode, ParsedAST &AST) {
  Annotations Test(MarkedCode);
  switch (Test.points().size()) {
  case 1: // Point selection.
    return SelectionTree(AST.getASTContext(),
                         cantFail(positionToOffset(Test.code(), Test.point())));
  case 2: // Range selection.
    return SelectionTree(
        AST.getASTContext(),
        cantFail(positionToOffset(Test.code(), Test.points()[0])),
        cantFail(positionToOffset(Test.code(), Test.points()[1])));
  default:
    ADD_FAILURE() << "Expected 1-2 points for selection.\n" << MarkedCode;
    return SelectionTree(AST.getASTContext(), 0u, 0u);
  }
}

Range nodeRange(const SelectionTree::Node *N, ParsedAST &AST) {
  if (!N)
    return Range{};
  SourceManager &SM = AST.getASTContext().getSourceManager();
  StringRef Buffer = SM.getBufferData(SM.getMainFileID());
  SourceRange SR = N->ASTNode.getSourceRange();
  SR.setBegin(SM.getFileLoc(SR.getBegin()));
  SR.setEnd(SM.getFileLoc(SR.getEnd()));
  CharSourceRange R =
      Lexer::getAsCharRange(SR, SM, AST.getASTContext().getLangOpts());
  return Range{offsetToPosition(Buffer, SM.getFileOffset(R.getBegin())),
               offsetToPosition(Buffer, SM.getFileOffset(R.getEnd()))};
}

std::string nodeKind(const SelectionTree::Node *N) {
  if (!N)
    return "<null>";
  return N->ASTNode.getNodeKind().asStringRef().str();
}

std::vector<const SelectionTree::Node *> allNodes(const SelectionTree &T) {
  std::vector<const SelectionTree::Node *> Result = {T.root()};
  for (unsigned I = 0; I < Result.size(); ++I) {
    const SelectionTree::Node *N = Result[I];
    Result.insert(Result.end(), N->Children.begin(), N->Children.end());
  }
  return Result;
}

// Returns true if Common is a descendent of Root.
// Verifies nothing is selected above Common.
bool verifyCommonAncestor(const SelectionTree::Node *Root,
                          const SelectionTree::Node *Common,
                          StringRef MarkedCode) {
  if (Root == Common)
    return true;
  if (Root->Selected)
    ADD_FAILURE() << "Selected nodes outside common ancestor\n" << MarkedCode;
  bool Seen = false;
  for (const SelectionTree::Node *Child : Root->Children)
    if (verifyCommonAncestor(Child, Common, MarkedCode)) {
      if (Seen)
        ADD_FAILURE() << "Saw common ancestor twice\n" << MarkedCode;
      Seen = true;
    }
  return Seen;
}

TEST(SelectionTest, CommonAncestor) {
  struct Case {
    // Selection is between ^marks^.
    // common ancestor marked with a [[range]].
    const char *Code;
    const char *CommonAncestorKind;
  };
  Case Cases[] = {
      {
          R"cpp(
            struct AAA { struct BBB { static int ccc(); };};
            int x = AAA::[[B^B^B]]::ccc();
          )cpp",
          "TypeLoc",
      },
      {
          R"cpp(
            struct AAA { struct BBB { static int ccc(); };};
            int x = AAA::[[B^BB^]]::ccc();
          )cpp",
          "TypeLoc",
      },
      {
          R"cpp(
            struct AAA { struct BBB { static int ccc(); };};
            int x = [[AAA::BBB::c^c^c]]();
          )cpp",
          "DeclRefExpr",
      },
      {
          R"cpp(
            struct AAA { struct BBB { static int ccc(); };};
            int x = [[AAA::BBB::cc^c(^)]];
          )cpp",
          "CallExpr",
      },

      {
          R"cpp(
            void foo() { [[if (1^11) { return; } else {^ }]] }
          )cpp",
          "IfStmt",
      },
      {
          R"cpp(
            void foo();
            #define CALL_FUNCTION(X) X()
            void bar() { CALL_FUNCTION([[f^o^o]]); }
          )cpp",
          "DeclRefExpr",
      },
      {
          R"cpp(
            void foo();
            #define CALL_FUNCTION(X) X()
            void bar() { CALL_FUNC^TION([[fo^o]]); }
          )cpp",
          "DeclRefExpr",
      },
      {
          R"cpp(
            void foo();
            #define CALL_FUNCTION(X) X()
            void bar() [[{ C^ALL_FUNC^TION(foo); }]]
          )cpp",
          "CompoundStmt",
      },
      {
          R"cpp(
            void foo();
            #define CALL_FUNCTION(X) X^()^
            void bar() { CALL_FUNCTION(foo); }
          )cpp",
          nullptr,
      },

      // Point selections.
      {"void foo() { [[^foo]](); }", "DeclRefExpr"},
      {"void foo() { [[f^oo]](); }", "DeclRefExpr"},
      {"void foo() { [[fo^o]](); }", "DeclRefExpr"},
      {"void foo() { [[foo^()]]; }", "CallExpr"},
      {"void foo() { [[foo^]] (); }", "DeclRefExpr"},
      {"int bar; void foo() [[{ foo (); }]]^", "CompoundStmt"},
      {"[[^void]] foo();", "TypeLoc"},
      {"^", nullptr},
      {"void foo() { [[foo^^]] (); }", "DeclRefExpr"},

      // FIXME: Ideally we'd get a declstmt or the VarDecl itself here.
      // This doesn't happen now; the RAV doesn't traverse a node containing ;.
      {"int x = 42;^", nullptr},
      {"int x = 42^;", nullptr},

      // Node types that have caused problems in the past.
      {"template <typename T> void foo() { [[^T]] t; }", "TypeLoc"},

      // No crash
      {
          R"cpp(
            template <class T> struct Foo {};
            template <[[template<class> class /*cursor here*/^U]]>
             struct Foo<U<int>*> {};
          )cpp",
          "TemplateTemplateParmDecl"
      },
  };
  for (const Case &C : Cases) {
    Annotations Test(C.Code);
    auto AST = TestTU::withCode(Test.code()).build();
    auto T = makeSelectionTree(C.Code, AST);

    if (Test.ranges().empty()) {
      // If no [[range]] is marked in the example, there should be no selection.
      EXPECT_FALSE(T.commonAncestor()) << C.Code << "\n" << T;
      EXPECT_FALSE(T.root()) << C.Code << "\n" << T;
    } else {
      // If there is an expected selection, both common ancestor and root
      // should exist with the appropriate node types in them.
      EXPECT_EQ(C.CommonAncestorKind, nodeKind(T.commonAncestor()))
          << C.Code << "\n"
          << T;
      EXPECT_EQ("TranslationUnitDecl", nodeKind(T.root())) << C.Code;
      // Convert the reported common ancestor to a range and verify it.
      EXPECT_EQ(nodeRange(T.commonAncestor(), AST), Test.range())
          << C.Code << "\n"
          << T;

      // Check that common ancestor is reachable on exactly one path from root,
      // and no nodes outside it are selected.
      EXPECT_TRUE(verifyCommonAncestor(T.root(), T.commonAncestor(), C.Code))
          << C.Code;
    }
  }
}

TEST(SelectionTest, Selected) {
  // Selection with ^marks^.
  // Partially selected nodes marked with a [[range]].
  // Completely selected nodes marked with a $C[[range]].
  const char *Cases[] = {
      R"cpp( int abc, xyz = [[^ab^c]]; )cpp",
      R"cpp( int abc, xyz = [[a^bc^]]; )cpp",
      R"cpp( int abc, xyz = $C[[^abc^]]; )cpp",
      R"cpp(
        void foo() {
          [[if ([[1^11]]) $C[[{
            $C[[return]];
          }]] else [[{^
          }]]]]
        }
      )cpp",
      R"cpp(
          template <class T>
          struct unique_ptr {};
          void foo(^$C[[unique_ptr<unique_ptr<$C[[int]]>>]]^ a) {}
      )cpp",
  };
  for (const char *C : Cases) {
    Annotations Test(C);
    auto AST = TestTU::withCode(Test.code()).build();
    auto T = makeSelectionTree(C, AST);

    std::vector<Range> Complete, Partial;
    for (const SelectionTree::Node *N : allNodes(T))
      if (N->Selected == SelectionTree::Complete)
        Complete.push_back(nodeRange(N, AST));
      else if (N->Selected == SelectionTree::Partial)
        Partial.push_back(nodeRange(N, AST));
    EXPECT_THAT(Complete, UnorderedElementsAreArray(Test.ranges("C"))) << C;
    EXPECT_THAT(Partial, UnorderedElementsAreArray(Test.ranges())) << C;
  }
}

} // namespace
} // namespace clangd
} // namespace clang
