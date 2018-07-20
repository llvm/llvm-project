//===- unittests/Tooling/RecursiveASTVisitorPostOrderASTVisitor.cpp -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for the post-order traversing functionality
// of RecursiveASTVisitor.
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

class RecordingVisitor : public TestVisitor<RecordingVisitor> {

  bool VisitPostOrder;

public:
  explicit RecordingVisitor(bool VisitPostOrder)
      : VisitPostOrder(VisitPostOrder) {}

  // List of visited nodes during traversal.
  std::vector<std::string> VisitedNodes;

  bool shouldTraversePostOrder() const { return VisitPostOrder; }

  bool VisitUnaryOperator(UnaryOperator *Op) {
    VisitedNodes.push_back(Op->getOpcodeStr(Op->getOpcode()));
    return true;
  }

  bool VisitBinaryOperator(BinaryOperator *Op) {
    VisitedNodes.push_back(Op->getOpcodeStr());
    return true;
  }

  bool VisitIntegerLiteral(IntegerLiteral *Lit) {
    VisitedNodes.push_back(Lit->getValue().toString(10, false));
    return true;
  }

  bool VisitVarDecl(VarDecl *D) {
    VisitedNodes.push_back(D->getNameAsString());
    return true;
  }

  bool VisitCXXMethodDecl(CXXMethodDecl *D) {
    VisitedNodes.push_back(D->getQualifiedNameAsString());
    return true;
  }

  bool VisitReturnStmt(ReturnStmt *S) {
    VisitedNodes.push_back("return");
    return true;
  }

  bool VisitCXXRecordDecl(CXXRecordDecl *D) {
    if (!D->isImplicit())
      VisitedNodes.push_back(D->getQualifiedNameAsString());
    return true;
  }

  bool VisitTemplateTypeParmType(TemplateTypeParmType *T) {
    VisitedNodes.push_back(T->getDecl()->getQualifiedNameAsString());
    return true;
  }
};
} // namespace

TEST(RecursiveASTVisitor, PostOrderTraversal) {
  // We traverse the translation unit and store all visited nodes.
  RecordingVisitor Visitor(true);
  Visitor.runOver("class A {\n"
                  "  class B {\n"
                  "    int foo() {\n"
                  "      while(4) { int i = 9; int j = -5; }\n"
                  "      return (1 + 3) + 2; }\n"
                  "  };\n"
                  "};\n");

  std::vector<std::string> expected = {"4", "9",      "i",         "5",    "-",
                                       "j", "1",      "3",         "+",    "2",
                                       "+", "return", "A::B::foo", "A::B", "A"};
  // Compare the list of actually visited nodes with the expected list of
  // visited nodes.
  ASSERT_EQ(expected.size(), Visitor.VisitedNodes.size());
  for (std::size_t I = 0; I < expected.size(); I++) {
    ASSERT_EQ(expected[I], Visitor.VisitedNodes[I]);
  }
}

TEST(RecursiveASTVisitor, NoPostOrderTraversal) {
  // We traverse the translation unit and store all visited nodes.
  RecordingVisitor Visitor(false);
  Visitor.runOver("class A {\n"
                  "  class B {\n"
                  "    int foo() { return 1 + 2; }\n"
                  "  };\n"
                  "};\n");

  std::vector<std::string> expected = {"A", "A::B", "A::B::foo", "return",
                                       "+", "1",    "2"};
  // Compare the list of actually visited nodes with the expected list of
  // visited nodes.
  ASSERT_EQ(expected.size(), Visitor.VisitedNodes.size());
  for (std::size_t I = 0; I < expected.size(); I++) {
    ASSERT_EQ(expected[I], Visitor.VisitedNodes[I]);
  }
}
