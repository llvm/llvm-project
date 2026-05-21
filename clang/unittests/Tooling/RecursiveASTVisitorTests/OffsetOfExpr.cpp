//===- unittest/Tooling/RecursiveASTVisitorTests/OffsetOfExpr.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include "clang/AST/Expr.h"

using namespace clang;

namespace {

struct OffsetOfNodeRecorder : TestVisitor {
  // Expose the protected helper so a test can supply extra compiler args.
  using TestVisitor::CreateTestAction;

  struct Component {
    OffsetOfNode::Kind Kind;
    // Field name for Field/Identifier kinds; empty otherwise.
    std::string Name;
    // For Array kind: the slot in OffsetOfExpr's trailing index-expression
    // array, NOT the literal subscript value. Slots are assigned in source
    // order (so e.g. `a[2].b[7].c` produces Array nodes referring to slots
    // 0 and 1, whose Expr*s evaluate to 2 and 7 respectively).
    unsigned ArrayExprSlot = ~0u;
    // Whether the pointer received in TraverseOffsetOfNode equals the one
    // that the enclosing OffsetOfExpr exposes via getComponent().
    bool PointerStable = false;
  };

  bool TraverseOffsetOfNode(const OffsetOfNode *Node) override {
    ++Traversed;
    return TestVisitor::TraverseOffsetOfNode(Node);
  }

  bool VisitOffsetOfNode(const OffsetOfNode *Node) override {
    Component C;
    C.Kind = Node->getKind();
    if (C.Kind == OffsetOfNode::Field)
      C.Name = Node->getField()->getNameAsString();
    else if (C.Kind == OffsetOfNode::Identifier)
      C.Name = Node->getFieldName()->getName().str();
    else if (C.Kind == OffsetOfNode::Array)
      C.ArrayExprSlot = Node->getArrayExprIndex();
    C.PointerStable = StableAddresses.count(Node) != 0;
    Visited.push_back(std::move(C));
    return true;
  }

  bool VisitOffsetOfExpr(OffsetOfExpr *E) override {
    for (unsigned I = 0, N = E->getNumComponents(); I != N; ++I)
      StableAddresses.insert(&E->getComponent(I));
    return true;
  }

  llvm::DenseSet<const OffsetOfNode *> StableAddresses;
  std::vector<Component> Visited;
  int Traversed = 0;
};

TEST(RecursiveASTVisitor, OffsetOfFlatField) {
  OffsetOfNodeRecorder Recorder;
  EXPECT_TRUE(Recorder.runOver(
      "struct Foo { int bar; };\n"
      "unsigned long x = __builtin_offsetof(struct Foo, bar);\n",
      OffsetOfNodeRecorder::Lang_C));
  ASSERT_EQ(1u, Recorder.Visited.size());
  EXPECT_EQ(1, Recorder.Traversed);
  EXPECT_EQ(OffsetOfNode::Field, Recorder.Visited[0].Kind);
  EXPECT_EQ("bar", Recorder.Visited[0].Name);
  EXPECT_TRUE(Recorder.Visited[0].PointerStable);
}

TEST(RecursiveASTVisitor, OffsetOfNestedFields) {
  OffsetOfNodeRecorder Recorder;
  EXPECT_TRUE(Recorder.runOver(
      "struct Inner { int c; };\n"
      "struct Mid { struct Inner b; };\n"
      "struct Outer { struct Mid a; };\n"
      "unsigned long x = __builtin_offsetof(struct Outer, a.b.c);\n",
      OffsetOfNodeRecorder::Lang_C));
  ASSERT_EQ(3u, Recorder.Visited.size());
  EXPECT_EQ(3, Recorder.Traversed);
  EXPECT_EQ(OffsetOfNode::Field, Recorder.Visited[0].Kind);
  EXPECT_EQ("a", Recorder.Visited[0].Name);
  EXPECT_EQ(OffsetOfNode::Field, Recorder.Visited[1].Kind);
  EXPECT_EQ("b", Recorder.Visited[1].Name);
  EXPECT_EQ(OffsetOfNode::Field, Recorder.Visited[2].Kind);
  EXPECT_EQ("c", Recorder.Visited[2].Name);
  for (const auto &C : Recorder.Visited)
    EXPECT_TRUE(C.PointerStable);
}

TEST(RecursiveASTVisitor, OffsetOfArrayAndField) {
  OffsetOfNodeRecorder Recorder;
  // Two array subscripts mean two Array OffsetOfNodes; their getArrayExprIndex
  // values should be 0 and 1 in source order, indexing into OffsetOfExpr's
  // trailing Expr* array (which holds the literals `2` and `7`).
  EXPECT_TRUE(Recorder.runOver(
      "struct Inner { int c; };\n"
      "struct Mid { struct Inner b[10]; };\n"
      "struct Outer { struct Mid a[5]; };\n"
      "unsigned long x = __builtin_offsetof(struct Outer, a[2].b[7].c);\n",
      OffsetOfNodeRecorder::Lang_C));
  ASSERT_EQ(5u, Recorder.Visited.size());
  EXPECT_EQ(5, Recorder.Traversed);
  EXPECT_EQ(OffsetOfNode::Field, Recorder.Visited[0].Kind);
  EXPECT_EQ("a", Recorder.Visited[0].Name);
  EXPECT_EQ(OffsetOfNode::Array, Recorder.Visited[1].Kind);
  EXPECT_EQ(0u, Recorder.Visited[1].ArrayExprSlot);
  EXPECT_EQ(OffsetOfNode::Field, Recorder.Visited[2].Kind);
  EXPECT_EQ("b", Recorder.Visited[2].Name);
  EXPECT_EQ(OffsetOfNode::Array, Recorder.Visited[3].Kind);
  EXPECT_EQ(1u, Recorder.Visited[3].ArrayExprSlot);
  EXPECT_EQ(OffsetOfNode::Field, Recorder.Visited[4].Kind);
  EXPECT_EQ("c", Recorder.Visited[4].Name);
}

TEST(RecursiveASTVisitor, OffsetOfDependentIdentifier) {
  OffsetOfNodeRecorder Recorder;
  // Bypass TestVisitor::runOver so we can pass -fno-delayed-template-parsing.
  // Otherwise on MSVC-compatible triples the uninstantiated template body
  // is not parsed and the OffsetOfExpr never enters the AST.
  EXPECT_TRUE(tooling::runToolOnCodeWithArgs(
      Recorder.CreateTestAction(),
      "template <typename T>\n"
      "unsigned long off() { return __builtin_offsetof(T, x.y); }\n",
      {"-std=c++11", "-fno-delayed-template-parsing"}));
  // Inside a dependent type the components are recorded as Identifier nodes
  // because the field decls are not yet known. We expect both `x` and `y`.
  ASSERT_EQ(2u, Recorder.Visited.size());
  EXPECT_EQ(OffsetOfNode::Identifier, Recorder.Visited[0].Kind);
  EXPECT_EQ("x", Recorder.Visited[0].Name);
  EXPECT_EQ(OffsetOfNode::Identifier, Recorder.Visited[1].Kind);
  EXPECT_EQ("y", Recorder.Visited[1].Name);
}

// Verifies that overriding only Visit (the typical use case) suffices: the
// default TraverseOffsetOfNode in (Dynamic)RecursiveASTVisitor must dispatch
// to VisitOffsetOfNode.
struct VisitOnly : TestVisitor {
  int Visits = 0;
  bool VisitOffsetOfNode(const OffsetOfNode *Node) override {
    ++Visits;
    return true;
  }
};

TEST(RecursiveASTVisitor, OffsetOfDefaultTraverseDispatchesToVisit) {
  VisitOnly Recorder;
  EXPECT_TRUE(Recorder.runOver(
      "struct Inner { int c; };\n"
      "struct Outer { struct Inner a; };\n"
      "unsigned long x = __builtin_offsetof(struct Outer, a.c);\n",
      VisitOnly::Lang_C));
  EXPECT_EQ(2, Recorder.Visits);
}

} // namespace
