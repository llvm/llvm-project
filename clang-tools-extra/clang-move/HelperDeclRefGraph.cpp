//===-- HelperDeclRefGraph.cpp - AST-based call graph for helper decls ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HelperDeclRefGraph.h"
#include "ClangMove.h"
#include "clang/AST/Decl.h"
#include "llvm/Support/Debug.h"
#include <vector>

#define DEBUG_TYPE "clang-move"

namespace clang {
namespace move {

void HelperDeclRefGraph::print(raw_ostream &OS) const {
  OS << " --- Call graph Dump --- \n";
  for (auto I = DeclMap.begin(); I != DeclMap.end(); ++I) {
    const CallGraphNode *N = (I->second).get();

    OS << "  Declarations: ";
    N->print(OS);
    OS << " (" << N << ") ";
    OS << " calls: ";
    for (auto CI = N->begin(), CE = N->end(); CI != CE; ++CI) {
      (*CI)->print(OS);
      OS << " (" << CI << ") ";
    }
    OS << '\n';
  }
  OS.flush();
}

void HelperDeclRefGraph::addEdge(const Decl *Caller, const Decl *Callee) {
  assert(Caller);
  assert(Callee);

  // Ignore the case where Caller equals Callee. This happens in the static
  // class member definitions in global namespace like "int CLASS::static_var =
  // 1;", its DC is a VarDel whose outmost enclosing declaration is the "CLASS"
  // CXXRecordDecl.
  if (Caller == Callee) return;

  // Allocate a new node, mark it as root, and process it's calls.
  CallGraphNode *CallerNode = getOrInsertNode(const_cast<Decl *>(Caller));
  CallGraphNode *CalleeNode = getOrInsertNode(const_cast<Decl *>(Callee));
  CallerNode->addCallee(CalleeNode);
}

void HelperDeclRefGraph::dump() const { print(llvm::errs()); }

CallGraphNode *HelperDeclRefGraph::getOrInsertNode(Decl *F) {
  F = F->getCanonicalDecl();
  std::unique_ptr<CallGraphNode> &Node = DeclMap[F];
  if (Node)
    return Node.get();

  Node = llvm::make_unique<CallGraphNode>(F);
  return Node.get();
}

CallGraphNode *HelperDeclRefGraph::getNode(const Decl *D) const {
  auto I = DeclMap.find(D->getCanonicalDecl());
  return I == DeclMap.end() ? nullptr : I->second.get();
}

llvm::DenseSet<const CallGraphNode *>
HelperDeclRefGraph::getReachableNodes(const Decl *Root) const {
  const auto *RootNode = getNode(Root);
  if (!RootNode)
    return {};
  llvm::DenseSet<const CallGraphNode *> ConnectedNodes;
  std::function<void(const CallGraphNode *)> VisitNode =
      [&](const CallGraphNode *Node) {
        if (ConnectedNodes.count(Node))
          return;
        ConnectedNodes.insert(Node);
        for (auto It = Node->begin(), End = Node->end(); It != End; ++It)
          VisitNode(*It);
      };

  VisitNode(RootNode);
  return ConnectedNodes;
}

const Decl *HelperDeclRGBuilder::getOutmostClassOrFunDecl(const Decl *D) {
  const auto *DC = D->getDeclContext();
  const auto *Result = D;
  while (DC) {
    if (const auto *RD = dyn_cast<CXXRecordDecl>(DC))
      Result = RD;
    else if (const auto *FD = dyn_cast<FunctionDecl>(DC))
      Result = FD;
    DC = DC->getParent();
  }
  return Result;
}

void HelperDeclRGBuilder::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  // Construct the graph by adding a directed edge from caller to callee.
  //
  // "dc" is the closest ancestor declaration of "func_ref" or "used_class", it
  // might be not the targetted Caller Decl, we always use the outmost enclosing
  // FunctionDecl/CXXRecordDecl of "dc". For example,
  //
  //   int MoveClass::F() { int a = helper(); return a; }
  //
  // The matched "dc" of "helper" DeclRefExpr is a VarDecl, we traverse up AST
  // to find the outmost "MoveClass" CXXRecordDecl and use it as Caller.
  if (const auto *FuncRef = Result.Nodes.getNodeAs<DeclRefExpr>("func_ref")) {
    const auto *DC = Result.Nodes.getNodeAs<Decl>("dc");
    assert(DC);
    LLVM_DEBUG(llvm::dbgs() << "Find helper function usage: "
                            << FuncRef->getDecl()->getNameAsString() << " ("
                            << FuncRef->getDecl() << ")\n");
    RG->addEdge(
        getOutmostClassOrFunDecl(DC->getCanonicalDecl()),
        getOutmostClassOrFunDecl(FuncRef->getDecl()->getCanonicalDecl()));
  } else if (const auto *UsedClass =
                 Result.Nodes.getNodeAs<CXXRecordDecl>("used_class")) {
    const auto *DC = Result.Nodes.getNodeAs<Decl>("dc");
    assert(DC);
    LLVM_DEBUG(llvm::dbgs()
               << "Find helper class usage: " << UsedClass->getNameAsString()
               << " (" << UsedClass << ")\n");
    RG->addEdge(getOutmostClassOrFunDecl(DC->getCanonicalDecl()), UsedClass);
  }
}

} // namespace move
} // namespace clang
