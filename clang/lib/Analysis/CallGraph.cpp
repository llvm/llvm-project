//== CallGraph.cpp - AST-based Call graph  ----------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the AST-based CallGraph.
//
//===----------------------------------------------------------------------===//
#include "clang/Analysis/CallGraph.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/StmtVisitor.h"

#include "llvm/Support/GraphWriter.h"

using namespace clang;

namespace {
/// A helper class, which walks the AST and locates all the call sites in the
/// given function body.
class CGBuilder : public StmtVisitor<CGBuilder> {
  CallGraph *G;
  CallGraphNode *CallerNode;

public:
  CGBuilder(CallGraph *g, CallGraphNode *N)
    : G(g), CallerNode(N) {}

  void VisitStmt(Stmt *S) { VisitChildren(S); }

  void VisitCallExpr(CallExpr *CE) {
    // TODO: We need to handle ObjC method calls as well.
    if (FunctionDecl *CalleeDecl = CE->getDirectCallee())
      if (G->includeInGraph(CalleeDecl)) {
        CallGraphNode *CalleeNode = G->getOrInsertNode(CalleeDecl);
        CallerNode->addCallee(CalleeNode, G);
      }
  }

  void VisitChildren(Stmt *S) {
    for (Stmt::child_range I = S->children(); I; ++I)
      if (*I)
        static_cast<CGBuilder*>(this)->Visit(*I);
  }
};

} // end anonymous namespace

CallGraph::CallGraph() {
  Root = getOrInsertNode(0);
}

CallGraph::~CallGraph() {
  if (!FunctionMap.empty()) {
    for (FunctionMapTy::iterator I = FunctionMap.begin(), E = FunctionMap.end();
        I != E; ++I)
      delete I->second;
    FunctionMap.clear();
  }
}

bool CallGraph::includeInGraph(const Decl *D) {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // We skip function template definitions, as their semantics is
    // only determined when they are instantiated.
    if (!FD->isThisDeclarationADefinition() ||
        FD->isDependentContext())
      return false;

    IdentifierInfo *II = FD->getIdentifier();
    if (II && II->getName().startswith("__inline"))
      return false;
  }

  if (const ObjCMethodDecl *ID = dyn_cast<ObjCMethodDecl>(D)) {
    if (!ID->isThisDeclarationADefinition())
      return false;
  }

  return true;
}

void CallGraph::addNodeForDecl(Decl* D, bool IsGlobal) {
  assert(D);

  // Do nothing if the node already exists.
  if (FunctionMap.find(D) != FunctionMap.end())
    return;

  // Allocate a new node, mark it as root, and process it's calls.
  CallGraphNode *Node = getOrInsertNode(D);
  if (IsGlobal)
    Root->addCallee(Node, this);

  // Process all the calls by this function as well.
  CGBuilder builder(this, Node);
  if (Stmt *Body = D->getBody())
    builder.Visit(Body);
}

CallGraphNode *CallGraph::getNode(const Decl *F) const {
  FunctionMapTy::const_iterator I = FunctionMap.find(F);
  if (I == FunctionMap.end()) return 0;
  return I->second;
}

CallGraphNode *CallGraph::getOrInsertNode(Decl *F) {
  CallGraphNode *&Node = FunctionMap[F];
  if (Node)
    return Node;

  Node = new CallGraphNode(F);
  // If not root, add to the parentless list.
  if (F != 0)
    ParentlessNodes.insert(Node);
  return Node;
}

void CallGraph::print(raw_ostream &OS) const {
  OS << " --- Call graph Dump --- \n";
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    OS << "  Function: ";
    if (I->second == Root)
      OS << "< root >";
    else
      I->second->print(OS);
    OS << " calls: ";
    for (CallGraphNode::iterator CI = I->second->begin(),
        CE = I->second->end(); CI != CE; ++CI) {
      assert(*CI != Root && "No one can call the root node.");
      (*CI)->print(OS);
      OS << " ";
    }
    OS << '\n';
  }
  OS.flush();
}

void CallGraph::dump() const {
  print(llvm::errs());
}

void CallGraph::viewGraph() const {
  llvm::ViewGraph(this, "CallGraph");
}

StringRef CallGraphNode::getName() const {
  if (const FunctionDecl *D = dyn_cast_or_null<FunctionDecl>(FD))
    if (const IdentifierInfo *II = D->getIdentifier())
      return II->getName();
    return "< >";
}

void CallGraphNode::print(raw_ostream &os) const {
  os << getName();
}

void CallGraphNode::dump() const {
  print(llvm::errs());
}

namespace llvm {

template <>
struct DOTGraphTraits<const CallGraph*> : public DefaultDOTGraphTraits {

  DOTGraphTraits (bool isSimple=false) : DefaultDOTGraphTraits(isSimple) {}

  static std::string getNodeLabel(const CallGraphNode *Node,
                                  const CallGraph *CG) {
    if (CG->getRoot() == Node) {
      return "< root >";
    }
    return Node->getName();
  }

};
}
