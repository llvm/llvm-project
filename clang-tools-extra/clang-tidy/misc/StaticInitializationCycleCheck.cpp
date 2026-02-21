//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StaticInitializationCycleCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/CallGraph.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SCCIterator.h"

using namespace clang;
using namespace clang::ast_matchers;

// Check if a reference to a static variable (that was reached while traversal
// of a function declaration) should be ignored by the check. This returns true
// if the value of the variable has no effect on the return value of the
// function, or the reference is ignored for other reason to eliminate FP
// results.
// Ignore happens if the variable appears at LHS of an assignment or it appears
// inside a compile-time constant expression (like 'sizeof').
// Additional condition is if the reference appears in a not immediately called
// lambda function.
static bool shouldIgnoreRef(const DeclRefExpr *DRE, const Decl *ParentD) {
  ASTContext &ACtx = ParentD->getASTContext();
  ParentMapContext &PMC = ACtx.getParentMapContext();
  DynTypedNodeList Parents = PMC.getParents(*DRE);
  // While going upwards on the parent graph, this stores the last encountered
  // lambda expression that did not appear (until now) as a callee of a
  // 'operator ()'.
  const LambdaExpr *ParentLambda = nullptr;
  while (!Parents.empty()) {
    if (Parents.size() > 1)
      return true;
    if (const Expr *E = Parents[0].get<Expr>()) {
      if (!E->isValueDependent() && E->isIntegerConstantExpr(ACtx))
        return true;
      if (const auto *ParentBO = dyn_cast<BinaryOperator>(E)) {
        if (ParentBO->isAssignmentOp() &&
            ParentBO->getLHS()->IgnoreParenCasts() == DRE)
          return true;
      } else if (const auto *LambdaE = dyn_cast<LambdaExpr>(E)) {
        // Found another lambda while the last found do not appear to be called
        // by '()'.
        if (ParentLambda)
          return true;
        ParentLambda = LambdaE;
      } else if (const auto *OpCallE = dyn_cast<CXXOperatorCallExpr>(E)) {
        // Check if the last found lambda is called with this 'operator ()'.
        if (ParentLambda &&
            OpCallE->getOperator() == OverloadedOperatorKind::OO_Call &&
            OpCallE->getCalleeDecl() == ParentLambda->getCallOperator())
          ParentLambda = nullptr;
      }
    } else if (const Decl *D = Parents[0].get<Decl>()) {
      // Check if we reached the root of the context (variable or function
      // declaration) to check.
      if ([D, ParentD]() {
            if (const auto *ParentF = dyn_cast<FunctionDecl>(ParentD)) {
              if (const auto *FD = dyn_cast<FunctionDecl>(D))
                return FD == ParentF->getDefinition();
              return false;
            } else {
              return D->getCanonicalDecl() == ParentD->getCanonicalDecl();
            }
          }())
        return ParentLambda != nullptr;
    }
    Parents = PMC.getParents(Parents[0]);
  }
  llvm_unreachable("declaration of ParentD should be reached");
  return false;
}

namespace {

class VarUseNode;

// Store the reference to a variable or the call location of a function.
// 'Ref' is a DeclRefExpr or a CallExpr.
// 'Node' contains information about corresponding VarDecl or FunctionDecl.
struct VarUseRecord {
  const Expr *Ref;
  VarUseNode *Node;

  VarUseRecord() = default;
  VarUseRecord(const Expr *Ref, VarUseNode *N) : Ref(Ref), Node(N) {}
  operator VarUseNode *() const { return Node; }
};

// One node in the variable usage graph.
// If 'D' is a VarDecl:
// 'Uses' contains all static variables and global function calls in the
// initializer expression.
// If 'D' is a FunctionDecl:
// 'Uses' contains all static variable references and global function calls in
// the function body.
class VarUseNode {
  const NamedDecl *D;
  llvm::SmallVector<VarUseRecord, 2> Uses;

public:
  VarUseNode(const NamedDecl *D) : D(D) {}

  const NamedDecl *getDecl() const { return D; }
  bool isVar() const { return isa<VarDecl>(D); }
  bool isFunction() const { return isa<FunctionDecl>(D); }
  const VarDecl *getVar() const { return cast<VarDecl>(D); }
  const FunctionDecl *getFunction() const { return cast<FunctionDecl>(D); }

  using const_iterator = llvm::SmallVectorImpl<VarUseRecord>::const_iterator;

  const_iterator begin() const { return Uses.begin(); }
  const_iterator end() const { return Uses.end(); }

  llvm::iterator_range<const_iterator> uses() const {
    return llvm::make_range(begin(), end());
  }

  bool empty() const { return Uses.empty(); }
  unsigned size() const { return Uses.size(); }

  friend class VarUseCollector;
  friend class VarUseGraphBuilder;
  friend class VarUseGraph;
};

// "Variable usage graph":
// Stores dependencies of variables from other variables or function calls,
// and dependencies of function results from variables or functions.
// Only static variables (static member, static local variable, or global
// variable) and global or static functions are stored.
// Stored are the canonical declarations of variables and definitions of
// functions.
class VarUseGraph {
  using UseMapTy = llvm::DenseMap<const Decl *, std::unique_ptr<VarUseNode>>;

  UseMapTy UseMap;

public:
  VarUseGraph() {
    // A special "root" is added at nullptr location.
    // It contains edges to all other nodes, without a "Ref" expression.
    // This is used by the SCC algorithm.
    UseMap[nullptr] = std::make_unique<VarUseNode>(nullptr);
  }

  VarUseNode *addNode(const NamedDecl *D) {
    std::unique_ptr<VarUseNode> &N = UseMap[D];
    if (N)
      return N.get();
    N = std::make_unique<VarUseNode>(D);
    UseMap[nullptr]->Uses.emplace_back(nullptr, N.get());
    return N.get();
  }

  using const_iterator = UseMapTy::const_iterator;

  const_iterator begin() const { return UseMap.begin(); }
  const_iterator end() const { return UseMap.end(); }

  unsigned size() const { return UseMap.size(); }

  VarUseNode *getRoot() { return UseMap[nullptr].get(); }

  friend class VarUseGraphBuilder;
};

// Collect static variable references and static function calls.
// This is used with initializer expressions and function body statements.
// At initializer expressions only statements (and expressions) should be
// traversed. But for functions declarations are needed too (to reach
// initializations of variables) (only inside the given function).
class VarUseCollector : public DynamicRecursiveASTVisitor {
  VarUseNode *Node;
  VarUseGraph &G;
  const DeclContext *DC;

public:
  VarUseCollector(VarUseNode *N, VarUseGraph &G)
      : Node(N), G(G), DC(N->isFunction() ? N->getFunction() : nullptr) {}

  bool TraverseType(QualType T, bool TraverseQualifier) override {
    return true;
  }
  bool TraverseTypeLoc(TypeLoc TL, bool TraverseQualifier) override {
    return true;
  }
  bool TraverseAttr(Attr *At) override { return true; }
  bool TraverseDecl(Decl *D) override {
    if (DC && DC->containsDecl(D))
      return DynamicRecursiveASTVisitor::TraverseDecl(D);
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *DRE) override {
    if (const auto *VarD = dyn_cast<VarDecl>(DRE->getDecl())) {
      if (!shouldIgnoreRef(DRE, Node->getDecl()) &&
          (VarD->hasGlobalStorage() || VarD->isStaticLocal()))
        Node->Uses.emplace_back(DRE, G.addNode(VarD->getCanonicalDecl()));
    }
    return true;
  }

  bool VisitCallExpr(CallExpr *CE) override {
    if (const FunctionDecl *F = CE->getDirectCallee()) {
      if (F->isGlobal() || F->isStatic()) {
        const FunctionDecl *Def = F->getDefinition();
        if (Def)
          Node->Uses.emplace_back(CE, G.addNode(Def));
      }
    }
    return true;
  }
};

// Build the complete graph by visiting all static variables and functions and
// add all "usages" (children in the graph) to it.
// Every variable and function is visited once (at canonical declaration or the
// definition). When visiting an object, a node for it may already exist
// (without added children) if a reference to it was found already.
class VarUseGraphBuilder : public DynamicRecursiveASTVisitor {
  VarUseGraph &G;

public:
  VarUseGraphBuilder(VarUseGraph &G) : G(G) {}

  bool VisitVarDecl(VarDecl *VD) override {
    if ((VD->hasGlobalStorage() || VD->isStaticLocal()) &&
        VD->isCanonicalDecl()) {
      if (VarDecl *InitD = VD->getInitializingDeclaration()) {
        VarUseNode *N = G.addNode(VD);
        VarUseCollector Collector(N, G);
        Collector.TraverseStmt(InitD->getInit());
      }
    }
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *FD) override {
    if (FD->isGlobal() || FD->isStatic()) {
      if (Stmt *Body = FD->getBody()) {
        VarUseNode *N = G.addNode(FD);
        VarUseCollector Collector(N, G);
        Collector.TraverseStmt(Body);
      }
    }
    return true;
  }
};

} // namespace

namespace llvm {

// These structures are required by scc_iterator.

template <> struct GraphTraits<const VarUseNode *> {
  using NodeType = const VarUseNode;
  using NodeRef = const VarUseNode *;
  using ChildIteratorType = NodeType::const_iterator;

  static NodeType *getEntryNode(const VarUseNode *N) { return N; }
  static ChildIteratorType
  child_begin(NodeType *N) { // NOLINT(readability-identifier-naming)
    return N->begin();
  }
  static ChildIteratorType
  child_end(NodeType *N) { // NOLINT(readability-identifier-naming)
    return N->end();
  }
};

template <>
struct GraphTraits<const VarUseGraph *>
    : public GraphTraits<const VarUseNode *> {
  static NodeType *getEntryNode(const VarUseGraph *G) {
    return const_cast<VarUseGraph *>(G)->getRoot();
  }

  static VarUseNode *getValue(VarUseGraph::const_iterator::value_type &P) {
    return P.second.get();
  }

  using nodes_iterator =
      mapped_iterator<VarUseGraph::const_iterator, decltype(&getValue)>;

  static nodes_iterator
  nodes_begin(const VarUseGraph *G) { // NOLINT(readability-identifier-naming)
    return {G->begin(), &getValue};
  }

  static nodes_iterator
  nodes_end(const VarUseGraph *G) { // NOLINT(readability-identifier-naming)
    return {G->end(), &getValue};
  }

  static unsigned size(const VarUseGraph *G) { return G->size(); }
};

} // namespace llvm

static void
reportCycles(ArrayRef<const VarUseNode *> SCC,
             clang::tidy::misc::StaticInitializationCycleCheck &Chk) {
  // Check if the SCC contains any variable, otherwise it is a function
  // recursion.
  auto NodeIsVar = [](const VarUseNode *N) { return N->isVar(); };
  const auto *VarNode = llvm::find_if(SCC, NodeIsVar);
  if (VarNode == SCC.end())
    return;

  Chk.diag((*VarNode)->getDecl()->getLocation(),
           "static variable initialization cycle detected involving %0")
      << (*VarNode)->getDecl();

  // SCC may contain multiple cycles.
  // Find one path with the front node as start.

  // Lookup if a node is part of current SCC.
  const llvm::SmallPtrSet<const VarUseNode *, 4> SCCElts(SCC.begin(),
                                                         SCC.end());

  // Visit all paths in the SCC until we reach the front again.
  llvm::DenseMap<const VarUseNode *, VarUseNode::const_iterator> NextNode;
  llvm::SmallVector<const VarUseNode *> FoundPath;
  FoundPath.push_back(SCC.front());
  while (!FoundPath.empty()) {
    if (!NextNode.contains(FoundPath.back())) {
      NextNode[FoundPath.back()] = FoundPath.back()->begin();
    } else {
      NextNode[FoundPath.back()]++;
      if (NextNode[FoundPath.back()] == FoundPath.back()->end()) {
        FoundPath.pop_back();
        continue;
      }
    }
    const VarUseNode *N = (*NextNode[FoundPath.back()]).Node;
    if (N == SCC.front())
      break;
    if (!SCCElts.contains(N) || NextNode.contains(N))
      continue;
    FoundPath.push_back(N);
  }

  for (const VarUseNode *N : FoundPath) {
    const VarUseRecord &U = *NextNode[N];
    // 'U' is the source of the value, 'N->getDecl()' is the destination
    const char *VarFuncUseStr = U.Node->isVar() ? "value" : "result";
    if (N->isVar())
      Chk.diag(U.Ref->getBeginLoc(),
               "%0 of %1 may be used to initialize variable %2 here",
               DiagnosticIDs::Note)
          << VarFuncUseStr << U.Node->getDecl() << N->getDecl();
    else
      Chk.diag(U.Ref->getBeginLoc(),
               "%0 of %1 may be used to compute result of %2",
               DiagnosticIDs::Note)
          << VarFuncUseStr << U.Node->getDecl() << N->getDecl();
  }
}

namespace clang::tidy::misc {

void StaticInitializationCycleCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(translationUnitDecl().bind("TUDecl"), this);
}

void StaticInitializationCycleCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *TU = Result.Nodes.getNodeAs<TranslationUnitDecl>("TUDecl");

  VarUseGraph Uses;
  VarUseGraphBuilder Builder(Uses);
  Builder.TraverseDecl(const_cast<TranslationUnitDecl *>(TU));

  for (llvm::scc_iterator<const VarUseGraph *>
           SCCI = llvm::scc_begin(const_cast<const VarUseGraph *>(&Uses)),
           SCCE = llvm::scc_end(const_cast<const VarUseGraph *>(&Uses));
       SCCI != SCCE; ++SCCI) {
    if (!SCCI.hasCycle())
      continue;
    reportCycles(*SCCI, *this);
  }
}

} // namespace clang::tidy::misc
