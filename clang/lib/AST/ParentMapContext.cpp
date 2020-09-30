//===- ParentMapContext.cpp - Map of parents using DynTypedNode -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Similar to ParentMap.cpp, but generalizes to non-Stmt nodes, which can have
// multiple parents.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TemplateBase.h"

using namespace clang;

ParentMapContext::ParentMapContext(ASTContext &Ctx) : ASTCtx(Ctx) {}

ParentMapContext::~ParentMapContext() = default;

void ParentMapContext::clear() { Parents.reset(); }

const Expr *ParentMapContext::traverseIgnored(const Expr *E) const {
  return traverseIgnored(const_cast<Expr *>(E));
}

Expr *ParentMapContext::traverseIgnored(Expr *E) const {
  if (!E)
    return nullptr;

  switch (Traversal) {
  case TK_AsIs:
    return E;
  case TK_IgnoreImplicitCastsAndParentheses:
    return E->IgnoreParenImpCasts();
  case TK_IgnoreUnlessSpelledInSource:
    return E->IgnoreUnlessSpelledInSource();
  }
  llvm_unreachable("Invalid Traversal type!");
}

DynTypedNode ParentMapContext::traverseIgnored(const DynTypedNode &N) const {
  if (const auto *E = N.get<Expr>()) {
    return DynTypedNode::create(*traverseIgnored(E));
  }
  return N;
}

class ParentMapContext::ParentMap {
  /// Contains parents of a node.
  using ParentVector = llvm::SmallVector<DynTypedNode, 2>;

  /// Maps from a node to its parents. This is used for nodes that have
  /// pointer identity only, which are more common and we can save space by
  /// only storing a unique pointer to them.
  using ParentMapPointers =
      llvm::DenseMap<const void *,
                     llvm::PointerUnion<const Decl *, const Stmt *,
                                        DynTypedNode *, ParentVector *>>;

  /// Parent map for nodes without pointer identity. We store a full
  /// DynTypedNode for all keys.
  using ParentMapOtherNodes =
      llvm::DenseMap<DynTypedNode,
                     llvm::PointerUnion<const Decl *, const Stmt *,
                                        DynTypedNode *, ParentVector *>>;

  ParentMapPointers PointerParents;
  ParentMapOtherNodes OtherParents;
  class ASTVisitor;

  static DynTypedNode
  getSingleDynTypedNodeFromParentMap(ParentMapPointers::mapped_type U) {
    if (const auto *D = U.dyn_cast<const Decl *>())
      return DynTypedNode::create(*D);
    if (const auto *S = U.dyn_cast<const Stmt *>())
      return DynTypedNode::create(*S);
    return *U.get<DynTypedNode *>();
  }

  template <typename NodeTy, typename MapTy>
  static DynTypedNodeList getDynNodeFromMap(const NodeTy &Node,
                                                        const MapTy &Map) {
    auto I = Map.find(Node);
    if (I == Map.end()) {
      return llvm::ArrayRef<DynTypedNode>();
    }
    if (const auto *V = I->second.template dyn_cast<ParentVector *>()) {
      return llvm::makeArrayRef(*V);
    }
    return getSingleDynTypedNodeFromParentMap(I->second);
  }

public:
  ParentMap(ASTContext &Ctx);
  ~ParentMap() {
    for (const auto &Entry : PointerParents) {
      if (Entry.second.is<DynTypedNode *>()) {
        delete Entry.second.get<DynTypedNode *>();
      } else if (Entry.second.is<ParentVector *>()) {
        delete Entry.second.get<ParentVector *>();
      }
    }
    for (const auto &Entry : OtherParents) {
      if (Entry.second.is<DynTypedNode *>()) {
        delete Entry.second.get<DynTypedNode *>();
      } else if (Entry.second.is<ParentVector *>()) {
        delete Entry.second.get<ParentVector *>();
      }
    }
  }

  DynTypedNodeList getParents(TraversalKind TK, const DynTypedNode &Node) {
    if (Node.getNodeKind().hasPointerIdentity()) {
      auto ParentList =
          getDynNodeFromMap(Node.getMemoizationData(), PointerParents);
      if (ParentList.size() == 1 && TK == TK_IgnoreUnlessSpelledInSource) {
        const auto *E = ParentList[0].get<Expr>();
        const auto *Child = Node.get<Expr>();
        if (E && Child)
          return AscendIgnoreUnlessSpelledInSource(E, Child);
      }
      return ParentList;
    }
    return getDynNodeFromMap(Node, OtherParents);
  }

  DynTypedNodeList AscendIgnoreUnlessSpelledInSource(const Expr *E,
                                                     const Expr *Child) {

    auto ShouldSkip = [](const Expr *E, const Expr *Child) {
      if (isa<ImplicitCastExpr>(E))
        return true;

      if (isa<FullExpr>(E))
        return true;

      if (isa<MaterializeTemporaryExpr>(E))
        return true;

      if (isa<CXXBindTemporaryExpr>(E))
        return true;

      if (isa<ParenExpr>(E))
        return true;

      if (isa<ExprWithCleanups>(E))
        return true;

      auto SR = Child->getSourceRange();

      if (const auto *C = dyn_cast<CXXConstructExpr>(E)) {
        if (C->getSourceRange() == SR || !isa<CXXTemporaryObjectExpr>(C))
          return true;
      }

      if (const auto *C = dyn_cast<CXXMemberCallExpr>(E)) {
        if (C->getSourceRange() == SR)
          return true;
      }

      if (const auto *C = dyn_cast<MemberExpr>(E)) {
        if (C->getSourceRange() == SR)
          return true;
      }
      return false;
    };

    while (ShouldSkip(E, Child)) {
      auto It = PointerParents.find(E);
      if (It == PointerParents.end())
        break;
      const auto *S = It->second.dyn_cast<const Stmt *>();
      if (!S) {
        if (auto *Vec = It->second.dyn_cast<ParentVector *>())
          return llvm::makeArrayRef(*Vec);
        return getSingleDynTypedNodeFromParentMap(It->second);
      }
      const auto *P = dyn_cast<Expr>(S);
      if (!P)
        return DynTypedNode::create(*S);
      Child = E;
      E = P;
    }
    return DynTypedNode::create(*E);
  }
};

/// Template specializations to abstract away from pointers and TypeLocs.
/// @{
template <typename T> static DynTypedNode createDynTypedNode(const T &Node) {
  return DynTypedNode::create(*Node);
}
template <> DynTypedNode createDynTypedNode(const TypeLoc &Node) {
  return DynTypedNode::create(Node);
}
template <>
DynTypedNode createDynTypedNode(const NestedNameSpecifierLoc &Node) {
  return DynTypedNode::create(Node);
}
/// @}

/// A \c RecursiveASTVisitor that builds a map from nodes to their
/// parents as defined by the \c RecursiveASTVisitor.
///
/// Note that the relationship described here is purely in terms of AST
/// traversal - there are other relationships (for example declaration context)
/// in the AST that are better modeled by special matchers.
class ParentMapContext::ParentMap::ASTVisitor
    : public RecursiveASTVisitor<ASTVisitor> {
public:
  ASTVisitor(ParentMap &Map) : Map(Map) {}

private:
  friend class RecursiveASTVisitor<ASTVisitor>;

  using VisitorBase = RecursiveASTVisitor<ASTVisitor>;

  bool shouldVisitTemplateInstantiations() const { return true; }

  bool shouldVisitImplicitCode() const { return true; }

  /// Record the parent of the node we're visiting.
  /// MapNode is the child, the parent is on top of ParentStack.
  /// Parents is the parent storage (either PointerParents or OtherParents).
  template <typename MapNodeTy, typename MapTy>
  void addParent(MapNodeTy MapNode, MapTy *Parents) {
    if (ParentStack.empty())
      return;

    // FIXME: Currently we add the same parent multiple times, but only
    // when no memoization data is available for the type.
    // For example when we visit all subexpressions of template
    // instantiations; this is suboptimal, but benign: the only way to
    // visit those is with hasAncestor / hasParent, and those do not create
    // new matches.
    // The plan is to enable DynTypedNode to be storable in a map or hash
    // map. The main problem there is to implement hash functions /
    // comparison operators for all types that DynTypedNode supports that
    // do not have pointer identity.
    auto &NodeOrVector = (*Parents)[MapNode];
    if (NodeOrVector.isNull()) {
      if (const auto *D = ParentStack.back().get<Decl>())
        NodeOrVector = D;
      else if (const auto *S = ParentStack.back().get<Stmt>())
        NodeOrVector = S;
      else
        NodeOrVector = new DynTypedNode(ParentStack.back());
    } else {
      if (!NodeOrVector.template is<ParentVector *>()) {
        auto *Vector = new ParentVector(
            1, getSingleDynTypedNodeFromParentMap(NodeOrVector));
        delete NodeOrVector.template dyn_cast<DynTypedNode *>();
        NodeOrVector = Vector;
      }

      auto *Vector = NodeOrVector.template get<ParentVector *>();
      // Skip duplicates for types that have memoization data.
      // We must check that the type has memoization data before calling
      // std::find() because DynTypedNode::operator== can't compare all
      // types.
      bool Found = ParentStack.back().getMemoizationData() &&
                   std::find(Vector->begin(), Vector->end(),
                             ParentStack.back()) != Vector->end();
      if (!Found)
        Vector->push_back(ParentStack.back());
    }
  }

  template <typename T, typename MapNodeTy, typename BaseTraverseFn,
            typename MapTy>
  bool TraverseNode(T Node, MapNodeTy MapNode, BaseTraverseFn BaseTraverse,
                    MapTy *Parents) {
    if (!Node)
      return true;
    addParent(MapNode, Parents);
    ParentStack.push_back(createDynTypedNode(Node));
    bool Result = BaseTraverse();
    ParentStack.pop_back();
    return Result;
  }

  bool TraverseDecl(Decl *DeclNode) {
    return TraverseNode(
        DeclNode, DeclNode, [&] { return VisitorBase::TraverseDecl(DeclNode); },
        &Map.PointerParents);
  }
  bool TraverseTypeLoc(TypeLoc TypeLocNode) {
    return TraverseNode(
        TypeLocNode, DynTypedNode::create(TypeLocNode),
        [&] { return VisitorBase::TraverseTypeLoc(TypeLocNode); },
        &Map.OtherParents);
  }
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNSLocNode) {
    return TraverseNode(
        NNSLocNode, DynTypedNode::create(NNSLocNode),
        [&] { return VisitorBase::TraverseNestedNameSpecifierLoc(NNSLocNode); },
        &Map.OtherParents);
  }

  // Using generic TraverseNode for Stmt would prevent data-recursion.
  bool dataTraverseStmtPre(Stmt *StmtNode) {
    addParent(StmtNode, &Map.PointerParents);
    ParentStack.push_back(DynTypedNode::create(*StmtNode));
    return true;
  }
  bool dataTraverseStmtPost(Stmt *StmtNode) {
    ParentStack.pop_back();
    return true;
  }

  ParentMap &Map;
  llvm::SmallVector<DynTypedNode, 16> ParentStack;
};

ParentMapContext::ParentMap::ParentMap(ASTContext &Ctx) {
  ASTVisitor(*this).TraverseAST(Ctx);
}

DynTypedNodeList ParentMapContext::getParents(const DynTypedNode &Node) {
  if (!Parents)
    // We build the parent map for the traversal scope (usually whole TU), as
    // hasAncestor can escape any subtree.
    Parents = std::make_unique<ParentMap>(ASTCtx);
  return Parents->getParents(getTraversalKind(), Node);
}
