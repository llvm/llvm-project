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

void ParentMapContext::clear() { Parents.clear(); }

const Expr *ParentMapContext::traverseIgnored(const Expr *E) const {
  return traverseIgnored(const_cast<Expr *>(E));
}

Expr *ParentMapContext::traverseIgnored(Expr *E) const {
  if (!E)
    return nullptr;

  switch (Traversal) {
  case ast_type_traits::TK_AsIs:
    return E;
  case ast_type_traits::TK_IgnoreImplicitCastsAndParentheses:
    return E->IgnoreParenImpCasts();
  case ast_type_traits::TK_IgnoreUnlessSpelledInSource:
    return E->IgnoreUnlessSpelledInSource();
  }
  llvm_unreachable("Invalid Traversal type!");
}

ast_type_traits::DynTypedNode
ParentMapContext::traverseIgnored(const ast_type_traits::DynTypedNode &N) const {
  if (const auto *E = N.get<Expr>()) {
    return ast_type_traits::DynTypedNode::create(*traverseIgnored(E));
  }
  return N;
}

class ParentMapContext::ParentMap {
  /// Contains parents of a node.
  using ParentVector = llvm::SmallVector<ast_type_traits::DynTypedNode, 2>;

  /// Maps from a node to its parents. This is used for nodes that have
  /// pointer identity only, which are more common and we can save space by
  /// only storing a unique pointer to them.
  using ParentMapPointers = llvm::DenseMap<
      const void *,
      llvm::PointerUnion<const Decl *, const Stmt *,
                         ast_type_traits::DynTypedNode *, ParentVector *>>;

  /// Parent map for nodes without pointer identity. We store a full
  /// DynTypedNode for all keys.
  using ParentMapOtherNodes = llvm::DenseMap<
      ast_type_traits::DynTypedNode,
      llvm::PointerUnion<const Decl *, const Stmt *,
                         ast_type_traits::DynTypedNode *, ParentVector *>>;

  ParentMapPointers PointerParents;
  ParentMapOtherNodes OtherParents;
  class ASTVisitor;

  static ast_type_traits::DynTypedNode
  getSingleDynTypedNodeFromParentMap(ParentMapPointers::mapped_type U) {
    if (const auto *D = U.dyn_cast<const Decl *>())
      return ast_type_traits::DynTypedNode::create(*D);
    if (const auto *S = U.dyn_cast<const Stmt *>())
      return ast_type_traits::DynTypedNode::create(*S);
    return *U.get<ast_type_traits::DynTypedNode *>();
  }

  template <typename NodeTy, typename MapTy>
  static DynTypedNodeList getDynNodeFromMap(const NodeTy &Node,
                                                        const MapTy &Map) {
    auto I = Map.find(Node);
    if (I == Map.end()) {
      return llvm::ArrayRef<ast_type_traits::DynTypedNode>();
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
      if (Entry.second.is<ast_type_traits::DynTypedNode *>()) {
        delete Entry.second.get<ast_type_traits::DynTypedNode *>();
      } else if (Entry.second.is<ParentVector *>()) {
        delete Entry.second.get<ParentVector *>();
      }
    }
    for (const auto &Entry : OtherParents) {
      if (Entry.second.is<ast_type_traits::DynTypedNode *>()) {
        delete Entry.second.get<ast_type_traits::DynTypedNode *>();
      } else if (Entry.second.is<ParentVector *>()) {
        delete Entry.second.get<ParentVector *>();
      }
    }
  }

  DynTypedNodeList getParents(const ast_type_traits::DynTypedNode &Node) {
    if (Node.getNodeKind().hasPointerIdentity())
      return getDynNodeFromMap(Node.getMemoizationData(), PointerParents);
    return getDynNodeFromMap(Node, OtherParents);
  }
};

/// Template specializations to abstract away from pointers and TypeLocs.
/// @{
template <typename T>
static ast_type_traits::DynTypedNode createDynTypedNode(const T &Node) {
  return ast_type_traits::DynTypedNode::create(*Node);
}
template <>
ast_type_traits::DynTypedNode createDynTypedNode(const TypeLoc &Node) {
  return ast_type_traits::DynTypedNode::create(Node);
}
template <>
ast_type_traits::DynTypedNode
createDynTypedNode(const NestedNameSpecifierLoc &Node) {
  return ast_type_traits::DynTypedNode::create(Node);
}
/// @}

/// A \c RecursiveASTVisitor that builds a map from nodes to their
/// parents as defined by the \c RecursiveASTVisitor.
///
/// Note that the relationship described here is purely in terms of AST
/// traversal - there are other relationships (for example declaration context)
/// in the AST that are better modeled by special matchers.
///
/// FIXME: Currently only builds up the map using \c Stmt and \c Decl nodes.
class ParentMapContext::ParentMap::ASTVisitor
    : public RecursiveASTVisitor<ASTVisitor> {
public:
  ASTVisitor(ParentMap &Map, ParentMapContext &MapCtx)
      : Map(Map), MapCtx(MapCtx) {}

private:
  friend class RecursiveASTVisitor<ASTVisitor>;

  using VisitorBase = RecursiveASTVisitor<ASTVisitor>;

  bool shouldVisitTemplateInstantiations() const { return true; }

  bool shouldVisitImplicitCode() const { return true; }

  template <typename T, typename MapNodeTy, typename BaseTraverseFn,
            typename MapTy>
  bool TraverseNode(T Node, MapNodeTy MapNode, BaseTraverseFn BaseTraverse,
                    MapTy *Parents) {
    if (!Node)
      return true;
    if (ParentStack.size() > 0) {
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
          NodeOrVector = new ast_type_traits::DynTypedNode(ParentStack.back());
      } else {
        if (!NodeOrVector.template is<ParentVector *>()) {
          auto *Vector = new ParentVector(
              1, getSingleDynTypedNodeFromParentMap(NodeOrVector));
          delete NodeOrVector
              .template dyn_cast<ast_type_traits::DynTypedNode *>();
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

  bool TraverseStmt(Stmt *StmtNode) {
    Stmt *FilteredNode = StmtNode;
    if (auto *ExprNode = dyn_cast_or_null<Expr>(FilteredNode))
      FilteredNode = MapCtx.traverseIgnored(ExprNode);
    return TraverseNode(FilteredNode, FilteredNode,
                        [&] { return VisitorBase::TraverseStmt(FilteredNode); },
                        &Map.PointerParents);
  }

  bool TraverseTypeLoc(TypeLoc TypeLocNode) {
    return TraverseNode(
        TypeLocNode, ast_type_traits::DynTypedNode::create(TypeLocNode),
        [&] { return VisitorBase::TraverseTypeLoc(TypeLocNode); },
        &Map.OtherParents);
  }

  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNSLocNode) {
    return TraverseNode(
        NNSLocNode, ast_type_traits::DynTypedNode::create(NNSLocNode),
        [&] { return VisitorBase::TraverseNestedNameSpecifierLoc(NNSLocNode); },
        &Map.OtherParents);
  }

  ParentMap &Map;
  ParentMapContext &MapCtx;
  llvm::SmallVector<ast_type_traits::DynTypedNode, 16> ParentStack;
};

ParentMapContext::ParentMap::ParentMap(ASTContext &Ctx) {
  ASTVisitor(*this, Ctx.getParentMapContext()).TraverseAST(Ctx);
}

DynTypedNodeList
ParentMapContext::getParents(const ast_type_traits::DynTypedNode &Node) {
  std::unique_ptr<ParentMap> &P = Parents[Traversal];
  if (!P)
    // We build the parent map for the traversal scope (usually whole TU), as
    // hasAncestor can escape any subtree.
    P = std::make_unique<ParentMap>(ASTCtx);
  return P->getParents(Node);
}

