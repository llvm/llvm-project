//===--- ASTMatchFinder.cpp - Structural query framework ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Implements an algorithm to efficiently search for matches on AST nodes.
//  Uses memoization to support recursive matches like HasDescendant.
//
//  The general idea is to visit all AST nodes with a RecursiveASTVisitor,
//  calling the Matches(...) method of each matcher we are running on each
//  AST node. The matcher can recurse via the ASTMatchFinder interface.
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Timer.h"
#include <deque>
#include <memory>
#include <set>

namespace clang {
namespace ast_matchers {
namespace internal {
namespace {

typedef MatchFinder::MatchCallback MatchCallback;

// The maximum number of memoization entries to store.
// 10k has been experimentally found to give a good trade-off
// of performance vs. memory consumption by running matcher
// that match on every statement over a very large codebase.
//
// FIXME: Do some performance optimization in general and
// revisit this number; also, put up micro-benchmarks that we can
// optimize this on.
static const unsigned MaxMemoizationEntries = 10000;

enum class MatchType {
  Ancestors,

  Descendants,
  Child,
};

// We use memoization to avoid running the same matcher on the same
// AST node twice.  This struct is the key for looking up match
// result.  It consists of an ID of the MatcherInterface (for
// identifying the matcher), a pointer to the AST node and the
// bound nodes before the matcher was executed.
//
// We currently only memoize on nodes whose pointers identify the
// nodes (\c Stmt and \c Decl, but not \c QualType or \c TypeLoc).
// For \c QualType and \c TypeLoc it is possible to implement
// generation of keys for each type.
// FIXME: Benchmark whether memoization of non-pointer typed nodes
// provides enough benefit for the additional amount of code.
struct MatchKey {
  DynTypedMatcher::MatcherIDType MatcherID;
  DynTypedNode Node;
  BoundNodesTreeBuilder BoundNodes;
  TraversalKind Traversal = TK_AsIs;
  MatchType Type;

  bool operator<(const MatchKey &Other) const {
    return std::tie(Traversal, Type, MatcherID, Node, BoundNodes) <
           std::tie(Other.Traversal, Other.Type, Other.MatcherID, Other.Node,
                    Other.BoundNodes);
  }
};

// Used to store the result of a match and possibly bound nodes.
struct MemoizedMatchResult {
  bool ResultOfMatch;
  BoundNodesTreeBuilder Nodes;
};

// A RecursiveASTVisitor that traverses all children or all descendants of
// a node.
class MatchChildASTVisitor
    : public RecursiveASTVisitor<MatchChildASTVisitor> {
public:
  typedef RecursiveASTVisitor<MatchChildASTVisitor> VisitorBase;

  // Creates an AST visitor that matches 'matcher' on all children or
  // descendants of a traversed node. max_depth is the maximum depth
  // to traverse: use 1 for matching the children and INT_MAX for
  // matching the descendants.
  MatchChildASTVisitor(const DynTypedMatcher *Matcher, ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder, int MaxDepth,
                       TraversalKind Traversal, ASTMatchFinder::BindKind Bind)
      : Matcher(Matcher), Finder(Finder), Builder(Builder), CurrentDepth(0),
        MaxDepth(MaxDepth), Traversal(Traversal), Bind(Bind), Matches(false) {}

  // Returns true if a match is found in the subtree rooted at the
  // given AST node. This is done via a set of mutually recursive
  // functions. Here's how the recursion is done (the  *wildcard can
  // actually be Decl, Stmt, or Type):
  //
  //   - Traverse(node) calls BaseTraverse(node) when it needs
  //     to visit the descendants of node.
  //   - BaseTraverse(node) then calls (via VisitorBase::Traverse*(node))
  //     Traverse*(c) for each child c of 'node'.
  //   - Traverse*(c) in turn calls Traverse(c), completing the
  //     recursion.
  bool findMatch(const DynTypedNode &DynNode) {
    reset();
    if (const Decl *D = DynNode.get<Decl>())
      traverse(*D);
    else if (const Stmt *S = DynNode.get<Stmt>())
      traverse(*S);
    else if (const NestedNameSpecifier *NNS =
             DynNode.get<NestedNameSpecifier>())
      traverse(*NNS);
    else if (const NestedNameSpecifierLoc *NNSLoc =
             DynNode.get<NestedNameSpecifierLoc>())
      traverse(*NNSLoc);
    else if (const QualType *Q = DynNode.get<QualType>())
      traverse(*Q);
    else if (const TypeLoc *T = DynNode.get<TypeLoc>())
      traverse(*T);
    else if (const auto *C = DynNode.get<CXXCtorInitializer>())
      traverse(*C);
    else if (const TemplateArgumentLoc *TALoc =
                 DynNode.get<TemplateArgumentLoc>())
      traverse(*TALoc);
    // FIXME: Add other base types after adding tests.

    // It's OK to always overwrite the bound nodes, as if there was
    // no match in this recursive branch, the result set is empty
    // anyway.
    *Builder = ResultBindings;

    return Matches;
  }

  // The following are overriding methods from the base visitor class.
  // They are public only to allow CRTP to work. They are *not *part
  // of the public API of this class.
  bool TraverseDecl(Decl *DeclNode) {
    ScopedIncrement ScopedDepth(&CurrentDepth);
    return (DeclNode == nullptr) || traverse(*DeclNode);
  }

  Stmt *getStmtToTraverse(Stmt *StmtNode) {
    Stmt *StmtToTraverse = StmtNode;
    if (auto *ExprNode = dyn_cast_or_null<Expr>(StmtNode)) {
      auto *LambdaNode = dyn_cast_or_null<LambdaExpr>(StmtNode);
      if (LambdaNode &&
          Finder->getASTContext().getParentMapContext().getTraversalKind() ==
              TK_IgnoreUnlessSpelledInSource)
        StmtToTraverse = LambdaNode;
      else
        StmtToTraverse =
            Finder->getASTContext().getParentMapContext().traverseIgnored(
                ExprNode);
    }
    if (Traversal == TraversalKind::TK_IgnoreImplicitCastsAndParentheses) {
      if (Expr *ExprNode = dyn_cast_or_null<Expr>(StmtNode))
        StmtToTraverse = ExprNode->IgnoreParenImpCasts();
    }
    return StmtToTraverse;
  }

  bool TraverseStmt(Stmt *StmtNode, DataRecursionQueue *Queue = nullptr) {
    // If we need to keep track of the depth, we can't perform data recursion.
    if (CurrentDepth == 0 || (CurrentDepth <= MaxDepth && MaxDepth < INT_MAX))
      Queue = nullptr;

    ScopedIncrement ScopedDepth(&CurrentDepth);
    Stmt *StmtToTraverse = getStmtToTraverse(StmtNode);
    if (!StmtToTraverse)
      return true;
    if (!match(*StmtToTraverse))
      return false;
    return VisitorBase::TraverseStmt(StmtToTraverse, Queue);
  }
  // We assume that the QualType and the contained type are on the same
  // hierarchy level. Thus, we try to match either of them.
  bool TraverseType(QualType TypeNode) {
    if (TypeNode.isNull())
      return true;
    ScopedIncrement ScopedDepth(&CurrentDepth);
    // Match the Type.
    if (!match(*TypeNode))
      return false;
    // The QualType is matched inside traverse.
    return traverse(TypeNode);
  }
  // We assume that the TypeLoc, contained QualType and contained Type all are
  // on the same hierarchy level. Thus, we try to match all of them.
  bool TraverseTypeLoc(TypeLoc TypeLocNode) {
    if (TypeLocNode.isNull())
      return true;
    ScopedIncrement ScopedDepth(&CurrentDepth);
    // Match the Type.
    if (!match(*TypeLocNode.getType()))
      return false;
    // Match the QualType.
    if (!match(TypeLocNode.getType()))
      return false;
    // The TypeLoc is matched inside traverse.
    return traverse(TypeLocNode);
  }
  bool TraverseNestedNameSpecifier(NestedNameSpecifier *NNS) {
    ScopedIncrement ScopedDepth(&CurrentDepth);
    return (NNS == nullptr) || traverse(*NNS);
  }
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS) {
    if (!NNS)
      return true;
    ScopedIncrement ScopedDepth(&CurrentDepth);
    if (!match(*NNS.getNestedNameSpecifier()))
      return false;
    return traverse(NNS);
  }
  bool TraverseConstructorInitializer(CXXCtorInitializer *CtorInit) {
    if (!CtorInit)
      return true;
    ScopedIncrement ScopedDepth(&CurrentDepth);
    return traverse(*CtorInit);
  }
  bool TraverseTemplateArgumentLoc(TemplateArgumentLoc TAL) {
    ScopedIncrement ScopedDepth(&CurrentDepth);
    return traverse(TAL);
  }
  bool TraverseLambdaExpr(LambdaExpr *Node) {
    if (Finder->getASTContext().getParentMapContext().getTraversalKind() !=
        TK_IgnoreUnlessSpelledInSource)
      return VisitorBase::TraverseLambdaExpr(Node);
    if (!Node)
      return true;
    ScopedIncrement ScopedDepth(&CurrentDepth);

    for (unsigned I = 0, N = Node->capture_size(); I != N; ++I) {
      const auto *C = Node->capture_begin() + I;
      if (!C->isExplicit())
        continue;
      if (Node->isInitCapture(C) && !match(*C->getCapturedVar()))
        return false;
      if (!match(*Node->capture_init_begin()[I]))
        return false;
    }

    if (const auto *TPL = Node->getTemplateParameterList()) {
      for (const auto *TP : *TPL) {
        if (!match(*TP))
          return false;
      }
    }

    for (const auto *P : Node->getCallOperator()->parameters()) {
      if (!match(*P))
        return false;
    }

    if (!match(*Node->getBody()))
      return false;

    return true;
  }

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }

private:
  // Used for updating the depth during traversal.
  struct ScopedIncrement {
    explicit ScopedIncrement(int *Depth) : Depth(Depth) { ++(*Depth); }
    ~ScopedIncrement() { --(*Depth); }

   private:
    int *Depth;
  };

  // Resets the state of this object.
  void reset() {
    Matches = false;
    CurrentDepth = 0;
  }

  // Forwards the call to the corresponding Traverse*() method in the
  // base visitor class.
  bool baseTraverse(const Decl &DeclNode) {
    return VisitorBase::TraverseDecl(const_cast<Decl*>(&DeclNode));
  }
  bool baseTraverse(const Stmt &StmtNode) {
    return VisitorBase::TraverseStmt(const_cast<Stmt*>(&StmtNode));
  }
  bool baseTraverse(QualType TypeNode) {
    return VisitorBase::TraverseType(TypeNode);
  }
  bool baseTraverse(TypeLoc TypeLocNode) {
    return VisitorBase::TraverseTypeLoc(TypeLocNode);
  }
  bool baseTraverse(const NestedNameSpecifier &NNS) {
    return VisitorBase::TraverseNestedNameSpecifier(
        const_cast<NestedNameSpecifier*>(&NNS));
  }
  bool baseTraverse(NestedNameSpecifierLoc NNS) {
    return VisitorBase::TraverseNestedNameSpecifierLoc(NNS);
  }
  bool baseTraverse(const CXXCtorInitializer &CtorInit) {
    return VisitorBase::TraverseConstructorInitializer(
        const_cast<CXXCtorInitializer *>(&CtorInit));
  }
  bool baseTraverse(TemplateArgumentLoc TAL) {
    return VisitorBase::TraverseTemplateArgumentLoc(TAL);
  }

  // Sets 'Matched' to true if 'Matcher' matches 'Node' and:
  //   0 < CurrentDepth <= MaxDepth.
  //
  // Returns 'true' if traversal should continue after this function
  // returns, i.e. if no match is found or 'Bind' is 'BK_All'.
  template <typename T>
  bool match(const T &Node) {
    if (CurrentDepth == 0 || CurrentDepth > MaxDepth) {
      return true;
    }
    if (Bind != ASTMatchFinder::BK_All) {
      BoundNodesTreeBuilder RecursiveBuilder(*Builder);
      if (Matcher->matches(DynTypedNode::create(Node), Finder,
                           &RecursiveBuilder)) {
        Matches = true;
        ResultBindings.addMatch(RecursiveBuilder);
        return false; // Abort as soon as a match is found.
      }
    } else {
      BoundNodesTreeBuilder RecursiveBuilder(*Builder);
      if (Matcher->matches(DynTypedNode::create(Node), Finder,
                           &RecursiveBuilder)) {
        // After the first match the matcher succeeds.
        Matches = true;
        ResultBindings.addMatch(RecursiveBuilder);
      }
    }
    return true;
  }

  // Traverses the subtree rooted at 'Node'; returns true if the
  // traversal should continue after this function returns.
  template <typename T>
  bool traverse(const T &Node) {
    static_assert(IsBaseType<T>::value,
                  "traverse can only be instantiated with base type");
    if (!match(Node))
      return false;
    return baseTraverse(Node);
  }

  const DynTypedMatcher *const Matcher;
  ASTMatchFinder *const Finder;
  BoundNodesTreeBuilder *const Builder;
  BoundNodesTreeBuilder ResultBindings;
  int CurrentDepth;
  const int MaxDepth;
  const TraversalKind Traversal;
  const ASTMatchFinder::BindKind Bind;
  bool Matches;
};

// Controls the outermost traversal of the AST and allows to match multiple
// matchers.
class MatchASTVisitor : public RecursiveASTVisitor<MatchASTVisitor>,
                        public ASTMatchFinder {
public:
  MatchASTVisitor(const MatchFinder::MatchersByType *Matchers,
                  const MatchFinder::MatchFinderOptions &Options)
      : Matchers(Matchers), Options(Options), ActiveASTContext(nullptr) {}

  ~MatchASTVisitor() override {
    if (Options.CheckProfiling) {
      Options.CheckProfiling->Records = std::move(TimeByBucket);
    }
  }

  void onStartOfTranslationUnit() {
    const bool EnableCheckProfiling = Options.CheckProfiling.hasValue();
    TimeBucketRegion Timer;
    for (MatchCallback *MC : Matchers->AllCallbacks) {
      if (EnableCheckProfiling)
        Timer.setBucket(&TimeByBucket[MC->getID()]);
      MC->onStartOfTranslationUnit();
    }
  }

  void onEndOfTranslationUnit() {
    const bool EnableCheckProfiling = Options.CheckProfiling.hasValue();
    TimeBucketRegion Timer;
    for (MatchCallback *MC : Matchers->AllCallbacks) {
      if (EnableCheckProfiling)
        Timer.setBucket(&TimeByBucket[MC->getID()]);
      MC->onEndOfTranslationUnit();
    }
  }

  void set_active_ast_context(ASTContext *NewActiveASTContext) {
    ActiveASTContext = NewActiveASTContext;
  }

  // The following Visit*() and Traverse*() functions "override"
  // methods in RecursiveASTVisitor.

  bool VisitTypedefNameDecl(TypedefNameDecl *DeclNode) {
    // When we see 'typedef A B', we add name 'B' to the set of names
    // A's canonical type maps to.  This is necessary for implementing
    // isDerivedFrom(x) properly, where x can be the name of the base
    // class or any of its aliases.
    //
    // In general, the is-alias-of (as defined by typedefs) relation
    // is tree-shaped, as you can typedef a type more than once.  For
    // example,
    //
    //   typedef A B;
    //   typedef A C;
    //   typedef C D;
    //   typedef C E;
    //
    // gives you
    //
    //   A
    //   |- B
    //   `- C
    //      |- D
    //      `- E
    //
    // It is wrong to assume that the relation is a chain.  A correct
    // implementation of isDerivedFrom() needs to recognize that B and
    // E are aliases, even though neither is a typedef of the other.
    // Therefore, we cannot simply walk through one typedef chain to
    // find out whether the type name matches.
    const Type *TypeNode = DeclNode->getUnderlyingType().getTypePtr();
    const Type *CanonicalType =  // root of the typedef tree
        ActiveASTContext->getCanonicalType(TypeNode);
    TypeAliases[CanonicalType].insert(DeclNode);
    return true;
  }

  bool VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *CAD) {
    const ObjCInterfaceDecl *InterfaceDecl = CAD->getClassInterface();
    CompatibleAliases[InterfaceDecl].insert(CAD);
    return true;
  }

  bool TraverseDecl(Decl *DeclNode);
  bool TraverseStmt(Stmt *StmtNode, DataRecursionQueue *Queue = nullptr);
  bool TraverseType(QualType TypeNode);
  bool TraverseTypeLoc(TypeLoc TypeNode);
  bool TraverseNestedNameSpecifier(NestedNameSpecifier *NNS);
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS);
  bool TraverseConstructorInitializer(CXXCtorInitializer *CtorInit);
  bool TraverseTemplateArgumentLoc(TemplateArgumentLoc TAL);

  // Matches children or descendants of 'Node' with 'BaseMatcher'.
  bool memoizedMatchesRecursively(const DynTypedNode &Node, ASTContext &Ctx,
                                  const DynTypedMatcher &Matcher,
                                  BoundNodesTreeBuilder *Builder, int MaxDepth,
                                  TraversalKind Traversal, BindKind Bind) {
    // For AST-nodes that don't have an identity, we can't memoize.
    if (!Node.getMemoizationData() || !Builder->isComparable())
      return matchesRecursively(Node, Matcher, Builder, MaxDepth, Traversal,
                                Bind);

    MatchKey Key;
    Key.MatcherID = Matcher.getID();
    Key.Node = Node;
    // Note that we key on the bindings *before* the match.
    Key.BoundNodes = *Builder;
    Key.Traversal = Ctx.getParentMapContext().getTraversalKind();
    // Memoize result even doing a single-level match, it might be expensive.
    Key.Type = MaxDepth == 1 ? MatchType::Child : MatchType::Descendants;
    MemoizationMap::iterator I = ResultCache.find(Key);
    if (I != ResultCache.end()) {
      *Builder = I->second.Nodes;
      return I->second.ResultOfMatch;
    }

    MemoizedMatchResult Result;
    Result.Nodes = *Builder;
    Result.ResultOfMatch = matchesRecursively(Node, Matcher, &Result.Nodes,
                                              MaxDepth, Traversal, Bind);

    MemoizedMatchResult &CachedResult = ResultCache[Key];
    CachedResult = std::move(Result);

    *Builder = CachedResult.Nodes;
    return CachedResult.ResultOfMatch;
  }

  // Matches children or descendants of 'Node' with 'BaseMatcher'.
  bool matchesRecursively(const DynTypedNode &Node,
                          const DynTypedMatcher &Matcher,
                          BoundNodesTreeBuilder *Builder, int MaxDepth,
                          TraversalKind Traversal, BindKind Bind) {
    MatchChildASTVisitor Visitor(
      &Matcher, this, Builder, MaxDepth, Traversal, Bind);
    return Visitor.findMatch(Node);
  }

  bool classIsDerivedFrom(const CXXRecordDecl *Declaration,
                          const Matcher<NamedDecl> &Base,
                          BoundNodesTreeBuilder *Builder,
                          bool Directly) override;

  bool objcClassIsDerivedFrom(const ObjCInterfaceDecl *Declaration,
                              const Matcher<NamedDecl> &Base,
                              BoundNodesTreeBuilder *Builder,
                              bool Directly) override;

  // Implements ASTMatchFinder::matchesChildOf.
  bool matchesChildOf(const DynTypedNode &Node, ASTContext &Ctx,
                      const DynTypedMatcher &Matcher,
                      BoundNodesTreeBuilder *Builder, TraversalKind Traversal,
                      BindKind Bind) override {
    if (ResultCache.size() > MaxMemoizationEntries)
      ResultCache.clear();
    return memoizedMatchesRecursively(Node, Ctx, Matcher, Builder, 1, Traversal,
                                      Bind);
  }
  // Implements ASTMatchFinder::matchesDescendantOf.
  bool matchesDescendantOf(const DynTypedNode &Node, ASTContext &Ctx,
                           const DynTypedMatcher &Matcher,
                           BoundNodesTreeBuilder *Builder,
                           BindKind Bind) override {
    if (ResultCache.size() > MaxMemoizationEntries)
      ResultCache.clear();
    return memoizedMatchesRecursively(Node, Ctx, Matcher, Builder, INT_MAX,
                                      TraversalKind::TK_AsIs, Bind);
  }
  // Implements ASTMatchFinder::matchesAncestorOf.
  bool matchesAncestorOf(const DynTypedNode &Node, ASTContext &Ctx,
                         const DynTypedMatcher &Matcher,
                         BoundNodesTreeBuilder *Builder,
                         AncestorMatchMode MatchMode) override {
    // Reset the cache outside of the recursive call to make sure we
    // don't invalidate any iterators.
    if (ResultCache.size() > MaxMemoizationEntries)
      ResultCache.clear();
    return memoizedMatchesAncestorOfRecursively(Node, Ctx, Matcher, Builder,
                                                MatchMode);
  }

  // Matches all registered matchers on the given node and calls the
  // result callback for every node that matches.
  void match(const DynTypedNode &Node) {
    // FIXME: Improve this with a switch or a visitor pattern.
    if (auto *N = Node.get<Decl>()) {
      match(*N);
    } else if (auto *N = Node.get<Stmt>()) {
      match(*N);
    } else if (auto *N = Node.get<Type>()) {
      match(*N);
    } else if (auto *N = Node.get<QualType>()) {
      match(*N);
    } else if (auto *N = Node.get<NestedNameSpecifier>()) {
      match(*N);
    } else if (auto *N = Node.get<NestedNameSpecifierLoc>()) {
      match(*N);
    } else if (auto *N = Node.get<TypeLoc>()) {
      match(*N);
    } else if (auto *N = Node.get<CXXCtorInitializer>()) {
      match(*N);
    } else if (auto *N = Node.get<TemplateArgumentLoc>()) {
      match(*N);
    }
  }

  template <typename T> void match(const T &Node) {
    matchDispatch(&Node);
  }

  // Implements ASTMatchFinder::getASTContext.
  ASTContext &getASTContext() const override { return *ActiveASTContext; }

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }

private:
  class TimeBucketRegion {
  public:
    TimeBucketRegion() : Bucket(nullptr) {}
    ~TimeBucketRegion() { setBucket(nullptr); }

    /// Start timing for \p NewBucket.
    ///
    /// If there was a bucket already set, it will finish the timing for that
    /// other bucket.
    /// \p NewBucket will be timed until the next call to \c setBucket() or
    /// until the \c TimeBucketRegion is destroyed.
    /// If \p NewBucket is the same as the currently timed bucket, this call
    /// does nothing.
    void setBucket(llvm::TimeRecord *NewBucket) {
      if (Bucket != NewBucket) {
        auto Now = llvm::TimeRecord::getCurrentTime(true);
        if (Bucket)
          *Bucket += Now;
        if (NewBucket)
          *NewBucket -= Now;
        Bucket = NewBucket;
      }
    }

  private:
    llvm::TimeRecord *Bucket;
  };

  /// Runs all the \p Matchers on \p Node.
  ///
  /// Used by \c matchDispatch() below.
  template <typename T, typename MC>
  void matchWithoutFilter(const T &Node, const MC &Matchers) {
    const bool EnableCheckProfiling = Options.CheckProfiling.hasValue();
    TimeBucketRegion Timer;
    for (const auto &MP : Matchers) {
      if (EnableCheckProfiling)
        Timer.setBucket(&TimeByBucket[MP.second->getID()]);
      BoundNodesTreeBuilder Builder;
      if (MP.first.matches(Node, this, &Builder)) {
        MatchVisitor Visitor(ActiveASTContext, MP.second);
        Builder.visitMatches(&Visitor);
      }
    }
  }

  void matchWithFilter(const DynTypedNode &DynNode) {
    auto Kind = DynNode.getNodeKind();
    auto it = MatcherFiltersMap.find(Kind);
    const auto &Filter =
        it != MatcherFiltersMap.end() ? it->second : getFilterForKind(Kind);

    if (Filter.empty())
      return;

    const bool EnableCheckProfiling = Options.CheckProfiling.hasValue();
    TimeBucketRegion Timer;
    auto &Matchers = this->Matchers->DeclOrStmt;
    for (unsigned short I : Filter) {
      auto &MP = Matchers[I];
      if (EnableCheckProfiling)
        Timer.setBucket(&TimeByBucket[MP.second->getID()]);
      BoundNodesTreeBuilder Builder;
      if (MP.first.matches(DynNode, this, &Builder)) {
        MatchVisitor Visitor(ActiveASTContext, MP.second);
        Builder.visitMatches(&Visitor);
      }
    }
  }

  const std::vector<unsigned short> &getFilterForKind(ASTNodeKind Kind) {
    auto &Filter = MatcherFiltersMap[Kind];
    auto &Matchers = this->Matchers->DeclOrStmt;
    assert((Matchers.size() < USHRT_MAX) && "Too many matchers.");
    for (unsigned I = 0, E = Matchers.size(); I != E; ++I) {
      if (Matchers[I].first.canMatchNodesOfKind(Kind)) {
        Filter.push_back(I);
      }
    }
    return Filter;
  }

  /// @{
  /// Overloads to pair the different node types to their matchers.
  void matchDispatch(const Decl *Node) {
    return matchWithFilter(DynTypedNode::create(*Node));
  }
  void matchDispatch(const Stmt *Node) {
    return matchWithFilter(DynTypedNode::create(*Node));
  }

  void matchDispatch(const Type *Node) {
    matchWithoutFilter(QualType(Node, 0), Matchers->Type);
  }
  void matchDispatch(const TypeLoc *Node) {
    matchWithoutFilter(*Node, Matchers->TypeLoc);
  }
  void matchDispatch(const QualType *Node) {
    matchWithoutFilter(*Node, Matchers->Type);
  }
  void matchDispatch(const NestedNameSpecifier *Node) {
    matchWithoutFilter(*Node, Matchers->NestedNameSpecifier);
  }
  void matchDispatch(const NestedNameSpecifierLoc *Node) {
    matchWithoutFilter(*Node, Matchers->NestedNameSpecifierLoc);
  }
  void matchDispatch(const CXXCtorInitializer *Node) {
    matchWithoutFilter(*Node, Matchers->CtorInit);
  }
  void matchDispatch(const TemplateArgumentLoc *Node) {
    matchWithoutFilter(*Node, Matchers->TemplateArgumentLoc);
  }
  void matchDispatch(const void *) { /* Do nothing. */ }
  /// @}

  // Returns whether an ancestor of \p Node matches \p Matcher.
  //
  // The order of matching ((which can lead to different nodes being bound in
  // case there are multiple matches) is breadth first search.
  //
  // To allow memoization in the very common case of having deeply nested
  // expressions inside a template function, we first walk up the AST, memoizing
  // the result of the match along the way, as long as there is only a single
  // parent.
  //
  // Once there are multiple parents, the breadth first search order does not
  // allow simple memoization on the ancestors. Thus, we only memoize as long
  // as there is a single parent.
  bool memoizedMatchesAncestorOfRecursively(const DynTypedNode &Node,
                                            ASTContext &Ctx,
                                            const DynTypedMatcher &Matcher,
                                            BoundNodesTreeBuilder *Builder,
                                            AncestorMatchMode MatchMode) {
    // For AST-nodes that don't have an identity, we can't memoize.
    // When doing a single-level match, we don't need to memoize because
    // ParentMap (in ASTContext) already memoizes the result.
    if (!Builder->isComparable() ||
        MatchMode == AncestorMatchMode::AMM_ParentOnly)
      return matchesAncestorOfRecursively(Node, Ctx, Matcher, Builder,
                                          MatchMode);

    MatchKey Key;
    Key.MatcherID = Matcher.getID();
    Key.Node = Node;
    Key.BoundNodes = *Builder;
    Key.Traversal = Ctx.getParentMapContext().getTraversalKind();
    Key.Type = MatchType::Ancestors;

    // Note that we cannot use insert and reuse the iterator, as recursive
    // calls to match might invalidate the result cache iterators.
    MemoizationMap::iterator I = ResultCache.find(Key);
    if (I != ResultCache.end()) {
      *Builder = I->second.Nodes;
      return I->second.ResultOfMatch;
    }

    MemoizedMatchResult Result;
    Result.Nodes = *Builder;
    Result.ResultOfMatch = matchesAncestorOfRecursively(
        Node, Ctx, Matcher, &Result.Nodes, MatchMode);

    MemoizedMatchResult &CachedResult = ResultCache[Key];
    CachedResult = std::move(Result);

    *Builder = CachedResult.Nodes;
    return CachedResult.ResultOfMatch;
  }

  bool matchesAncestorOfRecursively(const DynTypedNode &Node, ASTContext &Ctx,
                                    const DynTypedMatcher &Matcher,
                                    BoundNodesTreeBuilder *Builder,
                                    AncestorMatchMode MatchMode) {
    const auto &Parents = ActiveASTContext->getParents(Node);
    if (Parents.empty()) {
      // Nodes may have no parents if:
      //  a) the node is the TranslationUnitDecl
      //  b) we have a limited traversal scope that excludes the parent edges
      //  c) there is a bug in the AST, and the node is not reachable
      // Usually the traversal scope is the whole AST, which precludes b.
      // Bugs are common enough that it's worthwhile asserting when we can.
#ifndef NDEBUG
      if (!Node.get<TranslationUnitDecl>() &&
          /* Traversal scope is full AST if any of the bounds are the TU */
          llvm::any_of(ActiveASTContext->getTraversalScope(), [](Decl *D) {
            return D->getKind() == Decl::TranslationUnit;
          })) {
        llvm::errs() << "Tried to match orphan node:\n";
        Node.dump(llvm::errs(), *ActiveASTContext);
        llvm_unreachable("Parent map should be complete!");
      }
#endif
      return false;
    }
    if (Parents.size() == 1) {
      // Only one parent - do recursive memoization.
      const DynTypedNode Parent = Parents[0];
      BoundNodesTreeBuilder BuilderCopy = *Builder;
      if (Matcher.matches(Parent, this, &BuilderCopy)) {
        *Builder = std::move(BuilderCopy);
        return true;
      }
      if (MatchMode != ASTMatchFinder::AMM_ParentOnly) {
        return memoizedMatchesAncestorOfRecursively(Parent, Ctx, Matcher,
                                                    Builder, MatchMode);
        // Once we get back from the recursive call, the result will be the
        // same as the parent's result.
      }
    } else {
      // Multiple parents - BFS over the rest of the nodes.
      llvm::DenseSet<const void *> Visited;
      std::deque<DynTypedNode> Queue(Parents.begin(), Parents.end());
      while (!Queue.empty()) {
        BoundNodesTreeBuilder BuilderCopy = *Builder;
        if (Matcher.matches(Queue.front(), this, &BuilderCopy)) {
          *Builder = std::move(BuilderCopy);
          return true;
        }
        if (MatchMode != ASTMatchFinder::AMM_ParentOnly) {
          for (const auto &Parent :
               ActiveASTContext->getParents(Queue.front())) {
            // Make sure we do not visit the same node twice.
            // Otherwise, we'll visit the common ancestors as often as there
            // are splits on the way down.
            if (Visited.insert(Parent.getMemoizationData()).second)
              Queue.push_back(Parent);
          }
        }
        Queue.pop_front();
      }
    }
    return false;
  }

  // Implements a BoundNodesTree::Visitor that calls a MatchCallback with
  // the aggregated bound nodes for each match.
  class MatchVisitor : public BoundNodesTreeBuilder::Visitor {
  public:
    MatchVisitor(ASTContext* Context,
                 MatchFinder::MatchCallback* Callback)
      : Context(Context),
        Callback(Callback) {}

    void visitMatch(const BoundNodes& BoundNodesView) override {
      Callback->run(MatchFinder::MatchResult(BoundNodesView, Context));
    }

  private:
    ASTContext* Context;
    MatchFinder::MatchCallback* Callback;
  };

  // Returns true if 'TypeNode' has an alias that matches the given matcher.
  bool typeHasMatchingAlias(const Type *TypeNode,
                            const Matcher<NamedDecl> &Matcher,
                            BoundNodesTreeBuilder *Builder) {
    const Type *const CanonicalType =
      ActiveASTContext->getCanonicalType(TypeNode);
    auto Aliases = TypeAliases.find(CanonicalType);
    if (Aliases == TypeAliases.end())
      return false;
    for (const TypedefNameDecl *Alias : Aliases->second) {
      BoundNodesTreeBuilder Result(*Builder);
      if (Matcher.matches(*Alias, this, &Result)) {
        *Builder = std::move(Result);
        return true;
      }
    }
    return false;
  }

  bool
  objcClassHasMatchingCompatibilityAlias(const ObjCInterfaceDecl *InterfaceDecl,
                                         const Matcher<NamedDecl> &Matcher,
                                         BoundNodesTreeBuilder *Builder) {
    auto Aliases = CompatibleAliases.find(InterfaceDecl);
    if (Aliases == CompatibleAliases.end())
      return false;
    for (const ObjCCompatibleAliasDecl *Alias : Aliases->second) {
      BoundNodesTreeBuilder Result(*Builder);
      if (Matcher.matches(*Alias, this, &Result)) {
        *Builder = std::move(Result);
        return true;
      }
    }
    return false;
  }

  /// Bucket to record map.
  ///
  /// Used to get the appropriate bucket for each matcher.
  llvm::StringMap<llvm::TimeRecord> TimeByBucket;

  const MatchFinder::MatchersByType *Matchers;

  /// Filtered list of matcher indices for each matcher kind.
  ///
  /// \c Decl and \c Stmt toplevel matchers usually apply to a specific node
  /// kind (and derived kinds) so it is a waste to try every matcher on every
  /// node.
  /// We precalculate a list of matchers that pass the toplevel restrict check.
  llvm::DenseMap<ASTNodeKind, std::vector<unsigned short>> MatcherFiltersMap;

  const MatchFinder::MatchFinderOptions &Options;
  ASTContext *ActiveASTContext;

  // Maps a canonical type to its TypedefDecls.
  llvm::DenseMap<const Type*, std::set<const TypedefNameDecl*> > TypeAliases;

  // Maps an Objective-C interface to its ObjCCompatibleAliasDecls.
  llvm::DenseMap<const ObjCInterfaceDecl *,
                 llvm::SmallPtrSet<const ObjCCompatibleAliasDecl *, 2>>
      CompatibleAliases;

  // Maps (matcher, node) -> the match result for memoization.
  typedef std::map<MatchKey, MemoizedMatchResult> MemoizationMap;
  MemoizationMap ResultCache;
};

static CXXRecordDecl *
getAsCXXRecordDeclOrPrimaryTemplate(const Type *TypeNode) {
  if (auto *RD = TypeNode->getAsCXXRecordDecl())
    return RD;

  // Find the innermost TemplateSpecializationType that isn't an alias template.
  auto *TemplateType = TypeNode->getAs<TemplateSpecializationType>();
  while (TemplateType && TemplateType->isTypeAlias())
    TemplateType =
        TemplateType->getAliasedType()->getAs<TemplateSpecializationType>();

  // If this is the name of a (dependent) template specialization, use the
  // definition of the template, even though it might be specialized later.
  if (TemplateType)
    if (auto *ClassTemplate = dyn_cast_or_null<ClassTemplateDecl>(
          TemplateType->getTemplateName().getAsTemplateDecl()))
      return ClassTemplate->getTemplatedDecl();

  return nullptr;
}

// Returns true if the given C++ class is directly or indirectly derived
// from a base type with the given name.  A class is not considered to be
// derived from itself.
bool MatchASTVisitor::classIsDerivedFrom(const CXXRecordDecl *Declaration,
                                         const Matcher<NamedDecl> &Base,
                                         BoundNodesTreeBuilder *Builder,
                                         bool Directly) {
  if (!Declaration->hasDefinition())
    return false;
  for (const auto &It : Declaration->bases()) {
    const Type *TypeNode = It.getType().getTypePtr();

    if (typeHasMatchingAlias(TypeNode, Base, Builder))
      return true;

    // FIXME: Going to the primary template here isn't really correct, but
    // unfortunately we accept a Decl matcher for the base class not a Type
    // matcher, so it's the best thing we can do with our current interface.
    CXXRecordDecl *ClassDecl = getAsCXXRecordDeclOrPrimaryTemplate(TypeNode);
    if (!ClassDecl)
      continue;
    if (ClassDecl == Declaration) {
      // This can happen for recursive template definitions.
      continue;
    }
    BoundNodesTreeBuilder Result(*Builder);
    if (Base.matches(*ClassDecl, this, &Result)) {
      *Builder = std::move(Result);
      return true;
    }
    if (!Directly && classIsDerivedFrom(ClassDecl, Base, Builder, Directly))
      return true;
  }
  return false;
}

// Returns true if the given Objective-C class is directly or indirectly
// derived from a matching base class. A class is not considered to be derived
// from itself.
bool MatchASTVisitor::objcClassIsDerivedFrom(
    const ObjCInterfaceDecl *Declaration, const Matcher<NamedDecl> &Base,
    BoundNodesTreeBuilder *Builder, bool Directly) {
  // Check if any of the superclasses of the class match.
  for (const ObjCInterfaceDecl *ClassDecl = Declaration->getSuperClass();
       ClassDecl != nullptr; ClassDecl = ClassDecl->getSuperClass()) {
    // Check if there are any matching compatibility aliases.
    if (objcClassHasMatchingCompatibilityAlias(ClassDecl, Base, Builder))
      return true;

    // Check if there are any matching type aliases.
    const Type *TypeNode = ClassDecl->getTypeForDecl();
    if (typeHasMatchingAlias(TypeNode, Base, Builder))
      return true;

    if (Base.matches(*ClassDecl, this, Builder))
      return true;

    // Not `return false` as a temporary workaround for PR43879.
    if (Directly)
      break;
  }

  return false;
}

bool MatchASTVisitor::TraverseDecl(Decl *DeclNode) {
  if (!DeclNode) {
    return true;
  }
  match(*DeclNode);
  return RecursiveASTVisitor<MatchASTVisitor>::TraverseDecl(DeclNode);
}

bool MatchASTVisitor::TraverseStmt(Stmt *StmtNode, DataRecursionQueue *Queue) {
  if (!StmtNode) {
    return true;
  }
  match(*StmtNode);
  return RecursiveASTVisitor<MatchASTVisitor>::TraverseStmt(StmtNode, Queue);
}

bool MatchASTVisitor::TraverseType(QualType TypeNode) {
  match(TypeNode);
  return RecursiveASTVisitor<MatchASTVisitor>::TraverseType(TypeNode);
}

bool MatchASTVisitor::TraverseTypeLoc(TypeLoc TypeLocNode) {
  // The RecursiveASTVisitor only visits types if they're not within TypeLocs.
  // We still want to find those types via matchers, so we match them here. Note
  // that the TypeLocs are structurally a shadow-hierarchy to the expressed
  // type, so we visit all involved parts of a compound type when matching on
  // each TypeLoc.
  match(TypeLocNode);
  match(TypeLocNode.getType());
  return RecursiveASTVisitor<MatchASTVisitor>::TraverseTypeLoc(TypeLocNode);
}

bool MatchASTVisitor::TraverseNestedNameSpecifier(NestedNameSpecifier *NNS) {
  match(*NNS);
  return RecursiveASTVisitor<MatchASTVisitor>::TraverseNestedNameSpecifier(NNS);
}

bool MatchASTVisitor::TraverseNestedNameSpecifierLoc(
    NestedNameSpecifierLoc NNS) {
  if (!NNS)
    return true;

  match(NNS);

  // We only match the nested name specifier here (as opposed to traversing it)
  // because the traversal is already done in the parallel "Loc"-hierarchy.
  if (NNS.hasQualifier())
    match(*NNS.getNestedNameSpecifier());
  return
      RecursiveASTVisitor<MatchASTVisitor>::TraverseNestedNameSpecifierLoc(NNS);
}

bool MatchASTVisitor::TraverseConstructorInitializer(
    CXXCtorInitializer *CtorInit) {
  if (!CtorInit)
    return true;

  match(*CtorInit);

  return RecursiveASTVisitor<MatchASTVisitor>::TraverseConstructorInitializer(
      CtorInit);
}

bool MatchASTVisitor::TraverseTemplateArgumentLoc(TemplateArgumentLoc Loc) {
  match(Loc);
  return RecursiveASTVisitor<MatchASTVisitor>::TraverseTemplateArgumentLoc(Loc);
}

class MatchASTConsumer : public ASTConsumer {
public:
  MatchASTConsumer(MatchFinder *Finder,
                   MatchFinder::ParsingDoneTestCallback *ParsingDone)
      : Finder(Finder), ParsingDone(ParsingDone) {}

private:
  void HandleTranslationUnit(ASTContext &Context) override {
    if (ParsingDone != nullptr) {
      ParsingDone->run();
    }
    Finder->matchAST(Context);
  }

  MatchFinder *Finder;
  MatchFinder::ParsingDoneTestCallback *ParsingDone;
};

} // end namespace
} // end namespace internal

MatchFinder::MatchResult::MatchResult(const BoundNodes &Nodes,
                                      ASTContext *Context)
  : Nodes(Nodes), Context(Context),
    SourceManager(&Context->getSourceManager()) {}

MatchFinder::MatchCallback::~MatchCallback() {}
MatchFinder::ParsingDoneTestCallback::~ParsingDoneTestCallback() {}

MatchFinder::MatchFinder(MatchFinderOptions Options)
    : Options(std::move(Options)), ParsingDone(nullptr) {}

MatchFinder::~MatchFinder() {}

void MatchFinder::addMatcher(const DeclarationMatcher &NodeMatch,
                             MatchCallback *Action) {
  Matchers.DeclOrStmt.emplace_back(NodeMatch, Action);
  Matchers.AllCallbacks.insert(Action);
}

void MatchFinder::addMatcher(const TypeMatcher &NodeMatch,
                             MatchCallback *Action) {
  Matchers.Type.emplace_back(NodeMatch, Action);
  Matchers.AllCallbacks.insert(Action);
}

void MatchFinder::addMatcher(const StatementMatcher &NodeMatch,
                             MatchCallback *Action) {
  Matchers.DeclOrStmt.emplace_back(NodeMatch, Action);
  Matchers.AllCallbacks.insert(Action);
}

void MatchFinder::addMatcher(const NestedNameSpecifierMatcher &NodeMatch,
                             MatchCallback *Action) {
  Matchers.NestedNameSpecifier.emplace_back(NodeMatch, Action);
  Matchers.AllCallbacks.insert(Action);
}

void MatchFinder::addMatcher(const NestedNameSpecifierLocMatcher &NodeMatch,
                             MatchCallback *Action) {
  Matchers.NestedNameSpecifierLoc.emplace_back(NodeMatch, Action);
  Matchers.AllCallbacks.insert(Action);
}

void MatchFinder::addMatcher(const TypeLocMatcher &NodeMatch,
                             MatchCallback *Action) {
  Matchers.TypeLoc.emplace_back(NodeMatch, Action);
  Matchers.AllCallbacks.insert(Action);
}

void MatchFinder::addMatcher(const CXXCtorInitializerMatcher &NodeMatch,
                             MatchCallback *Action) {
  Matchers.CtorInit.emplace_back(NodeMatch, Action);
  Matchers.AllCallbacks.insert(Action);
}

void MatchFinder::addMatcher(const TemplateArgumentLocMatcher &NodeMatch,
                             MatchCallback *Action) {
  Matchers.TemplateArgumentLoc.emplace_back(NodeMatch, Action);
  Matchers.AllCallbacks.insert(Action);
}

bool MatchFinder::addDynamicMatcher(const internal::DynTypedMatcher &NodeMatch,
                                    MatchCallback *Action) {
  if (NodeMatch.canConvertTo<Decl>()) {
    addMatcher(NodeMatch.convertTo<Decl>(), Action);
    return true;
  } else if (NodeMatch.canConvertTo<QualType>()) {
    addMatcher(NodeMatch.convertTo<QualType>(), Action);
    return true;
  } else if (NodeMatch.canConvertTo<Stmt>()) {
    addMatcher(NodeMatch.convertTo<Stmt>(), Action);
    return true;
  } else if (NodeMatch.canConvertTo<NestedNameSpecifier>()) {
    addMatcher(NodeMatch.convertTo<NestedNameSpecifier>(), Action);
    return true;
  } else if (NodeMatch.canConvertTo<NestedNameSpecifierLoc>()) {
    addMatcher(NodeMatch.convertTo<NestedNameSpecifierLoc>(), Action);
    return true;
  } else if (NodeMatch.canConvertTo<TypeLoc>()) {
    addMatcher(NodeMatch.convertTo<TypeLoc>(), Action);
    return true;
  } else if (NodeMatch.canConvertTo<CXXCtorInitializer>()) {
    addMatcher(NodeMatch.convertTo<CXXCtorInitializer>(), Action);
    return true;
  } else if (NodeMatch.canConvertTo<TemplateArgumentLoc>()) {
    addMatcher(NodeMatch.convertTo<TemplateArgumentLoc>(), Action);
    return true;
  }
  return false;
}

std::unique_ptr<ASTConsumer> MatchFinder::newASTConsumer() {
  return std::make_unique<internal::MatchASTConsumer>(this, ParsingDone);
}

void MatchFinder::match(const clang::DynTypedNode &Node, ASTContext &Context) {
  internal::MatchASTVisitor Visitor(&Matchers, Options);
  Visitor.set_active_ast_context(&Context);
  Visitor.match(Node);
}

void MatchFinder::matchAST(ASTContext &Context) {
  internal::MatchASTVisitor Visitor(&Matchers, Options);
  Visitor.set_active_ast_context(&Context);
  Visitor.onStartOfTranslationUnit();
  Visitor.TraverseAST(Context);
  Visitor.onEndOfTranslationUnit();
}

void MatchFinder::registerTestCallbackAfterParsing(
    MatchFinder::ParsingDoneTestCallback *NewParsingDone) {
  ParsingDone = NewParsingDone;
}

StringRef MatchFinder::MatchCallback::getID() const { return "<unknown>"; }

} // end namespace ast_matchers
} // end namespace clang
