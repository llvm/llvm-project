//===--- ASTMatchFinder.cpp - Structural query framework ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include <set>

namespace clang {
namespace ast_matchers {
namespace internal {
namespace {

typedef MatchFinder::MatchCallback MatchCallback;

/// \brief A \c RecursiveASTVisitor that builds a map from nodes to their
/// parents as defined by the \c RecursiveASTVisitor.
///
/// Note that the relationship described here is purely in terms of AST
/// traversal - there are other relationships (for example declaration context)
/// in the AST that are better modeled by special matchers.
///
/// FIXME: Currently only builds up the map using \c Stmt and \c Decl nodes.
class ParentMapASTVisitor : public RecursiveASTVisitor<ParentMapASTVisitor> {
public:
  /// \brief Maps from a node to its parent.
  typedef llvm::DenseMap<const void*, ast_type_traits::DynTypedNode> ParentMap;

  /// \brief Builds and returns the translation unit's parent map.
  ///
  ///  The caller takes ownership of the returned \c ParentMap.
  static ParentMap *buildMap(TranslationUnitDecl &TU) {
    ParentMapASTVisitor Visitor(new ParentMap);
    Visitor.TraverseDecl(&TU);
    return Visitor.Parents;
  }

private:
  typedef RecursiveASTVisitor<ParentMapASTVisitor> VisitorBase;

  ParentMapASTVisitor(ParentMap *Parents) : Parents(Parents) {}

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }

  template <typename T>
  bool TraverseNode(T *Node, bool (VisitorBase::*traverse)(T*)) {
    if (Node == NULL)
      return true;
    if (ParentStack.size() > 0)
      (*Parents)[Node] = ParentStack.back();
    ParentStack.push_back(ast_type_traits::DynTypedNode::create(*Node));
    bool Result = (this->*traverse)(Node);
    ParentStack.pop_back();
    return Result;
  }

  bool TraverseDecl(Decl *DeclNode) {
    return TraverseNode(DeclNode, &VisitorBase::TraverseDecl);
  }

  bool TraverseStmt(Stmt *StmtNode) {
    return TraverseNode(StmtNode, &VisitorBase::TraverseStmt);
  }

  ParentMap *Parents;
  llvm::SmallVector<ast_type_traits::DynTypedNode, 16> ParentStack;

  friend class RecursiveASTVisitor<ParentMapASTVisitor>;
};

// We use memoization to avoid running the same matcher on the same
// AST node twice.  This pair is the key for looking up match
// result.  It consists of an ID of the MatcherInterface (for
// identifying the matcher) and a pointer to the AST node.
//
// We currently only memoize on nodes whose pointers identify the
// nodes (\c Stmt and \c Decl, but not \c QualType or \c TypeLoc).
// For \c QualType and \c TypeLoc it is possible to implement
// generation of keys for each type.
// FIXME: Benchmark whether memoization of non-pointer typed nodes
// provides enough benefit for the additional amount of code.
typedef std::pair<uint64_t, const void*> UntypedMatchInput;

// Used to store the result of a match and possibly bound nodes.
struct MemoizedMatchResult {
  bool ResultOfMatch;
  BoundNodesTree Nodes;
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
  MatchChildASTVisitor(const DynTypedMatcher *Matcher,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder,
                       int MaxDepth,
                       ASTMatchFinder::TraversalKind Traversal,
                       ASTMatchFinder::BindKind Bind)
      : Matcher(Matcher),
        Finder(Finder),
        Builder(Builder),
        CurrentDepth(0),
        MaxDepth(MaxDepth),
        Traversal(Traversal),
        Bind(Bind),
        Matches(false) {}

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
  bool findMatch(const ast_type_traits::DynTypedNode &DynNode) {
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
    // FIXME: Add other base types after adding tests.
    return Matches;
  }

  // The following are overriding methods from the base visitor class.
  // They are public only to allow CRTP to work. They are *not *part
  // of the public API of this class.
  bool TraverseDecl(Decl *DeclNode) {
    ScopedIncrement ScopedDepth(&CurrentDepth);
    return (DeclNode == NULL) || traverse(*DeclNode);
  }
  bool TraverseStmt(Stmt *StmtNode) {
    ScopedIncrement ScopedDepth(&CurrentDepth);
    const Stmt *StmtToTraverse = StmtNode;
    if (Traversal ==
        ASTMatchFinder::TK_IgnoreImplicitCastsAndParentheses) {
      const Expr *ExprNode = dyn_cast_or_null<Expr>(StmtNode);
      if (ExprNode != NULL) {
        StmtToTraverse = ExprNode->IgnoreParenImpCasts();
      }
    }
    return (StmtToTraverse == NULL) || traverse(*StmtToTraverse);
  }
  // We assume that the QualType and the contained type are on the same
  // hierarchy level. Thus, we try to match either of them.
  bool TraverseType(QualType TypeNode) {
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
    return (NNS == NULL) || traverse(*NNS);
  }
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS) {
    ScopedIncrement ScopedDepth(&CurrentDepth);
    if (!match(*NNS.getNestedNameSpecifier()))
      return false;
    return !NNS || traverse(NNS);
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
      if (Matcher->matches(ast_type_traits::DynTypedNode::create(Node),
                           Finder, Builder)) {
        Matches = true;
        return false;  // Abort as soon as a match is found.
      }
    } else {
      BoundNodesTreeBuilder RecursiveBuilder;
      if (Matcher->matches(ast_type_traits::DynTypedNode::create(Node),
                           Finder, &RecursiveBuilder)) {
        // After the first match the matcher succeeds.
        Matches = true;
        Builder->addMatch(RecursiveBuilder.build());
      }
    }
    return true;
  }

  // Traverses the subtree rooted at 'Node'; returns true if the
  // traversal should continue after this function returns.
  template <typename T>
  bool traverse(const T &Node) {
    TOOLING_COMPILE_ASSERT(IsBaseType<T>::value,
                           traverse_can_only_be_instantiated_with_base_type);
    if (!match(Node))
      return false;
    return baseTraverse(Node);
  }

  const DynTypedMatcher *const Matcher;
  ASTMatchFinder *const Finder;
  BoundNodesTreeBuilder *const Builder;
  int CurrentDepth;
  const int MaxDepth;
  const ASTMatchFinder::TraversalKind Traversal;
  const ASTMatchFinder::BindKind Bind;
  bool Matches;
};

// Controls the outermost traversal of the AST and allows to match multiple
// matchers.
class MatchASTVisitor : public RecursiveASTVisitor<MatchASTVisitor>,
                        public ASTMatchFinder {
public:
  MatchASTVisitor(std::vector<std::pair<const internal::DynTypedMatcher*,
                                        MatchCallback*> > *MatcherCallbackPairs)
     : MatcherCallbackPairs(MatcherCallbackPairs),
       ActiveASTContext(NULL) {
  }

  void onStartOfTranslationUnit() {
    for (std::vector<std::pair<const internal::DynTypedMatcher*,
                               MatchCallback*> >::const_iterator
             I = MatcherCallbackPairs->begin(), E = MatcherCallbackPairs->end();
         I != E; ++I) {
      I->second->onStartOfTranslationUnit();
    }
  }

  void set_active_ast_context(ASTContext *NewActiveASTContext) {
    ActiveASTContext = NewActiveASTContext;
  }

  // The following Visit*() and Traverse*() functions "override"
  // methods in RecursiveASTVisitor.

  bool VisitTypedefDecl(TypedefDecl *DeclNode) {
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

  bool TraverseDecl(Decl *DeclNode);
  bool TraverseStmt(Stmt *StmtNode);
  bool TraverseType(QualType TypeNode);
  bool TraverseTypeLoc(TypeLoc TypeNode);
  bool TraverseNestedNameSpecifier(NestedNameSpecifier *NNS);
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS);

  // Matches children or descendants of 'Node' with 'BaseMatcher'.
  bool memoizedMatchesRecursively(const ast_type_traits::DynTypedNode &Node,
                                  const DynTypedMatcher &Matcher,
                                  BoundNodesTreeBuilder *Builder, int MaxDepth,
                                  TraversalKind Traversal, BindKind Bind) {
    const UntypedMatchInput input(Matcher.getID(), Node.getMemoizationData());

    // For AST-nodes that don't have an identity, we can't memoize.
    if (!input.second)
      return matchesRecursively(Node, Matcher, Builder, MaxDepth, Traversal,
                                Bind);

    std::pair<MemoizationMap::iterator, bool> InsertResult
      = ResultCache.insert(std::make_pair(input, MemoizedMatchResult()));
    if (InsertResult.second) {
      BoundNodesTreeBuilder DescendantBoundNodesBuilder;
      InsertResult.first->second.ResultOfMatch =
        matchesRecursively(Node, Matcher, &DescendantBoundNodesBuilder,
                           MaxDepth, Traversal, Bind);
      InsertResult.first->second.Nodes =
        DescendantBoundNodesBuilder.build();
    }
    InsertResult.first->second.Nodes.copyTo(Builder);
    return InsertResult.first->second.ResultOfMatch;
  }

  // Matches children or descendants of 'Node' with 'BaseMatcher'.
  bool matchesRecursively(const ast_type_traits::DynTypedNode &Node,
                          const DynTypedMatcher &Matcher,
                          BoundNodesTreeBuilder *Builder, int MaxDepth,
                          TraversalKind Traversal, BindKind Bind) {
    MatchChildASTVisitor Visitor(
      &Matcher, this, Builder, MaxDepth, Traversal, Bind);
    return Visitor.findMatch(Node);
  }

  virtual bool classIsDerivedFrom(const CXXRecordDecl *Declaration,
                                  const Matcher<NamedDecl> &Base,
                                  BoundNodesTreeBuilder *Builder);

  // Implements ASTMatchFinder::matchesChildOf.
  virtual bool matchesChildOf(const ast_type_traits::DynTypedNode &Node,
                              const DynTypedMatcher &Matcher,
                              BoundNodesTreeBuilder *Builder,
                              TraversalKind Traversal,
                              BindKind Bind) {
    return matchesRecursively(Node, Matcher, Builder, 1, Traversal,
                              Bind);
  }
  // Implements ASTMatchFinder::matchesDescendantOf.
  virtual bool matchesDescendantOf(const ast_type_traits::DynTypedNode &Node,
                                   const DynTypedMatcher &Matcher,
                                   BoundNodesTreeBuilder *Builder,
                                   BindKind Bind) {
    return memoizedMatchesRecursively(Node, Matcher, Builder, INT_MAX,
                                      TK_AsIs, Bind);
  }
  // Implements ASTMatchFinder::matchesAncestorOf.
  virtual bool matchesAncestorOf(const ast_type_traits::DynTypedNode &Node,
                                 const DynTypedMatcher &Matcher,
                                 BoundNodesTreeBuilder *Builder,
                                 AncestorMatchMode MatchMode) {
    if (!Parents) {
      // We always need to run over the whole translation unit, as
      // \c hasAncestor can escape any subtree.
      Parents.reset(ParentMapASTVisitor::buildMap(
        *ActiveASTContext->getTranslationUnitDecl()));
    }
    ast_type_traits::DynTypedNode Ancestor = Node;
    while (Ancestor.get<TranslationUnitDecl>() !=
           ActiveASTContext->getTranslationUnitDecl()) {
      assert(Ancestor.getMemoizationData() &&
             "Invariant broken: only nodes that support memoization may be "
             "used in the parent map.");
      ParentMapASTVisitor::ParentMap::const_iterator I =
        Parents->find(Ancestor.getMemoizationData());
      if (I == Parents->end()) {
        assert(false &&
               "Found node that is not in the parent map.");
        return false;
      }
      Ancestor = I->second;
      if (Matcher.matches(Ancestor, this, Builder))
        return true;
      if (MatchMode == ASTMatchFinder::AMM_ParentOnly)
        return false;
    }
    return false;
  }

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }

private:
  // Implements a BoundNodesTree::Visitor that calls a MatchCallback with
  // the aggregated bound nodes for each match.
  class MatchVisitor : public BoundNodesTree::Visitor {
  public:
    MatchVisitor(ASTContext* Context,
                 MatchFinder::MatchCallback* Callback)
      : Context(Context),
        Callback(Callback) {}

    virtual void visitMatch(const BoundNodes& BoundNodesView) {
      Callback->run(MatchFinder::MatchResult(BoundNodesView, Context));
    }

  private:
    ASTContext* Context;
    MatchFinder::MatchCallback* Callback;
  };

  // Returns true if 'TypeNode' has an alias that matches the given matcher.
  bool typeHasMatchingAlias(const Type *TypeNode,
                            const Matcher<NamedDecl> Matcher,
                            BoundNodesTreeBuilder *Builder) {
    const Type *const CanonicalType =
      ActiveASTContext->getCanonicalType(TypeNode);
    const std::set<const TypedefDecl*> &Aliases = TypeAliases[CanonicalType];
    for (std::set<const TypedefDecl*>::const_iterator
           It = Aliases.begin(), End = Aliases.end();
         It != End; ++It) {
      if (Matcher.matches(**It, this, Builder))
        return true;
    }
    return false;
  }

  // Matches all registered matchers on the given node and calls the
  // result callback for every node that matches.
  template <typename T>
  void match(const T &node) {
    for (std::vector<std::pair<const internal::DynTypedMatcher*,
                               MatchCallback*> >::const_iterator
             I = MatcherCallbackPairs->begin(), E = MatcherCallbackPairs->end();
         I != E; ++I) {
      BoundNodesTreeBuilder Builder;
      if (I->first->matches(ast_type_traits::DynTypedNode::create(node),
                            this, &Builder)) {
        BoundNodesTree BoundNodes = Builder.build();
        MatchVisitor Visitor(ActiveASTContext, I->second);
        BoundNodes.visitMatches(&Visitor);
      }
    }
  }

  std::vector<std::pair<const internal::DynTypedMatcher*,
                        MatchCallback*> > *const MatcherCallbackPairs;
  ASTContext *ActiveASTContext;

  // Maps a canonical type to its TypedefDecls.
  llvm::DenseMap<const Type*, std::set<const TypedefDecl*> > TypeAliases;

  // Maps (matcher, node) -> the match result for memoization.
  typedef llvm::DenseMap<UntypedMatchInput, MemoizedMatchResult> MemoizationMap;
  MemoizationMap ResultCache;

  llvm::OwningPtr<ParentMapASTVisitor::ParentMap> Parents;
};

// Returns true if the given class is directly or indirectly derived
// from a base type with the given name.  A class is not considered to be
// derived from itself.
bool MatchASTVisitor::classIsDerivedFrom(const CXXRecordDecl *Declaration,
                                         const Matcher<NamedDecl> &Base,
                                         BoundNodesTreeBuilder *Builder) {
  if (!Declaration->hasDefinition())
    return false;
  typedef CXXRecordDecl::base_class_const_iterator BaseIterator;
  for (BaseIterator It = Declaration->bases_begin(),
                    End = Declaration->bases_end(); It != End; ++It) {
    const Type *TypeNode = It->getType().getTypePtr();

    if (typeHasMatchingAlias(TypeNode, Base, Builder))
      return true;

    // Type::getAs<...>() drills through typedefs.
    if (TypeNode->getAs<DependentNameType>() != NULL ||
        TypeNode->getAs<DependentTemplateSpecializationType>() != NULL ||
        TypeNode->getAs<TemplateTypeParmType>() != NULL)
      // Dependent names and template TypeNode parameters will be matched when
      // the template is instantiated.
      continue;
    CXXRecordDecl *ClassDecl = NULL;
    TemplateSpecializationType const *TemplateType =
      TypeNode->getAs<TemplateSpecializationType>();
    if (TemplateType != NULL) {
      if (TemplateType->getTemplateName().isDependent())
        // Dependent template specializations will be matched when the
        // template is instantiated.
        continue;

      // For template specialization types which are specializing a template
      // declaration which is an explicit or partial specialization of another
      // template declaration, getAsCXXRecordDecl() returns the corresponding
      // ClassTemplateSpecializationDecl.
      //
      // For template specialization types which are specializing a template
      // declaration which is neither an explicit nor partial specialization of
      // another template declaration, getAsCXXRecordDecl() returns NULL and
      // we get the CXXRecordDecl of the templated declaration.
      CXXRecordDecl *SpecializationDecl =
        TemplateType->getAsCXXRecordDecl();
      if (SpecializationDecl != NULL) {
        ClassDecl = SpecializationDecl;
      } else {
        ClassDecl = llvm::dyn_cast<CXXRecordDecl>(
            TemplateType->getTemplateName()
                .getAsTemplateDecl()->getTemplatedDecl());
      }
    } else {
      ClassDecl = TypeNode->getAsCXXRecordDecl();
    }
    assert(ClassDecl != NULL);
    assert(ClassDecl != Declaration);
    if (Base.matches(*ClassDecl, this, Builder))
      return true;
    if (classIsDerivedFrom(ClassDecl, Base, Builder))
      return true;
  }
  return false;
}

bool MatchASTVisitor::TraverseDecl(Decl *DeclNode) {
  if (DeclNode == NULL) {
    return true;
  }
  match(*DeclNode);
  return RecursiveASTVisitor<MatchASTVisitor>::TraverseDecl(DeclNode);
}

bool MatchASTVisitor::TraverseStmt(Stmt *StmtNode) {
  if (StmtNode == NULL) {
    return true;
  }
  match(*StmtNode);
  return RecursiveASTVisitor<MatchASTVisitor>::TraverseStmt(StmtNode);
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
  match(NNS);
  // We only match the nested name specifier here (as opposed to traversing it)
  // because the traversal is already done in the parallel "Loc"-hierarchy.
  match(*NNS.getNestedNameSpecifier());
  return
      RecursiveASTVisitor<MatchASTVisitor>::TraverseNestedNameSpecifierLoc(NNS);
}

class MatchASTConsumer : public ASTConsumer {
public:
  MatchASTConsumer(
    std::vector<std::pair<const internal::DynTypedMatcher*,
                          MatchCallback*> > *MatcherCallbackPairs,
    MatchFinder::ParsingDoneTestCallback *ParsingDone)
    : Visitor(MatcherCallbackPairs),
      ParsingDone(ParsingDone) {}

private:
  virtual void HandleTranslationUnit(ASTContext &Context) {
    if (ParsingDone != NULL) {
      ParsingDone->run();
    }
    Visitor.set_active_ast_context(&Context);
    Visitor.onStartOfTranslationUnit();
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    Visitor.set_active_ast_context(NULL);
  }

  MatchASTVisitor Visitor;
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

MatchFinder::MatchFinder() : ParsingDone(NULL) {}

MatchFinder::~MatchFinder() {
  for (std::vector<std::pair<const internal::DynTypedMatcher*,
                             MatchCallback*> >::const_iterator
           It = MatcherCallbackPairs.begin(), End = MatcherCallbackPairs.end();
       It != End; ++It) {
    delete It->first;
  }
}

void MatchFinder::addMatcher(const DeclarationMatcher &NodeMatch,
                             MatchCallback *Action) {
  MatcherCallbackPairs.push_back(std::make_pair(
    new internal::Matcher<Decl>(NodeMatch), Action));
}

void MatchFinder::addMatcher(const TypeMatcher &NodeMatch,
                             MatchCallback *Action) {
  MatcherCallbackPairs.push_back(std::make_pair(
    new internal::Matcher<QualType>(NodeMatch), Action));
}

void MatchFinder::addMatcher(const StatementMatcher &NodeMatch,
                             MatchCallback *Action) {
  MatcherCallbackPairs.push_back(std::make_pair(
    new internal::Matcher<Stmt>(NodeMatch), Action));
}

void MatchFinder::addMatcher(const NestedNameSpecifierMatcher &NodeMatch,
                             MatchCallback *Action) {
  MatcherCallbackPairs.push_back(std::make_pair(
    new NestedNameSpecifierMatcher(NodeMatch), Action));
}

void MatchFinder::addMatcher(const NestedNameSpecifierLocMatcher &NodeMatch,
                             MatchCallback *Action) {
  MatcherCallbackPairs.push_back(std::make_pair(
    new NestedNameSpecifierLocMatcher(NodeMatch), Action));
}

void MatchFinder::addMatcher(const TypeLocMatcher &NodeMatch,
                             MatchCallback *Action) {
  MatcherCallbackPairs.push_back(std::make_pair(
    new TypeLocMatcher(NodeMatch), Action));
}

ASTConsumer *MatchFinder::newASTConsumer() {
  return new internal::MatchASTConsumer(&MatcherCallbackPairs, ParsingDone);
}

void MatchFinder::findAll(const Decl &Node, ASTContext &Context) {
  internal::MatchASTVisitor Visitor(&MatcherCallbackPairs);
  Visitor.set_active_ast_context(&Context);
  Visitor.TraverseDecl(const_cast<Decl*>(&Node));
}

void MatchFinder::findAll(const Stmt &Node, ASTContext &Context) {
  internal::MatchASTVisitor Visitor(&MatcherCallbackPairs);
  Visitor.set_active_ast_context(&Context);
  Visitor.TraverseStmt(const_cast<Stmt*>(&Node));
}

void MatchFinder::registerTestCallbackAfterParsing(
    MatchFinder::ParsingDoneTestCallback *NewParsingDone) {
  ParsingDone = NewParsingDone;
}

} // end namespace ast_matchers
} // end namespace clang
