//===--- ASTMatchersInternal.h - Structural query framework -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Implements the base layer of the matcher framework.
//
//  Matchers are methods that return a Matcher<T> which provides a method
//  Matches(...) which is a predicate on an AST node. The Matches method's
//  parameters define the context of the match, which allows matchers to recurse
//  or store the current node as bound to a specific string, so that it can be
//  retrieved later.
//
//  In general, matchers have two parts:
//  1. A function Matcher<T> MatcherName(<arguments>) which returns a Matcher<T>
//     based on the arguments and optionally on template type deduction based
//     on the arguments. Matcher<T>s form an implicit reverse hierarchy
//     to clang's AST class hierarchy, meaning that you can use a Matcher<Base>
//     everywhere a Matcher<Derived> is required.
//  2. An implementation of a class derived from MatcherInterface<T>.
//
//  The matcher functions are defined in ASTMatchers.h. To make it possible
//  to implement both the matcher function and the implementation of the matcher
//  interface in one place, ASTMatcherMacros.h defines macros that allow
//  implementing a matcher in a single place.
//
//  This file contains the base classes needed to construct the actual matchers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MATCHERS_AST_MATCHERS_INTERNAL_H
#define LLVM_CLANG_AST_MATCHERS_AST_MATCHERS_INTERNAL_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTTypeTraits.h"
#include "llvm/ADT/VariadicFunction.h"
#include "llvm/Support/type_traits.h"
#include <map>
#include <string>
#include <vector>

namespace clang {
namespace ast_matchers {

/// FIXME: Move into the llvm support library.
template <bool> struct CompileAssert {};
#define TOOLING_COMPILE_ASSERT(Expr, Msg) \
  typedef CompileAssert<(bool(Expr))> Msg[bool(Expr) ? 1 : -1]

class BoundNodes;

namespace internal {

class BoundNodesTreeBuilder;
/// \brief Internal version of BoundNodes. Holds all the bound nodes.
class BoundNodesMap {
public:
  /// \brief Adds \c Node to the map with key \c ID.
  ///
  /// The node's base type should be in NodeBaseType or it will be unaccessible.
  template <typename T>
  void addNode(StringRef ID, const T* Node) {
    NodeMap[ID] = ast_type_traits::DynTypedNode::create(*Node);
  }
  void addNode(StringRef ID, ast_type_traits::DynTypedNode Node) {
    NodeMap[ID] = Node;
  }

  /// \brief Returns the AST node bound to \c ID.
  ///
  /// Returns NULL if there was no node bound to \c ID or if there is a node but
  /// it cannot be converted to the specified type.
  template <typename T>
  const T *getNodeAs(StringRef ID) const {
    IDToNodeMap::const_iterator It = NodeMap.find(ID);
    if (It == NodeMap.end()) {
      return NULL;
    }
    return It->second.get<T>();
  }

  /// \brief Copies all ID/Node pairs to BoundNodesTreeBuilder \c Builder.
  void copyTo(BoundNodesTreeBuilder *Builder) const;

  /// \brief Copies all ID/Node pairs to BoundNodesMap \c Other.
  void copyTo(BoundNodesMap *Other) const;

private:
  /// \brief A map from IDs to the bound nodes.
  typedef std::map<std::string, ast_type_traits::DynTypedNode> IDToNodeMap;

  IDToNodeMap NodeMap;
};

/// \brief A tree of bound nodes in match results.
///
/// If a match can contain multiple matches on the same node with different
/// matching subexpressions, BoundNodesTree contains a branch for each of
/// those matching subexpressions.
///
/// BoundNodesTree's are created during the matching process; when a match
/// is found, we iterate over the tree and create a BoundNodes object containing
/// the union of all bound nodes on the path from the root to a each leaf.
class BoundNodesTree {
public:
  /// \brief A visitor interface to visit all BoundNodes results for a
  /// BoundNodesTree.
  class Visitor {
  public:
    virtual ~Visitor() {}

    /// \brief Called multiple times during a single call to VisitMatches(...).
    ///
    /// 'BoundNodesView' contains the bound nodes for a single match.
    virtual void visitMatch(const BoundNodes& BoundNodesView) = 0;
  };

  BoundNodesTree();

  /// \brief Create a BoundNodesTree from pre-filled maps of bindings.
  BoundNodesTree(const BoundNodesMap& Bindings,
                 const std::vector<BoundNodesTree> RecursiveBindings);

  /// \brief Adds all bound nodes to \c Builder.
  void copyTo(BoundNodesTreeBuilder* Builder) const;

  /// \brief Visits all matches that this BoundNodesTree represents.
  ///
  /// The ownership of 'ResultVisitor' remains at the caller.
  void visitMatches(Visitor* ResultVisitor);

private:
  void visitMatchesRecursively(
      Visitor* ResultVistior,
      const BoundNodesMap& AggregatedBindings);

  // FIXME: Find out whether we want to use different data structures here -
  // first benchmarks indicate that it doesn't matter though.

  BoundNodesMap Bindings;

  std::vector<BoundNodesTree> RecursiveBindings;
};

/// \brief Creates BoundNodesTree objects.
///
/// The tree builder is used during the matching process to insert the bound
/// nodes from the Id matcher.
class BoundNodesTreeBuilder {
public:
  BoundNodesTreeBuilder();

  /// \brief Add a binding from an id to a node.
  template <typename T>
  void setBinding(const std::string &Id, const T *Node) {
    Bindings.addNode(Id, Node);
  }
  void setBinding(const std::string &Id, ast_type_traits::DynTypedNode Node) {
    Bindings.addNode(Id, Node);
  }

  /// \brief Adds a branch in the tree.
  void addMatch(const BoundNodesTree& Bindings);

  /// \brief Returns a BoundNodes object containing all current bindings.
  BoundNodesTree build() const;

private:
  BoundNodesTreeBuilder(const BoundNodesTreeBuilder &) LLVM_DELETED_FUNCTION;
  void operator=(const BoundNodesTreeBuilder &) LLVM_DELETED_FUNCTION;

  BoundNodesMap Bindings;

  std::vector<BoundNodesTree> RecursiveBindings;
};

class ASTMatchFinder;

/// \brief Generic interface for matchers on an AST node of type T.
///
/// Implement this if your matcher may need to inspect the children or
/// descendants of the node or bind matched nodes to names. If you are
/// writing a simple matcher that only inspects properties of the
/// current node and doesn't care about its children or descendants,
/// implement SingleNodeMatcherInterface instead.
template <typename T>
class MatcherInterface : public llvm::RefCountedBaseVPTR {
public:
  virtual ~MatcherInterface() {}

  /// \brief Returns true if 'Node' can be matched.
  ///
  /// May bind 'Node' to an ID via 'Builder', or recurse into
  /// the AST via 'Finder'.
  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const = 0;
};

/// \brief Interface for matchers that only evaluate properties on a single
/// node.
template <typename T>
class SingleNodeMatcherInterface : public MatcherInterface<T> {
public:
  /// \brief Returns true if the matcher matches the provided node.
  ///
  /// A subclass must implement this instead of Matches().
  virtual bool matchesNode(const T &Node) const = 0;

private:
  /// Implements MatcherInterface::Matches.
  virtual bool matches(const T &Node,
                       ASTMatchFinder * /* Finder */,
                       BoundNodesTreeBuilder * /*  Builder */) const {
    return matchesNode(Node);
  }
};

/// \brief Base class for all matchers that works on a \c DynTypedNode.
///
/// Matcher implementations will check whether the \c DynTypedNode is
/// convertible into the respecitve types and then do the actual match
/// on the actual node, or return false if it is not convertible.
class DynTypedMatcher {
public:
  virtual ~DynTypedMatcher() {}

  /// \brief Returns true if the matcher matches the given \c DynNode.
  virtual bool matches(const ast_type_traits::DynTypedNode DynNode,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const = 0;

  /// \brief Returns a unique ID for the matcher.
  virtual uint64_t getID() const = 0;
};

/// \brief Wrapper of a MatcherInterface<T> *that allows copying.
///
/// A Matcher<Base> can be used anywhere a Matcher<Derived> is
/// required. This establishes an is-a relationship which is reverse
/// to the AST hierarchy. In other words, Matcher<T> is contravariant
/// with respect to T. The relationship is built via a type conversion
/// operator rather than a type hierarchy to be able to templatize the
/// type hierarchy instead of spelling it out.
template <typename T>
class Matcher : public DynTypedMatcher {
public:
  /// \brief Takes ownership of the provided implementation pointer.
  explicit Matcher(MatcherInterface<T> *Implementation)
      : Implementation(Implementation) {}

  /// \brief Implicitly converts \c Other to a Matcher<T>.
  ///
  /// Requires \c T to be derived from \c From.
  template <typename From>
  Matcher(const Matcher<From> &Other,
          typename llvm::enable_if_c<
            llvm::is_base_of<From, T>::value &&
            !llvm::is_same<From, T>::value >::type* = 0)
      : Implementation(new ImplicitCastMatcher<From>(Other)) {}

  /// \brief Implicitly converts \c Matcher<Type> to \c Matcher<QualType>.
  ///
  /// The resulting matcher is not strict, i.e. ignores qualifiers.
  template <typename TypeT>
  Matcher(const Matcher<TypeT> &Other,
          typename llvm::enable_if_c<
            llvm::is_same<T, QualType>::value &&
            llvm::is_same<TypeT, Type>::value >::type* = 0)
      : Implementation(new TypeToQualType<TypeT>(Other)) {}

  /// \brief Forwards the call to the underlying MatcherInterface<T> pointer.
  bool matches(const T &Node,
               ASTMatchFinder *Finder,
               BoundNodesTreeBuilder *Builder) const {
    return Implementation->matches(Node, Finder, Builder);
  }

  /// \brief Returns an ID that uniquely identifies the matcher.
  uint64_t getID() const {
    /// FIXME: Document the requirements this imposes on matcher
    /// implementations (no new() implementation_ during a Matches()).
    return reinterpret_cast<uint64_t>(Implementation.getPtr());
  }

  /// \brief Returns whether the matcher matches on the given \c DynNode.
  virtual bool matches(const ast_type_traits::DynTypedNode DynNode,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    const T *Node = DynNode.get<T>();
    if (!Node) return false;
    return matches(*Node, Finder, Builder);
  }

  /// \brief Allows the conversion of a \c Matcher<Type> to a \c
  /// Matcher<QualType>.
  ///
  /// Depending on the constructor argument, the matcher is either strict, i.e.
  /// does only matches in the absence of qualifiers, or not, i.e. simply
  /// ignores any qualifiers.
  template <typename TypeT>
  class TypeToQualType : public MatcherInterface<QualType> {
   public:
    TypeToQualType(const Matcher<TypeT> &InnerMatcher)
        : InnerMatcher(InnerMatcher) {}

    virtual bool matches(const QualType &Node,
                         ASTMatchFinder *Finder,
                         BoundNodesTreeBuilder *Builder) const {
      if (Node.isNull())
        return false;
      return InnerMatcher.matches(*Node, Finder, Builder);
    }
   private:
    const Matcher<TypeT> InnerMatcher;
  };

private:
  /// \brief Allows conversion from Matcher<Base> to Matcher<T> if T
  /// is derived from Base.
  template <typename Base>
  class ImplicitCastMatcher : public MatcherInterface<T> {
  public:
    explicit ImplicitCastMatcher(const Matcher<Base> &From)
        : From(From) {}

    virtual bool matches(const T &Node,
                         ASTMatchFinder *Finder,
                         BoundNodesTreeBuilder *Builder) const {
      return From.matches(Node, Finder, Builder);
    }

  private:
    const Matcher<Base> From;
  };

  llvm::IntrusiveRefCntPtr< MatcherInterface<T> > Implementation;
};  // class Matcher

/// \brief A convenient helper for creating a Matcher<T> without specifying
/// the template type argument.
template <typename T>
inline Matcher<T> makeMatcher(MatcherInterface<T> *Implementation) {
  return Matcher<T>(Implementation);
}

/// \brief Matches declarations for QualType and CallExpr.
///
/// Type argument DeclMatcherT is required by PolymorphicMatcherWithParam1 but
/// not actually used.
template <typename T, typename DeclMatcherT>
class HasDeclarationMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT((llvm::is_same< DeclMatcherT,
                                         Matcher<Decl> >::value),
                          instantiated_with_wrong_types);
public:
  explicit HasDeclarationMatcher(const Matcher<Decl> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return matchesSpecialized(Node, Finder, Builder);
  }

private:
  /// \brief Extracts the CXXRecordDecl of a QualType and returns whether the
  /// inner matcher matches on it.
  bool matchesSpecialized(const QualType &Node, ASTMatchFinder *Finder,
                          BoundNodesTreeBuilder *Builder) const {
    /// FIXME: Add other ways to convert...
    if (Node.isNull())
      return false;
    return matchesDecl(Node->getAsCXXRecordDecl(), Finder, Builder);
  }

  /// \brief Extracts the Decl of the callee of a CallExpr and returns whether
  /// the inner matcher matches on it.
  bool matchesSpecialized(const CallExpr &Node, ASTMatchFinder *Finder,
                          BoundNodesTreeBuilder *Builder) const {
    return matchesDecl(Node.getCalleeDecl(), Finder, Builder);
  }

  /// \brief Extracts the Decl of the constructor call and returns whether the
  /// inner matcher matches on it.
  bool matchesSpecialized(const CXXConstructExpr &Node,
                          ASTMatchFinder *Finder,
                          BoundNodesTreeBuilder *Builder) const {
    return matchesDecl(Node.getConstructor(), Finder, Builder);
  }

  /// \brief Extracts the \c ValueDecl a \c MemberExpr refers to and returns
  /// whether the inner matcher matches on it.
  bool matchesSpecialized(const MemberExpr &Node,
                          ASTMatchFinder *Finder,
                          BoundNodesTreeBuilder *Builder) const {
    return matchesDecl(Node.getMemberDecl(), Finder, Builder);
  }

  /// \brief Returns whether the inner matcher \c Node. Returns false if \c Node
  /// is \c NULL.
  bool matchesDecl(const Decl *Node,
                   ASTMatchFinder *Finder,
                   BoundNodesTreeBuilder *Builder) const {
    return Node != NULL && InnerMatcher.matches(*Node, Finder, Builder);
  }

  const Matcher<Decl> InnerMatcher;
};

/// \brief IsBaseType<T>::value is true if T is a "base" type in the AST
/// node class hierarchies.
template <typename T>
struct IsBaseType {
  static const bool value =
      (llvm::is_same<T, Decl>::value ||
       llvm::is_same<T, Stmt>::value ||
       llvm::is_same<T, QualType>::value ||
       llvm::is_same<T, Type>::value ||
       llvm::is_same<T, TypeLoc>::value ||
       llvm::is_same<T, NestedNameSpecifier>::value ||
       llvm::is_same<T, NestedNameSpecifierLoc>::value ||
       llvm::is_same<T, CXXCtorInitializer>::value);
};
template <typename T>
const bool IsBaseType<T>::value;

/// \brief Interface that allows matchers to traverse the AST.
/// FIXME: Find a better name.
///
/// This provides three entry methods for each base node type in the AST:
/// - \c matchesChildOf:
///   Matches a matcher on every child node of the given node. Returns true
///   if at least one child node could be matched.
/// - \c matchesDescendantOf:
///   Matches a matcher on all descendant nodes of the given node. Returns true
///   if at least one descendant matched.
/// - \c matchesAncestorOf:
///   Matches a matcher on all ancestors of the given node. Returns true if
///   at least one ancestor matched.
///
/// FIXME: Currently we only allow Stmt and Decl nodes to start a traversal.
/// In the future, we wan to implement this for all nodes for which it makes
/// sense. In the case of matchesAncestorOf, we'll want to implement it for
/// all nodes, as all nodes have ancestors.
class ASTMatchFinder {
public:
  /// \brief Defines how we descend a level in the AST when we pass
  /// through expressions.
  enum TraversalKind {
    /// Will traverse any child nodes.
    TK_AsIs,
    /// Will not traverse implicit casts and parentheses.
    TK_IgnoreImplicitCastsAndParentheses
  };

  /// \brief Defines how bindings are processed on recursive matches.
  enum BindKind {
    /// Stop at the first match and only bind the first match.
    BK_First,
    /// Create results for all combinations of bindings that match.
    BK_All
  };

  /// \brief Defines which ancestors are considered for a match.
  enum AncestorMatchMode {
    /// All ancestors.
    AMM_All,
    /// Direct parent only.
    AMM_ParentOnly
  };

  virtual ~ASTMatchFinder() {}

  /// \brief Returns true if the given class is directly or indirectly derived
  /// from a base type matching \c base.
  ///
  /// A class is considered to be also derived from itself.
  virtual bool classIsDerivedFrom(const CXXRecordDecl *Declaration,
                                  const Matcher<NamedDecl> &Base,
                                  BoundNodesTreeBuilder *Builder) = 0;

  template <typename T>
  bool matchesChildOf(const T &Node,
                      const DynTypedMatcher &Matcher,
                      BoundNodesTreeBuilder *Builder,
                      TraversalKind Traverse,
                      BindKind Bind) {
    TOOLING_COMPILE_ASSERT(
        (llvm::is_base_of<Decl, T>::value ||
         llvm::is_base_of<Stmt, T>::value ||
         llvm::is_base_of<NestedNameSpecifier, T>::value ||
         llvm::is_base_of<NestedNameSpecifierLoc, T>::value ||
         llvm::is_base_of<TypeLoc, T>::value ||
         llvm::is_base_of<QualType, T>::value),
        unsupported_type_for_recursive_matching);
   return matchesChildOf(ast_type_traits::DynTypedNode::create(Node),
                          Matcher, Builder, Traverse, Bind);
  }

  template <typename T>
  bool matchesDescendantOf(const T &Node,
                           const DynTypedMatcher &Matcher,
                           BoundNodesTreeBuilder *Builder,
                           BindKind Bind) {
    TOOLING_COMPILE_ASSERT(
        (llvm::is_base_of<Decl, T>::value ||
         llvm::is_base_of<Stmt, T>::value ||
         llvm::is_base_of<NestedNameSpecifier, T>::value ||
         llvm::is_base_of<NestedNameSpecifierLoc, T>::value ||
         llvm::is_base_of<TypeLoc, T>::value ||
         llvm::is_base_of<QualType, T>::value),
        unsupported_type_for_recursive_matching);
    return matchesDescendantOf(ast_type_traits::DynTypedNode::create(Node),
                               Matcher, Builder, Bind);
  }

  // FIXME: Implement support for BindKind.
  template <typename T>
  bool matchesAncestorOf(const T &Node,
                         const DynTypedMatcher &Matcher,
                         BoundNodesTreeBuilder *Builder,
                         AncestorMatchMode MatchMode) {
    TOOLING_COMPILE_ASSERT((llvm::is_base_of<Decl, T>::value ||
                            llvm::is_base_of<Stmt, T>::value),
                           only_Decl_or_Stmt_allowed_for_recursive_matching);
    return matchesAncestorOf(ast_type_traits::DynTypedNode::create(Node),
                             Matcher, Builder, MatchMode);
  }

protected:
  virtual bool matchesChildOf(const ast_type_traits::DynTypedNode &Node,
                              const DynTypedMatcher &Matcher,
                              BoundNodesTreeBuilder *Builder,
                              TraversalKind Traverse,
                              BindKind Bind) = 0;

  virtual bool matchesDescendantOf(const ast_type_traits::DynTypedNode &Node,
                                   const DynTypedMatcher &Matcher,
                                   BoundNodesTreeBuilder *Builder,
                                   BindKind Bind) = 0;

  virtual bool matchesAncestorOf(const ast_type_traits::DynTypedNode &Node,
                                 const DynTypedMatcher &Matcher,
                                 BoundNodesTreeBuilder *Builder,
                                 AncestorMatchMode MatchMode) = 0;
};

/// \brief Converts a \c Matcher<T> to a matcher of desired type \c To by
/// "adapting" a \c To into a \c T.
///
/// The \c ArgumentAdapterT argument specifies how the adaptation is done.
///
/// For example:
///   \c ArgumentAdaptingMatcher<HasMatcher, T>(InnerMatcher);
/// Given that \c InnerMatcher is of type \c Matcher<T>, this returns a matcher
/// that is convertible into any matcher of type \c To by constructing
/// \c HasMatcher<To, T>(InnerMatcher).
///
/// If a matcher does not need knowledge about the inner type, prefer to use
/// PolymorphicMatcherWithParam1.
template <template <typename ToArg, typename FromArg> class ArgumentAdapterT,
          typename T>
class ArgumentAdaptingMatcher {
public:
  explicit ArgumentAdaptingMatcher(const Matcher<T> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  template <typename To>
  operator Matcher<To>() const {
    return Matcher<To>(new ArgumentAdapterT<To, T>(InnerMatcher));
  }

private:
  const Matcher<T> InnerMatcher;
};

/// \brief A PolymorphicMatcherWithParamN<MatcherT, P1, ..., PN> object can be
/// created from N parameters p1, ..., pN (of type P1, ..., PN) and
/// used as a Matcher<T> where a MatcherT<T, P1, ..., PN>(p1, ..., pN)
/// can be constructed.
///
/// For example:
/// - PolymorphicMatcherWithParam0<IsDefinitionMatcher>()
///   creates an object that can be used as a Matcher<T> for any type T
///   where an IsDefinitionMatcher<T>() can be constructed.
/// - PolymorphicMatcherWithParam1<ValueEqualsMatcher, int>(42)
///   creates an object that can be used as a Matcher<T> for any type T
///   where a ValueEqualsMatcher<T, int>(42) can be constructed.
template <template <typename T> class MatcherT>
class PolymorphicMatcherWithParam0 {
public:
  template <typename T>
  operator Matcher<T>() const {
    return Matcher<T>(new MatcherT<T>());
  }
};

template <template <typename T, typename P1> class MatcherT,
          typename P1>
class PolymorphicMatcherWithParam1 {
public:
  explicit PolymorphicMatcherWithParam1(const P1 &Param1)
      : Param1(Param1) {}

  template <typename T>
  operator Matcher<T>() const {
    return Matcher<T>(new MatcherT<T, P1>(Param1));
  }

private:
  const P1 Param1;
};

template <template <typename T, typename P1, typename P2> class MatcherT,
          typename P1, typename P2>
class PolymorphicMatcherWithParam2 {
public:
  PolymorphicMatcherWithParam2(const P1 &Param1, const P2 &Param2)
      : Param1(Param1), Param2(Param2) {}

  template <typename T>
  operator Matcher<T>() const {
    return Matcher<T>(new MatcherT<T, P1, P2>(Param1, Param2));
  }

private:
  const P1 Param1;
  const P2 Param2;
};

/// \brief Matches any instance of the given NodeType.
///
/// This is useful when a matcher syntactically requires a child matcher,
/// but the context doesn't care. See for example: anything().
///
/// FIXME: Alternatively we could also create a IsAMatcher or something
/// that checks that a dyn_cast is possible. This is purely needed for the
/// difference between calling for example:
///   record()
/// and
///   record(SomeMatcher)
/// In the second case we need the correct type we were dyn_cast'ed to in order
/// to get the right type for the inner matcher. In the first case we don't need
/// that, but we use the type conversion anyway and insert a TrueMatcher.
template <typename T>
class TrueMatcher : public SingleNodeMatcherInterface<T>  {
public:
  virtual bool matchesNode(const T &Node) const {
    return true;
  }
};

/// \brief Provides a MatcherInterface<T> for a Matcher<To> that matches if T is
/// dyn_cast'able into To and the given Matcher<To> matches on the dyn_cast'ed
/// node.
template <typename T, typename To>
class DynCastMatcher : public MatcherInterface<T> {
public:
  explicit DynCastMatcher(const Matcher<To> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    const To *InnerMatchValue = llvm::dyn_cast<To>(&Node);
    return InnerMatchValue != NULL &&
      InnerMatcher.matches(*InnerMatchValue, Finder, Builder);
  }

private:
  const Matcher<To> InnerMatcher;
};

/// \brief Matcher<T> that wraps an inner Matcher<T> and binds the matched node
/// to an ID if the inner matcher matches on the node.
template <typename T>
class IdMatcher : public MatcherInterface<T> {
public:
  /// \brief Creates an IdMatcher that binds to 'ID' if 'InnerMatcher' matches
  /// the node.
  IdMatcher(StringRef ID, const Matcher<T> &InnerMatcher)
      : ID(ID), InnerMatcher(InnerMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    bool Result = InnerMatcher.matches(Node, Finder, Builder);
    if (Result) {
      Builder->setBinding(ID, &Node);
    }
    return Result;
  }

private:
  const std::string ID;
  const Matcher<T> InnerMatcher;
};

/// \brief A Matcher that allows binding the node it matches to an id.
///
/// BindableMatcher provides a \a bind() method that allows binding the
/// matched node to an id if the match was successful.
template <typename T>
class BindableMatcher : public Matcher<T> {
public:
  BindableMatcher(MatcherInterface<T> *Implementation)
    : Matcher<T>(Implementation) {}

  /// \brief Returns a matcher that will bind the matched node on a match.
  ///
  /// The returned matcher is equivalent to this matcher, but will
  /// bind the matched node on a match.
  Matcher<T> bind(StringRef ID) const {
    return Matcher<T>(new IdMatcher<T>(ID, *this));
  }
};

/// \brief Matches nodes of type T that have child nodes of type ChildT for
/// which a specified child matcher matches.
///
/// ChildT must be an AST base type.
template <typename T, typename ChildT>
class HasMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT(IsBaseType<ChildT>::value,
                         has_only_accepts_base_type_matcher);
public:
  explicit HasMatcher(const Matcher<ChildT> &ChildMatcher)
      : ChildMatcher(ChildMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return Finder->matchesChildOf(
        Node, ChildMatcher, Builder,
        ASTMatchFinder::TK_IgnoreImplicitCastsAndParentheses,
        ASTMatchFinder::BK_First);
  }

 private:
  const Matcher<ChildT> ChildMatcher;
};

/// \brief Matches nodes of type T that have child nodes of type ChildT for
/// which a specified child matcher matches. ChildT must be an AST base
/// type.
/// As opposed to the HasMatcher, the ForEachMatcher will produce a match
/// for each child that matches.
template <typename T, typename ChildT>
class ForEachMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT(IsBaseType<ChildT>::value,
                         for_each_only_accepts_base_type_matcher);
 public:
  explicit ForEachMatcher(const Matcher<ChildT> &ChildMatcher)
      : ChildMatcher(ChildMatcher) {}

  virtual bool matches(const T& Node,
                       ASTMatchFinder* Finder,
                       BoundNodesTreeBuilder* Builder) const {
    return Finder->matchesChildOf(
      Node, ChildMatcher, Builder,
      ASTMatchFinder::TK_IgnoreImplicitCastsAndParentheses,
      ASTMatchFinder::BK_All);
  }

private:
  const Matcher<ChildT> ChildMatcher;
};

/// \brief Matches nodes of type T if the given Matcher<T> does not match.
///
/// Type argument MatcherT is required by PolymorphicMatcherWithParam1
/// but not actually used. It will always be instantiated with a type
/// convertible to Matcher<T>.
template <typename T, typename MatcherT>
class NotMatcher : public MatcherInterface<T> {
public:
  explicit NotMatcher(const Matcher<T> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return !InnerMatcher.matches(Node, Finder, Builder);
  }

private:
  const Matcher<T> InnerMatcher;
};

/// \brief Matches nodes of type T for which both provided matchers match.
///
/// Type arguments MatcherT1 and MatcherT2 are required by
/// PolymorphicMatcherWithParam2 but not actually used. They will
/// always be instantiated with types convertible to Matcher<T>.
template <typename T, typename MatcherT1, typename MatcherT2>
class AllOfMatcher : public MatcherInterface<T> {
public:
  AllOfMatcher(const Matcher<T> &InnerMatcher1, const Matcher<T> &InnerMatcher2)
      : InnerMatcher1(InnerMatcher1), InnerMatcher2(InnerMatcher2) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return InnerMatcher1.matches(Node, Finder, Builder) &&
           InnerMatcher2.matches(Node, Finder, Builder);
  }

private:
  const Matcher<T> InnerMatcher1;
  const Matcher<T> InnerMatcher2;
};

/// \brief Matches nodes of type T for which at least one of the two provided
/// matchers matches.
///
/// Type arguments MatcherT1 and MatcherT2 are
/// required by PolymorphicMatcherWithParam2 but not actually
/// used. They will always be instantiated with types convertible to
/// Matcher<T>.
template <typename T, typename MatcherT1, typename MatcherT2>
class AnyOfMatcher : public MatcherInterface<T> {
public:
  AnyOfMatcher(const Matcher<T> &InnerMatcher1, const Matcher<T> &InnerMatcher2)
      : InnerMatcher1(InnerMatcher1), InnertMatcher2(InnerMatcher2) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return InnerMatcher1.matches(Node, Finder, Builder) ||
           InnertMatcher2.matches(Node, Finder, Builder);
  }

private:
  const Matcher<T> InnerMatcher1;
  const Matcher<T> InnertMatcher2;
};

/// \brief Creates a Matcher<T> that matches if all inner matchers match.
template<typename T>
BindableMatcher<T> makeAllOfComposite(
    ArrayRef<const Matcher<T> *> InnerMatchers) {
  if (InnerMatchers.empty())
    return BindableMatcher<T>(new TrueMatcher<T>);
  MatcherInterface<T> *InnerMatcher = new TrueMatcher<T>;
  for (int i = InnerMatchers.size() - 1; i >= 0; --i) {
    InnerMatcher = new AllOfMatcher<T, Matcher<T>, Matcher<T> >(
      *InnerMatchers[i], makeMatcher(InnerMatcher));
  }
  return BindableMatcher<T>(InnerMatcher);
}

/// \brief Creates a Matcher<T> that matches if
/// T is dyn_cast'able into InnerT and all inner matchers match.
///
/// Returns BindableMatcher, as matchers that use dyn_cast have
/// the same object both to match on and to run submatchers on,
/// so there is no ambiguity with what gets bound.
template<typename T, typename InnerT>
BindableMatcher<T> makeDynCastAllOfComposite(
    ArrayRef<const Matcher<InnerT> *> InnerMatchers) {
  return BindableMatcher<T>(new DynCastMatcher<T, InnerT>(
    makeAllOfComposite(InnerMatchers)));
}

/// \brief Matches nodes of type T that have at least one descendant node of
/// type DescendantT for which the given inner matcher matches.
///
/// DescendantT must be an AST base type.
template <typename T, typename DescendantT>
class HasDescendantMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT(IsBaseType<DescendantT>::value,
                         has_descendant_only_accepts_base_type_matcher);
public:
  explicit HasDescendantMatcher(const Matcher<DescendantT> &DescendantMatcher)
      : DescendantMatcher(DescendantMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return Finder->matchesDescendantOf(
        Node, DescendantMatcher, Builder, ASTMatchFinder::BK_First);
  }

 private:
  const Matcher<DescendantT> DescendantMatcher;
};

/// \brief Matches nodes of type \c T that have a parent node of type \c ParentT
/// for which the given inner matcher matches.
///
/// \c ParentT must be an AST base type.
template <typename T, typename ParentT>
class HasParentMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT(IsBaseType<ParentT>::value,
                         has_parent_only_accepts_base_type_matcher);
public:
  explicit HasParentMatcher(const Matcher<ParentT> &ParentMatcher)
      : ParentMatcher(ParentMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return Finder->matchesAncestorOf(
        Node, ParentMatcher, Builder, ASTMatchFinder::AMM_ParentOnly);
  }

 private:
  const Matcher<ParentT> ParentMatcher;
};

/// \brief Matches nodes of type \c T that have at least one ancestor node of
/// type \c AncestorT for which the given inner matcher matches.
///
/// \c AncestorT must be an AST base type.
template <typename T, typename AncestorT>
class HasAncestorMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT(IsBaseType<AncestorT>::value,
                         has_ancestor_only_accepts_base_type_matcher);
public:
  explicit HasAncestorMatcher(const Matcher<AncestorT> &AncestorMatcher)
      : AncestorMatcher(AncestorMatcher) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return Finder->matchesAncestorOf(
        Node, AncestorMatcher, Builder, ASTMatchFinder::AMM_All);
  }

 private:
  const Matcher<AncestorT> AncestorMatcher;
};

/// \brief Matches nodes of type T that have at least one descendant node of
/// type DescendantT for which the given inner matcher matches.
///
/// DescendantT must be an AST base type.
/// As opposed to HasDescendantMatcher, ForEachDescendantMatcher will match
/// for each descendant node that matches instead of only for the first.
template <typename T, typename DescendantT>
class ForEachDescendantMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT(IsBaseType<DescendantT>::value,
                         for_each_descendant_only_accepts_base_type_matcher);
 public:
  explicit ForEachDescendantMatcher(
      const Matcher<DescendantT>& DescendantMatcher)
      : DescendantMatcher(DescendantMatcher) {}

  virtual bool matches(const T& Node,
                       ASTMatchFinder* Finder,
                       BoundNodesTreeBuilder* Builder) const {
    return Finder->matchesDescendantOf(Node, DescendantMatcher, Builder,
                                       ASTMatchFinder::BK_All);
  }

private:
  const Matcher<DescendantT> DescendantMatcher;
};

/// \brief Matches on nodes that have a getValue() method if getValue() equals
/// the value the ValueEqualsMatcher was constructed with.
template <typename T, typename ValueT>
class ValueEqualsMatcher : public SingleNodeMatcherInterface<T> {
  TOOLING_COMPILE_ASSERT((llvm::is_base_of<CharacterLiteral, T>::value ||
                         llvm::is_base_of<CXXBoolLiteralExpr,
                                          T>::value ||
                         llvm::is_base_of<FloatingLiteral, T>::value ||
                         llvm::is_base_of<IntegerLiteral, T>::value),
                         the_node_must_have_a_getValue_method);
public:
  explicit ValueEqualsMatcher(const ValueT &ExpectedValue)
      : ExpectedValue(ExpectedValue) {}

  virtual bool matchesNode(const T &Node) const {
    return Node.getValue() == ExpectedValue;
  }

private:
  const ValueT ExpectedValue;
};

template <typename T>
class IsDefinitionMatcher : public SingleNodeMatcherInterface<T> {
  TOOLING_COMPILE_ASSERT(
    (llvm::is_base_of<TagDecl, T>::value) ||
    (llvm::is_base_of<VarDecl, T>::value) ||
    (llvm::is_base_of<FunctionDecl, T>::value),
    is_definition_requires_isThisDeclarationADefinition_method);
public:
  virtual bool matchesNode(const T &Node) const {
    return Node.isThisDeclarationADefinition();
  }
};

/// \brief Matches on template instantiations for FunctionDecl, VarDecl or
/// CXXRecordDecl nodes.
template <typename T>
class IsTemplateInstantiationMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT((llvm::is_base_of<FunctionDecl, T>::value) ||
                         (llvm::is_base_of<VarDecl, T>::value) ||
                         (llvm::is_base_of<CXXRecordDecl, T>::value),
                         requires_getTemplateSpecializationKind_method);
 public:
  virtual bool matches(const T& Node,
                       ASTMatchFinder* Finder,
                       BoundNodesTreeBuilder* Builder) const {
    return (Node.getTemplateSpecializationKind() ==
                TSK_ImplicitInstantiation ||
            Node.getTemplateSpecializationKind() ==
                TSK_ExplicitInstantiationDefinition);
  }
};

/// \brief Matches on explicit template specializations for FunctionDecl,
/// VarDecl or CXXRecordDecl nodes.
template <typename T>
class IsExplicitTemplateSpecializationMatcher : public MatcherInterface<T> {
  TOOLING_COMPILE_ASSERT((llvm::is_base_of<FunctionDecl, T>::value) ||
                         (llvm::is_base_of<VarDecl, T>::value) ||
                         (llvm::is_base_of<CXXRecordDecl, T>::value),
                         requires_getTemplateSpecializationKind_method);
 public:
  virtual bool matches(const T& Node,
                       ASTMatchFinder* Finder,
                       BoundNodesTreeBuilder* Builder) const {
    return (Node.getTemplateSpecializationKind() == TSK_ExplicitSpecialization);
  }
};

class IsArrowMatcher : public SingleNodeMatcherInterface<MemberExpr> {
public:
  virtual bool matchesNode(const MemberExpr &Node) const {
    return Node.isArrow();
  }
};

class IsConstQualifiedMatcher
    : public SingleNodeMatcherInterface<QualType> {
 public:
  virtual bool matchesNode(const QualType& Node) const {
    return Node.isConstQualified();
  }
};

/// \brief A VariadicDynCastAllOfMatcher<SourceT, TargetT> object is a
/// variadic functor that takes a number of Matcher<TargetT> and returns a
/// Matcher<SourceT> that matches TargetT nodes that are matched by all of the
/// given matchers, if SourceT can be dynamically casted into TargetT.
///
/// For example:
///   const VariadicDynCastAllOfMatcher<
///       Decl, CXXRecordDecl> record;
/// Creates a functor record(...) that creates a Matcher<Decl> given
/// a variable number of arguments of type Matcher<CXXRecordDecl>.
/// The returned matcher matches if the given Decl can by dynamically
/// casted to CXXRecordDecl and all given matchers match.
template <typename SourceT, typename TargetT>
class VariadicDynCastAllOfMatcher
    : public llvm::VariadicFunction<
        BindableMatcher<SourceT>, Matcher<TargetT>,
        makeDynCastAllOfComposite<SourceT, TargetT> > {
public:
  VariadicDynCastAllOfMatcher() {}
};

/// \brief A \c VariadicAllOfMatcher<T> object is a variadic functor that takes
/// a number of \c Matcher<T> and returns a \c Matcher<T> that matches \c T
/// nodes that are matched by all of the given matchers.
///
/// For example:
///   const VariadicAllOfMatcher<NestedNameSpecifier> nestedNameSpecifier;
/// Creates a functor nestedNameSpecifier(...) that creates a
/// \c Matcher<NestedNameSpecifier> given a variable number of arguments of type
/// \c Matcher<NestedNameSpecifier>.
/// The returned matcher matches if all given matchers match.
template <typename T>
class VariadicAllOfMatcher : public llvm::VariadicFunction<
                               BindableMatcher<T>, Matcher<T>,
                               makeAllOfComposite<T> > {
public:
  VariadicAllOfMatcher() {}
};

/// \brief Matches nodes of type \c TLoc for which the inner
/// \c Matcher<T> matches.
template <typename TLoc, typename T>
class LocMatcher : public MatcherInterface<TLoc> {
public:
  explicit LocMatcher(const Matcher<T> &InnerMatcher)
    : InnerMatcher(InnerMatcher) {}

  virtual bool matches(const TLoc &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    if (!Node)
      return false;
    return InnerMatcher.matches(*extract(Node), Finder, Builder);
  }

private:
  const NestedNameSpecifier *extract(const NestedNameSpecifierLoc &Loc) const {
    return Loc.getNestedNameSpecifier();
  }

  const Matcher<T> InnerMatcher;
};

/// \brief Matches \c NestedNameSpecifiers with a prefix matching another
/// \c Matcher<NestedNameSpecifier>.
class NestedNameSpecifierPrefixMatcher
  : public MatcherInterface<NestedNameSpecifier> {
public:
  explicit NestedNameSpecifierPrefixMatcher(
    const Matcher<NestedNameSpecifier> &InnerMatcher)
    : InnerMatcher(InnerMatcher) {}

  virtual bool matches(const NestedNameSpecifier &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    NestedNameSpecifier *NextNode = Node.getPrefix();
    if (NextNode == NULL)
      return false;
    return InnerMatcher.matches(*NextNode, Finder, Builder);
  }

private:
  const Matcher<NestedNameSpecifier> InnerMatcher;
};

/// \brief Matches \c NestedNameSpecifierLocs with a prefix matching another
/// \c Matcher<NestedNameSpecifierLoc>.
class NestedNameSpecifierLocPrefixMatcher
  : public MatcherInterface<NestedNameSpecifierLoc> {
public:
  explicit NestedNameSpecifierLocPrefixMatcher(
    const Matcher<NestedNameSpecifierLoc> &InnerMatcher)
    : InnerMatcher(InnerMatcher) {}

  virtual bool matches(const NestedNameSpecifierLoc &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    NestedNameSpecifierLoc NextNode = Node.getPrefix();
    if (!NextNode)
      return false;
    return InnerMatcher.matches(NextNode, Finder, Builder);
  }

private:
  const Matcher<NestedNameSpecifierLoc> InnerMatcher;
};

/// \brief Matches \c TypeLocs based on an inner matcher matching a certain
/// \c QualType.
///
/// Used to implement the \c loc() matcher.
class TypeLocTypeMatcher : public MatcherInterface<TypeLoc> {
public:
  explicit TypeLocTypeMatcher(const Matcher<QualType> &InnerMatcher)
      : InnerMatcher(InnerMatcher) {}

  virtual bool matches(const TypeLoc &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    if (!Node)
      return false;
    return InnerMatcher.matches(Node.getType(), Finder, Builder);
  }

private:
  const Matcher<QualType> InnerMatcher;
};

/// \brief Matches nodes of type \c T for which the inner matcher matches on a
/// another node of type \c T that can be reached using a given traverse
/// function.
template <typename T>
class TypeTraverseMatcher : public MatcherInterface<T> {
public:
  explicit TypeTraverseMatcher(const Matcher<QualType> &InnerMatcher,
                               QualType (T::*TraverseFunction)() const)
      : InnerMatcher(InnerMatcher), TraverseFunction(TraverseFunction) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    QualType NextNode = (Node.*TraverseFunction)();
    if (NextNode.isNull())
      return false;
    return InnerMatcher.matches(NextNode, Finder, Builder);
  }

private:
  const Matcher<QualType> InnerMatcher;
  QualType (T::*TraverseFunction)() const;
};

/// \brief Matches nodes of type \c T in a ..Loc hierarchy, for which the inner
/// matcher matches on a another node of type \c T that can be reached using a
/// given traverse function.
template <typename T>
class TypeLocTraverseMatcher : public MatcherInterface<T> {
public:
  explicit TypeLocTraverseMatcher(const Matcher<TypeLoc> &InnerMatcher,
                                  TypeLoc (T::*TraverseFunction)() const)
      : InnerMatcher(InnerMatcher), TraverseFunction(TraverseFunction) {}

  virtual bool matches(const T &Node,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    TypeLoc NextNode = (Node.*TraverseFunction)();
    if (!NextNode)
      return false;
    return InnerMatcher.matches(NextNode, Finder, Builder);
  }

private:
  const Matcher<TypeLoc> InnerMatcher;
  TypeLoc (T::*TraverseFunction)() const;
};

template <typename T, typename InnerT>
T makeTypeAllOfComposite(ArrayRef<const Matcher<InnerT> *> InnerMatchers) {
  return T(makeAllOfComposite<InnerT>(InnerMatchers));
}

} // end namespace internal
} // end namespace ast_matchers
} // end namespace clang

#endif // LLVM_CLANG_AST_MATCHERS_AST_MATCHERS_INTERNAL_H
