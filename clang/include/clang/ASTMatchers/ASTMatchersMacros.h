//===--- ASTMatchersMacros.h - Structural query framework -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Defines macros that enable us to define new matchers in a single place.
//  Since a matcher is a function which returns a Matcher<T> object, where
//  T is the type of the actual implementation of the matcher, the macros allow
//  us to write matchers like functions and take care of the definition of the
//  class boilerplate.
//
//  Note that when you define a matcher with an AST_MATCHER* macro, only the
//  function which creates the matcher goes into the current namespace - the
//  class that implements the actual matcher, which gets returned by the
//  generator function, is put into the 'internal' namespace. This allows us
//  to only have the functions (which is all the user cares about) in the
//  'ast_matchers' namespace and hide the boilerplate.
//
//  To define a matcher in user code, always put it into the clang::ast_matchers
//  namespace and refer to the internal types via the 'internal::':
//
//  namespace clang {
//  namespace ast_matchers {
//  AST_MATCHER_P(MemberExpr, Member,
//                internal::Matcher<ValueDecl>, InnerMatcher) {
//    return InnerMatcher.matches(*Node.getMemberDecl(), Finder, Builder);
//  }
//  } // end namespace ast_matchers
//  } // end namespace clang
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MATCHERS_AST_MATCHERS_MACROS_H
#define LLVM_CLANG_AST_MATCHERS_AST_MATCHERS_MACROS_H

/// \brief AST_MATCHER(Type, DefineMatcher) { ... }
/// defines a zero parameter function named DefineMatcher() that returns a
/// Matcher<Type> object.
///
/// The code between the curly braces has access to the following variables:
///
///   Node:                  the AST node being matched; its type is Type.
///   Finder:                an ASTMatchFinder*.
///   Builder:               a BoundNodesTreeBuilder*.
///
/// The code should return true if 'Node' matches.
#define AST_MATCHER(Type, DefineMatcher)                                       \
  namespace internal {                                                         \
  class matcher_##DefineMatcher##Matcher                                       \
      : public MatcherInterface<Type> {                                        \
   public:                                                                     \
    explicit matcher_##DefineMatcher##Matcher() {}                             \
    virtual bool matches(                                                      \
        const Type &Node, ASTMatchFinder *Finder,                              \
        BoundNodesTreeBuilder *Builder) const;                                 \
  };                                                                           \
  }                                                                            \
  inline internal::Matcher<Type> DefineMatcher() {                             \
    return internal::makeMatcher(                                              \
      new internal::matcher_##DefineMatcher##Matcher());                       \
  }                                                                            \
  inline bool internal::matcher_##DefineMatcher##Matcher::matches(             \
      const Type &Node, ASTMatchFinder *Finder,                                \
      BoundNodesTreeBuilder *Builder) const

/// \brief AST_MATCHER_P(Type, DefineMatcher, ParamType, Param) { ... }
/// defines a single-parameter function named DefineMatcher() that returns a
/// Matcher<Type> object.
///
/// The code between the curly braces has access to the following variables:
///
///   Node:                  the AST node being matched; its type is Type.
///   Param:                 the parameter passed to the function; its type
///                          is ParamType.
///   Finder:                an ASTMatchFinder*.
///   Builder:               a BoundNodesTreeBuilder*.
///
/// The code should return true if 'Node' matches.
#define AST_MATCHER_P(Type, DefineMatcher, ParamType, Param)                   \
  namespace internal {                                                         \
  class matcher_##DefineMatcher##Matcher                                       \
      : public MatcherInterface<Type> {                                        \
   public:                                                                     \
    explicit matcher_##DefineMatcher##Matcher(                                 \
        const ParamType &A##Param) : Param(A##Param) {}                        \
    virtual bool matches(                                                      \
        const Type &Node, ASTMatchFinder *Finder,                              \
        BoundNodesTreeBuilder *Builder) const;                                 \
   private:                                                                    \
    const ParamType Param;                                                     \
  };                                                                           \
  }                                                                            \
  inline internal::Matcher<Type> DefineMatcher(const ParamType &Param) {       \
    return internal::makeMatcher(                                              \
      new internal::matcher_##DefineMatcher##Matcher(Param));                  \
  }                                                                            \
  inline bool internal::matcher_##DefineMatcher##Matcher::matches(             \
      const Type &Node, ASTMatchFinder *Finder,                                \
      BoundNodesTreeBuilder *Builder) const

/// \brief AST_MATCHER_P2(
///     Type, DefineMatcher, ParamType1, Param1, ParamType2, Param2) { ... }
/// defines a two-parameter function named DefineMatcher() that returns a
/// Matcher<Type> object.
///
/// The code between the curly braces has access to the following variables:
///
///   Node:                  the AST node being matched; its type is Type.
///   Param1, Param2:        the parameters passed to the function; their types
///                          are ParamType1 and ParamType2.
///   Finder:                an ASTMatchFinder*.
///   Builder:               a BoundNodesTreeBuilder*.
///
/// The code should return true if 'Node' matches.
#define AST_MATCHER_P2(                                                        \
    Type, DefineMatcher, ParamType1, Param1, ParamType2, Param2)               \
  namespace internal {                                                         \
  class matcher_##DefineMatcher##Matcher                                       \
      : public MatcherInterface<Type> {                                        \
   public:                                                                     \
    matcher_##DefineMatcher##Matcher(                                          \
        const ParamType1 &A##Param1, const ParamType2 &A##Param2)              \
        : Param1(A##Param1), Param2(A##Param2) {}                              \
    virtual bool matches(                                                      \
        const Type &Node, ASTMatchFinder *Finder,                              \
        BoundNodesTreeBuilder *Builder) const;                                 \
   private:                                                                    \
    const ParamType1 Param1;                                                   \
    const ParamType2 Param2;                                                   \
  };                                                                           \
  }                                                                            \
  inline internal::Matcher<Type> DefineMatcher(                                \
      const ParamType1 &Param1, const ParamType2 &Param2) {                    \
    return internal::makeMatcher(                                              \
      new internal::matcher_##DefineMatcher##Matcher(                          \
        Param1, Param2));                                                      \
  }                                                                            \
  inline bool internal::matcher_##DefineMatcher##Matcher::matches(             \
      const Type &Node, ASTMatchFinder *Finder,                                \
      BoundNodesTreeBuilder *Builder) const

/// \brief AST_POLYMORPHIC_MATCHER_P(DefineMatcher, ParamType, Param) { ... }
/// defines a single-parameter function named DefineMatcher() that is
/// polymorphic in the return type.
///
/// The variables are the same as for
/// AST_MATCHER_P, with the addition of NodeType, which specifies the node type
/// of the matcher Matcher<NodeType> returned by the function matcher().
///
/// FIXME: Pull out common code with above macro?
#define AST_POLYMORPHIC_MATCHER_P(DefineMatcher, ParamType, Param)             \
  namespace internal {                                                         \
  template <typename NodeType, typename ParamT>                                \
  class matcher_##DefineMatcher##Matcher                                       \
      : public MatcherInterface<NodeType> {                                    \
   public:                                                                     \
    explicit matcher_##DefineMatcher##Matcher(                                 \
        const ParamType &A##Param) : Param(A##Param) {}                        \
    virtual bool matches(                                                      \
        const NodeType &Node, ASTMatchFinder *Finder,                          \
        BoundNodesTreeBuilder *Builder) const;                                 \
   private:                                                                    \
    const ParamType Param;                                                     \
  };                                                                           \
  }                                                                            \
  inline internal::PolymorphicMatcherWithParam1<                               \
      internal::matcher_##DefineMatcher##Matcher,                              \
      ParamType >                                                              \
    DefineMatcher(const ParamType &Param) {                                    \
    return internal::PolymorphicMatcherWithParam1<                             \
        internal::matcher_##DefineMatcher##Matcher,                            \
        ParamType >(Param);                                                    \
  }                                                                            \
  template <typename NodeType, typename ParamT>                                \
  bool internal::matcher_##DefineMatcher##Matcher<NodeType, ParamT>::matches(  \
      const NodeType &Node, ASTMatchFinder *Finder,                            \
      BoundNodesTreeBuilder *Builder) const

/// \brief AST_POLYMORPHIC_MATCHER_P2(
///     DefineMatcher, ParamType1, Param1, ParamType2, Param2) { ... }
/// defines a two-parameter function named matcher() that is polymorphic in
/// the return type.
///
/// The variables are the same as for AST_MATCHER_P2, with the
/// addition of NodeType, which specifies the node type of the matcher
/// Matcher<NodeType> returned by the function DefineMatcher().
#define AST_POLYMORPHIC_MATCHER_P2(                                            \
      DefineMatcher, ParamType1, Param1, ParamType2, Param2)                   \
  namespace internal {                                                         \
  template <typename NodeType, typename ParamT1, typename ParamT2>             \
  class matcher_##DefineMatcher##Matcher                                       \
      : public MatcherInterface<NodeType> {                                    \
   public:                                                                     \
    matcher_##DefineMatcher##Matcher(                                          \
        const ParamType1 &A##Param1, const ParamType2 &A##Param2)              \
        : Param1(A##Param1), Param2(A##Param2) {}                              \
    virtual bool matches(                                                      \
        const NodeType &Node, ASTMatchFinder *Finder,                          \
        BoundNodesTreeBuilder *Builder) const;                                 \
   private:                                                                    \
    const ParamType1 Param1;                                                   \
    const ParamType2 Param2;                                                   \
  };                                                                           \
  }                                                                            \
  inline internal::PolymorphicMatcherWithParam2<                               \
      internal::matcher_##DefineMatcher##Matcher,                              \
      ParamType1, ParamType2 >                                                 \
    DefineMatcher(const ParamType1 &Param1, const ParamType2 &Param2) {        \
    return internal::PolymorphicMatcherWithParam2<                             \
        internal::matcher_##DefineMatcher##Matcher,                            \
        ParamType1, ParamType2 >(                                              \
        Param1, Param2);                                                       \
  }                                                                            \
  template <typename NodeType, typename ParamT1, typename ParamT2>             \
  bool internal::matcher_##DefineMatcher##Matcher<                             \
      NodeType, ParamT1, ParamT2>::matches(                                    \
      const NodeType &Node, ASTMatchFinder *Finder,                            \
      BoundNodesTreeBuilder *Builder) const

/// \brief Creates a variadic matcher for both a specific \c Type as well as
/// the corresponding \c TypeLoc.
#define AST_TYPE_MATCHER(NodeType, MatcherName)                                \
  const internal::VariadicDynCastAllOfMatcher<Type, NodeType> MatcherName;     \
  const internal::VariadicDynCastAllOfMatcher<TypeLoc,                         \
                                              NodeType##Loc> MatcherName##Loc

/// \brief AST_TYPE_TRAVERSE_MATCHER(MatcherName, FunctionName) defines
/// the matcher \c MatcherName that can be used to traverse from one \c Type
/// to another.
///
/// For a specific \c SpecificType, the traversal is done using 
/// \c SpecificType::FunctionName. The existance of such a function determines
/// whether a corresponding matcher can be used on \c SpecificType.
#define AST_TYPE_TRAVERSE_MATCHER(MatcherName, FunctionName)                   \
class Polymorphic##MatcherName##TypeMatcher {                                  \
public:                                                                        \
  Polymorphic##MatcherName##TypeMatcher(                                       \
      const internal::Matcher<QualType> &InnerMatcher)                         \
    : InnerMatcher(InnerMatcher) {}                                            \
  template <typename T> operator internal::Matcher<T>() {                      \
    return internal::Matcher<T>(new internal::TypeTraverseMatcher<T>(          \
      InnerMatcher, &T::FunctionName));                                        \
  }                                                                            \
private:                                                                       \
  const internal::Matcher<QualType> InnerMatcher;                              \
};                                                                             \
class Variadic##MatcherName##TypeTraverseMatcher                               \
    : public llvm::VariadicFunction<                                           \
        Polymorphic##MatcherName##TypeMatcher,                                 \
        internal::Matcher<QualType>,                                           \
        internal::makeTypeAllOfComposite<                                      \
          Polymorphic##MatcherName##TypeMatcher, QualType> > {                 \
public:                                                                        \
  Variadic##MatcherName##TypeTraverseMatcher() {}                              \
};                                                                             \
const Variadic##MatcherName##TypeTraverseMatcher MatcherName

/// \brief AST_TYPELOC_TRAVERSE_MATCHER(MatcherName, FunctionName) works
/// identical to \c AST_TYPE_TRAVERSE_MATCHER but operates on \c TypeLocs.
#define AST_TYPELOC_TRAVERSE_MATCHER(MatcherName, FunctionName)                \
class Polymorphic##MatcherName##TypeLocMatcher {                               \
public:                                                                        \
  Polymorphic##MatcherName##TypeLocMatcher(                                    \
      const internal::Matcher<TypeLoc> &InnerMatcher)                          \
    : InnerMatcher(InnerMatcher) {}                                            \
  template <typename T> operator internal::Matcher<T>() {                      \
    return internal::Matcher<T>(new internal::TypeLocTraverseMatcher<T>(       \
      InnerMatcher, &T::FunctionName##Loc));                                   \
  }                                                                            \
private:                                                                       \
  const internal::Matcher<TypeLoc> InnerMatcher;                               \
};                                                                             \
class Variadic##MatcherName##TypeLocTraverseMatcher                            \
    : public llvm::VariadicFunction<                                           \
        Polymorphic##MatcherName##TypeLocMatcher,                              \
        internal::Matcher<TypeLoc>,                                            \
        internal::makeTypeAllOfComposite<                                      \
          Polymorphic##MatcherName##TypeLocMatcher, TypeLoc> > {               \
public:                                                                        \
  Variadic##MatcherName##TypeLocTraverseMatcher() {}                           \
};                                                                             \
const Variadic##MatcherName##TypeLocTraverseMatcher MatcherName##Loc;          \
AST_TYPE_TRAVERSE_MATCHER(MatcherName, FunctionName##Type)

#endif // LLVM_CLANG_AST_MATCHERS_AST_MATCHERS_MACROS_H
