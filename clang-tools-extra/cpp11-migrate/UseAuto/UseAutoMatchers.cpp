//===-- UseAutoMatchers.cpp - Matchers for use-auto transform -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  \brief This file contains the implementation for matcher-generating
///  functions and custom AST_MATCHERs.
///
//===----------------------------------------------------------------------===//
#include "UseAutoMatchers.h"
#include "clang/AST/ASTContext.h"

using namespace clang::ast_matchers;
using namespace clang;

const char *IteratorDeclId = "iterator_decl";
const char *DeclWithNewId = "decl_new";
const char *NewExprId = "new_expr";

namespace clang {
namespace ast_matchers {

/// \brief Matches variable declarations that have explicit initializers that
/// are not initializer lists.
///
/// Given
/// \code
///   iterator I = Container.begin();
///   MyType A(42);
///   MyType B{2};
///   MyType C;
/// \endcode
/// varDecl(hasWrittenNonListInitializer()) matches \c I and \c A but not \c B
/// or \c C.
AST_MATCHER(VarDecl, hasWrittenNonListInitializer) {
  const Expr *Init = Node.getAnyInitializer();
  if (!Init)
    return false;

  // The following test is based on DeclPrinter::VisitVarDecl() to find if an
  // initializer is implicit or not.
  bool ImplicitInit = false;
  if (const CXXConstructExpr *Construct = dyn_cast<CXXConstructExpr>(Init)) {
    if (Construct->isListInitialization())
      return false;
    ImplicitInit = Construct->getNumArgs() == 0 ||
                   Construct->getArg(0)->isDefaultArgument();
  } else
    if (Node.getInitStyle() == VarDecl::ListInit)
      return false;

  return !ImplicitInit;
}

/// \brief Matches QualTypes that are type sugar for QualTypes that match \c
/// SugarMatcher.
///
/// Given
/// \code
///   class C {};
///   typedef C my_type
///   typedef my_type my_other_type;
/// \endcode
///
/// \c qualType(isSugarFor(recordType(hasDeclaration(namedDecl(hasName("C"))))))
/// matches \c my_type and \c my_other_type.
AST_MATCHER_P(QualType, isSugarFor, internal::Matcher<QualType>, SugarMatcher) {
  QualType QT = Node;
  for (;;) {
    if (SugarMatcher.matches(QT, Finder, Builder))
      return true;

    QualType NewQT = QT.getSingleStepDesugaredType(Finder->getASTContext());
    if (NewQT == QT)
      break;
    QT = NewQT;
  }
  return false;
}

/// \brief Matches named declarations that have one of the standard iterator
/// names: iterator, reverse_iterator, const_iterator, const_reverse_iterator.
///
/// Given
/// \code
/// iterator I;
/// const_iterator CI;
/// \endcode
///
/// \c namedDecl(hasStdIteratorName()) matches \c I and \c CI.
AST_MATCHER(NamedDecl, hasStdIteratorName) {
  static const char *IteratorNames[] = {
    "iterator",
    "reverse_iterator",
    "const_iterator",
    "const_reverse_iterator"
  };

  for (unsigned int i = 0;
       i < llvm::array_lengthof(IteratorNames);
       ++i) {
    if (hasName(IteratorNames[i]).matches(Node, Finder, Builder))
      return true;
  }
  return false;
}

/// \brief Matches named declarations that have one of the standard container
/// names.
///
/// Given
/// \code
/// class vector {};
/// class forward_list {};
/// class my_vec {};
/// \endcode
///
/// \c recordDecl(hasStdContainerName()) matches \c vector and \c forward_list
/// but not \c my_vec.
AST_MATCHER_P(NamedDecl, hasStdContainerName, bool, WithStd) {
  static const char *ContainerNames[] = {
    "array",
    "deque",
    "forward_list",
    "list",
    "vector",

    "map",
    "multimap",
    "set",
    "multiset",

    "unordered_map",
    "unordered_multimap",
    "unordered_set",
    "unordered_multiset",

    "queue",
    "priority_queue",
    "stack"
  };

  for (unsigned int i = 0;
       i < llvm::array_lengthof(ContainerNames);
       ++i) {
    std::string Name(ContainerNames[i]);
    if (WithStd)
      Name = "std::" + Name;
    if (hasName(Name).matches(Node, Finder, Builder))
      return true;
  }
  return false;
}

} // namespace ast_matchers
} // namespace clang

namespace {
// \brief Returns a TypeMatcher that matches typedefs for standard iterators
// inside records with a standard container name.
TypeMatcher typedefIterator() {
  return typedefType(
           hasDeclaration(
             allOf(
               namedDecl(hasStdIteratorName()),
               hasDeclContext(
                 recordDecl(hasStdContainerName(true))
               )
             )
           )
         );
}

// \brief Returns a TypeMatcher that matches records named for standard
// iterators nested inside records named for standard containers.
TypeMatcher nestedIterator() {
  return recordType(
           hasDeclaration(
             allOf(
               namedDecl(hasStdIteratorName()),
               hasDeclContext(
                 recordDecl(hasStdContainerName(true))
               )
             )
           )
         );
}

// \brief Returns a TypeMatcher that matches types declared with using
// declarations and which name standard iterators for standard containers.
TypeMatcher iteratorFromUsingDeclaration() {
  // Types resulting from using declarations are
  // represented by ElaboratedType.
  return elaboratedType(
           allOf(
             // Unwrap the nested name specifier to test for
             // one of the standard containers.
             hasQualifier(allOf(
               specifiesType(
                 templateSpecializationType(
                   hasDeclaration(
                     namedDecl(hasStdContainerName(false))
                   )
                 )
               ),
               hasPrefix(
                 specifiesNamespace(hasName("std"))
               )
             )),
             // The named type is what comes after the final
             // '::' in the type. It should name one of the
             // standard iterator names.
             namesType(anyOf(
               typedefType(
                 hasDeclaration(
                   namedDecl(hasStdIteratorName())
                 )
               ),
               recordType(
                 hasDeclaration(
                   namedDecl(hasStdIteratorName())
                 )
               )
             ))
           )
         );
}
} // namespace

DeclarationMatcher makeIteratorDeclMatcher() {
  return varDecl(allOf(
                   hasWrittenNonListInitializer(),
                   unless(hasType(autoType())),
                   hasType(
                     isSugarFor(
                       anyOf(
                         typedefIterator(),
                         nestedIterator(),
                         iteratorFromUsingDeclaration()
                       )
                     )
                   )
                 )).bind(IteratorDeclId);
}

DeclarationMatcher makeDeclWithNewMatcher() {
  return varDecl(
           hasInitializer(
             ignoringParenImpCasts(
               newExpr().bind(NewExprId)
             )
           ),

           // FIXME: TypeLoc information is not reliable where CV qualifiers are
           // concerned so these types can't be handled for now.
           unless(hasType(pointerType(pointee(hasLocalQualifiers())))),

           // FIXME: Handle function pointers. For now we ignore them because
           // the replacement replaces the entire type specifier source range
           // which includes the identifier.
           unless(
             hasType(
               pointsTo(
                 pointsTo(
                   parenType(innerType(functionType()))
                 )
               )
             )
           )
         ).bind(DeclWithNewId);
}
