//===--- UseAutoCheck.cpp - clang-tidy-------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UseAutoCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;

namespace clang {
namespace tidy {
namespace modernize {
namespace {

const char IteratorDeclStmtId[] = "iterator_decl";
const char DeclWithNewId[] = "decl_new";
const char DeclWithCastId[] = "decl_cast";
const char DeclWithTemplateCastId[] = "decl_template";

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
///
/// varDecl(hasWrittenNonListInitializer()) maches \c I and \c A but not \c B
/// or \c C.
AST_MATCHER(VarDecl, hasWrittenNonListInitializer) {
  const Expr *Init = Node.getAnyInitializer();
  if (!Init)
    return false;

  Init = Init->IgnoreImplicit();

  // The following test is based on DeclPrinter::VisitVarDecl() to find if an
  // initializer is implicit or not.
  if (const auto *Construct = dyn_cast<CXXConstructExpr>(Init)) {
    return !Construct->isListInitialization() && Construct->getNumArgs() > 0 &&
           !Construct->getArg(0)->isDefaultArgument();
  }
  return Node.getInitStyle() != VarDecl::ListInit;
}

/// \brief Matches QualTypes that are type sugar for QualTypes that match \c
/// SugarMatcher.
///
/// Given
/// \code
///   class C {};
///   typedef C my_type;
///   typedef my_type my_other_type;
/// \endcode
///
/// qualType(isSugarFor(recordType(hasDeclaration(namedDecl(hasName("C"))))))
/// matches \c my_type and \c my_other_type.
AST_MATCHER_P(QualType, isSugarFor, Matcher<QualType>, SugarMatcher) {
  QualType QT = Node;
  while (true) {
    if (SugarMatcher.matches(QT, Finder, Builder))
      return true;

    QualType NewQT = QT.getSingleStepDesugaredType(Finder->getASTContext());
    if (NewQT == QT)
      return false;
    QT = NewQT;
  }
}

/// \brief Matches named declarations that have one of the standard iterator
/// names: iterator, reverse_iterator, const_iterator, const_reverse_iterator.
///
/// Given
/// \code
///   iterator I;
///   const_iterator CI;
/// \endcode
///
/// namedDecl(hasStdIteratorName()) matches \c I and \c CI.
AST_MATCHER(NamedDecl, hasStdIteratorName) {
  static const char *const IteratorNames[] = {"iterator", "reverse_iterator",
                                              "const_iterator",
                                              "const_reverse_iterator"};

  for (const char *Name : IteratorNames) {
    if (hasName(Name).matches(Node, Finder, Builder))
      return true;
  }
  return false;
}

/// \brief Matches named declarations that have one of the standard container
/// names.
///
/// Given
/// \code
///   class vector {};
///   class forward_list {};
///   class my_ver{};
/// \endcode
///
/// recordDecl(hasStdContainerName()) matches \c vector and \c forward_list
/// but not \c my_vec.
AST_MATCHER(NamedDecl, hasStdContainerName) {
  static const char *const ContainerNames[] = {
      "array",         "deque",
      "forward_list",  "list",
      "vector",

      "map",           "multimap",
      "set",           "multiset",

      "unordered_map", "unordered_multimap",
      "unordered_set", "unordered_multiset",

      "queue",         "priority_queue",
      "stack"};

  for (const char *Name : ContainerNames) {
    if (hasName(Name).matches(Node, Finder, Builder))
      return true;
  }
  return false;
}

/// Matches declarations whose declaration context is the C++ standard library
/// namespace std.
///
/// Note that inline namespaces are silently ignored during the lookup since
/// both libstdc++ and libc++ are known to use them for versioning purposes.
///
/// Given:
/// \code
///   namespace ns {
///     struct my_type {};
///     using namespace std;
///   }
///
///   using std::vector;
///   using ns:my_type;
///   using ns::list;
/// \code
///
/// usingDecl(hasAnyUsingShadowDecl(hasTargetDecl(isFromStdNamespace())))
/// matches "using std::vector" and "using ns::list".
AST_MATCHER(Decl, isFromStdNamespace) {
  const DeclContext *D = Node.getDeclContext();

  while (D->isInlineNamespace())
    D = D->getParent();

  if (!D->isNamespace() || !D->getParent()->isTranslationUnit())
    return false;

  const IdentifierInfo *Info = cast<NamespaceDecl>(D)->getIdentifier();

  return (Info && Info->isStr("std"));
}

/// Matches declaration reference or member expressions with explicit template
/// arguments.
AST_POLYMORPHIC_MATCHER(hasExplicitTemplateArgs,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(DeclRefExpr,
                                                        MemberExpr)) {
  return Node.hasExplicitTemplateArgs();
}

/// \brief Returns a DeclarationMatcher that matches standard iterators nested
/// inside records with a standard container name.
DeclarationMatcher standardIterator() {
  return allOf(
      namedDecl(hasStdIteratorName()),
      hasDeclContext(recordDecl(hasStdContainerName(), isFromStdNamespace())));
}

/// \brief Returns a TypeMatcher that matches typedefs for standard iterators
/// inside records with a standard container name.
TypeMatcher typedefIterator() {
  return typedefType(hasDeclaration(standardIterator()));
}

/// \brief Returns a TypeMatcher that matches records named for standard
/// iterators nested inside records named for standard containers.
TypeMatcher nestedIterator() {
  return recordType(hasDeclaration(standardIterator()));
}

/// \brief Returns a TypeMatcher that matches types declared with using
/// declarations and which name standard iterators for standard containers.
TypeMatcher iteratorFromUsingDeclaration() {
  auto HasIteratorDecl = hasDeclaration(namedDecl(hasStdIteratorName()));
  // Types resulting from using declarations are represented by elaboratedType.
  return elaboratedType(allOf(
      // Unwrap the nested name specifier to test for one of the standard
      // containers.
      hasQualifier(specifiesType(templateSpecializationType(hasDeclaration(
          namedDecl(hasStdContainerName(), isFromStdNamespace()))))),
      // the named type is what comes after the final '::' in the type. It
      // should name one of the standard iterator names.
      namesType(
          anyOf(typedefType(HasIteratorDecl), recordType(HasIteratorDecl)))));
}

/// \brief This matcher returns declaration statements that contain variable
/// declarations with written non-list initializer for standard iterators.
StatementMatcher makeIteratorDeclMatcher() {
  return declStmt(unless(has(
                      varDecl(anyOf(unless(hasWrittenNonListInitializer()),
                                    unless(hasType(isSugarFor(anyOf(
                                        typedefIterator(), nestedIterator(),
                                        iteratorFromUsingDeclaration())))))))))
      .bind(IteratorDeclStmtId);
}

StatementMatcher makeDeclWithNewMatcher() {
  return declStmt(
             unless(has(varDecl(anyOf(
                 unless(hasInitializer(ignoringParenImpCasts(cxxNewExpr()))),
                 // FIXME: TypeLoc information is not reliable where CV
                 // qualifiers are concerned so these types can't be
                 // handled for now.
                 hasType(pointerType(
                     pointee(hasCanonicalType(hasLocalQualifiers())))),

                 // FIXME: Handle function pointers. For now we ignore them
                 // because the replacement replaces the entire type
                 // specifier source range which includes the identifier.
                 hasType(pointsTo(
                     pointsTo(parenType(innerType(functionType()))))))))))
      .bind(DeclWithNewId);
}

StatementMatcher makeDeclWithCastMatcher() {
  return declStmt(
             unless(has(varDecl(unless(hasInitializer(explicitCastExpr()))))))
      .bind(DeclWithCastId);
}

StatementMatcher makeDeclWithTemplateCastMatcher() {
  auto ST =
      substTemplateTypeParmType(hasReplacementType(equalsBoundNode("arg")));

  auto ExplicitCall =
      anyOf(has(memberExpr(hasExplicitTemplateArgs())),
            has(ignoringImpCasts(declRefExpr(hasExplicitTemplateArgs()))));

  auto TemplateArg =
      hasTemplateArgument(0, refersToType(qualType().bind("arg")));

  auto TemplateCall = callExpr(
      ExplicitCall,
      callee(functionDecl(TemplateArg,
                          returns(anyOf(ST, pointsTo(ST), references(ST))))));

  return declStmt(unless(has(varDecl(
                      unless(hasInitializer(ignoringImplicit(TemplateCall)))))))
      .bind(DeclWithTemplateCastId);
}

StatementMatcher makeCombinedMatcher() {
  return declStmt(
      // At least one varDecl should be a child of the declStmt to ensure
      // it's a declaration list and avoid matching other declarations,
      // e.g. using directives.
      has(varDecl(unless(isImplicit()))),
      // Skip declarations that are already using auto.
      unless(has(varDecl(anyOf(hasType(autoType()),
                               hasType(qualType(hasDescendant(autoType()))))))),
      anyOf(makeIteratorDeclMatcher(), makeDeclWithNewMatcher(),
            makeDeclWithCastMatcher(), makeDeclWithTemplateCastMatcher()));
}

} // namespace

UseAutoCheck::UseAutoCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      RemoveStars(Options.get("RemoveStars", 0)) {}

void UseAutoCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "RemoveStars", RemoveStars ? 1 : 0);
}

void UseAutoCheck::registerMatchers(MatchFinder *Finder) {
  // Only register the matchers for C++; the functionality currently does not
  // provide any benefit to other languages, despite being benign.
  if (getLangOpts().CPlusPlus) {
    Finder->addMatcher(makeCombinedMatcher(), this);
  }
}

void UseAutoCheck::replaceIterators(const DeclStmt *D, ASTContext *Context) {
  for (const auto *Dec : D->decls()) {
    const auto *V = cast<VarDecl>(Dec);
    const Expr *ExprInit = V->getInit();

    // Skip expressions with cleanups from the intializer expression.
    if (const auto *E = dyn_cast<ExprWithCleanups>(ExprInit))
      ExprInit = E->getSubExpr();

    const auto *Construct = dyn_cast<CXXConstructExpr>(ExprInit);
    if (!Construct)
      continue;

    // Ensure that the constructor receives a single argument.
    if (Construct->getNumArgs() != 1)
      return;

    // Drill down to the as-written initializer.
    const Expr *E = (*Construct->arg_begin())->IgnoreParenImpCasts();
    if (E != E->IgnoreConversionOperator()) {
      // We hit a conversion operator. Early-out now as they imply an implicit
      // conversion from a different type. Could also mean an explicit
      // conversion from the same type but that's pretty rare.
      return;
    }

    if (const auto *NestedConstruct = dyn_cast<CXXConstructExpr>(E)) {
      // If we ran into an implicit conversion contructor, can't convert.
      //
      // FIXME: The following only checks if the constructor can be used
      // implicitly, not if it actually was. Cases where the converting
      // constructor was used explicitly won't get converted.
      if (NestedConstruct->getConstructor()->isConvertingConstructor(false))
        return;
    }
    if (!Context->hasSameType(V->getType(), E->getType()))
      return;
  }

  // Get the type location using the first declaration.
  const auto *V = cast<VarDecl>(*D->decl_begin());

  // WARNING: TypeLoc::getSourceRange() will include the identifier for things
  // like function pointers. Not a concern since this action only works with
  // iterators but something to keep in mind in the future.

  SourceRange Range(V->getTypeSourceInfo()->getTypeLoc().getSourceRange());
  diag(Range.getBegin(), "use auto when declaring iterators")
      << FixItHint::CreateReplacement(Range, "auto");
}

void UseAutoCheck::replaceExpr(const DeclStmt *D, ASTContext *Context,
                               std::function<QualType(const Expr *)> GetType,
                               StringRef Message) {
  const auto *FirstDecl = dyn_cast<VarDecl>(*D->decl_begin());
  // Ensure that there is at least one VarDecl within the DeclStmt.
  if (!FirstDecl)
    return;

  const QualType FirstDeclType = FirstDecl->getType().getCanonicalType();

  std::vector<FixItHint> StarRemovals;
  for (const auto *Dec : D->decls()) {
    const auto *V = cast<VarDecl>(Dec);
    // Ensure that every DeclStmt child is a VarDecl.
    if (!V)
      return;

    const auto *Expr = V->getInit()->IgnoreParenImpCasts();
    // Ensure that every VarDecl has an initializer.
    if (!Expr)
      return;

    // If VarDecl and Initializer have mismatching unqualified types.
    if (!Context->hasSameUnqualifiedType(V->getType(), GetType(Expr)))
      return;

    // All subsequent variables in this declaration should have the same
    // canonical type.  For example, we don't want to use `auto` in
    // `T *p = new T, **pp = new T*;`.
    if (FirstDeclType != V->getType().getCanonicalType())
      return;

    if (RemoveStars) {
      // Remove explicitly written '*' from declarations where there's more than
      // one declaration in the declaration list.
      if (Dec == *D->decl_begin())
        continue;

      auto Q = V->getTypeSourceInfo()->getTypeLoc().getAs<PointerTypeLoc>();
      while (!Q.isNull()) {
        StarRemovals.push_back(FixItHint::CreateRemoval(Q.getStarLoc()));
        Q = Q.getNextTypeLoc().getAs<PointerTypeLoc>();
      }
    }
  }

  // FIXME: There is, however, one case we can address: when the VarDecl pointee
  // is the same as the initializer, just more CV-qualified. However, TypeLoc
  // information is not reliable where CV qualifiers are concerned so we can't
  // do anything about this case for now.
  TypeLoc Loc = FirstDecl->getTypeSourceInfo()->getTypeLoc();
  if (!RemoveStars) {
    while (Loc.getTypeLocClass() == TypeLoc::Pointer ||
           Loc.getTypeLocClass() == TypeLoc::Qualified)
      Loc = Loc.getNextTypeLoc();
  }
  while (Loc.getTypeLocClass() == TypeLoc::LValueReference ||
         Loc.getTypeLocClass() == TypeLoc::RValueReference ||
         Loc.getTypeLocClass() == TypeLoc::Qualified) {
    Loc = Loc.getNextTypeLoc();
  }
  SourceRange Range(Loc.getSourceRange());
  auto Diag = diag(Range.getBegin(), Message);

  // Space after 'auto' to handle cases where the '*' in the pointer type is
  // next to the identifier. This avoids changing 'int *p' into 'autop'.
  // FIXME: This doesn't work for function pointers because the variable name
  // is inside the type.
  Diag << FixItHint::CreateReplacement(Range, RemoveStars ? "auto " : "auto")
       << StarRemovals;
}

void UseAutoCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Decl = Result.Nodes.getNodeAs<DeclStmt>(IteratorDeclStmtId)) {
    replaceIterators(Decl, Result.Context);
  } else if (const auto *Decl =
                 Result.Nodes.getNodeAs<DeclStmt>(DeclWithNewId)) {
    replaceExpr(Decl, Result.Context,
                [](const Expr *Expr) { return Expr->getType(); },
                "use auto when initializing with new to avoid "
                "duplicating the type name");
  } else if (const auto *Decl =
                 Result.Nodes.getNodeAs<DeclStmt>(DeclWithCastId)) {
    replaceExpr(
        Decl, Result.Context,
        [](const Expr *Expr) {
          return cast<ExplicitCastExpr>(Expr)->getTypeAsWritten();
        },
        "use auto when initializing with a cast to avoid duplicating the type "
        "name");
  } else if (const auto *Decl =
                 Result.Nodes.getNodeAs<DeclStmt>(DeclWithTemplateCastId)) {
    replaceExpr(
        Decl, Result.Context,
        [](const Expr *Expr) {
          return cast<CallExpr>(Expr->IgnoreImplicit())
              ->getDirectCallee()
              ->getReturnType();
        },
        "use auto when initializing with a template cast to avoid duplicating "
        "the type name");
  } else {
    llvm_unreachable("Bad Callback. No node provided.");
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
