//===--- OwningMemoryCheck.cpp - clang-tidy--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OwningMemoryCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <string>

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;

namespace clang::tidy::cppcoreguidelines {

namespace {
AST_MATCHER_P(LambdaExpr, hasCallOperator, Matcher<CXXMethodDecl>,
              InnerMatcher) {
  return InnerMatcher.matches(*Node.getCallOperator(), Finder, Builder);
}

AST_MATCHER_P(LambdaExpr, hasLambdaBody, Matcher<Stmt>, InnerMatcher) {
  return InnerMatcher.matches(*Node.getBody(), Finder, Builder);
}
} // namespace

void OwningMemoryCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "LegacyResourceProducers", LegacyResourceProducers);
  Options.store(Opts, "LegacyResourceConsumers", LegacyResourceConsumers);
}

/// Match common cases, where the owner semantic is relevant, like function
/// calls, delete expressions and others.
void OwningMemoryCheck::registerMatchers(MatchFinder *Finder) {
  const auto OwnerDecl = typeAliasTemplateDecl(hasName("::gsl::owner"));
  const auto IsOwnerType = hasType(OwnerDecl);

  const auto LegacyCreatorFunctions =
      hasAnyName(utils::options::parseStringList(LegacyResourceProducers));
  const auto LegacyConsumerFunctions =
      hasAnyName(utils::options::parseStringList(LegacyResourceConsumers));

  // Legacy functions that are use for resource management but cannot be
  // updated to use `gsl::owner<>`, like standard C memory management.
  const auto CreatesLegacyOwner =
      callExpr(callee(functionDecl(LegacyCreatorFunctions)));
  // C-style functions like `::malloc()` sometimes create owners as void*
  // which is expected to be cast to the correct type in C++. This case
  // must be caught explicitly.
  const auto LegacyOwnerCast =
      castExpr(hasSourceExpression(CreatesLegacyOwner));
  // Functions that do manual resource management but cannot be updated to use
  // owner. Best example is `::free()`.
  const auto LegacyOwnerConsumers = functionDecl(LegacyConsumerFunctions);

  const auto CreatesOwner =
      anyOf(cxxNewExpr(),
            callExpr(callee(
                functionDecl(returns(qualType(hasDeclaration(OwnerDecl)))))),
            CreatesLegacyOwner, LegacyOwnerCast);

  const auto ConsideredOwner = eachOf(IsOwnerType, CreatesOwner);
  const auto ScopeDeclaration = anyOf(translationUnitDecl(), namespaceDecl(),
                                      recordDecl(), functionDecl());

  // Find delete expressions that delete non-owners.
  Finder->addMatcher(
      traverse(TK_AsIs,
               cxxDeleteExpr(hasDescendant(declRefExpr(unless(ConsideredOwner))
                                               .bind("deleted_variable")))
                   .bind("delete_expr")),
      this);

  // Ignoring the implicit casts is vital because the legacy owners do not work
  // with the 'owner<>' annotation and therefore always implicitly cast to the
  // legacy type (even 'void *').
  //
  // Furthermore, legacy owner functions are assumed to use raw pointers for
  // resources. This check assumes that all pointer arguments of a legacy
  // functions shall be 'gsl::owner<>'.
  Finder->addMatcher(
      traverse(TK_AsIs, callExpr(callee(LegacyOwnerConsumers),
                                 hasAnyArgument(expr(
                                     unless(ignoringImpCasts(ConsideredOwner)),
                                     hasType(pointerType()))))
                            .bind("legacy_consumer")),
      this);

  // Matching assignment to owners, with the rhs not being an owner nor creating
  // one.
  Finder->addMatcher(
      traverse(TK_AsIs,
               binaryOperator(isAssignmentOperator(), hasLHS(IsOwnerType),
                              hasRHS(unless(ConsideredOwner)))
                   .bind("owner_assignment")),
      this);

  // Matching initialization of owners with non-owners, nor creating owners.
  Finder->addMatcher(
      traverse(TK_AsIs,
               namedDecl(
                   varDecl(hasInitializer(unless(ConsideredOwner)), IsOwnerType)
                       .bind("owner_initialization"))),
      this);

  const auto HasConstructorInitializerForOwner =
      has(cxxConstructorDecl(forEachConstructorInitializer(
          cxxCtorInitializer(
              isMemberInitializer(), forField(IsOwnerType),
              withInitializer(
                  // Avoid templatesdeclaration with
                  // excluding parenListExpr.
                  allOf(unless(ConsideredOwner), unless(parenListExpr()))))
              .bind("owner_member_initializer"))));

  // Match class member initialization that expects owners, but does not get
  // them.
  Finder->addMatcher(
      traverse(TK_AsIs, cxxRecordDecl(HasConstructorInitializerForOwner)),
      this);

  // Matching on assignment operations where the RHS is a newly created owner,
  // but the LHS is not an owner.
  Finder->addMatcher(binaryOperator(isAssignmentOperator(),
                                    hasLHS(unless(IsOwnerType)),
                                    hasRHS(CreatesOwner))
                         .bind("bad_owner_creation_assignment"),
                     this);

  // Matching on initialization operations where the initial value is a newly
  // created owner, but the LHS is not an owner.
  Finder->addMatcher(
      traverse(TK_AsIs, namedDecl(varDecl(hasInitializer(CreatesOwner),
                                          unless(IsOwnerType))
                                      .bind("bad_owner_creation_variable"))),
      this);

  // Match on all function calls that expect owners as arguments, but didn't
  // get them.
  Finder->addMatcher(
      callExpr(forEachArgumentWithParam(
          expr(unless(ConsideredOwner)).bind("expected_owner_argument"),
          parmVarDecl(IsOwnerType))),
      this);

  // Matching for function calls where one argument is a created owner, but the
  // parameter type is not an owner.
  Finder->addMatcher(callExpr(forEachArgumentWithParam(
                         expr(CreatesOwner).bind("bad_owner_creation_argument"),
                         parmVarDecl(unless(IsOwnerType))
                             .bind("bad_owner_creation_parameter"))),
                     this);

  auto IsNotInSubLambda = stmt(
      hasAncestor(
          stmt(anyOf(equalsBoundNode("body"), lambdaExpr())).bind("scope")),
      hasAncestor(stmt(equalsBoundNode("scope"), equalsBoundNode("body"))));

  // Matching on functions, that return an owner/resource, but don't declare
  // their return type as owner.
  Finder->addMatcher(
      functionDecl(
          decl().bind("function_decl"),
          hasBody(
              stmt(stmt().bind("body"),
                   hasDescendant(
                       returnStmt(hasReturnValue(ConsideredOwner),
                                  // Ignore sub-lambda expressions
                                  IsNotInSubLambda,
                                  // Ignore sub-functions
                                  hasAncestor(functionDecl().bind("context")),
                                  hasAncestor(functionDecl(
                                      equalsBoundNode("context"),
                                      equalsBoundNode("function_decl"))))
                           .bind("bad_owner_return")))),
          returns(qualType(unless(hasDeclaration(OwnerDecl))).bind("result"))),
      this);

  // Matching on lambdas, that return an owner/resource, but don't declare
  // their return type as owner.
  Finder->addMatcher(
      lambdaExpr(
          hasAncestor(decl(ScopeDeclaration).bind("scope-decl")),
          hasLambdaBody(
              stmt(stmt().bind("body"),
                   hasDescendant(
                       returnStmt(
                           hasReturnValue(ConsideredOwner),
                           // Ignore sub-lambdas
                           IsNotInSubLambda,
                           // Ignore sub-functions
                           hasAncestor(decl(ScopeDeclaration).bind("context")),
                           hasAncestor(decl(equalsBoundNode("context"),
                                            equalsBoundNode("scope-decl"))))
                           .bind("bad_owner_return")))),
          hasCallOperator(returns(
              qualType(unless(hasDeclaration(OwnerDecl))).bind("result"))))
          .bind("lambda"),
      this);

  // Match on classes that have an owner as member, but don't declare a
  // destructor to properly release the owner.
  Finder->addMatcher(
      cxxRecordDecl(
          has(fieldDecl(IsOwnerType).bind("undestructed_owner_member")),
          anyOf(unless(has(cxxDestructorDecl())),
                has(cxxDestructorDecl(anyOf(isDefaulted(), isDeleted())))))
          .bind("non_destructor_class"),
      this);
}

void OwningMemoryCheck::check(const MatchFinder::MatchResult &Result) {
  const auto &Nodes = Result.Nodes;

  bool CheckExecuted = false;
  CheckExecuted |= handleDeletion(Nodes);
  CheckExecuted |= handleLegacyConsumers(Nodes);
  CheckExecuted |= handleExpectedOwner(Nodes);
  CheckExecuted |= handleAssignmentAndInit(Nodes);
  CheckExecuted |= handleAssignmentFromNewOwner(Nodes);
  CheckExecuted |= handleReturnValues(Nodes);
  CheckExecuted |= handleOwnerMembers(Nodes);

  (void)CheckExecuted;
  assert(CheckExecuted &&
         "None of the subroutines executed, logic error in matcher!");
}

bool OwningMemoryCheck::handleDeletion(const BoundNodes &Nodes) {
  // Result of delete matchers.
  const auto *DeleteStmt = Nodes.getNodeAs<CXXDeleteExpr>("delete_expr");
  const auto *DeletedVariable =
      Nodes.getNodeAs<DeclRefExpr>("deleted_variable");

  // Deletion of non-owners, with `delete variable;`
  if (DeleteStmt) {
    diag(DeleteStmt->getBeginLoc(),
         "deleting a pointer through a type that is "
         "not marked 'gsl::owner<>'; consider using a "
         "smart pointer instead")
        << DeletedVariable->getSourceRange();

    // FIXME: The declaration of the variable that was deleted can be
    // rewritten.
    const ValueDecl *Decl = DeletedVariable->getDecl();
    diag(Decl->getBeginLoc(), "variable declared here", DiagnosticIDs::Note)
        << Decl->getSourceRange();

    return true;
  }
  return false;
}

bool OwningMemoryCheck::handleLegacyConsumers(const BoundNodes &Nodes) {
  // Result of matching for legacy consumer-functions like `::free()`.
  const auto *LegacyConsumer = Nodes.getNodeAs<CallExpr>("legacy_consumer");

  // FIXME: `freopen` should be handled separately because it takes the filename
  // as a pointer, which should not be an owner. The argument that is an owner
  // is known and the false positive coming from the filename can be avoided.
  if (LegacyConsumer) {
    diag(LegacyConsumer->getBeginLoc(),
         "calling legacy resource function without passing a 'gsl::owner<>'")
        << LegacyConsumer->getSourceRange();
    return true;
  }
  return false;
}

bool OwningMemoryCheck::handleExpectedOwner(const BoundNodes &Nodes) {
  // Result of function call matchers.
  const auto *ExpectedOwner = Nodes.getNodeAs<Expr>("expected_owner_argument");

  // Expected function argument to be owner.
  if (ExpectedOwner) {
    diag(ExpectedOwner->getBeginLoc(),
         "expected argument of type 'gsl::owner<>'; got %0")
        << ExpectedOwner->getType() << ExpectedOwner->getSourceRange();
    return true;
  }
  return false;
}

/// Assignment and initialization of owner variables.
bool OwningMemoryCheck::handleAssignmentAndInit(const BoundNodes &Nodes) {
  const auto *OwnerAssignment =
      Nodes.getNodeAs<BinaryOperator>("owner_assignment");
  const auto *OwnerInitialization =
      Nodes.getNodeAs<VarDecl>("owner_initialization");
  const auto *OwnerInitializer =
      Nodes.getNodeAs<CXXCtorInitializer>("owner_member_initializer");

  // Assignments to owners.
  if (OwnerAssignment) {
    diag(OwnerAssignment->getBeginLoc(),
         "expected assignment source to be of type 'gsl::owner<>'; got %0")
        << OwnerAssignment->getRHS()->getType()
        << OwnerAssignment->getSourceRange();
    return true;
  }

  // Initialization of owners.
  if (OwnerInitialization) {
    diag(OwnerInitialization->getBeginLoc(),
         "expected initialization with value of type 'gsl::owner<>'; got %0")
        << OwnerInitialization->getAnyInitializer()->getType()
        << OwnerInitialization->getSourceRange();
    return true;
  }

  // Initializer of class constructors that initialize owners.
  if (OwnerInitializer) {
    diag(OwnerInitializer->getSourceLocation(),
         "expected initialization of owner member variable with value of type "
         "'gsl::owner<>'; got %0")
        // FIXME: the expression from getInit has type 'void', but the type
        // of the supplied argument would be of interest.
        << OwnerInitializer->getInit()->getType()
        << OwnerInitializer->getSourceRange();
    return true;
  }
  return false;
}

/// Problematic assignment and initializations, since the assigned value is a
/// newly created owner.
bool OwningMemoryCheck::handleAssignmentFromNewOwner(const BoundNodes &Nodes) {
  const auto *BadOwnerAssignment =
      Nodes.getNodeAs<BinaryOperator>("bad_owner_creation_assignment");
  const auto *BadOwnerInitialization =
      Nodes.getNodeAs<VarDecl>("bad_owner_creation_variable");

  const auto *BadOwnerArgument =
      Nodes.getNodeAs<Expr>("bad_owner_creation_argument");
  const auto *BadOwnerParameter =
      Nodes.getNodeAs<ParmVarDecl>("bad_owner_creation_parameter");

  // Bad assignments to non-owners, where the RHS is a newly created owner.
  if (BadOwnerAssignment) {
    diag(BadOwnerAssignment->getBeginLoc(),
         "assigning newly created 'gsl::owner<>' to non-owner %0")
        << BadOwnerAssignment->getLHS()->getType()
        << BadOwnerAssignment->getSourceRange();
    return true;
  }

  // Bad initialization of non-owners, where the RHS is a newly created owner.
  if (BadOwnerInitialization) {
    diag(BadOwnerInitialization->getBeginLoc(),
         "initializing non-owner %0 with a newly created 'gsl::owner<>'")
        << BadOwnerInitialization->getType()
        << BadOwnerInitialization->getSourceRange();

    // FIXME: FixitHint to rewrite the type of the initialized variable
    // as 'gsl::owner<OriginalType>'
    return true;
  }

  // Function call, where one arguments is a newly created owner, but the
  // parameter type is not.
  if (BadOwnerArgument) {
    assert(BadOwnerParameter &&
           "parameter for the problematic argument not found");
    diag(BadOwnerArgument->getBeginLoc(), "initializing non-owner argument of "
                                          "type %0 with a newly created "
                                          "'gsl::owner<>'")
        << BadOwnerParameter->getType() << BadOwnerArgument->getSourceRange();
    return true;
  }
  return false;
}

bool OwningMemoryCheck::handleReturnValues(const BoundNodes &Nodes) {
  // Function return statements, that are owners/resources, but the function
  // declaration does not declare its return value as owner.
  const auto *BadReturnType = Nodes.getNodeAs<ReturnStmt>("bad_owner_return");
  const auto *ResultType = Nodes.getNodeAs<QualType>("result");

  // Function return values, that should be owners but aren't.
  if (BadReturnType) {
    // The returned value is a resource or variable that was not annotated with
    // owner<> and the function return type is not owner<>.
    diag(BadReturnType->getBeginLoc(),
         "returning a newly created resource of "
         "type %0 or 'gsl::owner<>' from a "
         "%select{function|lambda}1 whose return type is not 'gsl::owner<>'")
        << *ResultType << (Nodes.getNodeAs<Expr>("lambda") != nullptr)
        << BadReturnType->getSourceRange();

    // FIXME: Rewrite the return type as 'gsl::owner<OriginalType>'
    return true;
  }
  return false;
}

bool OwningMemoryCheck::handleOwnerMembers(const BoundNodes &Nodes) {
  // Classes, that have owners as member, but do not declare destructors
  // accordingly.
  const auto *BadClass = Nodes.getNodeAs<CXXRecordDecl>("non_destructor_class");

  // Classes, that contains owners, but do not declare destructors.
  if (BadClass) {
    const auto *DeclaredOwnerMember =
        Nodes.getNodeAs<FieldDecl>("undestructed_owner_member");
    assert(DeclaredOwnerMember &&
           "match on class with bad destructor but without a declared owner");

    diag(DeclaredOwnerMember->getBeginLoc(),
         "member variable of type 'gsl::owner<>' requires the class %0 to "
         "implement a destructor to release the owned resource")
        << BadClass;
    return true;
  }
  return false;
}

} // namespace clang::tidy::cppcoreguidelines
