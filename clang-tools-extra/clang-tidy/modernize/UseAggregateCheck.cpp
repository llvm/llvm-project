//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseAggregateCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

/// Check whether \p Ctor is a trivial forwarding constructor: each parameter
/// is used to initialise the corresponding member (in declaration order) and
/// the body is empty.
static bool isTrivialForwardingConstructor(const CXXConstructorDecl *Ctor) {
  if (!Ctor || !Ctor->hasBody())
    return false;

  // Body must be an empty compound statement.
  const auto *Body = dyn_cast<CompoundStmt>(Ctor->getBody());
  if (!Body || !Body->body_empty())
    return false;

  const CXXRecordDecl *Record = Ctor->getParent();

  // Collect non-static data members in declaration order.
  SmallVector<const FieldDecl *, 8> Fields;
  for (const auto *Field : Record->fields())
    Fields.push_back(Field);

  // Number of parameters must match number of fields.
  if (Ctor->getNumParams() != Fields.size())
    return false;

  // Number of member initializers must match number of fields (no base
  // class inits, no extra inits).
  unsigned NumMemberInits = 0;
  for (const auto *Init : Ctor->inits())
    if (Init->isMemberInitializer())
      ++NumMemberInits;
    else
      return false; // base class or delegating init -- bail out
  if (NumMemberInits != Fields.size())
    return false;

  // Walk initializers and check each one initializes the matching field
  // from the matching parameter.
  unsigned FieldIdx = 0;
  for (const auto *Init : Ctor->inits()) {
    if (!Init->isMemberInitializer())
      return false;

    // Must match the field at the current position.
    if (Init->getMember() != Fields[FieldIdx])
      return false;

    const Expr *InitExpr = Init->getInit()->IgnoreImplicit();

    // Handle CXXConstructExpr wrapping the parameter (for class types).
    if (const auto *Construct = dyn_cast<CXXConstructExpr>(InitExpr)) {
      if (Construct->getNumArgs() != 1)
        return false;
      // Must be a copy or move constructor call.
      const CXXConstructorDecl *InitCtor = Construct->getConstructor();
      if (!InitCtor->isCopyOrMoveConstructor())
        return false;
      InitExpr = Construct->getArg(0)->IgnoreImplicit();
    }

    // The init expression must be a DeclRefExpr to the corresponding param.
    const auto *DRE = dyn_cast<DeclRefExpr>(InitExpr);
    if (!DRE)
      return false;
    const auto *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl());
    if (!PVD || PVD != Ctor->getParamDecl(FieldIdx))
      return false;

    ++FieldIdx;
  }

  return true;
}

/// Check whether the class would be a valid aggregate if all user-declared
/// constructors were removed.
static bool canBeAggregate(const CXXRecordDecl *Record,
                           const LangOptions &LangOpts) {
  if (!Record || !Record->hasDefinition())
    return false;

  // Must not have virtual functions.
  if (Record->isPolymorphic())
    return false;

  // Must not have private or protected non-static data members.
  for (const auto *Field : Record->fields())
    if (Field->getAccess() != AS_public)
      return false;

  // Must not have virtual, private, or protected base classes.
  for (const auto &Base : Record->bases()) {
    if (Base.isVirtual())
      return false;
    if (Base.getAccessSpecifier() != AS_public)
      return false;
  }

  // C++17 and later allow non-virtual public base classes in aggregates.
  // Before C++17, aggregates cannot have base classes at all.
  if (!LangOpts.CPlusPlus17 && Record->getNumBases() > 0)
    return false;

  return true;
}

void UseAggregateCheck::registerMatchers(MatchFinder *Finder) {
  // Match class/struct definitions that have at least one user-provided
  // constructor.
  Finder->addMatcher(
      cxxRecordDecl(
          isDefinition(), unless(isImplicit()), unless(isLambda()),
          has(cxxConstructorDecl(isUserProvided(), unless(isDeleted()))))
          .bind("record"),
      this);
}

void UseAggregateCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Record = Result.Nodes.getNodeAs<CXXRecordDecl>("record");
  if (!Record)
    return;

  // Skip records in system headers.
  if (Record->getLocation().isInvalid() ||
      Result.SourceManager->isInSystemHeader(Record->getLocation()))
    return;

  // Skip template specializations to avoid false positives.
  if (isa<ClassTemplateSpecializationDecl>(Record))
    return;

  // Skip if no fields (empty structs are already aggregates by default).
  if (Record->field_empty())
    return;

  // Check aggregate preconditions (ignoring constructors).
  if (!canBeAggregate(Record, getLangOpts()))
    return;

  // Collect all user-declared constructors.
  SmallVector<const CXXConstructorDecl *, 4> UserCtors;
  for (const auto *Decl : Record->decls()) {
    const auto *Ctor = dyn_cast<CXXConstructorDecl>(Decl);
    if (!Ctor || Ctor->isImplicit())
      continue;

    // If there is any user-declared constructor that is not a trivial
    // forwarding constructor and not defaulted/deleted, bail out. We cannot
    // safely suggest removing it.
    if (Ctor->isDeleted() || Ctor->isDefaulted()) {
      // Explicit default/delete still counts as user-declared in C++20
      // aggregate rules, but we focus on user-provided constructors.
      // In C++20 mode, even =default prevents aggregate, so we should
      // flag those too.
      if (getLangOpts().CPlusPlus20)
        UserCtors.push_back(Ctor);
      continue;
    }

    if (!isTrivialForwardingConstructor(Ctor))
      return; // Non-trivial constructor -- not safe to remove

    UserCtors.push_back(Ctor);
  }

  if (UserCtors.empty())
    return;

  // Check that there is no user-provided destructor.
  if (const auto *Dtor = Record->getDestructor())
    if (Dtor->isUserProvided())
      return;

  // Find the primary forwarding constructor to diagnose on.
  const CXXConstructorDecl *PrimaryCtor = nullptr;
  for (const auto *Ctor : UserCtors) {
    if (!Ctor->isDeleted() && !Ctor->isDefaulted()) {
      PrimaryCtor = Ctor;
      break;
    }
  }

  if (!PrimaryCtor)
    return;

  // Emit diagnostic on the class, with a note on the constructor.
  diag(Record->getLocation(),
       "'%0' can be an aggregate type if the forwarding constructor "
       "is removed")
      << Record->getName();
  diag(PrimaryCtor->getLocation(),
       "remove this constructor to enable aggregate initialization",
       DiagnosticIDs::Note);
}

} // namespace clang::tidy::modernize
