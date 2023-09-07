//===--- CppInitClassMembersCheck.cpp - clang-tidy ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "CppInitClassMembersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::ast_matchers;

namespace clang::tidy::google {

namespace {

// Matches records that have a default constructor.
AST_MATCHER(CXXRecordDecl, hasDefaultConstructor) {
  return Node.hasDefaultConstructor();
}

// Returns the names of `Fields` in a comma separated string.
std::string
toCommaSeparatedString(const ArrayRef<const FieldDecl *> &Fields) {
  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  llvm::interleave(
      Fields, OS, [&OS](const FieldDecl *Decl) { OS << Decl->getName(); },
      ", ");
  return Buffer;
}

// Returns `true` for types that have uninitialized values by default. For
// example, returns `true` for `int` because an uninitialized `int` field or
// local variable can contain uninitialized values.
bool isDefaultValueUninitialized(QualType Ty) {
  if (Ty.isNull())
    return false;

  // FIXME: For now, this check focuses on several allowlisted types. We will
  // expand coverage in future.
  return Ty->isIntegerType() || Ty->isBooleanType();
}

} // anonymous namespace

void CppInitClassMembersCheck::checkMissingMemberInitializer(
    ASTContext &Context, const CXXRecordDecl &ClassDecl,
    const CXXConstructorDecl *Ctor) {
  SmallVector<const FieldDecl *, 16> FieldsToReport;

  for (const FieldDecl *F : ClassDecl.fields()) {
    if (isDefaultValueUninitialized(F->getType()) &&
        !F->hasInClassInitializer())
      FieldsToReport.push_back(F);
  }

  if (FieldsToReport.empty())
    return;

  DiagnosticBuilder Diag =
      diag(Ctor ? Ctor->getBeginLoc() : ClassDecl.getLocation(),
           "%select{these fields should be initialized|constructor should "
           "initialize these fields}0: %1")
      << (Ctor != nullptr) << toCommaSeparatedString(FieldsToReport);

  // FIXME: generate fixes.
}

void CppInitClassMembersCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(cxxConstructorDecl(isDefinition(), isDefaultConstructor(),
                                        unless(isUserProvided()))
                         .bind("ctor"),
                     this);

  Finder->addMatcher(cxxRecordDecl(isDefinition(), hasDefaultConstructor(),
                                   unless(isInstantiated()),
                                   unless(has(cxxConstructorDecl())))
                         .bind("record"),
                     this);
}

void CppInitClassMembersCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructorDecl>("ctor")) {
    checkMissingMemberInitializer(*Result.Context, *Ctor->getParent(), Ctor);
  } else if (const auto *Record =
                 Result.Nodes.getNodeAs<CXXRecordDecl>("record")) {
    // For a record, perform the same action as for a constructor. However,
    // emit the diagnostic for the record, not for the constructor.
    checkMissingMemberInitializer(*Result.Context, *Record, nullptr);
  }
}

} // namespace clang::tidy::google

