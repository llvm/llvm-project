//===--- IncorrectEnableSharedFromThisCheck.cpp - clang-tidy --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncorrectEnableSharedFromThisCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void IncorrectEnableSharedFromThisCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(translationUnitDecl(), this);
}

void IncorrectEnableSharedFromThisCheck::check(
    const MatchFinder::MatchResult &Result) {

  class Visitor : public RecursiveASTVisitor<Visitor> {
    IncorrectEnableSharedFromThisCheck &Check;
    llvm::SmallPtrSet<const CXXRecordDecl *, 16> EnableSharedClassSet;

  public:
    explicit Visitor(IncorrectEnableSharedFromThisCheck &Check)
        : Check(Check) {}

    bool TraverseCXXRecordHelper(CXXRecordDecl *RDecl) {
      if (!RDecl->hasDefinition()) {
        return true;
      }
      for (const auto &Base : RDecl->bases()) {
        VisitCXXBaseSpecifier(Base, RDecl);
      }
      for (const auto &Base : RDecl->bases()) {
        const CXXRecordDecl *BaseType = Base.getType()->getAsCXXRecordDecl();
        if (BaseType && isStdEnableSharedFromThis(BaseType)) {
          EnableSharedClassSet.insert(RDecl->getCanonicalDecl());
          return true;
        }
      }
      return true;
    }

    bool VisitCXXBaseSpecifier(const CXXBaseSpecifier &Base,
                               CXXRecordDecl *RDecl) {
      const CXXRecordDecl *BaseType = Base.getType()->getAsCXXRecordDecl();
      const clang::AccessSpecifier AccessSpec = Base.getAccessSpecifier();

      if (BaseType && isStdEnableSharedFromThis(BaseType)) {
        if (AccessSpec != clang::AS_public) {
          const SourceRange ReplacementRange = Base.getSourceRange();
          const std::string ReplacementString =
              // Base.getType().getAsString() results in
              // std::enable_shared_from_this<ClassName> or alias/typedefs of
              // std::enable_shared_from_this<ClassName>
              "public " + Base.getType().getAsString();
          const FixItHint Hint =
              FixItHint::CreateReplacement(ReplacementRange, ReplacementString);
          Check.diag(
              RDecl->getLocation(),
              "this is not publicly inheriting from "
              "std::enable_shared_from_this, will cause unintended behaviour "
              "on shared_from_this. fix this by making it public inheritance",
              DiagnosticIDs::Warning)
              << Hint;
        }
      } else if (EnableSharedClassSet.contains(BaseType->getCanonicalDecl())) {
        if (AccessSpec != clang::AS_public) {
          const SourceRange ReplacementRange = Base.getSourceRange();
          const std::string ReplacementString =
              "public " + Base.getType().getAsString();
          const FixItHint Hint =
              FixItHint::CreateReplacement(ReplacementRange, ReplacementString);
          Check.diag(
              RDecl->getLocation(),
              "this is not publicly inheriting from "
              "std::enable_shared_from_this, will cause unintended behaviour "
              "on shared_from_this. fix this by making it public inheritance",
              DiagnosticIDs::Warning)
              << Hint;
        }
      }
      return true;
    }

  private:
    // FIXME: configure this for boost in the future
    bool isStdEnableSharedFromThis(const CXXRecordDecl *BaseType) {
      return BaseType->getName() == "enable_shared_from_this" &&
             BaseType->isInStdNamespace();
    }
  };

  Visitor(*this).TraverseAST(*Result.Context);
}

} // namespace clang::tidy::bugprone
