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

    bool VisitCXXRecordDecl(CXXRecordDecl *RDecl) {
      if (!RDecl->hasDefinition())
        return true;

      if (isStdEnableSharedFromThis(RDecl))
        EnableSharedClassSet.insert(RDecl->getCanonicalDecl());

      for (const CXXBaseSpecifier &Base : RDecl->bases()) {
        const CXXRecordDecl *BaseRecord =
            Base.getType()->getAsCXXRecordDecl()->getCanonicalDecl();
        const bool IsStdEnableSharedFromThisBool =
            isStdEnableSharedFromThis(BaseRecord);

        if (EnableSharedClassSet.contains(BaseRecord) ||
            IsStdEnableSharedFromThisBool) {

          if (Base.getAccessSpecifier() != clang::AS_public) {
            const SourceRange ReplacementRange = Base.getSourceRange();
            const std::string ReplacementString =
                // Base.getType().getAsString() results in
                // std::enable_shared_from_this<ClassName> or
                // alias/typedefs of std::enable_shared_from_this<ClassName>
                "public " + Base.getType().getAsString();
            const FixItHint Hint = FixItHint::CreateReplacement(
                ReplacementRange, ReplacementString);
            Check.diag(RDecl->getLocation(),
                       "%2 is not publicly inheriting from "
                       "%select{%1 which inherits from "
                       "'std::enable_shared_from_this',|'std::enable_shared_"
                       "from_this',}0 "
                       "which will cause unintended behaviour "
                       "when using 'shared_from_this'; make the inheritance "
                       "public",
                       DiagnosticIDs::Warning)
                << IsStdEnableSharedFromThisBool << BaseRecord << RDecl << Hint;
          }

          EnableSharedClassSet.insert(RDecl->getCanonicalDecl());
        }
      }
      return true;
    }

  private:
    // FIXME: configure this for boost in the future
    bool isStdEnableSharedFromThis(const CXXRecordDecl *RDecl) {
      // this is the equivalent of
      // RDecl->getQualifiedNameAsString() == "std::enable_shared_from_this"
      return RDecl->getName() == "enable_shared_from_this" &&
             RDecl->isInStdNamespace();
    }
  };

  Visitor(*this).TraverseAST(*Result.Context);
}

} // namespace clang::tidy::bugprone
