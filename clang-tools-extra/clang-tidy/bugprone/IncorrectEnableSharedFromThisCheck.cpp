//===--- IncorrectEnableSharedFromThisCheck.cpp - clang-tidy --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncorrectEnableSharedFromThisCheck.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void IncorrectEnableSharedFromThisCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(translationUnitDecl(), this);
}

void IncorrectEnableSharedFromThisCheck::check(
  const MatchFinder::MatchResult &Result) {

  class Visitor : public RecursiveASTVisitor<Visitor> {
    IncorrectEnableSharedFromThisCheck &Check;
    llvm::SmallPtrSet<const CXXRecordDecl*, 16> EnableSharedClassSet;

  public:
    explicit Visitor(IncorrectEnableSharedFromThisCheck &Check) : Check(Check), EnableSharedClassSet(EnableSharedClassSet) {}
  
    bool VisitCXXRecordDecl(CXXRecordDecl *RDecl) {
      for (const auto &Base : RDecl->bases()) {
        VisitCXXBaseSpecifier(Base, RDecl);
      }
      for (const auto &Base : RDecl->bases()) {
        const CXXRecordDecl *BaseType = Base.getType()->getAsCXXRecordDecl();
        if (BaseType && BaseType->getQualifiedNameAsString() ==
                            "std::enable_shared_from_this") {
          EnableSharedClassSet.insert(RDecl->getCanonicalDecl());
        }
      }
      return true;     
    }

    bool VisitCXXBaseSpecifier(const CXXBaseSpecifier &Base, CXXRecordDecl *RDecl) {
      const CXXRecordDecl *BaseType = Base.getType()->getAsCXXRecordDecl();
      const clang::AccessSpecifier AccessSpec = Base.getAccessSpecifier();

      if ( BaseType && BaseType->getQualifiedNameAsString() ==
                          "enable_shared_from_this") {
        if (AccessSpec == clang::AS_public) {
          const SourceLocation InsertLocation = Base.getBaseTypeLoc();
          const FixItHint Hint = FixItHint::CreateInsertion(InsertLocation, "std::");
          Check.diag(RDecl->getLocation(),
              "Should be std::enable_shared_from_this", DiagnosticIDs::Warning)
              << Hint;
          
        } else {
          const SourceRange ReplacementRange = Base.getSourceRange();
          const std::string ReplacementString =
              "public std::" + Base.getType().getAsString();
          const FixItHint Hint =
              FixItHint::CreateReplacement(ReplacementRange, ReplacementString);
          Check.diag(RDecl->getLocation(),
              "Should be std::enable_shared_from_this and "
              "inheritance from std::enable_shared_from_this should be public "
              "inheritance, otherwise the internal weak_ptr won't be "
              "initialized",
              DiagnosticIDs::Warning)
              << Hint;
        }
      }
      else if ( BaseType && BaseType->getQualifiedNameAsString() ==
                          "std::enable_shared_from_this") {
        if (AccessSpec != clang::AS_public) {
          const SourceRange ReplacementRange = Base.getSourceRange();
          const std::string ReplacementString =
              "public " + Base.getType().getAsString();
          const FixItHint Hint =
              FixItHint::CreateReplacement(ReplacementRange, ReplacementString);
          Check.diag(
              RDecl->getLocation(),
              "inheritance from std::enable_shared_from_this should be public "
              "inheritance, otherwise the internal weak_ptr won't be initialized",
              DiagnosticIDs::Warning)
              << Hint;
        }
      } 
      else if (EnableSharedClassSet.contains(BaseType->getCanonicalDecl())) {
        if (AccessSpec != clang::AS_public) {
          const SourceRange ReplacementRange = Base.getSourceRange();
          const std::string ReplacementString =
              "public " + Base.getType().getAsString();
          const FixItHint Hint =
              FixItHint::CreateReplacement(ReplacementRange, ReplacementString);
          Check.diag(
              RDecl->getLocation(),
              "inheritance from std::enable_shared_from_this should be public "
              "inheritance, otherwise the internal weak_ptr won't be initialized",
              DiagnosticIDs::Warning)
              << Hint;
        }            
      }
      return true;
    }
  };
 
  Visitor(*this).TraverseAST(*Result.Context);
  
}

} // namespace clang::tidy::bugprone
