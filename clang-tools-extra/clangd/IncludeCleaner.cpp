//===--- IncludeCleaner.cpp - Unused/Missing Headers Analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncludeCleaner.h"
#include "support/Logger.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {
namespace clangd {
namespace {

/// Crawler traverses the AST and feeds in the locations of (sometimes
/// implicitly) used symbols into \p Result.
class ReferencedLocationCrawler
    : public RecursiveASTVisitor<ReferencedLocationCrawler> {
public:
  ReferencedLocationCrawler(ReferencedLocations &Result) : Result(Result) {}

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    add(DRE->getDecl());
    add(DRE->getFoundDecl());
    return true;
  }

  bool VisitMemberExpr(MemberExpr *ME) {
    add(ME->getMemberDecl());
    add(ME->getFoundDecl().getDecl());
    return true;
  }

  bool VisitTagType(TagType *TT) {
    add(TT->getDecl());
    return true;
  }

  bool VisitCXXConstructExpr(CXXConstructExpr *CCE) {
    add(CCE->getConstructor());
    return true;
  }

  bool VisitTemplateSpecializationType(TemplateSpecializationType *TST) {
    if (isNew(TST)) {
      add(TST->getTemplateName().getAsTemplateDecl()); // Primary template.
      add(TST->getAsCXXRecordDecl());                  // Specialization
    }
    return true;
  }

  bool VisitTypedefType(TypedefType *TT) {
    add(TT->getDecl());
    return true;
  }

  // Consider types of any subexpression used, even if the type is not named.
  // This is helpful in getFoo().bar(), where Foo must be complete.
  // FIXME(kirillbobyrev): Should we tweak this? It may not be desirable to
  // consider types "used" when they are not directly spelled in code.
  bool VisitExpr(Expr *E) {
    TraverseType(E->getType());
    return true;
  }

  bool TraverseType(QualType T) {
    if (isNew(T.getTypePtrOrNull())) { // don't care about quals
      Base::TraverseType(T);
    }
    return true;
  }

  bool VisitUsingDecl(UsingDecl *D) {
    for (const auto *Shadow : D->shadows()) {
      add(Shadow->getTargetDecl());
    }
    return true;
  }

private:
  using Base = RecursiveASTVisitor<ReferencedLocationCrawler>;

  void add(const Decl *D) {
    if (!D || !isNew(D->getCanonicalDecl())) {
      return;
    }
    for (const Decl *Redecl : D->redecls()) {
      Result.insert(Redecl->getLocation());
    }
  }

  bool isNew(const void *P) { return P && Visited.insert(P).second; }

  ReferencedLocations &Result;
  llvm::DenseSet<const void *> Visited;
};

} // namespace

ReferencedLocations findReferencedLocations(ParsedAST &AST) {
  ReferencedLocations Result;
  ReferencedLocationCrawler Crawler(Result);
  Crawler.TraverseAST(AST.getASTContext());
  // FIXME(kirillbobyrev): Handle macros.
  return Result;
}

} // namespace clangd
} // namespace clang
