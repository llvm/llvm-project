//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SmartPtrArrayMismatchCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

constexpr char ConstructExprN[] = "found_construct_expr";
constexpr char NewExprN[] = "found_new_expr";
constexpr char ConstructorN[] = "found_constructor";

bool isInSingleDeclStmt(const DeclaratorDecl *D) {
  const DynTypedNodeList Parents =
      D->getASTContext().getParentMapContext().getParents(*D);
  for (const DynTypedNode &PNode : Parents)
    if (const auto *PDecl = PNode.get<DeclStmt>())
      return PDecl->isSingleDecl();
  return false;
}

const DeclaratorDecl *getConstructedVarOrField(const Expr *FoundConstructExpr,
                                               ASTContext &Ctx) {
  const DynTypedNodeList ConstructParents =
      Ctx.getParentMapContext().getParents(*FoundConstructExpr);
  if (ConstructParents.size() != 1)
    return nullptr;
  const auto *ParentDecl = ConstructParents.begin()->get<DeclaratorDecl>();
  if (isa_and_nonnull<VarDecl, FieldDecl>(ParentDecl))
    return ParentDecl;

  return nullptr;
}

} // namespace

const char SmartPtrArrayMismatchCheck::PointerTypeN[] = "pointer_type";

SmartPtrArrayMismatchCheck::SmartPtrArrayMismatchCheck(
    StringRef Name, ClangTidyContext *Context, StringRef SmartPointerName)
    : ClangTidyCheck(Name, Context), SmartPointerName(SmartPointerName) {}

void SmartPtrArrayMismatchCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {}

void SmartPtrArrayMismatchCheck::registerMatchers(MatchFinder *Finder) {
  // For both shared and unique pointers, we need to find constructor with
  // exactly one parameter that has the pointer type. Other constructors are
  // not applicable for this check.
  auto FindConstructor =
      cxxConstructorDecl(ofClass(getSmartPointerClassMatcher()),
                         parameterCountIs(1), isExplicit())
          .bind(ConstructorN);
  auto FindConstructExpr =
      cxxConstructExpr(
          hasDeclaration(FindConstructor), argumentCountIs(1),
          hasArgument(0,
                      cxxNewExpr(isArray(),
                                 hasType(hasCanonicalType(pointerType(
                                     pointee(equalsBoundNode(PointerTypeN))))))
                          .bind(NewExprN)))
          .bind(ConstructExprN);
  Finder->addMatcher(FindConstructExpr, this);
}

void SmartPtrArrayMismatchCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *FoundNewExpr = Result.Nodes.getNodeAs<CXXNewExpr>(NewExprN);
  const auto *FoundConstructExpr =
      Result.Nodes.getNodeAs<CXXConstructExpr>(ConstructExprN);
  const auto *FoundConstructorDecl =
      Result.Nodes.getNodeAs<CXXConstructorDecl>(ConstructorN);

  ASTContext &Ctx = FoundConstructorDecl->getASTContext();
  const DeclaratorDecl *VarOrField =
      getConstructedVarOrField(FoundConstructExpr, Ctx);

  auto D = diag(FoundNewExpr->getBeginLoc(),
                "%0 pointer to non-array is initialized with array")
           << SmartPointerName;
  D << FoundNewExpr->getSourceRange();

  if (VarOrField) {
    auto TSTypeLoc = VarOrField->getTypeSourceInfo()
                         ->getTypeLoc()
                         .getAsAdjusted<clang::TemplateSpecializationTypeLoc>();
    assert(TSTypeLoc.getNumArgs() >= 1 &&
           "Matched type should have at least 1 template argument.");

    SourceRange TemplateArgumentRange = TSTypeLoc.getArgLoc(0)
                                            .getTypeSourceInfo()
                                            ->getTypeLoc()
                                            .getSourceRange();
    D << TemplateArgumentRange;

    if (isInSingleDeclStmt(VarOrField)) {
      const SourceManager &SM = Ctx.getSourceManager();
      if (!utils::rangeCanBeFixed(TemplateArgumentRange, &SM))
        return;

      SourceLocation InsertLoc = Lexer::getLocForEndOfToken(
          TemplateArgumentRange.getEnd(), 0, SM, Ctx.getLangOpts());
      D << FixItHint::CreateInsertion(InsertLoc, "[]");
    }
  }
}

} // namespace clang::tidy::bugprone
