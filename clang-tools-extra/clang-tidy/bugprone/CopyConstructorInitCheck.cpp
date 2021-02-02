//===--- CopyConstructorInitCheck.cpp - clang-tidy-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CopyConstructorInitCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

void CopyConstructorInitCheck::registerMatchers(MatchFinder *Finder) {
  // In the future this might be extended to move constructors?
  Finder->addMatcher(
      cxxConstructorDecl(
          isCopyConstructor(),
          hasAnyConstructorInitializer(cxxCtorInitializer(
              isBaseInitializer(),
              withInitializer(cxxConstructExpr(hasDeclaration(
                  cxxConstructorDecl(isDefaultConstructor())))))),
          unless(isInstantiated()))
          .bind("ctor"),
      this);
}

void CopyConstructorInitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructorDecl>("ctor");
  std::string ParamName = Ctor->getParamDecl(0)->getNameAsString();

  // We want only one warning (and FixIt) for each ctor.
  std::string FixItInitList;
  bool HasRelevantBaseInit = false;
  bool ShouldNotDoFixit = false;
  bool HasWrittenInitializer = false;
  SmallVector<FixItHint, 2> SafeFixIts;
  for (const auto *Init : Ctor->inits()) {
    bool CtorInitIsWritten = Init->isWritten();
    HasWrittenInitializer = HasWrittenInitializer || CtorInitIsWritten;
    if (!Init->isBaseInitializer())
      continue;
    const Type *BaseType = Init->getBaseClass();
    // Do not do fixits if there is a type alias involved or one of the bases
    // are explicitly initialized. In the latter case we not do fixits to avoid
    // -Wreorder warnings.
    if (const auto *TempSpecTy = dyn_cast<TemplateSpecializationType>(BaseType))
      ShouldNotDoFixit = ShouldNotDoFixit || TempSpecTy->isTypeAlias();
    ShouldNotDoFixit = ShouldNotDoFixit || isa<TypedefType>(BaseType);
    ShouldNotDoFixit = ShouldNotDoFixit || CtorInitIsWritten;
    const CXXRecordDecl *BaseClass =
        BaseType->getAsCXXRecordDecl()->getDefinition();
    if (BaseClass->field_empty() &&
        BaseClass->forallBases(
            [](const CXXRecordDecl *Class) { return Class->field_empty(); }))
      continue;
    bool NonCopyableBase = false;
    for (const auto *Ctor : BaseClass->ctors()) {
      if (Ctor->isCopyConstructor() &&
          (Ctor->getAccess() == AS_private || Ctor->isDeleted())) {
        NonCopyableBase = true;
        break;
      }
    }
    if (NonCopyableBase)
      continue;
    const auto *CExpr = dyn_cast<CXXConstructExpr>(Init->getInit());
    if (!CExpr || !CExpr->getConstructor()->isDefaultConstructor())
      continue;
    HasRelevantBaseInit = true;
    if (CtorInitIsWritten) {
      if (!ParamName.empty())
        SafeFixIts.push_back(
            FixItHint::CreateInsertion(CExpr->getEndLoc(), ParamName));
    } else {
      if (Init->getSourceLocation().isMacroID() ||
          Ctor->getLocation().isMacroID() || ShouldNotDoFixit)
        break;
      FixItInitList += BaseClass->getNameAsString();
      FixItInitList += "(" + ParamName + "), ";
    }
  }
  if (!HasRelevantBaseInit)
    return;

  auto Diag = diag(Ctor->getLocation(),
                   "calling a base constructor other than the copy constructor")
              << SafeFixIts;

  if (FixItInitList.empty() || ParamName.empty() || ShouldNotDoFixit)
    return;

  std::string FixItMsg{FixItInitList.substr(0, FixItInitList.size() - 2)};
  SourceLocation FixItLoc;
  // There is no initialization list in this constructor.
  if (!HasWrittenInitializer) {
    FixItLoc = Ctor->getBody()->getBeginLoc();
    FixItMsg = " : " + FixItMsg;
  } else {
    // We apply the missing ctors at the beginning of the initialization list.
    FixItLoc = (*Ctor->init_begin())->getSourceLocation();
    FixItMsg += ',';
  }
  FixItMsg += ' ';

  Diag << FixItHint::CreateInsertion(FixItLoc, FixItMsg);
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
