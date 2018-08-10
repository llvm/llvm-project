//===--- UseTransparentFunctorsCheck.cpp - clang-tidy----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UseTransparentFunctorsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

UseTransparentFunctorsCheck::UseTransparentFunctorsCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), SafeMode(Options.get("SafeMode", 0)) {}

void UseTransparentFunctorsCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SafeMode", SafeMode ? 1 : 0);
}

void UseTransparentFunctorsCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus14)
    return;

  const auto TransparentFunctors =
      classTemplateSpecializationDecl(
          unless(hasAnyTemplateArgument(refersToType(voidType()))),
          hasAnyName("::std::plus", "::std::minus", "::std::multiplies",
                     "::std::divides", "::std::modulus", "::std::negate",
                     "::std::equal_to", "::std::not_equal_to", "::std::greater",
                     "::std::less", "::std::greater_equal", "::std::less_equal",
                     "::std::logical_and", "::std::logical_or",
                     "::std::logical_not", "::std::bit_and", "::std::bit_or",
                     "::std::bit_xor", "::std::bit_not"))
          .bind("FunctorClass");

  // Non-transparent functor mentioned as a template parameter. FIXIT.
  Finder->addMatcher(
      loc(qualType(
              unless(elaboratedType()),
              hasDeclaration(classTemplateSpecializationDecl(
                  unless(hasAnyTemplateArgument(templateArgument(refersToType(
                      qualType(pointsTo(qualType(isAnyCharacter()))))))),
                  hasAnyTemplateArgument(
                      templateArgument(refersToType(qualType(hasDeclaration(
                                           TransparentFunctors))))
                          .bind("Functor"))))))
          .bind("FunctorParentLoc"),
      this);

  if (SafeMode)
    return;

  // Non-transparent functor constructed. No FIXIT. There is no easy way
  // to rule out the problematic char* vs string case.
  Finder->addMatcher(cxxConstructExpr(hasDeclaration(cxxMethodDecl(
                                          ofClass(TransparentFunctors))),
                                      unless(isInTemplateInstantiation()))
                         .bind("FuncInst"),
                     this);
}

static const StringRef Message = "prefer transparent functors '%0'";

template <typename T> static T getInnerTypeLocAs(TypeLoc Loc) {
  T Result;
  while (Result.isNull() && !Loc.isNull()) {
    Result = Loc.getAs<T>();
    Loc = Loc.getNextTypeLoc();
  }
  return Result;
}

void UseTransparentFunctorsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *FuncClass =
      Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>("FunctorClass");
  if (const auto *FuncInst =
          Result.Nodes.getNodeAs<CXXConstructExpr>("FuncInst")) {
    diag(FuncInst->getBeginLoc(), Message)
        << (FuncClass->getName() + "<>").str();
    return;
  }

  const auto *Functor = Result.Nodes.getNodeAs<TemplateArgument>("Functor");
  const auto FunctorParentLoc =
      Result.Nodes.getNodeAs<TypeLoc>("FunctorParentLoc")
          ->getAs<TemplateSpecializationTypeLoc>();

  if (!FunctorParentLoc)
    return;

  unsigned ArgNum = 0;
  const auto *FunctorParentType =
      FunctorParentLoc.getType()->castAs<TemplateSpecializationType>();
  for (; ArgNum < FunctorParentType->getNumArgs(); ++ArgNum) {
    const TemplateArgument &Arg = FunctorParentType->getArg(ArgNum);
    if (Arg.getKind() != TemplateArgument::Type)
      continue;
    QualType ParentArgType = Arg.getAsType();
    if (ParentArgType->isRecordType() &&
        ParentArgType->getAsCXXRecordDecl() ==
            Functor->getAsType()->getAsCXXRecordDecl())
      break;
  }
  // Functor is a default template argument.
  if (ArgNum == FunctorParentType->getNumArgs())
    return;
  TemplateArgumentLoc FunctorLoc = FunctorParentLoc.getArgLoc(ArgNum);
  auto FunctorTypeLoc = getInnerTypeLocAs<TemplateSpecializationTypeLoc>(
      FunctorLoc.getTypeSourceInfo()->getTypeLoc());
  if (FunctorTypeLoc.isNull())
    return;

  SourceLocation ReportLoc = FunctorLoc.getLocation();
  diag(ReportLoc, Message) << (FuncClass->getName() + "<>").str()
                           << FixItHint::CreateRemoval(
                                  FunctorTypeLoc.getArgLoc(0).getSourceRange());
}

} // namespace modernize
} // namespace tidy
} // namespace clang
