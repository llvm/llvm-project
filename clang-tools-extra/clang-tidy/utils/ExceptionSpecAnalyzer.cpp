//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExceptionSpecAnalyzer.h"

#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"

namespace clang::tidy::utils {

ExceptionSpecAnalyzer::State
ExceptionSpecAnalyzer::analyze(const FunctionDecl *FuncDecl) {
  // Check if function exist in cache or add temporary value to cache to protect
  // against endless recursion.
  const auto [CacheEntry, NotFound] =
      FunctionCache.try_emplace(FuncDecl, State::NotThrowing);
  if (NotFound) {
    ExceptionSpecAnalyzer::State State = analyzeImpl(FuncDecl);
    // Update result with calculated value
    FunctionCache[FuncDecl] = State;
    return State;
  }

  return CacheEntry->getSecond();
}

ExceptionSpecAnalyzer::State
ExceptionSpecAnalyzer::analyzeUnresolvedOrDefaulted(
    const CXXMethodDecl *MethodDecl, const FunctionProtoType *FuncProto) {
  if (!FuncProto || !MethodDecl)
    return State::Unknown;

  const DefaultableMemberKind Kind = getDefaultableMemberKind(MethodDecl);

  if (Kind == DefaultableMemberKind::None)
    return State::Unknown;

  return analyzeRecord(MethodDecl->getParent(), Kind, SkipMethods::Yes);
}

ExceptionSpecAnalyzer::State
ExceptionSpecAnalyzer::analyzeFieldDecl(const FieldDecl *FDecl,
                                        DefaultableMemberKind Kind) {
  if (!FDecl)
    return State::Unknown;

  if (const CXXRecordDecl *RecDecl =
          FDecl->getType()->getUnqualifiedDesugaredType()->getAsCXXRecordDecl())
    return analyzeRecord(RecDecl, Kind);

  // Trivial types do not throw
  if (FDecl->getType().isTrivialType(FDecl->getASTContext()))
    return State::NotThrowing;

  return State::Unknown;
}

ExceptionSpecAnalyzer::State
ExceptionSpecAnalyzer::analyzeBase(const CXXBaseSpecifier &Base,
                                   DefaultableMemberKind Kind) {
  const auto *RecType = Base.getType()->getAs<RecordType>();
  if (!RecType)
    return State::Unknown;

  return analyzeRecord(RecType->getAsCXXRecordDecl(), Kind);
}

ExceptionSpecAnalyzer::State
ExceptionSpecAnalyzer::analyzeRecord(const CXXRecordDecl *RecordDecl,
                                     DefaultableMemberKind Kind,
                                     SkipMethods SkipMethods) {
  if (!RecordDecl)
    return State::Unknown;

  // Trivial implies noexcept
  if (hasTrivialMemberKind(RecordDecl, Kind))
    return State::NotThrowing;

  if (SkipMethods == SkipMethods::No)
    for (const auto *MethodDecl : RecordDecl->methods())
      if (getDefaultableMemberKind(MethodDecl) == Kind)
        return analyze(MethodDecl);

  for (const auto &BaseSpec : RecordDecl->bases()) {
    State Result = analyzeBase(BaseSpec, Kind);
    if (Result == State::Throwing || Result == State::Unknown)
      return Result;
  }

  for (const auto &BaseSpec : RecordDecl->vbases()) {
    State Result = analyzeBase(BaseSpec, Kind);
    if (Result == State::Throwing || Result == State::Unknown)
      return Result;
  }

  for (const auto *FDecl : RecordDecl->fields())
    if (!FDecl->isInvalidDecl() && !FDecl->isUnnamedBitField()) {
      State Result = analyzeFieldDecl(FDecl, Kind);
      if (Result == State::Throwing || Result == State::Unknown)
        return Result;
    }

  return State::NotThrowing;
}

ExceptionSpecAnalyzer::State
ExceptionSpecAnalyzer::analyzeImpl(const FunctionDecl *FuncDecl) {
  const auto *FuncProto = FuncDecl->getType()->getAs<FunctionProtoType>();
  if (!FuncProto)
    return State::Unknown;

  const ExceptionSpecificationType EST = FuncProto->getExceptionSpecType();

  if (EST == EST_Unevaluated || (EST == EST_None && FuncDecl->isDefaulted()))
    return analyzeUnresolvedOrDefaulted(cast<CXXMethodDecl>(FuncDecl),
                                        FuncProto);

  return analyzeFunctionEST(FuncDecl, FuncProto);
}

ExceptionSpecAnalyzer::State
ExceptionSpecAnalyzer::analyzeFunctionEST(const FunctionDecl *FuncDecl,
                                          const FunctionProtoType *FuncProto) {
  if (!FuncDecl || !FuncProto)
    return State::Unknown;

  if (isUnresolvedExceptionSpec(FuncProto->getExceptionSpecType()))
    return State::Unknown;

  // A non defaulted destructor without the noexcept specifier is still noexcept
  if (isa<CXXDestructorDecl>(FuncDecl) &&
      FuncDecl->getExceptionSpecType() == EST_None)
    return State::NotThrowing;

  switch (FuncProto->canThrow()) {
  case CT_Cannot:
    return State::NotThrowing;
  case CT_Dependent: {
    const Expr *NoexceptExpr = FuncProto->getNoexceptExpr();
    if (!NoexceptExpr)
      return State::NotThrowing;

    // We can't resolve value dependence so just return unknown
    if (NoexceptExpr->isValueDependent())
      return State::Unknown;

    // Try to evaluate the expression to a boolean value
    bool Result = false;
    if (NoexceptExpr->EvaluateAsBooleanCondition(
            Result, FuncDecl->getASTContext(), true))
      return Result ? State::NotThrowing : State::Throwing;

    // The noexcept expression is not value dependent but we can't evaluate it
    // as a boolean condition so we have no idea if its throwing or not
    return State::Unknown;
  }
  default:
    return State::Throwing;
  };
}

bool ExceptionSpecAnalyzer::hasTrivialMemberKind(const CXXRecordDecl *RecDecl,
                                                 DefaultableMemberKind Kind) {
  if (!RecDecl)
    return false;

  switch (Kind) {
  case DefaultableMemberKind::DefaultConstructor:
    return RecDecl->hasTrivialDefaultConstructor();
  case DefaultableMemberKind::CopyConstructor:
    return RecDecl->hasTrivialCopyConstructor();
  case DefaultableMemberKind::MoveConstructor:
    return RecDecl->hasTrivialMoveConstructor();
  case DefaultableMemberKind::CopyAssignment:
    return RecDecl->hasTrivialCopyAssignment();
  case DefaultableMemberKind::MoveAssignment:
    return RecDecl->hasTrivialMoveAssignment();
  case DefaultableMemberKind::Destructor:
    return RecDecl->hasTrivialDestructor();

  default:
    return false;
  }
}

bool ExceptionSpecAnalyzer::isConstructor(DefaultableMemberKind Kind) {
  switch (Kind) {
  case DefaultableMemberKind::DefaultConstructor:
  case DefaultableMemberKind::CopyConstructor:
  case DefaultableMemberKind::MoveConstructor:
    return true;

  default:
    return false;
  }
}

bool ExceptionSpecAnalyzer::isSpecialMember(DefaultableMemberKind Kind) {
  switch (Kind) {
  case DefaultableMemberKind::DefaultConstructor:
  case DefaultableMemberKind::CopyConstructor:
  case DefaultableMemberKind::MoveConstructor:
  case DefaultableMemberKind::CopyAssignment:
  case DefaultableMemberKind::MoveAssignment:
  case DefaultableMemberKind::Destructor:
    return true;
  default:
    return false;
  }
}

bool ExceptionSpecAnalyzer::isComparison(DefaultableMemberKind Kind) {
  switch (Kind) {
  case DefaultableMemberKind::CompareEqual:
  case DefaultableMemberKind::CompareNotEqual:
  case DefaultableMemberKind::CompareRelational:
  case DefaultableMemberKind::CompareThreeWay:
    return true;
  default:
    return false;
  }
}

ExceptionSpecAnalyzer::DefaultableMemberKind
ExceptionSpecAnalyzer::getDefaultableMemberKind(const FunctionDecl *FuncDecl) {
  if (const auto *MethodDecl = dyn_cast<CXXMethodDecl>(FuncDecl)) {
    if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(FuncDecl)) {
      if (Ctor->isDefaultConstructor())
        return DefaultableMemberKind::DefaultConstructor;

      if (Ctor->isCopyConstructor())
        return DefaultableMemberKind::CopyConstructor;

      if (Ctor->isMoveConstructor())
        return DefaultableMemberKind::MoveConstructor;
    }

    if (MethodDecl->isCopyAssignmentOperator())
      return DefaultableMemberKind::CopyAssignment;

    if (MethodDecl->isMoveAssignmentOperator())
      return DefaultableMemberKind::MoveAssignment;

    if (isa<CXXDestructorDecl>(FuncDecl))
      return DefaultableMemberKind::Destructor;
  }

  const LangOptions &LangOpts = FuncDecl->getLangOpts();

  switch (FuncDecl->getDeclName().getCXXOverloadedOperator()) {
  case OO_EqualEqual:
    return DefaultableMemberKind::CompareEqual;

  case OO_ExclaimEqual:
    return DefaultableMemberKind::CompareNotEqual;

  case OO_Spaceship:
    // No point allowing this if <=> doesn't exist in the current language mode.
    if (!LangOpts.CPlusPlus20)
      break;
    return DefaultableMemberKind::CompareThreeWay;

  case OO_Less:
  case OO_LessEqual:
  case OO_Greater:
  case OO_GreaterEqual:
    // No point allowing this if <=> doesn't exist in the current language mode.
    if (!LangOpts.CPlusPlus20)
      break;
    return DefaultableMemberKind::CompareRelational;

  default:
    break;
  }

  // Not a defaultable member kind
  return DefaultableMemberKind::None;
}

} // namespace clang::tidy::utils
