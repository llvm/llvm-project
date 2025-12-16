//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThrowByValueCatchByReferenceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

ThrowByValueCatchByReferenceCheck::ThrowByValueCatchByReferenceCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      CheckAnonymousTemporaries(Options.get("CheckThrowTemporaries", true)),
      WarnOnLargeObject(Options.get("WarnOnLargeObject", false)),
      // Cannot access `ASTContext` from here so set it to an extremal value.
      MaxSizeOptions(
          Options.get("MaxSize", std::numeric_limits<uint64_t>::max())),
      MaxSize(MaxSizeOptions) {}

void ThrowByValueCatchByReferenceCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(cxxThrowExpr().bind("throw"), this);
  Finder->addMatcher(cxxCatchStmt().bind("catch"), this);
}

void ThrowByValueCatchByReferenceCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CheckThrowTemporaries", true);
  Options.store(Opts, "WarnOnLargeObjects", WarnOnLargeObject);
  Options.store(Opts, "MaxSize", MaxSizeOptions);
}

void ThrowByValueCatchByReferenceCheck::check(
    const MatchFinder::MatchResult &Result) {
  diagnoseThrowLocations(Result.Nodes.getNodeAs<CXXThrowExpr>("throw"));
  diagnoseCatchLocations(Result.Nodes.getNodeAs<CXXCatchStmt>("catch"),
                         *Result.Context);
}

bool ThrowByValueCatchByReferenceCheck::isFunctionParameter(
    const DeclRefExpr *DeclRefExpr) {
  return isa<ParmVarDecl>(DeclRefExpr->getDecl());
}

bool ThrowByValueCatchByReferenceCheck::isCatchVariable(
    const DeclRefExpr *DeclRefExpr) {
  auto *ValueDecl = DeclRefExpr->getDecl();
  if (auto *VarDecl = dyn_cast<clang::VarDecl>(ValueDecl))
    return VarDecl->isExceptionVariable();
  return false;
}

bool ThrowByValueCatchByReferenceCheck::isFunctionOrCatchVar(
    const DeclRefExpr *DeclRefExpr) {
  return isFunctionParameter(DeclRefExpr) || isCatchVariable(DeclRefExpr);
}

void ThrowByValueCatchByReferenceCheck::diagnoseThrowLocations(
    const CXXThrowExpr *ThrowExpr) {
  if (!ThrowExpr)
    return;
  auto *SubExpr = ThrowExpr->getSubExpr();
  if (!SubExpr)
    return;
  auto QualType = SubExpr->getType();
  if (QualType->isPointerType()) {
    // The code is throwing a pointer.
    // In case it is string literal, it is safe and we return.
    auto *Inner = SubExpr->IgnoreParenImpCasts();
    if (isa<StringLiteral>(Inner))
      return;
    // If it's a variable from a catch statement, we return as well.
    auto *DeclRef = dyn_cast<DeclRefExpr>(Inner);
    if (DeclRef && isCatchVariable(DeclRef)) {
      return;
    }
    diag(SubExpr->getBeginLoc(), "throw expression throws a pointer; it should "
                                 "throw a non-pointer value instead");
  }
  // If the throw statement does not throw by pointer then it throws by value
  // which is ok.
  // There are addition checks that emit diagnosis messages if the thrown value
  // is not an RValue. See:
  // https://www.securecoding.cert.org/confluence/display/cplusplus/ERR09-CPP.+Throw+anonymous+temporaries
  // This behavior can be influenced by an option.

  // If we encounter a CXXThrowExpr, we move through all casts until you either
  // encounter a DeclRefExpr or a CXXConstructExpr.
  // If it's a DeclRefExpr, we emit a message if the referenced variable is not
  // a catch variable or function parameter.
  // When encountering a CopyOrMoveConstructor: emit message if after casts,
  // the expression is a LValue
  if (CheckAnonymousTemporaries) {
    bool Emit = false;
    auto *CurrentSubExpr = SubExpr->IgnoreImpCasts();
    const auto *VariableReference = dyn_cast<DeclRefExpr>(CurrentSubExpr);
    const auto *ConstructorCall = dyn_cast<CXXConstructExpr>(CurrentSubExpr);
    // If we have a DeclRefExpr, we flag for emitting a diagnosis message in
    // case the referenced variable is neither a function parameter nor a
    // variable declared in the catch statement.
    if (VariableReference)
      Emit = !isFunctionOrCatchVar(VariableReference);
    else if (ConstructorCall &&
             ConstructorCall->getConstructor()->isCopyOrMoveConstructor()) {
      // If we have a copy / move construction, we emit a diagnosis message if
      // the object that we copy construct from is neither a function parameter
      // nor a variable declared in a catch statement
      auto ArgIter =
          ConstructorCall
              ->arg_begin(); // there's only one for copy constructors
      auto *CurrentSubExpr = (*ArgIter)->IgnoreImpCasts();
      if (CurrentSubExpr->isLValue()) {
        if (auto *Tmp = dyn_cast<DeclRefExpr>(CurrentSubExpr))
          Emit = !isFunctionOrCatchVar(Tmp);
        else if (isa<CallExpr>(CurrentSubExpr))
          Emit = true;
      }
    }
    if (Emit)
      diag(SubExpr->getBeginLoc(),
           "throw expression should throw anonymous temporary values instead");
  }
}

void ThrowByValueCatchByReferenceCheck::diagnoseCatchLocations(
    const CXXCatchStmt *CatchStmt, ASTContext &Context) {
  if (!CatchStmt)
    return;
  auto CaughtType = CatchStmt->getCaughtType();
  if (CaughtType.isNull())
    return;
  auto *VarDecl = CatchStmt->getExceptionDecl();
  if (const auto *PT = CaughtType.getCanonicalType()->getAs<PointerType>()) {
    const char *DiagMsgCatchReference =
        "catch handler catches a pointer value; "
        "should throw a non-pointer value and "
        "catch by reference instead";
    // We do not diagnose when catching pointer to strings since we also allow
    // throwing string literals.
    if (!PT->getPointeeType()->isAnyCharacterType())
      diag(VarDecl->getBeginLoc(), DiagMsgCatchReference);
  } else if (!CaughtType->isReferenceType()) {
    const char *DiagMsgCatchReference = "catch handler catches by value; "
                                        "should catch by reference instead";
    // If it's not a pointer and not a reference then it must be caught "by
    // value". In this case we should emit a diagnosis message unless the type
    // is trivial.
    if (!CaughtType.isTrivialType(Context)) {
      diag(VarDecl->getBeginLoc(), DiagMsgCatchReference);
    } else if (WarnOnLargeObject) {
      // If the type is trivial, then catching it by reference is not dangerous.
      // However, catching large objects by value decreases the performance.

      // We can now access `ASTContext` so if `MaxSize` is an extremal value
      // then set it to the size of `size_t`.
      if (MaxSize == std::numeric_limits<uint64_t>::max())
        MaxSize = Context.getTypeSize(Context.getSizeType());
      if (Context.getTypeSize(CaughtType) > MaxSize)
        diag(VarDecl->getBeginLoc(), DiagMsgCatchReference);
    }
  }
}

} // namespace clang::tidy::misc
