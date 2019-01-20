//===--- ExceptionEscapeCheck.cpp - clang-tidy-----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExceptionEscapeCheck.h"

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"

using namespace clang::ast_matchers;

namespace {
typedef llvm::SmallVector<const clang::Type *, 8> TypeVec;
} // namespace

namespace clang {

static bool isBaseOf(const Type *DerivedType, const Type *BaseType) {
  const auto *DerivedClass = DerivedType->getAsCXXRecordDecl();
  const auto *BaseClass = BaseType->getAsCXXRecordDecl();
  if (!DerivedClass || !BaseClass)
    return false;

  return !DerivedClass->forallBases(
      [BaseClass](const CXXRecordDecl *Cur) { return Cur != BaseClass; });
}

static const TypeVec
throwsException(const Stmt *St, const TypeVec &Caught,
                llvm::SmallSet<const FunctionDecl *, 32> &CallStack);

static const TypeVec
throwsException(const FunctionDecl *Func,
                llvm::SmallSet<const FunctionDecl *, 32> &CallStack) {
  if (CallStack.count(Func))
    return TypeVec();

  if (const Stmt *Body = Func->getBody()) {
    CallStack.insert(Func);
    const TypeVec Result = throwsException(Body, TypeVec(), CallStack);
    CallStack.erase(Func);
    return Result;
  }

  TypeVec Result;
  if (const auto *FPT = Func->getType()->getAs<FunctionProtoType>()) {
    for (const QualType Ex : FPT->exceptions()) {
      Result.push_back(Ex.getTypePtr());
    }
  }
  return Result;
}

static const TypeVec
throwsException(const Stmt *St, const TypeVec &Caught,
                llvm::SmallSet<const FunctionDecl *, 32> &CallStack) {
  TypeVec Results;

  if (!St)
    return Results;

  if (const auto *Throw = dyn_cast<CXXThrowExpr>(St)) {
    if (const auto *ThrownExpr = Throw->getSubExpr()) {
      const auto *ThrownType =
          ThrownExpr->getType()->getUnqualifiedDesugaredType();
      if (ThrownType->isReferenceType()) {
        ThrownType = ThrownType->castAs<ReferenceType>()
                         ->getPointeeType()
                         ->getUnqualifiedDesugaredType();
      }
      if (const auto *TD = ThrownType->getAsTagDecl()) {
        if (TD->getDeclName().isIdentifier() && TD->getName() == "bad_alloc"
            && TD->isInStdNamespace())
          return Results;
      }
      Results.push_back(ThrownExpr->getType()->getUnqualifiedDesugaredType());
    } else {
      Results.append(Caught.begin(), Caught.end());
    }
  } else if (const auto *Try = dyn_cast<CXXTryStmt>(St)) {
    TypeVec Uncaught = throwsException(Try->getTryBlock(), Caught, CallStack);
    for (unsigned i = 0; i < Try->getNumHandlers(); ++i) {
      const CXXCatchStmt *Catch = Try->getHandler(i);
      if (!Catch->getExceptionDecl()) {
        const TypeVec Rethrown =
            throwsException(Catch->getHandlerBlock(), Uncaught, CallStack);
        Results.append(Rethrown.begin(), Rethrown.end());
        Uncaught.clear();
      } else {
        const auto *CaughtType =
            Catch->getCaughtType()->getUnqualifiedDesugaredType();
        if (CaughtType->isReferenceType()) {
          CaughtType = CaughtType->castAs<ReferenceType>()
                           ->getPointeeType()
                           ->getUnqualifiedDesugaredType();
        }
        auto NewEnd =
            llvm::remove_if(Uncaught, [&CaughtType](const Type *ThrownType) {
              return ThrownType == CaughtType ||
                     isBaseOf(ThrownType, CaughtType);
            });
        if (NewEnd != Uncaught.end()) {
          Uncaught.erase(NewEnd, Uncaught.end());
          const TypeVec Rethrown = throwsException(
              Catch->getHandlerBlock(), TypeVec(1, CaughtType), CallStack);
          Results.append(Rethrown.begin(), Rethrown.end());
        }
      }
    }
    Results.append(Uncaught.begin(), Uncaught.end());
  } else if (const auto *Call = dyn_cast<CallExpr>(St)) {
    if (const FunctionDecl *Func = Call->getDirectCallee()) {
      TypeVec Excs = throwsException(Func, CallStack);
      Results.append(Excs.begin(), Excs.end());
    }
  } else {
    for (const Stmt *Child : St->children()) {
      TypeVec Excs = throwsException(Child, Caught, CallStack);
      Results.append(Excs.begin(), Excs.end());
    }
  }
  return Results;
}

static const TypeVec throwsException(const FunctionDecl *Func) {
  llvm::SmallSet<const FunctionDecl *, 32> CallStack;
  return throwsException(Func, CallStack);
}

namespace ast_matchers {
AST_MATCHER_P(FunctionDecl, throws, internal::Matcher<Type>, InnerMatcher) {
  TypeVec ExceptionList = throwsException(&Node);
  auto NewEnd = llvm::remove_if(
      ExceptionList, [this, Finder, Builder](const Type *Exception) {
        return !InnerMatcher.matches(*Exception, Finder, Builder);
      });
  ExceptionList.erase(NewEnd, ExceptionList.end());
  return ExceptionList.size();
}

AST_MATCHER_P(Type, isIgnored, llvm::StringSet<>, IgnoredExceptions) {
  if (const auto *TD = Node.getAsTagDecl()) {
    if (TD->getDeclName().isIdentifier())
      return IgnoredExceptions.count(TD->getName()) > 0;
  }
  return false;
}

AST_MATCHER_P(FunctionDecl, isEnabled, llvm::StringSet<>,
              FunctionsThatShouldNotThrow) {
  return FunctionsThatShouldNotThrow.count(Node.getNameAsString()) > 0;
}
} // namespace ast_matchers

namespace tidy {
namespace bugprone {

ExceptionEscapeCheck::ExceptionEscapeCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), RawFunctionsThatShouldNotThrow(Options.get(
                                         "FunctionsThatShouldNotThrow", "")),
      RawIgnoredExceptions(Options.get("IgnoredExceptions", "")) {
  llvm::SmallVector<StringRef, 8> FunctionsThatShouldNotThrowVec,
      IgnoredExceptionsVec;
  StringRef(RawFunctionsThatShouldNotThrow)
      .split(FunctionsThatShouldNotThrowVec, ",", -1, false);
  FunctionsThatShouldNotThrow.insert(FunctionsThatShouldNotThrowVec.begin(),
                                     FunctionsThatShouldNotThrowVec.end());
  StringRef(RawIgnoredExceptions).split(IgnoredExceptionsVec, ",", -1, false);
  IgnoredExceptions.insert(IgnoredExceptionsVec.begin(),
                           IgnoredExceptionsVec.end());
}

void ExceptionEscapeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "FunctionsThatShouldNotThrow",
                RawFunctionsThatShouldNotThrow);
  Options.store(Opts, "IgnoredExceptions", RawIgnoredExceptions);
}

void ExceptionEscapeCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus || !getLangOpts().CXXExceptions)
    return;

  Finder->addMatcher(
      functionDecl(anyOf(isNoThrow(), cxxDestructorDecl(),
                         cxxConstructorDecl(isMoveConstructor()),
                         cxxMethodDecl(isMoveAssignmentOperator()),
                         hasName("main"), hasName("swap"),
                         isEnabled(FunctionsThatShouldNotThrow)),
                   throws(unless(isIgnored(IgnoredExceptions))))
          .bind("thrower"),
      this);
}

void ExceptionEscapeCheck::check(const MatchFinder::MatchResult &Result) {
  const FunctionDecl *MatchedDecl =
      Result.Nodes.getNodeAs<FunctionDecl>("thrower");
  if (!MatchedDecl)
    return;

  // FIXME: We should provide more information about the exact location where
  // the exception is thrown, maybe the full path the exception escapes
  diag(MatchedDecl->getLocation(), "an exception may be thrown in function %0 "
       "which should not throw exceptions") << MatchedDecl;
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
