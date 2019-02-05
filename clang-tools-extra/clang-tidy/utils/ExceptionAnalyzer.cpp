//===--- ExceptionAnalyzer.cpp - clang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExceptionAnalyzer.h"

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
static bool isBaseOf(const Type *DerivedType, const Type *BaseType) {
  const auto *DerivedClass = DerivedType->getAsCXXRecordDecl();
  const auto *BaseClass = BaseType->getAsCXXRecordDecl();
  if (!DerivedClass || !BaseClass)
    return false;

  return !DerivedClass->forallBases(
      [BaseClass](const CXXRecordDecl *Cur) { return Cur != BaseClass; });
}

namespace tidy {
namespace utils {

ExceptionAnalyzer::TypeVec ExceptionAnalyzer::throwsException(
    const FunctionDecl *Func,
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

ExceptionAnalyzer::TypeVec ExceptionAnalyzer::throwsException(
    const Stmt *St, const TypeVec &Caught,
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
        if (TD->getDeclName().isIdentifier() && TD->getName() == "bad_alloc" &&
            TD->isInStdNamespace())
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

bool ExceptionAnalyzer::throwsException(const FunctionDecl *Func) {
  // Check if the function has already been analyzed and reuse that result.
  if (FunctionCache.count(Func) > 0)
    return FunctionCache[Func];

  llvm::SmallSet<const FunctionDecl *, 32> CallStack;
  TypeVec ExceptionList = throwsException(Func, CallStack);

  // Remove all ignored exceptions from the list of exceptions that can be
  // thrown.
  auto NewEnd = llvm::remove_if(ExceptionList, [this](const Type *Exception) {
    return isIgnoredExceptionType(Exception);
  });
  ExceptionList.erase(NewEnd, ExceptionList.end());

  // Cache the result of the analysis.
  bool FunctionThrows = ExceptionList.size() > 0;
  FunctionCache.insert(std::make_pair(Func, FunctionThrows));

  return FunctionThrows;
}

bool ExceptionAnalyzer::isIgnoredExceptionType(const Type *Exception) {
  if (const auto *TD = Exception->getAsTagDecl()) {
    if (TD->getDeclName().isIdentifier())
      return IgnoredExceptions.count(TD->getName()) > 0;
  }
  return false;
}

} // namespace utils
} // namespace tidy

} // namespace clang
