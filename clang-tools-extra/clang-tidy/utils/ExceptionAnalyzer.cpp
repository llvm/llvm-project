//===--- ExceptionAnalyzer.cpp - clang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExceptionAnalyzer.h"

namespace clang::tidy::utils {

void ExceptionAnalyzer::ExceptionInfo::registerException(
    const Type *ExceptionType) {
  assert(ExceptionType != nullptr && "Only valid types are accepted");
  Behaviour = State::Throwing;
  ThrownExceptions.insert(ExceptionType);
}

void ExceptionAnalyzer::ExceptionInfo::registerExceptions(
    const Throwables &Exceptions) {
  if (Exceptions.empty())
    return;
  Behaviour = State::Throwing;
  ThrownExceptions.insert(Exceptions.begin(), Exceptions.end());
}

ExceptionAnalyzer::ExceptionInfo &ExceptionAnalyzer::ExceptionInfo::merge(
    const ExceptionAnalyzer::ExceptionInfo &Other) {
  // Only the following two cases require an update to the local
  // 'Behaviour'. If the local entity is already throwing there will be no
  // change and if the other entity is throwing the merged entity will throw
  // as well.
  // If one of both entities is 'Unknown' and the other one does not throw
  // the merged entity is 'Unknown' as well.
  if (Other.Behaviour == State::Throwing)
    Behaviour = State::Throwing;
  else if (Other.Behaviour == State::Unknown && Behaviour == State::NotThrowing)
    Behaviour = State::Unknown;

  ContainsUnknown = ContainsUnknown || Other.ContainsUnknown;
  ThrownExceptions.insert(Other.ThrownExceptions.begin(),
                          Other.ThrownExceptions.end());
  return *this;
}

// FIXME: This could be ported to clang later.
namespace {

bool isUnambiguousPublicBaseClass(const Type *DerivedType,
                                  const Type *BaseType) {
  const auto *DerivedClass =
      DerivedType->getCanonicalTypeUnqualified()->getAsCXXRecordDecl();
  const auto *BaseClass =
      BaseType->getCanonicalTypeUnqualified()->getAsCXXRecordDecl();
  if (!DerivedClass || !BaseClass)
    return false;

  CXXBasePaths Paths;
  Paths.setOrigin(DerivedClass);

  bool IsPublicBaseClass = false;
  DerivedClass->lookupInBases(
      [&BaseClass, &IsPublicBaseClass](const CXXBaseSpecifier *BS,
                                       CXXBasePath &) {
        if (BS->getType()
                    ->getCanonicalTypeUnqualified()
                    ->getAsCXXRecordDecl() == BaseClass &&
            BS->getAccessSpecifier() == AS_public) {
          IsPublicBaseClass = true;
          return true;
        }

        return false;
      },
      Paths);

  return !Paths.isAmbiguous(BaseType->getCanonicalTypeUnqualified()) &&
         IsPublicBaseClass;
}

inline bool isPointerOrPointerToMember(const Type *T) {
  return T->isPointerType() || T->isMemberPointerType();
}

std::optional<QualType> getPointeeOrArrayElementQualType(QualType T) {
  if (T->isAnyPointerType() || T->isMemberPointerType())
    return T->getPointeeType();

  if (T->isArrayType())
    return T->getAsArrayTypeUnsafe()->getElementType();

  return std::nullopt;
}

bool isBaseOf(const Type *DerivedType, const Type *BaseType) {
  const auto *DerivedClass = DerivedType->getAsCXXRecordDecl();
  const auto *BaseClass = BaseType->getAsCXXRecordDecl();
  if (!DerivedClass || !BaseClass)
    return false;

  return !DerivedClass->forallBases(
      [BaseClass](const CXXRecordDecl *Cur) { return Cur != BaseClass; });
}

// Check if T1 is more or Equally qualified than T2.
bool moreOrEquallyQualified(QualType T1, QualType T2) {
  return T1.getQualifiers().isStrictSupersetOf(T2.getQualifiers()) ||
         T1.getQualifiers() == T2.getQualifiers();
}

bool isStandardPointerConvertible(QualType From, QualType To) {
  assert((From->isPointerType() || From->isMemberPointerType()) &&
         (To->isPointerType() || To->isMemberPointerType()) &&
         "Pointer conversion should be performed on pointer types only.");

  if (!moreOrEquallyQualified(To->getPointeeType(), From->getPointeeType()))
    return false;

  // (1)
  // A null pointer constant can be converted to a pointer type ...
  // The conversion of a null pointer constant to a pointer to cv-qualified type
  // is a single conversion, and not the sequence of a pointer conversion
  // followed by a qualification conversion. A null pointer constant of integral
  // type can be converted to a prvalue of type std::nullptr_t
  if (To->isPointerType() && From->isNullPtrType())
    return true;

  // (2)
  // A prvalue of type “pointer to cv T”, where T is an object type, can be
  // converted to a prvalue of type “pointer to cv void”.
  if (To->isVoidPointerType() && From->isObjectPointerType())
    return true;

  // (3)
  // A prvalue of type “pointer to cv D”, where D is a complete class type, can
  // be converted to a prvalue of type “pointer to cv B”, where B is a base
  // class of D. If B is an inaccessible or ambiguous base class of D, a program
  // that necessitates this conversion is ill-formed.
  if (const auto *RD = From->getPointeeCXXRecordDecl()) {
    if (RD->isCompleteDefinition() &&
        isBaseOf(From->getPointeeType().getTypePtr(),
                 To->getPointeeType().getTypePtr())) {
      // If B is an inaccessible or ambiguous base class of D, a program
      // that necessitates this conversion is ill-formed
      return isUnambiguousPublicBaseClass(From->getPointeeType().getTypePtr(),
                                          To->getPointeeType().getTypePtr());
    }
  }

  return false;
}

bool isFunctionPointerConvertible(QualType From, QualType To) {
  if (!From->isFunctionPointerType() && !From->isFunctionType() &&
      !From->isMemberFunctionPointerType())
    return false;

  if (!To->isFunctionPointerType() && !To->isMemberFunctionPointerType())
    return false;

  if (To->isFunctionPointerType()) {
    if (From->isFunctionPointerType())
      return To->getPointeeType() == From->getPointeeType();

    if (From->isFunctionType())
      return To->getPointeeType() == From;

    return false;
  }

  if (To->isMemberFunctionPointerType()) {
    if (!From->isMemberFunctionPointerType())
      return false;

    const auto *FromMember = cast<MemberPointerType>(From);
    const auto *ToMember = cast<MemberPointerType>(To);

    // Note: converting Derived::* to Base::* is a different kind of conversion,
    // called Pointer-to-member conversion.
    return FromMember->getClass() == ToMember->getClass() &&
           FromMember->getPointeeType() == ToMember->getPointeeType();
  }

  return false;
}

// Checks if From is qualification convertible to To based on the current
// LangOpts. If From is any array, we perform the array to pointer conversion
// first. The function only performs checks based on C++ rules, which can differ
// from the C rules.
//
// The function should only be called in C++ mode.
bool isQualificationConvertiblePointer(QualType From, QualType To,
                                       LangOptions LangOpts) {

  // [N4659 7.5 (1)]
  // A cv-decomposition of a type T is a sequence of cv_i and P_i such that T is
  //    cv_0 P_0 cv_1 P_1 ... cv_n−1 P_n−1 cv_n U” for n > 0,
  // where each cv_i is a set of cv-qualifiers, and each P_i is “pointer to”,
  // “pointer to member of class C_i of type”, “array of N_i”, or
  // “array of unknown bound of”.
  //
  // If P_i designates an array, the cv-qualifiers cv_i+1 on the element type
  // are also taken as the cv-qualifiers cvi of the array.
  //
  // The n-tuple of cv-qualifiers after the first one in the longest
  // cv-decomposition of T, that is, cv_1, cv_2, ... , cv_n, is called the
  // cv-qualification signature of T.

  auto isValidP_i = [](QualType P) {
    return P->isPointerType() || P->isMemberPointerType() ||
           P->isConstantArrayType() || P->isIncompleteArrayType();
  };

  auto isSameP_i = [](QualType P1, QualType P2) {
    if (P1->isPointerType())
      return P2->isPointerType();

    if (P1->isMemberPointerType())
      return P2->isMemberPointerType() &&
             P1->getAs<MemberPointerType>()->getClass() ==
                 P2->getAs<MemberPointerType>()->getClass();

    if (P1->isConstantArrayType())
      return P2->isConstantArrayType() &&
             cast<ConstantArrayType>(P1)->getSize() ==
                 cast<ConstantArrayType>(P2)->getSize();

    if (P1->isIncompleteArrayType())
      return P2->isIncompleteArrayType();

    return false;
  };

  // (2)
  // Two types From and To are similar if they have cv-decompositions with the
  // same n such that corresponding P_i components are the same [(added by
  // N4849 7.3.5) or one is “array of N_i” and the other is “array of unknown
  // bound of”], and the types denoted by U are the same.
  //
  // (3)
  // A prvalue expression of type From can be converted to type To if the
  // following conditions are satisfied:
  //  - From and To are similar
  //  - For every i > 0, if const is in cv_i of From then const is in cv_i of
  //  To, and similarly for volatile.
  //  - [(derived from addition by N4849 7.3.5) If P_i of From is “array of
  //  unknown bound of”, P_i of To is “array of unknown bound of”.]
  //  - If the cv_i of From and cv_i of To are different, then const is in every
  //  cv_k of To for 0 < k < i.

  int I = 0;
  bool ConstUntilI = true;
  auto SatisfiesCVRules = [&I, &ConstUntilI](const QualType &From,
                                             const QualType &To) {
    if (I > 1) {
      if (From.getQualifiers() != To.getQualifiers() && !ConstUntilI)
        return false;
    }

    if (I > 0) {
      if (From.isConstQualified() && !To.isConstQualified())
        return false;

      if (From.isVolatileQualified() && !To.isVolatileQualified())
        return false;

      ConstUntilI = To.isConstQualified();
    }

    return true;
  };

  while (isValidP_i(From) && isValidP_i(To)) {
    // Remove every sugar.
    From = From.getCanonicalType();
    To = To.getCanonicalType();

    if (!SatisfiesCVRules(From, To))
      return false;

    if (!isSameP_i(From, To)) {
      if (LangOpts.CPlusPlus20) {
        if (From->isConstantArrayType() && !To->isIncompleteArrayType())
          return false;

        if (From->isIncompleteArrayType() && !To->isIncompleteArrayType())
          return false;

      } else {
        return false;
      }
    }

    ++I;
    std::optional<QualType> FromPointeeOrElem =
        getPointeeOrArrayElementQualType(From);
    std::optional<QualType> ToPointeeOrElem =
        getPointeeOrArrayElementQualType(To);

    assert(FromPointeeOrElem &&
           "From pointer or array has no pointee or element!");
    assert(ToPointeeOrElem && "To pointer or array has no pointee or element!");

    From = *FromPointeeOrElem;
    To = *ToPointeeOrElem;
  }

  // In this case the length (n) of From and To are not the same.
  if (isValidP_i(From) || isValidP_i(To))
    return false;

  // We hit U.
  if (!SatisfiesCVRules(From, To))
    return false;

  return From.getTypePtr() == To.getTypePtr();
}
} // namespace

static bool canThrow(const FunctionDecl *Func) {
  // consteval specifies that every call to the function must produce a
  // compile-time constant, which cannot evaluate a throw expression without
  // producing a compilation error.
  if (Func->isConsteval())
    return false;

  const auto *FunProto = Func->getType()->getAs<FunctionProtoType>();
  if (!FunProto)
    return true;

  switch (FunProto->canThrow()) {
  case CT_Cannot:
    return false;
  case CT_Dependent: {
    const Expr *NoexceptExpr = FunProto->getNoexceptExpr();
    if (!NoexceptExpr)
      return true; // no noexept - can throw

    if (NoexceptExpr->isValueDependent())
      return true; // depend on template - some instance can throw

    bool Result = false;
    if (!NoexceptExpr->EvaluateAsBooleanCondition(Result, Func->getASTContext(),
                                                  /*InConstantContext=*/true))
      return true;  // complex X condition in noexcept(X), cannot validate,
                    // assume that may throw
    return !Result; // noexcept(false) - can throw
  }
  default:
    return true;
  };
}

bool ExceptionAnalyzer::ExceptionInfo::filterByCatch(
    const Type *HandlerTy, const ASTContext &Context) {
  llvm::SmallVector<const Type *, 8> TypesToDelete;
  for (const Type *ExceptionTy : ThrownExceptions) {
    CanQualType ExceptionCanTy = ExceptionTy->getCanonicalTypeUnqualified();
    CanQualType HandlerCanTy = HandlerTy->getCanonicalTypeUnqualified();

    // The handler is of type cv T or cv T& and E and T are the same type
    // (ignoring the top-level cv-qualifiers) ...
    if (ExceptionCanTy == HandlerCanTy) {
      TypesToDelete.push_back(ExceptionTy);
    }

    // The handler is of type cv T or cv T& and T is an unambiguous public base
    // class of E ...
    else if (isUnambiguousPublicBaseClass(ExceptionCanTy->getTypePtr(),
                                          HandlerCanTy->getTypePtr())) {
      TypesToDelete.push_back(ExceptionTy);
    }

    if (HandlerCanTy->getTypeClass() == Type::RValueReference ||
        (HandlerCanTy->getTypeClass() == Type::LValueReference &&
         !HandlerCanTy->getTypePtr()->getPointeeType().isConstQualified()))
      continue;
    // The handler is of type cv T or const T& where T is a pointer or
    // pointer-to-member type and E is a pointer or pointer-to-member type that
    // can be converted to T by one or more of ...
    if (isPointerOrPointerToMember(HandlerCanTy->getTypePtr()) &&
        isPointerOrPointerToMember(ExceptionCanTy->getTypePtr())) {
      // A standard pointer conversion not involving conversions to pointers to
      // private or protected or ambiguous classes ...
      if (isStandardPointerConvertible(ExceptionCanTy, HandlerCanTy)) {
        TypesToDelete.push_back(ExceptionTy);
      }
      // A function pointer conversion ...
      else if (isFunctionPointerConvertible(ExceptionCanTy, HandlerCanTy)) {
        TypesToDelete.push_back(ExceptionTy);
      }
      // A a qualification conversion ...
      else if (isQualificationConvertiblePointer(ExceptionCanTy, HandlerCanTy,
                                                 Context.getLangOpts())) {
        TypesToDelete.push_back(ExceptionTy);
      }
    }

    // The handler is of type cv T or const T& where T is a pointer or
    // pointer-to-member type and E is std::nullptr_t.
    else if (isPointerOrPointerToMember(HandlerCanTy->getTypePtr()) &&
             ExceptionCanTy->isNullPtrType()) {
      TypesToDelete.push_back(ExceptionTy);
    }
  }

  for (const Type *T : TypesToDelete)
    ThrownExceptions.erase(T);

  reevaluateBehaviour();
  return !TypesToDelete.empty();
}

ExceptionAnalyzer::ExceptionInfo &
ExceptionAnalyzer::ExceptionInfo::filterIgnoredExceptions(
    const llvm::StringSet<> &IgnoredTypes, bool IgnoreBadAlloc) {
  llvm::SmallVector<const Type *, 8> TypesToDelete;
  // Note: Using a 'SmallSet' with 'llvm::remove_if()' is not possible.
  // Therefore this slightly hacky implementation is required.
  for (const Type *T : ThrownExceptions) {
    if (const auto *TD = T->getAsTagDecl()) {
      if (TD->getDeclName().isIdentifier()) {
        if ((IgnoreBadAlloc &&
             (TD->getName() == "bad_alloc" && TD->isInStdNamespace())) ||
            (IgnoredTypes.contains(TD->getName())))
          TypesToDelete.push_back(T);
      }
    }
  }
  for (const Type *T : TypesToDelete)
    ThrownExceptions.erase(T);

  reevaluateBehaviour();
  return *this;
}

void ExceptionAnalyzer::ExceptionInfo::clear() {
  Behaviour = State::NotThrowing;
  ContainsUnknown = false;
  ThrownExceptions.clear();
}

void ExceptionAnalyzer::ExceptionInfo::reevaluateBehaviour() {
  if (ThrownExceptions.empty())
    if (ContainsUnknown)
      Behaviour = State::Unknown;
    else
      Behaviour = State::NotThrowing;
  else
    Behaviour = State::Throwing;
}

ExceptionAnalyzer::ExceptionInfo ExceptionAnalyzer::throwsException(
    const FunctionDecl *Func, const ExceptionInfo::Throwables &Caught,
    llvm::SmallSet<const FunctionDecl *, 32> &CallStack) {
  if (!Func || CallStack.contains(Func) ||
      (!CallStack.empty() && !canThrow(Func)))
    return ExceptionInfo::createNonThrowing();

  if (const Stmt *Body = Func->getBody()) {
    CallStack.insert(Func);
    ExceptionInfo Result = throwsException(Body, Caught, CallStack);

    // For a constructor, we also have to check the initializers.
    if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(Func)) {
      for (const CXXCtorInitializer *Init : Ctor->inits()) {
        ExceptionInfo Excs =
            throwsException(Init->getInit(), Caught, CallStack);
        Result.merge(Excs);
      }
    }

    CallStack.erase(Func);
    return Result;
  }

  auto Result = ExceptionInfo::createUnknown();
  if (const auto *FPT = Func->getType()->getAs<FunctionProtoType>()) {
    for (const QualType &Ex : FPT->exceptions())
      Result.registerException(Ex.getTypePtr());
  }
  return Result;
}

/// Analyzes a single statement on it's throwing behaviour. This is in principle
/// possible except some 'Unknown' functions are called.
ExceptionAnalyzer::ExceptionInfo ExceptionAnalyzer::throwsException(
    const Stmt *St, const ExceptionInfo::Throwables &Caught,
    llvm::SmallSet<const FunctionDecl *, 32> &CallStack) {
  auto Results = ExceptionInfo::createNonThrowing();
  if (!St)
    return Results;

  if (const auto *Throw = dyn_cast<CXXThrowExpr>(St)) {
    if (const auto *ThrownExpr = Throw->getSubExpr()) {
      const auto *ThrownType =
          ThrownExpr->getType()->getUnqualifiedDesugaredType();
      if (ThrownType->isReferenceType())
        ThrownType = ThrownType->castAs<ReferenceType>()
                         ->getPointeeType()
                         ->getUnqualifiedDesugaredType();
      Results.registerException(
          ThrownExpr->getType()->getUnqualifiedDesugaredType());
    } else
      // A rethrow of a caught exception happens which makes it possible
      // to throw all exception that are caught in the 'catch' clause of
      // the parent try-catch block.
      Results.registerExceptions(Caught);
  } else if (const auto *Try = dyn_cast<CXXTryStmt>(St)) {
    ExceptionInfo Uncaught =
        throwsException(Try->getTryBlock(), Caught, CallStack);
    for (unsigned I = 0; I < Try->getNumHandlers(); ++I) {
      const CXXCatchStmt *Catch = Try->getHandler(I);

      // Everything is caught through 'catch(...)'.
      if (!Catch->getExceptionDecl()) {
        ExceptionInfo Rethrown = throwsException(
            Catch->getHandlerBlock(), Uncaught.getExceptionTypes(), CallStack);
        Results.merge(Rethrown);
        Uncaught.clear();
      } else {
        const auto *CaughtType =
            Catch->getCaughtType()->getUnqualifiedDesugaredType();
        if (CaughtType->isReferenceType()) {
          CaughtType = CaughtType->castAs<ReferenceType>()
                           ->getPointeeType()
                           ->getUnqualifiedDesugaredType();
        }

        // If the caught exception will catch multiple previously potential
        // thrown types (because it's sensitive to inheritance) the throwing
        // situation changes. First of all filter the exception types and
        // analyze if the baseclass-exception is rethrown.
        if (Uncaught.filterByCatch(
                CaughtType, Catch->getExceptionDecl()->getASTContext())) {
          ExceptionInfo::Throwables CaughtExceptions;
          CaughtExceptions.insert(CaughtType);
          ExceptionInfo Rethrown = throwsException(Catch->getHandlerBlock(),
                                                   CaughtExceptions, CallStack);
          Results.merge(Rethrown);
        }
      }
    }
    Results.merge(Uncaught);
  } else if (const auto *Call = dyn_cast<CallExpr>(St)) {
    if (const FunctionDecl *Func = Call->getDirectCallee()) {
      ExceptionInfo Excs = throwsException(Func, Caught, CallStack);
      Results.merge(Excs);
    }
  } else if (const auto *Construct = dyn_cast<CXXConstructExpr>(St)) {
    ExceptionInfo Excs =
        throwsException(Construct->getConstructor(), Caught, CallStack);
    Results.merge(Excs);
  } else if (const auto *DefaultInit = dyn_cast<CXXDefaultInitExpr>(St)) {
    ExceptionInfo Excs =
        throwsException(DefaultInit->getExpr(), Caught, CallStack);
    Results.merge(Excs);
  } else if (const auto *Coro = dyn_cast<CoroutineBodyStmt>(St)) {
    for (const Stmt *Child : Coro->childrenExclBody()) {
      if (Child != Coro->getExceptionHandler()) {
        ExceptionInfo Excs = throwsException(Child, Caught, CallStack);
        Results.merge(Excs);
      }
    }
    ExceptionInfo Excs = throwsException(Coro->getBody(), Caught, CallStack);
    Results.merge(throwsException(Coro->getExceptionHandler(),
                                  Excs.getExceptionTypes(), CallStack));
    for (const Type *Throwable : Excs.getExceptionTypes()) {
      if (const auto ThrowableRec = Throwable->getAsCXXRecordDecl()) {
        ExceptionInfo DestructorExcs =
            throwsException(ThrowableRec->getDestructor(), Caught, CallStack);
        Results.merge(DestructorExcs);
      }
    }
  } else {
    for (const Stmt *Child : St->children()) {
      ExceptionInfo Excs = throwsException(Child, Caught, CallStack);
      Results.merge(Excs);
    }
  }
  return Results;
}

ExceptionAnalyzer::ExceptionInfo
ExceptionAnalyzer::analyzeImpl(const FunctionDecl *Func) {
  ExceptionInfo ExceptionList;

  // Check if the function has already been analyzed and reuse that result.
  const auto CacheEntry = FunctionCache.find(Func);
  if (CacheEntry == FunctionCache.end()) {
    llvm::SmallSet<const FunctionDecl *, 32> CallStack;
    ExceptionList =
        throwsException(Func, ExceptionInfo::Throwables(), CallStack);

    // Cache the result of the analysis. This is done prior to filtering
    // because it is best to keep as much information as possible.
    // The results here might be relevant to different analysis passes
    // with different needs as well.
    FunctionCache.try_emplace(Func, ExceptionList);
  } else
    ExceptionList = CacheEntry->getSecond();

  return ExceptionList;
}

ExceptionAnalyzer::ExceptionInfo
ExceptionAnalyzer::analyzeImpl(const Stmt *Stmt) {
  llvm::SmallSet<const FunctionDecl *, 32> CallStack;
  return throwsException(Stmt, ExceptionInfo::Throwables(), CallStack);
}

template <typename T>
ExceptionAnalyzer::ExceptionInfo
ExceptionAnalyzer::analyzeDispatch(const T *Node) {
  ExceptionInfo ExceptionList = analyzeImpl(Node);

  if (ExceptionList.getBehaviour() == State::NotThrowing ||
      ExceptionList.getBehaviour() == State::Unknown)
    return ExceptionList;

  // Remove all ignored exceptions from the list of exceptions that can be
  // thrown.
  ExceptionList.filterIgnoredExceptions(IgnoredExceptions, IgnoreBadAlloc);

  return ExceptionList;
}

ExceptionAnalyzer::ExceptionInfo
ExceptionAnalyzer::analyze(const FunctionDecl *Func) {
  return analyzeDispatch(Func);
}

ExceptionAnalyzer::ExceptionInfo ExceptionAnalyzer::analyze(const Stmt *Stmt) {
  return analyzeDispatch(Stmt);
}

} // namespace clang::tidy::utils
