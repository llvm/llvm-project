//===- PointerFlowPairs.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlowPairs.h"
#include "SSAFAnalysesCommon.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TypeBase.h"
#include "llvm/ADT/SmallVector.h"

namespace {
using namespace clang;
using namespace ssaf;

//===----------------------------------------------------------------------===//
// Helper functions for `PointerFlowPairMatcher::matches`.
//===----------------------------------------------------------------------===//

bool findUntypedPairsInStmt(const Stmt *S, const NamedDecl *RootDecl,
                            llvm::SmallVectorImpl<PointerFlowPair> &Result);
bool findUntypedPairsInDecl(const Decl *D,
                            llvm::SmallVectorImpl<PointerFlowPair> &Result);

/// Dispatch \p DynNode to `findUntypedPairsInStmt`/`findUntypedPairsInDecl` and
/// collect pairs without checking types.
bool findUntypedPairs(const DynTypedNode &DynNode, const NamedDecl *RootDecl,
                      llvm::SmallVectorImpl<PointerFlowPair> &Result) {
  if (const Stmt *S = DynNode.get<Stmt>())
    return findUntypedPairsInStmt(S, RootDecl, Result);
  if (const Decl *D = DynNode.get<Decl>())
    return findUntypedPairsInDecl(D, Result);
  return false;
}

template <typename ParmsProvider, typename ArgsProvider>
bool matchesArgsWithParams(unsigned ArgIdxStart, ParmsProvider *PP,
                           ArgsProvider *AP,
                           llvm::SmallVectorImpl<PointerFlowPair> &Result) {
  unsigned ArgIdx = ArgIdxStart;
  bool Found = false;

  for (unsigned ParmIdx = 0;
       ParmIdx < PP->getNumParams() && ArgIdx < AP->getNumArgs();
       ++ArgIdx, ++ParmIdx) {
    if (const ParmVarDecl *PD = PP->getParamDecl(ParmIdx)) {
      Result.emplace_back(PD, AP->getArg(ArgIdx));
      Found = true;
    }
  }
  return Found;
}

bool findUntypedPairsInStmt(const Stmt *S, const NamedDecl *RootDecl,
                            llvm::SmallVectorImpl<PointerFlowPair> &Result) {
  // Match 'p = q':
  if (const auto *BO = dyn_cast<BinaryOperator>(S);
      BO && BO->getOpcode() == BO_Assign) {
    Result.emplace_back(BO->getLHS(), BO->getRHS());
    return true;
  }

  // Match arg-to-param passing (in CallExpr):
  if (const auto *CE = dyn_cast<CallExpr>(S)) {
    const FunctionDecl *FD = CE->getDirectCallee();

    if (!FD)
      return false;

    unsigned ArgIdx = 0;

    if (isa<CXXOperatorCallExpr>(CE))
      if (const auto *MD = dyn_cast<CXXMethodDecl>(FD);
          MD && !MD->isExplicitObjectMemberFunction())
        ArgIdx = 1;
    return matchesArgsWithParams(ArgIdx, FD, CE, Result);
  }
  // Match arg-to-param passing (in CXXConstructExpr):
  if (const auto *CCE = dyn_cast<CXXConstructExpr>(S)) {
    return matchesArgsWithParams(/*ArgIdxStart=*/0, CCE->getConstructor(), CCE,
                                 Result);
  }
  if (const auto *RS = dyn_cast<ReturnStmt>(S)) {
    const Expr *RetExpr = RS->getRetValue();
    if (RetExpr)
      if (const auto *FD = dyn_cast<FunctionDecl>(RootDecl)) {
        Result.emplace_back(FD, RetExpr, /*IsLHSRet=*/true);
        return true;
      }
    return false;
  }
  return false;
}

bool findUntypedPairsInDecl(const Decl *D,
                            llvm::SmallVectorImpl<PointerFlowPair> &Result) {
  const Expr *InitExpr = nullptr;

  if (const auto *VD = dyn_cast<ValueDecl>(D)) {
    if (const auto *Var = dyn_cast<VarDecl>(VD))
      InitExpr = Var->getInit();
    if (const auto *Fd = dyn_cast<FieldDecl>(VD))
      InitExpr = Fd->getInClassInitializer();

    // Match initializer-list:
    if (const auto *InitLst = dyn_cast_or_null<InitListExpr>(InitExpr)) {
      Result.emplace_back(VD, InitLst);
      return true;
    }
    if (InitExpr) {
      // Match initializers to variables/fields of a pointer type:
      Result.emplace_back(VD, InitExpr);
      return true;
    }
  }

  bool Found = false;
  // Match C++ constructor member-initializers here. The FieldDecl a
  // member-initializer targets is only recorded on the CXXCtorInitializer
  // itself, which is neither a Stmt nor a Decl,
  if (const auto *CtorD = dyn_cast<CXXConstructorDecl>(D)) {
    for (const auto *E : CtorD->inits()) {
      if (const FieldDecl *FD = E->getMember()) {
        Result.emplace_back(FD, E->getInit());
        Found = true;
      }
    }
  }
  return Found;
}

/// Pipeline the output of `findUntypedPairs`: further decompose
/// list-initializers around record types and filter out pairs that are not
/// pointers or arrays.
///
/// Upon return, each pair in \c Result has a pointer or array type.  In
/// addition, if its \c RHS is a list-initializer, the pair has an array type.
///
/// For example,
/// - suppose 'LHS' has type 'struct S {int *x; int *y;};',
///   - it finds in '(LHS, {1 , 2})' two pairs '(x, 1), (y, 2)'.
///
/// - suppose 'LHS' has type 'S[2]',
///   - it finds in '(LHS, {{1 , 2}, {3, 4}})' four pairs
///     '(x, 1), (y, 2), (x, 3), (y, 4)'.
///
/// - suppose 'LHS' has type 'int *[2]',
///   - it finds in '(LHS, {nullptr, nullptr})' one pair
///     '(LHS, {nullptr, nullptr})'.
///
/// - suppose 'LHS' has type 'int *',
///   - it finds in '(LHS, nullptr)' one pair '(LHS, nullptr)'.
bool matchPtrOrArrPairs(const PointerFlowPairMatcher &Matcher, PointerFlowPair Pair,
                        llvm::SmallVectorImpl<PointerFlowPair> &Result);
} // namespace

namespace clang::ssaf {

bool PointerFlowPairMatcher::matches(
    const DynTypedNode &DynNode, const NamedDecl *RootDecl,
    llvm::SmallVectorImpl<PointerFlowPair> &Result) const {
  llvm::SmallVector<PointerFlowPair, 8> UntypedPairs;
  findUntypedPairs(DynNode, RootDecl, UntypedPairs);

  bool Found = false;
  for (const PointerFlowPair &P : UntypedPairs)
    Found |= matchPtrOrArrPairs(*this, P, Result);
  return Found;
}
} // namespace clang::ssaf

namespace {

struct GetType {
  QualType operator()(const ValueDecl *D, bool IsRet) const {
    return IsRet ? cast<FunctionDecl>(D)->getReturnType() : D->getType();
  }

  QualType operator()(const Expr *E) const { return E->getType(); }
};

//===----------------------------------------------------------------------===//
// Helper functions for `PointerFlowPairMatcher::matchPtrOrArrPairs`.
//===----------------------------------------------------------------------===//

/// Helper function for matchPtrOrArrPairs that handles record
/// types.
bool matchInitializerListForRecordDeclRecursive(
    const PointerFlowPairMatcher &Matcher, const RecordDecl *RecordTy,
    const InitListExpr *ILE, llvm::SmallVectorImpl<PointerFlowPair> &Result) {
  if (const auto *CXXRD = dyn_cast<CXXRecordDecl>(RecordTy))
    if (CXXRD->getNumBases() != 0) {
      // FIXME: support this:
      logWarningFromError(makeErrAtNode(
          Matcher.Ctx, ILE,
          "attempt to create pointer assignment edges between "
          "CXXRecordDecls with base classes and initializer-lists"));
      return false;
    }
  // Handle union:
  if (RecordTy->isUnion()) {
    const auto *InitField = ILE->getInitializedFieldInUnion();

    if (!InitField || ILE->inits().empty())
      return false;
    return matchPtrOrArrPairs(Matcher, {InitField, ILE->getInit(0)}, Result);
  }
  // Handle struct/class:
  ILE = ILE->isSemanticForm() ? ILE : ILE->getSemanticForm();

  auto FieldIter = RecordTy->field_begin();
  bool Found = false;

  for (const auto *Init : ILE->inits()) {
    // Skip unnamed bit-fields:
    while (FieldIter != RecordTy->field_end() && FieldIter->isUnnamedBitField())
      ++FieldIter;
    assert(FieldIter != RecordTy->field_end());
    Found |= matchPtrOrArrPairs(Matcher, {*(FieldIter++), Init}, Result);
  }
  return Found;
}

/// Helper function of `matchPtrOrArrPairs` that specifically
/// handles an list-initializer to a (multi-dimensional) array of a record
/// type.
bool matchInitializerListForRecordArrayRecursive(
    const PointerFlowPairMatcher &Matcher, const ArrayType *ArrayType,
    const InitListExpr *ILE, llvm::SmallVectorImpl<PointerFlowPair> &Result) {
  assert(Matcher.Ctx.getBaseElementType(ArrayType)->isRecordType() &&
         "expected a (multi-dimensional) array of a record type");
  auto EltTy = ArrayType->getElementType();
  bool Found = false;

  if (const auto *RD = EltTy->getAsRecordDecl()) {
    for (const auto *Init : ILE->inits()) {
      if (const auto *SubILE = dyn_cast<InitListExpr>(Init))
        Found |= matchInitializerListForRecordDeclRecursive(Matcher, RD, SubILE,
                                                            Result);
      // No need to handle non-list-initialized records:
    }
    return Found;
  }
  if (auto *SubArrayType = Matcher.Ctx.getAsArrayType(EltTy)) {
    for (const auto *Init : ILE->inits())
      if (const auto *SubILE = dyn_cast<InitListExpr>(Init))
        Found |= matchInitializerListForRecordArrayRecursive(
            Matcher, SubArrayType, SubILE, Result);
    return Found;
  }
  return false;
}

bool matchPtrOrArrPairs(const PointerFlowPairMatcher &Matcher, PointerFlowPair Pair,
                        llvm::SmallVectorImpl<PointerFlowPair> &Result) {
  // - Base case: `RHS` is not a InitListExpr;
  // - Call `matchInitializerListForRecordDeclRecursive` to handle
  //   list-initializing record;
  // - Call `matchInitializerListForRecordArrayRecursive` to handle
  //   list-initializing (multi-d) array of records;
  // - Recursion on list-initialization of scalar.
  const auto *ILE = dyn_cast<InitListExpr>(Pair.RHS);
  QualType Type = Pair.visitLHS(GetType{});

  if (!ILE) {
    // Base case:
    if (!hasPtrOrArrType(Type))
      return false;
    Result.push_back(Pair);
    return true;
  }

  if (auto *RD = Type->getAsRecordDecl())
    return matchInitializerListForRecordDeclRecursive(Matcher, RD, ILE, Result);
  if (auto *ArrayType = Matcher.Ctx.getAsArrayType(Type)) {
    auto BaseTy = Matcher.Ctx.getBaseElementType(ArrayType);

    if (BaseTy->isRecordType())
      return matchInitializerListForRecordArrayRecursive(Matcher, ArrayType,
                                                         ILE, Result);
    Result.push_back(Pair);
    return true;
  }

  // Must be the case of using a initializer-list for a scalar.
  // The initializer-list can be either singleton or empty:
  if (ILE->getNumInits() == 0)
    return false;
  Pair.RHS = ILE->getInit(0);
  return matchPtrOrArrPairs(Matcher, Pair, Result);
}

} // namespace
