//===- BoundsSafetySuggestions.cpp - Improve your -fbounds-safety code ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/AST/StmtVisitor.h"
// TO_UPSTREAM(BoundsSafety)
#include "clang/Sema/BoundsSafetySuggestions.h"
#include "clang/Sema/DynamicCountPointerAssignmentAnalysisExported.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace llvm;
using namespace clang;

namespace {

// A map from variables to values stored in them (their potential
// "definitions", as in "reaching definitions" or "use-def chains").
using DefinitionList = SmallVector<const Expr *, 4>;
using DefinitionMap = DenseMap<const VarDecl *, DefinitionList>;

// A visitor that recursively scans an AST subtree to identify values of
// encountered local variables. It's purely syntactic; it doesn't account
// for "happens-before" relationships between definitions, and the answer
// doesn't depend on the point in the program in which possible definitions
// are queried.
//
// Still, it is very useful for identifying values of variables
// in situations when the value is actually unconditional,
// but otherwise unobvious from the AST. Or confirming that
// all possible values fall into a certain category.
//
// The visitor performs exactly one pass over the AST, which is fast enough
// for the compiler.
//
// TODO: Teach the DefinitionVisitor to understand happens-before
// (rdar://117166345).
class DefinitionVisitor : public ConstStmtVisitor<DefinitionVisitor> { // CRTP!
  void VisitChildren(const Stmt *S) {
    for (const Stmt *ChildS : S->children())
      if (ChildS)
        Visit(ChildS);
  }

  bool isSupportedVariable(const Decl *D) {
    // We currently support local variables.
    if (const auto *VD = dyn_cast_or_null<VarDecl>(D))
      if (VD->isLocalVarDecl())
        return true;

    return false;
  }

public:
  DefinitionMap DefMap;

  void VisitStmt(const Stmt *S) {
    // This is a manual implementation of RecursiveASTVIsitor behavior.
    // It only applies to statements and gives us fine control
    // over what exactly do we recurse into.
    VisitChildren(S);
  }

  // These statements are sources of variable values.
  void VisitDeclStmt(const DeclStmt *DS);
  void VisitBinaryOperator(const BinaryOperator *BO);

  // Unevaluated context visitors
  void VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *) {
    // Unevaluated context such as sizeof()/alignof()/__alignof
    return;
  }

  void VisitGenericSelectionExpr(const GenericSelectionExpr *GSE) {
    // Only the result expr is evaluated in `_Generic`.
    // E.g.
    // _Generic(0, // Not evaluated
    //   int: foo(), // Evaluated
    //   char: bar() // Not evaluated
    // );
    Visit(GSE->getResultExpr());
  }
};

} // namespace

void DefinitionVisitor::VisitDeclStmt(const DeclStmt *DS) {
  for (const Decl *D : DS->decls())
    if (const auto *VD = dyn_cast<VarDecl>(D))
      if (const Expr *E = VD->getInit()) {
        // An initialization is a definition. An uninitialized variable
        // declaration isn't a definition.
        if (isSupportedVariable(VD))
          DefMap[VD].push_back(E);

        // The initializer may have more interesting sub-expressions.
        // FIXME: For non-variable declarations, should we visit children
        // in a different way? VisitChildren() is probably unhelpful
        // because children aren't statements(?).
        Visit(E);
      }
}

void DefinitionVisitor::VisitBinaryOperator(const BinaryOperator *BO) {
  // Compound assignment operations (+= etc.) don't count as definitions.
  // They just reuse whatever's already there.
  if (BO->getOpcode() == BO_Assign) {
    if (const auto *LHSDRE =
            dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParens())) {
      const auto *VD = dyn_cast<VarDecl>(LHSDRE->getDecl());

      // An assignment to a variable is a definition of that variable.
      if (isSupportedVariable(VD))
        DefMap[VD].push_back(BO->getRHS());
    }
  }

  // Continue visitation normally.
  VisitChildren(BO);
}

namespace {

// The visitor that enumerates unsafe buffer operations and informs the Handler
// about problems associated with them.
//
// TODO: Teach UnsafeOperationVisitor to understand happens-before
// (rdar://117166345).
class UnsafeOperationVisitor : public ConstStmtVisitor<UnsafeOperationVisitor> {
  using UnsafeOpKind = BoundsSafetySuggestionHandler::UnsafeOpKind;
  using WillTrapKind = BoundsSafetySuggestionHandler::WillTrapKind;
  using PtrArithOOBKind = BoundsSafetySuggestionHandler::PtrArithOOBKind;

  BoundsSafetySuggestionHandler &Handler;
  const DefinitionMap &DefMap;
  Sema &S;
  ASTContext &Ctx;
  llvm::SmallPtrSet<const Expr *, 4> VisitedOVESourceExprs;
  llvm::SmallPtrSet<const Expr *, 4> UnsafeBitCastsToSkip;
  llvm::SmallPtrSet<const Expr *, 4> FBPtrCastsToSkip;

  void VisitChildren(const Stmt *S) {
    for (const Stmt *ChildS : S->children())
      if (ChildS)
        Visit(ChildS);
  }

  // Individual checks performed on each unsafe operation.
  void checkSingleEntityFlowingToIndexableLocalVariable(const Stmt *UnsafeOp,
                                                        const Expr *Operand,
                                                        UnsafeOpKind Kind);

  void checkSingleEntityFlowingToIndexableLocalVariableHandleOp(
      const Stmt *UnsafeOp, UnsafeOpKind Kind, const VarDecl *IndexableVar,
      llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
          &SingleEntities);

  void checkSingleEntityFlowingToIndexableLocalVariableHandleIndexing(
      const Stmt *UnsafeOp, const VarDecl *IndexableVar,
      llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
          &SingleEntities);

  void checkSingleEntityFlowingToIndexableLocalVariableHandleArithmetic(
      const Stmt *UnsafeOp, const VarDecl *IndexableVar,
      llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
          &SingleEntities);

  void checkSingleEntityFlowingToIndexableLocalVariableHandleDeref(
      const Stmt *UnsafeOp, const VarDecl *IndexableVar,
      llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
          &SingleEntities);

  void checkSingleEntityFlowingToIndexableLocalVariableHandleMemberAccess(
      const Stmt *UnsafeOp, const VarDecl *IndexableVar,
      llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
          &SingleEntities);

  void checkSingleEntityFlowingToIndexableLocalVariableHandleEscapingLocal(
      const Stmt *UnsafeOp, const Expr *Operand, const VarDecl *IndexableVar,
      llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
          &SingleEntities,
      UnsafeOpKind Kind);

  void checkSingleEntityFlowingToIndexableLocalVariableHandleEscapingOOBLocal(
      const Stmt *UnsafeOp, const Expr *Operand, const VarDecl *IndexableVar,
      llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
          &SingleEntities,
      UnsafeOpKind Kind);

  void checkSingleEntityFlowingToIndexableLocalVariableHandleCast(
      const Stmt *UnsafeOp, const Expr *Operand, const VarDecl *IndexableVar,
      llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
          &SingleEntities,
      UnsafeOpKind Kind);

  void
  checkSingleEntityFlowingToIndexableLocalVariableHandleCastConvertedToSingle(
      const Stmt *UnsafeOp, const Expr *Operand, const VarDecl *IndexableVar,
      llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
          &SingleEntities,
      UnsafeOpKind Kind);

  bool
  checkSingleEntityFlowingToIndexableLocalVariableHandleUnsafeCastToLargerPointee(
      const Stmt *UnsafeOp, const Expr *Operand, const VarDecl *IndexableVar,
      llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
          &SingleEntities,
      UnsafeOpKind Kind, bool TestOnly);

  using ExtraDynCountLogicFn = std::function<WillTrapKind(
      WillTrapKind, size_t, size_t, std::optional<APInt> &)>;

  void checkSingleEntityFlowingToIndexableLocalVariableHandleCastToDynamicCount(
      const Stmt *UnsafeOp, const Expr *Operand, const VarDecl *IndexableVar,
      llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
          &SingleEntities,
      UnsafeOpKind Kind, ExtraDynCountLogicFn Predicate);

  void
  checkSingleEntityFlowingToIndexableLocalVariableHandleEscapingCastConvertedToDynamicCount(
      const Stmt *UnsafeOp, const Expr *Operand, const VarDecl *IndexableVar,
      llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
          &SingleEntities,
      UnsafeOpKind Kind);

  // Run all individual checks.
  void handleUnsafeOperation(const Stmt *UnsafeOp, const Expr *Operand,
                             UnsafeOpKind Kind) {
    checkSingleEntityFlowingToIndexableLocalVariable(UnsafeOp, Operand, Kind);
  }

  bool ExprIsConstantZero(const Expr *E) {
    auto Result = EvaluateAsInt(E);
    if (!Result)
      return false;
    if (Result->isZero())
      return true;
    return false;
  }

  std::optional<APInt> EvaluateAsInt(const Expr *E) {
    Expr::EvalResult Result;
    if (E->isValueDependent()) {
      // Expr::EvaluateAsInt will assert if we try to call it in this
      // case so just give up.
      return std::optional<APInt>();
    }

    bool success = E->EvaluateAsInt(Result, Ctx);
    if (!success)
      return std::optional<APInt>();
    if (!Result.Val.isInt())
      return std::optional<APInt>();
    return std::optional<APInt>(Result.Val.getInt());
  }

  /// Determines if the provided expression is a __bidi_indexable pointer
  /// that only allows access to it's 0th element.
  ///
  /// \param E - The expression to visit
  ///
  /// \return Tuple (X, Y).
  /// X will be true iff `E` has type `T* __bidi_indexable` and has the bounds
  /// of a `U* __single` where `sizeof(U) == sizeof(T)` and false otherwise.
  ///
  /// If X is true then Y will be the __single pointer type used for the bounds
  /// of the __bidi_indexable pointer.
  std::tuple<bool, QualType>
  IsWidePointerWithBoundsOfSingle(const Expr *) const;

  bool IsReallySinglePtr(const QualType Ty) const {
    if (!Ty->isSinglePointerType())
      return false;

    // Unfortunately __counted_by and friends, and __terminated_by use sugar
    // types wrapping a __single so we need to check for those explicitly and
    // bail in those cases.
    if (Ty->isBoundsAttributedType() || Ty->isValueTerminatedType())
      return false;

    return true;
  };

  bool FindSingleEntity(
      const Expr *Def, const Expr *AssignmentExpr,
      QualType SingleTyUsedForBidiBounds,
      llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
          &Entities);

public:
  UnsafeOperationVisitor(BoundsSafetySuggestionHandler &Handler,
                         const DefinitionMap &DefMap, Sema &S)
      : Handler(Handler), DefMap(DefMap), S(S), Ctx(S.getASTContext()) {}

  void VisitStmt(const Stmt *S) {
    // Recurse normally.
    VisitChildren(S);
  }

  void reset() {
    VisitedOVESourceExprs.clear();
    UnsafeBitCastsToSkip.clear();
  }

  // Unevaluated context visitors
  void VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *) {
    // Unevaluated context such as sizeof()/alignof()/__alignof
    return;
  }

  void VisitGenericSelectionExpr(const GenericSelectionExpr *GSE) {
    // Only the result expr is evaluated in `_Generic`.
    // E.g.
    // _Generic(0, // Not evaluated
    //   int: foo(), // Evaluated
    //   char: bar() // Not evaluated
    // );
    Visit(GSE->getResultExpr());
  }

  // These are the individual unsafe operations we'll react upon.
  void VisitArraySubscriptExpr(const ArraySubscriptExpr *ASE);
  void VisitUnaryOperator(const UnaryOperator *UO);
  void VisitBinaryOperator(const BinaryOperator *BO);
  void VisitMemberExpr(const MemberExpr *ME);
  void VisitOpaqueValueExpr(const OpaqueValueExpr *OVE);
  void VisitDeclStmt(const DeclStmt *DS);
  void VisitReturnStmt(const ReturnStmt *RS);
  void VisitCallExpr(const CallExpr *CE);
  void VisitCastExpr(const CastExpr *CE);
};

} // namespace

std::tuple<bool, QualType>
UnsafeOperationVisitor::IsWidePointerWithBoundsOfSingle(const Expr *E) const {
  const auto Failure = std::make_tuple(false, QualType());
  if (!E->getType()->isPointerTypeWithBounds())
    return Failure;

  // Find the top-most BoundsSafetyPointerCast
  const Expr *Curr = E;
  const Expr *Prev = nullptr;
  const CastExpr *FBPtrCast = nullptr;
  while (Curr && Curr != Prev) {
    Prev = Curr;
    // Walk through ParenExprs
    if (const auto *PE = dyn_cast<ParenExpr>(Curr)) {
      Curr = PE->getSubExpr();
      continue;
    }
    // BoundsSafetyPointerCast can be on either ImplicitCastExpr or
    // ExplicitCastExpr so use `CastExpr` which covers both.
    if (const auto *CE = dyn_cast<CastExpr>(Curr)) {
      if (CE->getCastKind() != clang::CK_BoundsSafetyPointerCast) {
        // Walk through all other casts.
        Curr = CE->getSubExpr();
        continue;
      }
      FBPtrCast = CE;
      break;
    }
  }

  if (!FBPtrCast)
    return Failure; // Failed to find BoundsSafetyPointerCast

  // Check that the BoundsSafetyPointerCast acts on a __single pointer.
  const auto SinglePtrQualTy = FBPtrCast->getSubExpr()->getType();
  if (!IsReallySinglePtr(SinglePtrQualTy))
    return Failure;

  // The resulting `T* __bidi_indexable` of E will get the bounds
  // of a `U* __single`.
  //
  // Now check if `sizeof(T) >= sizeof(U)`. If that's the case then
  // `E[<not zero>]` will be out-of-bounds (where
  // `E` is the expression `E`).
  //
  // When `sizeof(T) <= sizeof(U)` then `E[0]` will be in-bounds.
  //
  // When `sizeof(T) > sizeof(U)` then `E[0]` will also be out-of-bounds.
  // However, due to rdar://119744147 generated code won't trap, so we currently
  // don't try to emit a warning in that situation either (rdar://119775862).

  // Use getAs<PointerType> Walk through AttributedType sugar. E.g.:
  //
  //   AttributedType 0x120024c90 'uint8_t *__bidi_indexable' sugar
  //  `-PointerType 0x120024c60 'uint8_t *__bidi_indexable'
  const auto *IndexablePtrType = E->getType()->getAs<clang::PointerType>();
  assert(IndexablePtrType);

  const auto TypeUsedForIndexing = IndexablePtrType->getPointeeType();
  const auto TypeUsedForIndexingSize =
      Ctx.getTypeSizeOrNull(TypeUsedForIndexing);
  if (TypeUsedForIndexingSize == 0) {
    // This situation already raises `warn_bounds_safety_single_bitcast_lose_bounds`
    // (if there's an explicit cast to a pointer with a sized type) or
    // `err_bounds_safety_incomplete_single_to_indexable` elsewhere.
    return Failure;
  }

  // Walk through AttributedType sugar
  const auto *SinglePointerTy = SinglePtrQualTy->getAs<clang::PointerType>();
  assert(SinglePointerTy);

  const auto SinglePointeeType = SinglePointerTy->getPointeeType();
  const auto SinglePointeeTypeSize = Ctx.getTypeSizeOrNull(SinglePointeeType);
  if (SinglePointeeTypeSize == 0) {
    // This situation isn't legal and raises
    // `err_bounds_safety_incomplete_single_to_indexable`
    // elsewhere.
    return Failure;
  }

  if (TypeUsedForIndexingSize >= SinglePointeeTypeSize)
    return std::make_tuple(true, SinglePointeeType);

  return Failure;
}

bool UnsafeOperationVisitor::FindSingleEntity(
    const Expr *Def, const Expr *AssignmentExpr, QualType SinglePointeeTy,
    llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
        &SingleEntities) {
  assert(IsReallySinglePtr(Def->getType()));

  // Consider the different ways a __single can end up in the variable
  // definition

  // Definition comes directly from a variable
  if (const auto *DRE = dyn_cast<DeclRefExpr>(Def)) {
    const auto *SingleDecl = DRE->getDecl();
    if (!SingleDecl)
      return false;
    const auto *SingleVarDecl = dyn_cast<VarDecl>(SingleDecl);
    if (!SingleVarDecl)
      return false;

    if (const auto *Param = dyn_cast<ParmVarDecl>(SingleVarDecl)) {
      // Variable definition is from a __single parameter
      BoundsSafetySuggestionHandler::SingleEntity Entity = {
          /*Kind*/ BoundsSafetySuggestionHandler::AssignmentSourceKind::
              Parameter,
          /*Entity*/ Param,
          /*AssignmentExpr*/ AssignmentExpr,
          /*SinglePointeeTy*/ SinglePointeeTy};
      SingleEntities.emplace_back(Entity);
      return true;
    }

    if (SingleVarDecl->hasGlobalStorage() && !SingleVarDecl->isStaticLocal()) {
      // Variable definition is from a __single global
      BoundsSafetySuggestionHandler::SingleEntity Entity = {
          /*Kind*/ BoundsSafetySuggestionHandler::AssignmentSourceKind::
              GlobalVar,
          /*Entity*/ SingleVarDecl,
          /*AssignmentExpr*/ AssignmentExpr,
          /*SinglePointeeTy*/ SinglePointeeTy};
      SingleEntities.emplace_back(Entity);
      return true;
    }

    if (SingleVarDecl->isLocalVarDecl() || SingleVarDecl->isStaticLocal()) {
      // Variable definition is from a local __single or a
      // "locally scoped" static __single.
      BoundsSafetySuggestionHandler::SingleEntity Entity = {
          /*Kind*/ BoundsSafetySuggestionHandler::AssignmentSourceKind::
              LocalVar,
          /*Entity*/ SingleVarDecl,
          /*AssignmentExpr*/ AssignmentExpr,
          /*SinglePointeeTy*/ SinglePointeeTy};
      SingleEntities.emplace_back(Entity);
      return true;
    }

    return false;
  }

  // __single comes from value returned by function call
  if (const auto *CE = dyn_cast<CallExpr>(Def)) {
    const auto *DirectCallDecl = CE->getDirectCallee();
    if (!DirectCallDecl) {
      // Don't support indirect calls for now.
      return false;
    }
    if (DirectCallDecl->hasAttr<AllocSizeAttr>() &&
        IsReallySinglePtr(DirectCallDecl->getReturnType())) {
      // Functions declared like
      // void * custom_malloc(size_t s) __attribute__((alloc_size(1)))
      //
      // are currently are annotated as returning `void *__single` rather
      // than `void *__sized_by(s)`. To make the right thing happen at call
      // sites `BoundsSafetyPointerPromotionExpr` is used to generate a pointer
      // with the appropriate bounds from the `void *__single`. For functions
      // like this the warning needs to be suppressed because from the user's
      // perspective the returned value is not actually __single.
      //
      // This code path can be deleted once allocating functions are properly
      // annotated with __sized_by_or_null. rdar://117114186
      return false;
    }

    assert(IsReallySinglePtr(DirectCallDecl->getReturnType()));

    BoundsSafetySuggestionHandler::SingleEntity Entity = {
        /*Kind*/ BoundsSafetySuggestionHandler::AssignmentSourceKind::
            FunctionCallReturnValue,
        /*Entity*/ DirectCallDecl,
        /*AssignmentExpr*/ AssignmentExpr,
        /*SinglePointeeTy*/ SinglePointeeTy};
    SingleEntities.emplace_back(Entity);
    return true;
  }

  auto HasSingleFromStructOrUnion = [&](const Expr *E) -> bool {
    if (const auto *ME = dyn_cast<MemberExpr>(E)) {
      if (const auto *FD = dyn_cast<FieldDecl>(ME->getMemberDecl())) {
        BoundsSafetySuggestionHandler::AssignmentSourceKind AssignmentKind;
        if (FD->getParent()->isUnion()) {
          AssignmentKind =
              BoundsSafetySuggestionHandler::AssignmentSourceKind::UnionMember;
        } else {
          assert(FD->getParent()->isStruct());
          AssignmentKind =
              BoundsSafetySuggestionHandler::AssignmentSourceKind::StructMember;
        }

        BoundsSafetySuggestionHandler::SingleEntity Entity = {
            /*Kind*/ AssignmentKind,
            /*Entity*/ FD,
            /*AssignmentExpr*/ AssignmentExpr,
            /*SinglePointeeTy*/ SinglePointeeTy};
        SingleEntities.emplace_back(Entity);
        return true;
      }
    }
    return false;
  };

  // __single comes from array access
  if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(Def)) {
    // The `IgnoreImpCasts()` on `ASE->getBase()` is necessary here to walk
    // through MaterializeSequenceExpr, BoundsSafetyPointerPromotionExpr,
    // OpaqueValueExpr, and ImplicitCastExpr.
    const Expr *BaseExpr = ASE->getBase()->IgnoreImpCasts();

    // Array access is on a variable.
    if (const auto *DRE = dyn_cast<DeclRefExpr>(BaseExpr)) {
      if (const auto *ArrayDecl = dyn_cast<VarDecl>(DRE->getDecl())) {
        BoundsSafetySuggestionHandler::SingleEntity Entity = {
            /*Kind*/ BoundsSafetySuggestionHandler::AssignmentSourceKind::
                ArrayElement,
            /*Entity*/ ArrayDecl,
            /*AssignmentExpr*/ AssignmentExpr,
            /*SinglePointeeTy*/ SinglePointeeTy};
        SingleEntities.emplace_back(Entity);
        return true;
      }
    }

    // Array access is on struct/union member
    if (HasSingleFromStructOrUnion(BaseExpr))
      return true;
  }

  // __single comes from direct access to a struct/union member
  if (HasSingleFromStructOrUnion(Def))
    return true;

  return false;
}

void UnsafeOperationVisitor::checkSingleEntityFlowingToIndexableLocalVariable(
    const Stmt *UnsafeOp, const Expr *Operand, UnsafeOpKind Kind) {

  const Expr *PreviousOperand = nullptr;
  const Expr *CurrentOperand = Operand;

  while (CurrentOperand != PreviousOperand) {
    PreviousOperand = CurrentOperand;
    if (const auto *OVE = dyn_cast<OpaqueValueExpr>(CurrentOperand)) {
      CurrentOperand = OVE->getSourceExpr();
      continue;
    }
    // This is quite broad, but such transformation typically results in
    // an expression with roughly the same value, i.e. it refers to the same
    // object. The only exception from this rule is, implicit casts of
    // CK_LValueToRValue kind, which constitute memory load operations.
    // But we don't expect any of that coming in.
    CurrentOperand = CurrentOperand->IgnoreParenCasts();
  }

  const auto *VarDRE = dyn_cast<DeclRefExpr>(CurrentOperand);
  if (!VarDRE)
    return;

  const auto *Var = dyn_cast<VarDecl>(VarDRE->getDecl());
  if (!Var || !Var->isLocalVarDecl())
    return;

  QualType VarTy = Var->getType();

  // It might be that we've discarded an explicit cast from an integer
  // to a pointer, so let's double-check.
  if (!VarTy->isPointerTypeWithBounds())
    return;

  // Check that the variable is *implicitly* __bidi_indexable.
  // Explicitly __bidi_indexable variables are covered by
  // -Wbounds-attributes-implicit-conversion-single-to-explicit-indexable.
  //
  // We use `hasAttr` to walk through additional AttributedTypes that may be
  // present.
  if (!VarTy->hasAttr(attr::PtrAutoAttr))
    return;

  auto DefI = DefMap.find(Var); // The map is const, can't use [].
  if (DefI == DefMap.end())
    return;

  const DefinitionList &Defs = DefI->second;

  llvm::SmallVector<BoundsSafetySuggestionHandler::SingleEntity, 2>
      SingleEntities;

  if (Defs.size() == 0)
    return;

  // Walk over all the potential definitions that the __bidi_indexable variable
  // might take. If they are all __single and can all be identified as
  // an "entity" then a warning is emitted.
  for (const auto *Def : Defs) {
    // Skip emitting warnings if any of the definitions aren't T*
    // __bidi_indexable with the bounds of a T* __single.
    bool IsWPWBOS = false;
    QualType SingleTyUsedForBidiBounds;
    std::tie(IsWPWBOS, SingleTyUsedForBidiBounds) =
        IsWidePointerWithBoundsOfSingle(Def);
    if (!IsWPWBOS)
      return;

    // It's ok to walk through all casts (including explicit) at this point
    // because `IsWidePointerWithBoundsOfSingle` has already checked that this
    // is a scenario where we should warn.
    const Expr *DefWithoutParensAndCasts = Def->IgnoreParenCasts();

    // Skip emitting warnings if any of the definitions are not "effectively"
    // __single. We can only be sure a `__single` flowed into an indexable
    // variable in this case.
    if (!IsReallySinglePtr(DefWithoutParensAndCasts->getType()))
      return;

    const Expr *AssignmentExpr = nullptr;
    if (Def != Var->getInit()) {
      // The definition comes from assignment (i.e. not an initializer).
      AssignmentExpr = DefWithoutParensAndCasts;
    }
    // Skip emitting the warning if the entity for any of the definitions can't
    // be found.
    if (!FindSingleEntity(DefWithoutParensAndCasts, AssignmentExpr,
                          SingleTyUsedForBidiBounds, SingleEntities))
      return;
  }

  // Dispatch based on the unsafe operation.
  switch (Kind) {
  case BoundsSafetySuggestionHandler::UnsafeOpKind::Arithmetic:
    checkSingleEntityFlowingToIndexableLocalVariableHandleArithmetic(
        UnsafeOp, Var, SingleEntities);
    break;
  case BoundsSafetySuggestionHandler::UnsafeOpKind::Index:
    checkSingleEntityFlowingToIndexableLocalVariableHandleIndexing(
        UnsafeOp, Var, SingleEntities);
    break;
  case clang::BoundsSafetySuggestionHandler::UnsafeOpKind::Deref:
    checkSingleEntityFlowingToIndexableLocalVariableHandleDeref(UnsafeOp, Var,
                                                                SingleEntities);
    break;
  case clang::BoundsSafetySuggestionHandler::UnsafeOpKind::MemberAccess:
    checkSingleEntityFlowingToIndexableLocalVariableHandleMemberAccess(
        UnsafeOp, Var, SingleEntities);
    break;
  case clang::BoundsSafetySuggestionHandler::UnsafeOpKind::Assignment:
  case clang::BoundsSafetySuggestionHandler::UnsafeOpKind::Return:
  case clang::BoundsSafetySuggestionHandler::UnsafeOpKind::CallArg:
    checkSingleEntityFlowingToIndexableLocalVariableHandleEscapingLocal(
        UnsafeOp, Operand, Var, SingleEntities, Kind);
    break;
  case clang::BoundsSafetySuggestionHandler::UnsafeOpKind::Cast:
    checkSingleEntityFlowingToIndexableLocalVariableHandleCast(
        UnsafeOp, Operand, Var, SingleEntities, Kind);
    break;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
  default:
    llvm_unreachable("Unhandled UnsafeOpKind");
#pragma clang diagnostic pop
  }
}

static bool AccessingElementZeroIsOOB(
    ASTContext &Ctx, const QualType EltTy,
    llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
        &SingleEntities) {
  auto EltSize = Ctx.getTypeSizeOrNull(EltTy);
  if (EltSize == 0)
    return false;
  // If the EltTy used for dereferencing is larger than all the potential
  // __single bounds i then this out-of-bounds. This currently won't trap in
  // some cases due to rdar://104845295 but it will be in the future so we
  // should still warn.
  bool DefinitelyOOB = std::all_of(
      SingleEntities.begin(), SingleEntities.end(),
      [&](const BoundsSafetySuggestionHandler::SingleEntity &S) -> bool {
        auto SingleSize = Ctx.getTypeSizeOrNull(S.SinglePointeeTy);
        if (SingleSize == 0)
          return false;
        return EltSize > SingleSize;
      });
  return DefinitelyOOB;
}

static QualType GetLargestSinglePointee(
    llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
        &SingleEntities,
    ASTContext &Ctx) {
  QualType LargestSinglePointee = SingleEntities.front().SinglePointeeTy;
  auto LargestPointeeSizeInBits = Ctx.getTypeSizeOrNull(LargestSinglePointee);
  for (const auto &SingleEntity : SingleEntities) {
    const auto CurrentPointeeTy = SingleEntity.SinglePointeeTy;
    auto CurrentPointeeSizeInBits = Ctx.getTypeSizeOrNull(CurrentPointeeTy);
    if (CurrentPointeeSizeInBits > LargestPointeeSizeInBits) {
      LargestSinglePointee = CurrentPointeeTy;
      LargestPointeeSizeInBits = CurrentPointeeSizeInBits;
    }
  }
  return LargestSinglePointee;
}

struct PointerArithInBoundsInfo {
  // Any effective offset >= to this is out-of-bounds.
  // Note the offset is assumed to be in units of the pointee
  // type used for the pointer arithmetic.
  //
  // The "effective offset" is the offset assuming the operation
  // was an addition operation.
  size_t MinimumOOBPositiveOffset;

  BoundsSafetySuggestionHandler::PtrArithOOBKind IsOOB;

  PointerArithInBoundsInfo()
      : MinimumOOBPositiveOffset(0),
        IsOOB(BoundsSafetySuggestionHandler::PtrArithOOBKind::UNKNOWN) {}
};

/// Compute if pointer arithmetic would be in bounds and other related info.
///
/// \param PointeeTySizeInBits The pointee size used for pointer arithmetic.
/// \param AvailableBits The number bits available in the memory pointed to by
/// the pointer. \param Offset If set it is the offset that would be used.
///
/// Example:
///
/// PointeeTySizeInBits = 1*8 = 8
/// AvailableBits = 4*8 = 32
/// Offset = Unset
/// OpIsIncrement = true
/// OffsetIsSignedTy = false (size_t is unsigned)
///
/// Assume sizeof(int) = 4 and sizeof(char) = 1. This corresponds to
///
/// ```
/// void foo(int* p, size_t offset) {
///    int* local = p;
///    ((char*)local + offset);
/// }
PointerArithInBoundsInfo ComputeIfPointerArithIsInBounds(
    size_t PointeeTySizeInBits, size_t AvailableBits,
    std::optional<APInt> Offset, bool OpIsIncrement, bool OffsetIsSignedTy) {
  PointerArithInBoundsInfo Result;
  using PtrArithOOBKind = BoundsSafetySuggestionHandler::PtrArithOOBKind;

  // Compute the smallest offset that would be out-of-bounds.
  Result.MinimumOOBPositiveOffset = AvailableBits / PointeeTySizeInBits;

  if (Result.MinimumOOBPositiveOffset == 0) {
    // Special case: This means any offset >= 0 would cause an out-of-bounds
    // access. This means **any** offset would be out-of-bounds.
    //
    // This will happen if PointeeTySizeInBits > Availablebits.
    //
    // E.g.
    // ```
    // uint32_t* local;
    // *((uint64_t*)local + offset) = 0;
    // ```
    Result.IsOOB = PtrArithOOBKind::ALWAYS_OOB_BASE_OOB;
    return Result;
  }

  if (!Offset.has_value()) {
    // The offset isn't known

    if (OffsetIsSignedTy) {
      // Effective offset can be negative or positive or zero
      Result.IsOOB = PtrArithOOBKind::
          OOB_IF_EFFECTIVE_OFFSET_GTE_MINIMUM_OOB_POSITIVE_OFFSET_OR_LT_ZERO;
    } else {
      // Effective offset can be positive or zero
      if (OpIsIncrement) {
        // Addition operation
        // Effective offset can be zero or positive (cannot be negative)
        Result.IsOOB = PtrArithOOBKind::
            OOB_IF_EFFECTIVE_OFFSET_GTE_MINIMUM_OOB_POSITIVE_OFFSET;
      } else {
        // Subtract operation
        // Effective offset can be negative or zero (cannot be positive)
        Result.IsOOB = PtrArithOOBKind::OOB_IF_EFFECTIVE_OFFSET_LT_ZERO;
      }
    }
    return Result;
  }

  // Offset is a constant

  // Compute the "EffectiveOffset" which is the offset that would be used
  // assuming the operation was always a pointer increment operation. I.e.:
  //
  // ptr + EffectiveOffset
  //
  APInt EffectiveOffset = Offset.value();
  if (!OpIsIncrement) {
    // The operation does pointer subtraction.
    // Note if `EffectiveOffset.isMinSignedValue()`
    // then the negation will get the same value back.

    EffectiveOffset.negate();
  }

  if (EffectiveOffset.isNegative()) {
    // Any negative offset is out-of-bounds
    Result.IsOOB = PtrArithOOBKind::ALWAYS_OOB_CONSTANT_OFFSET;
    return Result;
  }

  // If this condition is true
  //
  // EffectiveOffset >= (AvailableBits / PointeeTySizeInBits)
  //
  // then then a memory access at the offset would access out-of-bounds
  // memory.
  assert(!EffectiveOffset.isNegative());
  auto MinimumOOBOffset =
      APInt(EffectiveOffset.getBitWidth(), Result.MinimumOOBPositiveOffset);

  if (EffectiveOffset.uge(MinimumOOBOffset))
    Result.IsOOB = PtrArithOOBKind::ALWAYS_OOB_CONSTANT_OFFSET;
  else
    Result.IsOOB = PtrArithOOBKind::NEVER_OOB;

  return Result;
}

void UnsafeOperationVisitor::
    checkSingleEntityFlowingToIndexableLocalVariableHandleIndexing(
        const Stmt *UnsafeOp, const VarDecl *IndexableVar,
        llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
            &SingleEntities) {
  auto *ASE = cast<ArraySubscriptExpr>(UnsafeOp);
  // Determine the pointee sized used for indexing
  const auto BasePtrTy = ASE->getBase()->getType();
  const auto IndexingPointeeTy = BasePtrTy->getPointeeType();

  if (!BasePtrTy->isPointerTypeWithBounds()) {
    // This path can be hit if the __bidi_indexable is casted to a different
    // attribute type before being indexed. E.g.:
    //
    // ```
    // int* __single p;
    // int* local = p;
    // ((int* __single) local)[0] = 0;
    // ```
    //
    // The analysis here currently assumes the base pointer is __bidi_indexable
    // so don't try to proceed further.
    return;
  }

  const auto IndexingPointeeTyInBits = Ctx.getTypeSizeOrNull(IndexingPointeeTy);
  assert(IndexingPointeeTyInBits > 0);

  // Determine the max possible bounds that the __bidi_indexable could
  // be storing. We use the maximum to reduce chances of false positives.
  const auto LargestSinglePointeeTy =
      GetLargestSinglePointee(SingleEntities, Ctx);
  const auto LargestSinglePointeeTyInBits =
      Ctx.getTypeSizeOrNull(LargestSinglePointeeTy);

  // Try to evaluate the index as a constant
  const auto KnownOffset = EvaluateAsInt(ASE->getIdx());
  bool IndexTyIsSigned = ASE->getIdx()->getType()->isSignedIntegerType();
  const auto PABInfo = ComputeIfPointerArithIsInBounds(
      IndexingPointeeTyInBits, LargestSinglePointeeTyInBits, KnownOffset,
      /*OpIsIncrement=*/true, /*OffsetIsSignedTy=*/IndexTyIsSigned);

  if (PABInfo.IsOOB == PtrArithOOBKind::NEVER_OOB)
    return;

  if (PABInfo.IsOOB == PtrArithOOBKind::ALWAYS_OOB_BASE_OOB) {
    // Accessing any index is out-of-bounds
    const auto *IndexableVarExpr = ASE->getBase();
    // Accessing element is guaranteed to be out-of-bounds. This currently
    // won't trap due to rdar://104845295 but it will be in the future so we
    // should still warn.
    Handler.handleSingleEntitiesFlowingToIndexableVariableWithEltZeroOOB(
        SingleEntities, IndexableVar, UnsafeOp, IndexableVarExpr,
        UnsafeOpKind::Index);
    return;
  }

  Handler.handleSingleEntitiesFlowingToIndexableVariableIndexOrPtrArith(
      SingleEntities, IndexableVar, UnsafeOp, UnsafeOpKind::Index,
      PABInfo.IsOOB, PABInfo.MinimumOOBPositiveOffset);
}

void UnsafeOperationVisitor::
    checkSingleEntityFlowingToIndexableLocalVariableHandleArithmetic(
        const Stmt *UnsafeOp, const VarDecl *IndexableVar,
        llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
            &SingleEntities) {
  const Expr *E = cast<Expr>(UnsafeOp);
  const auto PointeeTy = E->getType()->getAs<clang::PointerType>()->getPointeeType();
  if (PointeeTy->isIncompleteType())
    return;
  const auto PointeeTySizeInBits = Ctx.getTypeSizeOrNull(PointeeTy);

  const auto LargestSinglePointee =
      GetLargestSinglePointee(SingleEntities, Ctx);
  const auto LargestSinglePointeeInBits =
      Ctx.getTypeSizeOrNull(LargestSinglePointee);

  size_t MinimumOOBOffset = 0;
  PtrArithOOBKind IsOOB = PtrArithOOBKind::UNKNOWN;

  auto ShouldWarn = [&](std::optional<APInt> KnownOffset, bool OpIsIncrement,
                        bool OffsetIsSignedTy) -> bool {
    auto Info = ComputeIfPointerArithIsInBounds(
        PointeeTySizeInBits, LargestSinglePointeeInBits, KnownOffset,
        OpIsIncrement, OffsetIsSignedTy);

    MinimumOOBOffset = Info.MinimumOOBPositiveOffset;
    IsOOB = Info.IsOOB;
    assert(IsOOB != PtrArithOOBKind::UNKNOWN);

    if (IsOOB == PtrArithOOBKind::NEVER_OOB)
      return false;

    return true;
  };

  if (auto *UO = dyn_cast<UnaryOperator>(UnsafeOp)) {
    std::optional<bool> IsIncrement;
    switch (UO->getOpcode()) {
    case UO_PostInc:
    case UO_PreInc:
      IsIncrement = true;
      break;
    case UO_PreDec:
    case UO_PostDec:
      IsIncrement = false;
      break;
    default:
      llvm_unreachable("Unhandled UnaryOperator");
    }
    assert(IsIncrement.has_value());
    if (!ShouldWarn(/*KnownOffset=*/APInt(/*numBits=*/64, 1),
                    /*OpIsIncrement=*/IsIncrement.value(),
                    /*OffsetIsSignedTy=*/false))
      return;

    Handler.handleSingleEntitiesFlowingToIndexableVariableIndexOrPtrArith(
        SingleEntities, IndexableVar, UnsafeOp, UnsafeOpKind::Arithmetic, IsOOB,
        MinimumOOBOffset);

    return;
  }
  if (auto *BO = dyn_cast<BinaryOperator>(UnsafeOp)) {
    const Expr *Offset;

    bool IsIncrement = false;
    switch (BO->getOpcode()) {
    case BO_Add:
      // n + ptr or ptr + n
    case BO_AddAssign:
      // ptr += n (i.e. ptr = ptr + n)
      IsIncrement = true;
      break;
    case BO_Sub:
      // ptr - n
    case BO_SubAssign:
      // ptr -= n (i.e. ptr = ptr - n)
      IsIncrement = false;
      break;
    default:
      llvm_unreachable("Unhandled BinaryOperator");
    }

    if (BO->getLHS()->getType()->isPointerType()) {
      Offset = BO->getRHS();
    } else {
      Offset = BO->getLHS();
    }
    assert(Offset->getType()->isIntegralOrEnumerationType());
    bool OffsetIsSignedTy =
        Offset->getType()->isSignedIntegerOrEnumerationType();

    auto EvaluatedOffset = EvaluateAsInt(Offset);
    if (!ShouldWarn(/*KnownOffset=*/EvaluatedOffset,
                    /*OpIsIncrement=*/IsIncrement,
                    /*OffsetIsSignedTy=*/OffsetIsSignedTy))
      return;

    Handler.handleSingleEntitiesFlowingToIndexableVariableIndexOrPtrArith(
        SingleEntities, IndexableVar, UnsafeOp, UnsafeOpKind::Arithmetic, IsOOB,
        MinimumOOBOffset);
    return;
  }
  llvm_unreachable("Unhandled UnsafeOp");
}

void UnsafeOperationVisitor::
    checkSingleEntityFlowingToIndexableLocalVariableHandleDeref(
        const Stmt *UnsafeOp, const VarDecl *IndexableVar,
        llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
            &SingleEntities) {

  if (auto *UO = dyn_cast<UnaryOperator>(UnsafeOp)) {
    assert(UO->getOpcode() == UO_Deref);
    const auto *BasePtrTy = UO->getSubExpr()->getType()->getAs<clang::PointerType>();
    assert(BasePtrTy);
    const auto BasePointeeTy = BasePtrTy->getPointeeType();

    if (!BasePtrTy->isPointerTypeWithBounds()) {
      // It's possible that the there are casts that change the attribute type.
      // E.g.
      //
      // void consume(int* p) {
      //   int* ptr = p;
      //   *((int* __single) ptr) = 0;
      // }
      return;
    }

    if (!AccessingElementZeroIsOOB(Ctx, BasePointeeTy, SingleEntities))
      return;

    const auto *IndexableVarExpr = UO->getSubExpr();
    Handler.handleSingleEntitiesFlowingToIndexableVariableWithEltZeroOOB(
        SingleEntities, IndexableVar, UnsafeOp, IndexableVarExpr,
        UnsafeOpKind::Deref);
    return;
  }

  llvm_unreachable("Unhandled UnsafeOp");
}

void UnsafeOperationVisitor::
    checkSingleEntityFlowingToIndexableLocalVariableHandleMemberAccess(
        const Stmt *UnsafeOp, const VarDecl *IndexableVar,
        llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
            &SingleEntities) {
  const auto *ME = cast<MemberExpr>(UnsafeOp);
  const auto *BasePtrTy = ME->getBase()->getType()->getAs<clang::PointerType>();
  assert(BasePtrTy);
  const auto BasePointeeTy = BasePtrTy->getPointeeType();

  if (!BasePtrTy->isPointerTypeWithBounds()) {
    // It's possible that the there are casts that change the attribute type.
    // E.g.
    //
    // void consume(struct Foo* p) {
    //   struct Foo* ptr = p;
    //   *((struct Bar* __single) ptr) = 0;
    // }
    return;
  }

  // This is rather subtle but access to **any** field in the struct (not just
  // the fields above the upper bound of the wide pointer) will trap because
  // `CodeGenFunction::EmitMemberExpr` checks that "one-past-the-end" (assuming
  // the pointee type of the wide pointer) is <= the upper bound. This check
  // will always fail if the bounds stored in the wide pointer are for a smaller
  // type.
  //
  // So it is not sufficient to check if the field being accessed is above the
  // upper bound. Instead check if **any** part of the struct will be
  // out-of-bounds which corresponds directly to the check that
  // `CodeGenFunction::EmitMemberExpr` does.
  if (!AccessingElementZeroIsOOB(Ctx, BasePointeeTy, SingleEntities))
    return;

  const auto *IndexableVarExpr = ME->getBase();
  Handler.handleSingleEntitiesFlowingToIndexableVariableWithEltZeroOOB(
      SingleEntities, IndexableVar, UnsafeOp, IndexableVarExpr,
      UnsafeOpKind::MemberAccess);
}

void UnsafeOperationVisitor::
    checkSingleEntityFlowingToIndexableLocalVariableHandleEscapingLocal(
        const Stmt *UnsafeOp, const Expr *Operand, const VarDecl *IndexableVar,
        llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
            &SingleEntities,
        UnsafeOpKind Kind) {
  // Note. This is "escape" in the sense that the indexable variable is being
  // assigned to something else which means it "escapes" the analysis (the
  // bounds are no longer tracked due to the fact transitive assignments are not
  // tracked) and not necessarily "escapes" in the sense that the pointer
  // escapes the current function (although that is what happens in some cases).

  checkSingleEntityFlowingToIndexableLocalVariableHandleEscapingOOBLocal(
      UnsafeOp, Operand, IndexableVar, SingleEntities, Kind);

  checkSingleEntityFlowingToIndexableLocalVariableHandleEscapingCastConvertedToDynamicCount(
      UnsafeOp, Operand, IndexableVar, SingleEntities, Kind);
}

void UnsafeOperationVisitor::
    checkSingleEntityFlowingToIndexableLocalVariableHandleEscapingOOBLocal(
        const Stmt *UnsafeOp, const Expr *Operand, const VarDecl *IndexableVar,
        llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
            &SingleEntities,
        UnsafeOpKind Kind) {
  const auto IndexableEltTy =
      IndexableVar->getType()->getAs<clang::PointerType>()->getPointeeType();

  // Note: We deliberately don't check the pointer attributes on `Operand`
  // as `IndexableVar` could have been casted to have a different pointer
  // attribute and we still want to warn about it escaping the analysis.

  // Using `IndexableEltTy` as the element type is "technically" wrong.
  // The correct element type is actually:
  //
  // Operand->getType()->getPointeeType()
  //
  // However, doing this causes many more warnings to be emitted because
  // we will warn about unsafe casts at the use site that we already warned
  // about. E.g.:
  //
  // ```
  // int* consume(char*p) {
  //   char* local = p;
  //   // The cast already causes a warning to be emitted.
  //   return (int*) local;
  // }
  // ```
  //
  // To avoid emitting these extra warnings we use `IndexableEltTy` instead
  // which indirectly avoids emitting warnings for casts we already warned about
  // but still allows us to warn about when the unsafe cast at assignment to
  // the local `__bidi_indexable.` E.g.:
  //
  // ```
  // int* consume(char* p) {
  //   int* local = (int* __bidi_indexable)(char* __bidi_indexable) p;
  //   return local; // Warn here
  // }
  // ```
  //
  // Revisiting this design decision is tracked by rdar://123654605.
  //
  if (!AccessingElementZeroIsOOB(Ctx, IndexableEltTy, SingleEntities))
    return;
  // The local __bidi_indexable does not have bounds sufficient to access a
  // single element so warn when this pointer escapes the analysis.

  Handler.handleSingleEntitiesFlowingToIndexableVariableWithEltZeroOOB(
      SingleEntities, IndexableVar, UnsafeOp, Operand, Kind);
}

void UnsafeOperationVisitor::
    checkSingleEntityFlowingToIndexableLocalVariableHandleCastConvertedToSingle(
        const Stmt *UnsafeOp, const Expr *Operand, const VarDecl *IndexableVar,
        llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
            &SingleEntities,
        UnsafeOpKind Kind) {
  // Look for unsafe `__bidi_indexable -> __single conversion (causes a bounds
  // check).
  auto *CE = cast<CastExpr>(UnsafeOp);

  if (!IsReallySinglePtr(CE->getType())) {
    return;
  }

  assert(CE->getSubExpr() == Operand);

  // Make sure this is a `__bidi_indexable/__indexable -> __single` conversion
  if (CE->getCastKind() != clang::CK_BoundsSafetyPointerCast)
    return;
  const auto OperandTy = Operand->getType();
  if (!OperandTy->isPointerTypeWithBounds())
    return;
  const auto EltTy = OperandTy->getAs<clang::PointerType>()->getPointeeType();

  if (!AccessingElementZeroIsOOB(Ctx, EltTy, SingleEntities))
    return;

  // At this cast a bounds check will be emitted. The pointee type on the
  // cast is larger than the bounds that will be stored in `IndexableVar` at
  // runtime. This should be a trap but currently isn't due to rdar://104845295
  // but will be in the future so warn about this.
  Handler.handleSingleEntitiesFlowingToIndexableVariableUnsafelyCasted(
      SingleEntities, IndexableVar, UnsafeOp, Kind, Operand);

  // Try to suppress warnings about unsafe bit casts in the expr tree of the
  // operand given that we've already warned about this trapping cast.
  const auto IndexableVarPointeeTy =
      IndexableVar->getType()->getAs<clang::PointerType>()->getPointeeType();
  if (EltTy != IndexableVarPointeeTy) {
    const Expr *Current = Operand;
    const Expr *Previous = nullptr;
    while (Current != Previous) {
      Previous = Current;
      if (const auto *PE = dyn_cast<ParenExpr>(Current)) {
        Current = PE->getSubExpr();
        continue;
      }
      if (const auto *CE = dyn_cast<CastExpr>(Current)) {
        if (CE->getCastKind() == clang::CK_BitCast) {
          if (checkSingleEntityFlowingToIndexableLocalVariableHandleUnsafeCastToLargerPointee(
                  CE, CE->getSubExpr(), IndexableVar, SingleEntities, Kind,
                  /*TestOnly=*/true)) {
            // If this bit cast was visited it would be warned about so add it
            // to the set of bitcasts to skip.
            UnsafeBitCastsToSkip.insert(CE);
          }
        }
        Current = CE->getSubExpr();
      }
    }
  }
}

void UnsafeOperationVisitor::
    checkSingleEntityFlowingToIndexableLocalVariableHandleCast(
        const Stmt *UnsafeOp, const Expr *Operand, const VarDecl *IndexableVar,
        llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
            &SingleEntities,
        UnsafeOpKind Kind) {
  checkSingleEntityFlowingToIndexableLocalVariableHandleUnsafeCastToLargerPointee(
      UnsafeOp, Operand, IndexableVar, SingleEntities, Kind,
      /*TestOnly=*/false);
  checkSingleEntityFlowingToIndexableLocalVariableHandleCastConvertedToSingle(
      UnsafeOp, Operand, IndexableVar, SingleEntities, Kind);
  checkSingleEntityFlowingToIndexableLocalVariableHandleCastToDynamicCount(
      UnsafeOp, Operand, IndexableVar, SingleEntities, Kind,
      /*Predicate=*/nullptr);
}

bool UnsafeOperationVisitor::
    checkSingleEntityFlowingToIndexableLocalVariableHandleUnsafeCastToLargerPointee(
        const Stmt *UnsafeOp, const Expr *Operand, const VarDecl *IndexableVar,
        llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
            &SingleEntities,
        UnsafeOpKind Kind, bool TestOnly) {

  // Report casts that bitcast __bidi_indexable/__indexable pointer
  // to a larger pointee type where we know the bounds at runtime will
  // be too small.
  auto *CE = cast<CastExpr>(UnsafeOp);

  if (UnsafeBitCastsToSkip.contains(CE)) {
    // This cast was already reported elsewhere so skip it.
    return false;
  }

  if (CE->getCastKind() != clang::CK_BitCast)
    return false;

  // The result needs to be a __bidi_indexable/__indexable pointer
  if (!CE->getType()->isPointerTypeWithBounds())
    return false;

  // This is a BitCast so the pointer attribute should not have changed.
  assert(CE->getSubExpr()->getType()->isPointerTypeWithBounds());

  // Go through all __single entities and find the largest pointee type.
  QualType LargestSinglePointee = GetLargestSinglePointee(SingleEntities, Ctx);
  const auto LargestSinglePointeeSizeInBits =
      Ctx.getTypeSizeOrNull(LargestSinglePointee);
  if (LargestSinglePointeeSizeInBits == 0)
    return false;

  // Get size of pointee of CastExpr
  auto CEPointeeTy = CE->getType()->getAs<clang::PointerType>()->getPointeeType();
  const auto CEPointeeTySizeInBits = Ctx.getTypeSize(CEPointeeTy);
  if (CEPointeeTySizeInBits == 0)
    return false;

  if (CEPointeeTySizeInBits <= LargestSinglePointeeSizeInBits)
    return false;

  if (TestOnly)
    return true;

  // Found unsafe bitcast
  UnsafeBitCastsToSkip.insert(CE);
  Handler.handleSingleEntitiesFlowingToIndexableVariableUnsafelyCasted(
      SingleEntities, IndexableVar, UnsafeOp, Kind, Operand);
  return true;
}

using WillTrapKind = clang::BoundsSafetySuggestionHandler::WillTrapKind;
static WillTrapKind
TryDetermineIfBidiIndexableToDynamicCountWithConstantCountTraps(
    const CountAttributedType *DCPT, const APInt &CountExprAsConstant,
    size_t LargestSinglePointeeTySizeInBits, size_t CastPointeeTySizeInBits) {
  const auto CountExprBitWidth = CountExprAsConstant.getBitWidth();
  if (CountExprAsConstant.isZero()) {
    // The bounds in the `__bidi_indexable` are guaranteed to be >= 0
    // bytes/objects so this can't trap.
    return WillTrapKind::NoTrap;
  }
  if (DCPT->isCountInBytes()) {
    // __sized_by(_or_null)
    auto LargestSinglePointeeTySizeInBytes =
        llvm::APInt(CountExprBitWidth, LargestSinglePointeeTySizeInBits / 8);
    if (CountExprAsConstant.ule(LargestSinglePointeeTySizeInBytes)) {
      // The bounds of the __bidi_indexable are larger or equal to the
      // constant byte count. In this case the only possible trap is caused
      // by the pointer being null.
      if (DCPT->isOrNull()) {
        // Won't trap even if nullptr passed
        return WillTrapKind::NoTrap;
      }
      // Won't trap unless ptr is null
      return WillTrapKind::TrapIffPtrNull;
    }

    // The bounds of the __bidi_indexable are smaller than the constant
    // byte count expected. So the bounds are insufficient. Whether or not
    // this traps depends on the attribute type and the pointer value.
    if (DCPT->isOrNull()) {
      // If the pointer is not null then the bounds are insufficient, so
      // this traps.
      return WillTrapKind::TrapIffPtrNotNull;
    }
    // Always traps
    return WillTrapKind::Trap;
  }

  // __counted_by(_or_null)
  const auto CastPointeeTySizeInBitsAP =
      APInt(CountExprBitWidth, CastPointeeTySizeInBits);
  const auto MinimumBufferSizeInBitsAP =
      CastPointeeTySizeInBitsAP * CountExprAsConstant;
  const auto LargestSinglePointeeTySizeInBitsAP =
      APInt(CountExprBitWidth, LargestSinglePointeeTySizeInBits);

  if (LargestSinglePointeeTySizeInBitsAP.uge(MinimumBufferSizeInBitsAP)) {
    // The number of bits the __counted_by_(or_null) expects is <= to
    // the number of bits of the `__single` stored in the `__bidi_indexable`
    // pointer. So at the point of conversion the bounds in the
    // `__bidi_indexable` are big enough.
    //
    // If the pointee sizes are the same then this is just __counted_by(1) or
    // __counted_by_or_null(1)
#ifndef NDEBUG
    if (LargestSinglePointeeTySizeInBitsAP == CastPointeeTySizeInBitsAP)
      assert(CountExprAsConstant.isOne());
#endif

    if (DCPT->isOrNull()) {
      // __counted_by_or_null: Won't trap. Even if nullptr passed
      return WillTrapKind::NoTrap;
    }
    // __counted_by: Traps if and only if ptr is null
    return WillTrapKind::TrapIffPtrNull;
  }

  // LargestSinglePointeeTySizeInBitsAP < MinimumBufferSizeInBitsAP
  // The number of bits the __counted_by_(or_null) expects is >
  // the number of bits of the `__single` stored in the `__bidi_indexable`.
  // So at the point of conversion the bounds in the `__bidi_indexable`
  // are **not** big enough.
  //
  // If the pointee sizes are the same then this is
  // `__counted_by(<const>)` or `__counted_by_or_null(<const>)`
  // where <const> is > 1
#ifndef NDEBUG
  if (LargestSinglePointeeTySizeInBitsAP == CastPointeeTySizeInBitsAP)
    assert(CountExprAsConstant.ugt(APInt(CountExprBitWidth, 1)));
#endif

  if (DCPT->isOrNull()) {
    // Traps if and only if the ptr is non-null
    return WillTrapKind::TrapIffPtrNotNull;
  }

  // Always traps
  return WillTrapKind::Trap;
}

void UnsafeOperationVisitor::
    checkSingleEntityFlowingToIndexableLocalVariableHandleCastToDynamicCount(
        const Stmt *UnsafeOp, const Expr *Operand, const VarDecl *IndexableVar,
        llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
            &SingleEntities,
        UnsafeOpKind Kind, ExtraDynCountLogicFn Predicate) {

  // Report casts that __bidi_indexable/__indexable pointers
  // to __counted_by/__sized_by
  auto *CE = cast<CastExpr>(UnsafeOp);

  if (CE->getCastKind() != clang::CK_BoundsSafetyPointerCast)
    return;

  if (FBPtrCastsToSkip.contains(CE))
    return;

  // Check for __bidi_indexable/__indexable being converted
  assert(Operand == CE->getSubExpr());
  if (!Operand->getType()->isPointerTypeWithBounds())
    return;

  if (!CE->getType()->isBoundsAttributedType())
    return;

  const auto *DCPT = CE->getType()->getAs<CountAttributedType>();
  if (!DCPT)
    return;
  const auto DCPTQualTy = CE->getType();

  // Use the largest pointee type out of all the potential __single to avoid
  // potential false positives. This choice can cause false negatives.
  const auto LargestSinglePointeeTy =
      GetLargestSinglePointee(SingleEntities, Ctx);
  const auto LargestSinglePointeeTySizeInBits =
      Ctx.getTypeSizeOrNull(LargestSinglePointeeTy);
  const auto CastPointeeTy = DCPT->desugar()->getPointeeType();
  const auto CastPointeeTySizeInBits = Ctx.getTypeSizeOrNull(CastPointeeTy);

  // Try to see if the count expr is a constant
  WillTrapKind WillTrap = WillTrapKind::Unknown;
  auto CountExprAsConstant =
      EvaluateAsInt(DCPT->getCountExpr()->IgnoreParens());
  if (CountExprAsConstant) {
    WillTrap = TryDetermineIfBidiIndexableToDynamicCountWithConstantCountTraps(
        DCPT, *CountExprAsConstant, LargestSinglePointeeTySizeInBits,
        CastPointeeTySizeInBits);
  }

  // Compute the maximum count/size expression that can be used without leading
  // to a trap. Any count/size > will trap.
  // E.g.
  // int foo(int* __counted_by(size) b, size_t size);
  //
  // void use_foo(int* b, size_t s) {
  //   // The maximum value `s` can be is 1.
  //   foo(b, s);
  // }
  size_t MaxSafeSizeOrCount = 0;
  if (DCPT->isCountInBytes()) {
    // __sized_by(or_null)
    MaxSafeSizeOrCount = LargestSinglePointeeTySizeInBits / 8;
  } else {
    // __counted_by(_or_null)
    //
    // If the pointee types are the same size this is 1.
    // If the __single pointee size is < DCPT pointee size then this is 0
    // If the __single pointee size is > DCPT pointee size then this >= 1.
    MaxSafeSizeOrCount =
        LargestSinglePointeeTySizeInBits / CastPointeeTySizeInBits;
  }

  // Run custom logic to suppress this warning or determine that the count
  // expression is a constant.
  if (Predicate) {
    WillTrap = Predicate(WillTrap, LargestSinglePointeeTySizeInBits,
                         CastPointeeTySizeInBits, CountExprAsConstant);
  }

  // Make sure we don't visit this cast again.
  FBPtrCastsToSkip.insert(CE);

  // Technically if `Kind == UnsafeOpKind::Return` then we will never trap due
  // a bug where the bounds checks are missing (rdar://83900556). We ignore that
  // problem here and let the handler decide what to do.

  switch (WillTrap) {
  case WillTrapKind::NoTrap:
    // Nothing to warn about.
    return;
  case WillTrapKind::TrapIffPtrNull:
    // TODO: We should probably warn about this but we probably want to do this
    // elsewhere (i.e. not in the analysis).
    // For now just drop this.
    return;
  case WillTrapKind::Unknown:           // Might trap
  case WillTrapKind::Trap:              // Definitely traps
  case WillTrapKind::TrapIffPtrNotNull: // Traps in almost all cases
    Handler.handleSingleEntitiesFlowingToIndexableDynamicCountConversion(
        SingleEntities, IndexableVar, UnsafeOp, Kind, Operand, DCPTQualTy,
        WillTrap, CountExprAsConstant, MaxSafeSizeOrCount);
    break;
  }
}

void UnsafeOperationVisitor::
    checkSingleEntityFlowingToIndexableLocalVariableHandleEscapingCastConvertedToDynamicCount(
        const Stmt *UnsafeOp, const Expr *Operand, const VarDecl *IndexableVar,
        llvm::SmallVectorImpl<BoundsSafetySuggestionHandler::SingleEntity>
            &SingleEntities,
        UnsafeOpKind Kind) {
  assert(Kind == UnsafeOpKind::CallArg || Kind == UnsafeOpKind::Return ||
         Kind == UnsafeOpKind::Assignment);

  const auto *DCPT = Operand->getType()->getAs<CountAttributedType>();
  if (!DCPT)
    return;

  // Set up code to lazily compute the count expression with call parameters
  // replaced with arguments used at call site.
  const Expr *CountExprUsingParams = nullptr;
  const Expr *CountExprUsingArguments = nullptr;
  const auto *Call = dyn_cast<CallExpr>(UnsafeOp);
  auto LazilyInitCountExprUsingArguments = [&]() {
    if (CountExprUsingArguments)
      return;
    ExprResult Replaced =
        ReplaceCountExprParamsWithArgsFromCall(CountExprUsingParams, Call, S);
    if (auto *NewCountExpr = Replaced.get()) {
      CountExprUsingArguments = NewCountExpr;
    }
  };

  auto CallExprPredicate =
      [&](WillTrapKind WillTrap, size_t LargestSinglePointeeTySizeInBits,
          size_t CastPointeeTySizeInBits,
          std::optional<APInt> &CountExprAsConstant) -> WillTrapKind {
    // This predicate tries to see if it can statically determine
    // if the cast will fail. It does this by trying to see if the
    // __counted_by/__sized_by size expression is a constant if the
    // call arguments are substituted into the expression.
    if (WillTrap != WillTrapKind::Unknown) {
      // Trust the existing analysis
      return WillTrap;
    }

    LazilyInitCountExprUsingArguments();
    const auto CountExprUsingArgumentsAsConstant =
        EvaluateAsInt(CountExprUsingArguments);

    if (CountExprUsingArgumentsAsConstant) {
      CountExprAsConstant =
          *CountExprUsingArgumentsAsConstant; // Pass back to caller
      WillTrap =
          TryDetermineIfBidiIndexableToDynamicCountWithConstantCountTraps(
              DCPT, *CountExprUsingArgumentsAsConstant,
              LargestSinglePointeeTySizeInBits, CastPointeeTySizeInBits);
    }

    return WillTrap;
  };

  ExtraDynCountLogicFn Predicate = nullptr;
  // Currently only `CallArg` needs a special predicate.
  if (Kind == UnsafeOpKind::CallArg) {
    assert(Call);
    CountExprUsingParams = DCPT->getCountExpr();
    Predicate = CallExprPredicate;
  }

  // FIXME: Ideally we'd support constant evaluation for returns and
  // variable assignment too. Unfortunately we don't anyway right now
  // of knowing what values the variables referred to in the count expression
  // could take. We'd need a reaching-definition analysis to do this.

  // Walk through the Operand to find potential CK_BoundsSafetyPointerCast's to
  // warn about. Although the visitor will normally find this cast later on its
  // done this way here so that when the cast is found the extra context (e.g.
  // the cast happens at a call site) needed to emit the diagnostic.
  const Expr *Current = Operand;
  const Expr *Previous = nullptr;
  while (Current != Previous) {
    Previous = Current;
    if (const auto *PE = dyn_cast<ParenExpr>(Current)) {
      Current = PE->getSubExpr();
      continue;
    }
    if (const auto *BCE = dyn_cast<BoundsCheckExpr>(Current)) {
      Current = BCE->getGuardedExpr();
      continue;
    }
    if (const auto *CE = dyn_cast<CastExpr>(Current)) {
      if (CE->getCastKind() == clang::CK_BoundsSafetyPointerCast) {
        // Found a cast we might want to warn about.
        checkSingleEntityFlowingToIndexableLocalVariableHandleCastToDynamicCount(
            CE, CE->getSubExpr(), IndexableVar, SingleEntities, Kind,
            Predicate);
      }
      Current = CE->getSubExpr();
    }
  }
}

void UnsafeOperationVisitor::VisitArraySubscriptExpr(
    const ArraySubscriptExpr *ASE) {
  handleUnsafeOperation(ASE, ASE->getBase(), UnsafeOpKind::Index);

  // Continue visitation normally.
  VisitChildren(ASE);
}

void UnsafeOperationVisitor::VisitUnaryOperator(const UnaryOperator *UO) {

  switch (UO->getOpcode()) {
  case UO_PostInc:
  case UO_PostDec:
  case UO_PreInc:
  case UO_PreDec:
    handleUnsafeOperation(UO, UO->getSubExpr(), UnsafeOpKind::Arithmetic);
    break;
  case UO_Deref:
    handleUnsafeOperation(UO, UO->getSubExpr(), UnsafeOpKind::Deref);
    break;
  default:
    break;
  }

  // Continue visitation normally.
  VisitChildren(UO);
}

void UnsafeOperationVisitor::VisitBinaryOperator(const BinaryOperator *BO) {
  const Expr *Lhs = BO->getLHS(), *Rhs = BO->getRHS();
  QualType LhsTy = Lhs->getType(), RhsTy = Rhs->getType();

  switch (BO->getOpcode()) {
  case BO_Add:
    // n + ptr, a special case that only works with BO_Add.
    if (RhsTy->isAnyPointerType() && LhsTy->isIntegralOrEnumerationType()) {
      handleUnsafeOperation(BO, Rhs, UnsafeOpKind::Arithmetic);
    }
    LLVM_FALLTHROUGH; // Fall through to handle ptr + n
  case BO_Sub:
  case BO_AddAssign:
  case BO_SubAssign:
    // ptr + n, ptr - n, ptr += n, ptr -= n, i.e. the typical case.
    if (LhsTy->isAnyPointerType() && RhsTy->isIntegralOrEnumerationType()) {
      handleUnsafeOperation(BO, Lhs, UnsafeOpKind::Arithmetic);
    }
    break;
  case BO_Assign: {
    if (RhsTy->isPointerType()) {
      handleUnsafeOperation(BO, Rhs, UnsafeOpKind::Assignment);
    }
    break;
  }
  default:
    break;
  }

  // Continue visitation normally.
  VisitChildren(BO);
}

void UnsafeOperationVisitor::VisitMemberExpr(const MemberExpr *ME) {
  if (ME->isArrow()) {
    assert(ME->getBase()->getType()->isPointerType());
    handleUnsafeOperation(ME, ME->getBase(), UnsafeOpKind::MemberAccess);
  }

  // Continue visitation normally.
  VisitChildren(ME);
}

void UnsafeOperationVisitor::VisitOpaqueValueExpr(const OpaqueValueExpr *OVE) {
  // OpaqueValueExpr doesn't have any children so Visit() won't traverse the
  // SourceExpr so we need to handle it manually here.
  const auto *SrcExpr = OVE->getSourceExpr();
  if (!SrcExpr)
    return;

  // OVEs wrap an expression that should only be evaluated once. Thus this
  // visitor needs to make sure it only visits the SourceExpr once, otherwise it
  // will warn about the same unsafe operation multiple times.
  if (VisitedOVESourceExprs.contains(SrcExpr))
    return; // Don't visit again

  VisitedOVESourceExprs.insert(SrcExpr);
  Visit(SrcExpr);
}

void UnsafeOperationVisitor::VisitDeclStmt(const DeclStmt *DS) {
  for (const Decl *D : DS->decls()) {
    if (const auto *VD = dyn_cast<VarDecl>(D)) {
      if (const Expr *E = VD->getInit()) {
        if (E->getType()->isPointerType()) {
          handleUnsafeOperation(DS, E, UnsafeOpKind::Assignment);
        } else if (const auto *ILE = dyn_cast<InitListExpr>(E)) {
          // Struct/Union initializer.
          // Note `InitListExpr` may be nested so iterative DFS
          // is used to walk through nested `InitListExpr` and visit
          // all the initializers.
          llvm::SmallVector<std::tuple<const InitListExpr *,const Expr *>> WorkList;

          // Collect initial set of initializers
          WorkList.reserve(ILE->getNumInits());
          auto CollectInitializers = [&](const InitListExpr *ILE) -> void {
            if (ILE->getNumInits() == 0)
              return;
            // Add in reverse order so that when the worklist is processed
            // the initializers are visited in order
            for (size_t Index = ILE->getNumInits(); Index > 0; --Index) {
              WorkList.push_back(std::make_tuple(ILE, ILE->getInit(Index - 1)));
            }
          };
          CollectInitializers(ILE);

          while (!WorkList.empty()) {
            const Expr* Initializer;
            const InitListExpr* ParentILE;

            std::tie(ParentILE, Initializer) = WorkList.back();
            WorkList.pop_back();
            if (const auto *SubILE = dyn_cast<InitListExpr>(Initializer)) {
              CollectInitializers(SubILE);
              continue;
            }
            // Field initializer of something that's not a union or struct
            // (i.e. a leaf of "InitListExpr" graph).
            if (Initializer->getType()->isPointerType()) {
              handleUnsafeOperation(ParentILE, Initializer,
                                    UnsafeOpKind::Assignment);
            }
          }
        }
      }
    }
  }

  // This handles visiting the children. We don't do that above because the
  // visited Decls aren't necessarily `VarDecl`s (e.g. `TypedefDecl`)
  VisitChildren(DS);
}

void UnsafeOperationVisitor::VisitReturnStmt(const ReturnStmt *RS) {
  const auto *const ReturnExpr = RS->getRetValue();
  if (ReturnExpr && ReturnExpr->getType()->isPointerType()) {
    handleUnsafeOperation(RS, ReturnExpr, UnsafeOpKind::Return);
  }

  // Visit children to look for other unsafe operations
  VisitChildren(RS);
}

void UnsafeOperationVisitor::VisitCallExpr(const CallExpr *CE) {
  for (const auto &Arg : CE->arguments()) {
    if (Arg->getType()->isPointerType()) {
      handleUnsafeOperation(CE, Arg, UnsafeOpKind::CallArg);
    }
  }

  // Visit children to look for other unsafe operations
  VisitChildren(CE);
}

void UnsafeOperationVisitor::VisitCastExpr(const CastExpr *CE) {
  if (CE->getSubExpr()->getType()->isPointerType()) {
    handleUnsafeOperation(CE, CE->getSubExpr(), UnsafeOpKind::Cast);
  }

  // Visit children to look for other unsafe operations
  VisitChildren(CE);
}

void clang::checkBoundsSafetySuggestions(const Decl *D,
                                         BoundsSafetySuggestionHandler &Handler,
                                         Sema &S) {
  const Stmt *Body = D->getBody();
  assert(Body);

  // First pass: Populate DefMap by connecting variables
  // to their potential values.
  DefinitionVisitor DefV;
  DefV.Visit(Body);

  // Second pass: Scan the function for unsafe operations.
  // Use DefMap to assess where the operands are coming from.
  //
  // Fundamentally all of this could have been done in one pass,
  // with some postprocessing, but it's nice to have some
  // separation of concerns.
  UnsafeOperationVisitor OpV(Handler, DefV.DefMap, S);
  OpV.Visit(Body);
  OpV.reset();
}