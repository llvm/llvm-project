//===--- SemaExpr.cpp - Semantic Analysis for Typed Memory Operations -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for typed memory operations.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTStructuralEquivalence.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/SemaHLSL.h"
#include "clang/Sema/SemaObjC.h"
#include "clang/Sema/SemaRISCV.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;
using namespace sema;

static void emitTMODescriptorRemarks(
    Sema &S, const Expr *CallExpr, const FunctionDecl *Callee,
    const FunctionDecl *TargetFunction, SourceRange TypeRange,
    ASTContext::TypedMemoryDescriptor TMD,
    const InferredAllocationType &InferredType) {
  if (S.Diags.isIgnored(diag::remark_tmo_passed_type, CallExpr->getBeginLoc()))
    return;
  SourceRange CallRange = CallExpr->getSourceRange();

  TypedMemoryDescriptorBits TypeDescriptor = TMD.asBits();

  bool IsArrayAlloc =
      (TMD.Summary.CallsiteFlags & TypedMemoryCallsiteFlags::Array) ==
      TypedMemoryCallsiteFlags::Array;
  bool IsConstantArray =
      IsArrayAlloc &&
      !!(TMD.Summary.CallsiteFlags & TypedMemoryCallsiteFlags::FixedSize);

  StringRef Note = "";
  if (IsConstantArray)
    Note = "constant sized array of ";
  else if (IsArrayAlloc)
    Note = "array of ";

  std::string InferredTypeName = InferredType.describe(S.getASTContext());
  S.Diag(CallRange.getBegin(), diag::remark_tmo_passed_type)
      << Twine(Note, "type ") << *InferredType.primaryType() << TargetFunction
      << (Callee == TargetFunction) << Callee << TypeRange;
  // Note: we still only encode the primary type, not the full inferred
  // structure
  S.Diag(CallRange.getBegin(), diag::note_tmo_type_encoding)
      << Note << *InferredType.primaryType() << TypeDescriptor.value()
      << TMD.TypeDescription << TypeRange;
}

static void emitTMOInferenceDiagnostics(Sema &S, const Expr *CallExpr,
                                        const FunctionDecl *Callee,
                                        std::optional<QualType> Type,
                                        const InferredTypeInfo &TypeInfo,
                                        const FunctionDecl *RewriteTarget) {
  assert(S.getLangOpts().TypedMemoryOperations);
  assert(RewriteTarget);

  // Don't do any work if logging is not enabled
  bool WarnOnInferenceFailure = !S.Diags.isIgnored(
      diag::warn_tmo_inference_failed, CallExpr->getBeginLoc());
  bool EmitTMORemarks =
      !S.Diags.isIgnored(diag::remark_tmo_passed_type, CallExpr->getBeginLoc());
  if (!WarnOnInferenceFailure && !EmitTMORemarks)
    return;

  const InferredAllocationType *InferredType =
      TypeInfo.Type ? &*TypeInfo.Type : nullptr;

  const Expr *InferenceSourceExpression = TypeInfo.InferenceSourceExpression;
  SourceRange TypeSourceRange;
  if (const ExplicitCastExpr *CastExpr =
          dyn_cast<ExplicitCastExpr>(InferenceSourceExpression))
    TypeSourceRange =
        CastExpr->getTypeInfoAsWritten()->getTypeLoc().getSourceRange();
  else
    TypeSourceRange = InferenceSourceExpression->getSourceRange();

  llvm::scope_exit LogRewriteIfNecessary([&]() {
    if (!EmitTMORemarks)
      return;
    if (!Type) {
      S.Diag(TypeSourceRange.getBegin(), diag::note_tmo_failed_inference_source)
          << InferenceSourceExpression << TypeSourceRange;
      return;
    }
    bool WasInferredFromCast = isa<const CastExpr>(InferenceSourceExpression);
    const Expr *EffectiveExpr = InferenceSourceExpression;
    SourceRange EffectiveRange = TypeSourceRange;
    if (EffectiveRange.isInvalid()) {
      // A number of parts of sema introduce explicit CStyleCasts and similar
      // instead of ImplicitCastExprs, but also don't include the source range
      // of the cast sub expression either so we just substitute in the call
      // expression itself here.
      EffectiveExpr = CallExpr;
      EffectiveRange = CallExpr->getSourceRange();
    }
    assert(InferredType);
    const char *NotePrefix = InferredType->isArray() ? "array of " : "";
    std::string NoteDisplay = InferredType->describe(S.getASTContext());
    S.Diag(EffectiveRange.getBegin(), diag::note_tmo_inference_result)
        << Twine(NotePrefix, NoteDisplay) << WasInferredFromCast
        << EffectiveExpr->IgnoreUnlessSpelledInSource();
  });

  if (!Type) {
    S.Diag(CallExpr->getBeginLoc(), diag::warn_tmo_inference_failed) << Callee;
    return;
  }

  if (!EmitTMORemarks)
    return;

  assert(!(*Type)->isDependentType());
  TypedMemoryCallsiteFlags Flags = TypeInfo.InferredCallsiteFlags;
  ASTContext::TypedMemoryDescriptor TypeDescriptor =
      S.getASTContext().getTypedMemoryDescriptor(
          *Type, Callee->getDeclName().getCXXOverloadedOperator(), Flags);
  assert(InferredType);
  emitTMODescriptorRemarks(S, CallExpr, Callee, RewriteTarget, TypeSourceRange,
                           TypeDescriptor, *InferredType);
}

void TypedMemoryCallsiteContext::recordInfoForInferredCall(
    Sema &S, const CallExpr *Call) {
  if (S.CurContext->isDependentContext())
    return;

  if (!S.getLangOpts().TypedMemoryOperations)
    return;

  const FunctionDecl *CalleeDecl = Call->getDirectCallee();
  if (!CalleeDecl)
    return;
  const TypedMemoryAttr *TMA = CalleeDecl->getAttr<TypedMemoryAttr>();
  if (!TMA)
    return;
  FunctionDecl *Target = TMA->getRewriteTarget();
  unsigned InferredParamIndex = TMA->getInferredParameterIdx().getLLVMIndex();
  const Expr *InferredParameter = Call->getArg(InferredParamIndex);
  const CastExpr *CastExpr = nullptr;
  if (auto FoundCast = Casts.find(Call); FoundCast != Casts.end())
    CastExpr = FoundCast->second;
  InferredTypeInfo InferredInfo = S.getASTContext().inferTypedMemoryType(
      Call, *InferredParameter, CastExpr);

  std::optional<QualType> PrimaryType;
  if (InferredInfo.Type)
    PrimaryType = InferredInfo.Type->primaryType();
  emitTMOInferenceDiagnostics(S, Call, CalleeDecl, PrimaryType, InferredInfo,
                              Target);
}

void TypedMemoryCallsiteContext::finalizeOutstandingTMOCandidates(Sema &S) {
  if (!S.getLangOpts().TypedMemoryOperations)
    return;
  if (auto *EnclosingFunctionScope = S.getEnclosingFunction()) {
    // Don't clear any TMO tracking information when finishing a statement
    // expression as we may be casting the result. This is suboptimal as it
    // means we'll maintain this across multiple substatements, but for now
    // this is fine.
    if (EnclosingFunctionScope->CompoundScopes.size() &&
        EnclosingFunctionScope->CompoundScopes.back().IsStmtExpr)
      return;
  }

  ShouldSearchCasts = false;

  for (const CallExpr *Call : Calls)
    recordInfoForInferredCall(S, Call);

  Calls.clear();
  Casts.clear();
}

void TypedMemoryCallsiteContext::recordCastForTMOInference(
    Sema &S, const CastExpr *Cast) {
  if (!S.getLangOpts().TypedMemoryOperations)
    return;

  if (S.CurContext->isDependentContext())
    return;

  if (Cast->getType()->isVoidPointerType())
    return;

  if (isa<ImplicitCastExpr>(Cast))
    return;

  const Expr *PotentialCall = Cast->getSubExpr();
  const Expr *LastPotentialCall = nullptr;
  // We need to walk through all casts and implicit nodes between the cast
  // node and the actual underlying expression.
  do {
    LastPotentialCall = PotentialCall;
    PotentialCall = PotentialCall->IgnoreParens();
    PotentialCall = PotentialCall->IgnoreImplicit();
    PotentialCall = PotentialCall->IgnoreCasts();
    if (auto *OpaqueValue = dyn_cast<OpaqueValueExpr>(PotentialCall))
      PotentialCall = OpaqueValue->getSourceExpr();
    if (auto *SE = dyn_cast<StmtExpr>(PotentialCall)) {
      const CompoundStmt *SubStmt = SE->getSubStmt();
      if (auto *LastExpr = dyn_cast_or_null<Expr>(SubStmt->body_back()))
        PotentialCall = LastExpr;
    }
  } while (LastPotentialCall != PotentialCall);

  const CallExpr *Call = dyn_cast_or_null<CallExpr>(PotentialCall);
  if (!Call)
    return;

  const FunctionDecl *Callee = Call->getDirectCallee();
  if (!Callee || !Callee->getAttr<TypedMemoryAttr>())
    return;
  assert(ShouldSearchCasts);
  // We prioritize the deepest non-implicit, non-void* cast
  auto [It, Inserted] = Casts.try_emplace(Call, Cast);
  if (!Inserted && It->second->getType()->isVoidPointerType())
    It->second = Cast;
}

void Sema::emitTMODiagnosticsForTypeQuery(SourceLocation QueryLocation,
                                          SourceRange ExpressionRange,
                                          QualType QueriedType) {
  // Don't do any work if logging is not enabled
  if (Diags.isIgnored(diag::remark_tmo_passed_type, QueryLocation))
    return;

  if (QueriedType->isDependentType() || QueriedType->isIncompleteType())
    return;

  ASTContext::TypedMemoryDescriptor Descriptor =
      Context.getTypedMemoryDescriptor(QueriedType, OO_None,
                                       TypedMemoryCallsiteFlags::None);
  TypedMemoryDescriptorBits TMDB;
  TMDB.Summary = Descriptor.Summary;
  TMDB.Hash = Descriptor.IdentityHash;
  Diag(QueryLocation, diag::remark_tmo_get_descriptor_info)
      << QueriedType << TMDB.value() << Descriptor.TypeDescription
      << ExpressionRange;
}

bool Sema::checkTMOGetTypeDescriptor(QualType T, SourceLocation Loc,
                                     SourceRange ArgRange) {
  if (RequireCompleteSizedType(
          Loc, Context.getBaseElementType(T),
          diag::err_sizeof_alignof_incomplete_or_sizeless_type,
          getTraitSpelling(UETT_TMOGetTypeDescriptor), ArgRange))
    return true;
  assert(!T->isVoidType());
  emitTMODiagnosticsForTypeQuery(Loc, ArgRange, T);
  return false;
}

void TypedMemoryCallsiteContext::recordTMOInferenceCandidate(Sema &S,
                                                             const Expr *Call) {
  if (!S.getLangOpts().TypedMemoryOperations)
    return;
  if (S.CurContext->isDependentContext())
    return;
  const auto *CE = dyn_cast_or_null<CallExpr>(Call);
  if (!CE)
    return;
  auto *CalleeDecl = CE->getDirectCallee();
  if (!CalleeDecl)
    return;
  if (!CalleeDecl->hasAttr<TypedMemoryAttr>())
    return;
  Calls.push_back(CE);
  ShouldSearchCasts = true;
}
