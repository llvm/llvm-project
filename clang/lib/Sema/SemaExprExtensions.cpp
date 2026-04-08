//===--- SemaExprExtensions.cpp - Semantic Analysis for Clang Extensions --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for Clang language extensions
//  (blocks, __builtin_choose_expr, __unknown_anytype resolution, etc.).
//  Split from SemaExpr.cpp for parallel compilation.
//
//===----------------------------------------------------------------------===//

#include "CheckExprLifetime.h"
#include "TreeTransform.h"
#include "UsedDeclVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/MangleNumberingContext.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TypeTraits.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/AnalysisBasedWarnings.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "clang/Sema/Designator.h"
#include "clang/Sema/EnterExpressionEvaluationContext.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaAMDGPU.h"
#include "clang/Sema/SemaARM.h"
#include "clang/Sema/SemaCUDA.h"
#include "clang/Sema/SemaFixItUtils.h"
#include "clang/Sema/SemaHLSL.h"
#include "clang/Sema/SemaObjC.h"
#include "clang/Sema/SemaOpenMP.h"
#include "clang/Sema/SemaPseudoObject.h"
#include "clang/Sema/Template.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/TypeSize.h"
#include <limits>
#include <optional>

using namespace clang;
using namespace sema;

// Forward declarations for helpers defined in SemaExpr.cpp that are shared
// across both translation units after the file was split.
// These are file-scope (non-static) helpers; the 'using namespace clang'
// above brings clang types into scope so unqualified names work here.
bool CheckForModifiableLvalue(Expr *E, SourceLocation Loc, Sema &S);
void captureVariablyModifiedType(ASTContext &Context, QualType T,
                                 sema::CapturingScopeInfo *CSI);

//===----------------------------------------------------------------------===//
// Clang Extensions.
//===----------------------------------------------------------------------===//

void Sema::ActOnBlockStart(SourceLocation CaretLoc, Scope *CurScope) {
  BlockDecl *Block = BlockDecl::Create(Context, CurContext, CaretLoc);

  if (LangOpts.CPlusPlus) {
    MangleNumberingContext *MCtx;
    Decl *ManglingContextDecl;
    std::tie(MCtx, ManglingContextDecl) =
        getCurrentMangleNumberContext(Block->getDeclContext());
    if (MCtx) {
      unsigned ManglingNumber = MCtx->getManglingNumber(Block);
      Block->setBlockMangling(ManglingNumber, ManglingContextDecl);
    }
  }

  PushBlockScope(CurScope, Block);
  CurContext->addDecl(Block);
  if (CurScope)
    PushDeclContext(CurScope, Block);
  else
    CurContext = Block;

  getCurBlock()->HasImplicitReturnType = true;

  // Enter a new evaluation context to insulate the block from any
  // cleanups from the enclosing full-expression.
  PushExpressionEvaluationContext(
      ExpressionEvaluationContext::PotentiallyEvaluated);
}

void Sema::ActOnBlockArguments(SourceLocation CaretLoc, Declarator &ParamInfo,
                               Scope *CurScope) {
  assert(ParamInfo.getIdentifier() == nullptr &&
         "block-id should have no identifier!");
  assert(ParamInfo.getContext() == DeclaratorContext::BlockLiteral);
  BlockScopeInfo *CurBlock = getCurBlock();

  TypeSourceInfo *Sig = GetTypeForDeclarator(ParamInfo);
  QualType T = Sig->getType();
  DiagnoseUnexpandedParameterPack(CaretLoc, Sig, UPPC_Block);

  // GetTypeForDeclarator always produces a function type for a block
  // literal signature.  Furthermore, it is always a FunctionProtoType
  // unless the function was written with a typedef.
  assert(T->isFunctionType() &&
         "GetTypeForDeclarator made a non-function block signature");

  // Look for an explicit signature in that function type.
  FunctionProtoTypeLoc ExplicitSignature;

  if ((ExplicitSignature = Sig->getTypeLoc()
                               .getAsAdjusted<FunctionProtoTypeLoc>())) {

    // Check whether that explicit signature was synthesized by
    // GetTypeForDeclarator.  If so, don't save that as part of the
    // written signature.
    if (ExplicitSignature.getLocalRangeBegin() ==
        ExplicitSignature.getLocalRangeEnd()) {
      // This would be much cheaper if we stored TypeLocs instead of
      // TypeSourceInfos.
      TypeLoc Result = ExplicitSignature.getReturnLoc();
      unsigned Size = Result.getFullDataSize();
      Sig = Context.CreateTypeSourceInfo(Result.getType(), Size);
      Sig->getTypeLoc().initializeFullCopy(Result, Size);

      ExplicitSignature = FunctionProtoTypeLoc();
    }
  }

  CurBlock->TheDecl->setSignatureAsWritten(Sig);
  CurBlock->FunctionType = T;

  const auto *Fn = T->castAs<FunctionType>();
  QualType RetTy = Fn->getReturnType();
  bool isVariadic =
      (isa<FunctionProtoType>(Fn) && cast<FunctionProtoType>(Fn)->isVariadic());

  CurBlock->TheDecl->setIsVariadic(isVariadic);

  // Context.DependentTy is used as a placeholder for a missing block
  // return type.  TODO:  what should we do with declarators like:
  //   ^ * { ... }
  // If the answer is "apply template argument deduction"....
  if (RetTy != Context.DependentTy) {
    CurBlock->ReturnType = RetTy;
    CurBlock->TheDecl->setBlockMissingReturnType(false);
    CurBlock->HasImplicitReturnType = false;
  }

  // Push block parameters from the declarator if we had them.
  SmallVector<ParmVarDecl*, 8> Params;
  if (ExplicitSignature) {
    for (unsigned I = 0, E = ExplicitSignature.getNumParams(); I != E; ++I) {
      ParmVarDecl *Param = ExplicitSignature.getParam(I);
      if (Param->getIdentifier() == nullptr && !Param->isImplicit() &&
          !Param->isInvalidDecl() && !getLangOpts().CPlusPlus) {
        // Diagnose this as an extension in C17 and earlier.
        if (!getLangOpts().C23)
          Diag(Param->getLocation(), diag::ext_parameter_name_omitted_c23);
      }
      Params.push_back(Param);
    }

  // Fake up parameter variables if we have a typedef, like
  //   ^ fntype { ... }
  } else if (const FunctionProtoType *Fn = T->getAs<FunctionProtoType>()) {
    for (const auto &I : Fn->param_types()) {
      ParmVarDecl *Param = BuildParmVarDeclForTypedef(
          CurBlock->TheDecl, ParamInfo.getBeginLoc(), I);
      Params.push_back(Param);
    }
  }

  // Set the parameters on the block decl.
  if (!Params.empty()) {
    CurBlock->TheDecl->setParams(Params);
    CheckParmsForFunctionDef(CurBlock->TheDecl->parameters(),
                             /*CheckParameterNames=*/false);
  }

  // Finally we can process decl attributes.
  ProcessDeclAttributes(CurScope, CurBlock->TheDecl, ParamInfo);

  // Put the parameter variables in scope.
  for (auto *AI : CurBlock->TheDecl->parameters()) {
    AI->setOwningFunction(CurBlock->TheDecl);

    // If this has an identifier, add it to the scope stack.
    if (AI->getIdentifier()) {
      CheckShadow(CurBlock->TheScope, AI);

      PushOnScopeChains(AI, CurBlock->TheScope);
    }

    if (AI->isInvalidDecl())
      CurBlock->TheDecl->setInvalidDecl();
  }
}

void Sema::ActOnBlockError(SourceLocation CaretLoc, Scope *CurScope) {
  // Leave the expression-evaluation context.
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();

  // Pop off CurBlock, handle nested blocks.
  PopDeclContext();
  PopFunctionScopeInfo();
}

ExprResult Sema::ActOnBlockStmtExpr(SourceLocation CaretLoc,
                                    Stmt *Body, Scope *CurScope) {
  // If blocks are disabled, emit an error.
  if (!LangOpts.Blocks)
    Diag(CaretLoc, diag::err_blocks_disable) << LangOpts.OpenCL;

  // Leave the expression-evaluation context.
  if (hasAnyUnrecoverableErrorsInThisFunction())
    DiscardCleanupsInEvaluationContext();
  assert(!Cleanup.exprNeedsCleanups() &&
         "cleanups within block not correctly bound!");
  PopExpressionEvaluationContext();

  BlockScopeInfo *BSI = cast<BlockScopeInfo>(FunctionScopes.back());
  BlockDecl *BD = BSI->TheDecl;

  maybeAddDeclWithEffects(BD);

  if (BSI->HasImplicitReturnType)
    deduceClosureReturnType(*BSI);

  QualType RetTy = Context.VoidTy;
  if (!BSI->ReturnType.isNull())
    RetTy = BSI->ReturnType;

  bool NoReturn = BD->hasAttr<NoReturnAttr>();
  QualType BlockTy;

  // If the user wrote a function type in some form, try to use that.
  if (!BSI->FunctionType.isNull()) {
    const FunctionType *FTy = BSI->FunctionType->castAs<FunctionType>();

    FunctionType::ExtInfo Ext = FTy->getExtInfo();
    if (NoReturn && !Ext.getNoReturn()) Ext = Ext.withNoReturn(true);

    // Turn protoless block types into nullary block types.
    if (isa<FunctionNoProtoType>(FTy)) {
      FunctionProtoType::ExtProtoInfo EPI;
      EPI.ExtInfo = Ext;
      BlockTy = Context.getFunctionType(RetTy, {}, EPI);

      // Otherwise, if we don't need to change anything about the function type,
      // preserve its sugar structure.
    } else if (FTy->getReturnType() == RetTy &&
               (!NoReturn || FTy->getNoReturnAttr())) {
      BlockTy = BSI->FunctionType;

    // Otherwise, make the minimal modifications to the function type.
    } else {
      const FunctionProtoType *FPT = cast<FunctionProtoType>(FTy);
      FunctionProtoType::ExtProtoInfo EPI = FPT->getExtProtoInfo();
      EPI.TypeQuals = Qualifiers();
      EPI.ExtInfo = Ext;
      BlockTy = Context.getFunctionType(RetTy, FPT->getParamTypes(), EPI);
    }

  // If we don't have a function type, just build one from nothing.
  } else {
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.ExtInfo = FunctionType::ExtInfo().withNoReturn(NoReturn);
    BlockTy = Context.getFunctionType(RetTy, {}, EPI);
  }

  DiagnoseUnusedParameters(BD->parameters());
  BlockTy = Context.getBlockPointerType(BlockTy);

  // If needed, diagnose invalid gotos and switches in the block.
  if (getCurFunction()->NeedsScopeChecking() &&
      !PP.isCodeCompletionEnabled())
    DiagnoseInvalidJumps(cast<CompoundStmt>(Body));

  BD->setBody(cast<CompoundStmt>(Body));

  if (Body && getCurFunction()->HasPotentialAvailabilityViolations)
    DiagnoseUnguardedAvailabilityViolations(BD);

  // Try to apply the named return value optimization. We have to check again
  // if we can do this, though, because blocks keep return statements around
  // to deduce an implicit return type.
  if (getLangOpts().CPlusPlus && RetTy->isRecordType() &&
      !BD->isDependentContext())
    computeNRVO(Body, BSI);

  if (RetTy.hasNonTrivialToPrimitiveDestructCUnion() ||
      RetTy.hasNonTrivialToPrimitiveCopyCUnion())
    checkNonTrivialCUnion(RetTy, BD->getCaretLocation(),
                          NonTrivialCUnionContext::FunctionReturn,
                          NTCUK_Destruct | NTCUK_Copy);

  PopDeclContext();

  // Set the captured variables on the block.
  SmallVector<BlockDecl::Capture, 4> Captures;
  for (Capture &Cap : BSI->Captures) {
    if (Cap.isInvalid() || Cap.isThisCapture())
      continue;
    // Cap.getVariable() is always a VarDecl because
    // blocks cannot capture structured bindings or other ValueDecl kinds.
    auto *Var = cast<VarDecl>(Cap.getVariable());
    Expr *CopyExpr = nullptr;
    if (getLangOpts().CPlusPlus && Cap.isCopyCapture()) {
      if (auto *Record = Cap.getCaptureType()->getAsCXXRecordDecl()) {
        // The capture logic needs the destructor, so make sure we mark it.
        // Usually this is unnecessary because most local variables have
        // their destructors marked at declaration time, but parameters are
        // an exception because it's technically only the call site that
        // actually requires the destructor.
        if (isa<ParmVarDecl>(Var))
          FinalizeVarWithDestructor(Var, Record);

        // Enter a separate potentially-evaluated context while building block
        // initializers to isolate their cleanups from those of the block
        // itself.
        // FIXME: Is this appropriate even when the block itself occurs in an
        // unevaluated operand?
        EnterExpressionEvaluationContext EvalContext(
            *this, ExpressionEvaluationContext::PotentiallyEvaluated);

        SourceLocation Loc = Cap.getLocation();

        ExprResult Result = BuildDeclarationNameExpr(
            CXXScopeSpec(), DeclarationNameInfo(Var->getDeclName(), Loc), Var);

        // According to the blocks spec, the capture of a variable from
        // the stack requires a const copy constructor.  This is not true
        // of the copy/move done to move a __block variable to the heap.
        if (!Result.isInvalid() &&
            !Result.get()->getType().isConstQualified()) {
          Result = ImpCastExprToType(Result.get(),
                                     Result.get()->getType().withConst(),
                                     CK_NoOp, VK_LValue);
        }

        if (!Result.isInvalid()) {
          Result = PerformCopyInitialization(
              InitializedEntity::InitializeBlock(Var->getLocation(),
                                                 Cap.getCaptureType()),
              Loc, Result.get());
        }

        // Build a full-expression copy expression if initialization
        // succeeded and used a non-trivial constructor.  Recover from
        // errors by pretending that the copy isn't necessary.
        if (!Result.isInvalid() &&
            !cast<CXXConstructExpr>(Result.get())->getConstructor()
                ->isTrivial()) {
          Result = MaybeCreateExprWithCleanups(Result);
          CopyExpr = Result.get();
        }
      }
    }

    BlockDecl::Capture NewCap(Var, Cap.isBlockCapture(), Cap.isNested(),
                              CopyExpr);
    Captures.push_back(NewCap);
  }
  BD->setCaptures(Context, Captures, BSI->CXXThisCaptureIndex != 0);

  // Pop the block scope now but keep it alive to the end of this function.
  AnalysisBasedWarnings::Policy WP =
      AnalysisWarnings.getPolicyInEffectAt(Body->getEndLoc());
  PoppedFunctionScopePtr ScopeRAII = PopFunctionScopeInfo(&WP, BD, BlockTy);

  BlockExpr *Result = new (Context)
      BlockExpr(BD, BlockTy, BSI->ContainsUnexpandedParameterPack);

  // If the block isn't obviously global, i.e. it captures anything at
  // all, then we need to do a few things in the surrounding context:
  if (Result->getBlockDecl()->hasCaptures()) {
    // First, this expression has a new cleanup object.
    ExprCleanupObjects.push_back(Result->getBlockDecl());
    Cleanup.setExprNeedsCleanups(true);

    // It also gets a branch-protected scope if any of the captured
    // variables needs destruction.
    for (const auto &CI : Result->getBlockDecl()->captures()) {
      const VarDecl *var = CI.getVariable();
      if (var->getType().isDestructedType() != QualType::DK_none) {
        setFunctionHasBranchProtectedScope();
        break;
      }
    }
  }

  if (getCurFunction())
    getCurFunction()->addBlock(BD);

  // This can happen if the block's return type is deduced, but
  // the return expression is invalid.
  if (BD->isInvalidDecl())
    return CreateRecoveryExpr(Result->getBeginLoc(), Result->getEndLoc(),
                              {Result}, Result->getType());
  return Result;
}

ExprResult Sema::ActOnVAArg(SourceLocation BuiltinLoc, Expr *E, ParsedType Ty,
                            SourceLocation RPLoc) {
  TypeSourceInfo *TInfo;
  GetTypeFromParser(Ty, &TInfo);
  return BuildVAArgExpr(BuiltinLoc, E, TInfo, RPLoc);
}

ExprResult Sema::BuildVAArgExpr(SourceLocation BuiltinLoc,
                                Expr *E, TypeSourceInfo *TInfo,
                                SourceLocation RPLoc) {
  Expr *OrigExpr = E;
  bool IsMS = false;

  // CUDA device global function does not support varargs.
  if (getLangOpts().CUDA && getLangOpts().CUDAIsDevice) {
    if (const FunctionDecl *F = dyn_cast<FunctionDecl>(CurContext)) {
      CUDAFunctionTarget T = CUDA().IdentifyTarget(F);
      if (T == CUDAFunctionTarget::Global)
        return ExprError(Diag(E->getBeginLoc(), diag::err_va_arg_in_device));
    }
  }

  // NVPTX does not support va_arg expression.
  if (getLangOpts().OpenMP && getLangOpts().OpenMPIsTargetDevice &&
      Context.getTargetInfo().getTriple().isNVPTX())
    targetDiag(E->getBeginLoc(), diag::err_va_arg_in_device);

  // It might be a __builtin_ms_va_list. (But don't ever mark a va_arg()
  // as Microsoft ABI on an actual Microsoft platform, where
  // __builtin_ms_va_list and __builtin_va_list are the same.)
  if (!E->isTypeDependent() && Context.getTargetInfo().hasBuiltinMSVaList() &&
      Context.getTargetInfo().getBuiltinVaListKind() != TargetInfo::CharPtrBuiltinVaList) {
    QualType MSVaListType = Context.getBuiltinMSVaListType();
    if (Context.hasSameType(MSVaListType, E->getType())) {
      if (CheckForModifiableLvalue(E, BuiltinLoc, *this))
        return ExprError();
      IsMS = true;
    }
  }

  // Get the va_list type
  QualType VaListType = Context.getBuiltinVaListType();
  if (!IsMS) {
    if (VaListType->isArrayType()) {
      // Deal with implicit array decay; for example, on x86-64,
      // va_list is an array, but it's supposed to decay to
      // a pointer for va_arg.
      VaListType = Context.getArrayDecayedType(VaListType);
      // Make sure the input expression also decays appropriately.
      ExprResult Result = UsualUnaryConversions(E);
      if (Result.isInvalid())
        return ExprError();
      E = Result.get();
    } else if (VaListType->isRecordType() && getLangOpts().CPlusPlus) {
      // If va_list is a record type and we are compiling in C++ mode,
      // check the argument using reference binding.
      InitializedEntity Entity = InitializedEntity::InitializeParameter(
          Context, Context.getLValueReferenceType(VaListType), false);
      ExprResult Init = PerformCopyInitialization(Entity, SourceLocation(), E);
      if (Init.isInvalid())
        return ExprError();
      E = Init.getAs<Expr>();
    } else {
      // Otherwise, the va_list argument must be an l-value because
      // it is modified by va_arg.
      if (!E->isTypeDependent() &&
          CheckForModifiableLvalue(E, BuiltinLoc, *this))
        return ExprError();
    }
  }

  if (!IsMS && !E->isTypeDependent() &&
      !Context.hasSameType(VaListType, E->getType()))
    return ExprError(
        Diag(E->getBeginLoc(),
             diag::err_first_argument_to_va_arg_not_of_type_va_list)
        << OrigExpr->getType() << E->getSourceRange());

  if (!TInfo->getType()->isDependentType()) {
    if (RequireCompleteType(TInfo->getTypeLoc().getBeginLoc(), TInfo->getType(),
                            diag::err_second_parameter_to_va_arg_incomplete,
                            TInfo->getTypeLoc()))
      return ExprError();

    if (RequireNonAbstractType(TInfo->getTypeLoc().getBeginLoc(),
                               TInfo->getType(),
                               diag::err_second_parameter_to_va_arg_abstract,
                               TInfo->getTypeLoc()))
      return ExprError();

    if (!TInfo->getType().isPODType(Context)) {
      Diag(TInfo->getTypeLoc().getBeginLoc(),
           TInfo->getType()->isObjCLifetimeType()
             ? diag::warn_second_parameter_to_va_arg_ownership_qualified
             : diag::warn_second_parameter_to_va_arg_not_pod)
        << TInfo->getType()
        << TInfo->getTypeLoc().getSourceRange();
    }

    if (TInfo->getType()->isArrayType()) {
      DiagRuntimeBehavior(TInfo->getTypeLoc().getBeginLoc(), E,
                          PDiag(diag::warn_second_parameter_to_va_arg_array)
                              << TInfo->getType()
                              << TInfo->getTypeLoc().getSourceRange());
    }

    // Check for va_arg where arguments of the given type will be promoted
    // (i.e. this va_arg is guaranteed to have undefined behavior).
    QualType PromoteType;
    if (Context.isPromotableIntegerType(TInfo->getType())) {
      PromoteType = Context.getPromotedIntegerType(TInfo->getType());
      // [cstdarg.syn]p1 defers the C++ behavior to what the C standard says,
      // and C23 7.16.1.1p2 says, in part:
      //   If type is not compatible with the type of the actual next argument
      //   (as promoted according to the default argument promotions), the
      //   behavior is undefined, except for the following cases:
      //     - both types are pointers to qualified or unqualified versions of
      //       compatible types;
      //     - one type is compatible with a signed integer type, the other
      //       type is compatible with the corresponding unsigned integer type,
      //       and the value is representable in both types;
      //     - one type is pointer to qualified or unqualified void and the
      //       other is a pointer to a qualified or unqualified character type;
      //     - or, the type of the next argument is nullptr_t and type is a
      //       pointer type that has the same representation and alignment
      //       requirements as a pointer to a character type.
      // Given that type compatibility is the primary requirement (ignoring
      // qualifications), you would think we could call typesAreCompatible()
      // directly to test this. However, in C++, that checks for *same type*,
      // which causes false positives when passing an enumeration type to
      // va_arg. Instead, get the underlying type of the enumeration and pass
      // that.
      QualType UnderlyingType = TInfo->getType();
      if (const auto *ED = UnderlyingType->getAsEnumDecl())
        UnderlyingType = ED->getIntegerType();
      if (Context.typesAreCompatible(PromoteType, UnderlyingType,
                                     /*CompareUnqualified*/ true))
        PromoteType = QualType();

      // If the types are still not compatible, we need to test whether the
      // promoted type and the underlying type are the same except for
      // signedness. Ask the AST for the correctly corresponding type and see
      // if that's compatible.
      if (!PromoteType.isNull() && !UnderlyingType->isBooleanType() &&
          PromoteType->isUnsignedIntegerType() !=
              UnderlyingType->isUnsignedIntegerType()) {
        UnderlyingType =
            UnderlyingType->isUnsignedIntegerType()
                ? Context.getCorrespondingSignedType(UnderlyingType)
                : Context.getCorrespondingUnsignedType(UnderlyingType);
        if (Context.typesAreCompatible(PromoteType, UnderlyingType,
                                       /*CompareUnqualified*/ true))
          PromoteType = QualType();
      }
    }
    if (TInfo->getType()->isSpecificBuiltinType(BuiltinType::Float))
      PromoteType = Context.DoubleTy;
    if (!PromoteType.isNull())
      DiagRuntimeBehavior(TInfo->getTypeLoc().getBeginLoc(), E,
                  PDiag(diag::warn_second_parameter_to_va_arg_never_compatible)
                          << TInfo->getType()
                          << PromoteType
                          << TInfo->getTypeLoc().getSourceRange());
  }

  QualType T = TInfo->getType().getNonLValueExprType(Context);
  return new (Context) VAArgExpr(BuiltinLoc, E, TInfo, RPLoc, T, IsMS);
}

ExprResult Sema::ActOnGNUNullExpr(SourceLocation TokenLoc) {
  // The type of __null will be int or long, depending on the size of
  // pointers on the target.
  QualType Ty;
  unsigned pw = Context.getTargetInfo().getPointerWidth(LangAS::Default);
  if (pw == Context.getTargetInfo().getIntWidth())
    Ty = Context.IntTy;
  else if (pw == Context.getTargetInfo().getLongWidth())
    Ty = Context.LongTy;
  else if (pw == Context.getTargetInfo().getLongLongWidth())
    Ty = Context.LongLongTy;
  else {
    llvm_unreachable("I don't know size of pointer!");
  }

  return new (Context) GNUNullExpr(Ty, TokenLoc);
}

static CXXRecordDecl *LookupStdSourceLocationImpl(Sema &S, SourceLocation Loc) {
  CXXRecordDecl *ImplDecl = nullptr;

  // Fetch the std::source_location::__impl decl.
  if (NamespaceDecl *Std = S.getStdNamespace()) {
    LookupResult ResultSL(S, &S.PP.getIdentifierTable().get("source_location"),
                          Loc, Sema::LookupOrdinaryName);
    if (S.LookupQualifiedName(ResultSL, Std)) {
      if (auto *SLDecl = ResultSL.getAsSingle<RecordDecl>()) {
        LookupResult ResultImpl(S, &S.PP.getIdentifierTable().get("__impl"),
                                Loc, Sema::LookupOrdinaryName);
        if ((SLDecl->isCompleteDefinition() || SLDecl->isBeingDefined()) &&
            S.LookupQualifiedName(ResultImpl, SLDecl)) {
          ImplDecl = ResultImpl.getAsSingle<CXXRecordDecl>();
        }
      }
    }
  }

  if (!ImplDecl || !ImplDecl->isCompleteDefinition()) {
    S.Diag(Loc, diag::err_std_source_location_impl_not_found);
    return nullptr;
  }

  // Verify that __impl is a trivial struct type, with no base classes, and with
  // only the four expected fields.
  if (ImplDecl->isUnion() || !ImplDecl->isStandardLayout() ||
      ImplDecl->getNumBases() != 0) {
    S.Diag(Loc, diag::err_std_source_location_impl_malformed);
    return nullptr;
  }

  unsigned Count = 0;
  for (FieldDecl *F : ImplDecl->fields()) {
    StringRef Name = F->getName();

    if (Name == "_M_file_name") {
      if (F->getType() !=
          S.Context.getPointerType(S.Context.CharTy.withConst()))
        break;
      Count++;
    } else if (Name == "_M_function_name") {
      if (F->getType() !=
          S.Context.getPointerType(S.Context.CharTy.withConst()))
        break;
      Count++;
    } else if (Name == "_M_line") {
      if (!F->getType()->isIntegerType())
        break;
      Count++;
    } else if (Name == "_M_column") {
      if (!F->getType()->isIntegerType())
        break;
      Count++;
    } else {
      Count = 100; // invalid
      break;
    }
  }
  if (Count != 4) {
    S.Diag(Loc, diag::err_std_source_location_impl_malformed);
    return nullptr;
  }

  return ImplDecl;
}

ExprResult Sema::ActOnSourceLocExpr(SourceLocIdentKind Kind,
                                    SourceLocation BuiltinLoc,
                                    SourceLocation RPLoc) {
  QualType ResultTy;
  switch (Kind) {
  case SourceLocIdentKind::File:
  case SourceLocIdentKind::FileName:
  case SourceLocIdentKind::Function:
  case SourceLocIdentKind::FuncSig: {
    QualType ArrTy = Context.getStringLiteralArrayType(Context.CharTy, 0);
    ResultTy =
        Context.getPointerType(ArrTy->getAsArrayTypeUnsafe()->getElementType());
    break;
  }
  case SourceLocIdentKind::Line:
  case SourceLocIdentKind::Column:
    ResultTy = Context.UnsignedIntTy;
    break;
  case SourceLocIdentKind::SourceLocStruct:
    if (!StdSourceLocationImplDecl) {
      StdSourceLocationImplDecl =
          LookupStdSourceLocationImpl(*this, BuiltinLoc);
      if (!StdSourceLocationImplDecl)
        return ExprError();
    }
    ResultTy = Context.getPointerType(
        Context.getCanonicalTagType(StdSourceLocationImplDecl).withConst());
    break;
  }

  return BuildSourceLocExpr(Kind, ResultTy, BuiltinLoc, RPLoc, CurContext);
}

ExprResult Sema::BuildSourceLocExpr(SourceLocIdentKind Kind, QualType ResultTy,
                                    SourceLocation BuiltinLoc,
                                    SourceLocation RPLoc,
                                    DeclContext *ParentContext) {
  return new (Context)
      SourceLocExpr(Context, Kind, ResultTy, BuiltinLoc, RPLoc, ParentContext);
}

ExprResult Sema::ActOnEmbedExpr(SourceLocation EmbedKeywordLoc,
                                StringLiteral *BinaryData, StringRef FileName) {
  EmbedDataStorage *Data = new (Context) EmbedDataStorage;
  Data->BinaryData = BinaryData;
  Data->FileName = FileName;
  return new (Context)
      EmbedExpr(Context, EmbedKeywordLoc, Data, /*NumOfElements=*/0,
                Data->getDataElementCount());
}

static bool maybeDiagnoseAssignmentToFunction(Sema &S, QualType DstType,
                                              const Expr *SrcExpr) {
  if (!DstType->isFunctionPointerType() ||
      !SrcExpr->getType()->isFunctionType())
    return false;

  auto *DRE = dyn_cast<DeclRefExpr>(SrcExpr->IgnoreParenImpCasts());
  if (!DRE)
    return false;

  auto *FD = dyn_cast<FunctionDecl>(DRE->getDecl());
  if (!FD)
    return false;

  return !S.checkAddressOfFunctionIsAvailable(FD,
                                              /*Complain=*/true,
                                              SrcExpr->getBeginLoc());
}

bool Sema::DiagnoseAssignmentResult(AssignConvertType ConvTy,
                                    SourceLocation Loc,
                                    QualType DstType, QualType SrcType,
                                    Expr *SrcExpr, AssignmentAction Action,
                                    bool *Complained) {
  if (Complained)
    *Complained = false;

  // Decode the result (notice that AST's are still created for extensions).
  bool CheckInferredResultType = false;
  bool isInvalid = false;
  unsigned DiagKind = 0;
  ConversionFixItGenerator ConvHints;
  bool MayHaveConvFixit = false;
  bool MayHaveFunctionDiff = false;
  const ObjCInterfaceDecl *IFace = nullptr;
  const ObjCProtocolDecl *PDecl = nullptr;

  switch (ConvTy) {
  case AssignConvertType::Compatible:
    DiagnoseAssignmentEnum(DstType, SrcType, SrcExpr);
    return false;
  case AssignConvertType::CompatibleVoidPtrToNonVoidPtr:
    // Still a valid conversion, but we may want to diagnose for C++
    // compatibility reasons.
    DiagKind = diag::warn_compatible_implicit_pointer_conv;
    break;
  case AssignConvertType::PointerToInt:
    if (getLangOpts().CPlusPlus) {
      DiagKind = diag::err_typecheck_convert_pointer_int;
      isInvalid = true;
    } else {
      DiagKind = diag::ext_typecheck_convert_pointer_int;
    }
    ConvHints.tryToFixConversion(SrcExpr, SrcType, DstType, *this);
    MayHaveConvFixit = true;
    break;
  case AssignConvertType::IntToPointer:
    if (getLangOpts().CPlusPlus) {
      DiagKind = diag::err_typecheck_convert_int_pointer;
      isInvalid = true;
    } else {
      DiagKind = diag::ext_typecheck_convert_int_pointer;
    }
    ConvHints.tryToFixConversion(SrcExpr, SrcType, DstType, *this);
    MayHaveConvFixit = true;
    break;
  case AssignConvertType::IncompatibleFunctionPointerStrict:
    DiagKind =
        diag::warn_typecheck_convert_incompatible_function_pointer_strict;
    ConvHints.tryToFixConversion(SrcExpr, SrcType, DstType, *this);
    MayHaveConvFixit = true;
    break;
  case AssignConvertType::IncompatibleFunctionPointer:
    if (getLangOpts().CPlusPlus) {
      DiagKind = diag::err_typecheck_convert_incompatible_function_pointer;
      isInvalid = true;
    } else {
      DiagKind = diag::ext_typecheck_convert_incompatible_function_pointer;
    }
    ConvHints.tryToFixConversion(SrcExpr, SrcType, DstType, *this);
    MayHaveConvFixit = true;
    break;
  case AssignConvertType::IncompatiblePointer:
    if (Action == AssignmentAction::Passing_CFAudited) {
      DiagKind = diag::err_arc_typecheck_convert_incompatible_pointer;
    } else if (getLangOpts().CPlusPlus) {
      DiagKind = diag::err_typecheck_convert_incompatible_pointer;
      isInvalid = true;
    } else {
      DiagKind = diag::ext_typecheck_convert_incompatible_pointer;
    }
    CheckInferredResultType = DstType->isObjCObjectPointerType() &&
      SrcType->isObjCObjectPointerType();
    if (CheckInferredResultType) {
      SrcType = SrcType.getUnqualifiedType();
      DstType = DstType.getUnqualifiedType();
    } else {
      ConvHints.tryToFixConversion(SrcExpr, SrcType, DstType, *this);
    }
    MayHaveConvFixit = true;
    break;
  case AssignConvertType::IncompatiblePointerSign:
    if (getLangOpts().CPlusPlus) {
      DiagKind = diag::err_typecheck_convert_incompatible_pointer_sign;
      isInvalid = true;
    } else {
      DiagKind = diag::ext_typecheck_convert_incompatible_pointer_sign;
    }
    break;
  case AssignConvertType::FunctionVoidPointer:
    if (getLangOpts().CPlusPlus) {
      DiagKind = diag::err_typecheck_convert_pointer_void_func;
      isInvalid = true;
    } else {
      DiagKind = diag::ext_typecheck_convert_pointer_void_func;
    }
    break;
  case AssignConvertType::IncompatiblePointerDiscardsQualifiers: {
    // Perform array-to-pointer decay if necessary.
    if (SrcType->isArrayType()) SrcType = Context.getArrayDecayedType(SrcType);

    isInvalid = true;

    Qualifiers lhq = SrcType->getPointeeType().getQualifiers();
    Qualifiers rhq = DstType->getPointeeType().getQualifiers();
    if (lhq.getAddressSpace() != rhq.getAddressSpace()) {
      DiagKind = diag::err_typecheck_incompatible_address_space;
      break;
    } else if (lhq.getObjCLifetime() != rhq.getObjCLifetime()) {
      DiagKind = diag::err_typecheck_incompatible_ownership;
      break;
    } else if (!lhq.getPointerAuth().isEquivalent(rhq.getPointerAuth())) {
      DiagKind = diag::err_typecheck_incompatible_ptrauth;
      break;
    }

    llvm_unreachable("unknown error case for discarding qualifiers!");
    // fallthrough
  }
  case AssignConvertType::IncompatiblePointerDiscardsOverflowBehavior:
    if (SrcType->isArrayType())
      SrcType = Context.getArrayDecayedType(SrcType);

    DiagKind = diag::ext_typecheck_convert_discards_overflow_behavior;
    break;
  case AssignConvertType::CompatiblePointerDiscardsQualifiers:
    // If the qualifiers lost were because we were applying the
    // (deprecated) C++ conversion from a string literal to a char*
    // (or wchar_t*), then there was no error (C++ 4.2p2).  FIXME:
    // Ideally, this check would be performed in
    // checkPointerTypesForAssignment. However, that would require a
    // bit of refactoring (so that the second argument is an
    // expression, rather than a type), which should be done as part
    // of a larger effort to fix checkPointerTypesForAssignment for
    // C++ semantics.
    if (getLangOpts().CPlusPlus &&
        IsStringLiteralToNonConstPointerConversion(SrcExpr, DstType))
      return false;
    if (getLangOpts().CPlusPlus) {
      DiagKind =  diag::err_typecheck_convert_discards_qualifiers;
      isInvalid = true;
    } else {
      DiagKind =  diag::ext_typecheck_convert_discards_qualifiers;
    }

    break;
  case AssignConvertType::IncompatibleNestedPointerQualifiers:
    if (getLangOpts().CPlusPlus) {
      isInvalid = true;
      DiagKind = diag::err_nested_pointer_qualifier_mismatch;
    } else {
      DiagKind = diag::ext_nested_pointer_qualifier_mismatch;
    }
    break;
  case AssignConvertType::IncompatibleNestedPointerAddressSpaceMismatch:
    DiagKind = diag::err_typecheck_incompatible_nested_address_space;
    isInvalid = true;
    break;
  case AssignConvertType::IntToBlockPointer:
    DiagKind = diag::err_int_to_block_pointer;
    isInvalid = true;
    break;
  case AssignConvertType::IncompatibleBlockPointer:
    DiagKind = diag::err_typecheck_convert_incompatible_block_pointer;
    isInvalid = true;
    break;
  case AssignConvertType::IncompatibleObjCQualifiedId: {
    if (SrcType->isObjCQualifiedIdType()) {
      const ObjCObjectPointerType *srcOPT =
                SrcType->castAs<ObjCObjectPointerType>();
      for (auto *srcProto : srcOPT->quals()) {
        PDecl = srcProto;
        break;
      }
      if (const ObjCInterfaceType *IFaceT =
            DstType->castAs<ObjCObjectPointerType>()->getInterfaceType())
        IFace = IFaceT->getDecl();
    }
    else if (DstType->isObjCQualifiedIdType()) {
      const ObjCObjectPointerType *dstOPT =
        DstType->castAs<ObjCObjectPointerType>();
      for (auto *dstProto : dstOPT->quals()) {
        PDecl = dstProto;
        break;
      }
      if (const ObjCInterfaceType *IFaceT =
            SrcType->castAs<ObjCObjectPointerType>()->getInterfaceType())
        IFace = IFaceT->getDecl();
    }
    if (getLangOpts().CPlusPlus) {
      DiagKind = diag::err_incompatible_qualified_id;
      isInvalid = true;
    } else {
      DiagKind = diag::warn_incompatible_qualified_id;
    }
    break;
  }
  case AssignConvertType::IncompatibleVectors:
    if (getLangOpts().CPlusPlus) {
      DiagKind = diag::err_incompatible_vectors;
      isInvalid = true;
    } else {
      DiagKind = diag::warn_incompatible_vectors;
    }
    break;
  case AssignConvertType::IncompatibleObjCWeakRef:
    DiagKind = diag::err_arc_weak_unavailable_assign;
    isInvalid = true;
    break;
  case AssignConvertType::CompatibleOBTDiscards:
    return false;
  case AssignConvertType::IncompatibleOBTKinds: {
    auto getOBTKindName = [](QualType Ty) -> StringRef {
      if (Ty->isPointerType())
        Ty = Ty->getPointeeType();
      if (const auto *OBT = Ty->getAs<OverflowBehaviorType>()) {
        return OBT->getBehaviorKind() ==
                       OverflowBehaviorType::OverflowBehaviorKind::Trap
                   ? "__ob_trap"
                   : "__ob_wrap";
      }
      llvm_unreachable("OBT kind unhandled");
    };

    Diag(Loc, diag::err_incompatible_obt_kinds_assignment)
        << DstType << SrcType << getOBTKindName(DstType)
        << getOBTKindName(SrcType);
    isInvalid = true;
    return true;
  }
  case AssignConvertType::Incompatible:
    if (maybeDiagnoseAssignmentToFunction(*this, DstType, SrcExpr)) {
      if (Complained)
        *Complained = true;
      return true;
    }

    DiagKind = diag::err_typecheck_convert_incompatible;
    ConvHints.tryToFixConversion(SrcExpr, SrcType, DstType, *this);
    MayHaveConvFixit = true;
    isInvalid = true;
    MayHaveFunctionDiff = true;
    break;
  }

  QualType FirstType, SecondType;
  switch (Action) {
  case AssignmentAction::Assigning:
  case AssignmentAction::Initializing:
    // The destination type comes first.
    FirstType = DstType;
    SecondType = SrcType;
    break;

  case AssignmentAction::Returning:
  case AssignmentAction::Passing:
  case AssignmentAction::Passing_CFAudited:
  case AssignmentAction::Converting:
  case AssignmentAction::Sending:
  case AssignmentAction::Casting:
    // The source type comes first.
    FirstType = SrcType;
    SecondType = DstType;
    break;
  }

  PartialDiagnostic FDiag = PDiag(DiagKind);
  AssignmentAction ActionForDiag = Action;
  if (Action == AssignmentAction::Passing_CFAudited)
    ActionForDiag = AssignmentAction::Passing;

  FDiag << FirstType << SecondType << ActionForDiag
        << SrcExpr->getSourceRange();

  if (DiagKind == diag::ext_typecheck_convert_incompatible_pointer_sign ||
      DiagKind == diag::err_typecheck_convert_incompatible_pointer_sign) {
    auto isPlainChar = [](const clang::Type *Type) {
      return Type->isSpecificBuiltinType(BuiltinType::Char_S) ||
             Type->isSpecificBuiltinType(BuiltinType::Char_U);
    };
    FDiag << (isPlainChar(FirstType->getPointeeOrArrayElementType()) ||
              isPlainChar(SecondType->getPointeeOrArrayElementType()));
  }

  // If we can fix the conversion, suggest the FixIts.
  if (!ConvHints.isNull()) {
    for (FixItHint &H : ConvHints.Hints)
      FDiag << H;
  }

  if (MayHaveConvFixit) { FDiag << (unsigned) (ConvHints.Kind); }

  if (MayHaveFunctionDiff)
    HandleFunctionTypeMismatch(FDiag, SecondType, FirstType);

  Diag(Loc, FDiag);
  if ((DiagKind == diag::warn_incompatible_qualified_id ||
       DiagKind == diag::err_incompatible_qualified_id) &&
      PDecl && IFace && !IFace->hasDefinition())
    Diag(IFace->getLocation(), diag::note_incomplete_class_and_qualified_id)
        << IFace << PDecl;

  if (SecondType == Context.OverloadTy)
    NoteAllOverloadCandidates(OverloadExpr::find(SrcExpr).Expression,
                              FirstType, /*TakingAddress=*/true);

  if (CheckInferredResultType)
    ObjC().EmitRelatedResultTypeNote(SrcExpr);

  if (Action == AssignmentAction::Returning &&
      ConvTy == AssignConvertType::IncompatiblePointer)
    ObjC().EmitRelatedResultTypeNoteForReturn(DstType);

  if (Complained)
    *Complained = true;
  return isInvalid;
}

ExprResult Sema::VerifyIntegerConstantExpression(Expr *E,
                                                 llvm::APSInt *Result,
                                                 AllowFoldKind CanFold) {
  class SimpleICEDiagnoser : public VerifyICEDiagnoser {
  public:
    SemaDiagnosticBuilder diagnoseNotICEType(Sema &S, SourceLocation Loc,
                                             QualType T) override {
      return S.Diag(Loc, diag::err_ice_not_integral)
             << T << S.LangOpts.CPlusPlus;
    }
    SemaDiagnosticBuilder diagnoseNotICE(Sema &S, SourceLocation Loc) override {
      return S.Diag(Loc, diag::err_expr_not_ice) << S.LangOpts.CPlusPlus;
    }
  } Diagnoser;

  return VerifyIntegerConstantExpression(E, Result, Diagnoser, CanFold);
}

ExprResult Sema::VerifyIntegerConstantExpression(Expr *E,
                                                 llvm::APSInt *Result,
                                                 unsigned DiagID,
                                                 AllowFoldKind CanFold) {
  class IDDiagnoser : public VerifyICEDiagnoser {
    unsigned DiagID;

  public:
    IDDiagnoser(unsigned DiagID)
      : VerifyICEDiagnoser(DiagID == 0), DiagID(DiagID) { }

    SemaDiagnosticBuilder diagnoseNotICE(Sema &S, SourceLocation Loc) override {
      return S.Diag(Loc, DiagID);
    }
  } Diagnoser(DiagID);

  return VerifyIntegerConstantExpression(E, Result, Diagnoser, CanFold);
}

Sema::SemaDiagnosticBuilder
Sema::VerifyICEDiagnoser::diagnoseNotICEType(Sema &S, SourceLocation Loc,
                                             QualType T) {
  return diagnoseNotICE(S, Loc);
}

Sema::SemaDiagnosticBuilder
Sema::VerifyICEDiagnoser::diagnoseFold(Sema &S, SourceLocation Loc) {
  return S.Diag(Loc, diag::ext_expr_not_ice) << S.LangOpts.CPlusPlus;
}

ExprResult
Sema::VerifyIntegerConstantExpression(Expr *E, llvm::APSInt *Result,
                                      VerifyICEDiagnoser &Diagnoser,
                                      AllowFoldKind CanFold) {
  SourceLocation DiagLoc = E->getBeginLoc();

  if (getLangOpts().CPlusPlus11) {
    // C++11 [expr.const]p5:
    //   If an expression of literal class type is used in a context where an
    //   integral constant expression is required, then that class type shall
    //   have a single non-explicit conversion function to an integral or
    //   unscoped enumeration type
    ExprResult Converted;
    class CXX11ConvertDiagnoser : public ICEConvertDiagnoser {
      VerifyICEDiagnoser &BaseDiagnoser;
    public:
      CXX11ConvertDiagnoser(VerifyICEDiagnoser &BaseDiagnoser)
          : ICEConvertDiagnoser(/*AllowScopedEnumerations*/ false,
                                BaseDiagnoser.Suppress, true),
            BaseDiagnoser(BaseDiagnoser) {}

      SemaDiagnosticBuilder diagnoseNotInt(Sema &S, SourceLocation Loc,
                                           QualType T) override {
        return BaseDiagnoser.diagnoseNotICEType(S, Loc, T);
      }

      SemaDiagnosticBuilder diagnoseIncomplete(
          Sema &S, SourceLocation Loc, QualType T) override {
        return S.Diag(Loc, diag::err_ice_incomplete_type) << T;
      }

      SemaDiagnosticBuilder diagnoseExplicitConv(
          Sema &S, SourceLocation Loc, QualType T, QualType ConvTy) override {
        return S.Diag(Loc, diag::err_ice_explicit_conversion) << T << ConvTy;
      }

      SemaDiagnosticBuilder noteExplicitConv(
          Sema &S, CXXConversionDecl *Conv, QualType ConvTy) override {
        return S.Diag(Conv->getLocation(), diag::note_ice_conversion_here)
                 << ConvTy->isEnumeralType() << ConvTy;
      }

      SemaDiagnosticBuilder diagnoseAmbiguous(
          Sema &S, SourceLocation Loc, QualType T) override {
        return S.Diag(Loc, diag::err_ice_ambiguous_conversion) << T;
      }

      SemaDiagnosticBuilder noteAmbiguous(
          Sema &S, CXXConversionDecl *Conv, QualType ConvTy) override {
        return S.Diag(Conv->getLocation(), diag::note_ice_conversion_here)
                 << ConvTy->isEnumeralType() << ConvTy;
      }

      SemaDiagnosticBuilder diagnoseConversion(
          Sema &S, SourceLocation Loc, QualType T, QualType ConvTy) override {
        llvm_unreachable("conversion functions are permitted");
      }
    } ConvertDiagnoser(Diagnoser);

    Converted = PerformContextualImplicitConversion(DiagLoc, E,
                                                    ConvertDiagnoser);
    if (Converted.isInvalid())
      return Converted;
    E = Converted.get();
    // The 'explicit' case causes us to get a RecoveryExpr.  Give up here so we
    // don't try to evaluate it later. We also don't want to return the
    // RecoveryExpr here, as it results in this call succeeding, thus callers of
    // this function will attempt to use 'Value'.
    if (isa<RecoveryExpr>(E))
      return ExprError();
    if (!E->getType()->isIntegralOrUnscopedEnumerationType())
      return ExprError();
  } else if (!E->getType()->isIntegralOrUnscopedEnumerationType()) {
    // An ICE must be of integral or unscoped enumeration type.
    if (!Diagnoser.Suppress)
      Diagnoser.diagnoseNotICEType(*this, DiagLoc, E->getType())
          << E->getSourceRange();
    return ExprError();
  }

  ExprResult RValueExpr = DefaultLvalueConversion(E);
  if (RValueExpr.isInvalid())
    return ExprError();

  E = RValueExpr.get();

  // Circumvent ICE checking in C++11 to avoid evaluating the expression twice
  // in the non-ICE case.
  if (!getLangOpts().CPlusPlus11 && E->isIntegerConstantExpr(Context)) {
    SmallVector<PartialDiagnosticAt, 8> Notes;
    if (Result)
      *Result = E->EvaluateKnownConstIntCheckOverflow(Context, &Notes);
    if (!isa<ConstantExpr>(E))
      E = Result ? ConstantExpr::Create(Context, E, APValue(*Result))
                 : ConstantExpr::Create(Context, E);

    if (Notes.empty())
      return E;

    // If our only note is the usual "invalid subexpression" note, just point
    // the caret at its location rather than producing an essentially
    // redundant note.
    if (Notes.size() == 1 && Notes[0].second.getDiagID() ==
          diag::note_invalid_subexpr_in_const_expr) {
      DiagLoc = Notes[0].first;
      Notes.clear();
    }

    if (getLangOpts().CPlusPlus) {
      if (!Diagnoser.Suppress) {
        Diagnoser.diagnoseNotICE(*this, DiagLoc) << E->getSourceRange();
        for (const PartialDiagnosticAt &Note : Notes)
          Diag(Note.first, Note.second);
      }
      return ExprError();
    }

    Diagnoser.diagnoseFold(*this, DiagLoc) << E->getSourceRange();
    for (const PartialDiagnosticAt &Note : Notes)
      Diag(Note.first, Note.second);

    return E;
  }

  Expr::EvalResult EvalResult;
  SmallVector<PartialDiagnosticAt, 8> Notes;
  EvalResult.Diag = &Notes;

  // Try to evaluate the expression, and produce diagnostics explaining why it's
  // not a constant expression as a side-effect.
  bool Folded =
      E->EvaluateAsRValue(EvalResult, Context, /*isConstantContext*/ true) &&
      EvalResult.Val.isInt() && !EvalResult.HasSideEffects &&
      (!getLangOpts().CPlusPlus || !EvalResult.HasUndefinedBehavior);

  if (!isa<ConstantExpr>(E))
    E = ConstantExpr::Create(Context, E, EvalResult.Val);

  // In C++11, we can rely on diagnostics being produced for any expression
  // which is not a constant expression. If no diagnostics were produced, then
  // this is a constant expression.
  if (Folded && getLangOpts().CPlusPlus11 && Notes.empty()) {
    if (Result)
      *Result = EvalResult.Val.getInt();
    return E;
  }

  // If our only note is the usual "invalid subexpression" note, just point
  // the caret at its location rather than producing an essentially
  // redundant note.
  if (Notes.size() == 1 && Notes[0].second.getDiagID() ==
        diag::note_invalid_subexpr_in_const_expr) {
    DiagLoc = Notes[0].first;
    Notes.clear();
  }

  if (!Folded || CanFold == AllowFoldKind::No) {
    if (!Diagnoser.Suppress) {
      Diagnoser.diagnoseNotICE(*this, DiagLoc) << E->getSourceRange();
      for (const PartialDiagnosticAt &Note : Notes)
        Diag(Note.first, Note.second);
    }

    return ExprError();
  }

  Diagnoser.diagnoseFold(*this, DiagLoc) << E->getSourceRange();
  for (const PartialDiagnosticAt &Note : Notes)
    Diag(Note.first, Note.second);

  if (Result)
    *Result = EvalResult.Val.getInt();
  return E;
}

namespace {
  // Handle the case where we conclude a expression which we speculatively
  // considered to be unevaluated is actually evaluated.
  class TransformToPE : public TreeTransform<TransformToPE> {
    typedef TreeTransform<TransformToPE> BaseTransform;

  public:
    TransformToPE(Sema &SemaRef) : BaseTransform(SemaRef) { }

    // Make sure we redo semantic analysis
    bool AlwaysRebuild() { return true; }
    bool ReplacingOriginal() { return true; }

    // We need to special-case DeclRefExprs referring to FieldDecls which
    // are not part of a member pointer formation; normal TreeTransforming
    // doesn't catch this case because of the way we represent them in the AST.
    // FIXME: This is a bit ugly; is it really the best way to handle this
    // case?
    //
    // Error on DeclRefExprs referring to FieldDecls.
    ExprResult TransformDeclRefExpr(DeclRefExpr *E) {
      if (isa<FieldDecl>(E->getDecl()) &&
          !SemaRef.isUnevaluatedContext())
        return SemaRef.Diag(E->getLocation(),
                            diag::err_invalid_non_static_member_use)
            << E->getDecl() << E->getSourceRange();

      return BaseTransform::TransformDeclRefExpr(E);
    }

    // Exception: filter out member pointer formation
    ExprResult TransformUnaryOperator(UnaryOperator *E) {
      if (E->getOpcode() == UO_AddrOf && E->getType()->isMemberPointerType())
        return E;

      return BaseTransform::TransformUnaryOperator(E);
    }

    // The body of a lambda-expression is in a separate expression evaluation
    // context so never needs to be transformed.
    // FIXME: Ideally we wouldn't transform the closure type either, and would
    // just recreate the capture expressions and lambda expression.
    StmtResult TransformLambdaBody(LambdaExpr *E, Stmt *Body) {
      return SkipLambdaBody(E, Body);
    }
  };
}

ExprResult Sema::TransformToPotentiallyEvaluated(Expr *E) {
  assert(isUnevaluatedContext() &&
         "Should only transform unevaluated expressions");
  ExprEvalContexts.back().Context =
      ExprEvalContexts[ExprEvalContexts.size()-2].Context;
  if (isUnevaluatedContext())
    return E;
  return TransformToPE(*this).TransformExpr(E);
}

TypeSourceInfo *Sema::TransformToPotentiallyEvaluated(TypeSourceInfo *TInfo) {
  assert(isUnevaluatedContext() &&
         "Should only transform unevaluated expressions");
  ExprEvalContexts.back().Context = parentEvaluationContext().Context;
  if (isUnevaluatedContext())
    return TInfo;
  return TransformToPE(*this).TransformType(TInfo);
}

void
Sema::PushExpressionEvaluationContext(
    ExpressionEvaluationContext NewContext, Decl *LambdaContextDecl,
    ExpressionEvaluationContextRecord::ExpressionKind ExprContext) {
  ExprEvalContexts.emplace_back(NewContext, ExprCleanupObjects.size(), Cleanup,
                                LambdaContextDecl, ExprContext);

  // Discarded statements and immediate contexts nested in other
  // discarded statements or immediate context are themselves
  // a discarded statement or an immediate context, respectively.
  ExprEvalContexts.back().InDiscardedStatement =
      parentEvaluationContext().isDiscardedStatementContext();

  // C++23 [expr.const]/p15
  // An expression or conversion is in an immediate function context if [...]
  // it is a subexpression of a manifestly constant-evaluated expression or
  // conversion.
  const auto &Prev = parentEvaluationContext();
  ExprEvalContexts.back().InImmediateFunctionContext =
      Prev.isImmediateFunctionContext() || Prev.isConstantEvaluated();

  ExprEvalContexts.back().InImmediateEscalatingFunctionContext =
      Prev.InImmediateEscalatingFunctionContext;

  Cleanup.reset();
  if (!MaybeODRUseExprs.empty())
    std::swap(MaybeODRUseExprs, ExprEvalContexts.back().SavedMaybeODRUseExprs);
}

void
Sema::PushExpressionEvaluationContext(
    ExpressionEvaluationContext NewContext, ReuseLambdaContextDecl_t,
    ExpressionEvaluationContextRecord::ExpressionKind ExprContext) {
  Decl *ClosureContextDecl = ExprEvalContexts.back().ManglingContextDecl;
  PushExpressionEvaluationContext(NewContext, ClosureContextDecl, ExprContext);
}

void Sema::PushExpressionEvaluationContextForFunction(
    ExpressionEvaluationContext NewContext, FunctionDecl *FD) {
  // [expr.const]/p14.1
  // An expression or conversion is in an immediate function context if it is
  // potentially evaluated and either: its innermost enclosing non-block scope
  // is a function parameter scope of an immediate function.
  PushExpressionEvaluationContext(
      FD && FD->isConsteval()
          ? ExpressionEvaluationContext::ImmediateFunctionContext
          : NewContext);
  const Sema::ExpressionEvaluationContextRecord &Parent =
      parentEvaluationContext();
  Sema::ExpressionEvaluationContextRecord &Current = currentEvaluationContext();

  Current.InDiscardedStatement = false;

  if (FD) {

    // Each ExpressionEvaluationContextRecord also keeps track of whether the
    // context is nested in an immediate function context, so smaller contexts
    // that appear inside immediate functions (like variable initializers) are
    // considered to be inside an immediate function context even though by
    // themselves they are not immediate function contexts. But when a new
    // function is entered, we need to reset this tracking, since the entered
    // function might be not an immediate function.

    Current.InImmediateEscalatingFunctionContext =
        getLangOpts().CPlusPlus20 && FD->isImmediateEscalating();

    if (isLambdaMethod(FD))
      Current.InImmediateFunctionContext =
          FD->isConsteval() ||
          (isLambdaMethod(FD) && (Parent.isConstantEvaluated() ||
                                  Parent.isImmediateFunctionContext()));
    else
      Current.InImmediateFunctionContext = FD->isConsteval();
  }
}

ExprResult Sema::ActOnCXXReflectExpr(SourceLocation CaretCaretLoc,
                                     TypeSourceInfo *TSI) {
  return BuildCXXReflectExpr(CaretCaretLoc, TSI);
}

ExprResult Sema::BuildCXXReflectExpr(SourceLocation CaretCaretLoc,
                                     TypeSourceInfo *TSI) {
  return CXXReflectExpr::Create(Context, CaretCaretLoc, TSI);
}

namespace {

const DeclRefExpr *CheckPossibleDeref(Sema &S, const Expr *PossibleDeref) {
  PossibleDeref = PossibleDeref->IgnoreParenImpCasts();
  if (const auto *E = dyn_cast<UnaryOperator>(PossibleDeref)) {
    if (E->getOpcode() == UO_Deref)
      return CheckPossibleDeref(S, E->getSubExpr());
  } else if (const auto *E = dyn_cast<ArraySubscriptExpr>(PossibleDeref)) {
    return CheckPossibleDeref(S, E->getBase());
  } else if (const auto *E = dyn_cast<MemberExpr>(PossibleDeref)) {
    return CheckPossibleDeref(S, E->getBase());
  } else if (const auto E = dyn_cast<DeclRefExpr>(PossibleDeref)) {
    QualType Inner;
    QualType Ty = E->getType();
    if (const auto *Ptr = Ty->getAs<PointerType>())
      Inner = Ptr->getPointeeType();
    else if (const auto *Arr = S.Context.getAsArrayType(Ty))
      Inner = Arr->getElementType();
    else
      return nullptr;

    if (Inner->hasAttr(attr::NoDeref))
      return E;
  }
  return nullptr;
}

} // namespace

void Sema::WarnOnPendingNoDerefs(ExpressionEvaluationContextRecord &Rec) {
  for (const Expr *E : Rec.PossibleDerefs) {
    const DeclRefExpr *DeclRef = CheckPossibleDeref(*this, E);
    if (DeclRef) {
      const ValueDecl *Decl = DeclRef->getDecl();
      Diag(E->getExprLoc(), diag::warn_dereference_of_noderef_type)
          << Decl->getName() << E->getSourceRange();
      Diag(Decl->getLocation(), diag::note_previous_decl) << Decl->getName();
    } else {
      Diag(E->getExprLoc(), diag::warn_dereference_of_noderef_type_no_decl)
          << E->getSourceRange();
    }
  }
  Rec.PossibleDerefs.clear();
}

void Sema::CheckUnusedVolatileAssignment(Expr *E) {
  if (!E->getType().isVolatileQualified() || !getLangOpts().CPlusPlus20)
    return;

  // Note: ignoring parens here is not justified by the standard rules, but
  // ignoring parentheses seems like a more reasonable approach, and this only
  // drives a deprecation warning so doesn't affect conformance.
  if (auto *BO = dyn_cast<BinaryOperator>(E->IgnoreParenImpCasts())) {
    if (BO->getOpcode() == BO_Assign) {
      auto &LHSs = ExprEvalContexts.back().VolatileAssignmentLHSs;
      llvm::erase(LHSs, BO->getLHS());
    }
  }
}

void Sema::MarkExpressionAsImmediateEscalating(Expr *E) {
  assert(getLangOpts().CPlusPlus20 &&
         ExprEvalContexts.back().InImmediateEscalatingFunctionContext &&
         "Cannot mark an immediate escalating expression outside of an "
         "immediate escalating context");
  if (auto *Call = dyn_cast<CallExpr>(E->IgnoreImplicit());
      Call && Call->getCallee()) {
    if (auto *DeclRef =
            dyn_cast<DeclRefExpr>(Call->getCallee()->IgnoreImplicit()))
      DeclRef->setIsImmediateEscalating(true);
  } else if (auto *Ctr = dyn_cast<CXXConstructExpr>(E->IgnoreImplicit())) {
    Ctr->setIsImmediateEscalating(true);
  } else if (auto *DeclRef = dyn_cast<DeclRefExpr>(E->IgnoreImplicit())) {
    DeclRef->setIsImmediateEscalating(true);
  } else {
    assert(false && "expected an immediately escalating expression");
  }
  if (FunctionScopeInfo *FI = getCurFunction())
    FI->FoundImmediateEscalatingExpression = true;
}

ExprResult Sema::CheckForImmediateInvocation(ExprResult E, FunctionDecl *Decl) {
  if (isUnevaluatedContext() || !E.isUsable() || !Decl ||
      !Decl->isImmediateFunction() || isAlwaysConstantEvaluatedContext() ||
      isCheckingDefaultArgumentOrInitializer() ||
      RebuildingImmediateInvocation || isImmediateFunctionContext())
    return E;

  /// Opportunistically remove the callee from ReferencesToConsteval if we can.
  /// It's OK if this fails; we'll also remove this in
  /// HandleImmediateInvocations, but catching it here allows us to avoid
  /// walking the AST looking for it in simple cases.
  if (auto *Call = dyn_cast<CallExpr>(E.get()->IgnoreImplicit()))
    if (auto *DeclRef =
            dyn_cast<DeclRefExpr>(Call->getCallee()->IgnoreImplicit()))
      ExprEvalContexts.back().ReferenceToConsteval.erase(DeclRef);

  // C++23 [expr.const]/p16
  // An expression or conversion is immediate-escalating if it is not initially
  // in an immediate function context and it is [...] an immediate invocation
  // that is not a constant expression and is not a subexpression of an
  // immediate invocation.
  APValue Cached;
  auto CheckConstantExpressionAndKeepResult = [&]() {
    llvm::SmallVector<PartialDiagnosticAt, 8> Notes;
    Expr::EvalResult Eval;
    Eval.Diag = &Notes;
    bool Res = E.get()->EvaluateAsConstantExpr(
        Eval, getASTContext(), ConstantExprKind::ImmediateInvocation);
    if (Res && Notes.empty()) {
      Cached = std::move(Eval.Val);
      return true;
    }
    return false;
  };

  if (!E.get()->isValueDependent() &&
      ExprEvalContexts.back().InImmediateEscalatingFunctionContext &&
      !CheckConstantExpressionAndKeepResult()) {
    MarkExpressionAsImmediateEscalating(E.get());
    return E;
  }

  if (Cleanup.exprNeedsCleanups()) {
    // Since an immediate invocation is a full expression itself - it requires
    // an additional ExprWithCleanups node, but it can participate to a bigger
    // full expression which actually requires cleanups to be run after so
    // create ExprWithCleanups without using MaybeCreateExprWithCleanups as it
    // may discard cleanups for outer expression too early.

    // Note that ExprWithCleanups created here must always have empty cleanup
    // objects:
    // - compound literals do not create cleanup objects in C++ and immediate
    // invocations are C++-only.
    // - blocks are not allowed inside constant expressions and compiler will
    // issue an error if they appear there.
    //
    // Hence, in correct code any cleanup objects created inside current
    // evaluation context must be outside the immediate invocation.
    E = ExprWithCleanups::Create(getASTContext(), E.get(),
                                 Cleanup.cleanupsHaveSideEffects(), {});
  }

  ConstantExpr *Res = ConstantExpr::Create(
      getASTContext(), E.get(),
      ConstantExpr::getStorageKind(Decl->getReturnType().getTypePtr(),
                                   getASTContext()),
      /*IsImmediateInvocation*/ true);
  if (Cached.hasValue())
    Res->MoveIntoResult(Cached, getASTContext());
  /// Value-dependent constant expressions should not be immediately
  /// evaluated until they are instantiated.
  if (!Res->isValueDependent())
    ExprEvalContexts.back().ImmediateInvocationCandidates.emplace_back(Res, 0);
  return Res;
}

static void EvaluateAndDiagnoseImmediateInvocation(
    Sema &SemaRef, Sema::ImmediateInvocationCandidate Candidate) {
  llvm::SmallVector<PartialDiagnosticAt, 8> Notes;
  Expr::EvalResult Eval;
  Eval.Diag = &Notes;
  ConstantExpr *CE = Candidate.getPointer();
  bool Result = CE->EvaluateAsConstantExpr(
      Eval, SemaRef.getASTContext(), ConstantExprKind::ImmediateInvocation);
  if (!Result || !Notes.empty()) {
    SemaRef.FailedImmediateInvocations.insert(CE);
    Expr *InnerExpr = CE->getSubExpr()->IgnoreImplicit();
    if (auto *FunctionalCast = dyn_cast<CXXFunctionalCastExpr>(InnerExpr))
      InnerExpr = FunctionalCast->getSubExpr()->IgnoreImplicit();
    FunctionDecl *FD = nullptr;
    if (auto *Call = dyn_cast<CallExpr>(InnerExpr))
      FD = cast<FunctionDecl>(Call->getCalleeDecl());
    else if (auto *Call = dyn_cast<CXXConstructExpr>(InnerExpr))
      FD = Call->getConstructor();
    else if (auto *Cast = dyn_cast<CastExpr>(InnerExpr))
      FD = dyn_cast_or_null<FunctionDecl>(Cast->getConversionFunction());

    assert(FD && FD->isImmediateFunction() &&
           "could not find an immediate function in this expression");
    if (FD->isInvalidDecl())
      return;
    SemaRef.Diag(CE->getBeginLoc(), diag::err_invalid_consteval_call)
        << FD << FD->isConsteval();
    if (auto Context =
            SemaRef.InnermostDeclarationWithDelayedImmediateInvocations()) {
      SemaRef.Diag(Context->Loc, diag::note_invalid_consteval_initializer)
          << Context->Decl;
      SemaRef.Diag(Context->Decl->getBeginLoc(), diag::note_declared_at);
    }
    if (!FD->isConsteval())
      SemaRef.DiagnoseImmediateEscalatingReason(FD);
    for (auto &Note : Notes)
      SemaRef.Diag(Note.first, Note.second);
    return;
  }
  CE->MoveIntoResult(Eval.Val, SemaRef.getASTContext());
}

static void RemoveNestedImmediateInvocation(
    Sema &SemaRef, Sema::ExpressionEvaluationContextRecord &Rec,
    SmallVector<Sema::ImmediateInvocationCandidate, 4>::reverse_iterator It) {
  struct ComplexRemove : TreeTransform<ComplexRemove> {
    using Base = TreeTransform<ComplexRemove>;
    llvm::SmallPtrSetImpl<DeclRefExpr *> &DRSet;
    SmallVector<Sema::ImmediateInvocationCandidate, 4> &IISet;
    SmallVector<Sema::ImmediateInvocationCandidate, 4>::reverse_iterator
        CurrentII;
    ComplexRemove(Sema &SemaRef, llvm::SmallPtrSetImpl<DeclRefExpr *> &DR,
                  SmallVector<Sema::ImmediateInvocationCandidate, 4> &II,
                  SmallVector<Sema::ImmediateInvocationCandidate,
                              4>::reverse_iterator Current)
        : Base(SemaRef), DRSet(DR), IISet(II), CurrentII(Current) {}
    void RemoveImmediateInvocation(ConstantExpr* E) {
      auto It = std::find_if(CurrentII, IISet.rend(),
                             [E](Sema::ImmediateInvocationCandidate Elem) {
                               return Elem.getPointer() == E;
                             });
      // It is possible that some subexpression of the current immediate
      // invocation was handled from another expression evaluation context. Do
      // not handle the current immediate invocation if some of its
      // subexpressions failed before.
      if (It == IISet.rend()) {
        if (SemaRef.FailedImmediateInvocations.contains(E))
          CurrentII->setInt(1);
      } else {
        It->setInt(1); // Mark as deleted
      }
    }
    ExprResult TransformConstantExpr(ConstantExpr *E) {
      if (!E->isImmediateInvocation())
        return Base::TransformConstantExpr(E);
      RemoveImmediateInvocation(E);
      return Base::TransformExpr(E->getSubExpr());
    }
    /// Base::TransfromCXXOperatorCallExpr doesn't traverse the callee so
    /// we need to remove its DeclRefExpr from the DRSet.
    ExprResult TransformCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
      DRSet.erase(cast<DeclRefExpr>(E->getCallee()->IgnoreImplicit()));
      return Base::TransformCXXOperatorCallExpr(E);
    }
    /// Base::TransformUserDefinedLiteral doesn't preserve the
    /// UserDefinedLiteral node.
    ExprResult TransformUserDefinedLiteral(UserDefinedLiteral *E) { return E; }
    /// Base::TransformInitializer skips ConstantExpr so we need to visit them
    /// here.
    ExprResult TransformInitializer(Expr *Init, bool NotCopyInit) {
      if (!Init)
        return Init;

      // We cannot use IgnoreImpCasts because we need to preserve
      // full expressions.
      while (true) {
        if (auto *ICE = dyn_cast<ImplicitCastExpr>(Init))
          Init = ICE->getSubExpr();
        else if (auto *ICE = dyn_cast<MaterializeTemporaryExpr>(Init))
          Init = ICE->getSubExpr();
        else
          break;
      }
      /// ConstantExprs are the first layer of implicit node to be removed so if
      /// Init isn't a ConstantExpr, no ConstantExpr will be skipped.
      if (auto *CE = dyn_cast<ConstantExpr>(Init);
          CE && CE->isImmediateInvocation())
        RemoveImmediateInvocation(CE);
      return Base::TransformInitializer(Init, NotCopyInit);
    }
    ExprResult TransformDeclRefExpr(DeclRefExpr *E) {
      DRSet.erase(E);
      return E;
    }
    ExprResult TransformLambdaExpr(LambdaExpr *E) {
      // Do not rebuild lambdas to avoid creating a new type.
      // Lambdas have already been processed inside their eval contexts.
      return E;
    }
    bool AlwaysRebuild() { return false; }
    bool ReplacingOriginal() { return true; }
    bool AllowSkippingCXXConstructExpr() {
      bool Res = AllowSkippingFirstCXXConstructExpr;
      AllowSkippingFirstCXXConstructExpr = true;
      return Res;
    }
    bool AllowSkippingFirstCXXConstructExpr = true;
  } Transformer(SemaRef, Rec.ReferenceToConsteval,
                Rec.ImmediateInvocationCandidates, It);

  /// CXXConstructExpr with a single argument are getting skipped by
  /// TreeTransform in some situtation because they could be implicit. This
  /// can only occur for the top-level CXXConstructExpr because it is used
  /// nowhere in the expression being transformed therefore will not be rebuilt.
  /// Setting AllowSkippingFirstCXXConstructExpr to false will prevent from
  /// skipping the first CXXConstructExpr.
  if (isa<CXXConstructExpr>(It->getPointer()->IgnoreImplicit()))
    Transformer.AllowSkippingFirstCXXConstructExpr = false;

  ExprResult Res = Transformer.TransformExpr(It->getPointer()->getSubExpr());
  // The result may not be usable in case of previous compilation errors.
  // In this case evaluation of the expression may result in crash so just
  // don't do anything further with the result.
  if (Res.isUsable()) {
    Res = SemaRef.MaybeCreateExprWithCleanups(Res);
    It->getPointer()->setSubExpr(Res.get());
  }
}

static void
HandleImmediateInvocations(Sema &SemaRef,
                           Sema::ExpressionEvaluationContextRecord &Rec) {
  if ((Rec.ImmediateInvocationCandidates.size() == 0 &&
       Rec.ReferenceToConsteval.size() == 0) ||
      Rec.isImmediateFunctionContext() || SemaRef.RebuildingImmediateInvocation)
    return;

  // An expression or conversion is 'manifestly constant-evaluated' if it is:
  // [...]
  // - the initializer of a variable that is usable in constant expressions or
  //   has constant initialization.
  if (SemaRef.getLangOpts().CPlusPlus23 &&
      Rec.ExprContext ==
          Sema::ExpressionEvaluationContextRecord::EK_VariableInit) {
    auto *VD = dyn_cast<VarDecl>(Rec.ManglingContextDecl);
    if (VD && (VD->isUsableInConstantExpressions(SemaRef.Context) ||
               VD->hasConstantInitialization())) {
      // An expression or conversion is in an 'immediate function context' if it
      // is potentially evaluated and either:
      // [...]
      // - it is a subexpression of a manifestly constant-evaluated expression
      //   or conversion.
      return;
    }
  }

  /// When we have more than 1 ImmediateInvocationCandidates or previously
  /// failed immediate invocations, we need to check for nested
  /// ImmediateInvocationCandidates in order to avoid duplicate diagnostics.
  /// Otherwise we only need to remove ReferenceToConsteval in the immediate
  /// invocation.
  if (Rec.ImmediateInvocationCandidates.size() > 1 ||
      !SemaRef.FailedImmediateInvocations.empty()) {

    /// Prevent sema calls during the tree transform from adding pointers that
    /// are already in the sets.
    llvm::SaveAndRestore DisableIITracking(
        SemaRef.RebuildingImmediateInvocation, true);

    /// Prevent diagnostic during tree transfrom as they are duplicates
    Sema::TentativeAnalysisScope DisableDiag(SemaRef);

    for (auto It = Rec.ImmediateInvocationCandidates.rbegin();
         It != Rec.ImmediateInvocationCandidates.rend(); It++)
      if (!It->getInt())
        RemoveNestedImmediateInvocation(SemaRef, Rec, It);
  } else if (Rec.ImmediateInvocationCandidates.size() == 1 &&
             Rec.ReferenceToConsteval.size()) {
    struct SimpleRemove : DynamicRecursiveASTVisitor {
      llvm::SmallPtrSetImpl<DeclRefExpr *> &DRSet;
      SimpleRemove(llvm::SmallPtrSetImpl<DeclRefExpr *> &S) : DRSet(S) {}
      bool VisitDeclRefExpr(DeclRefExpr *E) override {
        DRSet.erase(E);
        return DRSet.size();
      }
    } Visitor(Rec.ReferenceToConsteval);
    Visitor.TraverseStmt(
        Rec.ImmediateInvocationCandidates.front().getPointer()->getSubExpr());
  }
  for (auto CE : Rec.ImmediateInvocationCandidates)
    if (!CE.getInt())
      EvaluateAndDiagnoseImmediateInvocation(SemaRef, CE);
  for (auto *DR : Rec.ReferenceToConsteval) {
    // If the expression is immediate escalating, it is not an error;
    // The outer context itself becomes immediate and further errors,
    // if any, will be handled by DiagnoseImmediateEscalatingReason.
    if (DR->isImmediateEscalating())
      continue;
    auto *FD = cast<FunctionDecl>(DR->getDecl());
    const NamedDecl *ND = FD;
    if (const auto *MD = dyn_cast<CXXMethodDecl>(ND);
        MD && (MD->isLambdaStaticInvoker() || isLambdaCallOperator(MD)))
      ND = MD->getParent();

    // C++23 [expr.const]/p16
    // An expression or conversion is immediate-escalating if it is not
    // initially in an immediate function context and it is [...] a
    // potentially-evaluated id-expression that denotes an immediate function
    // that is not a subexpression of an immediate invocation.
    bool ImmediateEscalating = false;
    bool IsPotentiallyEvaluated =
        Rec.Context ==
            Sema::ExpressionEvaluationContext::PotentiallyEvaluated ||
        Rec.Context ==
            Sema::ExpressionEvaluationContext::PotentiallyEvaluatedIfUsed;
    if (SemaRef.inTemplateInstantiation() && IsPotentiallyEvaluated)
      ImmediateEscalating = Rec.InImmediateEscalatingFunctionContext;

    if (!Rec.InImmediateEscalatingFunctionContext ||
        (SemaRef.inTemplateInstantiation() && !ImmediateEscalating)) {
      SemaRef.Diag(DR->getBeginLoc(), diag::err_invalid_consteval_take_address)
          << ND << isa<CXXRecordDecl>(ND) << FD->isConsteval();
      if (!FD->getBuiltinID())
        SemaRef.Diag(ND->getLocation(), diag::note_declared_at);
      if (auto Context =
              SemaRef.InnermostDeclarationWithDelayedImmediateInvocations()) {
        SemaRef.Diag(Context->Loc, diag::note_invalid_consteval_initializer)
            << Context->Decl;
        SemaRef.Diag(Context->Decl->getBeginLoc(), diag::note_declared_at);
      }
      if (FD->isImmediateEscalating() && !FD->isConsteval())
        SemaRef.DiagnoseImmediateEscalatingReason(FD);

    } else {
      SemaRef.MarkExpressionAsImmediateEscalating(DR);
    }
  }
}

void Sema::PopExpressionEvaluationContext() {
  ExpressionEvaluationContextRecord& Rec = ExprEvalContexts.back();
  if (!Rec.Lambdas.empty()) {
    using ExpressionKind = ExpressionEvaluationContextRecord::ExpressionKind;
    if (!getLangOpts().CPlusPlus20 &&
        (Rec.ExprContext == ExpressionKind::EK_TemplateArgument ||
         Rec.isUnevaluated() ||
         (Rec.isConstantEvaluated() && !getLangOpts().CPlusPlus17))) {
      unsigned D;
      if (Rec.isUnevaluated()) {
        // C++11 [expr.prim.lambda]p2:
        //   A lambda-expression shall not appear in an unevaluated operand
        //   (Clause 5).
        D = diag::err_lambda_unevaluated_operand;
      } else if (Rec.isConstantEvaluated() && !getLangOpts().CPlusPlus17) {
        // C++1y [expr.const]p2:
        //   A conditional-expression e is a core constant expression unless the
        //   evaluation of e, following the rules of the abstract machine, would
        //   evaluate [...] a lambda-expression.
        D = diag::err_lambda_in_constant_expression;
      } else if (Rec.ExprContext == ExpressionKind::EK_TemplateArgument) {
        // C++17 [expr.prim.lamda]p2:
        // A lambda-expression shall not appear [...] in a template-argument.
        D = diag::err_lambda_in_invalid_context;
      } else
        llvm_unreachable("Couldn't infer lambda error message.");

      for (const auto *L : Rec.Lambdas)
        Diag(L->getBeginLoc(), D);
    }
  }

  // Append the collected materialized temporaries into previous context before
  // exit if the previous also is a lifetime extending context.
  if (getLangOpts().CPlusPlus23 && Rec.InLifetimeExtendingContext &&
      parentEvaluationContext().InLifetimeExtendingContext &&
      !Rec.ForRangeLifetimeExtendTemps.empty()) {
    parentEvaluationContext().ForRangeLifetimeExtendTemps.append(
        Rec.ForRangeLifetimeExtendTemps);
  }

  WarnOnPendingNoDerefs(Rec);
  HandleImmediateInvocations(*this, Rec);

  // Warn on any volatile-qualified simple-assignments that are not discarded-
  // value expressions nor unevaluated operands (those cases get removed from
  // this list by CheckUnusedVolatileAssignment).
  for (auto *BO : Rec.VolatileAssignmentLHSs)
    Diag(BO->getBeginLoc(), diag::warn_deprecated_simple_assign_volatile)
        << BO->getType();

  // When are coming out of an unevaluated context, clear out any
  // temporaries that we may have created as part of the evaluation of
  // the expression in that context: they aren't relevant because they
  // will never be constructed.
  if (Rec.isUnevaluated() || Rec.isConstantEvaluated()) {
    ExprCleanupObjects.erase(ExprCleanupObjects.begin() + Rec.NumCleanupObjects,
                             ExprCleanupObjects.end());
    Cleanup = Rec.ParentCleanup;
    CleanupVarDeclMarking();
    std::swap(MaybeODRUseExprs, Rec.SavedMaybeODRUseExprs);
  // Otherwise, merge the contexts together.
  } else {
    Cleanup.mergeFrom(Rec.ParentCleanup);
    MaybeODRUseExprs.insert_range(Rec.SavedMaybeODRUseExprs);
  }

  DiagnoseMisalignedMembers();

  // Pop the current expression evaluation context off the stack.
  ExprEvalContexts.pop_back();
}

void Sema::DiscardCleanupsInEvaluationContext() {
  ExprCleanupObjects.erase(
         ExprCleanupObjects.begin() + ExprEvalContexts.back().NumCleanupObjects,
         ExprCleanupObjects.end());
  Cleanup.reset();
  MaybeODRUseExprs.clear();
}

ExprResult Sema::HandleExprEvaluationContextForTypeof(Expr *E) {
  ExprResult Result = CheckPlaceholderExpr(E);
  if (Result.isInvalid())
    return ExprError();
  E = Result.get();
  if (!E->getType()->isVariablyModifiedType())
    return E;
  return TransformToPotentiallyEvaluated(E);
}

/// Are we in a context that is potentially constant evaluated per C++20
/// [expr.const]p12?
static bool isPotentiallyConstantEvaluatedContext(Sema &SemaRef) {
  /// C++2a [expr.const]p12:
  //   An expression or conversion is potentially constant evaluated if it is
  switch (SemaRef.ExprEvalContexts.back().Context) {
    case Sema::ExpressionEvaluationContext::ConstantEvaluated:
    case Sema::ExpressionEvaluationContext::ImmediateFunctionContext:

      // -- a manifestly constant-evaluated expression,
    case Sema::ExpressionEvaluationContext::PotentiallyEvaluated:
    case Sema::ExpressionEvaluationContext::PotentiallyEvaluatedIfUsed:
    case Sema::ExpressionEvaluationContext::DiscardedStatement:
      // -- a potentially-evaluated expression,
    case Sema::ExpressionEvaluationContext::UnevaluatedList:
      // -- an immediate subexpression of a braced-init-list,

      // -- [FIXME] an expression of the form & cast-expression that occurs
      //    within a templated entity
      // -- a subexpression of one of the above that is not a subexpression of
      // a nested unevaluated operand.
      return true;

    case Sema::ExpressionEvaluationContext::Unevaluated:
    case Sema::ExpressionEvaluationContext::UnevaluatedAbstract:
      // Expressions in this context are never evaluated.
      return false;
  }
  llvm_unreachable("Invalid context");
}

/// Return true if this function has a calling convention that requires mangling
/// in the size of the parameter pack.
static bool funcHasParameterSizeMangling(Sema &S, FunctionDecl *FD) {
  // These manglings are only applicable for targets whcih use Microsoft
  // mangling scheme for C.
  if (!S.Context.getTargetInfo().shouldUseMicrosoftCCforMangling())
    return false;

  // If this is C++ and this isn't an extern "C" function, parameters do not
  // need to be complete. In this case, C++ mangling will apply, which doesn't
  // use the size of the parameters.
  if (S.getLangOpts().CPlusPlus && !FD->isExternC())
    return false;

  // Stdcall, fastcall, and vectorcall need this special treatment.
  CallingConv CC = FD->getType()->castAs<FunctionType>()->getCallConv();
  switch (CC) {
  case CC_X86StdCall:
  case CC_X86FastCall:
  case CC_X86VectorCall:
    return true;
  default:
    break;
  }
  return false;
}

/// Require that all of the parameter types of function be complete. Normally,
/// parameter types are only required to be complete when a function is called
/// or defined, but to mangle functions with certain calling conventions, the
/// mangler needs to know the size of the parameter list. In this situation,
/// MSVC doesn't emit an error or instantiate templates. Instead, MSVC mangles
/// the function as _foo@0, i.e. zero bytes of parameters, which will usually
/// result in a linker error. Clang doesn't implement this behavior, and instead
/// attempts to error at compile time.
static void CheckCompleteParameterTypesForMangler(Sema &S, FunctionDecl *FD,
                                                  SourceLocation Loc) {
  class ParamIncompleteTypeDiagnoser : public Sema::TypeDiagnoser {
    FunctionDecl *FD;
    ParmVarDecl *Param;

  public:
    ParamIncompleteTypeDiagnoser(FunctionDecl *FD, ParmVarDecl *Param)
        : FD(FD), Param(Param) {}

    void diagnose(Sema &S, SourceLocation Loc, QualType T) override {
      CallingConv CC = FD->getType()->castAs<FunctionType>()->getCallConv();
      StringRef CCName;
      switch (CC) {
      case CC_X86StdCall:
        CCName = "stdcall";
        break;
      case CC_X86FastCall:
        CCName = "fastcall";
        break;
      case CC_X86VectorCall:
        CCName = "vectorcall";
        break;
      default:
        llvm_unreachable("CC does not need mangling");
      }

      S.Diag(Loc, diag::err_cconv_incomplete_param_type)
          << Param->getDeclName() << FD->getDeclName() << CCName;
    }
  };

  for (ParmVarDecl *Param : FD->parameters()) {
    ParamIncompleteTypeDiagnoser Diagnoser(FD, Param);
    S.RequireCompleteType(Loc, Param->getType(), Diagnoser);
  }
}

namespace {
enum class OdrUseContext {
  /// Declarations in this context are not odr-used.
  None,
  /// Declarations in this context are formally odr-used, but this is a
  /// dependent context.
  Dependent,
  /// Declarations in this context are odr-used but not actually used (yet).
  FormallyOdrUsed,
  /// Declarations in this context are used.
  Used
};
}

/// Are we within a context in which references to resolved functions or to
/// variables result in odr-use?
static OdrUseContext isOdrUseContext(Sema &SemaRef) {
  const Sema::ExpressionEvaluationContextRecord &Context =
      SemaRef.currentEvaluationContext();

  if (Context.isUnevaluated())
    return OdrUseContext::None;

  if (SemaRef.CurContext->isDependentContext())
    return OdrUseContext::Dependent;

  if (Context.isDiscardedStatementContext())
    return OdrUseContext::FormallyOdrUsed;

  else if (Context.Context ==
           Sema::ExpressionEvaluationContext::PotentiallyEvaluatedIfUsed)
    return OdrUseContext::FormallyOdrUsed;

  return OdrUseContext::Used;
}

static bool isImplicitlyDefinableConstexprFunction(FunctionDecl *Func) {
  if (!Func->isConstexpr())
    return false;

  if (Func->isImplicitlyInstantiable() || !Func->isUserProvided())
    return true;

  // Lambda conversion operators are never user provided.
  if (CXXConversionDecl *Conv = dyn_cast<CXXConversionDecl>(Func))
    return isLambdaConversionOperator(Conv);

  auto *CCD = dyn_cast<CXXConstructorDecl>(Func);
  return CCD && CCD->getInheritedConstructor();
}

void Sema::MarkFunctionReferenced(SourceLocation Loc, FunctionDecl *Func,
                                  bool MightBeOdrUse) {
  assert(Func && "No function?");

  Func->setReferenced();

  // Recursive functions aren't really used until they're used from some other
  // context.
  bool IsRecursiveCall = CurContext == Func;

  // C++11 [basic.def.odr]p3:
  //   A function whose name appears as a potentially-evaluated expression is
  //   odr-used if it is the unique lookup result or the selected member of a
  //   set of overloaded functions [...].
  //
  // We (incorrectly) mark overload resolution as an unevaluated context, so we
  // can just check that here.
  OdrUseContext OdrUse =
      MightBeOdrUse ? isOdrUseContext(*this) : OdrUseContext::None;
  if (IsRecursiveCall && OdrUse == OdrUseContext::Used)
    OdrUse = OdrUseContext::FormallyOdrUsed;

  // Trivial default constructors and destructors are never actually used.
  // FIXME: What about other special members?
  if (Func->isTrivial() && !Func->hasAttr<DLLExportAttr>() &&
      OdrUse == OdrUseContext::Used) {
    if (auto *Constructor = dyn_cast<CXXConstructorDecl>(Func))
      if (Constructor->isDefaultConstructor())
        OdrUse = OdrUseContext::FormallyOdrUsed;
    if (isa<CXXDestructorDecl>(Func))
      OdrUse = OdrUseContext::FormallyOdrUsed;
  }

  // C++20 [expr.const]p12:
  //   A function [...] is needed for constant evaluation if it is [...] a
  //   constexpr function that is named by an expression that is potentially
  //   constant evaluated
  bool NeededForConstantEvaluation =
      isPotentiallyConstantEvaluatedContext(*this) &&
      isImplicitlyDefinableConstexprFunction(Func);

  // Determine whether we require a function definition to exist, per
  // C++11 [temp.inst]p3:
  //   Unless a function template specialization has been explicitly
  //   instantiated or explicitly specialized, the function template
  //   specialization is implicitly instantiated when the specialization is
  //   referenced in a context that requires a function definition to exist.
  // C++20 [temp.inst]p7:
  //   The existence of a definition of a [...] function is considered to
  //   affect the semantics of the program if the [...] function is needed for
  //   constant evaluation by an expression
  // C++20 [basic.def.odr]p10:
  //   Every program shall contain exactly one definition of every non-inline
  //   function or variable that is odr-used in that program outside of a
  //   discarded statement
  // C++20 [special]p1:
  //   The implementation will implicitly define [defaulted special members]
  //   if they are odr-used or needed for constant evaluation.
  //
  // Note that we skip the implicit instantiation of templates that are only
  // used in unused default arguments or by recursive calls to themselves.
  // This is formally non-conforming, but seems reasonable in practice.
  bool NeedDefinition =
      !IsRecursiveCall &&
      (OdrUse == OdrUseContext::Used ||
       (NeededForConstantEvaluation && !Func->isPureVirtual()));

  // C++14 [temp.expl.spec]p6:
  //   If a template [...] is explicitly specialized then that specialization
  //   shall be declared before the first use of that specialization that would
  //   cause an implicit instantiation to take place, in every translation unit
  //   in which such a use occurs
  if (NeedDefinition &&
      (Func->getTemplateSpecializationKind() != TSK_Undeclared ||
       Func->getMemberSpecializationInfo()))
    checkSpecializationReachability(Loc, Func);

  if (getLangOpts().CUDA)
    CUDA().CheckCall(Loc, Func);

  // If we need a definition, try to create one.
  if (NeedDefinition && !Func->getBody()) {
    runWithSufficientStackSpace(Loc, [&] {
      if (CXXConstructorDecl *Constructor =
              dyn_cast<CXXConstructorDecl>(Func)) {
        Constructor = cast<CXXConstructorDecl>(Constructor->getFirstDecl());
        if (Constructor->isDefaulted() && !Constructor->isDeleted()) {
          if (Constructor->isDefaultConstructor()) {
            if (Constructor->isTrivial() &&
                !Constructor->hasAttr<DLLExportAttr>())
              return;
            DefineImplicitDefaultConstructor(Loc, Constructor);
          } else if (Constructor->isCopyConstructor()) {
            DefineImplicitCopyConstructor(Loc, Constructor);
          } else if (Constructor->isMoveConstructor()) {
            DefineImplicitMoveConstructor(Loc, Constructor);
          }
        } else if (Constructor->getInheritedConstructor()) {
          DefineInheritingConstructor(Loc, Constructor);
        }
      } else if (CXXDestructorDecl *Destructor =
                     dyn_cast<CXXDestructorDecl>(Func)) {
        Destructor = cast<CXXDestructorDecl>(Destructor->getFirstDecl());
        if (Destructor->isDefaulted() && !Destructor->isDeleted()) {
          if (Destructor->isTrivial() && !Destructor->hasAttr<DLLExportAttr>())
            return;
          DefineImplicitDestructor(Loc, Destructor);
        }
        if (Destructor->isVirtual() && getLangOpts().AppleKext)
          MarkVTableUsed(Loc, Destructor->getParent());
      } else if (CXXMethodDecl *MethodDecl = dyn_cast<CXXMethodDecl>(Func)) {
        if (MethodDecl->isOverloadedOperator() &&
            MethodDecl->getOverloadedOperator() == OO_Equal) {
          MethodDecl = cast<CXXMethodDecl>(MethodDecl->getFirstDecl());
          if (MethodDecl->isDefaulted() && !MethodDecl->isDeleted()) {
            if (MethodDecl->isCopyAssignmentOperator())
              DefineImplicitCopyAssignment(Loc, MethodDecl);
            else if (MethodDecl->isMoveAssignmentOperator())
              DefineImplicitMoveAssignment(Loc, MethodDecl);
          }
        } else if (isa<CXXConversionDecl>(MethodDecl) &&
                   MethodDecl->getParent()->isLambda()) {
          CXXConversionDecl *Conversion =
              cast<CXXConversionDecl>(MethodDecl->getFirstDecl());
          if (Conversion->isLambdaToBlockPointerConversion())
            DefineImplicitLambdaToBlockPointerConversion(Loc, Conversion);
          else
            DefineImplicitLambdaToFunctionPointerConversion(Loc, Conversion);
        } else if (MethodDecl->isVirtual() && getLangOpts().AppleKext)
          MarkVTableUsed(Loc, MethodDecl->getParent());
      }

      if (Func->isDefaulted() && !Func->isDeleted()) {
        DefaultedComparisonKind DCK = getDefaultedComparisonKind(Func);
        if (DCK != DefaultedComparisonKind::None)
          DefineDefaultedComparison(Loc, Func, DCK);
      }

      // Implicit instantiation of function templates and member functions of
      // class templates.
      if (Func->isImplicitlyInstantiable()) {
        TemplateSpecializationKind TSK =
            Func->getTemplateSpecializationKindForInstantiation();
        SourceLocation PointOfInstantiation = Func->getPointOfInstantiation();
        bool FirstInstantiation = PointOfInstantiation.isInvalid();
        if (FirstInstantiation) {
          PointOfInstantiation = Loc;
          if (auto *MSI = Func->getMemberSpecializationInfo())
            MSI->setPointOfInstantiation(Loc);
            // FIXME: Notify listener.
          else
            Func->setTemplateSpecializationKind(TSK, PointOfInstantiation);
        } else if (TSK != TSK_ImplicitInstantiation) {
          // Use the point of use as the point of instantiation, instead of the
          // point of explicit instantiation (which we track as the actual point
          // of instantiation). This gives better backtraces in diagnostics.
          PointOfInstantiation = Loc;
        }

        if (FirstInstantiation || TSK != TSK_ImplicitInstantiation ||
            Func->isConstexpr()) {
          if (isa<CXXRecordDecl>(Func->getDeclContext()) &&
              cast<CXXRecordDecl>(Func->getDeclContext())->isLocalClass() &&
              CodeSynthesisContexts.size())
            PendingLocalImplicitInstantiations.push_back(
                std::make_pair(Func, PointOfInstantiation));
          else if (Func->isConstexpr())
            // Do not defer instantiations of constexpr functions, to avoid the
            // expression evaluator needing to call back into Sema if it sees a
            // call to such a function.
            InstantiateFunctionDefinition(PointOfInstantiation, Func);
          else {
            Func->setInstantiationIsPending(true);
            PendingInstantiations.push_back(
                std::make_pair(Func, PointOfInstantiation));
            if (llvm::isTimeTraceVerbose()) {
              llvm::timeTraceAddInstantEvent("DeferInstantiation", [&] {
                std::string Name;
                llvm::raw_string_ostream OS(Name);
                Func->getNameForDiagnostic(OS, getPrintingPolicy(),
                                           /*Qualified=*/true);
                return Name;
              });
            }
            // Notify the consumer that a function was implicitly instantiated.
            Consumer.HandleCXXImplicitFunctionInstantiation(Func);
          }
        }
      } else {
        // Walk redefinitions, as some of them may be instantiable.
        for (auto *i : Func->redecls()) {
          if (!i->isUsed(false) && i->isImplicitlyInstantiable())
            MarkFunctionReferenced(Loc, i, MightBeOdrUse);
        }
      }
    });
  }

  // If a constructor was defined in the context of a default parameter
  // or of another default member initializer (ie a PotentiallyEvaluatedIfUsed
  // context), its initializers may not be referenced yet.
  if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(Func)) {
    EnterExpressionEvaluationContext EvalContext(
        *this,
        Constructor->isImmediateFunction()
            ? ExpressionEvaluationContext::ImmediateFunctionContext
            : ExpressionEvaluationContext::PotentiallyEvaluated,
        Constructor);
    for (CXXCtorInitializer *Init : Constructor->inits()) {
      if (Init->isInClassMemberInitializer())
        runWithSufficientStackSpace(Init->getSourceLocation(), [&]() {
          MarkDeclarationsReferencedInExpr(Init->getInit());
        });
    }
  }

  // C++14 [except.spec]p17:
  //   An exception-specification is considered to be needed when:
  //   - the function is odr-used or, if it appears in an unevaluated operand,
  //     would be odr-used if the expression were potentially-evaluated;
  //
  // Note, we do this even if MightBeOdrUse is false. That indicates that the
  // function is a pure virtual function we're calling, and in that case the
  // function was selected by overload resolution and we need to resolve its
  // exception specification for a different reason.
  const FunctionProtoType *FPT = Func->getType()->getAs<FunctionProtoType>();
  if (FPT && isUnresolvedExceptionSpec(FPT->getExceptionSpecType()))
    ResolveExceptionSpec(Loc, FPT);

  // A callee could be called by a host function then by a device function.
  // If we only try recording once, we will miss recording the use on device
  // side. Therefore keep trying until it is recorded.
  if (LangOpts.OffloadImplicitHostDeviceTemplates && LangOpts.CUDAIsDevice &&
      !getASTContext().CUDAImplicitHostDeviceFunUsedByDevice.count(Func))
    CUDA().RecordImplicitHostDeviceFuncUsedByDevice(Func);

  // If this is the first "real" use, act on that.
  if (OdrUse == OdrUseContext::Used && !Func->isUsed(/*CheckUsedAttr=*/false)) {
    // Keep track of used but undefined functions.
    if (!Func->isDefined() && !Func->isInAnotherModuleUnit()) {
      if (mightHaveNonExternalLinkage(Func))
        UndefinedButUsed.insert(std::make_pair(Func->getCanonicalDecl(), Loc));
      else if (Func->getMostRecentDecl()->isInlined() &&
               !LangOpts.GNUInline &&
               !Func->getMostRecentDecl()->hasAttr<GNUInlineAttr>())
        UndefinedButUsed.insert(std::make_pair(Func->getCanonicalDecl(), Loc));
      else if (isExternalWithNoLinkageType(Func))
        UndefinedButUsed.insert(std::make_pair(Func->getCanonicalDecl(), Loc));
    }

    // Some x86 Windows calling conventions mangle the size of the parameter
    // pack into the name. Computing the size of the parameters requires the
    // parameter types to be complete. Check that now.
    if (funcHasParameterSizeMangling(*this, Func))
      CheckCompleteParameterTypesForMangler(*this, Func, Loc);

    // In the MS C++ ABI, the compiler emits destructor variants where they are
    // used. If the destructor is used here but defined elsewhere, mark the
    // virtual base destructors referenced. If those virtual base destructors
    // are inline, this will ensure they are defined when emitting the complete
    // destructor variant. This checking may be redundant if the destructor is
    // provided later in this TU.
    if (Context.getTargetInfo().getCXXABI().isMicrosoft()) {
      if (auto *Dtor = dyn_cast<CXXDestructorDecl>(Func)) {
        CXXRecordDecl *Parent = Dtor->getParent();
        if (Parent->getNumVBases() > 0 && !Dtor->getBody())
          CheckCompleteDestructorVariant(Loc, Dtor);
      }
    }

    Func->markUsed(Context);
  }
}

/// Directly mark a variable odr-used. Given a choice, prefer to use
/// MarkVariableReferenced since it does additional checks and then
/// calls MarkVarDeclODRUsed.
/// If the variable must be captured:
///  - if FunctionScopeIndexToStopAt is null, capture it in the CurContext
///  - else capture it in the DeclContext that maps to the
///    *FunctionScopeIndexToStopAt on the FunctionScopeInfo stack.
static void
MarkVarDeclODRUsed(ValueDecl *V, SourceLocation Loc, Sema &SemaRef,
                   const unsigned *const FunctionScopeIndexToStopAt = nullptr) {
  // Keep track of used but undefined variables.
  // FIXME: We shouldn't suppress this warning for static data members.
  VarDecl *Var = V->getPotentiallyDecomposedVarDecl();
  assert(Var && "expected a capturable variable");

  if (Var->hasDefinition(SemaRef.Context) == VarDecl::DeclarationOnly &&
      (!Var->isExternallyVisible() || Var->isInline() ||
       SemaRef.isExternalWithNoLinkageType(Var)) &&
      !(Var->isStaticDataMember() && Var->hasInit())) {
    SourceLocation &old = SemaRef.UndefinedButUsed[Var->getCanonicalDecl()];
    if (old.isInvalid())
      old = Loc;
  }
  QualType CaptureType, DeclRefType;
  if (SemaRef.LangOpts.OpenMP)
    SemaRef.OpenMP().tryCaptureOpenMPLambdas(V);
  SemaRef.tryCaptureVariable(V, Loc, TryCaptureKind::Implicit,
                             /*EllipsisLoc*/ SourceLocation(),
                             /*BuildAndDiagnose*/ true, CaptureType,
                             DeclRefType, FunctionScopeIndexToStopAt);

  if (SemaRef.LangOpts.CUDA && Var->hasGlobalStorage()) {
    auto *FD = dyn_cast_or_null<FunctionDecl>(SemaRef.CurContext);
    auto VarTarget = SemaRef.CUDA().IdentifyTarget(Var);
    auto UserTarget = SemaRef.CUDA().IdentifyTarget(FD);
    if (VarTarget == SemaCUDA::CVT_Host &&
        (UserTarget == CUDAFunctionTarget::Device ||
         UserTarget == CUDAFunctionTarget::HostDevice ||
         UserTarget == CUDAFunctionTarget::Global)) {
      // Diagnose ODR-use of host global variables in device functions.
      // Reference of device global variables in host functions is allowed
      // through shadow variables therefore it is not diagnosed.
      if (SemaRef.LangOpts.CUDAIsDevice && !SemaRef.LangOpts.HIPStdPar) {
        SemaRef.targetDiag(Loc, diag::err_ref_bad_target)
            << /*host*/ 2 << /*variable*/ 1 << Var << UserTarget;
        SemaRef.targetDiag(Var->getLocation(),
                           Var->getType().isConstQualified()
                               ? diag::note_cuda_const_var_unpromoted
                               : diag::note_cuda_host_var);
      }
    } else if ((VarTarget == SemaCUDA::CVT_Device ||
                // Also capture __device__ const variables, which are classified
                // as CVT_Both due to an implicit CUDAConstantAttr. We check for
                // an explicit CUDADeviceAttr to distinguish them from plain
                // const variables (no __device__), which also get CVT_Both but
                // only have an implicit CUDADeviceAttr.
                (VarTarget == SemaCUDA::CVT_Both &&
                 Var->hasAttr<CUDADeviceAttr>() &&
                 !Var->getAttr<CUDADeviceAttr>()->isImplicit())) &&
               !Var->hasAttr<CUDASharedAttr>() &&
               (UserTarget == CUDAFunctionTarget::Host ||
                UserTarget == CUDAFunctionTarget::HostDevice)) {
      // Record a CUDA/HIP device side variable if it is ODR-used
      // by host code. This is done conservatively, when the variable is
      // referenced in any of the following contexts:
      //   - a non-function context
      //   - a host function
      //   - a host device function
      // This makes the ODR-use of the device side variable by host code to
      // be visible in the device compilation for the compiler to be able to
      // emit template variables instantiated by host code only and to
      // externalize the static device side variable ODR-used by host code.
      if (!Var->hasExternalStorage())
        SemaRef.getASTContext().CUDADeviceVarODRUsedByHost.insert(Var);
      else if (SemaRef.LangOpts.GPURelocatableDeviceCode &&
               (!FD || (!FD->getDescribedFunctionTemplate() &&
                        SemaRef.getASTContext().GetGVALinkageForFunction(FD) ==
                            GVA_StrongExternal)))
        SemaRef.getASTContext().CUDAExternalDeviceDeclODRUsedByHost.insert(Var);
    }
  }

  V->markUsed(SemaRef.Context);
}

void Sema::MarkCaptureUsedInEnclosingContext(ValueDecl *Capture,
                                             SourceLocation Loc,
                                             unsigned CapturingScopeIndex) {
  MarkVarDeclODRUsed(Capture, Loc, *this, &CapturingScopeIndex);
}

static void diagnoseUncapturableValueReferenceOrBinding(Sema &S,
                                                        SourceLocation loc,
                                                        ValueDecl *var) {
  DeclContext *VarDC = var->getDeclContext();

  //  If the parameter still belongs to the translation unit, then
  //  we're actually just using one parameter in the declaration of
  //  the next.
  if (isa<ParmVarDecl>(var) &&
      isa<TranslationUnitDecl>(VarDC))
    return;

  // For C code, don't diagnose about capture if we're not actually in code
  // right now; it's impossible to write a non-constant expression outside of
  // function context, so we'll get other (more useful) diagnostics later.
  //
  // For C++, things get a bit more nasty... it would be nice to suppress this
  // diagnostic for certain cases like using a local variable in an array bound
  // for a member of a local class, but the correct predicate is not obvious.
  if (!S.getLangOpts().CPlusPlus && !S.CurContext->isFunctionOrMethod())
    return;

  unsigned ValueKind = isa<BindingDecl>(var) ? 1 : 0;
  unsigned ContextKind = 3; // unknown
  if (isa<CXXMethodDecl>(VarDC) &&
      cast<CXXRecordDecl>(VarDC->getParent())->isLambda()) {
    ContextKind = 2;
  } else if (isa<FunctionDecl>(VarDC)) {
    ContextKind = 0;
  } else if (isa<BlockDecl>(VarDC)) {
    ContextKind = 1;
  }

  S.Diag(loc, diag::err_reference_to_local_in_enclosing_context)
    << var << ValueKind << ContextKind << VarDC;
  S.Diag(var->getLocation(), diag::note_entity_declared_at)
      << var;

  // FIXME: Add additional diagnostic info about class etc. which prevents
  // capture.
}

static bool isVariableAlreadyCapturedInScopeInfo(CapturingScopeInfo *CSI,
                                                 ValueDecl *Var,
                                                 bool &SubCapturesAreNested,
                                                 QualType &CaptureType,
                                                 QualType &DeclRefType) {
  // Check whether we've already captured it.
  if (CSI->CaptureMap.count(Var)) {
    // If we found a capture, any subcaptures are nested.
    SubCapturesAreNested = true;

    // Retrieve the capture type for this variable.
    CaptureType = CSI->getCapture(Var).getCaptureType();

    // Compute the type of an expression that refers to this variable.
    DeclRefType = CaptureType.getNonReferenceType();

    // Similarly to mutable captures in lambda, all the OpenMP captures by copy
    // are mutable in the sense that user can change their value - they are
    // private instances of the captured declarations.
    const Capture &Cap = CSI->getCapture(Var);
    // C++ [expr.prim.lambda]p10:
    //   The type of such a data member is [...] an lvalue reference to the
    //   referenced function type if the entity is a reference to a function.
    //   [...]
    if (Cap.isCopyCapture() && !DeclRefType->isFunctionType() &&
        !(isa<LambdaScopeInfo>(CSI) &&
          !cast<LambdaScopeInfo>(CSI)->lambdaCaptureShouldBeConst()) &&
        !(isa<CapturedRegionScopeInfo>(CSI) &&
          cast<CapturedRegionScopeInfo>(CSI)->CapRegionKind == CR_OpenMP))
      DeclRefType.addConst();
    return true;
  }
  return false;
}

// Only block literals, captured statements, and lambda expressions can
// capture; other scopes don't work.
static DeclContext *getParentOfCapturingContextOrNull(DeclContext *DC,
                                                      ValueDecl *Var,
                                                      SourceLocation Loc,
                                                      const bool Diagnose,
                                                      Sema &S) {
  if (isa<BlockDecl>(DC) || isa<CapturedDecl>(DC) || isLambdaCallOperator(DC))
    return getLambdaAwareParentOfDeclContext(DC);

  VarDecl *Underlying = Var->getPotentiallyDecomposedVarDecl();
  if (Underlying) {
    if (Underlying->hasLocalStorage() && Diagnose)
      diagnoseUncapturableValueReferenceOrBinding(S, Loc, Var);
  }
  return nullptr;
}

// Certain capturing entities (lambdas, blocks etc.) are not allowed to capture
// certain types of variables (unnamed, variably modified types etc.)
// so check for eligibility.
static bool isVariableCapturable(CapturingScopeInfo *CSI, ValueDecl *Var,
                                 SourceLocation Loc, const bool Diagnose,
                                 Sema &S) {

  assert((isa<VarDecl, BindingDecl>(Var)) &&
         "Only variables and structured bindings can be captured");

  bool IsBlock = isa<BlockScopeInfo>(CSI);
  bool IsLambda = isa<LambdaScopeInfo>(CSI);

  // Lambdas are not allowed to capture unnamed variables
  // (e.g. anonymous unions).
  // FIXME: The C++11 rule don't actually state this explicitly, but I'm
  // assuming that's the intent.
  if (IsLambda && !Var->getDeclName()) {
    if (Diagnose) {
      S.Diag(Loc, diag::err_lambda_capture_anonymous_var);
      S.Diag(Var->getLocation(), diag::note_declared_at);
    }
    return false;
  }

  // Prohibit variably-modified types in blocks; they're difficult to deal with.
  if (Var->getType()->isVariablyModifiedType() && IsBlock) {
    if (Diagnose) {
      S.Diag(Loc, diag::err_ref_vm_type);
      S.Diag(Var->getLocation(), diag::note_previous_decl) << Var;
    }
    return false;
  }
  // Prohibit structs with flexible array members too.
  // We cannot capture what is in the tail end of the struct.
  if (const auto *VTD = Var->getType()->getAsRecordDecl();
      VTD && VTD->hasFlexibleArrayMember()) {
    if (Diagnose) {
      if (IsBlock)
        S.Diag(Loc, diag::err_ref_flexarray_type);
      else
        S.Diag(Loc, diag::err_lambda_capture_flexarray_type) << Var;
      S.Diag(Var->getLocation(), diag::note_previous_decl) << Var;
    }
    return false;
  }
  const bool HasBlocksAttr = Var->hasAttr<BlocksAttr>();
  // Lambdas and captured statements are not allowed to capture __block
  // variables; they don't support the expected semantics.
  if (HasBlocksAttr && (IsLambda || isa<CapturedRegionScopeInfo>(CSI))) {
    if (Diagnose) {
      S.Diag(Loc, diag::err_capture_block_variable) << Var << !IsLambda;
      S.Diag(Var->getLocation(), diag::note_previous_decl) << Var;
    }
    return false;
  }
  // OpenCL v2.0 s6.12.5: Blocks cannot reference/capture other blocks
  if (S.getLangOpts().OpenCL && IsBlock &&
      Var->getType()->isBlockPointerType()) {
    if (Diagnose)
      S.Diag(Loc, diag::err_opencl_block_ref_block);
    return false;
  }

  if (isa<BindingDecl>(Var)) {
    if (!IsLambda || !S.getLangOpts().CPlusPlus) {
      if (Diagnose)
        diagnoseUncapturableValueReferenceOrBinding(S, Loc, Var);
      return false;
    } else if (Diagnose && S.getLangOpts().CPlusPlus) {
      S.Diag(Loc, S.LangOpts.CPlusPlus20
                      ? diag::warn_cxx17_compat_capture_binding
                      : diag::ext_capture_binding)
          << Var;
      S.Diag(Var->getLocation(), diag::note_entity_declared_at) << Var;
    }
  }

  return true;
}

// Returns true if the capture by block was successful.
static bool captureInBlock(BlockScopeInfo *BSI, ValueDecl *Var,
                           SourceLocation Loc, const bool BuildAndDiagnose,
                           QualType &CaptureType, QualType &DeclRefType,
                           const bool Nested, Sema &S, bool Invalid) {
  bool ByRef = false;

  // Blocks are not allowed to capture arrays, excepting OpenCL.
  // OpenCL v2.0 s1.12.5 (revision 40): arrays are captured by reference
  // (decayed to pointers).
  if (!Invalid && !S.getLangOpts().OpenCL && CaptureType->isArrayType()) {
    if (BuildAndDiagnose) {
      S.Diag(Loc, diag::err_ref_array_type);
      S.Diag(Var->getLocation(), diag::note_previous_decl) << Var;
      Invalid = true;
    } else {
      return false;
    }
  }

  // Forbid the block-capture of autoreleasing variables.
  if (!Invalid &&
      CaptureType.getObjCLifetime() == Qualifiers::OCL_Autoreleasing) {
    if (BuildAndDiagnose) {
      S.Diag(Loc, diag::err_arc_autoreleasing_capture)
        << /*block*/ 0;
      S.Diag(Var->getLocation(), diag::note_previous_decl) << Var;
      Invalid = true;
    } else {
      return false;
    }
  }

  // Warn about implicitly autoreleasing indirect parameters captured by blocks.
  if (const auto *PT = CaptureType->getAs<PointerType>()) {
    QualType PointeeTy = PT->getPointeeType();

    if (!Invalid && PointeeTy->getAs<ObjCObjectPointerType>() &&
        PointeeTy.getObjCLifetime() == Qualifiers::OCL_Autoreleasing &&
        !S.Context.hasDirectOwnershipQualifier(PointeeTy)) {
      if (BuildAndDiagnose) {
        SourceLocation VarLoc = Var->getLocation();
        S.Diag(Loc, diag::warn_block_capture_autoreleasing);
        S.Diag(VarLoc, diag::note_declare_parameter_strong);
      }
    }
  }

  const bool HasBlocksAttr = Var->hasAttr<BlocksAttr>();
  if (HasBlocksAttr || CaptureType->isReferenceType() ||
      (S.getLangOpts().OpenMP && S.OpenMP().isOpenMPCapturedDecl(Var))) {
    // Block capture by reference does not change the capture or
    // declaration reference types.
    ByRef = true;
  } else {
    // Block capture by copy introduces 'const'.
    CaptureType = CaptureType.getNonReferenceType().withConst();
    DeclRefType = CaptureType;
  }

  // Actually capture the variable.
  if (BuildAndDiagnose)
    BSI->addCapture(Var, HasBlocksAttr, ByRef, Nested, Loc, SourceLocation(),
                    CaptureType, Invalid);

  return !Invalid;
}

/// Capture the given variable in the captured region.
static bool captureInCapturedRegion(
    CapturedRegionScopeInfo *RSI, ValueDecl *Var, SourceLocation Loc,
    const bool BuildAndDiagnose, QualType &CaptureType, QualType &DeclRefType,
    const bool RefersToCapturedVariable, TryCaptureKind Kind, bool IsTopScope,
    Sema &S, bool Invalid) {
  // By default, capture variables by reference.
  bool ByRef = true;
  if (IsTopScope && Kind != TryCaptureKind::Implicit) {
    ByRef = (Kind == TryCaptureKind::ExplicitByRef);
  } else if (S.getLangOpts().OpenMP && RSI->CapRegionKind == CR_OpenMP) {
    // Using an LValue reference type is consistent with Lambdas (see below).
    if (S.OpenMP().isOpenMPCapturedDecl(Var)) {
      bool HasConst = DeclRefType.isConstQualified();
      DeclRefType = DeclRefType.getUnqualifiedType();
      // Don't lose diagnostics about assignments to const.
      if (HasConst)
        DeclRefType.addConst();
    }
    // Do not capture firstprivates in tasks.
    if (S.OpenMP().isOpenMPPrivateDecl(Var, RSI->OpenMPLevel,
                                       RSI->OpenMPCaptureLevel) != OMPC_unknown)
      return true;
    ByRef = S.OpenMP().isOpenMPCapturedByRef(Var, RSI->OpenMPLevel,
                                             RSI->OpenMPCaptureLevel);
  }

  if (ByRef)
    CaptureType = S.Context.getLValueReferenceType(DeclRefType);
  else
    CaptureType = DeclRefType;

  // Actually capture the variable.
  if (BuildAndDiagnose)
    RSI->addCapture(Var, /*isBlock*/ false, ByRef, RefersToCapturedVariable,
                    Loc, SourceLocation(), CaptureType, Invalid);

  return !Invalid;
}

/// Capture the given variable in the lambda.
static bool captureInLambda(LambdaScopeInfo *LSI, ValueDecl *Var,
                            SourceLocation Loc, const bool BuildAndDiagnose,
                            QualType &CaptureType, QualType &DeclRefType,
                            const bool RefersToCapturedVariable,
                            const TryCaptureKind Kind,
                            SourceLocation EllipsisLoc, const bool IsTopScope,
                            Sema &S, bool Invalid) {
  // Determine whether we are capturing by reference or by value.
  bool ByRef = false;
  if (IsTopScope && Kind != TryCaptureKind::Implicit) {
    ByRef = (Kind == TryCaptureKind::ExplicitByRef);
  } else {
    ByRef = (LSI->ImpCaptureStyle == LambdaScopeInfo::ImpCap_LambdaByref);
  }

  if (BuildAndDiagnose && S.Context.getTargetInfo().getTriple().isWasm() &&
      CaptureType.getNonReferenceType().isWebAssemblyReferenceType()) {
    S.Diag(Loc, diag::err_wasm_ca_reference) << 0;
    Invalid = true;
  }

  // Compute the type of the field that will capture this variable.
  if (ByRef) {
    // C++11 [expr.prim.lambda]p15:
    //   An entity is captured by reference if it is implicitly or
    //   explicitly captured but not captured by copy. It is
    //   unspecified whether additional unnamed non-static data
    //   members are declared in the closure type for entities
    //   captured by reference.
    //
    // FIXME: It is not clear whether we want to build an lvalue reference
    // to the DeclRefType or to CaptureType.getNonReferenceType(). GCC appears
    // to do the former, while EDG does the latter. Core issue 1249 will
    // clarify, but for now we follow GCC because it's a more permissive and
    // easily defensible position.
    CaptureType = S.Context.getLValueReferenceType(DeclRefType);
  } else {
    // C++11 [expr.prim.lambda]p14:
    //   For each entity captured by copy, an unnamed non-static
    //   data member is declared in the closure type. The
    //   declaration order of these members is unspecified. The type
    //   of such a data member is the type of the corresponding
    //   captured entity if the entity is not a reference to an
    //   object, or the referenced type otherwise. [Note: If the
    //   captured entity is a reference to a function, the
    //   corresponding data member is also a reference to a
    //   function. - end note ]
    if (const ReferenceType *RefType = CaptureType->getAs<ReferenceType>()){
      if (!RefType->getPointeeType()->isFunctionType())
        CaptureType = RefType->getPointeeType();
    }

    // Forbid the lambda copy-capture of autoreleasing variables.
    if (!Invalid &&
        CaptureType.getObjCLifetime() == Qualifiers::OCL_Autoreleasing) {
      if (BuildAndDiagnose) {
        S.Diag(Loc, diag::err_arc_autoreleasing_capture) << /*lambda*/ 1;
        S.Diag(Var->getLocation(), diag::note_previous_decl)
          << Var->getDeclName();
        Invalid = true;
      } else {
        return false;
      }
    }

    // Make sure that by-copy captures are of a complete and non-abstract type.
    if (!Invalid && BuildAndDiagnose) {
      if (!CaptureType->isDependentType() &&
          S.RequireCompleteSizedType(
              Loc, CaptureType,
              diag::err_capture_of_incomplete_or_sizeless_type,
              Var->getDeclName()))
        Invalid = true;
      else if (S.RequireNonAbstractType(Loc, CaptureType,
                                        diag::err_capture_of_abstract_type))
        Invalid = true;
    }
  }

  // Compute the type of a reference to this captured variable.
  if (ByRef)
    DeclRefType = CaptureType.getNonReferenceType();
  else {
    // C++ [expr.prim.lambda]p5:
    //   The closure type for a lambda-expression has a public inline
    //   function call operator [...]. This function call operator is
    //   declared const (9.3.1) if and only if the lambda-expression's
    //   parameter-declaration-clause is not followed by mutable.
    DeclRefType = CaptureType.getNonReferenceType();
    bool Const = LSI->lambdaCaptureShouldBeConst();
    // C++ [expr.prim.lambda]p10:
    //   The type of such a data member is [...] an lvalue reference to the
    //   referenced function type if the entity is a reference to a function.
    //   [...]
    if (Const && !CaptureType->isReferenceType() &&
        !DeclRefType->isFunctionType())
      DeclRefType.addConst();
  }

  // Add the capture.
  if (BuildAndDiagnose)
    LSI->addCapture(Var, /*isBlock=*/false, ByRef, RefersToCapturedVariable,
                    Loc, EllipsisLoc, CaptureType, Invalid);

  return !Invalid;
}

static bool canCaptureVariableByCopy(ValueDecl *Var,
                                     const ASTContext &Context) {
  // Offer a Copy fix even if the type is dependent.
  if (Var->getType()->isDependentType())
    return true;
  QualType T = Var->getType().getNonReferenceType();
  if (T.isTriviallyCopyableType(Context))
    return true;
  if (CXXRecordDecl *RD = T->getAsCXXRecordDecl()) {

    if (!(RD = RD->getDefinition()))
      return false;
    if (RD->hasSimpleCopyConstructor())
      return true;
    if (RD->hasUserDeclaredCopyConstructor())
      for (CXXConstructorDecl *Ctor : RD->ctors())
        if (Ctor->isCopyConstructor())
          return !Ctor->isDeleted();
  }
  return false;
}

/// Create up to 4 fix-its for explicit reference and value capture of \p Var or
/// default capture. Fixes may be omitted if they aren't allowed by the
/// standard, for example we can't emit a default copy capture fix-it if we
/// already explicitly copy capture capture another variable.
static void buildLambdaCaptureFixit(Sema &Sema, LambdaScopeInfo *LSI,
                                    ValueDecl *Var) {
  assert(LSI->ImpCaptureStyle == CapturingScopeInfo::ImpCap_None);
  // Don't offer Capture by copy of default capture by copy fixes if Var is
  // known not to be copy constructible.
  bool ShouldOfferCopyFix = canCaptureVariableByCopy(Var, Sema.getASTContext());

  SmallString<32> FixBuffer;
  StringRef Separator = LSI->NumExplicitCaptures > 0 ? ", " : "";
  if (Var->getDeclName().isIdentifier() && !Var->getName().empty()) {
    SourceLocation VarInsertLoc = LSI->IntroducerRange.getEnd();
    if (ShouldOfferCopyFix) {
      // Offer fixes to insert an explicit capture for the variable.
      // [] -> [VarName]
      // [OtherCapture] -> [OtherCapture, VarName]
      FixBuffer.assign({Separator, Var->getName()});
      Sema.Diag(VarInsertLoc, diag::note_lambda_variable_capture_fixit)
          << Var << /*value*/ 0
          << FixItHint::CreateInsertion(VarInsertLoc, FixBuffer);
    }
    // As above but capture by reference.
    FixBuffer.assign({Separator, "&", Var->getName()});
    Sema.Diag(VarInsertLoc, diag::note_lambda_variable_capture_fixit)
        << Var << /*reference*/ 1
        << FixItHint::CreateInsertion(VarInsertLoc, FixBuffer);
  }

  // Only try to offer default capture if there are no captures excluding this
  // and init captures.
  // [this]: OK.
  // [X = Y]: OK.
  // [&A, &B]: Don't offer.
  // [A, B]: Don't offer.
  if (llvm::any_of(LSI->Captures, [](Capture &C) {
        return !C.isThisCapture() && !C.isInitCapture();
      }))
    return;

  // The default capture specifiers, '=' or '&', must appear first in the
  // capture body.
  SourceLocation DefaultInsertLoc =
      LSI->IntroducerRange.getBegin().getLocWithOffset(1);

  if (ShouldOfferCopyFix) {
    bool CanDefaultCopyCapture = true;
    // [=, *this] OK since c++17
    // [=, this] OK since c++20
    if (LSI->isCXXThisCaptured() && !Sema.getLangOpts().CPlusPlus20)
      CanDefaultCopyCapture = Sema.getLangOpts().CPlusPlus17
                                  ? LSI->getCXXThisCapture().isCopyCapture()
                                  : false;
    // We can't use default capture by copy if any captures already specified
    // capture by copy.
    if (CanDefaultCopyCapture && llvm::none_of(LSI->Captures, [](Capture &C) {
          return !C.isThisCapture() && !C.isInitCapture() && C.isCopyCapture();
        })) {
      FixBuffer.assign({"=", Separator});
      Sema.Diag(DefaultInsertLoc, diag::note_lambda_default_capture_fixit)
          << /*value*/ 0
          << FixItHint::CreateInsertion(DefaultInsertLoc, FixBuffer);
    }
  }

  // We can't use default capture by reference if any captures already specified
  // capture by reference.
  if (llvm::none_of(LSI->Captures, [](Capture &C) {
        return !C.isInitCapture() && C.isReferenceCapture() &&
               !C.isThisCapture();
      })) {
    FixBuffer.assign({"&", Separator});
    Sema.Diag(DefaultInsertLoc, diag::note_lambda_default_capture_fixit)
        << /*reference*/ 1
        << FixItHint::CreateInsertion(DefaultInsertLoc, FixBuffer);
  }
}

bool Sema::tryCaptureVariable(
    ValueDecl *Var, SourceLocation ExprLoc, TryCaptureKind Kind,
    SourceLocation EllipsisLoc, bool BuildAndDiagnose, QualType &CaptureType,
    QualType &DeclRefType, const unsigned *const FunctionScopeIndexToStopAt) {
  // An init-capture is notionally from the context surrounding its
  // declaration, but its parent DC is the lambda class.
  DeclContext *VarDC = Var->getDeclContext();
  DeclContext *DC = CurContext;

  // Skip past RequiresExprBodys because they don't constitute function scopes.
  while (DC->isRequiresExprBody())
    DC = DC->getParent();

  // tryCaptureVariable is called every time a DeclRef is formed,
  // it can therefore have non-negigible impact on performances.
  // For local variables and when there is no capturing scope,
  // we can bailout early.
  if (CapturingFunctionScopes == 0 && (!BuildAndDiagnose || VarDC == DC))
    return true;

  // Exception: Function parameters are not tied to the function's DeclContext
  // until we enter the function definition. Capturing them anyway would result
  // in an out-of-bounds error while traversing DC and its parents.
  if (isa<ParmVarDecl>(Var) && !VarDC->isFunctionOrMethod())
    return true;

  const auto *VD = dyn_cast<VarDecl>(Var);
  if (VD) {
    if (VD->isInitCapture())
      VarDC = VarDC->getParent();
  } else {
    VD = Var->getPotentiallyDecomposedVarDecl();
  }
  assert(VD && "Cannot capture a null variable");

  const unsigned MaxFunctionScopesIndex = FunctionScopeIndexToStopAt
      ? *FunctionScopeIndexToStopAt : FunctionScopes.size() - 1;
  // We need to sync up the Declaration Context with the
  // FunctionScopeIndexToStopAt
  if (FunctionScopeIndexToStopAt) {
    assert(!FunctionScopes.empty() && "No function scopes to stop at?");
    unsigned FSIndex = FunctionScopes.size() - 1;
    // When we're parsing the lambda parameter list, the current DeclContext is
    // NOT the lambda but its parent. So move away the current LSI before
    // aligning DC and FunctionScopeIndexToStopAt.
    if (auto *LSI = dyn_cast<LambdaScopeInfo>(FunctionScopes[FSIndex]);
        FSIndex && LSI && !LSI->AfterParameterList)
      --FSIndex;
    assert(MaxFunctionScopesIndex <= FSIndex &&
           "FunctionScopeIndexToStopAt should be no greater than FSIndex into "
           "FunctionScopes.");
    while (FSIndex != MaxFunctionScopesIndex) {
      DC = getLambdaAwareParentOfDeclContext(DC);
      --FSIndex;
    }
  }

  // Capture global variables if it is required to use private copy of this
  // variable.
  bool IsGlobal = !VD->hasLocalStorage();
  if (IsGlobal && !(LangOpts.OpenMP &&
                    OpenMP().isOpenMPCapturedDecl(Var, /*CheckScopeInfo=*/true,
                                                  MaxFunctionScopesIndex)))
    return true;

  if (isa<VarDecl>(Var))
    Var = cast<VarDecl>(Var->getCanonicalDecl());

  // Walk up the stack to determine whether we can capture the variable,
  // performing the "simple" checks that don't depend on type. We stop when
  // we've either hit the declared scope of the variable or find an existing
  // capture of that variable.  We start from the innermost capturing-entity
  // (the DC) and ensure that all intervening capturing-entities
  // (blocks/lambdas etc.) between the innermost capturer and the variable`s
  // declcontext can either capture the variable or have already captured
  // the variable.
  CaptureType = Var->getType();
  DeclRefType = CaptureType.getNonReferenceType();
  bool Nested = false;
  bool Explicit = (Kind != TryCaptureKind::Implicit);
  unsigned FunctionScopesIndex = MaxFunctionScopesIndex;
  do {

    LambdaScopeInfo *LSI = nullptr;
    if (!FunctionScopes.empty())
      LSI = dyn_cast_or_null<LambdaScopeInfo>(
          FunctionScopes[FunctionScopesIndex]);

    bool IsInScopeDeclarationContext =
        !LSI || LSI->AfterParameterList || CurContext == LSI->CallOperator;

    if (LSI && !LSI->AfterParameterList) {
      // This allows capturing parameters from a default value which does not
      // seems correct
      if (isa<ParmVarDecl>(Var) && !Var->getDeclContext()->isFunctionOrMethod())
        return true;
    }
    // If the variable is declared in the current context, there is no need to
    // capture it.
    if (IsInScopeDeclarationContext &&
        FunctionScopesIndex == MaxFunctionScopesIndex && VarDC == DC)
      return true;

    // Only block literals, captured statements, and lambda expressions can
    // capture; other scopes don't work.
    DeclContext *ParentDC =
        !IsInScopeDeclarationContext
            ? DC->getParent()
            : getParentOfCapturingContextOrNull(DC, Var, ExprLoc,
                                                BuildAndDiagnose, *this);
    // We need to check for the parent *first* because, if we *have*
    // private-captured a global variable, we need to recursively capture it in
    // intermediate blocks, lambdas, etc.
    if (!ParentDC) {
      if (IsGlobal) {
        FunctionScopesIndex = MaxFunctionScopesIndex - 1;
        break;
      }
      return true;
    }

    FunctionScopeInfo  *FSI = FunctionScopes[FunctionScopesIndex];
    CapturingScopeInfo *CSI = cast<CapturingScopeInfo>(FSI);

    // Check whether we've already captured it.
    if (isVariableAlreadyCapturedInScopeInfo(CSI, Var, Nested, CaptureType,
                                             DeclRefType)) {
      CSI->getCapture(Var).markUsed(BuildAndDiagnose);
      break;
    }

    // When evaluating some attributes (like enable_if) we might refer to a
    // function parameter appertaining to the same declaration as that
    // attribute.
    if (const auto *Parm = dyn_cast<ParmVarDecl>(Var);
        Parm && Parm->getDeclContext() == DC)
      return true;

    // If we are instantiating a generic lambda call operator body,
    // we do not want to capture new variables.  What was captured
    // during either a lambdas transformation or initial parsing
    // should be used.
    if (isGenericLambdaCallOperatorSpecialization(DC)) {
      if (BuildAndDiagnose) {
        LambdaScopeInfo *LSI = cast<LambdaScopeInfo>(CSI);
        if (LSI->ImpCaptureStyle == CapturingScopeInfo::ImpCap_None) {
          Diag(ExprLoc, diag::err_lambda_impcap) << Var;
          Diag(Var->getLocation(), diag::note_previous_decl) << Var;
          Diag(LSI->Lambda->getBeginLoc(), diag::note_lambda_decl);
          buildLambdaCaptureFixit(*this, LSI, Var);
        } else
          diagnoseUncapturableValueReferenceOrBinding(*this, ExprLoc, Var);
      }
      return true;
    }

    // Try to capture variable-length arrays types.
    if (Var->getType()->isVariablyModifiedType()) {
      // We're going to walk down into the type and look for VLA
      // expressions.
      QualType QTy = Var->getType();
      if (ParmVarDecl *PVD = dyn_cast_or_null<ParmVarDecl>(Var))
        QTy = PVD->getOriginalType();
      captureVariablyModifiedType(Context, QTy, CSI);
    }

    if (getLangOpts().OpenMP) {
      if (auto *RSI = dyn_cast<CapturedRegionScopeInfo>(CSI)) {
        // OpenMP private variables should not be captured in outer scope, so
        // just break here. Similarly, global variables that are captured in a
        // target region should not be captured outside the scope of the region.
        if (RSI->CapRegionKind == CR_OpenMP) {
          // FIXME: We should support capturing structured bindings in OpenMP.
          if (isa<BindingDecl>(Var)) {
            if (BuildAndDiagnose) {
              Diag(ExprLoc, diag::err_capture_binding_openmp) << Var;
              Diag(Var->getLocation(), diag::note_entity_declared_at) << Var;
            }
            return true;
          }
          OpenMPClauseKind IsOpenMPPrivateDecl = OpenMP().isOpenMPPrivateDecl(
              Var, RSI->OpenMPLevel, RSI->OpenMPCaptureLevel);
          // If the variable is private (i.e. not captured) and has variably
          // modified type, we still need to capture the type for correct
          // codegen in all regions, associated with the construct. Currently,
          // it is captured in the innermost captured region only.
          if (IsOpenMPPrivateDecl != OMPC_unknown &&
              Var->getType()->isVariablyModifiedType()) {
            QualType QTy = Var->getType();
            if (ParmVarDecl *PVD = dyn_cast_or_null<ParmVarDecl>(Var))
              QTy = PVD->getOriginalType();
            for (int I = 1,
                     E = OpenMP().getNumberOfConstructScopes(RSI->OpenMPLevel);
                 I < E; ++I) {
              auto *OuterRSI = cast<CapturedRegionScopeInfo>(
                  FunctionScopes[FunctionScopesIndex - I]);
              assert(RSI->OpenMPLevel == OuterRSI->OpenMPLevel &&
                     "Wrong number of captured regions associated with the "
                     "OpenMP construct.");
              captureVariablyModifiedType(Context, QTy, OuterRSI);
            }
          }
          bool IsTargetCap =
              IsOpenMPPrivateDecl != OMPC_private &&
              OpenMP().isOpenMPTargetCapturedDecl(Var, RSI->OpenMPLevel,
                                                  RSI->OpenMPCaptureLevel);
          // Do not capture global if it is not privatized in outer regions.
          bool IsGlobalCap =
              IsGlobal && OpenMP().isOpenMPGlobalCapturedDecl(
                              Var, RSI->OpenMPLevel, RSI->OpenMPCaptureLevel);

          // When we detect target captures we are looking from inside the
          // target region, therefore we need to propagate the capture from the
          // enclosing region. Therefore, the capture is not initially nested.
          if (IsTargetCap)
            OpenMP().adjustOpenMPTargetScopeIndex(FunctionScopesIndex,
                                                  RSI->OpenMPLevel);

          if (IsTargetCap || IsOpenMPPrivateDecl == OMPC_private ||
              (IsGlobal && !IsGlobalCap)) {
            Nested = !IsTargetCap;
            bool HasConst = DeclRefType.isConstQualified();
            DeclRefType = DeclRefType.getUnqualifiedType();
            // Don't lose diagnostics about assignments to const.
            if (HasConst)
              DeclRefType.addConst();
            CaptureType = Context.getLValueReferenceType(DeclRefType);
            break;
          }
        }
      }
    }
    if (CSI->ImpCaptureStyle == CapturingScopeInfo::ImpCap_None && !Explicit) {
      // No capture-default, and this is not an explicit capture
      // so cannot capture this variable.
      if (BuildAndDiagnose) {
        Diag(ExprLoc, diag::err_lambda_impcap) << Var;
        Diag(Var->getLocation(), diag::note_previous_decl) << Var;
        auto *LSI = cast<LambdaScopeInfo>(CSI);
        if (LSI->Lambda) {
          Diag(LSI->Lambda->getBeginLoc(), diag::note_lambda_decl);
          buildLambdaCaptureFixit(*this, LSI, Var);
        }
        // FIXME: If we error out because an outer lambda can not implicitly
        // capture a variable that an inner lambda explicitly captures, we
        // should have the inner lambda do the explicit capture - because
        // it makes for cleaner diagnostics later.  This would purely be done
        // so that the diagnostic does not misleadingly claim that a variable
        // can not be captured by a lambda implicitly even though it is captured
        // explicitly.  Suggestion:
        //  - create const bool VariableCaptureWasInitiallyExplicit = Explicit
        //    at the function head
        //  - cache the StartingDeclContext - this must be a lambda
        //  - captureInLambda in the innermost lambda the variable.
      }
      return true;
    }
    Explicit = false;
    FunctionScopesIndex--;
    if (IsInScopeDeclarationContext)
      DC = ParentDC;
  } while (!VarDC->Equals(DC));

  // Walk back down the scope stack, (e.g. from outer lambda to inner lambda)
  // computing the type of the capture at each step, checking type-specific
  // requirements, and adding captures if requested.
  // If the variable had already been captured previously, we start capturing
  // at the lambda nested within that one.
  bool Invalid = false;
  for (unsigned I = ++FunctionScopesIndex, N = MaxFunctionScopesIndex + 1; I != N;
       ++I) {
    CapturingScopeInfo *CSI = cast<CapturingScopeInfo>(FunctionScopes[I]);

    // Certain capturing entities (lambdas, blocks etc.) are not allowed to capture
    // certain types of variables (unnamed, variably modified types etc.)
    // so check for eligibility.
    if (!Invalid)
      Invalid =
          !isVariableCapturable(CSI, Var, ExprLoc, BuildAndDiagnose, *this);

    // After encountering an error, if we're actually supposed to capture, keep
    // capturing in nested contexts to suppress any follow-on diagnostics.
    if (Invalid && !BuildAndDiagnose)
      return true;

    if (BlockScopeInfo *BSI = dyn_cast<BlockScopeInfo>(CSI)) {
      Invalid = !captureInBlock(BSI, Var, ExprLoc, BuildAndDiagnose, CaptureType,
                               DeclRefType, Nested, *this, Invalid);
      Nested = true;
    } else if (CapturedRegionScopeInfo *RSI = dyn_cast<CapturedRegionScopeInfo>(CSI)) {
      Invalid = !captureInCapturedRegion(
          RSI, Var, ExprLoc, BuildAndDiagnose, CaptureType, DeclRefType, Nested,
          Kind, /*IsTopScope*/ I == N - 1, *this, Invalid);
      Nested = true;
    } else {
      LambdaScopeInfo *LSI = cast<LambdaScopeInfo>(CSI);
      Invalid =
          !captureInLambda(LSI, Var, ExprLoc, BuildAndDiagnose, CaptureType,
                           DeclRefType, Nested, Kind, EllipsisLoc,
                           /*IsTopScope*/ I == N - 1, *this, Invalid);
      Nested = true;
    }

    if (Invalid && !BuildAndDiagnose)
      return true;
  }
  return Invalid;
}

bool Sema::tryCaptureVariable(ValueDecl *Var, SourceLocation Loc,
                              TryCaptureKind Kind, SourceLocation EllipsisLoc) {
  QualType CaptureType;
  QualType DeclRefType;
  return tryCaptureVariable(Var, Loc, Kind, EllipsisLoc,
                            /*BuildAndDiagnose=*/true, CaptureType,
                            DeclRefType, nullptr);
}

bool Sema::NeedToCaptureVariable(ValueDecl *Var, SourceLocation Loc) {
  QualType CaptureType;
  QualType DeclRefType;
  return !tryCaptureVariable(
      Var, Loc, TryCaptureKind::Implicit, SourceLocation(),
      /*BuildAndDiagnose=*/false, CaptureType, DeclRefType, nullptr);
}

QualType Sema::getCapturedDeclRefType(ValueDecl *Var, SourceLocation Loc) {
  assert(Var && "Null value cannot be captured");

  QualType CaptureType;
  QualType DeclRefType;

  // Determine whether we can capture this variable.
  if (tryCaptureVariable(Var, Loc, TryCaptureKind::Implicit, SourceLocation(),
                         /*BuildAndDiagnose=*/false, CaptureType, DeclRefType,
                         nullptr))
    return QualType();

  return DeclRefType;
}

namespace {
// Helper to copy the template arguments from a DeclRefExpr or MemberExpr.
// The produced TemplateArgumentListInfo* points to data stored within this
// object, so should only be used in contexts where the pointer will not be
// used after the CopiedTemplateArgs object is destroyed.
class CopiedTemplateArgs {
  bool HasArgs;
  TemplateArgumentListInfo TemplateArgStorage;
public:
  template<typename RefExpr>
  CopiedTemplateArgs(RefExpr *E) : HasArgs(E->hasExplicitTemplateArgs()) {
    if (HasArgs)
      E->copyTemplateArgumentsInto(TemplateArgStorage);
  }
  operator TemplateArgumentListInfo*()
#ifdef __has_cpp_attribute
#if __has_cpp_attribute(clang::lifetimebound)
  [[clang::lifetimebound]]
#endif
#endif
  {
    return HasArgs ? &TemplateArgStorage : nullptr;
  }
};
}

/// Walk the set of potential results of an expression and mark them all as
/// non-odr-uses if they satisfy the side-conditions of the NonOdrUseReason.
///
/// \return A new expression if we found any potential results, ExprEmpty() if
///         not, and ExprError() if we diagnosed an error.
static ExprResult rebuildPotentialResultsAsNonOdrUsed(Sema &S, Expr *E,
                                                      NonOdrUseReason NOUR) {
  // Per C++11 [basic.def.odr], a variable is odr-used "unless it is
  // an object that satisfies the requirements for appearing in a
  // constant expression (5.19) and the lvalue-to-rvalue conversion (4.1)
  // is immediately applied."  This function handles the lvalue-to-rvalue
  // conversion part.
  //
  // If we encounter a node that claims to be an odr-use but shouldn't be, we
  // transform it into the relevant kind of non-odr-use node and rebuild the
  // tree of nodes leading to it.
  //
  // This is a mini-TreeTransform that only transforms a restricted subset of
  // nodes (and only certain operands of them).

  // Rebuild a subexpression.
  auto Rebuild = [&](Expr *Sub) {
    return rebuildPotentialResultsAsNonOdrUsed(S, Sub, NOUR);
  };

  // Check whether a potential result satisfies the requirements of NOUR.
  auto IsPotentialResultOdrUsed = [&](NamedDecl *D) {
    // Any entity other than a VarDecl is always odr-used whenever it's named
    // in a potentially-evaluated expression.
    auto *VD = dyn_cast<VarDecl>(D);
    if (!VD)
      return true;

    // C++2a [basic.def.odr]p4:
    //   A variable x whose name appears as a potentially-evalauted expression
    //   e is odr-used by e unless
    //   -- x is a reference that is usable in constant expressions, or
    //   -- x is a variable of non-reference type that is usable in constant
    //      expressions and has no mutable subobjects, and e is an element of
    //      the set of potential results of an expression of
    //      non-volatile-qualified non-class type to which the lvalue-to-rvalue
    //      conversion is applied, or
    //   -- x is a variable of non-reference type, and e is an element of the
    //      set of potential results of a discarded-value expression to which
    //      the lvalue-to-rvalue conversion is not applied
    //
    // We check the first bullet and the "potentially-evaluated" condition in
    // BuildDeclRefExpr. We check the type requirements in the second bullet
    // in CheckLValueToRValueConversionOperand below.
    switch (NOUR) {
    case NOUR_None:
    case NOUR_Unevaluated:
      llvm_unreachable("unexpected non-odr-use-reason");

    case NOUR_Constant:
      // Constant references were handled when they were built.
      if (VD->getType()->isReferenceType())
        return true;
      if (auto *RD = VD->getType()->getAsCXXRecordDecl())
        if (RD->hasDefinition() && RD->hasMutableFields())
          return true;
      if (!VD->isUsableInConstantExpressions(S.Context))
        return true;
      break;

    case NOUR_Discarded:
      if (VD->getType()->isReferenceType())
        return true;
      break;
    }
    return false;
  };

  // Check whether this expression may be odr-used in CUDA/HIP.
  auto MaybeCUDAODRUsed = [&]() -> bool {
    if (!S.LangOpts.CUDA)
      return false;
    LambdaScopeInfo *LSI = S.getCurLambda();
    if (!LSI)
      return false;
    auto *DRE = dyn_cast<DeclRefExpr>(E);
    if (!DRE)
      return false;
    auto *VD = dyn_cast<VarDecl>(DRE->getDecl());
    if (!VD)
      return false;
    return LSI->CUDAPotentialODRUsedVars.count(VD);
  };

  // Mark that this expression does not constitute an odr-use.
  auto MarkNotOdrUsed = [&] {
    if (!MaybeCUDAODRUsed()) {
      S.MaybeODRUseExprs.remove(E);
      if (LambdaScopeInfo *LSI = S.getCurLambda())
        LSI->markVariableExprAsNonODRUsed(E);
    }
  };

  // C++2a [basic.def.odr]p2:
  //   The set of potential results of an expression e is defined as follows:
  switch (E->getStmtClass()) {
  //   -- If e is an id-expression, ...
  case Expr::DeclRefExprClass: {
    auto *DRE = cast<DeclRefExpr>(E);
    if (DRE->isNonOdrUse() || IsPotentialResultOdrUsed(DRE->getDecl()))
      break;

    // Rebuild as a non-odr-use DeclRefExpr.
    MarkNotOdrUsed();
    return DeclRefExpr::Create(
        S.Context, DRE->getQualifierLoc(), DRE->getTemplateKeywordLoc(),
        DRE->getDecl(), DRE->refersToEnclosingVariableOrCapture(),
        DRE->getNameInfo(), DRE->getType(), DRE->getValueKind(),
        DRE->getFoundDecl(), CopiedTemplateArgs(DRE), NOUR);
  }

  case Expr::FunctionParmPackExprClass: {
    auto *FPPE = cast<FunctionParmPackExpr>(E);
    // If any of the declarations in the pack is odr-used, then the expression
    // as a whole constitutes an odr-use.
    for (ValueDecl *D : *FPPE)
      if (IsPotentialResultOdrUsed(D))
        return ExprEmpty();

    // FIXME: Rebuild as a non-odr-use FunctionParmPackExpr? In practice,
    // nothing cares about whether we marked this as an odr-use, but it might
    // be useful for non-compiler tools.
    MarkNotOdrUsed();
    break;
  }

  //   -- If e is a subscripting operation with an array operand...
  case Expr::ArraySubscriptExprClass: {
    auto *ASE = cast<ArraySubscriptExpr>(E);
    Expr *OldBase = ASE->getBase()->IgnoreImplicit();
    if (!OldBase->getType()->isArrayType())
      break;
    ExprResult Base = Rebuild(OldBase);
    if (!Base.isUsable())
      return Base;
    Expr *LHS = ASE->getBase() == ASE->getLHS() ? Base.get() : ASE->getLHS();
    Expr *RHS = ASE->getBase() == ASE->getRHS() ? Base.get() : ASE->getRHS();
    SourceLocation LBracketLoc = ASE->getBeginLoc(); // FIXME: Not stored.
    return S.ActOnArraySubscriptExpr(nullptr, LHS, LBracketLoc, RHS,
                                     ASE->getRBracketLoc());
  }

  case Expr::MemberExprClass: {
    auto *ME = cast<MemberExpr>(E);
    // -- If e is a class member access expression [...] naming a non-static
    //    data member...
    if (isa<FieldDecl>(ME->getMemberDecl())) {
      ExprResult Base = Rebuild(ME->getBase());
      if (!Base.isUsable())
        return Base;
      return MemberExpr::Create(
          S.Context, Base.get(), ME->isArrow(), ME->getOperatorLoc(),
          ME->getQualifierLoc(), ME->getTemplateKeywordLoc(),
          ME->getMemberDecl(), ME->getFoundDecl(), ME->getMemberNameInfo(),
          CopiedTemplateArgs(ME), ME->getType(), ME->getValueKind(),
          ME->getObjectKind(), ME->isNonOdrUse());
    }

    if (ME->getMemberDecl()->isCXXInstanceMember())
      break;

    // -- If e is a class member access expression naming a static data member,
    //    ...
    if (ME->isNonOdrUse() || IsPotentialResultOdrUsed(ME->getMemberDecl()))
      break;

    // Rebuild as a non-odr-use MemberExpr.
    MarkNotOdrUsed();
    return MemberExpr::Create(
        S.Context, ME->getBase(), ME->isArrow(), ME->getOperatorLoc(),
        ME->getQualifierLoc(), ME->getTemplateKeywordLoc(), ME->getMemberDecl(),
        ME->getFoundDecl(), ME->getMemberNameInfo(), CopiedTemplateArgs(ME),
        ME->getType(), ME->getValueKind(), ME->getObjectKind(), NOUR);
  }

  case Expr::BinaryOperatorClass: {
    auto *BO = cast<BinaryOperator>(E);
    Expr *LHS = BO->getLHS();
    Expr *RHS = BO->getRHS();
    // -- If e is a pointer-to-member expression of the form e1 .* e2 ...
    if (BO->getOpcode() == BO_PtrMemD) {
      ExprResult Sub = Rebuild(LHS);
      if (!Sub.isUsable())
        return Sub;
      BO->setLHS(Sub.get());
    //   -- If e is a comma expression, ...
    } else if (BO->getOpcode() == BO_Comma) {
      ExprResult Sub = Rebuild(RHS);
      if (!Sub.isUsable())
        return Sub;
      BO->setRHS(Sub.get());
    } else {
      break;
    }
    return ExprResult(BO);
  }

  //   -- If e has the form (e1)...
  case Expr::ParenExprClass: {
    auto *PE = cast<ParenExpr>(E);
    ExprResult Sub = Rebuild(PE->getSubExpr());
    if (!Sub.isUsable())
      return Sub;
    return S.ActOnParenExpr(PE->getLParen(), PE->getRParen(), Sub.get());
  }

  //   -- If e is a glvalue conditional expression, ...
  // We don't apply this to a binary conditional operator. FIXME: Should we?
  case Expr::ConditionalOperatorClass: {
    auto *CO = cast<ConditionalOperator>(E);
    ExprResult LHS = Rebuild(CO->getLHS());
    if (LHS.isInvalid())
      return ExprError();
    ExprResult RHS = Rebuild(CO->getRHS());
    if (RHS.isInvalid())
      return ExprError();
    if (!LHS.isUsable() && !RHS.isUsable())
      return ExprEmpty();
    if (!LHS.isUsable())
      LHS = CO->getLHS();
    if (!RHS.isUsable())
      RHS = CO->getRHS();
    return S.ActOnConditionalOp(CO->getQuestionLoc(), CO->getColonLoc(),
                                CO->getCond(), LHS.get(), RHS.get());
  }

  // [Clang extension]
  //   -- If e has the form __extension__ e1...
  case Expr::UnaryOperatorClass: {
    auto *UO = cast<UnaryOperator>(E);
    if (UO->getOpcode() != UO_Extension)
      break;
    ExprResult Sub = Rebuild(UO->getSubExpr());
    if (!Sub.isUsable())
      return Sub;
    return S.BuildUnaryOp(nullptr, UO->getOperatorLoc(), UO_Extension,
                          Sub.get());
  }

  // [Clang extension]
  //   -- If e has the form _Generic(...), the set of potential results is the
  //      union of the sets of potential results of the associated expressions.
  case Expr::GenericSelectionExprClass: {
    auto *GSE = cast<GenericSelectionExpr>(E);

    SmallVector<Expr *, 4> AssocExprs;
    bool AnyChanged = false;
    for (Expr *OrigAssocExpr : GSE->getAssocExprs()) {
      ExprResult AssocExpr = Rebuild(OrigAssocExpr);
      if (AssocExpr.isInvalid())
        return ExprError();
      if (AssocExpr.isUsable()) {
        AssocExprs.push_back(AssocExpr.get());
        AnyChanged = true;
      } else {
        AssocExprs.push_back(OrigAssocExpr);
      }
    }

    void *ExOrTy = nullptr;
    bool IsExpr = GSE->isExprPredicate();
    if (IsExpr)
      ExOrTy = GSE->getControllingExpr();
    else
      ExOrTy = GSE->getControllingType();
    return AnyChanged ? S.CreateGenericSelectionExpr(
                            GSE->getGenericLoc(), GSE->getDefaultLoc(),
                            GSE->getRParenLoc(), IsExpr, ExOrTy,
                            GSE->getAssocTypeSourceInfos(), AssocExprs)
                      : ExprEmpty();
  }

  // [Clang extension]
  //   -- If e has the form __builtin_choose_expr(...), the set of potential
  //      results is the union of the sets of potential results of the
  //      second and third subexpressions.
  case Expr::ChooseExprClass: {
    auto *CE = cast<ChooseExpr>(E);

    ExprResult LHS = Rebuild(CE->getLHS());
    if (LHS.isInvalid())
      return ExprError();

    ExprResult RHS = Rebuild(CE->getLHS());
    if (RHS.isInvalid())
      return ExprError();

    if (!LHS.get() && !RHS.get())
      return ExprEmpty();
    if (!LHS.isUsable())
      LHS = CE->getLHS();
    if (!RHS.isUsable())
      RHS = CE->getRHS();

    return S.ActOnChooseExpr(CE->getBuiltinLoc(), CE->getCond(), LHS.get(),
                             RHS.get(), CE->getRParenLoc());
  }

  // Step through non-syntactic nodes.
  case Expr::ConstantExprClass: {
    auto *CE = cast<ConstantExpr>(E);
    ExprResult Sub = Rebuild(CE->getSubExpr());
    if (!Sub.isUsable())
      return Sub;
    return ConstantExpr::Create(S.Context, Sub.get());
  }

  // We could mostly rely on the recursive rebuilding to rebuild implicit
  // casts, but not at the top level, so rebuild them here.
  case Expr::ImplicitCastExprClass: {
    auto *ICE = cast<ImplicitCastExpr>(E);
    // Only step through the narrow set of cast kinds we expect to encounter.
    // Anything else suggests we've left the region in which potential results
    // can be found.
    switch (ICE->getCastKind()) {
    case CK_NoOp:
    case CK_DerivedToBase:
    case CK_UncheckedDerivedToBase: {
      ExprResult Sub = Rebuild(ICE->getSubExpr());
      if (!Sub.isUsable())
        return Sub;
      CXXCastPath Path(ICE->path());
      return S.ImpCastExprToType(Sub.get(), ICE->getType(), ICE->getCastKind(),
                                 ICE->getValueKind(), &Path);
    }

    default:
      break;
    }
    break;
  }

  default:
    break;
  }

  // Can't traverse through this node. Nothing to do.
  return ExprEmpty();
}

ExprResult Sema::CheckLValueToRValueConversionOperand(Expr *E) {
  // Check whether the operand is or contains an object of non-trivial C union
  // type.
  if (E->getType().isVolatileQualified() &&
      (E->getType().hasNonTrivialToPrimitiveDestructCUnion() ||
       E->getType().hasNonTrivialToPrimitiveCopyCUnion()))
    checkNonTrivialCUnion(E->getType(), E->getExprLoc(),
                          NonTrivialCUnionContext::LValueToRValueVolatile,
                          NTCUK_Destruct | NTCUK_Copy);

  // C++2a [basic.def.odr]p4:
  //   [...] an expression of non-volatile-qualified non-class type to which
  //   the lvalue-to-rvalue conversion is applied [...]
  if (E->getType().isVolatileQualified() || E->getType()->isRecordType())
    return E;

  ExprResult Result =
      rebuildPotentialResultsAsNonOdrUsed(*this, E, NOUR_Constant);
  if (Result.isInvalid())
    return ExprError();
  return Result.get() ? Result : E;
}

ExprResult Sema::ActOnConstantExpression(ExprResult Res) {
  if (!Res.isUsable())
    return Res;

  // If a constant-expression is a reference to a variable where we delay
  // deciding whether it is an odr-use, just assume we will apply the
  // lvalue-to-rvalue conversion.  In the one case where this doesn't happen
  // (a non-type template argument), we have special handling anyway.
  return CheckLValueToRValueConversionOperand(Res.get());
}

void Sema::CleanupVarDeclMarking() {
  // Iterate through a local copy in case MarkVarDeclODRUsed makes a recursive
  // call.
  MaybeODRUseExprSet LocalMaybeODRUseExprs;
  std::swap(LocalMaybeODRUseExprs, MaybeODRUseExprs);

  for (Expr *E : LocalMaybeODRUseExprs) {
    if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      MarkVarDeclODRUsed(cast<VarDecl>(DRE->getDecl()),
                         DRE->getLocation(), *this);
    } else if (auto *ME = dyn_cast<MemberExpr>(E)) {
      MarkVarDeclODRUsed(cast<VarDecl>(ME->getMemberDecl()), ME->getMemberLoc(),
                         *this);
    } else if (auto *FP = dyn_cast<FunctionParmPackExpr>(E)) {
      for (ValueDecl *VD : *FP)
        MarkVarDeclODRUsed(VD, FP->getParameterPackLocation(), *this);
    } else {
      llvm_unreachable("Unexpected expression");
    }
  }

  assert(MaybeODRUseExprs.empty() &&
         "MarkVarDeclODRUsed failed to cleanup MaybeODRUseExprs?");
}

static void DoMarkPotentialCapture(Sema &SemaRef, SourceLocation Loc,
                                   ValueDecl *Var, Expr *E) {
  VarDecl *VD = Var->getPotentiallyDecomposedVarDecl();
  if (!VD)
    return;

  const bool RefersToEnclosingScope =
      (SemaRef.CurContext != VD->getDeclContext() &&
       VD->getDeclContext()->isFunctionOrMethod() && VD->hasLocalStorage());
  if (RefersToEnclosingScope) {
    LambdaScopeInfo *const LSI =
        SemaRef.getCurLambda(/*IgnoreNonLambdaCapturingScope=*/true);
    if (LSI && (!LSI->CallOperator ||
                !LSI->CallOperator->Encloses(Var->getDeclContext()))) {
      // If a variable could potentially be odr-used, defer marking it so
      // until we finish analyzing the full expression for any
      // lvalue-to-rvalue
      // or discarded value conversions that would obviate odr-use.
      // Add it to the list of potential captures that will be analyzed
      // later (ActOnFinishFullExpr) for eventual capture and odr-use marking
      // unless the variable is a reference that was initialized by a constant
      // expression (this will never need to be captured or odr-used).
      //
      // FIXME: We can simplify this a lot after implementing P0588R1.
      assert(E && "Capture variable should be used in an expression.");
      if (!Var->getType()->isReferenceType() ||
          !VD->isUsableInConstantExpressions(SemaRef.Context))
        LSI->addPotentialCapture(E->IgnoreParens());
    }
  }
}

static void DoMarkVarDeclReferenced(
    Sema &SemaRef, SourceLocation Loc, VarDecl *Var, Expr *E,
    llvm::DenseMap<const VarDecl *, int> &RefsMinusAssignments) {
  assert((!E || isa<DeclRefExpr>(E) || isa<MemberExpr>(E) ||
          isa<FunctionParmPackExpr>(E)) &&
         "Invalid Expr argument to DoMarkVarDeclReferenced");
  Var->setReferenced();

  if (Var->isInvalidDecl())
    return;

  auto *MSI = Var->getMemberSpecializationInfo();
  TemplateSpecializationKind TSK = MSI ? MSI->getTemplateSpecializationKind()
                                       : Var->getTemplateSpecializationKind();

  OdrUseContext OdrUse = isOdrUseContext(SemaRef);
  bool UsableInConstantExpr =
      Var->mightBeUsableInConstantExpressions(SemaRef.Context);

  // Only track variables with internal linkage or local scope.
  // Use canonical decl so in-class declarations and out-of-class definitions
  // of static data members in anonymous namespaces are tracked as a single
  // entry.
  const VarDecl *CanonVar = Var->getCanonicalDecl();
  if ((CanonVar->isLocalVarDeclOrParm() ||
       CanonVar->isInternalLinkageFileVar()) &&
      !CanonVar->hasExternalStorage()) {
    RefsMinusAssignments.insert({CanonVar, 0}).first->getSecond()++;
  }

  // C++20 [expr.const]p12:
  //   A variable [...] is needed for constant evaluation if it is [...] a
  //   variable whose name appears as a potentially constant evaluated
  //   expression that is either a contexpr variable or is of non-volatile
  //   const-qualified integral type or of reference type
  bool NeededForConstantEvaluation =
      isPotentiallyConstantEvaluatedContext(SemaRef) && UsableInConstantExpr;

  bool NeedDefinition =
      OdrUse == OdrUseContext::Used || NeededForConstantEvaluation ||
      (TSK != clang::TSK_Undeclared && !UsableInConstantExpr &&
       Var->getType()->isUndeducedType());

  assert(!isa<VarTemplatePartialSpecializationDecl>(Var) &&
         "Can't instantiate a partial template specialization.");

  // If this might be a member specialization of a static data member, check
  // the specialization is visible. We already did the checks for variable
  // template specializations when we created them.
  if (NeedDefinition && TSK != TSK_Undeclared &&
      !isa<VarTemplateSpecializationDecl>(Var))
    SemaRef.checkSpecializationVisibility(Loc, Var);

  // Perform implicit instantiation of static data members, static data member
  // templates of class templates, and variable template specializations. Delay
  // instantiations of variable templates, except for those that could be used
  // in a constant expression.
  if (NeedDefinition && isTemplateInstantiation(TSK)) {
    // Per C++17 [temp.explicit]p10, we may instantiate despite an explicit
    // instantiation declaration if a variable is usable in a constant
    // expression (among other cases).
    bool TryInstantiating =
        TSK == TSK_ImplicitInstantiation ||
        (TSK == TSK_ExplicitInstantiationDeclaration && UsableInConstantExpr);

    if (TryInstantiating) {
      SourceLocation PointOfInstantiation =
          MSI ? MSI->getPointOfInstantiation() : Var->getPointOfInstantiation();
      bool FirstInstantiation = PointOfInstantiation.isInvalid();
      if (FirstInstantiation) {
        PointOfInstantiation = Loc;
        if (MSI)
          MSI->setPointOfInstantiation(PointOfInstantiation);
          // FIXME: Notify listener.
        else
          Var->setTemplateSpecializationKind(TSK, PointOfInstantiation);
      }

      if (UsableInConstantExpr || Var->getType()->isUndeducedType()) {
        // Do not defer instantiations of variables that could be used in a
        // constant expression.
        // The type deduction also needs a complete initializer.
        SemaRef.runWithSufficientStackSpace(PointOfInstantiation, [&] {
          SemaRef.InstantiateVariableDefinition(PointOfInstantiation, Var);
        });

        // The size of an incomplete array type can be updated by
        // instantiating the initializer. The DeclRefExpr's type should be
        // updated accordingly too, or users of it would be confused!
        if (E)
          SemaRef.getCompletedType(E);

        // Re-set the member to trigger a recomputation of the dependence bits
        // for the expression.
        if (auto *DRE = dyn_cast_or_null<DeclRefExpr>(E))
          DRE->setDecl(DRE->getDecl());
        else if (auto *ME = dyn_cast_or_null<MemberExpr>(E))
          ME->setMemberDecl(ME->getMemberDecl());
      } else if (FirstInstantiation) {
        SemaRef.PendingInstantiations
            .push_back(std::make_pair(Var, PointOfInstantiation));
      } else {
        bool Inserted = false;
        for (auto &I : SemaRef.SavedPendingInstantiations) {
          auto Iter = llvm::find_if(
              I, [Var](const Sema::PendingImplicitInstantiation &P) {
                return P.first == Var;
              });
          if (Iter != I.end()) {
            SemaRef.PendingInstantiations.push_back(*Iter);
            I.erase(Iter);
            Inserted = true;
            break;
          }
        }

        // FIXME: For a specialization of a variable template, we don't
        // distinguish between "declaration and type implicitly instantiated"
        // and "implicit instantiation of definition requested", so we have
        // no direct way to avoid enqueueing the pending instantiation
        // multiple times.
        if (isa<VarTemplateSpecializationDecl>(Var) && !Inserted)
          SemaRef.PendingInstantiations
            .push_back(std::make_pair(Var, PointOfInstantiation));
      }
    }
  }

  // C++2a [basic.def.odr]p4:
  //   A variable x whose name appears as a potentially-evaluated expression e
  //   is odr-used by e unless
  //   -- x is a reference that is usable in constant expressions
  //   -- x is a variable of non-reference type that is usable in constant
  //      expressions and has no mutable subobjects [FIXME], and e is an
  //      element of the set of potential results of an expression of
  //      non-volatile-qualified non-class type to which the lvalue-to-rvalue
  //      conversion is applied
  //   -- x is a variable of non-reference type, and e is an element of the set
  //      of potential results of a discarded-value expression to which the
  //      lvalue-to-rvalue conversion is not applied [FIXME]
  //
  // We check the first part of the second bullet here, and
  // Sema::CheckLValueToRValueConversionOperand deals with the second part.
  // FIXME: To get the third bullet right, we need to delay this even for
  // variables that are not usable in constant expressions.

  // If we already know this isn't an odr-use, there's nothing more to do.
  if (DeclRefExpr *DRE = dyn_cast_or_null<DeclRefExpr>(E))
    if (DRE->isNonOdrUse())
      return;
  if (MemberExpr *ME = dyn_cast_or_null<MemberExpr>(E))
    if (ME->isNonOdrUse())
      return;

  switch (OdrUse) {
  case OdrUseContext::None:
    // In some cases, a variable may not have been marked unevaluated, if it
    // appears in a defaukt initializer.
    assert((!E || isa<FunctionParmPackExpr>(E) ||
            SemaRef.isUnevaluatedContext()) &&
           "missing non-odr-use marking for unevaluated decl ref");
    break;

  case OdrUseContext::FormallyOdrUsed:
    // FIXME: Ignoring formal odr-uses results in incorrect lambda capture
    // behavior.
    break;

  case OdrUseContext::Used:
    // If we might later find that this expression isn't actually an odr-use,
    // delay the marking.
    if (E && Var->isUsableInConstantExpressions(SemaRef.Context))
      SemaRef.MaybeODRUseExprs.insert(E);
    else
      MarkVarDeclODRUsed(Var, Loc, SemaRef);
    break;

  case OdrUseContext::Dependent:
    // If this is a dependent context, we don't need to mark variables as
    // odr-used, but we may still need to track them for lambda capture.
    // FIXME: Do we also need to do this inside dependent typeid expressions
    // (which are modeled as unevaluated at this point)?
    DoMarkPotentialCapture(SemaRef, Loc, Var, E);
    break;
  }
}

static void DoMarkBindingDeclReferenced(Sema &SemaRef, SourceLocation Loc,
                                        BindingDecl *BD, Expr *E) {
  BD->setReferenced();

  if (BD->isInvalidDecl())
    return;

  OdrUseContext OdrUse = isOdrUseContext(SemaRef);
  if (OdrUse == OdrUseContext::Used) {
    QualType CaptureType, DeclRefType;
    SemaRef.tryCaptureVariable(BD, Loc, TryCaptureKind::Implicit,
                               /*EllipsisLoc*/ SourceLocation(),
                               /*BuildAndDiagnose*/ true, CaptureType,
                               DeclRefType,
                               /*FunctionScopeIndexToStopAt*/ nullptr);
  } else if (OdrUse == OdrUseContext::Dependent) {
    DoMarkPotentialCapture(SemaRef, Loc, BD, E);
  }
}

void Sema::MarkVariableReferenced(SourceLocation Loc, VarDecl *Var) {
  DoMarkVarDeclReferenced(*this, Loc, Var, nullptr, RefsMinusAssignments);
}

// C++ [temp.dep.expr]p3:
//   An id-expression is type-dependent if it contains:
//     - an identifier associated by name lookup with an entity captured by copy
//       in a lambda-expression that has an explicit object parameter whose type
//       is dependent ([dcl.fct]),
static void FixDependencyOfIdExpressionsInLambdaWithDependentObjectParameter(
    Sema &SemaRef, ValueDecl *D, Expr *E) {
  auto *ID = dyn_cast<DeclRefExpr>(E);
  if (!ID || ID->isTypeDependent() || !ID->refersToEnclosingVariableOrCapture())
    return;

  // If any enclosing lambda with a dependent explicit object parameter either
  // explicitly captures the variable by value, or has a capture default of '='
  // and does not capture the variable by reference, then the type of the DRE
  // is dependent on the type of that lambda's explicit object parameter.
  auto IsDependent = [&]() {
    for (auto *Scope : llvm::reverse(SemaRef.FunctionScopes)) {
      auto *LSI = dyn_cast<sema::LambdaScopeInfo>(Scope);
      if (!LSI)
        continue;

      if (LSI->Lambda && !LSI->Lambda->Encloses(SemaRef.CurContext) &&
          LSI->AfterParameterList)
        return false;

      const auto *MD = LSI->CallOperator;
      if (MD->getType().isNull())
        continue;

      const auto *Ty = MD->getType()->getAs<FunctionProtoType>();
      if (!Ty || !MD->isExplicitObjectMemberFunction() ||
          !Ty->getParamType(0)->isDependentType())
        continue;

      if (auto *C = LSI->CaptureMap.count(D) ? &LSI->getCapture(D) : nullptr) {
        if (C->isCopyCapture())
          return true;
        continue;
      }

      if (LSI->ImpCaptureStyle == LambdaScopeInfo::ImpCap_LambdaByval)
        return true;
    }
    return false;
  }();

  ID->setCapturedByCopyInLambdaWithExplicitObjectParameter(
      IsDependent, SemaRef.getASTContext());
}

static void
MarkExprReferenced(Sema &SemaRef, SourceLocation Loc, Decl *D, Expr *E,
                   bool MightBeOdrUse,
                   llvm::DenseMap<const VarDecl *, int> &RefsMinusAssignments) {
  if (SemaRef.OpenMP().isInOpenMPDeclareTargetContext())
    SemaRef.OpenMP().checkDeclIsAllowedInOpenMPTarget(E, D);

  if (SemaRef.getLangOpts().OpenACC)
    SemaRef.OpenACC().CheckDeclReference(Loc, E, D);

  if (VarDecl *Var = dyn_cast<VarDecl>(D)) {
    DoMarkVarDeclReferenced(SemaRef, Loc, Var, E, RefsMinusAssignments);
    if (SemaRef.getLangOpts().CPlusPlus)
      FixDependencyOfIdExpressionsInLambdaWithDependentObjectParameter(SemaRef,
                                                                       Var, E);
    return;
  }

  if (BindingDecl *Decl = dyn_cast<BindingDecl>(D)) {
    DoMarkBindingDeclReferenced(SemaRef, Loc, Decl, E);
    if (SemaRef.getLangOpts().CPlusPlus)
      FixDependencyOfIdExpressionsInLambdaWithDependentObjectParameter(SemaRef,
                                                                       Decl, E);
    return;
  }
  SemaRef.MarkAnyDeclReferenced(Loc, D, MightBeOdrUse);

  // If this is a call to a method via a cast, also mark the method in the
  // derived class used in case codegen can devirtualize the call.
  const MemberExpr *ME = dyn_cast<MemberExpr>(E);
  if (!ME)
    return;
  CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(ME->getMemberDecl());
  if (!MD)
    return;
  // Only attempt to devirtualize if this is truly a virtual call.
  bool IsVirtualCall = MD->isVirtual() &&
                          ME->performsVirtualDispatch(SemaRef.getLangOpts());
  if (!IsVirtualCall)
    return;

  // If it's possible to devirtualize the call, mark the called function
  // referenced.
  CXXMethodDecl *DM = MD->getDevirtualizedMethod(
      ME->getBase(), SemaRef.getLangOpts().AppleKext);
  if (DM)
    SemaRef.MarkAnyDeclReferenced(Loc, DM, MightBeOdrUse);
}

void Sema::MarkDeclRefReferenced(DeclRefExpr *E, const Expr *Base) {
  // [basic.def.odr] (CWG 1614)
  // A function is named by an expression or conversion [...]
  // unless it is a pure virtual function and either the expression is not an
  // id-expression naming the function with an explicitly qualified name or
  // the expression forms a pointer to member
  bool OdrUse = true;
  if (const CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(E->getDecl()))
    if (Method->isVirtual() &&
        !Method->getDevirtualizedMethod(Base, getLangOpts().AppleKext))
      OdrUse = false;

  if (auto *FD = dyn_cast<FunctionDecl>(E->getDecl())) {
    if (!isUnevaluatedContext() && !isConstantEvaluatedContext() &&
        !isImmediateFunctionContext() &&
        !isCheckingDefaultArgumentOrInitializer() &&
        FD->isImmediateFunction() && !RebuildingImmediateInvocation &&
        !FD->isDependentContext())
      ExprEvalContexts.back().ReferenceToConsteval.insert(E);
  }
  MarkExprReferenced(*this, E->getLocation(), E->getDecl(), E, OdrUse,
                     RefsMinusAssignments);
}

void Sema::MarkMemberReferenced(MemberExpr *E) {
  // C++11 [basic.def.odr]p2:
  //   A non-overloaded function whose name appears as a potentially-evaluated
  //   expression or a member of a set of candidate functions, if selected by
  //   overload resolution when referred to from a potentially-evaluated
  //   expression, is odr-used, unless it is a pure virtual function and its
  //   name is not explicitly qualified.
  bool MightBeOdrUse = true;
  if (E->performsVirtualDispatch(getLangOpts())) {
    if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(E->getMemberDecl()))
      if (Method->isPureVirtual())
        MightBeOdrUse = false;
  }
  SourceLocation Loc =
      E->getMemberLoc().isValid() ? E->getMemberLoc() : E->getBeginLoc();
  MarkExprReferenced(*this, Loc, E->getMemberDecl(), E, MightBeOdrUse,
                     RefsMinusAssignments);
}

void Sema::MarkFunctionParmPackReferenced(FunctionParmPackExpr *E) {
  for (ValueDecl *VD : *E)
    MarkExprReferenced(*this, E->getParameterPackLocation(), VD, E, true,
                       RefsMinusAssignments);
}

/// Perform marking for a reference to an arbitrary declaration.  It
/// marks the declaration referenced, and performs odr-use checking for
/// functions and variables. This method should not be used when building a
/// normal expression which refers to a variable.
void Sema::MarkAnyDeclReferenced(SourceLocation Loc, Decl *D,
                                 bool MightBeOdrUse) {
  if (MightBeOdrUse) {
    if (auto *VD = dyn_cast<VarDecl>(D)) {
      MarkVariableReferenced(Loc, VD);
      return;
    }
  }
  if (auto *FD = dyn_cast<FunctionDecl>(D)) {
    MarkFunctionReferenced(Loc, FD, MightBeOdrUse);
    return;
  }
  D->setReferenced();
}

namespace {
  // Mark all of the declarations used by a type as referenced.
  // FIXME: Not fully implemented yet! We need to have a better understanding
  // of when we're entering a context we should not recurse into.
  // FIXME: This is and EvaluatedExprMarker are more-or-less equivalent to
  // TreeTransforms rebuilding the type in a new context. Rather than
  // duplicating the TreeTransform logic, we should consider reusing it here.
  // Currently that causes problems when rebuilding LambdaExprs.
class MarkReferencedDecls : public DynamicRecursiveASTVisitor {
  Sema &S;
  SourceLocation Loc;

public:
  MarkReferencedDecls(Sema &S, SourceLocation Loc) : S(S), Loc(Loc) {}

  bool TraverseTemplateArgument(const TemplateArgument &Arg) override;
};
}

bool MarkReferencedDecls::TraverseTemplateArgument(
    const TemplateArgument &Arg) {
  {
    // A non-type template argument is a constant-evaluated context.
    EnterExpressionEvaluationContext Evaluated(
        S, Sema::ExpressionEvaluationContext::ConstantEvaluated);
    if (Arg.getKind() == TemplateArgument::Declaration) {
      if (Decl *D = Arg.getAsDecl())
        S.MarkAnyDeclReferenced(Loc, D, true);
    } else if (Arg.getKind() == TemplateArgument::Expression) {
      S.MarkDeclarationsReferencedInExpr(Arg.getAsExpr(), false);
    }
  }

  return DynamicRecursiveASTVisitor::TraverseTemplateArgument(Arg);
}

void Sema::MarkDeclarationsReferencedInType(SourceLocation Loc, QualType T) {
  MarkReferencedDecls Marker(*this, Loc);
  Marker.TraverseType(T);
}

namespace {
/// Helper class that marks all of the declarations referenced by
/// potentially-evaluated subexpressions as "referenced".
class EvaluatedExprMarker : public UsedDeclVisitor<EvaluatedExprMarker> {
public:
  typedef UsedDeclVisitor<EvaluatedExprMarker> Inherited;
  bool SkipLocalVariables;
  ArrayRef<const Expr *> StopAt;

  EvaluatedExprMarker(Sema &S, bool SkipLocalVariables,
                      ArrayRef<const Expr *> StopAt)
      : Inherited(S), SkipLocalVariables(SkipLocalVariables), StopAt(StopAt) {}

  void visitUsedDecl(SourceLocation Loc, Decl *D) {
    S.MarkFunctionReferenced(Loc, cast<FunctionDecl>(D));
  }

  void Visit(Expr *E) {
    if (llvm::is_contained(StopAt, E))
      return;
    Inherited::Visit(E);
  }

  void VisitConstantExpr(ConstantExpr *E) {
    // Don't mark declarations within a ConstantExpression, as this expression
    // will be evaluated and folded to a value.
  }

  void VisitDeclRefExpr(DeclRefExpr *E) {
    // If we were asked not to visit local variables, don't.
    if (SkipLocalVariables) {
      if (VarDecl *VD = dyn_cast<VarDecl>(E->getDecl()))
        if (VD->hasLocalStorage())
          return;
    }

    // FIXME: This can trigger the instantiation of the initializer of a
    // variable, which can cause the expression to become value-dependent
    // or error-dependent. Do we need to propagate the new dependence bits?
    S.MarkDeclRefReferenced(E);
  }

  void VisitMemberExpr(MemberExpr *E) {
    S.MarkMemberReferenced(E);
    Visit(E->getBase());
  }
};
} // namespace

void Sema::MarkDeclarationsReferencedInExpr(Expr *E,
                                            bool SkipLocalVariables,
                                            ArrayRef<const Expr*> StopAt) {
  EvaluatedExprMarker(*this, SkipLocalVariables, StopAt).Visit(E);
}

/// Emit a diagnostic when statements are reachable.
bool Sema::DiagIfReachable(SourceLocation Loc, ArrayRef<const Stmt *> Stmts,
                           const PartialDiagnostic &PD) {
  VarDecl *Decl = ExprEvalContexts.back().DeclForInitializer;
  // The initializer of a constexpr variable or of the first declaration of a
  // static data member is not syntactically a constant evaluated constant,
  // but nonetheless is always required to be a constant expression, so we
  // can skip diagnosing.
  if (Decl &&
      (Decl->isConstexpr() || (Decl->isStaticDataMember() &&
                               Decl->isFirstDecl() && !Decl->isInline())))
    return false;

  if (Stmts.empty()) {
    Diag(Loc, PD);
    return true;
  }

  if (getCurFunction()) {
    FunctionScopes.back()->PossiblyUnreachableDiags.push_back(
        sema::PossiblyUnreachableDiag(PD, Loc, Stmts));
    return true;
  }

  // For non-constexpr file-scope variables with reachability context (non-empty
  // Stmts), build a CFG for the initializer and check whether the context in
  // question is reachable.
  if (Decl && Decl->isFileVarDecl()) {
    AnalysisWarnings.registerVarDeclWarning(
        Decl, sema::PossiblyUnreachableDiag(PD, Loc, Stmts));
    return true;
  }

  Diag(Loc, PD);
  return true;
}

/// Emit a diagnostic that describes an effect on the run-time behavior
/// of the program being compiled.
///
/// This routine emits the given diagnostic when the code currently being
/// type-checked is "potentially evaluated", meaning that there is a
/// possibility that the code will actually be executable. Code in sizeof()
/// expressions, code used only during overload resolution, etc., are not
/// potentially evaluated. This routine will suppress such diagnostics or,
/// in the absolutely nutty case of potentially potentially evaluated
/// expressions (C++ typeid), queue the diagnostic to potentially emit it
/// later.
///
/// This routine should be used for all diagnostics that describe the run-time
/// behavior of a program, such as passing a non-POD value through an ellipsis.
/// Failure to do so will likely result in spurious diagnostics or failures
/// during overload resolution or within sizeof/alignof/typeof/typeid.
bool Sema::DiagRuntimeBehavior(SourceLocation Loc, ArrayRef<const Stmt*> Stmts,
                               const PartialDiagnostic &PD) {

  if (ExprEvalContexts.back().isDiscardedStatementContext())
    return false;

  switch (ExprEvalContexts.back().Context) {
  case ExpressionEvaluationContext::Unevaluated:
  case ExpressionEvaluationContext::UnevaluatedList:
  case ExpressionEvaluationContext::UnevaluatedAbstract:
  case ExpressionEvaluationContext::DiscardedStatement:
    // The argument will never be evaluated, so don't complain.
    break;

  case ExpressionEvaluationContext::ConstantEvaluated:
  case ExpressionEvaluationContext::ImmediateFunctionContext:
    // Relevant diagnostics should be produced by constant evaluation.
    break;

  case ExpressionEvaluationContext::PotentiallyEvaluated:
  case ExpressionEvaluationContext::PotentiallyEvaluatedIfUsed:
    return DiagIfReachable(Loc, Stmts, PD);
  }

  return false;
}

bool Sema::DiagRuntimeBehavior(SourceLocation Loc, const Stmt *Statement,
                               const PartialDiagnostic &PD) {
  return DiagRuntimeBehavior(
      Loc, Statement ? llvm::ArrayRef(Statement) : llvm::ArrayRef<Stmt *>(),
      PD);
}

bool Sema::CheckCallReturnType(QualType ReturnType, SourceLocation Loc,
                               CallExpr *CE, FunctionDecl *FD) {
  if (ReturnType->isVoidType() || !ReturnType->isIncompleteType())
    return false;

  // If we're inside a decltype's expression, don't check for a valid return
  // type or construct temporaries until we know whether this is the last call.
  if (ExprEvalContexts.back().ExprContext ==
      ExpressionEvaluationContextRecord::EK_Decltype) {
    ExprEvalContexts.back().DelayedDecltypeCalls.push_back(CE);
    return false;
  }

  class CallReturnIncompleteDiagnoser : public TypeDiagnoser {
    FunctionDecl *FD;
    CallExpr *CE;

  public:
    CallReturnIncompleteDiagnoser(FunctionDecl *FD, CallExpr *CE)
      : FD(FD), CE(CE) { }

    void diagnose(Sema &S, SourceLocation Loc, QualType T) override {
      if (!FD) {
        S.Diag(Loc, diag::err_call_incomplete_return)
          << T << CE->getSourceRange();
        return;
      }

      S.Diag(Loc, diag::err_call_function_incomplete_return)
          << CE->getSourceRange() << FD << T;
      S.Diag(FD->getLocation(), diag::note_entity_declared_at)
          << FD->getDeclName();
    }
  } Diagnoser(FD, CE);

  if (RequireCompleteType(Loc, ReturnType, Diagnoser))
    return true;

  return false;
}

// Diagnose the s/=/==/ and s/\|=/!=/ typos. Note that adding parentheses
// will prevent this condition from triggering, which is what we want.
void Sema::DiagnoseAssignmentAsCondition(Expr *E) {
  SourceLocation Loc;

  unsigned diagnostic = diag::warn_condition_is_assignment;
  bool IsOrAssign = false;

  if (BinaryOperator *Op = dyn_cast<BinaryOperator>(E)) {
    if (Op->getOpcode() != BO_Assign && Op->getOpcode() != BO_OrAssign)
      return;

    IsOrAssign = Op->getOpcode() == BO_OrAssign;

    // Greylist some idioms by putting them into a warning subcategory.
    if (ObjCMessageExpr *ME
          = dyn_cast<ObjCMessageExpr>(Op->getRHS()->IgnoreParenCasts())) {
      Selector Sel = ME->getSelector();

      // self = [<foo> init...]
      if (ObjC().isSelfExpr(Op->getLHS()) && ME->getMethodFamily() == OMF_init)
        diagnostic = diag::warn_condition_is_idiomatic_assignment;

      // <foo> = [<bar> nextObject]
      else if (Sel.isUnarySelector() && Sel.getNameForSlot(0) == "nextObject")
        diagnostic = diag::warn_condition_is_idiomatic_assignment;
    }

    Loc = Op->getOperatorLoc();
  } else if (CXXOperatorCallExpr *Op = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (Op->getOperator() != OO_Equal && Op->getOperator() != OO_PipeEqual)
      return;

    IsOrAssign = Op->getOperator() == OO_PipeEqual;
    Loc = Op->getOperatorLoc();
  } else if (PseudoObjectExpr *POE = dyn_cast<PseudoObjectExpr>(E))
    return DiagnoseAssignmentAsCondition(POE->getSyntacticForm());
  else {
    // Not an assignment.
    return;
  }

  Diag(Loc, diagnostic) << E->getSourceRange();

  SourceLocation Open = E->getBeginLoc();
  SourceLocation Close = getLocForEndOfToken(E->getSourceRange().getEnd());
  Diag(Loc, diag::note_condition_assign_silence)
        << FixItHint::CreateInsertion(Open, "(")
        << FixItHint::CreateInsertion(Close, ")");

  if (IsOrAssign)
    Diag(Loc, diag::note_condition_or_assign_to_comparison)
      << FixItHint::CreateReplacement(Loc, "!=");
  else
    Diag(Loc, diag::note_condition_assign_to_comparison)
      << FixItHint::CreateReplacement(Loc, "==");
}

void Sema::DiagnoseEqualityWithExtraParens(ParenExpr *ParenE) {
  // Don't warn if the parens came from a macro.
  SourceLocation parenLoc = ParenE->getBeginLoc();
  if (parenLoc.isInvalid() || parenLoc.isMacroID())
    return;
  // Don't warn for dependent expressions.
  if (ParenE->isTypeDependent())
    return;

  Expr *E = ParenE->IgnoreParens();
  if (ParenE->isProducedByFoldExpansion() && ParenE->getSubExpr() == E)
    return;

  if (BinaryOperator *opE = dyn_cast<BinaryOperator>(E))
    if (opE->getOpcode() == BO_EQ &&
        opE->getLHS()->IgnoreParenImpCasts()->isModifiableLvalue(Context)
                                                           == Expr::MLV_Valid) {
      SourceLocation Loc = opE->getOperatorLoc();

      Diag(Loc, diag::warn_equality_with_extra_parens) << E->getSourceRange();
      SourceRange ParenERange = ParenE->getSourceRange();
      Diag(Loc, diag::note_equality_comparison_silence)
        << FixItHint::CreateRemoval(ParenERange.getBegin())
        << FixItHint::CreateRemoval(ParenERange.getEnd());
      Diag(Loc, diag::note_equality_comparison_to_assign)
        << FixItHint::CreateReplacement(Loc, "=");
    }
}

ExprResult Sema::CheckBooleanCondition(SourceLocation Loc, Expr *E,
                                       bool IsConstexpr) {
  DiagnoseAssignmentAsCondition(E);
  if (ParenExpr *parenE = dyn_cast<ParenExpr>(E))
    DiagnoseEqualityWithExtraParens(parenE);

  ExprResult result = CheckPlaceholderExpr(E);
  if (result.isInvalid()) return ExprError();
  E = result.get();

  if (!E->isTypeDependent()) {
    if (E->getType() == Context.AMDGPUFeaturePredicateTy)
      return AMDGPU().ExpandAMDGPUPredicateBuiltIn(E);

    if (getLangOpts().CPlusPlus)
      return CheckCXXBooleanCondition(E, IsConstexpr); // C++ 6.4p4

    ExprResult ERes = DefaultFunctionArrayLvalueConversion(E);
    if (ERes.isInvalid())
      return ExprError();
    E = ERes.get();

    QualType T = E->getType();
    if (!T->isScalarType()) { // C99 6.8.4.1p1
      Diag(Loc, diag::err_typecheck_statement_requires_scalar)
        << T << E->getSourceRange();
      return ExprError();
    }
    CheckBoolLikeConversion(E, Loc);
  }

  return E;
}

Sema::ConditionResult Sema::ActOnCondition(Scope *S, SourceLocation Loc,
                                           Expr *SubExpr, ConditionKind CK,
                                           bool MissingOK) {
  // MissingOK indicates whether having no condition expression is valid
  // (for loop) or invalid (e.g. while loop).
  if (!SubExpr)
    return MissingOK ? ConditionResult() : ConditionError();

  ExprResult Cond;
  switch (CK) {
  case ConditionKind::Boolean:
    Cond = CheckBooleanCondition(Loc, SubExpr);
    break;

  case ConditionKind::ConstexprIf:
    // Note: this might produce a FullExpr
    Cond = CheckBooleanCondition(Loc, SubExpr, true);
    break;

  case ConditionKind::Switch:
    Cond = CheckSwitchCondition(Loc, SubExpr);
    break;
  }
  if (Cond.isInvalid()) {
    Cond = CreateRecoveryExpr(SubExpr->getBeginLoc(), SubExpr->getEndLoc(),
                              {SubExpr}, PreferredConditionType(CK));
    if (!Cond.get())
      return ConditionError();
  } else if (Cond.isUsable() && !isa<FullExpr>(Cond.get()))
    Cond = ActOnFinishFullExpr(Cond.get(), Loc, /*DiscardedValue*/ false);

  if (!Cond.isUsable())
    return ConditionError();

  return ConditionResult(*this, nullptr, Cond,
                         CK == ConditionKind::ConstexprIf);
}

namespace {
  /// A visitor for rebuilding an expression of type __unknown_anytype
  /// into one which resolves the type directly on the referring
  /// expression.  Strict preservation of the original source
  /// structure is not a goal.
  struct RebuildUnknownAnyExpr
    : StmtVisitor<RebuildUnknownAnyExpr, ExprResult> {

    Sema &S;

    /// The current destination type.
    QualType DestType;

    RebuildUnknownAnyExpr(Sema &S, QualType CastType)
      : S(S), DestType(CastType) {}

    ExprResult VisitStmt(Stmt *S) {
      llvm_unreachable("unexpected statement!");
    }

    ExprResult VisitExpr(Expr *E) {
      S.Diag(E->getExprLoc(), diag::err_unsupported_unknown_any_expr)
        << E->getSourceRange();
      return ExprError();
    }

    ExprResult VisitCallExpr(CallExpr *E);
    ExprResult VisitObjCMessageExpr(ObjCMessageExpr *E);

    /// Rebuild an expression which simply semantically wraps another
    /// expression which it shares the type and value kind of.
    template <class T> ExprResult rebuildSugarExpr(T *E) {
      ExprResult SubResult = Visit(E->getSubExpr());
      if (SubResult.isInvalid()) return ExprError();
      Expr *SubExpr = SubResult.get();
      E->setSubExpr(SubExpr);
      E->setType(SubExpr->getType());
      E->setValueKind(SubExpr->getValueKind());
      assert(E->getObjectKind() == OK_Ordinary);
      return E;
    }

    ExprResult VisitParenExpr(ParenExpr *E) {
      return rebuildSugarExpr(E);
    }

    ExprResult VisitUnaryExtension(UnaryOperator *E) {
      return rebuildSugarExpr(E);
    }

    ExprResult VisitUnaryAddrOf(UnaryOperator *E) {
      const PointerType *Ptr = DestType->getAs<PointerType>();
      if (!Ptr) {
        S.Diag(E->getOperatorLoc(), diag::err_unknown_any_addrof)
          << E->getSourceRange();
        return ExprError();
      }

      if (isa<CallExpr>(E->getSubExpr())) {
        S.Diag(E->getOperatorLoc(), diag::err_unknown_any_addrof_call)
          << E->getSourceRange();
        return ExprError();
      }

      assert(E->isPRValue());
      assert(E->getObjectKind() == OK_Ordinary);
      E->setType(DestType);

      // Build the sub-expression as if it were an object of the pointee type.
      DestType = Ptr->getPointeeType();
      ExprResult SubResult = Visit(E->getSubExpr());
      if (SubResult.isInvalid()) return ExprError();
      E->setSubExpr(SubResult.get());
      return E;
    }

    ExprResult VisitImplicitCastExpr(ImplicitCastExpr *E);

    ExprResult resolveDecl(Expr *E, ValueDecl *VD);

    ExprResult VisitMemberExpr(MemberExpr *E) {
      return resolveDecl(E, E->getMemberDecl());
    }

    ExprResult VisitDeclRefExpr(DeclRefExpr *E) {
      return resolveDecl(E, E->getDecl());
    }
  };
}

/// Rebuilds a call expression which yielded __unknown_anytype.
ExprResult RebuildUnknownAnyExpr::VisitCallExpr(CallExpr *E) {
  Expr *CalleeExpr = E->getCallee();

  enum FnKind {
    FK_MemberFunction,
    FK_FunctionPointer,
    FK_BlockPointer
  };

  FnKind Kind;
  QualType CalleeType = CalleeExpr->getType();
  if (CalleeType == S.Context.BoundMemberTy) {
    assert(isa<CXXMemberCallExpr>(E) || isa<CXXOperatorCallExpr>(E));
    Kind = FK_MemberFunction;
    CalleeType = Expr::findBoundMemberType(CalleeExpr);
  } else if (const PointerType *Ptr = CalleeType->getAs<PointerType>()) {
    CalleeType = Ptr->getPointeeType();
    Kind = FK_FunctionPointer;
  } else {
    CalleeType = CalleeType->castAs<BlockPointerType>()->getPointeeType();
    Kind = FK_BlockPointer;
  }
  const FunctionType *FnType = CalleeType->castAs<FunctionType>();

  // Verify that this is a legal result type of a function.
  if ((DestType->isArrayType() && !S.getLangOpts().allowArrayReturnTypes()) ||
      DestType->isFunctionType()) {
    unsigned diagID = diag::err_func_returning_array_function;
    if (Kind == FK_BlockPointer)
      diagID = diag::err_block_returning_array_function;

    S.Diag(E->getExprLoc(), diagID)
      << DestType->isFunctionType() << DestType;
    return ExprError();
  }

  // Otherwise, go ahead and set DestType as the call's result.
  E->setType(DestType.getNonLValueExprType(S.Context));
  E->setValueKind(Expr::getValueKindForType(DestType));
  assert(E->getObjectKind() == OK_Ordinary);

  // Rebuild the function type, replacing the result type with DestType.
  const FunctionProtoType *Proto = dyn_cast<FunctionProtoType>(FnType);
  if (Proto) {
    // __unknown_anytype(...) is a special case used by the debugger when
    // it has no idea what a function's signature is.
    //
    // We want to build this call essentially under the K&R
    // unprototyped rules, but making a FunctionNoProtoType in C++
    // would foul up all sorts of assumptions.  However, we cannot
    // simply pass all arguments as variadic arguments, nor can we
    // portably just call the function under a non-variadic type; see
    // the comment on IR-gen's TargetInfo::isNoProtoCallVariadic.
    // However, it turns out that in practice it is generally safe to
    // call a function declared as "A foo(B,C,D);" under the prototype
    // "A foo(B,C,D,...);".  The only known exception is with the
    // Windows ABI, where any variadic function is implicitly cdecl
    // regardless of its normal CC.  Therefore we change the parameter
    // types to match the types of the arguments.
    //
    // This is a hack, but it is far superior to moving the
    // corresponding target-specific code from IR-gen to Sema/AST.

    ArrayRef<QualType> ParamTypes = Proto->getParamTypes();
    SmallVector<QualType, 8> ArgTypes;
    if (ParamTypes.empty() && Proto->isVariadic()) { // the special case
      ArgTypes.reserve(E->getNumArgs());
      for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
        ArgTypes.push_back(S.Context.getReferenceQualifiedType(E->getArg(i)));
      }
      ParamTypes = ArgTypes;
    }
    DestType = S.Context.getFunctionType(DestType, ParamTypes,
                                         Proto->getExtProtoInfo());
  } else {
    DestType = S.Context.getFunctionNoProtoType(DestType,
                                                FnType->getExtInfo());
  }

  // Rebuild the appropriate pointer-to-function type.
  switch (Kind) {
  case FK_MemberFunction:
    // Nothing to do.
    break;

  case FK_FunctionPointer:
    DestType = S.Context.getPointerType(DestType);
    break;

  case FK_BlockPointer:
    DestType = S.Context.getBlockPointerType(DestType);
    break;
  }

  // Finally, we can recurse.
  ExprResult CalleeResult = Visit(CalleeExpr);
  if (!CalleeResult.isUsable()) return ExprError();
  E->setCallee(CalleeResult.get());

  // Bind a temporary if necessary.
  return S.MaybeBindToTemporary(E);
}

ExprResult RebuildUnknownAnyExpr::VisitObjCMessageExpr(ObjCMessageExpr *E) {
  // Verify that this is a legal result type of a call.
  if (DestType->isArrayType() || DestType->isFunctionType()) {
    S.Diag(E->getExprLoc(), diag::err_func_returning_array_function)
      << DestType->isFunctionType() << DestType;
    return ExprError();
  }

  // Rewrite the method result type if available.
  if (ObjCMethodDecl *Method = E->getMethodDecl()) {
    assert(Method->getReturnType() == S.Context.UnknownAnyTy);
    Method->setReturnType(DestType);
  }

  // Change the type of the message.
  E->setType(DestType.getNonReferenceType());
  E->setValueKind(Expr::getValueKindForType(DestType));

  return S.MaybeBindToTemporary(E);
}

ExprResult RebuildUnknownAnyExpr::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  // The only case we should ever see here is a function-to-pointer decay.
  if (E->getCastKind() == CK_FunctionToPointerDecay) {
    assert(E->isPRValue());
    assert(E->getObjectKind() == OK_Ordinary);

    E->setType(DestType);

    // Rebuild the sub-expression as the pointee (function) type.
    DestType = DestType->castAs<PointerType>()->getPointeeType();

    ExprResult Result = Visit(E->getSubExpr());
    if (!Result.isUsable()) return ExprError();

    E->setSubExpr(Result.get());
    return E;
  } else if (E->getCastKind() == CK_LValueToRValue) {
    assert(E->isPRValue());
    assert(E->getObjectKind() == OK_Ordinary);

    assert(isa<BlockPointerType>(E->getType()));

    E->setType(DestType);

    // The sub-expression has to be a lvalue reference, so rebuild it as such.
    DestType = S.Context.getLValueReferenceType(DestType);

    ExprResult Result = Visit(E->getSubExpr());
    if (!Result.isUsable()) return ExprError();

    E->setSubExpr(Result.get());
    return E;
  } else {
    llvm_unreachable("Unhandled cast type!");
  }
}

ExprResult RebuildUnknownAnyExpr::resolveDecl(Expr *E, ValueDecl *VD) {
  ExprValueKind ValueKind = VK_LValue;
  QualType Type = DestType;

  // We know how to make this work for certain kinds of decls:

  //  - functions
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(VD)) {
    if (const PointerType *Ptr = Type->getAs<PointerType>()) {
      DestType = Ptr->getPointeeType();
      ExprResult Result = resolveDecl(E, VD);
      if (Result.isInvalid()) return ExprError();
      return S.ImpCastExprToType(Result.get(), Type, CK_FunctionToPointerDecay,
                                 VK_PRValue);
    }

    if (!Type->isFunctionType()) {
      S.Diag(E->getExprLoc(), diag::err_unknown_any_function)
        << VD << E->getSourceRange();
      return ExprError();
    }
    if (const FunctionProtoType *FT = Type->getAs<FunctionProtoType>()) {
      // We must match the FunctionDecl's type to the hack introduced in
      // RebuildUnknownAnyExpr::VisitCallExpr to vararg functions of unknown
      // type. See the lengthy commentary in that routine.
      QualType FDT = FD->getType();
      const FunctionType *FnType = FDT->castAs<FunctionType>();
      const FunctionProtoType *Proto = dyn_cast_or_null<FunctionProtoType>(FnType);
      DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E);
      if (DRE && Proto && Proto->getParamTypes().empty() && Proto->isVariadic()) {
        SourceLocation Loc = FD->getLocation();
        FunctionDecl *NewFD = FunctionDecl::Create(
            S.Context, FD->getDeclContext(), Loc, Loc,
            FD->getNameInfo().getName(), DestType, FD->getTypeSourceInfo(),
            SC_None, S.getCurFPFeatures().isFPConstrained(),
            false /*isInlineSpecified*/, FD->hasPrototype(),
            /*ConstexprKind*/ ConstexprSpecKind::Unspecified);

        if (FD->getQualifier())
          NewFD->setQualifierInfo(FD->getQualifierLoc());

        SmallVector<ParmVarDecl*, 16> Params;
        for (const auto &AI : FT->param_types()) {
          ParmVarDecl *Param =
            S.BuildParmVarDeclForTypedef(FD, Loc, AI);
          Param->setScopeInfo(0, Params.size());
          Params.push_back(Param);
        }
        NewFD->setParams(Params);
        DRE->setDecl(NewFD);
        VD = DRE->getDecl();
      }
    }

    if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD))
      if (MD->isInstance()) {
        ValueKind = VK_PRValue;
        Type = S.Context.BoundMemberTy;
      }

    // Function references aren't l-values in C.
    if (!S.getLangOpts().CPlusPlus)
      ValueKind = VK_PRValue;

  //  - variables
  } else if (isa<VarDecl>(VD)) {
    if (const ReferenceType *RefTy = Type->getAs<ReferenceType>()) {
      Type = RefTy->getPointeeType();
    } else if (Type->isFunctionType()) {
      S.Diag(E->getExprLoc(), diag::err_unknown_any_var_function_type)
        << VD << E->getSourceRange();
      return ExprError();
    }

  //  - nothing else
  } else {
    S.Diag(E->getExprLoc(), diag::err_unsupported_unknown_any_decl)
      << VD << E->getSourceRange();
    return ExprError();
  }

  // Modifying the declaration like this is friendly to IR-gen but
  // also really dangerous.
  VD->setType(DestType);
  E->setType(Type);
  E->setValueKind(ValueKind);
  return E;
}

ExprResult Sema::checkUnknownAnyCast(SourceRange TypeRange, QualType CastType,
                                     Expr *CastExpr, CastKind &CastKind,
                                     ExprValueKind &VK, CXXCastPath &Path) {
  // The type we're casting to must be either void or complete.
  if (!CastType->isVoidType() &&
      RequireCompleteType(TypeRange.getBegin(), CastType,
                          diag::err_typecheck_cast_to_incomplete))
    return ExprError();

  // Rewrite the casted expression from scratch.
  ExprResult result = RebuildUnknownAnyExpr(*this, CastType).Visit(CastExpr);
  if (!result.isUsable()) return ExprError();

  CastExpr = result.get();
  VK = CastExpr->getValueKind();
  CastKind = CK_NoOp;

  return CastExpr;
}

ExprResult Sema::forceUnknownAnyToType(Expr *E, QualType ToType) {
  return RebuildUnknownAnyExpr(*this, ToType).Visit(E);
}

ExprResult Sema::checkUnknownAnyArg(SourceLocation callLoc,
                                    Expr *arg, QualType &paramType) {
  // If the syntactic form of the argument is not an explicit cast of
  // any sort, just do default argument promotion.
  ExplicitCastExpr *castArg = dyn_cast<ExplicitCastExpr>(arg->IgnoreParens());
  if (!castArg) {
    ExprResult result = DefaultArgumentPromotion(arg);
    if (result.isInvalid()) return ExprError();
    paramType = result.get()->getType();
    return result;
  }

  // Otherwise, use the type that was written in the explicit cast.
  assert(!arg->hasPlaceholderType());
  paramType = castArg->getTypeAsWritten();

  // Copy-initialize a parameter of that type.
  InitializedEntity entity =
    InitializedEntity::InitializeParameter(Context, paramType,
                                           /*consumed*/ false);
  return PerformCopyInitialization(entity, callLoc, arg);
}

static ExprResult diagnoseUnknownAnyExpr(Sema &S, Expr *E) {
  Expr *orig = E;
  unsigned diagID = diag::err_uncasted_use_of_unknown_any;
  while (true) {
    E = E->IgnoreParenImpCasts();
    if (CallExpr *call = dyn_cast<CallExpr>(E)) {
      E = call->getCallee();
      diagID = diag::err_uncasted_call_of_unknown_any;
    } else {
      break;
    }
  }

  SourceLocation loc;
  NamedDecl *d;
  if (DeclRefExpr *ref = dyn_cast<DeclRefExpr>(E)) {
    loc = ref->getLocation();
    d = ref->getDecl();
  } else if (MemberExpr *mem = dyn_cast<MemberExpr>(E)) {
    loc = mem->getMemberLoc();
    d = mem->getMemberDecl();
  } else if (ObjCMessageExpr *msg = dyn_cast<ObjCMessageExpr>(E)) {
    diagID = diag::err_uncasted_call_of_unknown_any;
    loc = msg->getSelectorStartLoc();
    d = msg->getMethodDecl();
    if (!d) {
      S.Diag(loc, diag::err_uncasted_send_to_unknown_any_method)
        << static_cast<unsigned>(msg->isClassMessage()) << msg->getSelector()
        << orig->getSourceRange();
      return ExprError();
    }
  } else {
    S.Diag(E->getExprLoc(), diag::err_unsupported_unknown_any_expr)
      << E->getSourceRange();
    return ExprError();
  }

  S.Diag(loc, diagID) << d << orig->getSourceRange();

  // Never recoverable.
  return ExprError();
}

ExprResult Sema::CheckPlaceholderExpr(Expr *E) {
  const BuiltinType *placeholderType = E->getType()->getAsPlaceholderType();
  if (!placeholderType) return E;

  switch (placeholderType->getKind()) {
  case BuiltinType::UnresolvedTemplate: {
    auto *ULE = cast<UnresolvedLookupExpr>(E->IgnoreParens());
    const DeclarationNameInfo &NameInfo = ULE->getNameInfo();
    // There's only one FoundDecl for UnresolvedTemplate type. See
    // BuildTemplateIdExpr.
    NamedDecl *Temp = *ULE->decls_begin();
    const bool IsTypeAliasTemplateDecl = isa<TypeAliasTemplateDecl>(Temp);

    NestedNameSpecifier NNS = ULE->getQualifierLoc().getNestedNameSpecifier();
    // FIXME: AssumedTemplate is not very appropriate for error recovery here,
    // as it models only the unqualified-id case, where this case can clearly be
    // qualified. Thus we can't just qualify an assumed template.
    TemplateName TN;
    if (auto *TD = dyn_cast<TemplateDecl>(Temp))
      TN = Context.getQualifiedTemplateName(NNS, ULE->hasTemplateKeyword(),
                                            TemplateName(TD));
    else
      TN = Context.getAssumedTemplateName(NameInfo.getName());

    Diag(NameInfo.getLoc(), diag::err_template_kw_refers_to_type_template)
        << TN << ULE->getSourceRange() << IsTypeAliasTemplateDecl;
    Diag(Temp->getLocation(), diag::note_referenced_type_template)
        << IsTypeAliasTemplateDecl;

    TemplateArgumentListInfo TAL(ULE->getLAngleLoc(), ULE->getRAngleLoc());
    bool HasAnyDependentTA = false;
    for (const TemplateArgumentLoc &Arg : ULE->template_arguments()) {
      HasAnyDependentTA |= Arg.getArgument().isDependent();
      TAL.addArgument(Arg);
    }

    QualType TST;
    {
      SFINAETrap Trap(*this);
      TST = CheckTemplateIdType(
          ElaboratedTypeKeyword::None, TN, NameInfo.getBeginLoc(), TAL,
          /*Scope=*/nullptr, /*ForNestedNameSpecifier=*/false);
    }
    if (TST.isNull())
      TST = Context.getTemplateSpecializationType(
          ElaboratedTypeKeyword::None, TN, ULE->template_arguments(),
          /*CanonicalArgs=*/{},
          HasAnyDependentTA ? Context.DependentTy : Context.IntTy);
    return CreateRecoveryExpr(NameInfo.getBeginLoc(), NameInfo.getEndLoc(), {},
                              TST);
  }

  // Overloaded expressions.
  case BuiltinType::Overload: {
    // Try to resolve a single function template specialization.
    // This is obligatory.
    ExprResult Result = E;
    if (ResolveAndFixSingleFunctionTemplateSpecialization(Result, false))
      return Result;

    // No guarantees that ResolveAndFixSingleFunctionTemplateSpecialization
    // leaves Result unchanged on failure.
    Result = E;
    if (resolveAndFixAddressOfSingleOverloadCandidate(Result))
      return Result;

    // If that failed, try to recover with a call.
    tryToRecoverWithCall(Result, PDiag(diag::err_ovl_unresolvable),
                         /*complain*/ true);
    return Result;
  }

  // Bound member functions.
  case BuiltinType::BoundMember: {
    ExprResult result = E;
    const Expr *BME = E->IgnoreParens();
    PartialDiagnostic PD = PDiag(diag::err_bound_member_function);
    // Try to give a nicer diagnostic if it is a bound member that we recognize.
    if (isa<CXXPseudoDestructorExpr>(BME)) {
      PD = PDiag(diag::err_dtor_expr_without_call) << /*pseudo-destructor*/ 1;
    } else if (const auto *ME = dyn_cast<MemberExpr>(BME)) {
      if (ME->getMemberNameInfo().getName().getNameKind() ==
          DeclarationName::CXXDestructorName)
        PD = PDiag(diag::err_dtor_expr_without_call) << /*destructor*/ 0;
    }
    tryToRecoverWithCall(result, PD,
                         /*complain*/ true);
    return result;
  }

  // ARC unbridged casts.
  case BuiltinType::ARCUnbridgedCast: {
    Expr *realCast = ObjC().stripARCUnbridgedCast(E);
    ObjC().diagnoseARCUnbridgedCast(realCast);
    return realCast;
  }

  // Expressions of unknown type.
  case BuiltinType::UnknownAny:
    return diagnoseUnknownAnyExpr(*this, E);

  // Pseudo-objects.
  case BuiltinType::PseudoObject:
    return PseudoObject().checkRValue(E);

  case BuiltinType::BuiltinFn: {
    // Accept __noop without parens by implicitly converting it to a call expr.
    auto *DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts());
    if (DRE) {
      auto *FD = cast<FunctionDecl>(DRE->getDecl());
      unsigned BuiltinID = FD->getBuiltinID();
      if (BuiltinID == Builtin::BI__noop) {
        E = ImpCastExprToType(E, Context.getPointerType(FD->getType()),
                              CK_BuiltinFnToFnPtr)
                .get();
        return CallExpr::Create(Context, E, /*Args=*/{}, Context.IntTy,
                                VK_PRValue, SourceLocation(),
                                FPOptionsOverride());
      }

      if (Context.BuiltinInfo.isInStdNamespace(BuiltinID)) {
        // Any use of these other than a direct call is ill-formed as of C++20,
        // because they are not addressable functions. In earlier language
        // modes, warn and force an instantiation of the real body.
        Diag(E->getBeginLoc(),
             getLangOpts().CPlusPlus20
                 ? diag::err_use_of_unaddressable_function
                 : diag::warn_cxx20_compat_use_of_unaddressable_function);
        if (FD->isImplicitlyInstantiable()) {
          // Require a definition here because a normal attempt at
          // instantiation for a builtin will be ignored, and we won't try
          // again later. We assume that the definition of the template
          // precedes this use.
          InstantiateFunctionDefinition(E->getBeginLoc(), FD,
                                        /*Recursive=*/false,
                                        /*DefinitionRequired=*/true,
                                        /*AtEndOfTU=*/false);
        }
        // Produce a properly-typed reference to the function.
        CXXScopeSpec SS;
        SS.Adopt(DRE->getQualifierLoc());
        TemplateArgumentListInfo TemplateArgs;
        DRE->copyTemplateArgumentsInto(TemplateArgs);
        return BuildDeclRefExpr(
            FD, FD->getType(), VK_LValue, DRE->getNameInfo(),
            DRE->hasQualifier() ? &SS : nullptr, DRE->getFoundDecl(),
            DRE->getTemplateKeywordLoc(),
            DRE->hasExplicitTemplateArgs() ? &TemplateArgs : nullptr);
      }
    }

    Diag(E->getBeginLoc(), diag::err_builtin_fn_use);
    return ExprError();
  }

  case BuiltinType::IncompleteMatrixIdx: {
    auto *MS = cast<MatrixSubscriptExpr>(E->IgnoreParens());
    // At this point, we know there was no second [] to complete the operator.
    // In HLSL, treat "m[row]" as selecting a row lane of column sized vector.
    if (getLangOpts().HLSL) {
      return CreateBuiltinMatrixSingleSubscriptExpr(
          MS->getBase(), MS->getRowIdx(), E->getExprLoc());
    }
    Diag(MS->getRowIdx()->getBeginLoc(), diag::err_matrix_incomplete_index);
    return ExprError();
  }

  // Expressions of unknown type.
  case BuiltinType::ArraySection:
    // If we've already diagnosed something on the array section type, we
    // shouldn't need to do any further diagnostic here.
    if (!E->containsErrors())
      Diag(E->getBeginLoc(), diag::err_array_section_use)
          << cast<ArraySectionExpr>(E->IgnoreParens())->isOMPArraySection();
    return ExprError();

  // Expressions of unknown type.
  case BuiltinType::OMPArrayShaping:
    return ExprError(Diag(E->getBeginLoc(), diag::err_omp_array_shaping_use));

  case BuiltinType::OMPIterator:
    return ExprError(Diag(E->getBeginLoc(), diag::err_omp_iterator_use));

  // Everything else should be impossible.
#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix) \
  case BuiltinType::Id:
#include "clang/Basic/OpenCLImageTypes.def"
#define EXT_OPAQUE_TYPE(ExtType, Id, Ext) \
  case BuiltinType::Id:
#include "clang/Basic/OpenCLExtensionTypes.def"
#define SVE_TYPE(Name, Id, SingletonId) \
  case BuiltinType::Id:
#include "clang/Basic/AArch64ACLETypes.def"
#define PPC_VECTOR_TYPE(Name, Id, Size) \
  case BuiltinType::Id:
#include "clang/Basic/PPCTypes.def"
#define RVV_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/RISCVVTypes.def"
#define WASM_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/WebAssemblyReferenceTypes.def"
#define AMDGPU_TYPE(Name, Id, SingletonId, Width, Align) case BuiltinType::Id:
#include "clang/Basic/AMDGPUTypes.def"
#define HLSL_INTANGIBLE_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/HLSLIntangibleTypes.def"
#define BUILTIN_TYPE(Id, SingletonId) case BuiltinType::Id:
#define PLACEHOLDER_TYPE(Id, SingletonId)
#include "clang/AST/BuiltinTypes.def"
    break;
  }

  llvm_unreachable("invalid placeholder type!");
}

bool Sema::CheckCaseExpression(Expr *E) {
  if (E->isTypeDependent())
    return true;
  if (E->isValueDependent() || E->isIntegerConstantExpr(Context))
    return E->getType()->isIntegralOrEnumerationType();
  return false;
}

ExprResult Sema::CreateRecoveryExpr(SourceLocation Begin, SourceLocation End,
                                    ArrayRef<Expr *> SubExprs, QualType T) {
  if (!Context.getLangOpts().RecoveryAST)
    return ExprError();

  if (isSFINAEContext())
    return ExprError();

  if (T.isNull() || T->isUndeducedType() ||
      !Context.getLangOpts().RecoveryASTType)
    // We don't know the concrete type, fallback to dependent type.
    T = Context.DependentTy;

  return RecoveryExpr::Create(Context, T, Begin, End, SubExprs);
}
