//===- SemaSPIRV.cpp - Semantic Analysis for SPIRV constructs--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for SPIRV constructs.
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaSPIRV.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/Sema.h"

// SPIR-V enumerants. Enums have only the required entries, see SPIR-V specs for
// values.
// FIXME: either use the SPIRV-Headers or generate a custom header using the
// grammar (like done with MLIR).
namespace spirv {
enum class StorageClass : int {
  Workgroup = 4,
  CrossWorkgroup = 5,
  Function = 7
};
}

namespace clang {

SemaSPIRV::SemaSPIRV(Sema &S) : SemaBase(S) {}

static bool CheckAllArgsHaveSameType(Sema *S, CallExpr *TheCall) {
  assert(TheCall->getNumArgs() > 1);
  QualType ArgTy0 = TheCall->getArg(0)->getType();

  for (unsigned I = 1, N = TheCall->getNumArgs(); I < N; ++I) {
    if (!S->getASTContext().hasSameUnqualifiedType(
            ArgTy0, TheCall->getArg(I)->getType())) {
      S->Diag(TheCall->getBeginLoc(), diag::err_vec_builtin_incompatible_vector)
          << TheCall->getDirectCallee() << /*useAllTerminology*/ true
          << SourceRange(TheCall->getArg(0)->getBeginLoc(),
                         TheCall->getArg(N - 1)->getEndLoc());
      return true;
    }
  }
  return false;
}

static bool CheckAllArgTypesAreCorrect(
    Sema *S, CallExpr *TheCall,
    llvm::ArrayRef<
        llvm::function_ref<bool(Sema *, SourceLocation, int, QualType)>>
        Checks) {
  unsigned NumArgs = TheCall->getNumArgs();
  assert(Checks.size() == NumArgs &&
         "Wrong number of checks for Number of args.");
  // Apply each check to the corresponding argument
  for (unsigned I = 0; I < NumArgs; ++I) {
    Expr *Arg = TheCall->getArg(I);
    if (Checks[I](S, Arg->getBeginLoc(), I + 1, Arg->getType()))
      return true;
  }
  return false;
}

static bool CheckFloatOrHalfRepresentation(Sema *S, SourceLocation Loc,
                                           int ArgOrdinal,
                                           clang::QualType PassedType) {
  clang::QualType BaseType =
      PassedType->isVectorType()
          ? PassedType->castAs<clang::VectorType>()->getElementType()
          : PassedType;
  if (!BaseType->isHalfType() && !BaseType->isFloat16Type() &&
      !BaseType->isFloat32Type())
    return S->Diag(Loc, diag::err_builtin_invalid_arg_type)
           << ArgOrdinal << /* scalar or vector of */ 5 << /* no int */ 0
           << /* half or float */ 2 << PassedType;
  return false;
}

static bool CheckFloatOrHalfScalarRepresentation(Sema *S, SourceLocation Loc,
                                                 int ArgOrdinal,
                                                 clang::QualType PassedType) {
  if (!PassedType->isHalfType() && !PassedType->isFloat16Type() &&
      !PassedType->isFloat32Type())
    return S->Diag(Loc, diag::err_builtin_invalid_arg_type)
           << ArgOrdinal << /* scalar */ 1 << /* no int */ 0
           << /* half or float */ 2 << PassedType;
  return false;
}

static std::optional<int>
processConstant32BitIntArgument(Sema &SemaRef, CallExpr *Call, int Argument) {
  ExprResult Arg =
      SemaRef.DefaultFunctionArrayLvalueConversion(Call->getArg(Argument));
  if (Arg.isInvalid())
    return true;
  Call->setArg(Argument, Arg.get());

  const Expr *IntArg = Arg.get();
  SmallVector<PartialDiagnosticAt, 8> Notes;
  Expr::EvalResult Eval;
  Eval.Diag = &Notes;
  if ((!IntArg->EvaluateAsConstantExpr(Eval, SemaRef.getASTContext())) ||
      !Eval.Val.isInt() || Eval.Val.getInt().getBitWidth() > 32) {
    SemaRef.Diag(IntArg->getBeginLoc(), diag::err_spirv_enum_not_int)
        << 0 << IntArg->getSourceRange();
    for (const PartialDiagnosticAt &PDiag : Notes)
      SemaRef.Diag(PDiag.first, PDiag.second);
    return true;
  }
  return {Eval.Val.getInt().getZExtValue()};
}

static bool checkGenericCastToPtr(Sema &SemaRef, CallExpr *Call) {
  if (SemaRef.checkArgCount(Call, 2))
    return true;

  {
    ExprResult Arg =
        SemaRef.DefaultFunctionArrayLvalueConversion(Call->getArg(0));
    if (Arg.isInvalid())
      return true;
    Call->setArg(0, Arg.get());

    QualType Ty = Arg.get()->getType();
    const auto *PtrTy = Ty->getAs<PointerType>();
    auto AddressSpaceNotInGeneric = [&](LangAS AS) {
      if (SemaRef.LangOpts.OpenCL)
        return AS != LangAS::opencl_generic;
      return AS != LangAS::Default;
    };
    if (!PtrTy ||
        AddressSpaceNotInGeneric(PtrTy->getPointeeType().getAddressSpace())) {
      SemaRef.Diag(Arg.get()->getBeginLoc(),
                   diag::err_spirv_builtin_generic_cast_invalid_arg)
          << Call->getSourceRange();
      return true;
    }
  }

  spirv::StorageClass StorageClass;
  if (std::optional<int> SCInt =
          processConstant32BitIntArgument(SemaRef, Call, 1);
      SCInt.has_value()) {
    StorageClass = static_cast<spirv::StorageClass>(SCInt.value());
    if (StorageClass != spirv::StorageClass::CrossWorkgroup &&
        StorageClass != spirv::StorageClass::Workgroup &&
        StorageClass != spirv::StorageClass::Function) {
      SemaRef.Diag(Call->getArg(1)->getBeginLoc(),
                   diag::err_spirv_enum_not_valid)
          << 0 << Call->getArg(1)->getSourceRange();
      return true;
    }
  } else {
    return true;
  }
  auto RT = Call->getArg(0)->getType();
  RT = RT->getPointeeType();
  auto Qual = RT.getQualifiers();
  LangAS AddrSpace;
  switch (StorageClass) {
  case spirv::StorageClass::CrossWorkgroup:
    AddrSpace =
        SemaRef.LangOpts.isSYCL() ? LangAS::sycl_global : LangAS::opencl_global;
    break;
  case spirv::StorageClass::Workgroup:
    AddrSpace =
        SemaRef.LangOpts.isSYCL() ? LangAS::sycl_local : LangAS::opencl_local;
    break;
  case spirv::StorageClass::Function:
    AddrSpace = SemaRef.LangOpts.isSYCL() ? LangAS::sycl_private
                                          : LangAS::opencl_private;
    break;
  }
  Qual.setAddressSpace(AddrSpace);
  Call->setType(SemaRef.getASTContext().getPointerType(
      SemaRef.getASTContext().getQualifiedType(RT.getUnqualifiedType(), Qual)));

  return false;
}

bool SemaSPIRV::CheckSPIRVBuiltinFunctionCall(const TargetInfo &TI,
                                              unsigned BuiltinID,
                                              CallExpr *TheCall) {
  if (BuiltinID >= SPIRV::FirstVKBuiltin && BuiltinID <= SPIRV::LastVKBuiltin &&
      TI.getTriple().getArch() != llvm::Triple::spirv) {
    SemaRef.Diag(TheCall->getBeginLoc(), diag::err_spirv_invalid_target) << 0;
    return true;
  }
  if (BuiltinID >= SPIRV::FirstCLBuiltin && BuiltinID <= SPIRV::LastTSBuiltin &&
      TI.getTriple().getArch() != llvm::Triple::spirv32 &&
      TI.getTriple().getArch() != llvm::Triple::spirv64) {
    SemaRef.Diag(TheCall->getBeginLoc(), diag::err_spirv_invalid_target) << 1;
    return true;
  }

  switch (BuiltinID) {
  case SPIRV::BI__builtin_spirv_distance: {
    if (SemaRef.checkArgCount(TheCall, 2))
      return true;

    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    auto *VTyA = ArgTyA->getAs<VectorType>();
    if (VTyA == nullptr) {
      SemaRef.Diag(A.get()->getBeginLoc(),
                   diag::err_typecheck_convert_incompatible)
          << ArgTyA
          << SemaRef.Context.getVectorType(ArgTyA, 2, VectorKind::Generic) << 1
          << 0 << 0;
      return true;
    }

    ExprResult B = TheCall->getArg(1);
    QualType ArgTyB = B.get()->getType();
    auto *VTyB = ArgTyB->getAs<VectorType>();
    if (VTyB == nullptr) {
      SemaRef.Diag(A.get()->getBeginLoc(),
                   diag::err_typecheck_convert_incompatible)
          << ArgTyB
          << SemaRef.Context.getVectorType(ArgTyB, 2, VectorKind::Generic) << 1
          << 0 << 0;
      return true;
    }

    QualType RetTy = VTyA->getElementType();
    TheCall->setType(RetTy);
    break;
  }
  case SPIRV::BI__builtin_spirv_length: {
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;
    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    auto *VTy = ArgTyA->getAs<VectorType>();
    if (VTy == nullptr) {
      SemaRef.Diag(A.get()->getBeginLoc(),
                   diag::err_typecheck_convert_incompatible)
          << ArgTyA
          << SemaRef.Context.getVectorType(ArgTyA, 2, VectorKind::Generic) << 1
          << 0 << 0;
      return true;
    }
    QualType RetTy = VTy->getElementType();
    TheCall->setType(RetTy);
    break;
  }
  case SPIRV::BI__builtin_spirv_reflect: {
    if (SemaRef.checkArgCount(TheCall, 2))
      return true;

    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    auto *VTyA = ArgTyA->getAs<VectorType>();
    if (VTyA == nullptr) {
      SemaRef.Diag(A.get()->getBeginLoc(),
                   diag::err_typecheck_convert_incompatible)
          << ArgTyA
          << SemaRef.Context.getVectorType(ArgTyA, 2, VectorKind::Generic) << 1
          << 0 << 0;
      return true;
    }

    ExprResult B = TheCall->getArg(1);
    QualType ArgTyB = B.get()->getType();
    auto *VTyB = ArgTyB->getAs<VectorType>();
    if (VTyB == nullptr) {
      SemaRef.Diag(A.get()->getBeginLoc(),
                   diag::err_typecheck_convert_incompatible)
          << ArgTyB
          << SemaRef.Context.getVectorType(ArgTyB, 2, VectorKind::Generic) << 1
          << 0 << 0;
      return true;
    }

    QualType RetTy = ArgTyA;
    TheCall->setType(RetTy);
    break;
  }
  case SPIRV::BI__builtin_spirv_refract: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;

    llvm::function_ref<bool(Sema *, SourceLocation, int, QualType)>
        ChecksArr[] = {CheckFloatOrHalfRepresentation,
                       CheckFloatOrHalfRepresentation,
                       CheckFloatOrHalfScalarRepresentation};
    if (CheckAllArgTypesAreCorrect(&SemaRef, TheCall,
                                   llvm::ArrayRef(ChecksArr)))
      return true;
    // Check that first two arguments are vectors/scalars of the same type
    QualType Arg0Type = TheCall->getArg(0)->getType();
    if (!SemaRef.getASTContext().hasSameUnqualifiedType(
            Arg0Type, TheCall->getArg(1)->getType()))
      return SemaRef.Diag(TheCall->getBeginLoc(),
                          diag::err_vec_builtin_incompatible_vector)
             << TheCall->getDirectCallee() << /* first two */ 0
             << SourceRange(TheCall->getArg(0)->getBeginLoc(),
                            TheCall->getArg(1)->getEndLoc());

    // Check that scalar type of 3rd arg is same as base type of first two args
    clang::QualType BaseType =
        Arg0Type->isVectorType()
            ? Arg0Type->castAs<clang::VectorType>()->getElementType()
            : Arg0Type;
    if (!SemaRef.getASTContext().hasSameUnqualifiedType(
            BaseType, TheCall->getArg(2)->getType()))
      return SemaRef.Diag(TheCall->getBeginLoc(),
                          diag::err_hlsl_builtin_scalar_vector_mismatch)
             << /* all */ 0 << TheCall->getDirectCallee() << Arg0Type
             << TheCall->getArg(2)->getType();

    QualType RetTy = TheCall->getArg(0)->getType();
    TheCall->setType(RetTy);
    break;
  }
  case SPIRV::BI__builtin_spirv_smoothstep: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;

    // Check if first argument has floating representation
    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    if (!ArgTyA->hasFloatingRepresentation()) {
      SemaRef.Diag(A.get()->getBeginLoc(), diag::err_builtin_invalid_arg_type)
          << /* ordinal */ 1 << /* scalar or vector */ 5 << /* no int */ 0
          << /* fp */ 1 << ArgTyA;
      return true;
    }

    if (CheckAllArgsHaveSameType(&SemaRef, TheCall))
      return true;

    QualType RetTy = ArgTyA;
    TheCall->setType(RetTy);
    break;
  }
  case SPIRV::BI__builtin_spirv_faceforward: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;

    // Check if first argument has floating representation
    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    if (!ArgTyA->hasFloatingRepresentation()) {
      SemaRef.Diag(A.get()->getBeginLoc(), diag::err_builtin_invalid_arg_type)
          << /* ordinal */ 1 << /* scalar or vector */ 5 << /* no int */ 0
          << /* fp */ 1 << ArgTyA;
      return true;
    }

    if (CheckAllArgsHaveSameType(&SemaRef, TheCall))
      return true;

    QualType RetTy = ArgTyA;
    TheCall->setType(RetTy);
    break;
  }
  case SPIRV::BI__builtin_spirv_generic_cast_to_ptr_explicit: {
    return checkGenericCastToPtr(SemaRef, TheCall);
  }
  case SPIRV::BI__builtin_spirv_fwidth: {
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;

    // Check if first argument has floating representation
    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    if (!ArgTyA->hasFloatingRepresentation()) {
      SemaRef.Diag(A.get()->getBeginLoc(), diag::err_builtin_invalid_arg_type)
          << /* ordinal */ 1 << /* scalar or vector */ 5 << /* no int */ 0
          << /* fp */ 1 << ArgTyA;
      return true;
    }

    QualType RetTy = ArgTyA;
    TheCall->setType(RetTy);
    break;
  }
  }
  return false;
}
} // namespace clang
