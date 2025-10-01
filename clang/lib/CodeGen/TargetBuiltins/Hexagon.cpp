//===------ Hexagon.cpp - Emit LLVM Code for builtins ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Builtin calls as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/IR/IntrinsicsHexagon.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

static std::pair<Intrinsic::ID, unsigned>
getIntrinsicForHexagonNonClangBuiltin(unsigned BuiltinID) {
  struct Info {
    unsigned BuiltinID;
    Intrinsic::ID IntrinsicID;
    unsigned VecLen;
  };
  static Info Infos[] = {
#define CUSTOM_BUILTIN_MAPPING(x,s) \
  { Hexagon::BI__builtin_HEXAGON_##x, Intrinsic::hexagon_##x, s },
    CUSTOM_BUILTIN_MAPPING(L2_loadrub_pci, 0)
    CUSTOM_BUILTIN_MAPPING(L2_loadrb_pci, 0)
    CUSTOM_BUILTIN_MAPPING(L2_loadruh_pci, 0)
    CUSTOM_BUILTIN_MAPPING(L2_loadrh_pci, 0)
    CUSTOM_BUILTIN_MAPPING(L2_loadri_pci, 0)
    CUSTOM_BUILTIN_MAPPING(L2_loadrd_pci, 0)
    CUSTOM_BUILTIN_MAPPING(L2_loadrub_pcr, 0)
    CUSTOM_BUILTIN_MAPPING(L2_loadrb_pcr, 0)
    CUSTOM_BUILTIN_MAPPING(L2_loadruh_pcr, 0)
    CUSTOM_BUILTIN_MAPPING(L2_loadrh_pcr, 0)
    CUSTOM_BUILTIN_MAPPING(L2_loadri_pcr, 0)
    CUSTOM_BUILTIN_MAPPING(L2_loadrd_pcr, 0)
    CUSTOM_BUILTIN_MAPPING(S2_storerb_pci, 0)
    CUSTOM_BUILTIN_MAPPING(S2_storerh_pci, 0)
    CUSTOM_BUILTIN_MAPPING(S2_storerf_pci, 0)
    CUSTOM_BUILTIN_MAPPING(S2_storeri_pci, 0)
    CUSTOM_BUILTIN_MAPPING(S2_storerd_pci, 0)
    CUSTOM_BUILTIN_MAPPING(S2_storerb_pcr, 0)
    CUSTOM_BUILTIN_MAPPING(S2_storerh_pcr, 0)
    CUSTOM_BUILTIN_MAPPING(S2_storerf_pcr, 0)
    CUSTOM_BUILTIN_MAPPING(S2_storeri_pcr, 0)
    CUSTOM_BUILTIN_MAPPING(S2_storerd_pcr, 0)
    // Legacy builtins that take a vector in place of a vector predicate.
    CUSTOM_BUILTIN_MAPPING(V6_vmaskedstoreq, 64)
    CUSTOM_BUILTIN_MAPPING(V6_vmaskedstorenq, 64)
    CUSTOM_BUILTIN_MAPPING(V6_vmaskedstorentq, 64)
    CUSTOM_BUILTIN_MAPPING(V6_vmaskedstorentnq, 64)
    CUSTOM_BUILTIN_MAPPING(V6_vmaskedstoreq_128B, 128)
    CUSTOM_BUILTIN_MAPPING(V6_vmaskedstorenq_128B, 128)
    CUSTOM_BUILTIN_MAPPING(V6_vmaskedstorentq_128B, 128)
    CUSTOM_BUILTIN_MAPPING(V6_vmaskedstorentnq_128B, 128)
#include "clang/Basic/BuiltinsHexagonMapCustomDep.def"
#undef CUSTOM_BUILTIN_MAPPING
  };

  auto CmpInfo = [] (Info A, Info B) { return A.BuiltinID < B.BuiltinID; };
  static const bool SortOnce = (llvm::sort(Infos, CmpInfo), true);
  (void)SortOnce;

  const Info *F = llvm::lower_bound(Infos, Info{BuiltinID, 0, 0}, CmpInfo);
  if (F == std::end(Infos) || F->BuiltinID != BuiltinID)
    return {Intrinsic::not_intrinsic, 0};

  return {F->IntrinsicID, F->VecLen};
}

Value *CodeGenFunction::EmitHexagonBuiltinExpr(unsigned BuiltinID,
                                               const CallExpr *E) {
  Intrinsic::ID ID;
  unsigned VecLen;
  std::tie(ID, VecLen) = getIntrinsicForHexagonNonClangBuiltin(BuiltinID);

  auto MakeCircOp = [this, E](unsigned IntID, bool IsLoad) {
    // The base pointer is passed by address, so it needs to be loaded.
    Address A = EmitPointerWithAlignment(E->getArg(0));
    Address BP = Address(A.emitRawPointer(*this), Int8PtrTy, A.getAlignment());
    llvm::Value *Base = Builder.CreateLoad(BP);
    // The treatment of both loads and stores is the same: the arguments for
    // the builtin are the same as the arguments for the intrinsic.
    // Load:
    //   builtin(Base, Inc, Mod, Start) -> intr(Base, Inc, Mod, Start)
    //   builtin(Base, Mod, Start)      -> intr(Base, Mod, Start)
    // Store:
    //   builtin(Base, Inc, Mod, Val, Start) -> intr(Base, Inc, Mod, Val, Start)
    //   builtin(Base, Mod, Val, Start)      -> intr(Base, Mod, Val, Start)
    SmallVector<llvm::Value*,5> Ops = { Base };
    for (unsigned i = 1, e = E->getNumArgs(); i != e; ++i)
      Ops.push_back(EmitScalarExpr(E->getArg(i)));

    llvm::Value *Result = Builder.CreateCall(CGM.getIntrinsic(IntID), Ops);
    // The load intrinsics generate two results (Value, NewBase), stores
    // generate one (NewBase). The new base address needs to be stored.
    llvm::Value *NewBase = IsLoad ? Builder.CreateExtractValue(Result, 1)
                                  : Result;
    llvm::Value *LV = EmitScalarExpr(E->getArg(0));
    Address Dest = EmitPointerWithAlignment(E->getArg(0));
    llvm::Value *RetVal =
        Builder.CreateAlignedStore(NewBase, LV, Dest.getAlignment());
    if (IsLoad)
      RetVal = Builder.CreateExtractValue(Result, 0);
    return RetVal;
  };

  // Handle the conversion of bit-reverse load intrinsics to bit code.
  // The intrinsic call after this function only reads from memory and the
  // write to memory is dealt by the store instruction.
  auto MakeBrevLd = [this, E](unsigned IntID, llvm::Type *DestTy) {
    // The intrinsic generates one result, which is the new value for the base
    // pointer. It needs to be returned. The result of the load instruction is
    // passed to intrinsic by address, so the value needs to be stored.
    llvm::Value *BaseAddress = EmitScalarExpr(E->getArg(0));

    // Expressions like &(*pt++) will be incremented per evaluation.
    // EmitPointerWithAlignment and EmitScalarExpr evaluates the expression
    // per call.
    Address DestAddr = EmitPointerWithAlignment(E->getArg(1));
    DestAddr = DestAddr.withElementType(Int8Ty);
    llvm::Value *DestAddress = DestAddr.emitRawPointer(*this);

    // Operands are Base, Dest, Modifier.
    // The intrinsic format in LLVM IR is defined as
    // { ValueType, i8* } (i8*, i32).
    llvm::Value *Result = Builder.CreateCall(
        CGM.getIntrinsic(IntID), {BaseAddress, EmitScalarExpr(E->getArg(2))});

    // The value needs to be stored as the variable is passed by reference.
    llvm::Value *DestVal = Builder.CreateExtractValue(Result, 0);

    // The store needs to be truncated to fit the destination type.
    // While i32 and i64 are natively supported on Hexagon, i8 and i16 needs
    // to be handled with stores of respective destination type.
    DestVal = Builder.CreateTrunc(DestVal, DestTy);

    Builder.CreateAlignedStore(DestVal, DestAddress, DestAddr.getAlignment());
    // The updated value of the base pointer is returned.
    return Builder.CreateExtractValue(Result, 1);
  };

  auto V2Q = [this, VecLen] (llvm::Value *Vec) {
    Intrinsic::ID ID = VecLen == 128 ? Intrinsic::hexagon_V6_vandvrt_128B
                                     : Intrinsic::hexagon_V6_vandvrt;
    return Builder.CreateCall(CGM.getIntrinsic(ID),
                              {Vec, Builder.getInt32(-1)});
  };
  auto Q2V = [this, VecLen] (llvm::Value *Pred) {
    Intrinsic::ID ID = VecLen == 128 ? Intrinsic::hexagon_V6_vandqrt_128B
                                     : Intrinsic::hexagon_V6_vandqrt;
    return Builder.CreateCall(CGM.getIntrinsic(ID),
                              {Pred, Builder.getInt32(-1)});
  };

  switch (BuiltinID) {
  // These intrinsics return a tuple {Vector, VectorPred} in LLVM IR,
  // and the corresponding C/C++ builtins use loads/stores to update
  // the predicate.
  case Hexagon::BI__builtin_HEXAGON_V6_vaddcarry:
  case Hexagon::BI__builtin_HEXAGON_V6_vaddcarry_128B:
  case Hexagon::BI__builtin_HEXAGON_V6_vsubcarry:
  case Hexagon::BI__builtin_HEXAGON_V6_vsubcarry_128B: {
    // Get the type from the 0-th argument.
    llvm::Type *VecType = ConvertType(E->getArg(0)->getType());
    Address PredAddr =
        EmitPointerWithAlignment(E->getArg(2)).withElementType(VecType);
    llvm::Value *PredIn = V2Q(Builder.CreateLoad(PredAddr));
    llvm::Value *Result = Builder.CreateCall(CGM.getIntrinsic(ID),
        {EmitScalarExpr(E->getArg(0)), EmitScalarExpr(E->getArg(1)), PredIn});

    llvm::Value *PredOut = Builder.CreateExtractValue(Result, 1);
    Builder.CreateAlignedStore(Q2V(PredOut), PredAddr.emitRawPointer(*this),
                               PredAddr.getAlignment());
    return Builder.CreateExtractValue(Result, 0);
  }
  // These are identical to the builtins above, except they don't consume
  // input carry, only generate carry-out. Since they still produce two
  // outputs, generate the store of the predicate, but no load.
  case Hexagon::BI__builtin_HEXAGON_V6_vaddcarryo:
  case Hexagon::BI__builtin_HEXAGON_V6_vaddcarryo_128B:
  case Hexagon::BI__builtin_HEXAGON_V6_vsubcarryo:
  case Hexagon::BI__builtin_HEXAGON_V6_vsubcarryo_128B: {
    // Get the type from the 0-th argument.
    llvm::Type *VecType = ConvertType(E->getArg(0)->getType());
    Address PredAddr =
        EmitPointerWithAlignment(E->getArg(2)).withElementType(VecType);
    llvm::Value *Result = Builder.CreateCall(CGM.getIntrinsic(ID),
        {EmitScalarExpr(E->getArg(0)), EmitScalarExpr(E->getArg(1))});

    llvm::Value *PredOut = Builder.CreateExtractValue(Result, 1);
    Builder.CreateAlignedStore(Q2V(PredOut), PredAddr.emitRawPointer(*this),
                               PredAddr.getAlignment());
    return Builder.CreateExtractValue(Result, 0);
  }

  case Hexagon::BI__builtin_HEXAGON_V6_vmaskedstoreq:
  case Hexagon::BI__builtin_HEXAGON_V6_vmaskedstorenq:
  case Hexagon::BI__builtin_HEXAGON_V6_vmaskedstorentq:
  case Hexagon::BI__builtin_HEXAGON_V6_vmaskedstorentnq:
  case Hexagon::BI__builtin_HEXAGON_V6_vmaskedstoreq_128B:
  case Hexagon::BI__builtin_HEXAGON_V6_vmaskedstorenq_128B:
  case Hexagon::BI__builtin_HEXAGON_V6_vmaskedstorentq_128B:
  case Hexagon::BI__builtin_HEXAGON_V6_vmaskedstorentnq_128B: {
    SmallVector<llvm::Value*,4> Ops;
    const Expr *PredOp = E->getArg(0);
    // There will be an implicit cast to a boolean vector. Strip it.
    if (auto *Cast = dyn_cast<ImplicitCastExpr>(PredOp)) {
      if (Cast->getCastKind() == CK_BitCast)
        PredOp = Cast->getSubExpr();
      Ops.push_back(V2Q(EmitScalarExpr(PredOp)));
    }
    for (int i = 1, e = E->getNumArgs(); i != e; ++i)
      Ops.push_back(EmitScalarExpr(E->getArg(i)));
    return Builder.CreateCall(CGM.getIntrinsic(ID), Ops);
  }

  case Hexagon::BI__builtin_HEXAGON_L2_loadrub_pci:
  case Hexagon::BI__builtin_HEXAGON_L2_loadrb_pci:
  case Hexagon::BI__builtin_HEXAGON_L2_loadruh_pci:
  case Hexagon::BI__builtin_HEXAGON_L2_loadrh_pci:
  case Hexagon::BI__builtin_HEXAGON_L2_loadri_pci:
  case Hexagon::BI__builtin_HEXAGON_L2_loadrd_pci:
  case Hexagon::BI__builtin_HEXAGON_L2_loadrub_pcr:
  case Hexagon::BI__builtin_HEXAGON_L2_loadrb_pcr:
  case Hexagon::BI__builtin_HEXAGON_L2_loadruh_pcr:
  case Hexagon::BI__builtin_HEXAGON_L2_loadrh_pcr:
  case Hexagon::BI__builtin_HEXAGON_L2_loadri_pcr:
  case Hexagon::BI__builtin_HEXAGON_L2_loadrd_pcr:
    return MakeCircOp(ID, /*IsLoad=*/true);
  case Hexagon::BI__builtin_HEXAGON_S2_storerb_pci:
  case Hexagon::BI__builtin_HEXAGON_S2_storerh_pci:
  case Hexagon::BI__builtin_HEXAGON_S2_storerf_pci:
  case Hexagon::BI__builtin_HEXAGON_S2_storeri_pci:
  case Hexagon::BI__builtin_HEXAGON_S2_storerd_pci:
  case Hexagon::BI__builtin_HEXAGON_S2_storerb_pcr:
  case Hexagon::BI__builtin_HEXAGON_S2_storerh_pcr:
  case Hexagon::BI__builtin_HEXAGON_S2_storerf_pcr:
  case Hexagon::BI__builtin_HEXAGON_S2_storeri_pcr:
  case Hexagon::BI__builtin_HEXAGON_S2_storerd_pcr:
    return MakeCircOp(ID, /*IsLoad=*/false);
  case Hexagon::BI__builtin_brev_ldub:
    return MakeBrevLd(Intrinsic::hexagon_L2_loadrub_pbr, Int8Ty);
  case Hexagon::BI__builtin_brev_ldb:
    return MakeBrevLd(Intrinsic::hexagon_L2_loadrb_pbr, Int8Ty);
  case Hexagon::BI__builtin_brev_lduh:
    return MakeBrevLd(Intrinsic::hexagon_L2_loadruh_pbr, Int16Ty);
  case Hexagon::BI__builtin_brev_ldh:
    return MakeBrevLd(Intrinsic::hexagon_L2_loadrh_pbr, Int16Ty);
  case Hexagon::BI__builtin_brev_ldw:
    return MakeBrevLd(Intrinsic::hexagon_L2_loadri_pbr, Int32Ty);
  case Hexagon::BI__builtin_brev_ldd:
    return MakeBrevLd(Intrinsic::hexagon_L2_loadrd_pbr, Int64Ty);
  } // switch

  return nullptr;
}
