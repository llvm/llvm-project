//===---- CIRGenBuiltinAArch64.cpp - Emit CIR for AArch64 builtins --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit ARM64 Builtin calls as CIR or a function call
// to be later resolved.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/CIR/MissingFeatures.h"

// TODO(cir): once all builtins are covered, decide whether we still
// need to use LLVM intrinsics or if there's a better approach to follow. Right
// now the intrinsics are reused to make it convenient to encode all thousands
// of them and passing down to LLVM lowering.
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAArch64.h"

#include "mlir/IR/Value.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Builtins.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace llvm;

// Generate vscale * scalingFactor
static mlir::Value genVscaleTimesFactor(mlir::Location loc,
                                        CIRGenBuilderTy builder,
                                        mlir::Type cirTy,
                                        int32_t scalingFactor) {
  mlir::Value vscale = builder.emitIntrinsicCallOp(loc, "vscale", cirTy);
  return builder.createNUWAMul(loc, vscale,
                               builder.getUInt64(scalingFactor, loc));
}

static bool aarch64SVEIntrinsicsProvenSorted = false;

namespace {
struct AArch64BuiltinInfo {
  unsigned builtinID;
  unsigned llvmIntrinsic;
  uint64_t typeModifier;

  bool operator<(unsigned rhsBuiltinID) const {
    return builtinID < rhsBuiltinID;
  }
  bool operator<(const AArch64BuiltinInfo &te) const {
    return builtinID < te.builtinID;
  }
};
} // end anonymous namespace

#define SVEMAP1(NameBase, llvmIntrinsic, TypeModifier)                         \
  {SVE::BI__builtin_sve_##NameBase, Intrinsic::llvmIntrinsic, TypeModifier}

#define SVEMAP2(NameBase, TypeModifier)                                        \
  {SVE::BI__builtin_sve_##NameBase, 0, TypeModifier}
static const AArch64BuiltinInfo aarch64SVEIntrinsicMap[] = {
#define GET_SVE_LLVM_INTRINSIC_MAP
#include "clang/Basic/arm_sve_builtin_cg.inc"
#undef GET_SVE_LLVM_INTRINSIC_MAP
};

static const AArch64BuiltinInfo *
findARMVectorIntrinsicInMap(ArrayRef<AArch64BuiltinInfo> intrinsicMap,
                            unsigned builtinID, bool &mapProvenSorted) {

#ifndef NDEBUG
  if (!mapProvenSorted) {
    assert(llvm::is_sorted(intrinsicMap));
    mapProvenSorted = true;
  }
#endif

  const AArch64BuiltinInfo *info = llvm::lower_bound(intrinsicMap, builtinID);

  if (info != intrinsicMap.end() && info->builtinID == builtinID)
    return info;

  return nullptr;
}

bool CIRGenFunction::getAArch64SVEProcessedOperands(
    unsigned builtinID, const CallExpr *expr, SmallVectorImpl<mlir::Value> &ops,
    SVETypeFlags typeFlags) {
  // Find out if any arguments are required to be integer constant expressions.
  unsigned iceArguments = 0;
  ASTContext::GetBuiltinTypeError error;
  getContext().GetBuiltinType(builtinID, error, &iceArguments);
  assert(error == ASTContext::GE_None && "Should not codegen an error");

  for (unsigned i = 0, e = expr->getNumArgs(); i != e; i++) {
    bool isIce = iceArguments & (1 << i);
    mlir::Value arg = emitScalarExpr(expr->getArg(i));

    if (isIce) {
      cgm.errorNYI(expr->getSourceRange(),
                   std::string("unimplemented AArch64 builtin call: ") +
                       getContext().BuiltinInfo.getName(builtinID));
    }

    // FIXME: Handle types like svint16x2_t, which are currently incorrectly
    // converted to i32. These should be treated as structs and unpacked.

    ops.push_back(arg);
  }
  return true;
}

std::optional<mlir::Value>
CIRGenFunction::emitAArch64SVEBuiltinExpr(unsigned builtinID,
                                          const CallExpr *expr) {
  if (builtinID >= SVE::BI__builtin_sve_reinterpret_s8_s8 &&
      builtinID <= SVE::BI__builtin_sve_reinterpret_f64_f64_x4) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  assert(!cir::MissingFeatures::aarch64SVEIntrinsics());

  auto *builtinIntrInfo = findARMVectorIntrinsicInMap(
      aarch64SVEIntrinsicMap, builtinID, aarch64SVEIntrinsicsProvenSorted);

  // The operands of the builtin call
  llvm::SmallVector<mlir::Value> ops;

  SVETypeFlags typeFlags(builtinIntrInfo->typeModifier);
  if (!CIRGenFunction::getAArch64SVEProcessedOperands(builtinID, expr, ops,
                                                      typeFlags))
    return mlir::Value{};

  if (typeFlags.isLoad() || typeFlags.isStore() || typeFlags.isGatherLoad() ||
      typeFlags.isScatterStore() || typeFlags.isPrefetch() ||
      typeFlags.isGatherPrefetch() || typeFlags.isStructLoad() ||
      typeFlags.isStructStore() || typeFlags.isTupleSet() ||
      typeFlags.isTupleGet() || typeFlags.isTupleCreate() ||
      typeFlags.isUndef())
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));

  mlir::Location loc = getLoc(expr->getExprLoc());

  // Handle built-ins for which there is a corresponding LLVM Intrinsic.
  // -------------------------------------------------------------------
  if (builtinIntrInfo->llvmIntrinsic != 0) {
    // Emit set FPMR for intrinsics that require it.
    if (typeFlags.setsFPMR())
      cgm.errorNYI(expr->getSourceRange(),
                   std::string("unimplemented AArch64 builtin call: ") +
                       getContext().BuiltinInfo.getName(builtinID));

    if (typeFlags.getMergeType() == SVETypeFlags::MergeZeroExp)
      cgm.errorNYI(expr->getSourceRange(),
                   std::string("unimplemented AArch64 builtin call: ") +
                       getContext().BuiltinInfo.getName(builtinID));

    if (typeFlags.getMergeType() == SVETypeFlags::MergeAnyExp)
      cgm.errorNYI(expr->getSourceRange(),
                   std::string("unimplemented AArch64 builtin call: ") +
                       getContext().BuiltinInfo.getName(builtinID));

    // Some ACLE builtins leave out the argument to specify the predicate
    // pattern, which is expected to be expanded to an SV_ALL pattern.
    if (typeFlags.isAppendSVALL())
      cgm.errorNYI(expr->getSourceRange(),
                   std::string("unimplemented AArch64 builtin call: ") +
                       getContext().BuiltinInfo.getName(builtinID));
    if (typeFlags.isInsertOp1SVALL())
      cgm.errorNYI(expr->getSourceRange(),
                   std::string("unimplemented AArch64 builtin call: ") +
                       getContext().BuiltinInfo.getName(builtinID));

    // Predicates must match the main datatype.
    for (mlir::Value &op : ops)
      if (auto predTy = dyn_cast<mlir::VectorType>(op.getType()))
        if (predTy.getElementType().isInteger(1))
          cgm.errorNYI(expr->getSourceRange(),
                       std::string("unimplemented AArch64 builtin call: ") +
                           getContext().BuiltinInfo.getName(builtinID));

    // Splat scalar operand to vector (intrinsics with _n infix)
    if (typeFlags.hasSplatOperand()) {
      cgm.errorNYI(expr->getSourceRange(),
                   std::string("unimplemented AArch64 builtin call: ") +
                       getContext().BuiltinInfo.getName(builtinID));
    }

    if (typeFlags.isReverseCompare())
      cgm.errorNYI(expr->getSourceRange(),
                   std::string("unimplemented AArch64 builtin call: ") +
                       getContext().BuiltinInfo.getName(builtinID));
    if (typeFlags.isReverseUSDOT())
      cgm.errorNYI(expr->getSourceRange(),
                   std::string("unimplemented AArch64 builtin call: ") +
                       getContext().BuiltinInfo.getName(builtinID));
    if (typeFlags.isReverseMergeAnyBinOp() &&
        typeFlags.getMergeType() == SVETypeFlags::MergeAny)
      cgm.errorNYI(expr->getSourceRange(),
                   std::string("unimplemented AArch64 builtin call: ") +
                       getContext().BuiltinInfo.getName(builtinID));
    if (typeFlags.isReverseMergeAnyAccOp() &&
        typeFlags.getMergeType() == SVETypeFlags::MergeAny)
      cgm.errorNYI(expr->getSourceRange(),
                   std::string("unimplemented AArch64 builtin call: ") +
                       getContext().BuiltinInfo.getName(builtinID));

    // Predicated intrinsics with _z suffix.
    if (typeFlags.getMergeType() == SVETypeFlags::MergeZero) {
      cgm.errorNYI(expr->getSourceRange(),
                   std::string("unimplemented AArch64 builtin call: ") +
                       getContext().BuiltinInfo.getName(builtinID));
    }

    std::string llvmIntrName(Intrinsic::getBaseName(
        (llvm::Intrinsic::ID)builtinIntrInfo->llvmIntrinsic));

    llvmIntrName.erase(0, /*std::strlen(".llvm")=*/5);

    auto retTy = convertType(expr->getType());

    auto call = builder.emitIntrinsicCallOp(loc, llvmIntrName, retTy,
                                            mlir::ValueRange{ops});
    if (call.getType() == retTy)
      return call;

    // Predicate results must be converted to svbool_t.
    if (isa<mlir::VectorType>(retTy) &&
        cast<mlir::VectorType>(retTy).isScalable())
      cgm.errorNYI(expr->getSourceRange(),
                   std::string("unimplemented AArch64 builtin call: ") +
                       getContext().BuiltinInfo.getName(builtinID));
    // TODO Handle struct types, e.g. svint8x2_t (update the converter first).

    llvm_unreachable("unsupported element count!");
  }

  // Handle the remaining built-ins.
  // -------------------------------
  switch (builtinID) {
  default:
    return std::nullopt;

  case SVE::BI__builtin_sve_svreinterpret_b:
  case SVE::BI__builtin_sve_svreinterpret_c:
  case SVE::BI__builtin_sve_svpsel_lane_b8:
  case SVE::BI__builtin_sve_svpsel_lane_b16:
  case SVE::BI__builtin_sve_svpsel_lane_b32:
  case SVE::BI__builtin_sve_svpsel_lane_b64:
  case SVE::BI__builtin_sve_svpsel_lane_c8:
  case SVE::BI__builtin_sve_svpsel_lane_c16:
  case SVE::BI__builtin_sve_svpsel_lane_c32:
  case SVE::BI__builtin_sve_svpsel_lane_c64:
  case SVE::BI__builtin_sve_svmov_b_z:
  case SVE::BI__builtin_sve_svnot_b_z:
  case SVE::BI__builtin_sve_svmovlb_u16:
  case SVE::BI__builtin_sve_svmovlb_u32:
  case SVE::BI__builtin_sve_svmovlb_u64:
  case SVE::BI__builtin_sve_svmovlb_s16:
  case SVE::BI__builtin_sve_svmovlb_s32:
  case SVE::BI__builtin_sve_svmovlb_s64:
  case SVE::BI__builtin_sve_svmovlt_u16:
  case SVE::BI__builtin_sve_svmovlt_u32:
  case SVE::BI__builtin_sve_svmovlt_u64:
  case SVE::BI__builtin_sve_svmovlt_s16:
  case SVE::BI__builtin_sve_svmovlt_s32:
  case SVE::BI__builtin_sve_svmovlt_s64:
  case SVE::BI__builtin_sve_svpmullt_u16:
  case SVE::BI__builtin_sve_svpmullt_u64:
  case SVE::BI__builtin_sve_svpmullt_n_u16:
  case SVE::BI__builtin_sve_svpmullt_n_u64:
  case SVE::BI__builtin_sve_svpmullb_u16:
  case SVE::BI__builtin_sve_svpmullb_u64:
  case SVE::BI__builtin_sve_svpmullb_n_u16:
  case SVE::BI__builtin_sve_svpmullb_n_u64:

  case SVE::BI__builtin_sve_svdup_n_b8:
  case SVE::BI__builtin_sve_svdup_n_b16:
  case SVE::BI__builtin_sve_svdup_n_b32:
  case SVE::BI__builtin_sve_svdup_n_b64:

  case SVE::BI__builtin_sve_svdupq_n_b8:
  case SVE::BI__builtin_sve_svdupq_n_b16:
  case SVE::BI__builtin_sve_svdupq_n_b32:
  case SVE::BI__builtin_sve_svdupq_n_b64:
  case SVE::BI__builtin_sve_svdupq_n_u8:
  case SVE::BI__builtin_sve_svdupq_n_s8:
  case SVE::BI__builtin_sve_svdupq_n_u64:
  case SVE::BI__builtin_sve_svdupq_n_f64:
  case SVE::BI__builtin_sve_svdupq_n_s64:
  case SVE::BI__builtin_sve_svdupq_n_u16:
  case SVE::BI__builtin_sve_svdupq_n_f16:
  case SVE::BI__builtin_sve_svdupq_n_bf16:
  case SVE::BI__builtin_sve_svdupq_n_s16:
  case SVE::BI__builtin_sve_svdupq_n_u32:
  case SVE::BI__builtin_sve_svdupq_n_f32:
  case SVE::BI__builtin_sve_svdupq_n_s32:
  case SVE::BI__builtin_sve_svpfalse_b:
  case SVE::BI__builtin_sve_svpfalse_c:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};

  case SVE::BI__builtin_sve_svlen_u8:
  case SVE::BI__builtin_sve_svlen_s8:
    return genVscaleTimesFactor(loc, builder, convertType(expr->getType()), 16);

  case SVE::BI__builtin_sve_svlen_u16:
  case SVE::BI__builtin_sve_svlen_s16:
  case SVE::BI__builtin_sve_svlen_f16:
  case SVE::BI__builtin_sve_svlen_bf16:
    return genVscaleTimesFactor(loc, builder, convertType(expr->getType()), 8);

  case SVE::BI__builtin_sve_svlen_u32:
  case SVE::BI__builtin_sve_svlen_s32:
  case SVE::BI__builtin_sve_svlen_f32:
    return genVscaleTimesFactor(loc, builder, convertType(expr->getType()), 4);

  case SVE::BI__builtin_sve_svlen_u64:
  case SVE::BI__builtin_sve_svlen_s64:
  case SVE::BI__builtin_sve_svlen_f64:
    return genVscaleTimesFactor(loc, builder, convertType(expr->getType()), 2);

  case SVE::BI__builtin_sve_svtbl2_u8:
  case SVE::BI__builtin_sve_svtbl2_s8:
  case SVE::BI__builtin_sve_svtbl2_u16:
  case SVE::BI__builtin_sve_svtbl2_s16:
  case SVE::BI__builtin_sve_svtbl2_u32:
  case SVE::BI__builtin_sve_svtbl2_s32:
  case SVE::BI__builtin_sve_svtbl2_u64:
  case SVE::BI__builtin_sve_svtbl2_s64:
  case SVE::BI__builtin_sve_svtbl2_f16:
  case SVE::BI__builtin_sve_svtbl2_bf16:
  case SVE::BI__builtin_sve_svtbl2_f32:
  case SVE::BI__builtin_sve_svtbl2_f64:
  case SVE::BI__builtin_sve_svset_neonq_s8:
  case SVE::BI__builtin_sve_svset_neonq_s16:
  case SVE::BI__builtin_sve_svset_neonq_s32:
  case SVE::BI__builtin_sve_svset_neonq_s64:
  case SVE::BI__builtin_sve_svset_neonq_u8:
  case SVE::BI__builtin_sve_svset_neonq_u16:
  case SVE::BI__builtin_sve_svset_neonq_u32:
  case SVE::BI__builtin_sve_svset_neonq_u64:
  case SVE::BI__builtin_sve_svset_neonq_f16:
  case SVE::BI__builtin_sve_svset_neonq_f32:
  case SVE::BI__builtin_sve_svset_neonq_f64:
  case SVE::BI__builtin_sve_svset_neonq_bf16:
  case SVE::BI__builtin_sve_svget_neonq_s8:
  case SVE::BI__builtin_sve_svget_neonq_s16:
  case SVE::BI__builtin_sve_svget_neonq_s32:
  case SVE::BI__builtin_sve_svget_neonq_s64:
  case SVE::BI__builtin_sve_svget_neonq_u8:
  case SVE::BI__builtin_sve_svget_neonq_u16:
  case SVE::BI__builtin_sve_svget_neonq_u32:
  case SVE::BI__builtin_sve_svget_neonq_u64:
  case SVE::BI__builtin_sve_svget_neonq_f16:
  case SVE::BI__builtin_sve_svget_neonq_f32:
  case SVE::BI__builtin_sve_svget_neonq_f64:
  case SVE::BI__builtin_sve_svget_neonq_bf16:
  case SVE::BI__builtin_sve_svdup_neonq_s8:
  case SVE::BI__builtin_sve_svdup_neonq_s16:
  case SVE::BI__builtin_sve_svdup_neonq_s32:
  case SVE::BI__builtin_sve_svdup_neonq_s64:
  case SVE::BI__builtin_sve_svdup_neonq_u8:
  case SVE::BI__builtin_sve_svdup_neonq_u16:
  case SVE::BI__builtin_sve_svdup_neonq_u32:
  case SVE::BI__builtin_sve_svdup_neonq_u64:
  case SVE::BI__builtin_sve_svdup_neonq_f16:
  case SVE::BI__builtin_sve_svdup_neonq_f32:
  case SVE::BI__builtin_sve_svdup_neonq_f64:
  case SVE::BI__builtin_sve_svdup_neonq_bf16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  // Unreachable: All cases in the switch above return.
}

std::optional<mlir::Value>
CIRGenFunction::emitAArch64SMEBuiltinExpr(unsigned builtinID,
                                          const CallExpr *expr) {
  assert(!cir::MissingFeatures::aarch64SMEIntrinsics());

  cgm.errorNYI(expr->getSourceRange(),
               std::string("unimplemented AArch64 builtin call: ") +
                   getContext().BuiltinInfo.getName(builtinID));
  return mlir::Value{};
}

// Some intrinsics are equivalent for codegen.
static const std::pair<unsigned, unsigned> neonEquivalentIntrinsicMap[] = {
    {
        NEON::BI__builtin_neon_splat_lane_bf16,
        NEON::BI__builtin_neon_splat_lane_v,
    },
    {
        NEON::BI__builtin_neon_splat_laneq_bf16,
        NEON::BI__builtin_neon_splat_laneq_v,
    },
    {
        NEON::BI__builtin_neon_splatq_lane_bf16,
        NEON::BI__builtin_neon_splatq_lane_v,
    },
    {
        NEON::BI__builtin_neon_splatq_laneq_bf16,
        NEON::BI__builtin_neon_splatq_laneq_v,
    },
    {
        NEON::BI__builtin_neon_vabd_f16,
        NEON::BI__builtin_neon_vabd_v,
    },
    {
        NEON::BI__builtin_neon_vabdq_f16,
        NEON::BI__builtin_neon_vabdq_v,
    },
    {
        NEON::BI__builtin_neon_vabs_f16,
        NEON::BI__builtin_neon_vabs_v,
    },
    {
        NEON::BI__builtin_neon_vabsq_f16,
        NEON::BI__builtin_neon_vabsq_v,
    },
    {
        NEON::BI__builtin_neon_vcage_f16,
        NEON::BI__builtin_neon_vcage_v,
    },
    {
        NEON::BI__builtin_neon_vcageq_f16,
        NEON::BI__builtin_neon_vcageq_v,
    },
    {
        NEON::BI__builtin_neon_vcagt_f16,
        NEON::BI__builtin_neon_vcagt_v,
    },
    {
        NEON::BI__builtin_neon_vcagtq_f16,
        NEON::BI__builtin_neon_vcagtq_v,
    },
    {
        NEON::BI__builtin_neon_vcale_f16,
        NEON::BI__builtin_neon_vcale_v,
    },
    {
        NEON::BI__builtin_neon_vcaleq_f16,
        NEON::BI__builtin_neon_vcaleq_v,
    },
    {
        NEON::BI__builtin_neon_vcalt_f16,
        NEON::BI__builtin_neon_vcalt_v,
    },
    {
        NEON::BI__builtin_neon_vcaltq_f16,
        NEON::BI__builtin_neon_vcaltq_v,
    },
    {
        NEON::BI__builtin_neon_vceqz_f16,
        NEON::BI__builtin_neon_vceqz_v,
    },
    {
        NEON::BI__builtin_neon_vceqzq_f16,
        NEON::BI__builtin_neon_vceqzq_v,
    },
    {
        NEON::BI__builtin_neon_vcgez_f16,
        NEON::BI__builtin_neon_vcgez_v,
    },
    {
        NEON::BI__builtin_neon_vcgezq_f16,
        NEON::BI__builtin_neon_vcgezq_v,
    },
    {
        NEON::BI__builtin_neon_vcgtz_f16,
        NEON::BI__builtin_neon_vcgtz_v,
    },
    {
        NEON::BI__builtin_neon_vcgtzq_f16,
        NEON::BI__builtin_neon_vcgtzq_v,
    },
    {
        NEON::BI__builtin_neon_vclez_f16,
        NEON::BI__builtin_neon_vclez_v,
    },
    {
        NEON::BI__builtin_neon_vclezq_f16,
        NEON::BI__builtin_neon_vclezq_v,
    },
    {
        NEON::BI__builtin_neon_vcltz_f16,
        NEON::BI__builtin_neon_vcltz_v,
    },
    {
        NEON::BI__builtin_neon_vcltzq_f16,
        NEON::BI__builtin_neon_vcltzq_v,
    },
    {
        NEON::BI__builtin_neon_vfma_f16,
        NEON::BI__builtin_neon_vfma_v,
    },
    {
        NEON::BI__builtin_neon_vfma_lane_f16,
        NEON::BI__builtin_neon_vfma_lane_v,
    },
    {
        NEON::BI__builtin_neon_vfma_laneq_f16,
        NEON::BI__builtin_neon_vfma_laneq_v,
    },
    {
        NEON::BI__builtin_neon_vfmaq_f16,
        NEON::BI__builtin_neon_vfmaq_v,
    },
    {
        NEON::BI__builtin_neon_vfmaq_lane_f16,
        NEON::BI__builtin_neon_vfmaq_lane_v,
    },
    {
        NEON::BI__builtin_neon_vfmaq_laneq_f16,
        NEON::BI__builtin_neon_vfmaq_laneq_v,
    },
    {NEON::BI__builtin_neon_vld1_bf16_x2, NEON::BI__builtin_neon_vld1_x2_v},
    {NEON::BI__builtin_neon_vld1_bf16_x3, NEON::BI__builtin_neon_vld1_x3_v},
    {NEON::BI__builtin_neon_vld1_bf16_x4, NEON::BI__builtin_neon_vld1_x4_v},
    {NEON::BI__builtin_neon_vld1_bf16, NEON::BI__builtin_neon_vld1_v},
    {NEON::BI__builtin_neon_vld1_dup_bf16, NEON::BI__builtin_neon_vld1_dup_v},
    {NEON::BI__builtin_neon_vld1_lane_bf16, NEON::BI__builtin_neon_vld1_lane_v},
    {NEON::BI__builtin_neon_vld1q_bf16_x2, NEON::BI__builtin_neon_vld1q_x2_v},
    {NEON::BI__builtin_neon_vld1q_bf16_x3, NEON::BI__builtin_neon_vld1q_x3_v},
    {NEON::BI__builtin_neon_vld1q_bf16_x4, NEON::BI__builtin_neon_vld1q_x4_v},
    {NEON::BI__builtin_neon_vld1q_bf16, NEON::BI__builtin_neon_vld1q_v},
    {NEON::BI__builtin_neon_vld1q_dup_bf16, NEON::BI__builtin_neon_vld1q_dup_v},
    {NEON::BI__builtin_neon_vld1q_lane_bf16,
     NEON::BI__builtin_neon_vld1q_lane_v},
    {NEON::BI__builtin_neon_vld2_bf16, NEON::BI__builtin_neon_vld2_v},
    {NEON::BI__builtin_neon_vld2_dup_bf16, NEON::BI__builtin_neon_vld2_dup_v},
    {NEON::BI__builtin_neon_vld2_lane_bf16, NEON::BI__builtin_neon_vld2_lane_v},
    {NEON::BI__builtin_neon_vld2q_bf16, NEON::BI__builtin_neon_vld2q_v},
    {NEON::BI__builtin_neon_vld2q_dup_bf16, NEON::BI__builtin_neon_vld2q_dup_v},
    {NEON::BI__builtin_neon_vld2q_lane_bf16,
     NEON::BI__builtin_neon_vld2q_lane_v},
    {NEON::BI__builtin_neon_vld3_bf16, NEON::BI__builtin_neon_vld3_v},
    {NEON::BI__builtin_neon_vld3_dup_bf16, NEON::BI__builtin_neon_vld3_dup_v},
    {NEON::BI__builtin_neon_vld3_lane_bf16, NEON::BI__builtin_neon_vld3_lane_v},
    {NEON::BI__builtin_neon_vld3q_bf16, NEON::BI__builtin_neon_vld3q_v},
    {NEON::BI__builtin_neon_vld3q_dup_bf16, NEON::BI__builtin_neon_vld3q_dup_v},
    {NEON::BI__builtin_neon_vld3q_lane_bf16,
     NEON::BI__builtin_neon_vld3q_lane_v},
    {NEON::BI__builtin_neon_vld4_bf16, NEON::BI__builtin_neon_vld4_v},
    {NEON::BI__builtin_neon_vld4_dup_bf16, NEON::BI__builtin_neon_vld4_dup_v},
    {NEON::BI__builtin_neon_vld4_lane_bf16, NEON::BI__builtin_neon_vld4_lane_v},
    {NEON::BI__builtin_neon_vld4q_bf16, NEON::BI__builtin_neon_vld4q_v},
    {NEON::BI__builtin_neon_vld4q_dup_bf16, NEON::BI__builtin_neon_vld4q_dup_v},
    {NEON::BI__builtin_neon_vld4q_lane_bf16,
     NEON::BI__builtin_neon_vld4q_lane_v},
    {
        NEON::BI__builtin_neon_vmax_f16,
        NEON::BI__builtin_neon_vmax_v,
    },
    {
        NEON::BI__builtin_neon_vmaxnm_f16,
        NEON::BI__builtin_neon_vmaxnm_v,
    },
    {
        NEON::BI__builtin_neon_vmaxnmq_f16,
        NEON::BI__builtin_neon_vmaxnmq_v,
    },
    {
        NEON::BI__builtin_neon_vmaxq_f16,
        NEON::BI__builtin_neon_vmaxq_v,
    },
    {
        NEON::BI__builtin_neon_vmin_f16,
        NEON::BI__builtin_neon_vmin_v,
    },
    {
        NEON::BI__builtin_neon_vminnm_f16,
        NEON::BI__builtin_neon_vminnm_v,
    },
    {
        NEON::BI__builtin_neon_vminnmq_f16,
        NEON::BI__builtin_neon_vminnmq_v,
    },
    {
        NEON::BI__builtin_neon_vminq_f16,
        NEON::BI__builtin_neon_vminq_v,
    },
    {
        NEON::BI__builtin_neon_vmulx_f16,
        NEON::BI__builtin_neon_vmulx_v,
    },
    {
        NEON::BI__builtin_neon_vmulxq_f16,
        NEON::BI__builtin_neon_vmulxq_v,
    },
    {
        NEON::BI__builtin_neon_vpadd_f16,
        NEON::BI__builtin_neon_vpadd_v,
    },
    {
        NEON::BI__builtin_neon_vpaddq_f16,
        NEON::BI__builtin_neon_vpaddq_v,
    },
    {
        NEON::BI__builtin_neon_vpmax_f16,
        NEON::BI__builtin_neon_vpmax_v,
    },
    {
        NEON::BI__builtin_neon_vpmaxnm_f16,
        NEON::BI__builtin_neon_vpmaxnm_v,
    },
    {
        NEON::BI__builtin_neon_vpmaxnmq_f16,
        NEON::BI__builtin_neon_vpmaxnmq_v,
    },
    {
        NEON::BI__builtin_neon_vpmaxq_f16,
        NEON::BI__builtin_neon_vpmaxq_v,
    },
    {
        NEON::BI__builtin_neon_vpmin_f16,
        NEON::BI__builtin_neon_vpmin_v,
    },
    {
        NEON::BI__builtin_neon_vpminnm_f16,
        NEON::BI__builtin_neon_vpminnm_v,
    },
    {
        NEON::BI__builtin_neon_vpminnmq_f16,
        NEON::BI__builtin_neon_vpminnmq_v,
    },
    {
        NEON::BI__builtin_neon_vpminq_f16,
        NEON::BI__builtin_neon_vpminq_v,
    },
    {
        NEON::BI__builtin_neon_vrecpe_f16,
        NEON::BI__builtin_neon_vrecpe_v,
    },
    {
        NEON::BI__builtin_neon_vrecpeq_f16,
        NEON::BI__builtin_neon_vrecpeq_v,
    },
    {
        NEON::BI__builtin_neon_vrecps_f16,
        NEON::BI__builtin_neon_vrecps_v,
    },
    {
        NEON::BI__builtin_neon_vrecpsq_f16,
        NEON::BI__builtin_neon_vrecpsq_v,
    },
    {
        NEON::BI__builtin_neon_vrnd_f16,
        NEON::BI__builtin_neon_vrnd_v,
    },
    {
        NEON::BI__builtin_neon_vrnda_f16,
        NEON::BI__builtin_neon_vrnda_v,
    },
    {
        NEON::BI__builtin_neon_vrndaq_f16,
        NEON::BI__builtin_neon_vrndaq_v,
    },
    {
        NEON::BI__builtin_neon_vrndi_f16,
        NEON::BI__builtin_neon_vrndi_v,
    },
    {
        NEON::BI__builtin_neon_vrndiq_f16,
        NEON::BI__builtin_neon_vrndiq_v,
    },
    {
        NEON::BI__builtin_neon_vrndm_f16,
        NEON::BI__builtin_neon_vrndm_v,
    },
    {
        NEON::BI__builtin_neon_vrndmq_f16,
        NEON::BI__builtin_neon_vrndmq_v,
    },
    {
        NEON::BI__builtin_neon_vrndn_f16,
        NEON::BI__builtin_neon_vrndn_v,
    },
    {
        NEON::BI__builtin_neon_vrndnq_f16,
        NEON::BI__builtin_neon_vrndnq_v,
    },
    {
        NEON::BI__builtin_neon_vrndp_f16,
        NEON::BI__builtin_neon_vrndp_v,
    },
    {
        NEON::BI__builtin_neon_vrndpq_f16,
        NEON::BI__builtin_neon_vrndpq_v,
    },
    {
        NEON::BI__builtin_neon_vrndq_f16,
        NEON::BI__builtin_neon_vrndq_v,
    },
    {
        NEON::BI__builtin_neon_vrndx_f16,
        NEON::BI__builtin_neon_vrndx_v,
    },
    {
        NEON::BI__builtin_neon_vrndxq_f16,
        NEON::BI__builtin_neon_vrndxq_v,
    },
    {
        NEON::BI__builtin_neon_vrsqrte_f16,
        NEON::BI__builtin_neon_vrsqrte_v,
    },
    {
        NEON::BI__builtin_neon_vrsqrteq_f16,
        NEON::BI__builtin_neon_vrsqrteq_v,
    },
    {
        NEON::BI__builtin_neon_vrsqrts_f16,
        NEON::BI__builtin_neon_vrsqrts_v,
    },
    {
        NEON::BI__builtin_neon_vrsqrtsq_f16,
        NEON::BI__builtin_neon_vrsqrtsq_v,
    },
    {
        NEON::BI__builtin_neon_vsqrt_f16,
        NEON::BI__builtin_neon_vsqrt_v,
    },
    {
        NEON::BI__builtin_neon_vsqrtq_f16,
        NEON::BI__builtin_neon_vsqrtq_v,
    },
    {NEON::BI__builtin_neon_vst1_bf16_x2, NEON::BI__builtin_neon_vst1_x2_v},
    {NEON::BI__builtin_neon_vst1_bf16_x3, NEON::BI__builtin_neon_vst1_x3_v},
    {NEON::BI__builtin_neon_vst1_bf16_x4, NEON::BI__builtin_neon_vst1_x4_v},
    {NEON::BI__builtin_neon_vst1_bf16, NEON::BI__builtin_neon_vst1_v},
    {NEON::BI__builtin_neon_vst1_lane_bf16, NEON::BI__builtin_neon_vst1_lane_v},
    {NEON::BI__builtin_neon_vst1q_bf16_x2, NEON::BI__builtin_neon_vst1q_x2_v},
    {NEON::BI__builtin_neon_vst1q_bf16_x3, NEON::BI__builtin_neon_vst1q_x3_v},
    {NEON::BI__builtin_neon_vst1q_bf16_x4, NEON::BI__builtin_neon_vst1q_x4_v},
    {NEON::BI__builtin_neon_vst1q_bf16, NEON::BI__builtin_neon_vst1q_v},
    {NEON::BI__builtin_neon_vst1q_lane_bf16,
     NEON::BI__builtin_neon_vst1q_lane_v},
    {NEON::BI__builtin_neon_vst2_bf16, NEON::BI__builtin_neon_vst2_v},
    {NEON::BI__builtin_neon_vst2_lane_bf16, NEON::BI__builtin_neon_vst2_lane_v},
    {NEON::BI__builtin_neon_vst2q_bf16, NEON::BI__builtin_neon_vst2q_v},
    {NEON::BI__builtin_neon_vst2q_lane_bf16,
     NEON::BI__builtin_neon_vst2q_lane_v},
    {NEON::BI__builtin_neon_vst3_bf16, NEON::BI__builtin_neon_vst3_v},
    {NEON::BI__builtin_neon_vst3_lane_bf16, NEON::BI__builtin_neon_vst3_lane_v},
    {NEON::BI__builtin_neon_vst3q_bf16, NEON::BI__builtin_neon_vst3q_v},
    {NEON::BI__builtin_neon_vst3q_lane_bf16,
     NEON::BI__builtin_neon_vst3q_lane_v},
    {NEON::BI__builtin_neon_vst4_bf16, NEON::BI__builtin_neon_vst4_v},
    {NEON::BI__builtin_neon_vst4_lane_bf16, NEON::BI__builtin_neon_vst4_lane_v},
    {NEON::BI__builtin_neon_vst4q_bf16, NEON::BI__builtin_neon_vst4q_v},
    {NEON::BI__builtin_neon_vst4q_lane_bf16,
     NEON::BI__builtin_neon_vst4q_lane_v},
    // The mangling rules cause us to have one ID for each type for
    // vldap1(q)_lane and vstl1(q)_lane, but codegen is equivalent for all of
    // them. Choose an arbitrary one to be handled as tha canonical variation.
    {NEON::BI__builtin_neon_vldap1_lane_u64,
     NEON::BI__builtin_neon_vldap1_lane_s64},
    {NEON::BI__builtin_neon_vldap1_lane_f64,
     NEON::BI__builtin_neon_vldap1_lane_s64},
    {NEON::BI__builtin_neon_vldap1_lane_p64,
     NEON::BI__builtin_neon_vldap1_lane_s64},
    {NEON::BI__builtin_neon_vldap1q_lane_u64,
     NEON::BI__builtin_neon_vldap1q_lane_s64},
    {NEON::BI__builtin_neon_vldap1q_lane_f64,
     NEON::BI__builtin_neon_vldap1q_lane_s64},
    {NEON::BI__builtin_neon_vldap1q_lane_p64,
     NEON::BI__builtin_neon_vldap1q_lane_s64},
    {NEON::BI__builtin_neon_vstl1_lane_u64,
     NEON::BI__builtin_neon_vstl1_lane_s64},
    {NEON::BI__builtin_neon_vstl1_lane_f64,
     NEON::BI__builtin_neon_vstl1_lane_s64},
    {NEON::BI__builtin_neon_vstl1_lane_p64,
     NEON::BI__builtin_neon_vstl1_lane_s64},
    {NEON::BI__builtin_neon_vstl1q_lane_u64,
     NEON::BI__builtin_neon_vstl1q_lane_s64},
    {NEON::BI__builtin_neon_vstl1q_lane_f64,
     NEON::BI__builtin_neon_vstl1q_lane_s64},
    {NEON::BI__builtin_neon_vstl1q_lane_p64,
     NEON::BI__builtin_neon_vstl1q_lane_s64},
};

std::optional<mlir::Value>
CIRGenFunction::emitAArch64BuiltinExpr(unsigned builtinID, const CallExpr *expr,
                                       ReturnValueSlot returnValue,
                                       llvm::Triple::ArchType arch) {
  if (builtinID >= clang::AArch64::FirstSVEBuiltin &&
      builtinID <= clang::AArch64::LastSVEBuiltin)
    return emitAArch64SVEBuiltinExpr(builtinID, expr);

  if (builtinID >= clang::AArch64::FirstSMEBuiltin &&
      builtinID <= clang::AArch64::LastSMEBuiltin)
    return emitAArch64SMEBuiltinExpr(builtinID, expr);

  if (builtinID == Builtin::BI__builtin_cpu_supports) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  switch (builtinID) {
  default:
    break;
  case clang::AArch64::BI__builtin_arm_nop:
  case clang::AArch64::BI__builtin_arm_yield:
  case clang::AArch64::BI__yield:
  case clang::AArch64::BI__builtin_arm_wfe:
  case clang::AArch64::BI__wfe:
  case clang::AArch64::BI__builtin_arm_wfi:
  case clang::AArch64::BI__wfi:
  case clang::AArch64::BI__builtin_arm_sev:
  case clang::AArch64::BI__sev:
  case clang::AArch64::BI__builtin_arm_sevl:
  case clang::AArch64::BI__sevl:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_trap) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_get_sme_state) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_rbit) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }
  if (builtinID == clang::AArch64::BI__builtin_arm_rbit64) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_clz ||
      builtinID == clang::AArch64::BI__builtin_arm_clz64) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_cls) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }
  if (builtinID == clang::AArch64::BI__builtin_arm_cls64) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_rint32zf ||
      builtinID == clang::AArch64::BI__builtin_arm_rint32z) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_rint64zf ||
      builtinID == clang::AArch64::BI__builtin_arm_rint64z) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_rint32xf ||
      builtinID == clang::AArch64::BI__builtin_arm_rint32x) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_rint64xf ||
      builtinID == clang::AArch64::BI__builtin_arm_rint64x) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_jcvt) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_ld64b ||
      builtinID == clang::AArch64::BI__builtin_arm_st64b ||
      builtinID == clang::AArch64::BI__builtin_arm_st64bv ||
      builtinID == clang::AArch64::BI__builtin_arm_st64bv0) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_rndr ||
      builtinID == clang::AArch64::BI__builtin_arm_rndrrs) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__clear_cache) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if ((builtinID == clang::AArch64::BI__builtin_arm_ldrex ||
       builtinID == clang::AArch64::BI__builtin_arm_ldaex) &&
      getContext().getTypeSize(expr->getType()) == 128) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }
  if (builtinID == clang::AArch64::BI__builtin_arm_ldrex ||
      builtinID == clang::AArch64::BI__builtin_arm_ldaex) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if ((builtinID == clang::AArch64::BI__builtin_arm_strex ||
       builtinID == clang::AArch64::BI__builtin_arm_stlex) &&
      getContext().getTypeSize(expr->getArg(0)->getType()) == 128) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_strex ||
      builtinID == clang::AArch64::BI__builtin_arm_stlex) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__getReg) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__break) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_clrex) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI_ReadWriteBarrier) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  // CRC32
  Intrinsic::ID crcIntrinsicID = Intrinsic::not_intrinsic;
  switch (builtinID) {
  case clang::AArch64::BI__builtin_arm_crc32b:
    crcIntrinsicID = Intrinsic::aarch64_crc32b;
    break;
  case clang::AArch64::BI__builtin_arm_crc32cb:
    crcIntrinsicID = Intrinsic::aarch64_crc32cb;
    break;
  case clang::AArch64::BI__builtin_arm_crc32h:
    crcIntrinsicID = Intrinsic::aarch64_crc32h;
    break;
  case clang::AArch64::BI__builtin_arm_crc32ch:
    crcIntrinsicID = Intrinsic::aarch64_crc32ch;
    break;
  case clang::AArch64::BI__builtin_arm_crc32w:
    crcIntrinsicID = Intrinsic::aarch64_crc32w;
    break;
  case clang::AArch64::BI__builtin_arm_crc32cw:
    crcIntrinsicID = Intrinsic::aarch64_crc32cw;
    break;
  case clang::AArch64::BI__builtin_arm_crc32d:
    crcIntrinsicID = Intrinsic::aarch64_crc32x;
    break;
  case clang::AArch64::BI__builtin_arm_crc32cd:
    crcIntrinsicID = Intrinsic::aarch64_crc32cx;
    break;
  }

  if (crcIntrinsicID != Intrinsic::not_intrinsic) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  // Memory Operations (MOPS)
  if (builtinID == AArch64::BI__builtin_arm_mops_memset_tag) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  // Memory Tagging Extensions (MTE) Intrinsics
  Intrinsic::ID mteIntrinsicID = Intrinsic::not_intrinsic;
  switch (builtinID) {
  case clang::AArch64::BI__builtin_arm_irg:
    mteIntrinsicID = Intrinsic::aarch64_irg;
    break;
  case clang::AArch64::BI__builtin_arm_addg:
    mteIntrinsicID = Intrinsic::aarch64_addg;
    break;
  case clang::AArch64::BI__builtin_arm_gmi:
    mteIntrinsicID = Intrinsic::aarch64_gmi;
    break;
  case clang::AArch64::BI__builtin_arm_ldg:
    mteIntrinsicID = Intrinsic::aarch64_ldg;
    break;
  case clang::AArch64::BI__builtin_arm_stg:
    mteIntrinsicID = Intrinsic::aarch64_stg;
    break;
  case clang::AArch64::BI__builtin_arm_subp:
    mteIntrinsicID = Intrinsic::aarch64_subp;
    break;
  }

  if (mteIntrinsicID != Intrinsic::not_intrinsic) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_arm_rsr ||
      builtinID == clang::AArch64::BI__builtin_arm_rsr64 ||
      builtinID == clang::AArch64::BI__builtin_arm_rsr128 ||
      builtinID == clang::AArch64::BI__builtin_arm_rsrp ||
      builtinID == clang::AArch64::BI__builtin_arm_wsr ||
      builtinID == clang::AArch64::BI__builtin_arm_wsr64 ||
      builtinID == clang::AArch64::BI__builtin_arm_wsr128 ||
      builtinID == clang::AArch64::BI__builtin_arm_wsrp) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI_ReadStatusReg ||
      builtinID == clang::AArch64::BI_WriteStatusReg ||
      builtinID == clang::AArch64::BI__sys) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI_AddressOfReturnAddress) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__builtin_sponentry) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == clang::AArch64::BI__mulh ||
      builtinID == clang::AArch64::BI__umulh) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == AArch64::BI__writex18byte ||
      builtinID == AArch64::BI__writex18word ||
      builtinID == AArch64::BI__writex18dword ||
      builtinID == AArch64::BI__writex18qword) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == AArch64::BI__readx18byte ||
      builtinID == AArch64::BI__readx18word ||
      builtinID == AArch64::BI__readx18dword ||
      builtinID == AArch64::BI__readx18qword) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == AArch64::BI__addx18byte ||
      builtinID == AArch64::BI__addx18word ||
      builtinID == AArch64::BI__addx18dword ||
      builtinID == AArch64::BI__addx18qword ||
      builtinID == AArch64::BI__incx18byte ||
      builtinID == AArch64::BI__incx18word ||
      builtinID == AArch64::BI__incx18dword ||
      builtinID == AArch64::BI__incx18qword) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == AArch64::BI_CopyDoubleFromInt64 ||
      builtinID == AArch64::BI_CopyFloatFromInt32 ||
      builtinID == AArch64::BI_CopyInt32FromFloat ||
      builtinID == AArch64::BI_CopyInt64FromDouble) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == AArch64::BI_CountLeadingOnes ||
      builtinID == AArch64::BI_CountLeadingOnes64 ||
      builtinID == AArch64::BI_CountLeadingZeros ||
      builtinID == AArch64::BI_CountLeadingZeros64) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == AArch64::BI_CountLeadingSigns ||
      builtinID == AArch64::BI_CountLeadingSigns64) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == AArch64::BI_CountOneBits ||
      builtinID == AArch64::BI_CountOneBits64) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == AArch64::BI__prefetch) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == AArch64::BI__hlt) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  if (builtinID == NEON::BI__builtin_neon_vcvth_bf16_f32) {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  // Handle MSVC intrinsics before argument evaluation to prevent double
  // evaluation.
  assert(!cir::MissingFeatures::msvcBuiltins());

  // Some intrinsics are equivalent - if they are use the base intrinsic ID.
  auto it = llvm::find_if(neonEquivalentIntrinsicMap, [builtinID](auto &p) {
    return p.first == builtinID;
  });
  if (it != end(neonEquivalentIntrinsicMap))
    builtinID = it->second;

  // Find out if any arguments are required to be integer constant
  // expressions.
  assert(!cir::MissingFeatures::handleBuiltinICEArguments());

  assert(!cir::MissingFeatures::neonSISDIntrinsics());

  llvm::SmallVector<mlir::Value> ops;
  mlir::Location loc = getLoc(expr->getExprLoc());

  // Handle non-overloaded intrinsics first.
  switch (builtinID) {
  default:
    break;
  case NEON::BI__builtin_neon_vabsh_f16: {
    ops.push_back(emitScalarExpr(expr->getArg(0)));
    return cir::FAbsOp::create(builder, loc, ops);
  }
  case NEON::BI__builtin_neon_vaddq_p128:
  case NEON::BI__builtin_neon_vldrq_p128:
  case NEON::BI__builtin_neon_vstrq_p128:
  case NEON::BI__builtin_neon_vcvts_f32_u32:
  case NEON::BI__builtin_neon_vcvtd_f64_u64:
  case NEON::BI__builtin_neon_vcvts_f32_s32:
  case NEON::BI__builtin_neon_vcvtd_f64_s64:
  case NEON::BI__builtin_neon_vcvth_f16_u16:
  case NEON::BI__builtin_neon_vcvth_f16_u32:
  case NEON::BI__builtin_neon_vcvth_f16_u64:
  case NEON::BI__builtin_neon_vcvth_f16_s16:
  case NEON::BI__builtin_neon_vcvth_f16_s32:
  case NEON::BI__builtin_neon_vcvth_f16_s64:
  case NEON::BI__builtin_neon_vcvtah_u16_f16:
  case NEON::BI__builtin_neon_vcvtmh_u16_f16:
  case NEON::BI__builtin_neon_vcvtnh_u16_f16:
  case NEON::BI__builtin_neon_vcvtph_u16_f16:
  case NEON::BI__builtin_neon_vcvth_u16_f16:
  case NEON::BI__builtin_neon_vcvtah_s16_f16:
  case NEON::BI__builtin_neon_vcvtmh_s16_f16:
  case NEON::BI__builtin_neon_vcvtnh_s16_f16:
  case NEON::BI__builtin_neon_vcvtph_s16_f16:
  case NEON::BI__builtin_neon_vcvth_s16_f16:
  case NEON::BI__builtin_neon_vcaleh_f16:
  case NEON::BI__builtin_neon_vcalth_f16:
  case NEON::BI__builtin_neon_vcageh_f16:
  case NEON::BI__builtin_neon_vcagth_f16:
  case NEON::BI__builtin_neon_vcvth_n_s16_f16:
  case NEON::BI__builtin_neon_vcvth_n_u16_f16:
  case NEON::BI__builtin_neon_vcvth_n_f16_s16:
  case NEON::BI__builtin_neon_vcvth_n_f16_u16:
  case NEON::BI__builtin_neon_vpaddd_s64:
  case NEON::BI__builtin_neon_vpaddd_f64:
  case NEON::BI__builtin_neon_vpadds_f32:
  case NEON::BI__builtin_neon_vceqzd_s64:
  case NEON::BI__builtin_neon_vceqzd_f64:
  case NEON::BI__builtin_neon_vceqzs_f32:
  case NEON::BI__builtin_neon_vceqzh_f16:
  case NEON::BI__builtin_neon_vcgezd_s64:
  case NEON::BI__builtin_neon_vcgezd_f64:
  case NEON::BI__builtin_neon_vcgezs_f32:
  case NEON::BI__builtin_neon_vcgezh_f16:
  case NEON::BI__builtin_neon_vclezd_s64:
  case NEON::BI__builtin_neon_vclezd_f64:
  case NEON::BI__builtin_neon_vclezs_f32:
  case NEON::BI__builtin_neon_vclezh_f16:
  case NEON::BI__builtin_neon_vcgtzd_s64:
  case NEON::BI__builtin_neon_vcgtzd_f64:
  case NEON::BI__builtin_neon_vcgtzs_f32:
  case NEON::BI__builtin_neon_vcgtzh_f16:
  case NEON::BI__builtin_neon_vcltzd_s64:
  case NEON::BI__builtin_neon_vcltzd_f64:
  case NEON::BI__builtin_neon_vcltzs_f32:
  case NEON::BI__builtin_neon_vcltzh_f16:
  case NEON::BI__builtin_neon_vceqzd_u64:
  case NEON::BI__builtin_neon_vceqd_f64:
  case NEON::BI__builtin_neon_vcled_f64:
  case NEON::BI__builtin_neon_vcltd_f64:
  case NEON::BI__builtin_neon_vcged_f64:
  case NEON::BI__builtin_neon_vcgtd_f64:
  case NEON::BI__builtin_neon_vceqs_f32:
  case NEON::BI__builtin_neon_vcles_f32:
  case NEON::BI__builtin_neon_vclts_f32:
  case NEON::BI__builtin_neon_vcges_f32:
  case NEON::BI__builtin_neon_vcgts_f32:
  case NEON::BI__builtin_neon_vceqh_f16:
  case NEON::BI__builtin_neon_vcleh_f16:
  case NEON::BI__builtin_neon_vclth_f16:
  case NEON::BI__builtin_neon_vcgeh_f16:
  case NEON::BI__builtin_neon_vcgth_f16:
  case NEON::BI__builtin_neon_vceqd_s64:
  case NEON::BI__builtin_neon_vceqd_u64:
  case NEON::BI__builtin_neon_vcgtd_s64:
  case NEON::BI__builtin_neon_vcgtd_u64:
  case NEON::BI__builtin_neon_vcltd_s64:
  case NEON::BI__builtin_neon_vcltd_u64:
  case NEON::BI__builtin_neon_vcged_u64:
  case NEON::BI__builtin_neon_vcged_s64:
  case NEON::BI__builtin_neon_vcled_u64:
  case NEON::BI__builtin_neon_vcled_s64:
  case NEON::BI__builtin_neon_vtstd_s64:
  case NEON::BI__builtin_neon_vtstd_u64:
  case NEON::BI__builtin_neon_vset_lane_i8:
  case NEON::BI__builtin_neon_vset_lane_i16:
  case NEON::BI__builtin_neon_vset_lane_i32:
  case NEON::BI__builtin_neon_vset_lane_i64:
  case NEON::BI__builtin_neon_vset_lane_bf16:
  case NEON::BI__builtin_neon_vset_lane_f32:
  case NEON::BI__builtin_neon_vsetq_lane_i8:
  case NEON::BI__builtin_neon_vsetq_lane_i16:
  case NEON::BI__builtin_neon_vsetq_lane_i32:
  case NEON::BI__builtin_neon_vsetq_lane_i64:
  case NEON::BI__builtin_neon_vsetq_lane_bf16:
  case NEON::BI__builtin_neon_vsetq_lane_f32:
  case NEON::BI__builtin_neon_vset_lane_f64:
  case NEON::BI__builtin_neon_vset_lane_mf8:
  case NEON::BI__builtin_neon_vsetq_lane_mf8:
  case NEON::BI__builtin_neon_vsetq_lane_f64:
  case NEON::BI__builtin_neon_vget_lane_i8:
  case NEON::BI__builtin_neon_vdupb_lane_i8:
  case NEON::BI__builtin_neon_vgetq_lane_i8:
  case NEON::BI__builtin_neon_vdupb_laneq_i8:
  case NEON::BI__builtin_neon_vget_lane_mf8:
  case NEON::BI__builtin_neon_vdupb_lane_mf8:
  case NEON::BI__builtin_neon_vgetq_lane_mf8:
  case NEON::BI__builtin_neon_vdupb_laneq_mf8:
  case NEON::BI__builtin_neon_vget_lane_i16:
  case NEON::BI__builtin_neon_vduph_lane_i16:
  case NEON::BI__builtin_neon_vgetq_lane_i16:
  case NEON::BI__builtin_neon_vduph_laneq_i16:
  case NEON::BI__builtin_neon_vget_lane_i32:
  case NEON::BI__builtin_neon_vdups_lane_i32:
  case NEON::BI__builtin_neon_vdups_lane_f32:
  case NEON::BI__builtin_neon_vgetq_lane_i32:
  case NEON::BI__builtin_neon_vdups_laneq_i32:
  case NEON::BI__builtin_neon_vget_lane_i64:
  case NEON::BI__builtin_neon_vdupd_lane_i64:
  case NEON::BI__builtin_neon_vdupd_lane_f64:
  case NEON::BI__builtin_neon_vgetq_lane_i64:
  case NEON::BI__builtin_neon_vdupd_laneq_i64:
  case NEON::BI__builtin_neon_vget_lane_f32:
  case NEON::BI__builtin_neon_vget_lane_f64:
  case NEON::BI__builtin_neon_vgetq_lane_f32:
  case NEON::BI__builtin_neon_vdups_laneq_f32:
  case NEON::BI__builtin_neon_vgetq_lane_f64:
  case NEON::BI__builtin_neon_vdupd_laneq_f64:
  case NEON::BI__builtin_neon_vaddh_f16:
  case NEON::BI__builtin_neon_vsubh_f16:
  case NEON::BI__builtin_neon_vmulh_f16:
  case NEON::BI__builtin_neon_vdivh_f16:
  case NEON::BI__builtin_neon_vfmah_f16:
  case NEON::BI__builtin_neon_vfmsh_f16:
  case NEON::BI__builtin_neon_vaddd_s64:
  case NEON::BI__builtin_neon_vaddd_u64:
  case NEON::BI__builtin_neon_vsubd_s64:
  case NEON::BI__builtin_neon_vsubd_u64:
  case NEON::BI__builtin_neon_vqdmlalh_s16:
  case NEON::BI__builtin_neon_vqdmlslh_s16:
  case NEON::BI__builtin_neon_vqshlud_n_s64:
  case NEON::BI__builtin_neon_vqshld_n_u64:
  case NEON::BI__builtin_neon_vqshld_n_s64:
  case NEON::BI__builtin_neon_vrshrd_n_u64:
  case NEON::BI__builtin_neon_vrshrd_n_s64:
  case NEON::BI__builtin_neon_vrsrad_n_u64:
  case NEON::BI__builtin_neon_vrsrad_n_s64:
  case NEON::BI__builtin_neon_vshld_n_s64:
  case NEON::BI__builtin_neon_vshld_n_u64:
  case NEON::BI__builtin_neon_vshrd_n_s64:
  case NEON::BI__builtin_neon_vshrd_n_u64:
  case NEON::BI__builtin_neon_vsrad_n_s64:
  case NEON::BI__builtin_neon_vsrad_n_u64:
  case NEON::BI__builtin_neon_vqdmlalh_lane_s16:
  case NEON::BI__builtin_neon_vqdmlalh_laneq_s16:
  case NEON::BI__builtin_neon_vqdmlslh_lane_s16:
  case NEON::BI__builtin_neon_vqdmlslh_laneq_s16:
  case NEON::BI__builtin_neon_vqdmlals_s32:
  case NEON::BI__builtin_neon_vqdmlsls_s32:
  case NEON::BI__builtin_neon_vqdmlals_lane_s32:
  case NEON::BI__builtin_neon_vqdmlals_laneq_s32:
  case NEON::BI__builtin_neon_vqdmlsls_lane_s32:
  case NEON::BI__builtin_neon_vqdmlsls_laneq_s32:
  case NEON::BI__builtin_neon_vget_lane_bf16:
  case NEON::BI__builtin_neon_vduph_lane_bf16:
  case NEON::BI__builtin_neon_vduph_lane_f16:
  case NEON::BI__builtin_neon_vgetq_lane_bf16:
  case NEON::BI__builtin_neon_vduph_laneq_bf16:
  case NEON::BI__builtin_neon_vduph_laneq_f16:
  case NEON::BI__builtin_neon_vcvt_bf16_f32:
  case NEON::BI__builtin_neon_vcvtq_low_bf16_f32:
  case NEON::BI__builtin_neon_vcvtq_high_bf16_f32:
  case clang::AArch64::BI_InterlockedAdd:
  case clang::AArch64::BI_InterlockedAdd_acq:
  case clang::AArch64::BI_InterlockedAdd_rel:
  case clang::AArch64::BI_InterlockedAdd_nf:
  case clang::AArch64::BI_InterlockedAdd64:
  case clang::AArch64::BI_InterlockedAdd64_acq:
  case clang::AArch64::BI_InterlockedAdd64_rel:
  case clang::AArch64::BI_InterlockedAdd64_nf:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  // Not all intrinsics handled by the common case work for AArch64 yet, so only
  // defer to common code if it's been added to our special map.
  assert(!cir::MissingFeatures::aarch64SIMDIntrinsics());

  assert(!cir::MissingFeatures::aarch64TblBuiltinExpr());

  switch (builtinID) {
  default:
    return std::nullopt;
  case NEON::BI__builtin_neon_vbsl_v:
  case NEON::BI__builtin_neon_vbslq_v:
  case NEON::BI__builtin_neon_vfma_lane_v:
  case NEON::BI__builtin_neon_vfmaq_lane_v:
  case NEON::BI__builtin_neon_vfma_laneq_v:
  case NEON::BI__builtin_neon_vfmaq_laneq_v:
  case NEON::BI__builtin_neon_vfmah_lane_f16:
  case NEON::BI__builtin_neon_vfmas_lane_f32:
  case NEON::BI__builtin_neon_vfmah_laneq_f16:
  case NEON::BI__builtin_neon_vfmas_laneq_f32:
  case NEON::BI__builtin_neon_vfmad_lane_f64:
  case NEON::BI__builtin_neon_vfmad_laneq_f64:
  case NEON::BI__builtin_neon_vmull_v:
  case NEON::BI__builtin_neon_vmax_v:
  case NEON::BI__builtin_neon_vmaxq_v:
  case NEON::BI__builtin_neon_vmaxh_f16:
  case NEON::BI__builtin_neon_vmin_v:
  case NEON::BI__builtin_neon_vminq_v:
  case NEON::BI__builtin_neon_vminh_f16:
  case NEON::BI__builtin_neon_vabd_v:
  case NEON::BI__builtin_neon_vabdq_v:
  case NEON::BI__builtin_neon_vpadal_v:
  case NEON::BI__builtin_neon_vpadalq_v:
  case NEON::BI__builtin_neon_vpmin_v:
  case NEON::BI__builtin_neon_vpminq_v:
  case NEON::BI__builtin_neon_vpmax_v:
  case NEON::BI__builtin_neon_vpmaxq_v:
  case NEON::BI__builtin_neon_vminnm_v:
  case NEON::BI__builtin_neon_vminnmq_v:
  case NEON::BI__builtin_neon_vminnmh_f16:
  case NEON::BI__builtin_neon_vmaxnm_v:
  case NEON::BI__builtin_neon_vmaxnmq_v:
  case NEON::BI__builtin_neon_vmaxnmh_f16:
  case NEON::BI__builtin_neon_vrecpss_f32:
  case NEON::BI__builtin_neon_vrecpsd_f64:
  case NEON::BI__builtin_neon_vrecpsh_f16:
  case NEON::BI__builtin_neon_vqshrun_n_v:
  case NEON::BI__builtin_neon_vqrshrun_n_v:
  case NEON::BI__builtin_neon_vqshrn_n_v:
  case NEON::BI__builtin_neon_vrshrn_n_v:
  case NEON::BI__builtin_neon_vqrshrn_n_v:
  case NEON::BI__builtin_neon_vrndah_f16:
  case NEON::BI__builtin_neon_vrnda_v:
  case NEON::BI__builtin_neon_vrndaq_v:
  case NEON::BI__builtin_neon_vrndih_f16:
  case NEON::BI__builtin_neon_vrndmh_f16:
  case NEON::BI__builtin_neon_vrndm_v:
  case NEON::BI__builtin_neon_vrndmq_v:
  case NEON::BI__builtin_neon_vrndnh_f16:
  case NEON::BI__builtin_neon_vrndn_v:
  case NEON::BI__builtin_neon_vrndnq_v:
  case NEON::BI__builtin_neon_vrndns_f32:
  case NEON::BI__builtin_neon_vrndph_f16:
  case NEON::BI__builtin_neon_vrndp_v:
  case NEON::BI__builtin_neon_vrndpq_v:
  case NEON::BI__builtin_neon_vrndxh_f16:
  case NEON::BI__builtin_neon_vrndx_v:
  case NEON::BI__builtin_neon_vrndxq_v:
  case NEON::BI__builtin_neon_vrndh_f16:
  case NEON::BI__builtin_neon_vrnd32x_f32:
  case NEON::BI__builtin_neon_vrnd32xq_f32:
  case NEON::BI__builtin_neon_vrnd32x_f64:
  case NEON::BI__builtin_neon_vrnd32xq_f64:
  case NEON::BI__builtin_neon_vrnd32z_f32:
  case NEON::BI__builtin_neon_vrnd32zq_f32:
  case NEON::BI__builtin_neon_vrnd32z_f64:
  case NEON::BI__builtin_neon_vrnd32zq_f64:
  case NEON::BI__builtin_neon_vrnd64x_f32:
  case NEON::BI__builtin_neon_vrnd64xq_f32:
  case NEON::BI__builtin_neon_vrnd64x_f64:
  case NEON::BI__builtin_neon_vrnd64xq_f64:
  case NEON::BI__builtin_neon_vrnd64z_f32:
  case NEON::BI__builtin_neon_vrnd64zq_f32:
  case NEON::BI__builtin_neon_vrnd64z_f64:
  case NEON::BI__builtin_neon_vrnd64zq_f64:
  case NEON::BI__builtin_neon_vrnd_v:
  case NEON::BI__builtin_neon_vrndq_v:
  case NEON::BI__builtin_neon_vcvt_f64_v:
  case NEON::BI__builtin_neon_vcvtq_f64_v:
  case NEON::BI__builtin_neon_vcvt_f64_f32:
  case NEON::BI__builtin_neon_vcvt_f32_f64:
  case NEON::BI__builtin_neon_vcvt_s32_v:
  case NEON::BI__builtin_neon_vcvt_u32_v:
  case NEON::BI__builtin_neon_vcvt_s64_v:
  case NEON::BI__builtin_neon_vcvt_u64_v:
  case NEON::BI__builtin_neon_vcvt_s16_f16:
  case NEON::BI__builtin_neon_vcvt_u16_f16:
  case NEON::BI__builtin_neon_vcvtq_s32_v:
  case NEON::BI__builtin_neon_vcvtq_u32_v:
  case NEON::BI__builtin_neon_vcvtq_s64_v:
  case NEON::BI__builtin_neon_vcvtq_u64_v:
  case NEON::BI__builtin_neon_vcvtq_s16_f16:
  case NEON::BI__builtin_neon_vcvtq_u16_f16:
  case NEON::BI__builtin_neon_vcvta_s16_f16:
  case NEON::BI__builtin_neon_vcvta_u16_f16:
  case NEON::BI__builtin_neon_vcvta_s32_v:
  case NEON::BI__builtin_neon_vcvtaq_s16_f16:
  case NEON::BI__builtin_neon_vcvtaq_s32_v:
  case NEON::BI__builtin_neon_vcvta_u32_v:
  case NEON::BI__builtin_neon_vcvtaq_u16_f16:
  case NEON::BI__builtin_neon_vcvtaq_u32_v:
  case NEON::BI__builtin_neon_vcvta_s64_v:
  case NEON::BI__builtin_neon_vcvtaq_s64_v:
  case NEON::BI__builtin_neon_vcvta_u64_v:
  case NEON::BI__builtin_neon_vcvtaq_u64_v:
  case NEON::BI__builtin_neon_vcvtm_s16_f16:
  case NEON::BI__builtin_neon_vcvtm_s32_v:
  case NEON::BI__builtin_neon_vcvtmq_s16_f16:
  case NEON::BI__builtin_neon_vcvtmq_s32_v:
  case NEON::BI__builtin_neon_vcvtm_u16_f16:
  case NEON::BI__builtin_neon_vcvtm_u32_v:
  case NEON::BI__builtin_neon_vcvtmq_u16_f16:
  case NEON::BI__builtin_neon_vcvtmq_u32_v:
  case NEON::BI__builtin_neon_vcvtm_s64_v:
  case NEON::BI__builtin_neon_vcvtmq_s64_v:
  case NEON::BI__builtin_neon_vcvtm_u64_v:
  case NEON::BI__builtin_neon_vcvtmq_u64_v:
  case NEON::BI__builtin_neon_vcvtn_s16_f16:
  case NEON::BI__builtin_neon_vcvtn_s32_v:
  case NEON::BI__builtin_neon_vcvtnq_s16_f16:
  case NEON::BI__builtin_neon_vcvtnq_s32_v:
  case NEON::BI__builtin_neon_vcvtn_u16_f16:
  case NEON::BI__builtin_neon_vcvtn_u32_v:
  case NEON::BI__builtin_neon_vcvtnq_u16_f16:
  case NEON::BI__builtin_neon_vcvtnq_u32_v:
  case NEON::BI__builtin_neon_vcvtn_s64_v:
  case NEON::BI__builtin_neon_vcvtnq_s64_v:
  case NEON::BI__builtin_neon_vcvtn_u64_v:
  case NEON::BI__builtin_neon_vcvtnq_u64_v:
  case NEON::BI__builtin_neon_vcvtp_s16_f16:
  case NEON::BI__builtin_neon_vcvtp_s32_v:
  case NEON::BI__builtin_neon_vcvtpq_s16_f16:
  case NEON::BI__builtin_neon_vcvtpq_s32_v:
  case NEON::BI__builtin_neon_vcvtp_u16_f16:
  case NEON::BI__builtin_neon_vcvtp_u32_v:
  case NEON::BI__builtin_neon_vcvtpq_u16_f16:
  case NEON::BI__builtin_neon_vcvtpq_u32_v:
  case NEON::BI__builtin_neon_vcvtp_s64_v:
  case NEON::BI__builtin_neon_vcvtpq_s64_v:
  case NEON::BI__builtin_neon_vcvtp_u64_v:
  case NEON::BI__builtin_neon_vcvtpq_u64_v:
  case NEON::BI__builtin_neon_vmulx_v:
  case NEON::BI__builtin_neon_vmulxq_v:
  case NEON::BI__builtin_neon_vmulxh_lane_f16:
  case NEON::BI__builtin_neon_vmulxh_laneq_f16:
  case NEON::BI__builtin_neon_vmul_lane_v:
  case NEON::BI__builtin_neon_vmul_laneq_v:
  case NEON::BI__builtin_neon_vnegd_s64:
  case NEON::BI__builtin_neon_vnegh_f16:
  case NEON::BI__builtin_neon_vpmaxnm_v:
  case NEON::BI__builtin_neon_vpmaxnmq_v:
  case NEON::BI__builtin_neon_vpminnm_v:
  case NEON::BI__builtin_neon_vpminnmq_v:
  case NEON::BI__builtin_neon_vsqrth_f16:
  case NEON::BI__builtin_neon_vsqrt_v:
  case NEON::BI__builtin_neon_vsqrtq_v:
  case NEON::BI__builtin_neon_vrbit_v:
  case NEON::BI__builtin_neon_vrbitq_v:
  case NEON::BI__builtin_neon_vmaxv_f16:
  case NEON::BI__builtin_neon_vmaxvq_f16:
  case NEON::BI__builtin_neon_vminv_f16:
  case NEON::BI__builtin_neon_vminvq_f16:
  case NEON::BI__builtin_neon_vmaxnmv_f16:
  case NEON::BI__builtin_neon_vmaxnmvq_f16:
  case NEON::BI__builtin_neon_vminnmv_f16:
  case NEON::BI__builtin_neon_vminnmvq_f16:
  case NEON::BI__builtin_neon_vmul_n_f64:
  case NEON::BI__builtin_neon_vaddlv_u8:
  case NEON::BI__builtin_neon_vaddlv_u16:
  case NEON::BI__builtin_neon_vaddlvq_u8:
  case NEON::BI__builtin_neon_vaddlvq_u16:
  case NEON::BI__builtin_neon_vaddlv_s8:
  case NEON::BI__builtin_neon_vaddlv_s16:
  case NEON::BI__builtin_neon_vaddlvq_s8:
  case NEON::BI__builtin_neon_vaddlvq_s16:
  case NEON::BI__builtin_neon_vsri_n_v:
  case NEON::BI__builtin_neon_vsriq_n_v:
  case NEON::BI__builtin_neon_vsli_n_v:
  case NEON::BI__builtin_neon_vsliq_n_v:
  case NEON::BI__builtin_neon_vsra_n_v:
  case NEON::BI__builtin_neon_vsraq_n_v:
  case NEON::BI__builtin_neon_vrsra_n_v:
  case NEON::BI__builtin_neon_vrsraq_n_v:
  case NEON::BI__builtin_neon_vld1_v:
  case NEON::BI__builtin_neon_vld1q_v:
  case NEON::BI__builtin_neon_vst1_v:
  case NEON::BI__builtin_neon_vst1q_v:
  case NEON::BI__builtin_neon_vld1_lane_v:
  case NEON::BI__builtin_neon_vld1q_lane_v:
  case NEON::BI__builtin_neon_vldap1_lane_s64:
  case NEON::BI__builtin_neon_vldap1q_lane_s64:
  case NEON::BI__builtin_neon_vld1_dup_v:
  case NEON::BI__builtin_neon_vld1q_dup_v:
  case NEON::BI__builtin_neon_vst1_lane_v:
  case NEON::BI__builtin_neon_vst1q_lane_v:
  case NEON::BI__builtin_neon_vstl1_lane_s64:
  case NEON::BI__builtin_neon_vstl1q_lane_s64:
  case NEON::BI__builtin_neon_vld2_v:
  case NEON::BI__builtin_neon_vld2q_v:
  case NEON::BI__builtin_neon_vld3_v:
  case NEON::BI__builtin_neon_vld3q_v:
  case NEON::BI__builtin_neon_vld4_v:
  case NEON::BI__builtin_neon_vld4q_v:
  case NEON::BI__builtin_neon_vld2_dup_v:
  case NEON::BI__builtin_neon_vld2q_dup_v:
  case NEON::BI__builtin_neon_vld3_dup_v:
  case NEON::BI__builtin_neon_vld3q_dup_v:
  case NEON::BI__builtin_neon_vld4_dup_v:
  case NEON::BI__builtin_neon_vld4q_dup_v:
  case NEON::BI__builtin_neon_vld2_lane_v:
  case NEON::BI__builtin_neon_vld2q_lane_v:
  case NEON::BI__builtin_neon_vld3_lane_v:
  case NEON::BI__builtin_neon_vld3q_lane_v:
  case NEON::BI__builtin_neon_vld4_lane_v:
  case NEON::BI__builtin_neon_vld4q_lane_v:
  case NEON::BI__builtin_neon_vst2_v:
  case NEON::BI__builtin_neon_vst2q_v:
  case NEON::BI__builtin_neon_vst2_lane_v:
  case NEON::BI__builtin_neon_vst2q_lane_v:
  case NEON::BI__builtin_neon_vst3_v:
  case NEON::BI__builtin_neon_vst3q_v:
  case NEON::BI__builtin_neon_vst3_lane_v:
  case NEON::BI__builtin_neon_vst3q_lane_v:
  case NEON::BI__builtin_neon_vst4_v:
  case NEON::BI__builtin_neon_vst4q_v:
  case NEON::BI__builtin_neon_vst4_lane_v:
  case NEON::BI__builtin_neon_vst4q_lane_v:
  case NEON::BI__builtin_neon_vtrn_v:
  case NEON::BI__builtin_neon_vtrnq_v:
  case NEON::BI__builtin_neon_vuzp_v:
  case NEON::BI__builtin_neon_vuzpq_v:
  case NEON::BI__builtin_neon_vzip_v:
  case NEON::BI__builtin_neon_vzipq_v:
  case NEON::BI__builtin_neon_vqtbl1q_v:
  case NEON::BI__builtin_neon_vqtbl2q_v:
  case NEON::BI__builtin_neon_vqtbl3q_v:
  case NEON::BI__builtin_neon_vqtbl4q_v:
  case NEON::BI__builtin_neon_vqtbx1q_v:
  case NEON::BI__builtin_neon_vqtbx2q_v:
  case NEON::BI__builtin_neon_vqtbx3q_v:
  case NEON::BI__builtin_neon_vqtbx4q_v:
  case NEON::BI__builtin_neon_vsqadd_v:
  case NEON::BI__builtin_neon_vsqaddq_v:
  case NEON::BI__builtin_neon_vuqadd_v:
  case NEON::BI__builtin_neon_vuqaddq_v:
  case NEON::BI__builtin_neon_vluti2_laneq_mf8:
  case NEON::BI__builtin_neon_vluti2_laneq_bf16:
  case NEON::BI__builtin_neon_vluti2_laneq_f16:
  case NEON::BI__builtin_neon_vluti2_laneq_p16:
  case NEON::BI__builtin_neon_vluti2_laneq_p8:
  case NEON::BI__builtin_neon_vluti2_laneq_s16:
  case NEON::BI__builtin_neon_vluti2_laneq_s8:
  case NEON::BI__builtin_neon_vluti2_laneq_u16:
  case NEON::BI__builtin_neon_vluti2_laneq_u8:
  case NEON::BI__builtin_neon_vluti2q_laneq_mf8:
  case NEON::BI__builtin_neon_vluti2q_laneq_bf16:
  case NEON::BI__builtin_neon_vluti2q_laneq_f16:
  case NEON::BI__builtin_neon_vluti2q_laneq_p16:
  case NEON::BI__builtin_neon_vluti2q_laneq_p8:
  case NEON::BI__builtin_neon_vluti2q_laneq_s16:
  case NEON::BI__builtin_neon_vluti2q_laneq_s8:
  case NEON::BI__builtin_neon_vluti2q_laneq_u16:
  case NEON::BI__builtin_neon_vluti2q_laneq_u8:
  case NEON::BI__builtin_neon_vluti2_lane_mf8:
  case NEON::BI__builtin_neon_vluti2_lane_bf16:
  case NEON::BI__builtin_neon_vluti2_lane_f16:
  case NEON::BI__builtin_neon_vluti2_lane_p16:
  case NEON::BI__builtin_neon_vluti2_lane_p8:
  case NEON::BI__builtin_neon_vluti2_lane_s16:
  case NEON::BI__builtin_neon_vluti2_lane_s8:
  case NEON::BI__builtin_neon_vluti2_lane_u16:
  case NEON::BI__builtin_neon_vluti2_lane_u8:
  case NEON::BI__builtin_neon_vluti2q_lane_mf8:
  case NEON::BI__builtin_neon_vluti2q_lane_bf16:
  case NEON::BI__builtin_neon_vluti2q_lane_f16:
  case NEON::BI__builtin_neon_vluti2q_lane_p16:
  case NEON::BI__builtin_neon_vluti2q_lane_p8:
  case NEON::BI__builtin_neon_vluti2q_lane_s16:
  case NEON::BI__builtin_neon_vluti2q_lane_s8:
  case NEON::BI__builtin_neon_vluti2q_lane_u16:
  case NEON::BI__builtin_neon_vluti2q_lane_u8:
  case NEON::BI__builtin_neon_vluti4q_lane_mf8:
  case NEON::BI__builtin_neon_vluti4q_lane_p8:
  case NEON::BI__builtin_neon_vluti4q_lane_s8:
  case NEON::BI__builtin_neon_vluti4q_lane_u8:
  case NEON::BI__builtin_neon_vluti4q_laneq_mf8:
  case NEON::BI__builtin_neon_vluti4q_laneq_p8:
  case NEON::BI__builtin_neon_vluti4q_laneq_s8:
  case NEON::BI__builtin_neon_vluti4q_laneq_u8:
  case NEON::BI__builtin_neon_vluti4q_lane_bf16_x2:
  case NEON::BI__builtin_neon_vluti4q_lane_f16_x2:
  case NEON::BI__builtin_neon_vluti4q_lane_p16_x2:
  case NEON::BI__builtin_neon_vluti4q_lane_s16_x2:
  case NEON::BI__builtin_neon_vluti4q_lane_u16_x2:
  case NEON::BI__builtin_neon_vluti4q_laneq_bf16_x2:
  case NEON::BI__builtin_neon_vluti4q_laneq_f16_x2:
  case NEON::BI__builtin_neon_vluti4q_laneq_p16_x2:
  case NEON::BI__builtin_neon_vluti4q_laneq_s16_x2:
  case NEON::BI__builtin_neon_vluti4q_laneq_u16_x2:
  case NEON::BI__builtin_neon_vmmlaq_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vmmlaq_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt1_low_bf16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt1_bf16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt1_high_bf16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt2_low_bf16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt2_bf16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt2_high_bf16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt1_low_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt1_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt1_high_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt2_low_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt2_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt2_high_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt_mf8_f32_fpm:
  case NEON::BI__builtin_neon_vcvt_mf8_f16_fpm:
  case NEON::BI__builtin_neon_vcvtq_mf8_f16_fpm:
  case NEON::BI__builtin_neon_vcvt_high_mf8_f32_fpm:
  case NEON::BI__builtin_neon_vdot_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vdotq_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vdot_lane_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vdotq_lane_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vdot_laneq_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vdotq_laneq_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vdot_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vdotq_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vdot_lane_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vdotq_lane_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vdot_laneq_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vdotq_laneq_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vmlalbq_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vmlaltq_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vmlallbbq_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vmlallbtq_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vmlalltbq_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vmlallttq_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vmlalbq_lane_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vmlalbq_laneq_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vmlaltq_lane_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vmlaltq_laneq_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vmlallbbq_lane_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vmlallbbq_laneq_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vmlallbtq_lane_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vmlallbtq_laneq_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vmlalltbq_lane_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vmlalltbq_laneq_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vmlallttq_lane_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vmlallttq_laneq_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vamin_f16:
  case NEON::BI__builtin_neon_vaminq_f16:
  case NEON::BI__builtin_neon_vamin_f32:
  case NEON::BI__builtin_neon_vaminq_f32:
  case NEON::BI__builtin_neon_vaminq_f64:
  case NEON::BI__builtin_neon_vamax_f16:
  case NEON::BI__builtin_neon_vamaxq_f16:
  case NEON::BI__builtin_neon_vamax_f32:
  case NEON::BI__builtin_neon_vamaxq_f32:
  case NEON::BI__builtin_neon_vamaxq_f64:
  case NEON::BI__builtin_neon_vscale_f16:
  case NEON::BI__builtin_neon_vscaleq_f16:
  case NEON::BI__builtin_neon_vscale_f32:
  case NEON::BI__builtin_neon_vscaleq_f32:
  case NEON::BI__builtin_neon_vscaleq_f64:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AArch64 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  // Unreachable: All cases in the switch above return.
}
