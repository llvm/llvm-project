//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit RISC-V Builtin calls as CIR or a function call
// to be later resolved.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "clang/Basic/TargetBuiltins.h"

using namespace clang;
using namespace clang::CIRGen;

std::optional<mlir::Value>
CIRGenFunction::emitRISCVBuiltinExpr(unsigned builtinID, const CallExpr *e) {
  if (builtinID == Builtin::BI__builtin_cpu_supports ||
      builtinID == Builtin::BI__builtin_cpu_init ||
      builtinID == Builtin::BI__builtin_cpu_is) {
    cgm.errorNYI(e->getSourceRange(),
                 std::string("unimplemented RISC-V builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  StringRef intrinsicName;
  mlir::Type returnType = convertType(e->getType());
  mlir::Location loc = getLoc(e->getSourceRange());
  llvm::SmallVector<mlir::Value> ops;

  // `iceArguments` is a bitmap indicating whether the argument at the i-th bit
  // is required to be a constant integer expression.
  unsigned iceArguments = 0;
  ASTContext::GetBuiltinTypeError error;
  getContext().GetBuiltinType(builtinID, error, &iceArguments);

  // RVV vector builtins use a special type overload mechanism (no type string).
  if (error == ASTContext::GE_Missing_type) {
    // Vector intrinsics don't have a type string.
    assert(builtinID >= clang::RISCV::FirstRVVBuiltin &&
           builtinID <= clang::RISCV::LastRVVBuiltin);
    iceArguments = 0;
    if (builtinID == RISCVVector::BI__builtin_rvv_vget_v ||
        builtinID == RISCVVector::BI__builtin_rvv_vset_v)
      iceArguments = 1 << 1;
  } else {
    assert(error == ASTContext::GE_None && "Unexpected error");
  }

  for (auto [idx, arg] : llvm::enumerate(e->arguments())) {
    // Handle aggregate argument, namely RVV tuple types in segment load/store
    if (hasAggregateEvaluationKind(arg->getType())) {
      LValue lv = emitAggExprToLValue(arg);
      ops.push_back(builder.createLoad(loc, lv.getAddress()));
      continue;
    }
    ops.push_back(emitScalarOrConstFoldImmArg(iceArguments, idx, arg));
  }

  // TODO: Handle ManualCodegen.
  bool hasCirManualCodegen = false;
  int policyAttrs = 0;

  switch (builtinID) {
  default:
    llvm_unreachable("unexpected builtin ID");

  // Zbb
  case RISCV::BI__builtin_riscv_orc_b_32:
  case RISCV::BI__builtin_riscv_orc_b_64: {
    intrinsicName = "riscv.orc.b";
    break;
  }

  // Zbc
  case RISCV::BI__builtin_riscv_clmul_32:
  case RISCV::BI__builtin_riscv_clmul_64: {
    intrinsicName = "clmul";
    break;
  }
  case RISCV::BI__builtin_riscv_clmulh_32:
  case RISCV::BI__builtin_riscv_clmulh_64: {
    intrinsicName = "riscv.clmulh";
    break;
  }
  case RISCV::BI__builtin_riscv_clmulr_32:
  case RISCV::BI__builtin_riscv_clmulr_64: {
    intrinsicName = "riscv.clmulr";
    break;
  }

  // Zbkx
  case RISCV::BI__builtin_riscv_xperm4_32:
  case RISCV::BI__builtin_riscv_xperm4_64: {
    intrinsicName = "riscv.xperm4";
    break;
  }
  case RISCV::BI__builtin_riscv_xperm8_32:
  case RISCV::BI__builtin_riscv_xperm8_64: {
    intrinsicName = "riscv.xperm8";
    break;
  }
  // Zbkb
  case RISCV::BI__builtin_riscv_brev8_32:
  case RISCV::BI__builtin_riscv_brev8_64: {
    intrinsicName = "riscv.brev8";
    break;
  }
  case RISCV::BI__builtin_riscv_zip_32: {
    intrinsicName = "riscv.zip";
    break;
  }
  case RISCV::BI__builtin_riscv_unzip_32: {
    intrinsicName = "riscv.unzip";
    break;
  }
  // Zknh
  case RISCV::BI__builtin_riscv_sha256sig0: {
    intrinsicName = "riscv.sha256sig0";
    break;
  }
  case RISCV::BI__builtin_riscv_sha256sig1: {
    intrinsicName = "riscv.sha256sig1";
    break;
  }
  case RISCV::BI__builtin_riscv_sha256sum0: {
    intrinsicName = "riscv.sha256sum0";
    break;
  }
  case RISCV::BI__builtin_riscv_sha256sum1: {
    intrinsicName = "riscv.sha256sum1";
    break;
  }
  // Zksed
  case RISCV::BI__builtin_riscv_sm4ks: {
    intrinsicName = "riscv.sm4ks";
    break;
  }
  case RISCV::BI__builtin_riscv_sm4ed: {
    intrinsicName = "riscv.sm4ed";
    break;
  }
  // Zksh
  case RISCV::BI__builtin_riscv_sm3p0: {
    intrinsicName = "riscv.sm3p0";
    break;
  }
  case RISCV::BI__builtin_riscv_sm3p1: {
    intrinsicName = "riscv.sm3p1";
    break;
  }
  // Zbb
  case RISCV::BI__builtin_riscv_clz_32:
  case RISCV::BI__builtin_riscv_clz_64: {
    auto op = cir::BitClzOp::create(builder, loc, ops[0],
                                    /*poison_zero=*/false);
    mlir::Value result = op.getResult();
    if (result.getType() != returnType)
      result = builder.createIntCast(result, returnType);
    return result;
  }
  case RISCV::BI__builtin_riscv_ctz_32:
  case RISCV::BI__builtin_riscv_ctz_64: {
    auto op = cir::BitCtzOp::create(builder, loc, ops[0],
                                    /*poison_zero=*/false);
    mlir::Value result = op.getResult();
    if (result.getType() != returnType)
      result = builder.createIntCast(result, returnType);
    return result;
  }

  // Zihintntl
  case RISCV::BI__builtin_riscv_ntl_load:
  case RISCV::BI__builtin_riscv_ntl_store: {
    cgm.errorNYI(e->getSourceRange(),
                 std::string("unimplemented RISC-V builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  // Zihintpause
  case RISCV::BI__builtin_riscv_pause: {
    intrinsicName = "riscv.pause";
    returnType = builder.getVoidTy();
    break;
  }

  // XCValu
  case RISCV::BI__builtin_riscv_cv_alu_addN:
  case RISCV::BI__builtin_riscv_cv_alu_addRN:
  case RISCV::BI__builtin_riscv_cv_alu_adduN:
  case RISCV::BI__builtin_riscv_cv_alu_adduRN:
  case RISCV::BI__builtin_riscv_cv_alu_clip:
  case RISCV::BI__builtin_riscv_cv_alu_clipu:
  case RISCV::BI__builtin_riscv_cv_alu_extbs:
  case RISCV::BI__builtin_riscv_cv_alu_extbz:
  case RISCV::BI__builtin_riscv_cv_alu_exths:
  case RISCV::BI__builtin_riscv_cv_alu_exthz:
  case RISCV::BI__builtin_riscv_cv_alu_sle:
  case RISCV::BI__builtin_riscv_cv_alu_sleu:
  case RISCV::BI__builtin_riscv_cv_alu_subN:
  case RISCV::BI__builtin_riscv_cv_alu_subRN:
  case RISCV::BI__builtin_riscv_cv_alu_subuN:
  case RISCV::BI__builtin_riscv_cv_alu_subuRN:
  // XAndesPerf
  case RISCV::BI__builtin_riscv_nds_ffb_32:
  case RISCV::BI__builtin_riscv_nds_ffb_64:
  case RISCV::BI__builtin_riscv_nds_ffzmism_32:
  case RISCV::BI__builtin_riscv_nds_ffzmism_64:
  case RISCV::BI__builtin_riscv_nds_ffmism_32:
  case RISCV::BI__builtin_riscv_nds_ffmism_64:
  case RISCV::BI__builtin_riscv_nds_flmism_32:
  case RISCV::BI__builtin_riscv_nds_flmism_64:
  // XAndesBFHCvt
  case RISCV::BI__builtin_riscv_nds_fcvt_s_bf16:
  case RISCV::BI__builtin_riscv_nds_fcvt_bf16_s: {
    cgm.errorNYI(e->getSourceRange(),
                 std::string("unimplemented RISC-V builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

#include "clang/Basic/riscv_vector_builtin_cir_cg.inc"
    // TODO: Handle Andes and SiFive vecotor builtin.
  }

  if (hasCirManualCodegen) {
    cgm.errorNYI(e->getSourceRange(),
                 std::string("unimplemented RISC-V vector builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }

  return builder.emitIntrinsicCallOp(loc, intrinsicName, returnType, ops);
}
