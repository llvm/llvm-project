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
  llvm::SmallVector<mlir::Value> ops;

  // `iceArguments` is a bitmap indicating whether the argument at the i-th bit
  // is required to be a constant integer expression.
  unsigned iceArguments = 0;
  ASTContext::GetBuiltinTypeError error;
  getContext().GetBuiltinType(builtinID, error, &iceArguments);
  assert(error == ASTContext::GE_None && "Should not codegen an error");
  for (auto [idx, arg] : llvm::enumerate(e->arguments()))
    ops.push_back(emitScalarOrConstFoldImmArg(iceArguments, idx, arg));

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
    mlir::Location loc = getLoc(e->getSourceRange());
    auto op = cir::BitClzOp::create(builder, loc, ops[0],
                                    /*poison_zero=*/false);
    mlir::Value result = op.getResult();
    if (result.getType() != returnType)
      result = builder.createIntCast(result, returnType);
    return result;
  }
  case RISCV::BI__builtin_riscv_ctz_32:
  case RISCV::BI__builtin_riscv_ctz_64: {
    mlir::Location loc = getLoc(e->getSourceRange());
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
  case RISCV::BI__builtin_riscv_cv_alu_addN: {
    intrinsicName = "riscv.cv.alu.addN";
    break;
  }
  case RISCV::BI__builtin_riscv_cv_alu_addRN: {
    intrinsicName = "riscv.cv.alu.addRN";
    break;
  }
  case RISCV::BI__builtin_riscv_cv_alu_adduN: {
    intrinsicName = "riscv.cv.alu.adduN";
    break;
  }
  case RISCV::BI__builtin_riscv_cv_alu_adduRN: {
    intrinsicName = "riscv.cv.alu.adduRN";
    break;
  }
  case RISCV::BI__builtin_riscv_cv_alu_clip: {
    intrinsicName = "riscv.cv.alu.clip";
    break;
  }
  case RISCV::BI__builtin_riscv_cv_alu_clipu: {
    intrinsicName = "riscv.cv.alu.clipu";
    break;
  }
  case RISCV::BI__builtin_riscv_cv_alu_extbs: {
    mlir::Value result = builder.createIntCast(ops[0], builder.getSInt8Ty());
    return builder.createIntCast(result, returnType);
  }
  case RISCV::BI__builtin_riscv_cv_alu_extbz: {
    mlir::Value result = builder.createIntCast(ops[0], builder.getUInt8Ty());
    return builder.createIntCast(result, returnType);
  }
  case RISCV::BI__builtin_riscv_cv_alu_exths: {
    mlir::Value result = builder.createIntCast(ops[0], builder.getSInt16Ty());
    return builder.createIntCast(result, returnType);
  }
  case RISCV::BI__builtin_riscv_cv_alu_exthz: {
    mlir::Value result = builder.createIntCast(ops[0], builder.getUInt16Ty());
    return builder.createIntCast(result, returnType);
  }
  case RISCV::BI__builtin_riscv_cv_alu_sle: {
    mlir::Location loc = getLoc(e->getSourceRange());
    mlir::Value result =
        builder.createCompare(loc, cir::CmpOpKind::le, ops[0], ops[1]);
    return builder.createBoolToInt(result, returnType);
  }
  case RISCV::BI__builtin_riscv_cv_alu_sleu: {
    mlir::Location loc = getLoc(e->getSourceRange());
    mlir::Value result =
        builder.createCompare(loc, cir::CmpOpKind::le, ops[0], ops[1]);
    return builder.createBoolToInt(result, returnType);
  }
  case RISCV::BI__builtin_riscv_cv_alu_subN: {
    intrinsicName = "riscv.cv.alu.subN";
    break;
  }
  case RISCV::BI__builtin_riscv_cv_alu_subRN: {
    intrinsicName = "riscv.cv.alu.subRN";
    break;
  }
  case RISCV::BI__builtin_riscv_cv_alu_subuN: {
    intrinsicName = "riscv.cv.alu.subuN";
    break;
  }
  case RISCV::BI__builtin_riscv_cv_alu_subuRN: {
    intrinsicName = "riscv.cv.alu.subuRN";
    break;
  }
  // XCVbitmanip
  case RISCV::BI__builtin_riscv_cv_bitmanip_extract:
  case RISCV::BI__builtin_riscv_cv_bitmanip_extractu:
  case RISCV::BI__builtin_riscv_cv_bitmanip_bclr:
  case RISCV::BI__builtin_riscv_cv_bitmanip_bset:
  case RISCV::BI__builtin_riscv_cv_bitmanip_insert:
  case RISCV::BI__builtin_riscv_cv_bitmanip_clb:
  case RISCV::BI__builtin_riscv_cv_bitmanip_bitrev:
  // XCVelw
  case RISCV::BI__builtin_riscv_cv_elw_elw:
  // XCVmac
  case RISCV::BI__builtin_riscv_cv_mac_mac:
  case RISCV::BI__builtin_riscv_cv_mac_msu:
  case RISCV::BI__builtin_riscv_cv_mac_muluN:
  case RISCV::BI__builtin_riscv_cv_mac_mulhhuN:
  case RISCV::BI__builtin_riscv_cv_mac_mulsN:
  case RISCV::BI__builtin_riscv_cv_mac_mulhhsN:
  case RISCV::BI__builtin_riscv_cv_mac_muluRN:
  case RISCV::BI__builtin_riscv_cv_mac_mulhhuRN:
  case RISCV::BI__builtin_riscv_cv_mac_mulsRN:
  case RISCV::BI__builtin_riscv_cv_mac_mulhhsRN:
  case RISCV::BI__builtin_riscv_cv_mac_macuN:
  case RISCV::BI__builtin_riscv_cv_mac_machhuN:
  case RISCV::BI__builtin_riscv_cv_mac_macsN:
  case RISCV::BI__builtin_riscv_cv_mac_machhsN:
  case RISCV::BI__builtin_riscv_cv_mac_macuRN:
  case RISCV::BI__builtin_riscv_cv_mac_machhuRN:
  case RISCV::BI__builtin_riscv_cv_mac_macsRN:
  case RISCV::BI__builtin_riscv_cv_mac_machhsRN:
  // XCVsimd builtins (lowered in classic CodeGen; not yet implemented
  // in CIR).
  case RISCV::BI__builtin_riscv_cv_simd_add_h:
  case RISCV::BI__builtin_riscv_cv_simd_add_b:
  case RISCV::BI__builtin_riscv_cv_simd_sub_h:
  case RISCV::BI__builtin_riscv_cv_simd_sub_b:
  case RISCV::BI__builtin_riscv_cv_simd_min_h:
  case RISCV::BI__builtin_riscv_cv_simd_min_b:
  case RISCV::BI__builtin_riscv_cv_simd_minu_h:
  case RISCV::BI__builtin_riscv_cv_simd_minu_b:
  case RISCV::BI__builtin_riscv_cv_simd_max_h:
  case RISCV::BI__builtin_riscv_cv_simd_max_b:
  case RISCV::BI__builtin_riscv_cv_simd_maxu_h:
  case RISCV::BI__builtin_riscv_cv_simd_maxu_b:
  case RISCV::BI__builtin_riscv_cv_simd_and_h:
  case RISCV::BI__builtin_riscv_cv_simd_and_b:
  case RISCV::BI__builtin_riscv_cv_simd_or_h:
  case RISCV::BI__builtin_riscv_cv_simd_or_b:
  case RISCV::BI__builtin_riscv_cv_simd_xor_h:
  case RISCV::BI__builtin_riscv_cv_simd_xor_b:
  case RISCV::BI__builtin_riscv_cv_simd_abs_h:
  case RISCV::BI__builtin_riscv_cv_simd_abs_b:
  case RISCV::BI__builtin_riscv_cv_simd_dotup_h:
  case RISCV::BI__builtin_riscv_cv_simd_dotup_b:
  case RISCV::BI__builtin_riscv_cv_simd_dotup_sc_h:
  case RISCV::BI__builtin_riscv_cv_simd_dotup_sc_b:
  case RISCV::BI__builtin_riscv_cv_simd_dotusp_h:
  case RISCV::BI__builtin_riscv_cv_simd_dotusp_b:
  case RISCV::BI__builtin_riscv_cv_simd_dotusp_sc_h:
  case RISCV::BI__builtin_riscv_cv_simd_dotusp_sc_b:
  case RISCV::BI__builtin_riscv_cv_simd_dotsp_h:
  case RISCV::BI__builtin_riscv_cv_simd_dotsp_b:
  case RISCV::BI__builtin_riscv_cv_simd_dotsp_sc_h:
  case RISCV::BI__builtin_riscv_cv_simd_dotsp_sc_b:
  case RISCV::BI__builtin_riscv_cv_simd_sdotup_h:
  case RISCV::BI__builtin_riscv_cv_simd_sdotup_b:
  case RISCV::BI__builtin_riscv_cv_simd_sdotup_sc_h:
  case RISCV::BI__builtin_riscv_cv_simd_sdotup_sc_b:
  case RISCV::BI__builtin_riscv_cv_simd_sdotusp_h:
  case RISCV::BI__builtin_riscv_cv_simd_sdotusp_b:
  case RISCV::BI__builtin_riscv_cv_simd_sdotusp_sc_h:
  case RISCV::BI__builtin_riscv_cv_simd_sdotusp_sc_b:
  case RISCV::BI__builtin_riscv_cv_simd_sdotsp_h:
  case RISCV::BI__builtin_riscv_cv_simd_sdotsp_b:
  case RISCV::BI__builtin_riscv_cv_simd_sdotsp_sc_h:
  case RISCV::BI__builtin_riscv_cv_simd_sdotsp_sc_b:
  case RISCV::BI__builtin_riscv_cv_simd_extract_h:
  case RISCV::BI__builtin_riscv_cv_simd_extract_b:
  case RISCV::BI__builtin_riscv_cv_simd_extractu_h:
  case RISCV::BI__builtin_riscv_cv_simd_extractu_b:
  case RISCV::BI__builtin_riscv_cv_simd_insert_h:
  case RISCV::BI__builtin_riscv_cv_simd_insert_b:
  case RISCV::BI__builtin_riscv_cv_simd_shuffle_h:
  case RISCV::BI__builtin_riscv_cv_simd_shuffle_b:
  case RISCV::BI__builtin_riscv_cv_simd_shuffle_sci_h:
  case RISCV::BI__builtin_riscv_cv_simd_shuffle_sci_b:
  case RISCV::BI__builtin_riscv_cv_simd_shuffle2_h:
  case RISCV::BI__builtin_riscv_cv_simd_shuffle2_b:
  case RISCV::BI__builtin_riscv_cv_simd_packhi_h:
  case RISCV::BI__builtin_riscv_cv_simd_packlo_h:
  case RISCV::BI__builtin_riscv_cv_simd_packhi_b:
  case RISCV::BI__builtin_riscv_cv_simd_packlo_b:
  case RISCV::BI__builtin_riscv_cv_simd_cplxmul_r:
  case RISCV::BI__builtin_riscv_cv_simd_cplxmul_i:
  case RISCV::BI__builtin_riscv_cv_simd_cplxconj:
  case RISCV::BI__builtin_riscv_cv_simd_subrotmj:
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

    // TODO: Handle vector builtins in tablegen.
  }

  mlir::Location loc = getLoc(e->getSourceRange());
  return builder.emitIntrinsicCallOp(loc, intrinsicName, returnType, ops);
}
