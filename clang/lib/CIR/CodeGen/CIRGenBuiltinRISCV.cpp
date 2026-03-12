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
  switch (builtinID) {
  default:
    return std::nullopt;

  // Zihintpause
  case RISCV::BI__builtin_riscv_pause:
  // Zihintntl
  case RISCV::BI__builtin_riscv_ntl_load:
  case RISCV::BI__builtin_riscv_ntl_store:
  // Zbb
  case RISCV::BI__builtin_riscv_orc_b_32:
  case RISCV::BI__builtin_riscv_orc_b_64:
  case RISCV::BI__builtin_riscv_clz_32:
  case RISCV::BI__builtin_riscv_clz_64:
  case RISCV::BI__builtin_riscv_ctz_32:
  case RISCV::BI__builtin_riscv_ctz_64:
  // Zbc
  case RISCV::BI__builtin_riscv_clmul_32:
  case RISCV::BI__builtin_riscv_clmul_64:
  case RISCV::BI__builtin_riscv_clmulh_32:
  case RISCV::BI__builtin_riscv_clmulh_64:
  case RISCV::BI__builtin_riscv_clmulr_32:
  case RISCV::BI__builtin_riscv_clmulr_64:
  // Zbkb
  case RISCV::BI__builtin_riscv_brev8_32:
  case RISCV::BI__builtin_riscv_brev8_64:
  case RISCV::BI__builtin_riscv_zip_32:
  case RISCV::BI__builtin_riscv_unzip_32:
  // Zbkx
  case RISCV::BI__builtin_riscv_xperm4_32:
  case RISCV::BI__builtin_riscv_xperm4_64:
  case RISCV::BI__builtin_riscv_xperm8_32:
  case RISCV::BI__builtin_riscv_xperm8_64:
  // Zknh
  case RISCV::BI__builtin_riscv_sha256sig0:
  case RISCV::BI__builtin_riscv_sha256sig1:
  case RISCV::BI__builtin_riscv_sha256sum0:
  case RISCV::BI__builtin_riscv_sha256sum1:
  // Zksed
  case RISCV::BI__builtin_riscv_sm4ks:
  case RISCV::BI__builtin_riscv_sm4ed:
  // Zksh
  case RISCV::BI__builtin_riscv_sm3p0:
  case RISCV::BI__builtin_riscv_sm3p1:
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

    // TODO: Handle vector builtins in tablegen.
  }
}
