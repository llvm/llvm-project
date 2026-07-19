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

  if (builtinID == RISCV::BI__builtin_riscv_ntl_load)
    iceArguments |= (1 << 1);
  if (builtinID == RISCV::BI__builtin_riscv_ntl_store)
    iceArguments |= (1 << 2);

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
    unsigned domainVal = 5; // Default __RISCV_NTLH_ALL
    unsigned domainArgNo =
        builtinID == RISCV::BI__builtin_riscv_ntl_load ? 1 : 2;
    if (e->getNumArgs() > domainArgNo) {
      const std::optional<llvm::APSInt> result =
          e->getArg(domainArgNo)->getIntegerConstantExpr(getContext());
      assert(result && "Expected NTLH domain argument to be a constant");
      domainVal = result->getZExtValue();
    }

    mlir::Location loc = getLoc(e->getSourceRange());
    mlir::Attribute domainAttr = builder.getI32IntegerAttr(domainVal);
    Address addr(ops[0],
                 cgm.getNaturalPointeeTypeAlignment(e->getArg(0)->getType()));
    if (builtinID == RISCV::BI__builtin_riscv_ntl_load) {
      auto load = builder.createLoad(loc, addr, /*isVolatile=*/false,
                                     /*isNontemporal=*/true);
      load->setAttr("cir.riscv_nontemporal_domain", domainAttr);
      return load.getResult();
    }

    mlir::Value val = emitToMemory(ops[1], e->getArg(1)->getType());
    auto store = builder.createStore(loc, val, addr, /*isVolatile=*/false,
                                     /*isNontemporal=*/true);
    store->setAttr("cir.riscv_nontemporal_domain", domainAttr);
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
