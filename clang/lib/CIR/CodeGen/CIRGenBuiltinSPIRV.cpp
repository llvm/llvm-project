//===---- CIRGenBuiltinSPIRV.cpp - Emit CIR for SPIR-V builtins -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit SPIR-V Builtin calls.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"

#include "mlir/IR/Value.h"
#include "clang/Basic/TargetBuiltins.h"

using namespace clang;
using namespace clang::CIRGen;

/// Emit a work-item query builtin that takes a dimension index and returns a
/// size_t value, e.g. __builtin_spirv_global_invocation_id(dim).
static mlir::Value emitWorkItemQuery(CIRGenFunction &cgf, const CallExpr *expr,
                                     llvm::StringRef intrinsicName) {
  mlir::Value dim = cgf.emitScalarExpr(expr->getArg(0));
  return cgf.getBuilder().emitIntrinsicCallOp(
      cgf.getLoc(expr->getExprLoc()), intrinsicName,
      cgf.convertType(expr->getType()), mlir::ValueRange{dim});
}

std::optional<mlir::Value>
CIRGenFunction::emitSPIRVBuiltinExpr(unsigned builtinId, const CallExpr *expr) {
  switch (builtinId) {
  case SPIRV::BI__builtin_spirv_num_workgroups:
    return emitWorkItemQuery(*this, expr, "spv.num.workgroups");
  case SPIRV::BI__builtin_spirv_workgroup_size:
    return emitWorkItemQuery(*this, expr, "spv.workgroup.size");
  case SPIRV::BI__builtin_spirv_workgroup_id:
    return emitWorkItemQuery(*this, expr, "spv.group.id");
  case SPIRV::BI__builtin_spirv_local_invocation_id:
    return emitWorkItemQuery(*this, expr, "spv.thread.id.in.group");
  case SPIRV::BI__builtin_spirv_global_invocation_id:
    return emitWorkItemQuery(*this, expr, "spv.thread.id");
  case SPIRV::BI__builtin_spirv_global_size:
    return emitWorkItemQuery(*this, expr, "spv.global.size");
  case SPIRV::BI__builtin_spirv_global_offset:
    return emitWorkItemQuery(*this, expr, "spv.global.offset");

  case SPIRV::BI__builtin_spirv_distance:
  case SPIRV::BI__builtin_spirv_length:
  case SPIRV::BI__builtin_spirv_reflect:
  case SPIRV::BI__builtin_spirv_refract:
  case SPIRV::BI__builtin_spirv_smoothstep:
  case SPIRV::BI__builtin_spirv_faceforward:
  case SPIRV::BI__builtin_spirv_generic_cast_to_ptr_explicit:
  case SPIRV::BI__builtin_spirv_subgroup_shuffle:
  case SPIRV::BI__builtin_spirv_ddx:
  case SPIRV::BI__builtin_spirv_ddy:
  case SPIRV::BI__builtin_spirv_fwidth:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented SPIR-V builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};

  default:
    // Builtins with a ClangBuiltin mapping on the intrinsic (e.g. the subgroup
    // queries) are handled by the generic path in emitBuiltinExpr.
    return std::nullopt;
  }
}
