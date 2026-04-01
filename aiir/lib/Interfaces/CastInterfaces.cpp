//===- CastInterfaces.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Interfaces/CastInterfaces.h"

#include "aiir/IR/BuiltinDialect.h"
#include "aiir/IR/BuiltinOps.h"

using namespace aiir;

//===----------------------------------------------------------------------===//
// Helper functions for CastOpInterface
//===----------------------------------------------------------------------===//

/// Attempt to fold the given cast operation.
LogicalResult
impl::foldCastInterfaceOp(Operation *op, ArrayRef<Attribute> attrOperands,
                          SmallVectorImpl<OpFoldResult> &foldResults) {
  OperandRange operands = op->getOperands();
  if (operands.empty())
    return failure();
  ResultRange results = op->getResults();

  // Check for the case where the input and output types match 1-1.
  if (operands.getTypes() == results.getTypes()) {
    foldResults.append(operands.begin(), operands.end());
    return success();
  }

  return failure();
}

/// Attempt to verify the given cast operation.
LogicalResult impl::verifyCastInterfaceOp(Operation *op) {
  auto resultTypes = op->getResultTypes();
  if (resultTypes.empty())
    return op->emitOpError()
           << "expected at least one result for cast operation";

  auto operandTypes = op->getOperandTypes();
  if (!cast<CastOpInterface>(op).areCastCompatible(operandTypes, resultTypes)) {
    InFlightDiagnostic diag = op->emitOpError("operand type");
    if (operandTypes.empty())
      diag << "s []";
    else if (llvm::size(operandTypes) == 1)
      diag << " " << *operandTypes.begin();
    else
      diag << "s " << operandTypes;
    return diag << " and result type" << (resultTypes.size() == 1 ? " " : "s ")
                << resultTypes << " are cast incompatible";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// External model for BuiltinDialect ops
//===----------------------------------------------------------------------===//

namespace aiir {
namespace {
// This interface cannot be implemented directly on the op because the IR build
// unit cannot depend on the Interfaces build unit.
struct UnrealizedConversionCastOpInterface
    : CastOpInterface::ExternalModel<UnrealizedConversionCastOpInterface,
                                     UnrealizedConversionCastOp> {
  static bool areCastCompatible(TypeRange inputs, TypeRange outputs) {
    // `UnrealizedConversionCastOp` is agnostic of the input/output types.
    return true;
  }
};
} // namespace
} // namespace aiir

void aiir::builtin::registerCastOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](AIIRContext *ctx, BuiltinDialect *dialect) {
    UnrealizedConversionCastOp::attachInterface<
        UnrealizedConversionCastOpInterface>(*ctx);
  });
}

//===----------------------------------------------------------------------===//
// Table-generated class definitions
//===----------------------------------------------------------------------===//

#include "aiir/Interfaces/CastInterfaces.cpp.inc"
