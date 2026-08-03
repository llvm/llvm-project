//===- CIRDialect.h - CIR dialect -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRDIALECT_H
#define CLANG_CIR_DIALECT_IR_CIRDIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIROpsDialect.h.inc"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Interfaces/CIRLoopOpInterface.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "clang/CIR/MissingFeatures.h"

using BuilderCallbackRef =
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>;
using BuilderOpStateCallbackRef = llvm::function_ref<void(
    mlir::OpBuilder &, mlir::Location, mlir::OperationState &)>;

namespace cir {
void buildTerminatedBody(mlir::OpBuilder &builder, mlir::Location loc);

/// The process floating-point environment, including its rounding mode and
/// exception state.
struct FloatingPointEnvironmentResource
    : public mlir::SideEffects::Resource::Base<
          FloatingPointEnvironmentResource> {
  mlir::StringRef getName() const final { return "FloatingPointEnvironment"; }
  bool isAddressable() const final { return false; }
};

template <typename ConcreteType>
class FenvOpTrait : public mlir::OpTrait::TraitBase<ConcreteType, FenvOpTrait> {
public:
  mlir::Speculation::Speculatability getSpeculatability() {
    // Masked exceptions cannot trap. When strict_except is false, exception
    // side effects are non-deterministic, so speculation is still safe.
    FPEnvConstrainedOpInterface fenvOp = getFenvOp();
    if (fenvOp.getFenvExceptionMode() == cir::FPExceptionMode::Masked &&
        !fenvOp.getFenvStrictExcept())
      return mlir::Speculation::Speculatable;

    return mlir::Speculation::NotSpeculatable;
  }

  void getEffects(
      llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
    if (!getFenvOp().getFenvAttr())
      return;
    effects.emplace_back(mlir::MemoryEffects::Read::get(),
                         FloatingPointEnvironmentResource::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get(),
                         FloatingPointEnvironmentResource::get());
  }

private:
  FPEnvConstrainedOpInterface getFenvOp() {
    return mlir::cast<FPEnvConstrainedOpInterface>(this->getOperation());
  }
};

/// Look up the RecordLayoutAttr for a named record in the module's
/// cir.record_layouts dictionary.  Asserts if the entry is missing.
RecordLayoutAttr getRecordLayout(mlir::ModuleOp module, mlir::StringAttr name);
} // namespace cir

// TableGen'erated files for MLIR dialects require that a macro be defined when
// they are included.  GET_OP_CLASSES tells the file to define the classes for
// the operations of that dialect.
#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROps.h.inc"

#endif // CLANG_CIR_DIALECT_IR_CIRDIALECT_H
