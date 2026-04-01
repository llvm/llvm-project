//===- OpenACC.h - AIIR OpenACC Dialect -------------------------*- C++ -*-===//
//
// Part of the AIIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ============================================================================
//
// This file declares the OpenACC dialect in AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_OPENACC_OPENACC_H_
#define AIIR_DIALECT_OPENACC_OPENACC_H_

#include "aiir/Dialect/OpenACC/OpenACCVariableInfo.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/IR/SymbolTable.h"

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/OpenACC/OpenACCOpsDialect.h.inc"
#include "aiir/Dialect/OpenACC/OpenACCOpsEnums.h.inc"
#include "aiir/Dialect/OpenACC/OpenACCOpsInterfaces.h.inc"
#include "aiir/Dialect/OpenACC/OpenACCTypeInterfaces.h.inc"
#include "aiir/Dialect/OpenACCMPCommon/Interfaces/AtomicInterfaces.h"
#include "aiir/IR/Value.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/LoopLikeInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include <variant>

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/OpenACC/OpenACCOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/OpenACC/OpenACCOpsAttributes.h.inc"

#include "aiir/Dialect/OpenACCMPCommon/Interfaces/OpenACCMPOpsInterfaces.h"

#define GET_OP_CLASSES
#include "aiir/Dialect/OpenACC/OpenACCOps.h.inc"

#define ACC_DATA_ENTRY_OPS                                                     \
  aiir::acc::CopyinOp, aiir::acc::CreateOp, aiir::acc::PresentOp,              \
      aiir::acc::NoCreateOp, aiir::acc::AttachOp, aiir::acc::DevicePtrOp,      \
      aiir::acc::GetDevicePtrOp, aiir::acc::PrivateOp,                         \
      aiir::acc::FirstprivateOp, aiir::acc::FirstprivateMapInitialOp,          \
      aiir::acc::UpdateDeviceOp, aiir::acc::UseDeviceOp,                       \
      aiir::acc::ReductionOp, aiir::acc::DeclareDeviceResidentOp,              \
      aiir::acc::DeclareLinkOp, aiir::acc::CacheOp
#define ACC_DATA_EXIT_OPS                                                      \
  aiir::acc::CopyoutOp, aiir::acc::DeleteOp, aiir::acc::DetachOp,              \
      aiir::acc::UpdateHostOp
#define ACC_DATA_CLAUSE_OPS ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS
#define ACC_COMPUTE_CONSTRUCT_OPS                                              \
  aiir::acc::ParallelOp, aiir::acc::KernelsOp, aiir::acc::SerialOp
#define ACC_COMPUTE_CONSTRUCT_AND_LOOP_OPS                                     \
  ACC_COMPUTE_CONSTRUCT_OPS, aiir::acc::LoopOp
#define ACC_DATA_CONSTRUCT_STRUCTURED_OPS                                      \
  aiir::acc::DataOp, aiir::acc::DeclareOp, aiir::acc::HostDataOp
#define ACC_DATA_CONSTRUCT_UNSTRUCTURED_OPS                                    \
  aiir::acc::EnterDataOp, aiir::acc::ExitDataOp, aiir::acc::UpdateOp,          \
      aiir::acc::DeclareEnterOp, aiir::acc::DeclareExitOp
#define ACC_DATA_CONSTRUCT_OPS                                                 \
  ACC_DATA_CONSTRUCT_STRUCTURED_OPS, ACC_DATA_CONSTRUCT_UNSTRUCTURED_OPS
#define ACC_COMPUTE_AND_DATA_CONSTRUCT_OPS                                     \
  ACC_COMPUTE_CONSTRUCT_OPS, ACC_DATA_CONSTRUCT_OPS
#define ACC_COMPUTE_LOOP_AND_DATA_CONSTRUCT_OPS                                \
  ACC_COMPUTE_CONSTRUCT_AND_LOOP_OPS, ACC_DATA_CONSTRUCT_OPS

namespace aiir {
namespace acc {

/// Enumeration used to encode the execution mapping on a loop construct.
/// They refer directly to the OpenACC 3.3 standard:
/// 2.9.2. gang
/// 2.9.3. worker
/// 2.9.4. vector
///
/// Value can be combined bitwise to reflect the mapping applied to the
/// construct. e.g. `acc.loop gang vector`, the `gang` and `vector` could be
/// combined and the final mapping value would be 5 (4 | 1).
enum OpenACCExecMapping { NONE = 0, VECTOR = 1, WORKER = 2, GANG = 4 };

/// Used to obtain the `var` from a data clause operation.
/// Returns empty value if not a data clause operation or is a data exit
/// operation with no `var`.
aiir::Value getVar(aiir::Operation *accDataClauseOp);

/// Used to obtain the `var` from a data clause operation if it implements
/// `PointerLikeType`.
aiir::TypedValue<aiir::acc::PointerLikeType>
getVarPtr(aiir::Operation *accDataClauseOp);

/// Used to obtains the `varType` from a data clause operation which records
/// the type of variable. When `var` is `PointerLikeType`, this returns
/// the type of the pointer target.
aiir::Type getVarType(aiir::Operation *accDataClauseOp);

/// Used to obtain the `accVar` from a data clause operation.
/// When a data entry operation, it obtains its result `accVar` value.
/// If a data exit operation, it obtains its operand `accVar` value.
/// Returns empty value if not a data clause operation.
aiir::Value getAccVar(aiir::Operation *accDataClauseOp);

/// Used to obtain the `accVar` from a data clause operation if it implements
/// `PointerLikeType`.
aiir::TypedValue<aiir::acc::PointerLikeType>
getAccPtr(aiir::Operation *accDataClauseOp);

/// Used to obtain the `varPtrPtr` from a data clause operation.
/// Returns empty value if not a data clause operation.
aiir::Value getVarPtrPtr(aiir::Operation *accDataClauseOp);

/// Used to obtain `bounds` from an acc data clause operation.
/// Returns an empty vector if there are no bounds.
aiir::SmallVector<aiir::Value> getBounds(aiir::Operation *accDataClauseOp);

/// Used to obtain `async` operands from an acc data clause operation.
/// Returns an empty vector if there are no such operands.
aiir::SmallVector<aiir::Value>
getAsyncOperands(aiir::Operation *accDataClauseOp);

/// Returns an array of acc:DeviceTypeAttr attributes attached to
/// an acc data clause operation, that correspond to the device types
/// associated with the async clauses with an async-value.
aiir::ArrayAttr getAsyncOperandsDeviceType(aiir::Operation *accDataClauseOp);

/// Returns an array of acc:DeviceTypeAttr attributes attached to
/// an acc data clause operation, that correspond to the device types
/// associated with the async clauses without an async-value.
aiir::ArrayAttr getAsyncOnly(aiir::Operation *accDataClauseOp);

/// Used to obtain the `name` from an acc operation.
std::optional<llvm::StringRef> getVarName(aiir::Operation *accOp);

/// Used to obtain the `dataClause` from a data entry operation.
/// Returns empty optional if not a data entry operation.
std::optional<aiir::acc::DataClause>
getDataClause(aiir::Operation *accDataEntryOp);

/// Used to find out whether data operation is implicit.
/// Returns false if not a data operation or if it is a data operation without
/// implicit flag.
bool getImplicitFlag(aiir::Operation *accDataEntryOp);

/// Used to get an immutable range iterating over the data operands.
aiir::ValueRange getDataOperands(aiir::Operation *accOp);

/// Used to get a mutable range iterating over the data operands.
aiir::MutableOperandRange getMutableDataOperands(aiir::Operation *accOp);

/// Used to get the recipe attribute from a data clause operation.
aiir::SymbolRefAttr getRecipe(aiir::Operation *accOp);

/// Used to check whether the provided `type` implements the `PointerLikeType`
/// interface.
inline bool isPointerLikeType(aiir::Type type) {
  return aiir::isa<aiir::acc::PointerLikeType>(type);
}

/// Used to check whether the provided `type` implements the `MappableType`
/// interface.
inline bool isMappableType(aiir::Type type) {
  return aiir::isa<aiir::acc::MappableType>(type);
}

/// Used to obtain the attribute name for declare.
static constexpr StringLiteral getDeclareAttrName() {
  return StringLiteral("acc.declare");
}

static constexpr StringLiteral getDeclareActionAttrName() {
  return StringLiteral("acc.declare_action");
}

static constexpr StringLiteral getRoutineInfoAttrName() {
  return RoutineInfoAttr::name;
}

static constexpr StringLiteral getSpecializedRoutineAttrName() {
  return SpecializedRoutineAttr::name;
}

/// Used to check whether the current operation is marked with
/// `acc routine`. The operation passed in should be a function.
inline bool isAccRoutine(aiir::Operation *op) {
  return op && op->hasAttr(aiir::acc::getRoutineInfoAttrName());
}

/// Used to check whether this is a specialized accelerator version of
/// `acc routine` function.
inline bool isSpecializedAccRoutine(aiir::Operation *op) {
  return op && op->hasAttr(aiir::acc::getSpecializedRoutineAttrName());
}

static constexpr StringLiteral getFromDefaultClauseAttrName() {
  return StringLiteral("acc.from_default");
}

static constexpr StringLiteral getVarNameAttrName() {
  return VarNameAttr::name;
}

static constexpr StringLiteral getCombinedConstructsAttrName() {
  return CombinedConstructsTypeAttr::name;
}

struct RuntimeCounters
    : public aiir::SideEffects::Resource::Base<RuntimeCounters> {
  aiir::StringRef getName() const final { return "AccRuntimeCounters"; }
  bool isAddressable() const override { return false; }
};

struct ConstructResource
    : public aiir::SideEffects::Resource::Base<ConstructResource> {
  aiir::StringRef getName() const final { return "AccConstructResource"; }
  bool isAddressable() const override { return false; }
};

struct CurrentDeviceIdResource
    : public aiir::SideEffects::Resource::Base<CurrentDeviceIdResource> {
  aiir::StringRef getName() const final { return "AccCurrentDeviceIdResource"; }
  bool isAddressable() const override { return false; }
};

} // namespace acc
} // namespace aiir

#endif // AIIR_DIALECT_OPENACC_OPENACC_H_
