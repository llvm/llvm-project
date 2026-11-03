#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINE_H
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINE_H

#include "inter/Dialect/XeMachine/IR/Xe2Timing.h"
#include "inter/Dialect/XeMachine/IR/XeMachineABI.h"
#include "inter/Dialect/XeMachine/IR/XeMachineTarget.h"
#include "inter/Dialect/XeMachine/IR/XeMachineTraits.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace inter::xemachine {

inline constexpr llvm::StringLiteral kTargetAttrName = "xemachine.target";
inline constexpr llvm::StringLiteral kKernelArgsAttrName =
    "xemachine.kernel_args";
inline constexpr llvm::StringLiteral kGrfCountAttrName = "xemachine.grf_count";
inline constexpr llvm::StringLiteral kGrfUsedAttrName = "xemachine.grf_used";
inline constexpr llvm::StringLiteral kReservedGrfCountAttrName =
    "xemachine.reserved_grf_count";
inline constexpr llvm::StringLiteral kSimdSizeAttrName = "xemachine.simd_size";
inline constexpr llvm::StringLiteral kRequiredWorkGroupSizeAttrName =
    "xemachine.required_work_group_size";
inline constexpr llvm::StringLiteral kUsesThreadIdsAttrName =
    "xemachine.uses_thread_ids";
inline constexpr llvm::StringLiteral kInlineDataPayloadSizeAttrName =
    "xemachine.inline_data_payload_size";
inline constexpr llvm::StringLiteral kPerThreadPayloadSizeAttrName =
    "xemachine.per_thread_payload_size";
inline constexpr llvm::StringLiteral kSlmSizeAttrName = "xemachine.slm_size";
inline constexpr llvm::StringLiteral kScratchSizeAttrName =
    "xemachine.scratch_size";
inline constexpr llvm::StringLiteral kBarrierCountAttrName =
    "xemachine.barrier_count";
inline constexpr llvm::StringLiteral kHasGlobalAtomicsAttrName =
    "xemachine.has_global_atomics";
inline constexpr llvm::StringLiteral kHasNoStatelessWriteAttrName =
    "xemachine.has_no_stateless_write";
inline constexpr llvm::StringLiteral kHasDpasAttrName = "xemachine.has_dpas";
inline constexpr llvm::StringLiteral kScratchAccessAttrName =
    "xemachine.scratch_access";
inline constexpr llvm::StringLiteral kAllowFixedOverlapAttrName =
    "xemachine.allow_fixed_overlap";

struct KernelResourceUsage {
  uint64_t grfUsed;
  int64_t barrierCount;
  bool hasGlobalAtomics;
  bool hasStatelessWrite;
  bool hasDpas;
};

mlir::FailureOr<KernelResourceUsage>
analyzeKernelResources(mlir::func::FuncOp function, int64_t grfCount);

mlir::LogicalResult verifyKernelArgLayout(mlir::ArrayAttr arguments,
                                          mlir::Operation *owner);

/// Relative register-storage constraint, in dwords.
struct RegisterStorageAlias {
  mlir::Value storage;
  mlir::Value alias;
  int64_t offset = 0;

  /// The alias is consumed while storage is overwritten.
  bool destructive = false;
};

} // namespace inter::xemachine

#include "inter/Dialect/XeMachine/IR/XeMachineDialect.h.inc"
#include "inter/Dialect/XeMachine/IR/XeMachineEnums.h.inc"

namespace inter::xemachine {

struct FinalSWSB {
  SWSBDistancePipe pipe = SWSBDistancePipe::none;
  int32_t distance = -1;
  int32_t token = -1;
  SWSBTokenMode tokenMode = SWSBTokenMode::none;

  bool empty() const { return distance < 0 && token < 0; }
};

} // namespace inter::xemachine

#define GET_ATTRDEF_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineTypes.h.inc"

#define GET_OP_INTERFACE_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineInterfaces.h.inc"

#define GET_OP_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineOps.h.inc"

#define GET_OP_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineTransformOps.h.inc"

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINE_H
