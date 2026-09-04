//===- Xe2Timing.h - Xe2 instruction timing ---------------------*- C++ -*-===//

#ifndef INTER_DIALECT_XEMACHINE_IR_XE2TIMING_H
#define INTER_DIALECT_XEMACHINE_IR_XE2TIMING_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir {
class Operation;
}

namespace inter::xemachine {

enum class InstructionIssueClass : uint8_t {
  none,
  moveOrLogic,
  arithmetic,
  accumulatorArithmetic,
  arfWrite,
  send,
  sync,
  systolic,
};

enum class MachineInstructionKind : uint8_t {
  none,
  mov,
  add,
  sub,
  add3,
  csel,
  shl,
  shr,
  and_,
  or_,
  mul,
  cmp,
  send,
  sync,
  dpas,
  branch,
};

enum class AsyncScoreboardKind : uint8_t { send, dpas };

enum class Xe2IssuePipe : uint8_t {
  none,
  integer,
  floating,
  send,
  systolic,
  count,
};

enum class Xe2DependencyKind : uint8_t { raw, war, waw, order };

struct Xe2InstructionTiming {
  MachineInstructionKind instructionKind = MachineInstructionKind::none;
  InstructionIssueClass issueClass = InstructionIssueClass::none;
  Xe2IssuePipe pipe = Xe2IssuePipe::none;
  uint16_t completionLatency = 0;
  uint16_t occupancy = 0;
  std::optional<uint16_t> sendSourceReadLatency;
};

mlir::FailureOr<Xe2InstructionTiming>
getXe2InstructionTiming(mlir::Operation *operation);

uint16_t getXe2RequiredGap(const Xe2InstructionTiming &producer,
                           Xe2DependencyKind dependency);

llvm::StringRef stringifyInstructionIssueClass(InstructionIssueClass value);
llvm::StringRef stringifyXe2IssuePipe(Xe2IssuePipe value);

} // namespace inter::xemachine

#endif // INTER_DIALECT_XEMACHINE_IR_XE2TIMING_H
