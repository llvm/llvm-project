#ifndef INTER_LIB_INTER_EMIT_EMISSIONPROGRAM_H
#define INTER_LIB_INTER_EMIT_EMISSIONPROGRAM_H

#include "inter/Dialect/XeMachine/IR/XeMachine.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace inter::detail {

enum class DataType { ub, uw, ud, q, f };

enum class AluOpcode { mov, add, shl, and_, or_, add3, mul };

struct ExecutionInfo {
  uint32_t size = 1;
  uint32_t maskOffset = 0;
  bool noMask = false;
};

struct GrfReference {
  int32_t number = 0;
  int32_t sub = 0;
  uint32_t widthDwords = 0;
};

struct ArfReference {
  xemachine::ARFFile file = xemachine::ARFFile::a0;
  int32_t number = 0;
  int32_t sub = 0;
};

using RegisterReference = std::variant<GrfReference, ArfReference>;

struct Immediate {
  uint64_t value = 0;
  DataType type = DataType::ud;
};

struct SourceRegion {
  uint32_t vstride = 1;
  uint32_t width = 1;
  uint32_t hstride = 0;
};

struct SourceOperand {
  std::variant<GrfReference, ArfReference, Immediate> value;
  DataType type = DataType::ud;
  SourceRegion region;
  bool negate = false;
};

struct Destination {
  RegisterReference value;
  uint32_t hstride = 1;
};

enum class DistancePipe { none, all, floating, inOrder };

struct SwsbInfo {
  DistancePipe pipe = DistancePipe::none;
  int32_t distance = -1;
  int32_t token = -1;
};

struct AluInstruction {
  AluOpcode opcode = AluOpcode::mov;
  ExecutionInfo execution;
  std::optional<Destination> destination;
  DataType destinationType = DataType::ud;
  llvm::SmallVector<SourceOperand, 3> sources;
  SwsbInfo swsb;
};

struct CompareInstruction {
  ExecutionInfo execution;
  xemachine::CondModifier condition = xemachine::CondModifier::eq;
  ArfReference flag;
  DataType dataType = DataType::ud;
  llvm::SmallVector<SourceOperand, 2> sources;
  SwsbInfo swsb;
};

struct SendInstruction {
  ExecutionInfo execution;
  xemachine::SendFn function = xemachine::SendFn::ugm;
  std::optional<GrfReference> destination;
  GrfReference address;
  std::optional<GrfReference> data;
  uint32_t exdesc = 0;
  uint32_t desc = 0;
  bool eot = false;
  SwsbInfo swsb;
  std::optional<uint32_t> rawSwsb;
};

struct SyncInstruction {
  xemachine::SyncKind kind = xemachine::SyncKind::nop;
};

struct Predicate {
  ArfReference flag;
  bool inverse = false;
};

struct GotoInstruction {
  std::optional<Predicate> predicate;
  uint32_t jip = 0;
  uint32_t uip = 0;
};

struct JmpiInstruction {
  std::optional<Predicate> predicate;
  uint32_t target = 0;
};

struct JoinInstruction {
  uint32_t uip = 0;
};

struct Label {
  uint32_t id = 0;
};

using EmissionItem =
    std::variant<Label, AluInstruction, CompareInstruction, SendInstruction,
                 SyncInstruction, GotoInstruction, JmpiInstruction,
                 JoinInstruction>;

struct EmissionProgram {
  llvm::SmallVector<EmissionItem> items;
};

mlir::LogicalResult lowerToEmissionProgram(mlir::ModuleOp moduleOp,
                                           EmissionProgram &program);

} // namespace inter::detail

#endif // INTER_LIB_INTER_EMIT_EMISSIONPROGRAM_H
