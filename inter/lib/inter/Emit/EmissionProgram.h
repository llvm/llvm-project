#ifndef INTER_LIB_INTER_EMIT_EMISSIONPROGRAM_H
#define INTER_LIB_INTER_EMIT_EMISSIONPROGRAM_H

#include "inter/Dialect/XeMachine/IR/XeMachine.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace inter::detail {

enum class DataType { ub, uw, ud, q, f };

enum class AluOpcode { mov, add, shl, shr, and_, or_, add3, csel, mul };

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

struct ExtendedDescriptorReference {
  ArfReference base;
  uint32_t immediate = 0;
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
  bool isSigned = false;
};

struct Destination {
  RegisterReference value;
  uint32_t hstride = 1;
};

enum class DistancePipe { none, all, floating, inOrder };
enum class TokenMode { none, set, source, destination };

struct SwsbInfo {
  DistancePipe pipe = DistancePipe::none;
  int32_t distance = -1;
  int32_t token = -1;
  TokenMode tokenMode = TokenMode::none;
};

struct AluInstruction {
  AluOpcode opcode = AluOpcode::mov;
  ExecutionInfo execution;
  std::optional<Destination> destination;
  DataType destinationType = DataType::ud;
  bool destinationSigned = false;
  llvm::SmallVector<SourceOperand, 3> sources;
  std::optional<xemachine::CondModifier> condition;
  std::optional<ArfReference> flag;
  SwsbInfo swsb;
};

struct CompareInstruction {
  ExecutionInfo execution;
  xemachine::CondModifier condition = xemachine::CondModifier::eq;
  ArfReference flag;
  DataType dataType = DataType::ud;
  bool isSigned = false;
  llvm::SmallVector<SourceOperand, 2> sources;
  SwsbInfo swsb;
};

struct DpasInstruction {
  ExecutionInfo execution;
  GrfReference destination;
  GrfReference accumulator;
  GrfReference sourceB;
  GrfReference sourceA;
  xemachine::DpasPrecision aPrecision = xemachine::DpasPrecision::F16;
  xemachine::DpasPrecision bPrecision = xemachine::DpasPrecision::F16;
  uint32_t systolicDepth = 8;
  uint32_t repeatCount = 8;
  SwsbInfo swsb;
};

struct SendInstruction {
  ExecutionInfo execution;
  xemachine::SendFn function = xemachine::SendFn::ugm;
  std::optional<GrfReference> destination;
  GrfReference address;
  std::optional<GrfReference> data;
  std::variant<uint32_t, ExtendedDescriptorReference> exdesc = uint32_t{0};
  uint32_t desc = 0;
  bool eot = false;
  SwsbInfo swsb;
  std::optional<uint32_t> rawSwsb;
};

struct SyncInstruction {
  xemachine::SyncKind kind = xemachine::SyncKind::nop;
  uint32_t sbidMask = 0;
  SwsbInfo swsb;
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
    std::variant<Label, AluInstruction, CompareInstruction, DpasInstruction,
                 SendInstruction, SyncInstruction, GotoInstruction,
                 JmpiInstruction, JoinInstruction>;

struct EmissionProgram {
  llvm::SmallVector<EmissionItem> items;
  std::optional<uint32_t> payloadEntryLabel;
};

mlir::LogicalResult lowerToEmissionProgram(mlir::func::FuncOp kernel,
                                           EmissionProgram &program);

} // namespace inter::detail

#endif // INTER_LIB_INTER_EMIT_EMISSIONPROGRAM_H
