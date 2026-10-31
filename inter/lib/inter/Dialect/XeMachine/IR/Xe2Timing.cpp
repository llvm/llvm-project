// Derive Xe2 instruction timing from the pinned IGC scheduler model.

#include "inter/Dialect/XeMachine/IR/XeMachine.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdint>

using namespace mlir;
using namespace inter::xemachine;

namespace {

// Mirrored from IGC's XELatencyInfo and LatencyTableXe<PlatformGen::XE>.
constexpr uint16_t kFpuLatency = 10;
constexpr uint16_t kAccumulatorLatency = 6;
constexpr uint16_t kArfLatency = 16;
constexpr uint16_t kSlm16Latency = 28;
constexpr uint16_t kSlm32Latency = 45;
constexpr uint16_t kSlmFenceLatency = 23;
constexpr uint16_t kUntypedL1Latency = 45;
constexpr uint16_t kUntypedL3Latency = 200;
constexpr uint16_t kTypedL1Latency = 75;
constexpr uint16_t kTypedL3Latency = 200;
constexpr uint16_t kUntypedFenceLatency = 35;
constexpr uint16_t kTypedFenceLatency = 60;
constexpr uint16_t kBarrierLatency = 30;
constexpr uint16_t kOtherSendLatency = 50;
constexpr uint16_t kSendArbitration = 8;

static uint16_t getDpasLatency(unsigned repeatCount) {
  switch (repeatCount) {
  case 1:
    return 22;
  case 2:
    return 23;
  default:
    return 33;
  }
}

static uint16_t getWidthScale(unsigned executionSize) {
  if (executionSize <= 8)
    return 0;
  if (executionSize == 16)
    return 1;
  return 3;
}

static uint16_t getOccupancy(unsigned executionSize) {
  if (executionSize <= 8)
    return 1;
  if (executionSize == 16)
    return 2;
  return 4;
}

static unsigned getExecutionSize(Operation *operation) {
  if (isa<DpasOp>(operation))
    // Repeat count controls accumulator rows; the execution mask is SIMD16.
    return 16;
  if (IntegerAttr value = operation->getAttrOfType<IntegerAttr>("execSize"))
    return value.getInt();
  if (isa<FenceAwaitOp>(operation))
    return 8;
  MachineInstructionKind instructionKind =
      cast<InstructionIssueOpInterface>(operation).getInstructionKind();
  if (instructionKind == MachineInstructionKind::send ||
      instructionKind == MachineInstructionKind::sync)
    return 1;
  return 16;
}

static Type getElementType(Operation *operation) {
  if (auto alu = dyn_cast<ALUOpInterface>(operation))
    return alu.getInstructionElementType();
  return {};
}

static bool writesDependencyArf(Operation *operation) {
  return llvm::any_of(operation->getResultTypes(), [](Type type) {
    ARFType arf = dyn_cast<ARFType>(type);
    return arf && (arf.getFile() == ARFFile::a0 || arf.getFile() == ARFFile::f);
  });
}

static bool writesAccumulator(Operation *operation) {
  return llvm::any_of(operation->getResultTypes(), [](Type type) {
    ARFType arf = dyn_cast<ARFType>(type);
    return arf &&
           (arf.getFile() == ARFFile::acc || arf.getFile() == ARFFile::mme);
  });
}

static bool hasI64Destination(Operation *operation) {
  Type elementType = getElementType(operation);
  if (!elementType || !elementType.isInteger(64))
    return false;
  return llvm::any_of(operation->getResultTypes(),
                      [](Type type) { return isa<RegType, ARFType>(type); });
}

static bool isCachedLscLoad(uint32_t descriptor) {
  uint32_t operation = descriptor & 0x3f;
  if (operation > 3 && operation != 0x1b && operation != 0x31)
    return false;
  uint32_t cacheControl = (descriptor >> 16) & 0xf;
  return cacheControl == 6 || cacheControl == 8 || cacheControl == 9;
}

static uint16_t getSendLatency(Operation *operation, unsigned executionSize) {
  if (isa<LoadA64Op, StoreA64Op, AtomicIAddA64Op>(operation))
    return kUntypedL3Latency;
  if (isa<LoadBlockA32Op>(operation))
    return kUntypedL1Latency;
  if (isa<LoadSLMOp, StoreSLMOp>(operation))
    return executionSize > 16 ? kSlm32Latency : kSlm16Latency;
  if (isa<FenceSLMOp>(operation))
    return kSlmFenceLatency;
  if (isa<BarrierSignalOp>(operation))
    return kBarrierLatency;
  if (isa<EotOp>(operation))
    return kOtherSendLatency;

  SendOp send = cast<SendOp>(operation);
  uint32_t descriptor = static_cast<uint32_t>(send.getDesc());
  bool isFence = (descriptor & 0x3f) == 0x1f;
  bool isCachedInL1 = isCachedLscLoad(descriptor);
  switch (send.getFn()) {
  case SendFn::ugm:
    if (isFence)
      return kUntypedFenceLatency;
    return isCachedInL1 ? kUntypedL1Latency : kUntypedL3Latency;
  case SendFn::tgm:
    if (isFence)
      return kTypedFenceLatency;
    return isCachedInL1 ? kTypedL1Latency : kTypedL3Latency;
  case SendFn::slm:
    if (isFence)
      return kSlmFenceLatency;
    return executionSize > 16 ? kSlm32Latency : kSlm16Latency;
  case SendFn::gtwy:
    return (static_cast<uint32_t>(send.getDesc()) & 0xff) == 4
               ? kBarrierLatency
               : kOtherSendLatency;
  }
  llvm_unreachable("unknown XeMachine send function");
}

static uint16_t getSendSourceReadLatency(Operation *operation) {
  uint16_t payloadGrfs = 0;
  if (SendOp send = dyn_cast<SendOp>(operation)) {
    payloadGrfs = (static_cast<uint32_t>(send.getDesc()) >> 25) & 0xf;
    if (Value data = send.getDataPayload()) {
      RegType reg = cast<RegType>(data.getType());
      payloadGrfs += llvm::divideCeil(reg.getWidthDwords(), 16u);
    }
    return kSendArbitration + payloadGrfs;
  }
  for (Value operand : operation->getOperands()) {
    RegType reg = dyn_cast<RegType>(operand.getType());
    if (reg)
      payloadGrfs += llvm::divideCeil(reg.getWidthDwords(), 16u);
  }
  return kSendArbitration + payloadGrfs;
}

} // namespace

FailureOr<Xe2InstructionTiming>
inter::xemachine::getXe2InstructionTiming(Operation *operation) {
  auto issue = dyn_cast<InstructionIssueOpInterface>(operation);
  if (!issue)
    return operation->emitError("timing model requires an instruction issue "
                                "interface"),
           failure();

  Xe2InstructionTiming timing;
  timing.instructionKind = issue.getInstructionKind();
  timing.issueClass = issue.getIssueClass();
  if (timing.issueClass == InstructionIssueClass::none)
    return timing;
  if (writesDependencyArf(operation))
    timing.issueClass = InstructionIssueClass::arfWrite;
  else if (timing.issueClass == InstructionIssueClass::arithmetic ||
           timing.issueClass == InstructionIssueClass::accumulatorArithmetic)
    timing.issueClass = writesAccumulator(operation)
                            ? InstructionIssueClass::accumulatorArithmetic
                            : InstructionIssueClass::arithmetic;

  unsigned executionSize = getExecutionSize(operation);
  if (executionSize > 32 || !llvm::isPowerOf2_32(executionSize))
    return operation->emitError(
               "timing model requires a power-of-two execution "
               "size no greater than 32"),
           failure();
  timing.occupancy = getOccupancy(executionSize);
  if (hasI64Destination(operation))
    timing.occupancy = executionSize <= 4 ? 1 : 2;

  Type elementType = getElementType(operation);
  if (timing.issueClass == InstructionIssueClass::send)
    timing.pipe = Xe2IssuePipe::send;
  else if (timing.issueClass == InstructionIssueClass::systolic)
    timing.pipe = Xe2IssuePipe::systolic;
  else if (timing.issueClass != InstructionIssueClass::sync)
    timing.pipe = elementType && isa<FloatType>(elementType)
                      ? Xe2IssuePipe::floating
                      : Xe2IssuePipe::integer;

  switch (timing.issueClass) {
  case InstructionIssueClass::none:
    llvm_unreachable("no-issue operations return before timing");
  case InstructionIssueClass::moveOrLogic:
    timing.completionLatency = kFpuLatency;
    break;
  case InstructionIssueClass::arithmetic:
    timing.completionLatency = kFpuLatency + getWidthScale(executionSize);
    break;
  case InstructionIssueClass::accumulatorArithmetic:
    timing.completionLatency =
        kAccumulatorLatency + getWidthScale(executionSize);
    break;
  case InstructionIssueClass::arfWrite:
    timing.completionLatency = kArfLatency;
    break;
  case InstructionIssueClass::send:
    timing.completionLatency = getSendLatency(operation, executionSize);
    timing.sendSourceReadLatency = getSendSourceReadLatency(operation);
    break;
  case InstructionIssueClass::sync:
    timing.completionLatency = kFpuLatency;
    break;
  case InstructionIssueClass::systolic:
    timing.completionLatency =
        getDpasLatency(cast<DpasOp>(operation).getRepeatCount());
    break;
  }
  return timing;
}

uint16_t
inter::xemachine::getXe2RequiredGap(const Xe2InstructionTiming &producer,
                                    Xe2DependencyKind dependency) {
  if (producer.issueClass == InstructionIssueClass::none)
    return 0;
  switch (dependency) {
  case Xe2DependencyKind::raw:
    return std::max(producer.completionLatency, producer.occupancy);
  case Xe2DependencyKind::war:
  case Xe2DependencyKind::waw:
    return std::max(producer.sendSourceReadLatency.value_or(2),
                    producer.occupancy);
  case Xe2DependencyKind::order:
    return producer.sendSourceReadLatency.value_or(producer.occupancy);
  }
  llvm_unreachable("unknown dependency kind");
}

StringRef
inter::xemachine::stringifyInstructionIssueClass(InstructionIssueClass value) {
  switch (value) {
  case InstructionIssueClass::none:
    return "none";
  case InstructionIssueClass::moveOrLogic:
    return "move-or-logic";
  case InstructionIssueClass::arithmetic:
    return "arithmetic";
  case InstructionIssueClass::accumulatorArithmetic:
    return "accumulator-arithmetic";
  case InstructionIssueClass::arfWrite:
    return "arf-write";
  case InstructionIssueClass::send:
    return "send";
  case InstructionIssueClass::sync:
    return "sync";
  case InstructionIssueClass::systolic:
    return "systolic";
  }
  llvm_unreachable("unknown instruction issue class");
}

StringRef inter::xemachine::stringifyXe2IssuePipe(Xe2IssuePipe value) {
  switch (value) {
  case Xe2IssuePipe::none:
    return "none";
  case Xe2IssuePipe::integer:
    return "integer";
  case Xe2IssuePipe::floating:
    return "floating";
  case Xe2IssuePipe::send:
    return "send";
  case Xe2IssuePipe::systolic:
    return "systolic";
  case Xe2IssuePipe::count:
    llvm_unreachable("issue pipe count has no name");
  }
  llvm_unreachable("unknown issue pipe");
}
