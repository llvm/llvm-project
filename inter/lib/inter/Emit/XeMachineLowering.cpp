#include "EmissionProgram.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <limits>

using namespace mlir;
using namespace inter::xemachine;

namespace inter::detail {
namespace {

class ProgramLowerer {
public:
  ProgramLowerer(MLIRContext *context, EmissionProgram &program)
      : context(context), program(program) {}

  LogicalResult lower(func::FuncOp function) {
    program.items.emplace_back(Label{0});
    return lowerBlock(function.getBody().front());
  }

private:
  DataType getDataType(Type type) const {
    if (type.isInteger(8))
      return DataType::ub;
    if (type.isInteger(16))
      return DataType::uw;
    if (type.isInteger(64))
      return DataType::q;
    if (type.isF32())
      return DataType::f;
    assert((type.isInteger(1) || type.isInteger(32)) &&
           "unsupported machine data type");
    return DataType::ud;
  }

  LogicalResult validateDataType(Operation *operation, Type type) const {
    if (type.isInteger(1) || type.isInteger(8) || type.isInteger(16) ||
        type.isInteger(32) || type.isInteger(64) || type.isF32())
      return success();
    return operation->emitError("unsupported machine data type ") << type;
  }

  GrfReference getGrfReference(RegType type, int32_t sub,
                               Type elementType) const {
    int32_t unitBytes = elementType.isInteger(8)    ? 1
                        : elementType.isInteger(16) ? 2
                        : elementType.isInteger(64) ? 8
                                                    : 4;
    int32_t advance = sub * unitBytes / 64;
    int32_t remainder = (sub * unitBytes) % 64 / unitBytes;
    return {type.getBaseGRF() + advance, remainder, type.getWidthDwords()};
  }

  ArfReference getArfReference(ARFType type, int32_t sub) const {
    return {type.getFile(), type.getIndex(), sub};
  }

  RegisterReference getRegisterReference(Type type, int32_t sub,
                                         Type elementType) const {
    if (auto arf = dyn_cast<ARFType>(type))
      return getArfReference(arf, sub);
    return getGrfReference(cast<RegType>(type), sub, elementType);
  }

  int32_t getSub(Operation *operation, StringRef name) const {
    if (IntegerAttr attr = operation->getAttrOfType<IntegerAttr>(name))
      return attr.getInt();
    return 0;
  }

  SourceRegion getSourceRegion(RegionAttr attr) const {
    if (attr)
      return {attr.getVstride(), attr.getWidth(), attr.getHstride()};
    return {};
  }

  SourceOperand getSourceOperand(Value value, int32_t sub, Type elementType,
                                 SourceRegion region) const {
    if (ImmOp immediate = value.getDefiningOp<ImmOp>()) {
      DataType type = getDataType(immediate.getElemType());
      return {Immediate{static_cast<uint64_t>(immediate.getValue()), type},
              type, region, false, false};
    }

    DataType type = getDataType(elementType);
    if (auto arf = dyn_cast<ARFType>(value.getType()))
      return {getArfReference(arf, sub), type, region, false, false};
    return {getGrfReference(cast<RegType>(value.getType()), sub, elementType),
            type, region, false, false};
  }

  static SwsbInfo getFinalSwsb(Operation *operation) {
    FinalSWSB final = cast<SWSBInfoOpInterface>(operation).getFinalSWSB();
    return {static_cast<DistancePipe>(final.pipe), final.distance, final.token,
            static_cast<TokenMode>(final.tokenMode)};
  }

  LogicalResult lowerBlock(Block &block) {
    for (Operation &operation : block) {
      if (operation.hasTrait<OpTrait::xemachine::NoAsmEmission>())
        continue;
      if (SendOp send = dyn_cast<SendOp>(&operation)) {
        if (failed(lowerSend(send)))
          return failure();
        continue;
      }
      if (SyncOp sync = dyn_cast<SyncOp>(&operation)) {
        lowerSync(sync);
        continue;
      }
      InstructionIssueOpInterface issue =
          dyn_cast<InstructionIssueOpInterface>(&operation);
      if (issue && issue.getInstructionKind() == MachineInstructionKind::send) {
        if (failed(lowerMessage(&operation)))
          return failure();
        continue;
      }
      if (FenceAwaitOp await = dyn_cast<FenceAwaitOp>(&operation)) {
        lowerFenceAwait(await);
        continue;
      }
      if (CmpOp compare = dyn_cast<CmpOp>(&operation)) {
        if (failed(lowerCompare(compare)))
          return failure();
        continue;
      }
      if (DpasOp dpas = dyn_cast<DpasOp>(&operation)) {
        lowerDpas(dpas);
        continue;
      }
      if (PayloadPrologueOp prologue =
              dyn_cast<PayloadPrologueOp>(&operation)) {
        if (prologue.getBody().empty())
          return prologue.emitError("emission requires a non-empty body");
        if (program.payloadEntryLabel)
          return prologue.emitError("emission supports one payload prologue");
        if (failed(lowerBlock(prologue.getBody().front())))
          return failure();
        uint32_t entryLabel = nextLabel++;
        program.items.emplace_back(Label{entryLabel});
        program.payloadEntryLabel = entryLabel;
        continue;
      }
      if (ExecIfOp ifOp = dyn_cast<ExecIfOp>(&operation)) {
        if (failed(lowerIf(ifOp.getOperation())))
          return failure();
        continue;
      }
      if (UniformIfOp ifOp = dyn_cast<UniformIfOp>(&operation)) {
        if (failed(lowerIf(ifOp.getOperation())))
          return failure();
        continue;
      }
      if (UniformLoopOp loop = dyn_cast<UniformLoopOp>(&operation)) {
        if (failed(lowerLoop(loop)))
          return failure();
        continue;
      }
      if (ContinueIfOp continueIf = dyn_cast<ContinueIfOp>(&operation)) {
        if (failed(lowerContinue(continueIf)))
          return failure();
        continue;
      }
      if (isa<func::ReturnOp>(&operation))
        continue;
      if (failed(lowerAlu(&operation)))
        return failure();
    }
    return success();
  }

  void lowerGoto(std::optional<Predicate> predicate, uint32_t jip,
                 uint32_t uip) {
    program.items.emplace_back(GotoInstruction{predicate, jip, uip});
    ++nextInstruction;
  }

  void lowerJmpi(std::optional<Predicate> predicate, uint32_t target) {
    program.items.emplace_back(JmpiInstruction{predicate, target});
    ++nextInstruction;
  }

  void lowerJoin(uint32_t label, uint32_t uip) {
    program.items.emplace_back(Label{label});
    program.items.emplace_back(JoinInstruction{uip});
    ++nextInstruction;
  }

  static bool hasSamePhysicalStorage(Type lhs, Type rhs) {
    if (isa<MemTokenType>(lhs) || isa<MemTokenType>(rhs))
      return isa<MemTokenType>(lhs) && isa<MemTokenType>(rhs);
    return lhs == rhs;
  }

  LogicalResult lowerLoop(UniformLoopOp loop) {
    if (loop.getBody().empty())
      return loop.emitError("emission requires a non-empty loop body");
    Block &body = loop.getBody().front();
    ContinueIfOp terminator = dyn_cast<ContinueIfOp>(body.getTerminator());
    if (!terminator)
      return loop.emitError("emission requires a continue_if terminator");
    if (loop.getInitialValues().size() != body.getNumArguments() ||
        loop.getNumResults() != body.getNumArguments() ||
        terminator.getCarried().size() != body.getNumArguments())
      return loop.emitError("loop-carried value count mismatch at emission");
    for (auto [init, argument, carried, result] :
         llvm::zip_equal(loop.getInitialValues(), body.getArguments(),
                         terminator.getCarried(), loop.getResults())) {
      if (!hasSamePhysicalStorage(init.getType(), argument.getType()) ||
          !hasSamePhysicalStorage(init.getType(), carried.getType()) ||
          !hasSamePhysicalStorage(init.getType(), result.getType()))
        return loop.emitError(
            "loop-carried values require identical physical storage");
    }

    uint32_t headerLabel = nextLabel++;
    program.items.emplace_back(Label{headerLabel});
    loopHeaders.push_back({loop, headerLabel});
    LogicalResult result = lowerBlock(body);
    loopHeaders.pop_back();
    return result;
  }

  LogicalResult lowerContinue(ContinueIfOp continueIf) {
    if (loopHeaders.empty() ||
        continueIf->getParentOp() != loopHeaders.back().loop.getOperation())
      return continueIf.emitError("continue_if is outside its emission loop");
    ARFType conditionType = dyn_cast<ARFType>(continueIf.getCond().getType());
    if (!conditionType || conditionType.getFile() != ARFFile::f ||
        conditionType.getIndex() < 0)
      return continueIf.emitError(
          "emission requires a physical flag loop condition");
    Predicate predicate{getArfReference(conditionType, 0), false};
    lowerJmpi(predicate, loopHeaders.back().headerLabel);
    return success();
  }

  LogicalResult lowerIf(Operation *operation) {
    Value condition = operation->getOperand(0);
    Predicate predicate{getArfReference(cast<ARFType>(condition.getType()), 0),
                        true};

    if (UniformIfOp uniformIf = dyn_cast<UniformIfOp>(operation);
        uniformIf && divergentDepth == 0) {
      uint32_t elseLabel = nextLabel++;
      uint32_t finalLabel = nextLabel++;
      lowerJmpi(predicate, elseLabel);
      if (failed(lowerBlock(uniformIf.getThenRegion().front())))
        return failure();
      lowerJmpi(std::nullopt, finalLabel);
      program.items.emplace_back(Label{elseLabel});
      if (!uniformIf.getElseRegion().empty() &&
          failed(lowerBlock(uniformIf.getElseRegion().front())))
        return failure();
      program.items.emplace_back(Label{finalLabel});
      return success();
    }

    uint32_t firstLabel = nextLabel++;
    uint32_t secondLabel = nextLabel++;
    uint32_t finalLabel = nextLabel++;
    lowerGoto(predicate, firstLabel, firstLabel);
    bool divergent = isa<ExecIfOp>(operation);
    auto lowerArm = [&](Region &region) {
      if (region.empty())
        return success();
      if (divergent)
        ++divergentDepth;
      LogicalResult result = lowerBlock(region.front());
      if (divergent)
        --divergentDepth;
      return result;
    };
    if (failed(lowerArm(operation->getRegion(0))))
      return failure();
    lowerGoto(std::nullopt, firstLabel, secondLabel);
    lowerJoin(firstLabel, secondLabel);
    if (failed(lowerArm(operation->getRegion(1))))
      return failure();
    lowerJoin(secondLabel, finalLabel);
    program.items.emplace_back(Label{finalLabel});
    return success();
  }

  void lowerSync(SyncOp sync) {
    program.items.emplace_back(SyncInstruction{
        sync.getKind(), static_cast<uint32_t>(sync.getSbidMask()),
        getFinalSwsb(sync)});
  }

  struct MessageForm {
    SendFn function;
    uint32_t desc;
    uint32_t exdesc;
    bool writesDestination;
    bool hasFixedLengths = false;
  };

  static uint32_t getGRFLength(Value value) {
    RegType type = cast<RegType>(value.getType());
    assert(type.getWidthDwords() % 16 == 0 &&
           "message payload must contain whole GRFs");
    return type.getWidthDwords() / 16;
  }

  LogicalResult lowerMessage(Operation *operation) {
    MessageForm form;
    if (isa<LoadA64Op>(operation))
      form = {SendFn::ugm, 0x00000580, 0x0, true};
    else if (isa<StoreA64Op>(operation))
      form = {SendFn::ugm, 0x00000584, 0x0, false};
    else if (isa<LoadSLMOp>(operation))
      form = {SendFn::slm, 0x00000500, 0x0, true};
    else if (isa<StoreSLMOp>(operation))
      form = {SendFn::slm, 0x00000504, 0x0, false};
    else if (isa<AtomicIAddA64Op>(operation))
      form = {SendFn::ugm, 0x0000058C, 0x0, true};
    else if (isa<FenceSLMOp>(operation))
      form = {SendFn::slm, 0x0210001F, 0x0, true,
              /*hasFixedLengths=*/true};
    else if (isa<BarrierSignalOp>(operation))
      form = {SendFn::gtwy, 0x02000004, 0x0, false,
              /*hasFixedLengths=*/true};
    else if (isa<EotOp>(operation))
      form = {SendFn::gtwy, 0x02000010, 0x0, false,
              /*hasFixedLengths=*/true};
    else if (LoadBlockA32Op block = dyn_cast<LoadBlockA32Op>(operation)) {
      uint32_t desc = block.getWords() == 32   ? 0x6229E500
                      : block.getWords() == 16 ? 0x6219D500
                                               : 0x6219C500;
      form = {SendFn::ugm, desc, 0xFF000000, true,
              /*hasFixedLengths=*/true};
    } else {
      return operation->emitError("unknown message operation");
    }

    SendInstruction instruction;
    instruction.function = form.function;
    instruction.exdesc = form.exdesc;
    instruction.eot = isa<EotOp>(operation);
    instruction.execution.size = 1;
    if (IntegerAttr attr = operation->getAttrOfType<IntegerAttr>("execSize"))
      instruction.execution.size = attr.getInt();
    instruction.execution.noMask =
        operation->hasAttr("noMask") || instruction.execution.size == 1;
    instruction.swsb = getFinalSwsb(operation);

    Type i32 = IntegerType::get(context, 32);
    Value address = operation->getOperand(0);
    if (!form.hasFixedLengths) {
      form.desc |= getGRFLength(address) << 25;
      if (form.writesDestination)
        form.desc |= getGRFLength(operation->getResult(0)) << 20;
    }
    instruction.desc = form.desc;
    instruction.address =
        getGrfReference(cast<RegType>(address.getType()), 0, i32);
    if (form.writesDestination)
      instruction.destination = getGrfReference(
          cast<RegType>(operation->getResult(0).getType()), 0, i32);

    Value data;
    if (StoreA64Op store = dyn_cast<StoreA64Op>(operation))
      data = store.getDataPayload();
    else if (StoreSLMOp store = dyn_cast<StoreSLMOp>(operation))
      data = store.getDataPayload();
    else if (AtomicIAddA64Op atomic = dyn_cast<AtomicIAddA64Op>(operation))
      data = atomic.getDataPayload();
    if (data)
      instruction.data = getGrfReference(cast<RegType>(data.getType()), 0, i32);

    program.items.push_back(std::move(instruction));
    return success();
  }

  void lowerFenceAwait(FenceAwaitOp await) {
    Type i32 = IntegerType::get(context, 32);
    AluInstruction instruction;
    instruction.opcode = AluOpcode::mov;
    instruction.execution = {8, 0, true};
    instruction.destinationType = DataType::ud;
    instruction.sources.push_back(
        getSourceOperand(await.getReadback(), 0, i32, SourceRegion{1, 1, 0}));
    instruction.swsb = getFinalSwsb(await);
    program.items.push_back(std::move(instruction));
  }

  void lowerDpas(DpasOp dpas) {
    Type i32 = IntegerType::get(context, 32);
    DpasInstruction instruction;
    instruction.execution = {16, 0, false};
    instruction.destination =
        getGrfReference(cast<RegType>(dpas.getDst().getType()), 0, i32);
    instruction.accumulator =
        getGrfReference(cast<RegType>(dpas.getAcc().getType()), 0, i32);
    instruction.sourceB =
        getGrfReference(cast<RegType>(dpas.getB().getType()), 0, i32);
    instruction.sourceA =
        getGrfReference(cast<RegType>(dpas.getA().getType()), 0, i32);
    instruction.aPrecision = dpas.getAPrecision();
    instruction.bPrecision = dpas.getBPrecision();
    instruction.systolicDepth = dpas.getSystolicDepth();
    instruction.repeatCount = dpas.getRepeatCount();
    instruction.swsb = getFinalSwsb(dpas);
    program.items.push_back(instruction);
  }

  LogicalResult lowerSend(SendOp send) {
    Type i32 = IntegerType::get(context, 32);
    SendInstruction instruction;
    instruction.execution = {static_cast<uint32_t>(send.getExecSize()), 0,
                             send.getNoMask()};
    instruction.function = send.getFn();
    RegType destinationType = cast<RegType>(send.getDst().getType());
    if (destinationType.getWidthDwords() != 0)
      instruction.destination = getGrfReference(destinationType, 0, i32);
    instruction.address =
        getGrfReference(cast<RegType>(send.getAddrPayload().getType()), 0, i32);
    if (Value data = send.getDataPayload())
      instruction.data = getGrfReference(cast<RegType>(data.getType()), 0, i32);
    if (Value exdesc = send.getExdescReg()) {
      ARFType type = cast<ARFType>(exdesc.getType());
      Operation *definition = exdesc.getDefiningOp();
      if (type.getFile() != ARFFile::a0 || type.getIndex() != 0 ||
          !definition || getSub(definition, "dstSub") != 2)
        return send.emitError("register exdesc requires a value in a0.2");
      instruction.exdesc =
          ExtendedDescriptorReference{getArfReference(type, /*sub=*/2),
                                      static_cast<uint32_t>(send.getExdesc())};
    } else
      instruction.exdesc = static_cast<uint32_t>(send.getExdesc());
    instruction.desc = send.getDesc();
    instruction.eot = send.getEot();
    instruction.swsb = getFinalSwsb(send);
    program.items.push_back(std::move(instruction));
    return success();
  }

  LogicalResult lowerCompare(CmpOp compare) {
    if (failed(validateDataType(compare, compare.getElemType())))
      return failure();
    CompareInstruction instruction;
    instruction.execution.size = compare.getExecSize();
    instruction.condition = compare.getCond();
    instruction.flag =
        getArfReference(cast<ARFType>(compare.getFlag().getType()), 0);
    instruction.dataType = getDataType(compare.getElemType());
    instruction.isSigned = compare->hasAttr("signed");

    ALUOpInterface alu = cast<ALUOpInterface>(compare.getOperation());
    for (auto [index, operand] : llvm::enumerate(compare.getOperands())) {
      std::optional<Type> explicitType =
          alu.getExplicitSourceElementType(index);
      Type sourceType = explicitType.value_or(compare.getElemType());
      if (failed(validateDataType(compare, sourceType)))
        return failure();
      if (ImmOp immediate = operand.getDefiningOp<ImmOp>())
        if (failed(validateDataType(compare, immediate.getElemType())))
          return failure();
      SourceOperand source =
          getSourceOperand(operand, alu.getSourceSubregister(index), sourceType,
                           getSourceRegion(alu.getSourceRegion(index)));
      source.isSigned = instruction.isSigned;
      instruction.sources.push_back(std::move(source));
    }
    instruction.swsb = getFinalSwsb(compare);
    program.items.push_back(std::move(instruction));
    return success();
  }

  LogicalResult lowerAlu(Operation *operation) {
    AluInstruction instruction;
    bool negateFirstSource = false;
    MachineInstructionKind instructionKind =
        cast<InstructionIssueOpInterface>(operation).getInstructionKind();
    if (instructionKind == MachineInstructionKind::mov)
      instruction.opcode = AluOpcode::mov;
    else if (instructionKind == MachineInstructionKind::add)
      instruction.opcode = AluOpcode::add;
    else if (instructionKind == MachineInstructionKind::shl)
      instruction.opcode = AluOpcode::shl;
    else if (instructionKind == MachineInstructionKind::shr)
      instruction.opcode = AluOpcode::shr;
    else if (instructionKind == MachineInstructionKind::and_)
      instruction.opcode = AluOpcode::and_;
    else if (instructionKind == MachineInstructionKind::or_)
      instruction.opcode = AluOpcode::or_;
    else if (instructionKind == MachineInstructionKind::sub) {
      instruction.opcode = AluOpcode::add;
      negateFirstSource = true;
    } else if (instructionKind == MachineInstructionKind::add3)
      instruction.opcode = AluOpcode::add3;
    else if (instructionKind == MachineInstructionKind::csel) {
      CselOp csel = cast<CselOp>(operation);
      instruction.opcode = AluOpcode::csel;
      instruction.condition = csel.getCond();
      instruction.flag =
          getArfReference(cast<ARFType>(csel.getFlag().getType()), 0);
      instruction.destinationSigned = csel.getSignedInt();
    } else if (instructionKind == MachineInstructionKind::mul)
      instruction.opcode = AluOpcode::mul;
    else
      return operation->emitError("unsupported operation in Xe emitter");

    ALUOpInterface alu = cast<ALUOpInterface>(operation);
    Type elementType = alu.getInstructionElementType();
    if (failed(validateDataType(operation, elementType)))
      return failure();
    instruction.destinationType = getDataType(elementType);
    instruction.execution.size = alu.getExecutionSize();
    if (IntegerAttr attr = operation->getAttrOfType<IntegerAttr>("maskOffset"))
      instruction.execution.maskOffset = attr.getInt();
    instruction.execution.noMask = operation->hasAttr("noMask");

    Value destination = operation->getResult(0);
    DstRegionAttr destinationRegion = alu.getDestinationRegion();
    uint32_t destinationStride =
        destinationRegion ? destinationRegion.getHstride() : 1;
    instruction.destination = Destination{
        getRegisterReference(destination.getType(),
                             alu.getDestinationSubregister(), elementType),
        destinationStride};

    for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
      std::optional<Type> explicitType =
          alu.getExplicitSourceElementType(index);
      Type sourceType = explicitType.value_or(elementType);
      if (failed(validateDataType(operation, sourceType)))
        return failure();
      if (ImmOp immediate = operand.getDefiningOp<ImmOp>())
        if (failed(validateDataType(operation, immediate.getElemType())))
          return failure();
      SourceOperand source =
          getSourceOperand(operand, alu.getSourceSubregister(index), sourceType,
                           getSourceRegion(alu.getSourceRegion(index)));
      if (operand.getDefiningOp<ImmOp>() && !explicitType)
        source.type =
            index == 1 && (instructionKind == MachineInstructionKind::shl ||
                           instructionKind == MachineInstructionKind::shr)
                ? DataType::ud
                : instruction.destinationType;
      source.negate = negateFirstSource && index == 0;
      source.isSigned = instructionKind == MachineInstructionKind::csel
                            ? cast<CselOp>(operation).getSignedInt()
                            : index == 0 && operation->hasAttr("signedSource");
      instruction.sources.push_back(std::move(source));
    }
    instruction.swsb = getFinalSwsb(operation);
    ARFType destinationArf = dyn_cast<ARFType>(destination.getType());
    if (destinationArf && destinationArf.getFile() == ARFFile::a0 &&
        !hasWrittenAddressRegister) {
      // Xe2 requires a floating-pipe distance on the first direct a0 write.
      if (instruction.swsb.pipe != DistancePipe::none &&
          instruction.swsb.pipe != DistancePipe::floating)
        return operation->emitError(
            "first a0 write has an unresolved in-order dependency");
      instruction.swsb.pipe = DistancePipe::floating;
      instruction.swsb.distance = 1;
      hasWrittenAddressRegister = true;
    }
    program.items.push_back(std::move(instruction));
    return success();
  }

  MLIRContext *context;
  EmissionProgram &program;
  struct LoopHeader {
    UniformLoopOp loop;
    uint32_t headerLabel;
  };
  SmallVector<LoopHeader> loopHeaders;
  int32_t nextInstruction = 0;
  uint32_t divergentDepth = 0;
  uint32_t nextLabel = 1;
  bool hasWrittenAddressRegister = false;
};

} // namespace

LogicalResult lowerToEmissionProgram(func::FuncOp kernel,
                                     EmissionProgram &program) {
  ProgramLowerer lowerer(kernel.getContext(), program);
  return lowerer.lower(kernel);
}

} // namespace inter::detail
