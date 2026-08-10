#include "EmissionProgram.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>

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
    function.walk([&](Operation *operation) {
      if (operation->hasTrait<OpTrait::xemachine::NoMachineInst>())
        return;
      for (Value result : operation->getResults())
        definingOperations[result] = operation;
    });
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
    return DataType::ud;
  }

  GrfReference getGrfReference(RegType type, int32_t sub,
                               Type elementType) const {
    int32_t unitBytes = elementType.isInteger(64)   ? 8
                        : elementType.isInteger(16) ? 2
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

  Type getSourceType(Operation *operation, StringRef name,
                     Type fallback) const {
    if (TypeAttr attr = operation->getAttrOfType<TypeAttr>(name))
      return attr.getValue();
    return fallback;
  }

  SourceRegion getSourceRegion(Operation *operation, StringRef name) const {
    if (RegionAttr attr = operation->getAttrOfType<RegionAttr>(name))
      return {attr.getVstride(), attr.getWidth(), attr.getHstride()};
    return {};
  }

  SourceOperand getSourceOperand(Value value, int32_t sub, Type elementType,
                                 SourceRegion region) const {
    if (ImmOp immediate = value.getDefiningOp<ImmOp>()) {
      DataType type = getDataType(immediate.getElemType());
      return {Immediate{static_cast<uint64_t>(immediate.getValue()), type},
              type, region, false};
    }

    DataType type = getDataType(elementType);
    if (auto arf = dyn_cast<ARFType>(value.getType()))
      return {getArfReference(arf, sub), type, region, false};
    return {getGrfReference(cast<RegType>(value.getType()), sub, elementType),
            type, region, false};
  }

  int32_t getYoungestProducerDistance(Operation *operation,
                                      ValueRange operands) const {
    int32_t youngest = -1;
    for (Value operand : operands) {
      if (isa<MemTokenType>(operand.getType()))
        continue;
      Operation *definingOperation = definingOperations.lookup(operand);
      if (!definingOperation || isa<SendOp>(definingOperation))
        continue;
      auto iterator = instructionIndices.find(definingOperation);
      if (iterator == instructionIndices.end())
        continue;
      int32_t distance =
          instructionIndices.lookup(operation) - iterator->second;
      if (distance >= 1 && distance <= 15)
        youngest = youngest < 0 ? distance : std::min(youngest, distance);
    }
    return youngest;
  }

  SwsbInfo getInOrderSwsb(Operation *operation, ValueRange operands) const {
    int32_t distance = getYoungestProducerDistance(operation, operands);
    if (distance < 0)
      return {};
    return {DistancePipe::inOrder, distance, -1};
  }

  SwsbInfo getSendSwsb(Operation *operation, ValueRange operands, bool eot) {
    int32_t distance = getYoungestProducerDistance(operation, operands);
    DistancePipe pipe = distance < 0 ? DistancePipe::none
                        : eot        ? DistancePipe::inOrder
                                     : DistancePipe::all;
    return {pipe, distance, nextToken++};
  }

  LogicalResult lowerBlock(Block &block) {
    for (Operation &operation : block) {
      if (operation.hasTrait<OpTrait::xemachine::NoAsmEmission>())
        continue;
      instructionIndices[&operation] = nextInstruction++;

      if (SendOp send = dyn_cast<SendOp>(&operation)) {
        lowerSend(send);
        continue;
      }
      if (SyncOp sync = dyn_cast<SyncOp>(&operation)) {
        lowerSync(sync);
        continue;
      }
      if (isa<LoadA64Op, StoreA64Op, LoadSLMOp, StoreSLMOp, AtomicIAddA64Op,
              LoadBlockA32Op, FenceSLMOp, BarrierSignalOp, EotOp>(operation)) {
        if (failed(lowerMessage(&operation)))
          return failure();
        continue;
      }
      if (FenceAwaitOp await = dyn_cast<FenceAwaitOp>(&operation)) {
        lowerFenceAwait(await);
        continue;
      }
      if (CmpOp compare = dyn_cast<CmpOp>(&operation)) {
        lowerCompare(compare);
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

  void lowerJoin(uint32_t label, uint32_t uip) {
    program.items.emplace_back(Label{label});
    program.items.emplace_back(JoinInstruction{uip});
    ++nextInstruction;
  }

  LogicalResult lowerIf(Operation *operation) {
    Value condition = operation->getOperand(0);
    Predicate predicate{getArfReference(cast<ARFType>(condition.getType()), 0),
                        true};

    uint32_t firstLabel = nextLabel++;
    uint32_t secondLabel = nextLabel++;
    uint32_t finalLabel = nextLabel++;
    lowerGoto(predicate, firstLabel, firstLabel);
    if (ExecIfOp ifOp = dyn_cast<ExecIfOp>(operation)) {
      if (failed(lowerBlock(ifOp.getThenRegion().front())))
        return failure();
    } else if (failed(lowerBlock(
                   cast<UniformIfOp>(operation).getThenRegion().front()))) {
      return failure();
    }
    lowerGoto(std::nullopt, firstLabel, secondLabel);
    lowerJoin(firstLabel, secondLabel);
    if (ExecIfOp ifOp = dyn_cast<ExecIfOp>(operation)) {
      if (!ifOp.getElseRegion().empty() &&
          failed(lowerBlock(ifOp.getElseRegion().front())))
        return failure();
    } else {
      UniformIfOp uniformIf = cast<UniformIfOp>(operation);
      if (!uniformIf.getElseRegion().empty() &&
          failed(lowerBlock(uniformIf.getElseRegion().front())))
        return failure();
    }
    lowerJoin(secondLabel, finalLabel);
    program.items.emplace_back(Label{finalLabel});
    return success();
  }

  void lowerSync(SyncOp sync) {
    program.items.emplace_back(SyncInstruction{sync.getKind()});
  }

  struct MessageForm {
    SendFn function;
    uint32_t desc;
    uint32_t exdesc;
    bool writesDestination;
  };

  LogicalResult lowerMessage(Operation *operation) {
    MessageForm form;
    if (isa<LoadA64Op>(operation))
      form = {SendFn::ugm, 0x08200580, 0x0, true};
    else if (isa<StoreA64Op>(operation))
      form = {SendFn::ugm, 0x08000584, 0x0, false};
    else if (isa<LoadSLMOp>(operation))
      form = {SendFn::slm, 0x04200500, 0x0, true};
    else if (isa<StoreSLMOp>(operation))
      form = {SendFn::slm, 0x04000504, 0x0, false};
    else if (isa<AtomicIAddA64Op>(operation))
      form = {SendFn::ugm, 0x0410058C, 0x0, true};
    else if (isa<FenceSLMOp>(operation))
      form = {SendFn::slm, 0x0210001F, 0x0, true};
    else if (isa<BarrierSignalOp>(operation))
      form = {SendFn::gtwy, 0x02000004, 0x0, false};
    else if (isa<EotOp>(operation))
      form = {SendFn::gtwy, 0x02000010, 0x0, false};
    else if (LoadBlockA32Op block = dyn_cast<LoadBlockA32Op>(operation)) {
      uint32_t desc = block.getWords() == 32   ? 0x6229E500
                      : block.getWords() == 16 ? 0x6219D500
                                               : 0x6219C500;
      form = {SendFn::ugm, desc, 0xFF000000, true};
    } else {
      return operation->emitError("unknown message operation");
    }

    SendInstruction instruction;
    instruction.function = form.function;
    instruction.desc = form.desc;
    instruction.exdesc = form.exdesc;
    instruction.eot = isa<EotOp>(operation);
    instruction.execution.size = 1;
    if (IntegerAttr attr = operation->getAttrOfType<IntegerAttr>("execSize"))
      instruction.execution.size = attr.getInt();
    instruction.execution.noMask =
        operation->hasAttr("noMask") || instruction.execution.size == 1;
    instruction.swsb =
        getSendSwsb(operation, operation->getOperands(), instruction.eot);

    Type i32 = IntegerType::get(context, 32);
    Value address = operation->getOperand(0);
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
    instruction.swsb = getInOrderSwsb(await, await.getOperands());
    program.items.push_back(std::move(instruction));
  }

  void lowerSend(SendOp send) {
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
    instruction.exdesc = send.getExdesc();
    instruction.desc = send.getDesc();
    instruction.eot = send.getEot();
    instruction.swsb = getSendSwsb(send, send.getOperands(), send.getEot());
    program.items.push_back(std::move(instruction));
  }

  void lowerCompare(CmpOp compare) {
    CompareInstruction instruction;
    instruction.execution.size = compare.getExecSize();
    instruction.condition = compare.getCond();
    instruction.flag =
        getArfReference(cast<ARFType>(compare.getFlag().getType()), 0);
    instruction.dataType = getDataType(compare.getElemType());

    constexpr std::array<StringLiteral, 2> regionNames = {"src0Region",
                                                          "src1Region"};
    constexpr std::array<StringLiteral, 2> subNames = {"src0Sub", "src1Sub"};
    constexpr std::array<StringLiteral, 2> typeNames = {"src0Type", "src1Type"};
    for (auto [index, operand] : llvm::enumerate(compare.getOperands())) {
      Type sourceType =
          getSourceType(compare, typeNames[index], compare.getElemType());
      instruction.sources.push_back(getSourceOperand(
          operand, getSub(compare, subNames[index]), sourceType,
          getSourceRegion(compare, regionNames[index])));
    }
    instruction.swsb = getInOrderSwsb(compare, compare.getOperands());
    program.items.push_back(std::move(instruction));
  }

  LogicalResult lowerAlu(Operation *operation) {
    AluInstruction instruction;
    bool negateFirstSource = false;
    if (isa<MovOp>(operation))
      instruction.opcode = AluOpcode::mov;
    else if (isa<AddOp>(operation))
      instruction.opcode = AluOpcode::add;
    else if (isa<ShlOp>(operation))
      instruction.opcode = AluOpcode::shl;
    else if (isa<AndOp>(operation))
      instruction.opcode = AluOpcode::and_;
    else if (isa<OrOp>(operation))
      instruction.opcode = AluOpcode::or_;
    else if (isa<SubOp>(operation)) {
      instruction.opcode = AluOpcode::add;
      negateFirstSource = true;
    } else if (isa<Add3Op>(operation))
      instruction.opcode = AluOpcode::add3;
    else if (isa<MulOp>(operation))
      instruction.opcode = AluOpcode::mul;
    else
      return operation->emitError("unsupported operation in Xe emitter");

    TypeAttr elementTypeAttr = operation->getAttrOfType<TypeAttr>("elemType");
    if (!elementTypeAttr)
      return operation->emitError("expected an elemType attribute");
    Type elementType = elementTypeAttr.getValue();
    instruction.destinationType = getDataType(elementType);
    if (IntegerAttr attr = operation->getAttrOfType<IntegerAttr>("execSize"))
      instruction.execution.size = attr.getInt();
    else
      instruction.execution.size = 16;
    if (IntegerAttr attr = operation->getAttrOfType<IntegerAttr>("maskOffset"))
      instruction.execution.maskOffset = attr.getInt();
    instruction.execution.noMask = operation->hasAttr("noMask");

    Value destination = operation->getResult(0);
    uint32_t destinationStride = 1;
    if (DstRegionAttr attr =
            operation->getAttrOfType<DstRegionAttr>("dstRegion"))
      destinationStride = attr.getHstride();
    instruction.destination = Destination{
        getRegisterReference(destination.getType(), getSub(operation, "dstSub"),
                             elementType),
        destinationStride};

    constexpr std::array<StringLiteral, 3> regionNames = {
        "src0Region", "src1Region", "src2Region"};
    constexpr std::array<StringLiteral, 3> subNames = {"src0Sub", "src1Sub",
                                                       "src2Sub"};
    constexpr std::array<StringLiteral, 3> typeNames = {"src0Type", "src1Type",
                                                        "src2Type"};
    for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
      Type sourceType = getSourceType(operation, typeNames[index], elementType);
      SourceOperand source = getSourceOperand(
          operand, getSub(operation, subNames[index]), sourceType,
          getSourceRegion(operation, regionNames[index]));
      source.negate = negateFirstSource && index == 0;
      instruction.sources.push_back(std::move(source));
    }
    instruction.swsb = getInOrderSwsb(operation, operation->getOperands());
    program.items.push_back(std::move(instruction));
    return success();
  }

  MLIRContext *context;
  EmissionProgram &program;
  DenseMap<Value, Operation *> definingOperations;
  DenseMap<Operation *, int32_t> instructionIndices;
  int32_t nextInstruction = 0;
  int32_t nextToken = 0;
  uint32_t nextLabel = 1;
};

} // namespace

LogicalResult lowerToEmissionProgram(ModuleOp moduleOp,
                                     EmissionProgram &program) {
  func::FuncOp kernel;
  moduleOp.walk([&](func::FuncOp function) {
    if (!kernel)
      kernel = function;
  });
  if (!kernel)
    return moduleOp.emitError("no func.func kernel found"), failure();

  ProgramLowerer lowerer(moduleOp.getContext(), program);
  return lowerer.lower(kernel);
}

} // namespace inter::detail
