// Select closed XW semantic IR to XeMachine operations.

#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <array>
#include <optional>

namespace inter {
#define GEN_PASS_DEF_SELECTTOMACHINE
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;
using namespace inter::xemachine;

namespace {

constexpr int kInlineMirrorSize = 32;
constexpr int kLocalIdLoadOffset = 0x20;
constexpr int kPerThreadPayloadSize = 192;

struct ValueShape {
  Type elementType;
  int64_t cardinality;
  bool mask;
};

struct WideValue {
  Value low;
  Value high;
};

class SelectToMachine final
    : public inter::impl::SelectToMachineBase<SelectToMachine> {
public:
  void runOnOperation() override {
    SmallVector<func::FuncOp> kernels;
    getOperation().walk([&](func::FuncOp function) {
      if (function->hasAttr("xw.kernel") ||
          function->hasAttr("xemachine.kernel"))
        kernels.push_back(function);
    });
    if (kernels.empty()) {
      getOperation().emitError("no kernel function found");
      return signalPassFailure();
    }
    for (func::FuncOp kernel : kernels)
      if (failed(lowerKernel(kernel)))
        return signalPassFailure();
  }

private:
  MLIRContext *context = nullptr;
  std::optional<Location> location;
  std::optional<OpBuilder> builder;
  DenseMap<Value, Value> values;
  DenseMap<Value, WideValue> wideValues;
  DenseMap<Value, Value> localPointers;
  DenseMap<Value, WideValue> widePointers;
  ArrayAttr kernelArguments;
  Value memoryToken;
  Value payloadTail;
  std::array<Value, 3> localIds;
  std::array<bool, 3> usedIdAxes{};
  int64_t simdWidth = 0;
  bool prologueEmitted = false;

  Type i8() const { return IntegerType::get(context, 8); }
  Type i16() const { return IntegerType::get(context, 16); }
  Type i32() const { return IntegerType::get(context, 32); }
  Type i64() const { return IntegerType::get(context, 64); }
  Type reg(int64_t dwords) const { return RegType::get(context, dwords, -1); }
  TypeAttr typeAttr(Type type) const { return TypeAttr::get(type); }
  RegionAttr canonicalRegion() const {
    return RegionAttr::get(context, 1, 1, 0);
  }
  RegionAttr uniformRegion() const {
    return RegionAttr::get(context, 0, 1, 0);
  }
  DstRegionAttr canonicalDestination() const {
    return DstRegionAttr::get(context, 1);
  }

  Value immediate(int64_t value, Type elementType) {
    return ImmOp::create(*builder, *location, ImmType::get(context), value,
                         elementType)
        .getResult();
  }

  Value architecturalRegister(int index, int64_t dwords = 16) {
    return ArchRegOp::create(*builder, *location,
                             RegType::get(context, dwords, index),
                             builder->getI32IntegerAttr(index))
        .getResult();
  }

  FailureOr<ValueShape> getShape(Type type, Operation *owner) const {
    if (xw::SimdType simd = dyn_cast<xw::SimdType>(type))
      return ValueShape{simd.getElementType(), simd.getCardinality(), false};
    if (xw::MaskType mask = dyn_cast<xw::MaskType>(type))
      return ValueShape{i8(), mask.getCardinality(), true};
    if (isa<xw::MemTokenType>(type))
      return owner->emitOpError("memory token has no register shape"),
             failure();
    if (isa<IntegerType, FloatType, IndexType, VectorType, xw::PtrType>(type))
      return ValueShape{type, 1, false};
    return owner->emitOpError("unsupported XW value type ") << type, failure();
  }

  FailureOr<int64_t> getElementBits(Type type, Operation *owner) const {
    if (IntegerType integer = dyn_cast<IntegerType>(type))
      return integer.getWidth();
    if (FloatType floating = dyn_cast<FloatType>(type))
      return floating.getWidth();
    if (isa<IndexType>(type))
      return 64;
    if (isa<xw::PtrType>(type))
      return 64;
    if (VectorType vector = dyn_cast<VectorType>(type)) {
      FailureOr<int64_t> elementBits =
          getElementBits(vector.getElementType(), owner);
      if (failed(elementBits) || vector.isScalable())
        return owner->emitOpError("unsupported XW vector element type"),
               failure();
      return vector.getNumElements() * *elementBits;
    }
    return owner->emitOpError("unsupported machine element type ") << type,
           failure();
  }

  FailureOr<int64_t> getFootprint(Type type, Operation *owner) const {
    FailureOr<ValueShape> shape = getShape(type, owner);
    if (failed(shape))
      return failure();
    FailureOr<int64_t> bits = getElementBits(shape->elementType, owner);
    if (failed(bits))
      return failure();
    return (*bits * shape->cardinality + 31) / 32;
  }

  bool isWideSimd(Type type) const {
    xw::SimdType simd = dyn_cast<xw::SimdType>(type);
    if (!simd || simd.getCardinality() != 32)
      return false;
    Type elementType = simd.getElementType();
    if (IntegerType integer = dyn_cast<IntegerType>(elementType))
      return integer.getWidth() == 64;
    if (FloatType floating = dyn_cast<FloatType>(elementType))
      return floating.getWidth() == 64;
    return isa<IndexType, xw::PtrType>(elementType);
  }

  RegionAttr sourceRegion(Value source, int64_t executionSize,
                          Operation *owner) const {
    if (values.lookup(source).getDefiningOp<ImmOp>())
      return RegionAttr();
    FailureOr<ValueShape> shape = getShape(source.getType(), owner);
    if (failed(shape) || shape->cardinality == 1)
      return uniformRegion();
    if (shape->cardinality == executionSize)
      return canonicalRegion();
    assert(executionSize % shape->cardinality == 0 &&
           "XW verifier must enforce compatible cardinalities");
    return RegionAttr::get(context, 1,
                           executionSize / shape->cardinality, 0);
  }

  Value emitMove(Type destination, Type elementType, int64_t executionSize,
                 Value source, RegionAttr region, bool noMask = false,
                 int64_t maskOffset = 0, IntegerAttr sourceSub = {}) {
    return MovOp::create(*builder, *location, destination, elementType,
                         executionSize, canonicalDestination(), region,
                         IntegerAttr(), sourceSub, TypeAttr(), noMask,
                         maskOffset, source)
        .getResult();
  }

  void emitSync(SyncKind kind) {
    SyncOp operation = SyncOp::create(
        *builder, *location, MemTokenType::get(context),
        SyncKindAttr::get(context, kind), memoryToken);
    memoryToken = operation.getToken();
  }

  FailureOr<KernelArgAttr> getKernelArgument(BlockArgument argument,
                                              Operation *owner) const {
    unsigned index = argument.getArgNumber();
    if (index >= kernelArguments.size())
      return owner->emitOpError("kernel argument index is out of range"),
             failure();
    KernelArgAttr descriptor = dyn_cast<KernelArgAttr>(kernelArguments[index]);
    if (!descriptor)
      return owner->emitOpError("invalid machine kernel argument descriptor"),
             failure();
    return descriptor;
  }

  FailureOr<ArrayAttr> importKernelArguments(func::FuncOp kernel) const {
    ArrayAttr descriptors = kernel->getAttrOfType<ArrayAttr>("xw.kernel_args");
    if (!descriptors)
      return kernel.emitOpError("missing xw.kernel_args"), failure();
    SmallVector<Attribute> machineDescriptors;
    machineDescriptors.reserve(descriptors.size());
    Builder attrBuilder(kernel.getContext());
    for (Attribute attribute : descriptors) {
      DictionaryAttr descriptor = dyn_cast<DictionaryAttr>(attribute);
      if (!descriptor)
        return kernel.emitOpError("invalid XW kernel argument descriptor"),
               failure();
      StringAttr kind = descriptor.getAs<StringAttr>("kind");
      IntegerAttr size = descriptor.getAs<IntegerAttr>("size");
      IntegerAttr offset = descriptor.getAs<IntegerAttr>("offset");
      if (!kind || !size || !offset)
        return kernel.emitOpError("incomplete XW kernel argument descriptor"),
               failure();
      bool pointer = kind.getValue() == "pointer";
      StringRef addressSpace = "none";
      if (IntegerAttr space = descriptor.getAs<IntegerAttr>("address_space")) {
        static constexpr std::array<StringLiteral, 5> names = {
            "private", "global", "constant", "local", "generic"};
        int64_t value = space.getInt();
        addressSpace = value >= 0 && value < static_cast<int64_t>(names.size())
                           ? names[value]
                           : "unknown";
      }
      uint64_t alignment = pointer ? 8 : std::min<uint64_t>(size.getInt(), 8);
      machineDescriptors.push_back(KernelArgAttr::get(
          kernel.getContext(), pointer ? KernelArgKind::by_pointer
                                       : KernelArgKind::by_value,
          attrBuilder.getStringAttr(addressSpace),
          attrBuilder.getStringAttr(pointer ? "read_write" : "none"),
          size.getInt(), alignment, offset.getInt()));
    }
    return attrBuilder.getArrayAttr(machineDescriptors);
  }

  FailureOr<std::pair<Value, int64_t>>
  getPayloadLocation(BlockArgument argument, Operation *owner) {
    FailureOr<KernelArgAttr> descriptor = getKernelArgument(argument, owner);
    if (failed(descriptor))
      return failure();
    uint64_t offset = descriptor->getOffset();
    uint64_t size = descriptor->getSize();
    if (offset < kInlineMirrorSize)
      return std::pair<Value, int64_t>{architecturalRegister(4), offset / size};
    if (!payloadTail || offset + size > 64)
      return owner->emitOpError("kernel argument is outside loaded payload"),
             failure();
    return std::pair<Value, int64_t>{payloadTail,
                                     (offset - kInlineMirrorSize) / size};
  }

  LogicalResult validateKernelArguments(func::FuncOp kernel) {
    if (!kernelArguments ||
        kernelArguments.size() != kernel.getNumArguments())
      return kernel.emitOpError("kernel argument descriptor count mismatch");
    if (failed(verifyKernelArgLayout(kernelArguments, kernel.getOperation())))
      return failure();
    for (BlockArgument argument : kernel.getArguments()) {
      FailureOr<KernelArgAttr> descriptor =
          getKernelArgument(argument, kernel.getOperation());
      if (failed(descriptor))
        return failure();
      Type type = argument.getType();
      if (isa<xw::SimdType, xw::MaskType, xw::MemTokenType>(type))
        return kernel.emitOpError("kernel ABI arguments must be bare values");
      if (xw::PtrType pointer = dyn_cast<xw::PtrType>(type)) {
        if (descriptor->getKind() != KernelArgKind::by_pointer ||
            descriptor->getSize() != 8)
          return kernel.emitOpError("pointer argument descriptor mismatch");
        StringRef expectedSpace = getAddressSpaceName(pointer.getAddressSpace());
        if (descriptor->getAddressSpace().getValue() != expectedSpace)
          return kernel.emitOpError("pointer address-space descriptor mismatch");
      } else {
        FailureOr<int64_t> bits = getElementBits(type, kernel.getOperation());
        if (failed(bits))
          return failure();
        uint64_t size = (*bits + 7) / 8;
        if (descriptor->getKind() != KernelArgKind::by_value ||
            descriptor->getSize() != size)
          return kernel.emitOpError("by-value argument descriptor mismatch");
      }
    }
    return success();
  }

  StringRef getAddressSpaceName(Attribute addressSpace) const {
    if (isa<xw::PrivateAddressSpaceAttr>(addressSpace))
      return "private";
    if (isa<xw::GlobalAddressSpaceAttr>(addressSpace))
      return "global";
    if (isa<xw::ConstantAddressSpaceAttr>(addressSpace))
      return "constant";
    if (isa<xw::LocalAddressSpaceAttr>(addressSpace))
      return "local";
    if (isa<xw::GenericAddressSpaceAttr>(addressSpace))
      return "generic";
    return "";
  }

  FailureOr<uint64_t> getSlmSize(func::FuncOp kernel) const {
    uint64_t size = 0;
    WalkResult result = kernel.walk([&](xw::AllocOp allocation) {
      uint64_t offset = allocation.getOffset().value_or(0);
      uint64_t alignment = allocation.getAlign();
      if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        allocation.emitOpError("SLM allocation alignment must be a power of two");
        return WalkResult::interrupt();
      }
      uint64_t aligned = (offset + alignment - 1) & ~(alignment - 1);
      size = std::max(size, aligned + allocation.getBytesize());
      return WalkResult::advance();
    });
    kernel.walk([&](xw::LocalMemoryBaseOp base) {
      IntegerAttr bytes = base->getAttrOfType<IntegerAttr>("xw.bytesize");
      IntegerAttr alignment = base->getAttrOfType<IntegerAttr>("xw.alignment");
      if (!bytes)
        return;
      uint64_t align = alignment ? alignment.getInt() : 1;
      uint64_t offset = base.getOffset();
      uint64_t aligned = (offset + align - 1) & ~(align - 1);
      size = std::max(size, aligned + bytes.getInt());
    });
    if (result.wasInterrupted())
      return failure();
    return size;
  }

  LogicalResult lowerKernel(func::FuncOp kernel) {
    if (!kernel.getFunctionType().getResults().empty())
      return kernel.emitOpError("kernel functions may not return values");
    if (&kernel.getBody().front() != &kernel.getBody().back())
      return kernel.emitOpError("unstructured kernel CFG reached selection");
    context = kernel.getContext();
    location = kernel.getLoc();
    values.clear();
    wideValues.clear();
    localPointers.clear();
    widePointers.clear();
    memoryToken = nullptr;
    payloadTail = nullptr;
    localIds.fill(Value());
    usedIdAxes.fill(false);
    prologueEmitted = false;
    kernelArguments = kernel->getAttrOfType<ArrayAttr>(kKernelArgsAttrName);
    if (!kernelArguments && kernel->hasAttr("xw.kernel")) {
      FailureOr<ArrayAttr> imported = importKernelArguments(kernel);
      if (failed(imported))
        return failure();
      kernelArguments = *imported;
    }
    if (!kernelArguments && kernel.getNumArguments() == 0)
      kernelArguments = ArrayAttr::get(context, {});
    if (failed(validateKernelArguments(kernel)))
      return failure();

    IntegerAttr width = kernel->getAttrOfType<IntegerAttr>(
        xw::XWDialect::getSimdWidthAttrName());
    if (!width)
      return kernel.emitOpError("missing xw.simd_width");
    simdWidth = width.getInt();

    WalkResult idWalk = kernel.walk([&](Operation *operation) {
      if (!isa<xw::GlobalIdOp, xw::LocalIdOp>(operation))
        return WalkResult::advance();
      int64_t dim = cast<IntegerAttr>(operation->getAttr("dim")).getInt();
      if (dim < 0 || dim >= 3) {
        operation->emitOpError("ID axis must be 0, 1, or 2");
        return WalkResult::interrupt();
      }
      usedIdAxes[dim] = true;
      return WalkResult::advance();
    });
    if (idWalk.wasInterrupted())
      return failure();

    FailureOr<uint64_t> slmSize = getSlmSize(kernel);
    if (failed(slmSize))
      return failure();

    OpBuilder moduleBuilder(kernel);
    func::FuncOp machineFunction = func::FuncOp::create(
        moduleBuilder, kernel.getLoc(), (kernel.getName() + "_xm").str(),
        moduleBuilder.getFunctionType({}, {}));
    machineFunction->setAttr(
        kTargetAttrName,
        TargetAttr::get(context, moduleBuilder.getStringAttr("bmg")));
    machineFunction->setAttr(kKernelArgsAttrName, kernelArguments);
    machineFunction->setAttr(kGrfCountAttrName,
                             moduleBuilder.getI32IntegerAttr(128));
    machineFunction->setAttr(kReservedGrfCountAttrName,
                             moduleBuilder.getI32IntegerAttr(5));
    machineFunction->setAttr(kSimdSizeAttrName,
                             moduleBuilder.getI32IntegerAttr(simdWidth));
    if (*slmSize != 0)
      machineFunction->setAttr(kSlmSizeAttrName,
                               moduleBuilder.getI64IntegerAttr(*slmSize));

    bool usesThreadIds = usedIdAxes[0] || usedIdAxes[1] || usedIdAxes[2];
    bool needsPayload = usesThreadIds || !kernelArguments.empty();
    if (usesThreadIds)
      machineFunction->setAttr(kUsesThreadIdsAttrName,
                               moduleBuilder.getUnitAttr());
    if (needsPayload) {
      machineFunction->setAttr(kInlineDataPayloadSizeAttrName,
                               moduleBuilder.getI32IntegerAttr(32));
      machineFunction->setAttr(kPayloadEntryOffsetAttrName,
                               moduleBuilder.getI32IntegerAttr(192));
    }
    if (usesThreadIds)
      machineFunction->setAttr(kPerThreadPayloadSizeAttrName,
                               moduleBuilder.getI32IntegerAttr(192));

    builder = OpBuilder::atBlockBegin(machineFunction.addEntryBlock());
    if (needsPayload && failed(emitPrologue()))
      return failure();
    if (failed(lowerBlock(kernel.getBody().front())))
      return failure();
    func::ReturnOp::create(*builder, *location);

    std::string name = kernel.getName().str();
    kernel.erase();
    machineFunction.setName(StringAttr::get(context, name));
    return success();
  }

  void emitArgumentEntry() {
    MovOp::create(*builder, *location, RegType::get(context, 16, 4), i32(), 8,
                  canonicalDestination(), canonicalRegion(), IntegerAttr(),
                  IntegerAttr(), TypeAttr(), true, 0,
                  architecturalRegister(1));
    for (unsigned index = 0; index < 11; ++index)
      emitSync(SyncKind::nop);
  }

  void emitLocalIdEntry() {
    Value r0 = architecturalRegister(0);
    Value r1 = architecturalRegister(1);
    MovOp::create(*builder, *location, RegType::get(context, 16, 4), i32(), 8,
                  canonicalDestination(), canonicalRegion(), IntegerAttr(),
                  IntegerAttr(), TypeAttr(), true, 0, r1);
    Value base =
        AndOp::create(*builder, *location, RegType::get(context, 16, 5), i32(),
                      1, canonicalDestination(), uniformRegion(), RegionAttr(),
                      IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                      TypeAttr(), true, 0, r0, immediate(0xFFFFFFC0, i32()))
            .getResult();
    Value perThreadBase =
        AddOp::create(*builder, *location, RegType::get(context, 16, 6), i32(),
                      1, canonicalDestination(), uniformRegion(), RegionAttr(),
                      IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                      TypeAttr(), true, 0, base,
                      immediate(kLocalIdLoadOffset, i32()))
            .getResult();
    Value threadSlot =
        AndOp::create(*builder, *location, RegType::get(context, 16, 7), i32(),
                      1, canonicalDestination(), uniformRegion(), RegionAttr(),
                      IntegerAttr(), builder->getI32IntegerAttr(4),
                      IntegerAttr(), TypeAttr(), TypeAttr(), true, 0, r0,
                      immediate(0xff, i32()))
            .getResult();
    Value offsetAccumulator =
        MulOp::create(*builder, *location,
                      ARFType::get(context, ARFFile::acc, 16, 0), i32(), 1,
                      canonicalDestination(), uniformRegion(), RegionAttr(),
                      IntegerAttr(), IntegerAttr(), IntegerAttr(), true, 0,
                      threadSlot, immediate(kPerThreadPayloadSize, i32()))
            .getResult();
    Value threadOffset = emitMove(RegType::get(context, 16, 8), i32(), 1,
                                  offsetAccumulator, uniformRegion(), true);
    Value address =
        AddOp::create(*builder, *location, RegType::get(context, 16, 9), i32(),
                      1, canonicalDestination(), uniformRegion(),
                      uniformRegion(), IntegerAttr(), IntegerAttr(),
                      IntegerAttr(), TypeAttr(), TypeAttr(), true, 0,
                      perThreadBase, threadOffset)
            .getResult();

    bool needsY = usedIdAxes[1];
    int firstWords = needsY ? 32 : 16;
    LoadBlockA32Op first = LoadBlockA32Op::create(
        *builder, *location, RegType::get(context, firstWords, 1),
        MemTokenType::get(context), address, Value(), firstWords);
    first->setAttr(kAllowFixedOverlapAttrName, builder->getUnitAttr());
    memoryToken = first.getToken();
    if (usedIdAxes[2]) {
      Value zAddress =
          AddOp::create(*builder, *location, reg(16), i32(), 1,
                        canonicalDestination(), uniformRegion(), RegionAttr(),
                        IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                        TypeAttr(), true, 0, address, immediate(128, i32()))
              .getResult();
      LoadBlockA32Op z = LoadBlockA32Op::create(
          *builder, *location, RegType::get(context, 16, 3),
          MemTokenType::get(context), zAddress, memoryToken, 16);
      z->setAttr(kAllowFixedOverlapAttrName, builder->getUnitAttr());
      memoryToken = z.getToken();
    }
    for (unsigned index = 0; index < 4; ++index)
      emitSync(SyncKind::nop);
  }

  LogicalResult emitPrologue() {
    if (prologueEmitted)
      return success();
    prologueEmitted = true;
    bool usesThreadIds = usedIdAxes[0] || usedIdAxes[1] || usedIdAxes[2];
    if (usesThreadIds)
      emitLocalIdEntry();
    else
      emitArgumentEntry();
    emitSync(SyncKind::allwr);

    Value r0 = architecturalRegister(0);
    Value base =
        AndOp::create(*builder, *location, reg(16), i32(), 1,
                      canonicalDestination(), uniformRegion(), RegionAttr(),
                      IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                      TypeAttr(), true, 0, r0,
                      immediate(0xFFFFFFC0, i32()))
            .getResult();
    LoadBlockA32Op tail = LoadBlockA32Op::create(
        *builder, *location, reg(16), MemTokenType::get(context), base,
        Value(), 8);
    memoryToken = tail.getToken();
    payloadTail = tail.getDst();
    for (unsigned axis = 0; axis < 3; ++axis)
      if (usedIdAxes[axis])
        localIds[axis] = architecturalRegister(1 + axis);
    return success();
  }

  FailureOr<Value> lowerBareArgument(BlockArgument argument,
                                     Operation *owner) {
    if (Value found = values.lookup(argument))
      return found;
    FailureOr<KernelArgAttr> descriptor = getKernelArgument(argument, owner);
    FailureOr<std::pair<Value, int64_t>> payload =
        getPayloadLocation(argument, owner);
    if (failed(descriptor) || failed(payload))
      return failure();
    Type elementType = isa<xw::PtrType>(argument.getType()) ? i64()
                                                            : argument.getType();
    FailureOr<int64_t> bits = getElementBits(elementType, owner);
    if (failed(bits))
      return failure();
    Value result = emitMove(reg((*bits + 31) / 32), elementType, 1,
                            payload->first, uniformRegion(), true, 0,
                            builder->getI32IntegerAttr(payload->second));
    values[argument] = result;
    return result;
  }

  FailureOr<int64_t> getConstantBits(xw::ConstantOp constant) const {
    Attribute value = constant.getValue();
    if (IntegerAttr integer = dyn_cast<IntegerAttr>(value))
      return integer.getValue().getSExtValue();
    if (FloatAttr floating = dyn_cast<FloatAttr>(value))
      return static_cast<int64_t>(
          floating.getValue().bitcastToAPInt().getZExtValue());
    if (DenseElementsAttr dense = dyn_cast<DenseElementsAttr>(value)) {
      if (!dense.isSplat())
        return constant.emitOpError(
                   "non-splat SIMD constants have no machine immediate form"),
               failure();
      if (dense.getElementType().isIntOrIndex())
        return dense.getSplatValue<APInt>().getSExtValue();
      if (isa<FloatType>(dense.getElementType()))
        return static_cast<int64_t>(
            dense.getSplatValue<APFloat>().bitcastToAPInt().getZExtValue());
    }
    return constant.emitOpError("unsupported XW constant attribute"), failure();
  }

  FailureOr<Value> getValue(Value source, Operation *owner) {
    if (Value found = values.lookup(source))
      return found;
    if (BlockArgument argument = dyn_cast<BlockArgument>(source))
      return lowerBareArgument(argument, owner);
    if (xw::ConstantOp constant = source.getDefiningOp<xw::ConstantOp>()) {
      FailureOr<ValueShape> shape = getShape(source.getType(), owner);
      FailureOr<int64_t> bits = getConstantBits(constant);
      if (failed(shape) || failed(bits))
        return failure();
      Value result = immediate(*bits, shape->elementType);
      if (isWideSimd(source.getType()))
        wideValues[source] = {result, result};
      else
        values[source] = result;
      return result;
    }
    return owner->emitOpError("operand was not selected"), failure();
  }

  FailureOr<WideValue> getWideValue(Value source, Operation *owner) {
    auto found = wideValues.find(source);
    if (found != wideValues.end())
      return found->second;
    FailureOr<Value> scalar = getValue(source, owner);
    if (failed(scalar))
      return failure();
    if (scalar->getDefiningOp<ImmOp>())
      return WideValue{*scalar, *scalar};
    if (source.getType().isIntOrIndexOrFloat() || isa<xw::PtrType>(source.getType()))
      return WideValue{*scalar, *scalar};
    return owner->emitOpError("SIMD32 i64 operand was not decomposed"), failure();
  }

  FailureOr<Value> materialize(Value source, Operation *owner) {
    FailureOr<Value> value = getValue(source, owner);
    FailureOr<ValueShape> shape = getShape(source.getType(), owner);
    FailureOr<int64_t> footprint = getFootprint(source.getType(), owner);
    if (failed(value) || failed(shape) || failed(footprint))
      return failure();
    if (!value->getDefiningOp<ImmOp>())
      return *value;
    return emitMove(reg(*footprint), shape->elementType, shape->cardinality,
                    *value, RegionAttr(), shape->cardinality == 1);
  }

  LogicalResult lowerView(Operation *operation, Value source) {
    FailureOr<ValueShape> resultShape =
        getShape(operation->getResult(0).getType(), operation);
    FailureOr<ValueShape> sourceShape = getShape(source.getType(), operation);
    FailureOr<Value> input = getValue(source, operation);
    FailureOr<int64_t> footprint =
        getFootprint(operation->getResult(0).getType(), operation);
    if (failed(resultShape) || failed(sourceShape) || failed(input) ||
        failed(footprint))
      return failure();
    if (isWideSimd(operation->getResult(0).getType())) {
      auto moveHalf = [&](int64_t maskOffset) {
        RegionAttr region = sourceShape->cardinality == 1
                                ? uniformRegion()
                                : RegionAttr::get(
                                      context, 1,
                                      16 / sourceShape->cardinality, 0);
        return emitMove(reg(32), resultShape->elementType, 16, *input, region,
                        false, maskOffset);
      };
      wideValues[operation->getResult(0)] = {moveHalf(0), moveHalf(16)};
      return success();
    }
    RegionAttr region = resultShape->cardinality == 1 &&
                                sourceShape->cardinality > 1
                            ? uniformRegion()
                            : sourceRegion(source, resultShape->cardinality,
                                           operation);
    Value result = emitMove(reg(*footprint), resultShape->elementType,
                            resultShape->cardinality, *input, region,
                            resultShape->cardinality == 1);
    values[operation->getResult(0)] = result;
    return success();
  }

  LogicalResult lowerBinary(xw::BinaryOp operation) {
    FailureOr<ValueShape> shape = getShape(operation.getType(), operation);
    if (failed(shape) || shape->mask)
      return failure();
    if (isWideSimd(operation.getType()))
      return lowerWideBinary(operation);
    FailureOr<Value> lhs = getValue(operation.getLhs(), operation);
    FailureOr<Value> rhs = getValue(operation.getRhs(), operation);
    FailureOr<int64_t> footprint = getFootprint(operation.getType(), operation);
    if (failed(lhs) || failed(rhs) || failed(footprint))
      return failure();
    RegionAttr lhsRegion =
        sourceRegion(operation.getLhs(), shape->cardinality, operation);
    RegionAttr rhsRegion =
        sourceRegion(operation.getRhs(), shape->cardinality, operation);
    Value result;
    switch (operation.getKind()) {
    case xw::BinaryKind::AddI:
      result = AddOp::create(
                   *builder, *location, reg(*footprint), shape->elementType,
                   shape->cardinality, canonicalDestination(), lhsRegion,
                   rhsRegion, IntegerAttr(), IntegerAttr(), IntegerAttr(),
                   TypeAttr(), TypeAttr(), shape->cardinality == 1, 0, *lhs,
                   *rhs)
                   .getResult();
      break;
    case xw::BinaryKind::SubI:
      result = SubOp::create(
                   *builder, *location, reg(*footprint), shape->elementType,
                   shape->cardinality, canonicalDestination(), rhsRegion,
                   lhsRegion, IntegerAttr(), IntegerAttr(), IntegerAttr(),
                   TypeAttr(), TypeAttr(), shape->cardinality == 1, 0, *rhs,
                   *lhs)
                   .getResult();
      break;
    case xw::BinaryKind::ShLI:
    case xw::BinaryKind::ShRUI:
      if (operation.getKind() == xw::BinaryKind::ShLI)
        result = ShlOp::create(
                      *builder, *location, reg(*footprint), shape->elementType,
                      shape->cardinality, canonicalDestination(), lhsRegion,
                      rhsRegion, IntegerAttr(), IntegerAttr(), IntegerAttr(),
                      TypeAttr(), shape->elementType.isInteger(64)
                                      ? typeAttr(i16())
                                      : TypeAttr(),
                      shape->cardinality == 1, 0, *lhs, *rhs)
                      .getResult();
      else
        result = ShrOp::create(
                      *builder, *location, reg(*footprint), shape->elementType,
                      shape->cardinality, canonicalDestination(), lhsRegion,
                      rhsRegion, IntegerAttr(), IntegerAttr(), IntegerAttr(),
                      TypeAttr(), shape->elementType.isInteger(64)
                                      ? typeAttr(i16())
                                      : TypeAttr(),
                      shape->cardinality == 1, 0, *lhs, *rhs)
                      .getResult();
      break;
    case xw::BinaryKind::AndI:
    case xw::BinaryKind::OrI:
      if (operation.getKind() == xw::BinaryKind::AndI)
        result = AndOp::create(
                     *builder, *location, reg(*footprint), shape->elementType,
                     shape->cardinality, canonicalDestination(), lhsRegion,
                     rhsRegion, IntegerAttr(), IntegerAttr(), IntegerAttr(),
                     TypeAttr(), TypeAttr(), shape->cardinality == 1, 0, *lhs,
                     *rhs)
                     .getResult();
      else
        result = OrOp::create(
                     *builder, *location, reg(*footprint), shape->elementType,
                     shape->cardinality, canonicalDestination(), lhsRegion,
                     rhsRegion, IntegerAttr(), IntegerAttr(), IntegerAttr(),
                     TypeAttr(), TypeAttr(), shape->cardinality == 1, 0, *lhs,
                     *rhs)
                     .getResult();
      break;
    case xw::BinaryKind::MulI: {
      Value accumulator =
          MulOp::create(*builder, *location,
                        ARFType::get(context, ARFFile::acc, *footprint, 0),
                        shape->elementType, shape->cardinality,
                        canonicalDestination(), lhsRegion, rhsRegion,
                        IntegerAttr(), IntegerAttr(), IntegerAttr(),
                        shape->cardinality == 1, 0, *lhs, *rhs)
              .getResult();
      result = emitMove(reg(*footprint), shape->elementType,
                        shape->cardinality, accumulator, canonicalRegion(),
                        shape->cardinality == 1);
      break;
    }
    default:
      return operation.emitOpError(
          "integer operation has no XeMachine instruction selection");
    }
    values[operation.getResult()] = result;
    return success();
  }

  LogicalResult lowerWideBinary(xw::BinaryOp operation) {
    if (operation.getKind() != xw::BinaryKind::AddI &&
        operation.getKind() != xw::BinaryKind::SubI &&
        operation.getKind() != xw::BinaryKind::ShLI &&
        operation.getKind() != xw::BinaryKind::ShRUI)
      return operation.emitOpError(
          "SIMD32 i64 operation has no decomposed machine selection");
    FailureOr<WideValue> lhs = getWideValue(operation.getLhs(), operation);
    FailureOr<WideValue> rhs = getWideValue(operation.getRhs(), operation);
    if (failed(lhs) || failed(rhs))
      return failure();
    auto emitHalf = [&](Value left, Value right, int64_t maskOffset) {
      RegionAttr leftRegion = left.getDefiningOp<ImmOp>() ? RegionAttr()
                                                          : canonicalRegion();
      RegionAttr rightRegion = right.getDefiningOp<ImmOp>() ? RegionAttr()
                                                             : canonicalRegion();
      if (operation.getKind() == xw::BinaryKind::AddI)
        return AddOp::create(
                   *builder, *location, reg(32), i64(), 16,
                   canonicalDestination(), leftRegion, rightRegion,
                   IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                   TypeAttr(), false, maskOffset, left, right)
            .getResult();
      if (operation.getKind() == xw::BinaryKind::SubI)
        return SubOp::create(
                   *builder, *location, reg(32), i64(), 16,
                   canonicalDestination(), rightRegion, leftRegion,
                   IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                   TypeAttr(), false, maskOffset, right, left)
            .getResult();
      if (operation.getKind() == xw::BinaryKind::ShLI)
        return ShlOp::create(
                   *builder, *location, reg(32), i64(), 16,
                   canonicalDestination(), leftRegion, rightRegion,
                   IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                   typeAttr(i16()), false, maskOffset, left, right)
            .getResult();
      return ShrOp::create(
                 *builder, *location, reg(32), i64(), 16,
                 canonicalDestination(), leftRegion, rightRegion,
                 IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                 typeAttr(i16()), false, maskOffset, left, right)
          .getResult();
    };
    WideValue result{emitHalf(lhs->low, rhs->low, 0),
                     emitHalf(lhs->high, rhs->high, 16)};
    wideValues[operation.getResult()] = result;
    return success();
  }

  LogicalResult lowerFloatBinary(Operation *operation, bool subtract) {
    FailureOr<ValueShape> shape =
        getShape(operation->getResult(0).getType(), operation);
    FailureOr<Value> lhs = getValue(operation->getOperand(0), operation);
    FailureOr<Value> rhs = getValue(operation->getOperand(1), operation);
    FailureOr<int64_t> footprint =
        getFootprint(operation->getResult(0).getType(), operation);
    if (failed(shape) || failed(lhs) || failed(rhs) || failed(footprint))
      return failure();
    RegionAttr lhsRegion =
        sourceRegion(operation->getOperand(0), shape->cardinality, operation);
    RegionAttr rhsRegion =
        sourceRegion(operation->getOperand(1), shape->cardinality, operation);
    Value result;
    if (subtract)
      result = SubOp::create(
                   *builder, *location, reg(*footprint), shape->elementType,
                   shape->cardinality, canonicalDestination(), rhsRegion,
                   lhsRegion, IntegerAttr(), IntegerAttr(), IntegerAttr(),
                   TypeAttr(), TypeAttr(), false, 0, *rhs, *lhs)
                   .getResult();
    else
      result = AddOp::create(
                   *builder, *location, reg(*footprint), shape->elementType,
                   shape->cardinality, canonicalDestination(), lhsRegion,
                   rhsRegion, IntegerAttr(), IntegerAttr(), IntegerAttr(),
                   TypeAttr(), TypeAttr(), false, 0, *lhs, *rhs)
                   .getResult();
    values[operation->getResult(0)] = result;
    return success();
  }

  LogicalResult lowerFloatMultiply(xw::FMulOp operation) {
    FailureOr<ValueShape> shape = getShape(operation.getType(), operation);
    FailureOr<Value> lhs = getValue(operation.getLhs(), operation);
    FailureOr<Value> rhs = getValue(operation.getRhs(), operation);
    FailureOr<int64_t> footprint = getFootprint(operation.getType(), operation);
    if (failed(shape) || failed(lhs) || failed(rhs) || failed(footprint))
      return failure();
    Value accumulator =
        MulOp::create(
            *builder, *location,
            ARFType::get(context, ARFFile::acc, *footprint, 0),
            shape->elementType, shape->cardinality, canonicalDestination(),
            sourceRegion(operation.getLhs(), shape->cardinality, operation),
            sourceRegion(operation.getRhs(), shape->cardinality, operation),
            IntegerAttr(), IntegerAttr(), IntegerAttr(), false, 0, *lhs, *rhs)
            .getResult();
    values[operation.getResult()] =
        emitMove(reg(*footprint), shape->elementType, shape->cardinality,
                 accumulator, canonicalRegion());
    return success();
  }

  LogicalResult lowerCast(xw::CastOp operation) {
    if (isWideSimd(operation.getType())) {
      FailureOr<Value> source = getValue(operation.getSource(), operation);
      FailureOr<ValueShape> sourceShape =
          getShape(operation.getSource().getType(), operation);
      if (failed(source) || failed(sourceShape))
        return failure();
      auto half = [&](int64_t offset) {
        MovOp move = MovOp::create(
            *builder, *location, reg(32), i64(), 16,
            canonicalDestination(), canonicalRegion(), IntegerAttr(),
            builder->getI32IntegerAttr(offset),
            typeAttr(sourceShape->elementType), false, offset, *source);
        if (operation.getKind() == xw::CastKind::IntConvert &&
            operation.getPolicy()) {
          DictionaryAttr policy = *operation.getPolicy();
          auto signedness = dyn_cast_or_null<xw::CastSignednessPolicyAttr>(
              policy.get("signedness"));
          if (signedness && signedness.getValue() == xw::CastSignedness::Signed)
            move->setAttr("signedSource", builder->getUnitAttr());
        }
        return move.getResult();
      };
      wideValues[operation.getResult()] = {half(0), half(16)};
      return success();
    }
    FailureOr<ValueShape> sourceShape =
        getShape(operation.getSource().getType(), operation);
    FailureOr<ValueShape> resultShape = getShape(operation.getType(), operation);
    FailureOr<Value> source = getValue(operation.getSource(), operation);
    FailureOr<int64_t> footprint = getFootprint(operation.getType(), operation);
    if (failed(sourceShape) || failed(resultShape) || failed(source) ||
        failed(footprint))
      return failure();
    MovOp move = MovOp::create(
        *builder, *location, reg(*footprint), resultShape->elementType,
        resultShape->cardinality, canonicalDestination(),
        sourceRegion(operation.getSource(), resultShape->cardinality, operation),
        IntegerAttr(), IntegerAttr(), typeAttr(sourceShape->elementType),
        resultShape->cardinality == 1, 0, *source);
    if (operation.getPolicy()) {
      DictionaryAttr policy = *operation.getPolicy();
      auto signedness = dyn_cast_or_null<xw::CastSignednessPolicyAttr>(
          policy.get("signedness"));
      if (signedness && signedness.getValue() == xw::CastSignedness::Signed)
        move->setAttr("signedSource", builder->getUnitAttr());
    }
    values[operation.getResult()] = move.getResult();
    return success();
  }

  std::optional<CondModifier> mapPredicate(arith::CmpIPredicate predicate) {
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      return CondModifier::eq;
    case arith::CmpIPredicate::ne:
      return CondModifier::ne;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      return CondModifier::lt;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return CondModifier::le;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return CondModifier::gt;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return CondModifier::ge;
    }
    return std::nullopt;
  }

  std::optional<CondModifier> mapPredicate(arith::CmpFPredicate predicate) {
    switch (predicate) {
    case arith::CmpFPredicate::OEQ:
      return CondModifier::eq;
    case arith::CmpFPredicate::ONE:
      return CondModifier::ne;
    case arith::CmpFPredicate::OLT:
      return CondModifier::lt;
    case arith::CmpFPredicate::OLE:
      return CondModifier::le;
    case arith::CmpFPredicate::OGT:
      return CondModifier::gt;
    case arith::CmpFPredicate::OGE:
      return CondModifier::ge;
    default:
      return std::nullopt;
    }
  }

  LogicalResult lowerCompare(xw::CmpIOp operation) {
    FailureOr<ValueShape> resultShape =
        getShape(operation.getResult().getType(), operation);
    FailureOr<ValueShape> operandShape =
        getShape(operation.getLhs().getType(), operation);
    FailureOr<Value> lhs = getValue(operation.getLhs(), operation);
    FailureOr<Value> rhs = getValue(operation.getRhs(), operation);
    if (failed(resultShape) || failed(operandShape) || failed(lhs) || failed(rhs))
      return failure();
    std::optional<CondModifier> condition = mapPredicate(operation.getPredicate());
    if (!condition)
      return operation.emitOpError("unsupported integer comparison predicate");
    int64_t executionSize = resultShape->cardinality;
    CmpOp compare = CmpOp::create(
        *builder, *location, ARFType::get(context, ARFFile::f, 2, -1),
        CondModifierAttr::get(context, *condition),
        typeAttr(operandShape->elementType),
        builder->getI32IntegerAttr(executionSize),
        sourceRegion(operation.getLhs(), executionSize, operation),
        sourceRegion(operation.getRhs(), executionSize, operation),
        IntegerAttr(), IntegerAttr(), TypeAttr(), TypeAttr(), *lhs, *rhs);
    arith::CmpIPredicate predicate = operation.getPredicate();
    if (predicate == arith::CmpIPredicate::slt ||
        predicate == arith::CmpIPredicate::sle ||
        predicate == arith::CmpIPredicate::sgt ||
        predicate == arith::CmpIPredicate::sge)
      compare->setAttr("signed", builder->getUnitAttr());
    values[operation.getResult()] = compare.getFlag();
    return success();
  }

  LogicalResult lowerCompare(xw::CmpFOp operation) {
    FailureOr<ValueShape> resultShape =
        getShape(operation.getResult().getType(), operation);
    FailureOr<ValueShape> operandShape =
        getShape(operation.getLhs().getType(), operation);
    FailureOr<Value> lhs = getValue(operation.getLhs(), operation);
    FailureOr<Value> rhs = getValue(operation.getRhs(), operation);
    if (failed(resultShape) || failed(operandShape) || failed(lhs) || failed(rhs))
      return failure();
    std::optional<CondModifier> condition = mapPredicate(operation.getPredicate());
    if (!condition)
      return operation.emitOpError(
          "floating comparison predicate has no exact XeMachine selection");
    CmpOp compare = CmpOp::create(
        *builder, *location, ARFType::get(context, ARFFile::f, 2, -1),
        CondModifierAttr::get(context, *condition),
        typeAttr(operandShape->elementType),
        builder->getI32IntegerAttr(resultShape->cardinality),
        sourceRegion(operation.getLhs(), resultShape->cardinality, operation),
        sourceRegion(operation.getRhs(), resultShape->cardinality, operation),
        IntegerAttr(), IntegerAttr(), TypeAttr(), TypeAttr(), *lhs, *rhs);
    values[operation.getResult()] = compare.getFlag();
    return success();
  }

  LogicalResult lowerSelect(xw::SelectOp operation) {
    FailureOr<Value> condition = getValue(operation.getCondition(), operation);
    FailureOr<Value> trueValue = getValue(operation.getTrueValue(), operation);
    FailureOr<Value> falseValue = getValue(operation.getFalseValue(), operation);
    FailureOr<ValueShape> shape = getShape(operation.getType(), operation);
    FailureOr<int64_t> footprint = getFootprint(operation.getType(), operation);
    if (failed(condition) || failed(trueValue) || failed(falseValue) ||
        failed(shape) || failed(footprint))
      return failure();
    Type resultType = reg(*footprint);
    Operation *machineIf;
    if (isa<xw::MaskType>(operation.getCondition().getType()))
      machineIf = ExecIfOp::create(*builder, *location, TypeRange{resultType},
                                   *condition);
    else
      machineIf = UniformIfOp::create(*builder, *location,
                                      TypeRange{resultType}, *condition);
    std::array<Value, 2> arms = {*trueValue, *falseValue};
    for (unsigned index = 0; index < 2; ++index) {
      Region &region = index == 0 ? machineIf->getRegion(0)
                                  : machineIf->getRegion(1);
      builder->setInsertionPointToStart(&region.emplaceBlock());
      Value selected = emitMove(
          resultType, shape->elementType, shape->cardinality, arms[index],
          sourceRegion(index == 0 ? operation.getTrueValue()
                                  : operation.getFalseValue(),
                       shape->cardinality, operation));
      YieldOp::create(*builder, *location, ValueRange{selected});
    }
    builder->setInsertionPointAfter(machineIf);
    values[operation.getResult()] = machineIf->getResult(0);
    return success();
  }

  LogicalResult lowerWhere(xw::WhereOp operation) {
    FailureOr<Value> condition = getValue(operation.getCondition(), operation);
    if (failed(condition))
      return failure();
    SmallVector<Type> resultTypes;
    for (Value result : operation.getResults()) {
      if (isa<xw::MemTokenType>(result.getType()))
        resultTypes.push_back(MemTokenType::get(context));
      else {
        FailureOr<int64_t> footprint = getFootprint(result.getType(), operation);
        if (failed(footprint))
          return failure();
        resultTypes.push_back(reg(*footprint));
      }
    }
    ExecIfOp machineIf = ExecIfOp::create(*builder, *location, resultTypes,
                                          *condition);
    for (unsigned index = 0; index < operation.getNumResults(); ++index)
      values[operation.getResult(index)] = machineIf.getResult(index);
    if (failed(lowerBranchRegions(operation.getOperation(), machineIf,
                                  /*uniform=*/false)))
      return failure();
    return success();
  }

  LogicalResult lowerScfIf(scf::IfOp operation) {
    FailureOr<Value> condition = getValue(operation.getCondition(), operation);
    if (failed(condition))
      return failure();
    SmallVector<Type> resultTypes;
    for (Value result : operation.getResults()) {
      if (isa<xw::MemTokenType>(result.getType()))
        resultTypes.push_back(MemTokenType::get(context));
      else {
        FailureOr<int64_t> footprint = getFootprint(result.getType(), operation);
        if (failed(footprint))
          return failure();
        resultTypes.push_back(reg(*footprint));
      }
    }
    UniformIfOp machineIf = UniformIfOp::create(
        *builder, *location, resultTypes, *condition);
    for (unsigned index = 0; index < operation.getNumResults(); ++index)
      values[operation.getResult(index)] = machineIf.getResult(index);
    return lowerBranchRegions(operation.getOperation(), machineIf,
                              /*uniform=*/true);
  }

  LogicalResult lowerBranchRegions(Operation *sourceIf, Operation *machineIf,
                                   bool uniform) {
    Value entryToken = memoryToken;
    for (unsigned regionIndex = 0; regionIndex < sourceIf->getNumRegions();
         ++regionIndex) {
      Region &semanticRegion = sourceIf->getRegion(regionIndex);
      if (semanticRegion.empty())
        continue;
      Region &destinationRegion = machineIf->getRegion(regionIndex);
      builder->setInsertionPointToStart(&destinationRegion.emplaceBlock());
      memoryToken = entryToken;
      if (failed(lowerBlock(semanticRegion.front())))
        return failure();
      Operation *sourceYield = semanticRegion.front().getTerminator();
      SmallVector<Value> yielded;
      for (unsigned index = 0; index < sourceYield->getNumOperands(); ++index) {
        Value source = sourceYield->getOperand(index);
        if (isa<xw::MemTokenType>(source.getType())) {
          Value token = values.lookup(source);
          if (!token)
            return sourceYield->emitOpError("yielded token was not selected");
          yielded.push_back(token);
          memoryToken = token;
          continue;
        }
        FailureOr<Value> value = getValue(source, sourceYield);
        FailureOr<ValueShape> shape = getShape(source.getType(), sourceYield);
        if (failed(value) || failed(shape))
          return failure();
        yielded.push_back(emitMove(
            machineIf->getResult(index).getType(), shape->elementType,
            shape->cardinality, *value,
            sourceRegion(source, shape->cardinality, sourceYield)));
      }
      YieldOp::create(*builder, *location, yielded);
    }
    builder->setInsertionPointAfter(machineIf);
    for (unsigned index = 0; index < sourceIf->getNumResults(); ++index)
      if (isa<xw::MemTokenType>(sourceIf->getResult(index).getType()))
        memoryToken = machineIf->getResult(index);
    assert((uniform || isa<xw::WhereOp>(sourceIf)) &&
           "divergent regions must originate from xw.where");
    return success();
  }

  LogicalResult lowerFor(scf::ForOp operation) {
    if (operation.getLowerBound().getType() != operation.getUpperBound().getType() ||
        operation.getLowerBound().getType() != operation.getStep().getType())
      return operation.emitOpError("loop bounds and step must have one type");
    FailureOr<Value> lower = getValue(operation.getLowerBound(), operation);
    FailureOr<Value> upper = getValue(operation.getUpperBound(), operation);
    FailureOr<Value> step = getValue(operation.getStep(), operation);
    if (failed(lower) || failed(upper) || failed(step))
      return failure();
    SmallVector<Value> initial{*lower};
    SmallVector<Type> resultTypes{lower->getType()};
    for (Value init : operation.getInitArgs()) {
      FailureOr<Value> selected = getValue(init, operation);
      if (failed(selected))
        return failure();
      initial.push_back(*selected);
      resultTypes.push_back(selected->getType());
    }
    UniformLoopOp loop = UniformLoopOp::create(*builder, *location, resultTypes,
                                               initial);
    Block &body = loop.getBody().emplaceBlock();
    for (Type type : resultTypes)
      body.addArgument(type, operation.getLoc());
    values[operation.getInductionVar()] = body.getArgument(0);
    for (unsigned index = 0; index < operation.getNumRegionIterArgs(); ++index)
      values[operation.getRegionIterArg(index)] = body.getArgument(index + 1);
    builder->setInsertionPointToStart(&body);
    if (failed(lowerBlock(operation.getRegion().front())))
      return failure();
    scf::YieldOp sourceYield =
        cast<scf::YieldOp>(operation.getRegion().front().getTerminator());
    Type inductionType = operation.getLowerBound().getType();
    if (!inductionType.isIntOrIndex())
      return operation.emitOpError("loop induction must be integer or index");
    Value next =
        AddOp::create(*builder, *location, lower->getType(), inductionType, 1,
                      canonicalDestination(), canonicalRegion(),
                      step->getDefiningOp<ImmOp>() ? RegionAttr()
                                                  : uniformRegion(),
                      IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                      TypeAttr(), true, 0, body.getArgument(0), *step)
            .getResult();
    CmpOp condition = CmpOp::create(
        *builder, *location, ARFType::get(context, ARFFile::f, 2, -1),
        CondModifierAttr::get(context, CondModifier::lt),
        typeAttr(inductionType),
        builder->getI32IntegerAttr(1), canonicalRegion(), uniformRegion(),
        IntegerAttr(), IntegerAttr(), TypeAttr(), TypeAttr(), next, *upper);
    condition->setAttr("signed", builder->getUnitAttr());
    SmallVector<Value> carried{next};
    for (Value source : sourceYield.getOperands()) {
      FailureOr<Value> selected = getValue(source, sourceYield);
      if (failed(selected))
        return failure();
      carried.push_back(*selected);
    }
    ContinueIfOp::create(*builder, *location, condition.getFlag(), carried);
    builder->setInsertionPointAfter(loop);
    for (unsigned index = 0; index < operation.getNumResults(); ++index)
      values[operation.getResult(index)] = loop.getResult(index + 1);
    return success();
  }

  LogicalResult lowerWhile(scf::WhileOp operation) {
    SmallVector<Value> initial;
    SmallVector<Type> resultTypes;
    for (Value operand : operation.getInits()) {
      FailureOr<Value> selected = getValue(operand, operation);
      if (failed(selected))
        return failure();
      initial.push_back(*selected);
      resultTypes.push_back(selected->getType());
    }
    UniformLoopOp loop = UniformLoopOp::create(*builder, *location, resultTypes,
                                               initial);
    Block &body = loop.getBody().emplaceBlock();
    for (Type type : resultTypes)
      body.addArgument(type, operation.getLoc());
    Block &before = operation.getBefore().front();
    for (unsigned index = 0; index < before.getNumArguments(); ++index)
      values[before.getArgument(index)] = body.getArgument(index);
    builder->setInsertionPointToStart(&body);
    if (failed(lowerBlock(before)))
      return failure();
    scf::ConditionOp condition =
        cast<scf::ConditionOp>(before.getTerminator());
    FailureOr<Value> selectedCondition =
        getValue(condition.getCondition(), condition);
    if (failed(selectedCondition))
      return failure();

    Block &after = operation.getAfter().front();
    if (after.getNumArguments() != condition.getArgs().size())
      return operation.emitOpError("while region argument count mismatch");
    SmallVector<Value> conditionArguments;
    SmallVector<Type> conditionTypes;
    for (Value argument : condition.getArgs()) {
      FailureOr<Value> selected = getValue(argument, condition);
      if (failed(selected))
        return failure();
      conditionArguments.push_back(*selected);
      conditionTypes.push_back(selected->getType());
    }
    UniformIfOp executeBody = UniformIfOp::create(
        *builder, *location, conditionTypes, *selectedCondition);
    builder->setInsertionPointToStart(
        &executeBody.getThenRegion().emplaceBlock());
    for (unsigned index = 0; index < after.getNumArguments(); ++index)
      values[after.getArgument(index)] = conditionArguments[index];
    if (failed(lowerBlock(after)))
      return failure();
    scf::YieldOp sourceYield = cast<scf::YieldOp>(after.getTerminator());
    SmallVector<Value> thenValues;
    for (Value operand : sourceYield.getOperands()) {
      FailureOr<Value> selected = getValue(operand, sourceYield);
      if (failed(selected))
        return failure();
      thenValues.push_back(*selected);
    }
    YieldOp::create(*builder, *location, thenValues);
    builder->setInsertionPointToStart(
        &executeBody.getElseRegion().emplaceBlock());
    YieldOp::create(*builder, *location, conditionArguments);
    builder->setInsertionPointAfter(executeBody);
    ContinueIfOp::create(*builder, *location, *selectedCondition,
                         executeBody.getResults());
    builder->setInsertionPointAfter(loop);
    for (unsigned index = 0; index < operation.getNumResults(); ++index)
      values[operation.getResult(index)] = loop.getResult(index);
    return success();
  }

  LogicalResult lowerPack(xw::PackOp operation) {
    SmallVector<Value> elements;
    SmallVector<Type> elementTypes;
    for (Value input : operation.getInputs()) {
      FailureOr<Value> selected = materialize(input, operation);
      if (failed(selected))
        return failure();
      elements.push_back(*selected);
      elementTypes.push_back(selected->getType());
    }
    FailureOr<int64_t> footprint = getFootprint(operation.getType(), operation);
    if (failed(footprint))
      return failure();
    values[operation.getResult()] =
        TupleFromElementsOp::create(*builder, *location, reg(*footprint),
                                    elements)
            .getTuple();
    return success();
  }

  LogicalResult lowerExtract(xw::ExtractOp operation) {
    FailureOr<Value> source = materialize(operation.getSource(), operation);
    if (failed(source))
      return failure();
    Type sourceType = operation.getSource().getType();
    Type packetType = sourceType;
    int64_t cardinality = 1;
    if (xw::SimdType simd = dyn_cast<xw::SimdType>(sourceType)) {
      packetType = simd.getElementType();
      cardinality = simd.getCardinality();
    }
    VectorType vector = dyn_cast<VectorType>(packetType);
    if (!vector || vector.getRank() != 1)
      return operation.emitOpError("machine extract requires a rank-one vector");
    FailureOr<int64_t> elementBits =
        getElementBits(vector.getElementType(), operation);
    if (failed(elementBits))
      return failure();
    int64_t elementDwords = (*elementBits * cardinality + 31) / 32;
    SmallVector<Type> parts(vector.getNumElements(), reg(elementDwords));
    TupleToElementsOp split = TupleToElementsOp::create(
        *builder, *location, parts, *source);
    values[operation.getResult()] = split.getResult(operation.getIndex());
    return success();
  }

  LogicalResult lowerPointerAdd(xw::PtrAddOp operation) {
    FailureOr<ValueShape> shape = getShape(operation.getType(), operation);
    if (failed(shape))
      return failure();
    bool local = isLocalPointer(operation.getType());
    if (!local && shape->cardinality == 32) {
      FailureOr<WideValue> base =
          materializeWidePointer(operation.getBase(), operation, 32);
      FailureOr<WideValue> offset =
          materializeWideInteger(operation.getOffset(), operation);
      if (failed(base) || failed(offset))
        return failure();
      auto add = [&](Value lhs, Value rhs, int64_t maskOffset) {
        return AddOp::create(
                   *builder, *location, reg(32), i64(), 16,
                   canonicalDestination(),
                   lhs.getDefiningOp<ImmOp>() ? RegionAttr()
                                               : canonicalRegion(),
                   rhs.getDefiningOp<ImmOp>() ? RegionAttr()
                                               : canonicalRegion(),
                   IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                   TypeAttr(), false, maskOffset, lhs, rhs)
            .getResult();
      };
      WideValue result{add(base->low, offset->low, 0),
                       add(base->high, offset->high, 16)};
      widePointers[operation.getResult()] = result;
      wideValues[operation.getResult()] = result;
      return success();
    }
    FailureOr<Value> base = getValue(operation.getBase(), operation);
    FailureOr<Value> offset = getValue(operation.getOffset(), operation);
    if (failed(base) || failed(offset))
      return failure();
    Type addressType = local ? i32() : i64();
    int64_t addressBits = local ? 32 : 64;
    int64_t footprint =
        (shape->cardinality * addressBits + 31) / 32;
    Value lhs = *base;
    Value rhs = *offset;
    Value lhsSource = operation.getBase();
    Value rhsSource = operation.getOffset();
    if (lhs.getDefiningOp<ImmOp>() && !rhs.getDefiningOp<ImmOp>()) {
      std::swap(lhs, rhs);
      std::swap(lhsSource, rhsSource);
    }
    Value result =
        AddOp::create(
            *builder, *location, reg(footprint), addressType,
            shape->cardinality,
            canonicalDestination(),
            sourceRegion(lhsSource, shape->cardinality, operation),
            sourceRegion(rhsSource, shape->cardinality, operation),
            IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(), TypeAttr(),
            shape->cardinality == 1, 0, lhs, rhs)
            .getResult();
    values[operation.getResult()] = result;
    return success();
  }

  FailureOr<WideValue> materializeWideInteger(Value source, Operation *owner) {
    if (isWideSimd(source.getType()))
      return getWideValue(source, owner);
    FailureOr<Value> value = getValue(source, owner);
    FailureOr<ValueShape> shape = getShape(source.getType(), owner);
    if (failed(value) || failed(shape))
      return failure();
    auto broadcast = [&](int64_t offset) {
      if (value->getDefiningOp<ImmOp>())
        return *value;
      return emitMove(reg(32), i64(), 16, *value,
                      shape->cardinality == 1 ? uniformRegion()
                                              : canonicalRegion(),
                      false, offset);
    };
    return WideValue{broadcast(0), broadcast(16)};
  }

  FailureOr<WideValue> materializeWidePointer(Value pointer, Operation *owner,
                                              int64_t cardinality) {
    auto found = widePointers.find(pointer);
    if (found != widePointers.end())
      return found->second;
    FailureOr<Value> scalar = getValue(pointer, owner);
    if (failed(scalar))
      return failure();
    FailureOr<ValueShape> shape = getShape(pointer.getType(), owner);
    if (failed(shape))
      return failure();
    auto broadcast = [&](int64_t offset) {
      return emitMove(reg(32), i64(), 16, *scalar,
                      shape->cardinality == 1 ? uniformRegion()
                                              : canonicalRegion(),
                      false, offset);
    };
    WideValue result{broadcast(0), broadcast(16)};
    widePointers[pointer] = result;
    (void)cardinality;
    return result;
  }

  xw::PtrType getPointerElementType(Type type) const {
    if (xw::SimdType simd = dyn_cast<xw::SimdType>(type))
      return dyn_cast<xw::PtrType>(simd.getElementType());
    return dyn_cast<xw::PtrType>(type);
  }

  bool isLocalPointer(Type type) const {
    xw::PtrType pointer = getPointerElementType(type);
    return pointer && isa<xw::LocalAddressSpaceAttr>(pointer.getAddressSpace());
  }

  bool isA64Pointer(Type type) const {
    xw::PtrType pointer = getPointerElementType(type);
    return pointer &&
           isa<xw::GlobalAddressSpaceAttr, xw::ConstantAddressSpaceAttr,
               xw::GenericAddressSpaceAttr>(pointer.getAddressSpace());
  }

  FailureOr<Value> getA64Payload(Value pointer, int64_t cardinality,
                                 Operation *owner) {
    if (cardinality == 32) {
      FailureOr<WideValue> address =
          materializeWidePointer(pointer, owner, cardinality);
      if (failed(address))
        return failure();
      return TupleFromElementsOp::create(
                 *builder, *location, reg(64),
                 ValueRange{address->low, address->high})
          .getTuple();
    }
    FailureOr<Value> address = getValue(pointer, owner);
    FailureOr<ValueShape> pointerShape = getShape(pointer.getType(), owner);
    if (failed(address) || failed(pointerShape))
      return failure();
    int64_t footprint = cardinality * 2;
    if (pointerShape->cardinality == cardinality &&
        !address->getDefiningOp<ImmOp>())
      return *address;
    return emitMove(reg(footprint), i64(), cardinality, *address,
                    pointerShape->cardinality == 1 ? uniformRegion()
                                                    : canonicalRegion());
  }

  FailureOr<Value> getLocalPayload(Value pointer, int64_t cardinality,
                                   Operation *owner) {
    FailureOr<Value> address = getValue(pointer, owner);
    FailureOr<ValueShape> pointerShape = getShape(pointer.getType(), owner);
    if (failed(address) || failed(pointerShape))
      return failure();
    if (pointerShape->cardinality == cardinality &&
        !address->getDefiningOp<ImmOp>())
      return *address;
    return emitMove(reg(cardinality), i32(), cardinality, *address,
                    pointerShape->cardinality == 1 ? uniformRegion()
                                                    : canonicalRegion());
  }

  FailureOr<Value> mapDependency(Operation *operation, Value dependency) {
    if (!dependency)
      return Value();
    Value selected = values.lookup(dependency);
    if (!selected)
      return operation->emitOpError("memory dependency was not selected"),
             failure();
    return selected;
  }

  LogicalResult lowerLoad(xw::LoadOp operation) {
    FailureOr<ValueShape> shape = getShape(operation.getValue().getType(), operation);
    if (failed(shape))
      return failure();
    FailureOr<int64_t> bits = getElementBits(shape->elementType, operation);
    FailureOr<Value> dependency =
        mapDependency(operation, operation.getDependency());
    if (failed(shape) || failed(bits) || failed(dependency))
      return failure();
    if (*bits != 32)
      return operation.emitOpError("only dword Xe memory loads are supported");
    if (isLocalPointer(operation.getPtr().getType())) {
      FailureOr<Value> address =
          getLocalPayload(operation.getPtr(), shape->cardinality, operation);
      if (failed(address))
        return failure();
      LoadSLMOp load = LoadSLMOp::create(
          *builder, *location, reg(shape->cardinality),
          MemTokenType::get(context), *address, *dependency,
          shape->cardinality);
      values[operation.getValue()] = load.getDst();
      values[operation.getToken()] = load.getToken();
      memoryToken = load.getToken();
      return success();
    }
    if (!isA64Pointer(operation.getPtr().getType()))
      return operation.emitOpError("unsupported XW load address space");
    FailureOr<Value> address =
        getA64Payload(operation.getPtr(), shape->cardinality, operation);
    if (failed(address))
      return failure();
    LoadA64Op load = LoadA64Op::create(
        *builder, *location, reg(shape->cardinality), MemTokenType::get(context),
        *address, *dependency, shape->cardinality);
    values[operation.getValue()] = load.getDst();
    values[operation.getToken()] = load.getToken();
    memoryToken = load.getToken();
    return success();
  }

  LogicalResult lowerStore(xw::StoreOp operation) {
    FailureOr<ValueShape> shape = getShape(operation.getValue().getType(), operation);
    if (failed(shape))
      return failure();
    FailureOr<int64_t> bits = getElementBits(shape->elementType, operation);
    FailureOr<Value> data = materialize(operation.getValue(), operation);
    FailureOr<Value> dependency =
        mapDependency(operation, operation.getDependency());
    if (failed(shape) || failed(bits) || failed(data) || failed(dependency))
      return failure();
    if (*bits != 32)
      return operation.emitOpError("only dword Xe memory stores are supported");
    if (isLocalPointer(operation.getPtr().getType())) {
      FailureOr<Value> address =
          getLocalPayload(operation.getPtr(), shape->cardinality, operation);
      if (failed(address))
        return failure();
      StoreSLMOp store = StoreSLMOp::create(
          *builder, *location, MemTokenType::get(context), *address, *data,
          *dependency, shape->cardinality);
      values[operation.getToken()] = store.getToken();
      memoryToken = store.getToken();
      return success();
    }
    if (!isA64Pointer(operation.getPtr().getType()))
      return operation.emitOpError("unsupported XW store address space");
    FailureOr<Value> address =
        getA64Payload(operation.getPtr(), shape->cardinality, operation);
    if (failed(address))
      return failure();
    StoreA64Op store = StoreA64Op::create(
        *builder, *location, MemTokenType::get(context), *address, *data,
        *dependency, shape->cardinality);
    values[operation.getToken()] = store.getToken();
    memoryToken = store.getToken();
    return success();
  }

  LogicalResult lowerAtomic(xw::AtomicRMWOp operation) {
    if (operation.getKind() != arith::AtomicRMWKind::addi)
      return operation.emitOpError("only atomic add has XeMachine support");
    FailureOr<ValueShape> shape = getShape(operation.getOld().getType(), operation);
    FailureOr<Value> data = materialize(operation.getValue(), operation);
    FailureOr<Value> dependency =
        mapDependency(operation, operation.getDependency());
    if (failed(shape) || failed(data) || failed(dependency))
      return failure();
    if (!shape->elementType.isInteger(32) ||
        !isA64Pointer(operation.getPtr().getType()))
      return operation.emitOpError(
          "atomic add requires i32 data and an A64 address");
    FailureOr<Value> address =
        getA64Payload(operation.getPtr(), shape->cardinality, operation);
    if (failed(address))
      return failure();
    AtomicIAddA64Op atomic = AtomicIAddA64Op::create(
        *builder, *location, reg(shape->cardinality),
        MemTokenType::get(context), *address, *data, *dependency,
        shape->cardinality);
    values[operation.getOld()] = atomic.getDst();
    values[operation.getToken()] = atomic.getToken();
    memoryToken = atomic.getToken();
    return success();
  }

  void emitBarrier(Value dependency) {
    FenceSLMOp fence = FenceSLMOp::create(
        *builder, *location, reg(16), MemTokenType::get(context),
        architecturalRegister(0), dependency);
    FenceAwaitOp await = FenceAwaitOp::create(
        *builder, *location, MemTokenType::get(context), fence.getReadback(),
        fence.getToken());
    Value payload = emitMove(reg(16), i32(), 16, immediate(0, i32()),
                             RegionAttr(), true);
    Value control =
        MovOp::create(*builder, *location, reg(16), i32(), 1,
                      canonicalDestination(), RegionAttr(),
                      builder->getI32IntegerAttr(2), IntegerAttr(), TypeAttr(),
                      true, 0, immediate(0x100, i32()))
            .getResult();
    payload = UpdateTupleOp::create(
                  *builder, *location, reg(16), payload, ValueRange{control},
                  builder->getArrayAttr({builder->getI64IntegerAttr(0)}))
                  .getResult();
    Value header = MovOp::create(
                       *builder, *location, reg(16), i8(), 2,
                       canonicalDestination(), uniformRegion(),
                       builder->getI32IntegerAttr(10),
                       builder->getI32IntegerAttr(11), TypeAttr(), true, 0,
                       architecturalRegister(0))
                       .getResult();
    payload = UpdateTupleOp::create(
                  *builder, *location, reg(16), payload, ValueRange{header},
                  builder->getArrayAttr({builder->getI64IntegerAttr(0)}))
                  .getResult();
    BarrierSignalOp signal = BarrierSignalOp::create(
        *builder, *location, MemTokenType::get(context), payload,
        await.getToken());
    memoryToken = signal.getToken();
    emitSync(SyncKind::bar);
  }

  LogicalResult lowerId(Operation *operation, int64_t dim, bool global) {
    if (dim < 0 || dim >= 3)
      return operation->emitOpError("ID axis must be 0, 1, or 2");
    if (failed(emitPrologue()))
      return failure();
    FailureOr<ValueShape> shape =
        getShape(operation->getResult(0).getType(), operation);
    if (failed(shape) || shape->cardinality != simdWidth)
      return operation->emitOpError(
          "work-item IDs require the ambient SIMD cardinality");
    Type elementType = shape->elementType;
    if (!elementType.isIntOrIndex())
      return operation->emitOpError("work-item ID must have integer payload");
    Value local = localIds[dim];
    Value result;
    if (!global) {
      result = emitMove(reg(simdWidth), i32(), simdWidth, local,
                        canonicalRegion(), false, 0, IntegerAttr());
    } else {
      Value r0 = architecturalRegister(0);
      Value inlineData = architecturalRegister(4);
      Value accumulator =
          MulOp::create(
              *builder, *location,
              ARFType::get(context, ARFFile::acc, 16, 0), i32(), 1,
              canonicalDestination(), uniformRegion(), uniformRegion(),
              IntegerAttr(), builder->getI32IntegerAttr(1 + dim),
              builder->getI32IntegerAttr(3 + dim), true, 0, r0, inlineData)
              .getResult();
      Value base = emitMove(reg(16), i32(), 1, accumulator, uniformRegion(),
                            true);
      result = Add3Op::create(
                   *builder, *location, reg(simdWidth), i32(), simdWidth,
                   canonicalDestination(), uniformRegion(), canonicalRegion(),
                   uniformRegion(), IntegerAttr(), IntegerAttr(), IntegerAttr(),
                   builder->getI32IntegerAttr(dim), TypeAttr(), typeAttr(i16()),
                   TypeAttr(), false, 0, base, local, inlineData)
                   .getResult();
    }
    if (elementType.isInteger(32) || isa<IndexType>(elementType)) {
      values[operation->getResult(0)] = result;
      return success();
    }
    if (!elementType.isInteger(64))
      return operation->emitOpError("unsupported work-item ID result type");
    if (simdWidth < 32) {
      values[operation->getResult(0)] =
          MovOp::create(*builder, *location, reg(simdWidth * 2), i64(),
                        simdWidth, canonicalDestination(), canonicalRegion(),
                        IntegerAttr(), IntegerAttr(), typeAttr(i32()), false, 0,
                        result)
              .getResult();
      return success();
    }
    auto widen = [&](int64_t offset) {
      return MovOp::create(
                 *builder, *location, reg(32), i64(), 16,
                 canonicalDestination(), canonicalRegion(), IntegerAttr(),
                 builder->getI32IntegerAttr(offset), typeAttr(i32()), false,
                 offset, result)
          .getResult();
    };
    wideValues[operation->getResult(0)] = {widen(0), widen(16)};
    return success();
  }

  LogicalResult lowerUniformQuery(Operation *operation, int64_t dim,
                                  int sourceSub) {
    if (dim < 0 || dim >= 3)
      return operation->emitOpError("query axis must be 0, 1, or 2");
    if (failed(emitPrologue()))
      return failure();
    FailureOr<ValueShape> shape =
        getShape(operation->getResult(0).getType(), operation);
    FailureOr<int64_t> footprint =
        getFootprint(operation->getResult(0).getType(), operation);
    if (failed(shape) || failed(footprint) || shape->cardinality != 1)
      return operation->emitOpError("uniform query must return a bare value");
    Value source = sourceSub < 3 ? architecturalRegister(0)
                                 : architecturalRegister(4);
    int sub = sourceSub < 3 ? sourceSub : sourceSub - 3;
    values[operation->getResult(0)] =
        MovOp::create(*builder, *location, reg(*footprint), shape->elementType,
                      1, canonicalDestination(), uniformRegion(), IntegerAttr(),
                      builder->getI32IntegerAttr(sub), typeAttr(i32()), true, 0,
                      source)
            .getResult();
    return success();
  }

  void emitEot() {
    Value payload = emitMove(reg(16), i32(), 16, architecturalRegister(0),
                             canonicalRegion(), true);
    EotOp::create(*builder, *location, payload, memoryToken);
  }

  LogicalResult lowerBlock(Block &block) {
    for (Operation &operation : block) {
      if (xw::ConstantOp constant = dyn_cast<xw::ConstantOp>(operation)) {
        if (isWideSimd(constant.getResult().getType())) {
          FailureOr<int64_t> bits = getConstantBits(constant);
          if (failed(bits))
            return failure();
          Value result = immediate(*bits, i64());
          wideValues[constant.getResult()] = {result, result};
        } else if (failed(getValue(constant.getResult(), constant))) {
          return failure();
        }
      } else if (xw::SplatOp splat = dyn_cast<xw::SplatOp>(operation)) {
        if (failed(lowerView(splat, splat.getSource())))
          return failure();
      } else if (xw::ReadFirstOp read = dyn_cast<xw::ReadFirstOp>(operation)) {
        if (failed(lowerView(read, read.getSource())))
          return failure();
      } else if (xw::ExpandOp expand = dyn_cast<xw::ExpandOp>(operation)) {
        if (failed(lowerView(expand, expand.getSource())))
          return failure();
      } else if (xw::BinaryOp binary = dyn_cast<xw::BinaryOp>(operation)) {
        if (failed(lowerBinary(binary)))
          return failure();
      } else if (xw::CastOp cast = dyn_cast<xw::CastOp>(operation)) {
        if (failed(lowerCast(cast)))
          return failure();
      } else if (isa<xw::FAddOp>(operation)) {
        if (failed(lowerFloatBinary(&operation, false)))
          return failure();
      } else if (isa<xw::FSubOp>(operation)) {
        if (failed(lowerFloatBinary(&operation, true)))
          return failure();
      } else if (xw::FMulOp multiply = dyn_cast<xw::FMulOp>(operation)) {
        if (failed(lowerFloatMultiply(multiply)))
          return failure();
      } else if (xw::CmpIOp compare = dyn_cast<xw::CmpIOp>(operation)) {
        if (failed(lowerCompare(compare)))
          return failure();
      } else if (xw::CmpFOp compare = dyn_cast<xw::CmpFOp>(operation)) {
        if (failed(lowerCompare(compare)))
          return failure();
      } else if (xw::SelectOp select = dyn_cast<xw::SelectOp>(operation)) {
        if (failed(lowerSelect(select)))
          return failure();
      } else if (xw::WhereOp where = dyn_cast<xw::WhereOp>(operation)) {
        if (failed(lowerWhere(where)))
          return failure();
      } else if (isa<xw::YieldOp, scf::YieldOp, scf::ConditionOp>(operation)) {
        continue;
      } else if (scf::IfOp ifOperation = dyn_cast<scf::IfOp>(operation)) {
        if (failed(lowerScfIf(ifOperation)))
          return failure();
      } else if (scf::ForOp forOperation = dyn_cast<scf::ForOp>(operation)) {
        if (failed(lowerFor(forOperation)))
          return failure();
      } else if (scf::WhileOp whileOperation =
                     dyn_cast<scf::WhileOp>(operation)) {
        if (failed(lowerWhile(whileOperation)))
          return failure();
      } else if (xw::PackOp pack = dyn_cast<xw::PackOp>(operation)) {
        if (failed(lowerPack(pack)))
          return failure();
      } else if (xw::ExtractOp extract = dyn_cast<xw::ExtractOp>(operation)) {
        if (failed(lowerExtract(extract)))
          return failure();
      } else if (xw::PtrAddOp pointer = dyn_cast<xw::PtrAddOp>(operation)) {
        if (failed(lowerPointerAdd(pointer)))
          return failure();
      } else if (xw::NullOp null = dyn_cast<xw::NullOp>(operation)) {
        values[null.getResult()] = immediate(0, i64());
      } else if (xw::LocalMemoryBaseOp local =
                     dyn_cast<xw::LocalMemoryBaseOp>(operation)) {
        values[local.getResult()] = immediate(local.getOffset(), i32());
      } else if (xw::AllocOp allocation = dyn_cast<xw::AllocOp>(operation)) {
        values[allocation.getResult()] =
            immediate(allocation.getOffset().value_or(0), i32());
      } else if (xw::TokenOp token = dyn_cast<xw::TokenOp>(operation)) {
        memoryToken =
            TokenOp::create(*builder, *location, MemTokenType::get(context))
                .getToken();
        values[token.getResult()] = memoryToken;
      } else if (isa<xw::IssueTokenOp, xw::AfterOp, xw::JoinOp>(operation)) {
        SmallVector<Value> dependencies;
        for (Value source : operation.getOperands()) {
          FailureOr<Value> selected = mapDependency(&operation, source);
          if (failed(selected))
            return failure();
          dependencies.push_back(*selected);
        }
        if (isa<xw::IssueTokenOp, xw::AfterOp>(operation))
          memoryToken = AfterOp::create(*builder, *location,
                                        MemTokenType::get(context), dependencies)
                            .getToken();
        else
          memoryToken = TokenJoinOp::create(
                            *builder, *location, MemTokenType::get(context),
                            dependencies)
                            .getToken();
        values[operation.getResult(0)] = memoryToken;
      } else if (xw::LoadOp load = dyn_cast<xw::LoadOp>(operation)) {
        if (failed(lowerLoad(load)))
          return failure();
      } else if (xw::StoreOp store = dyn_cast<xw::StoreOp>(operation)) {
        if (failed(lowerStore(store)))
          return failure();
      } else if (xw::AtomicRMWOp atomic =
                     dyn_cast<xw::AtomicRMWOp>(operation)) {
        if (failed(lowerAtomic(atomic)))
          return failure();
      } else if (xw::BarrierOp barrier = dyn_cast<xw::BarrierOp>(operation)) {
        SmallVector<Value> dependencies;
        for (Value source : barrier.getDependencies()) {
          FailureOr<Value> selected = mapDependency(barrier, source);
          if (failed(selected))
            return failure();
          dependencies.push_back(*selected);
        }
    Value dependency;
        if (dependencies.size() == 1)
          dependency = dependencies.front();
        else if (!dependencies.empty())
          dependency = TokenJoinOp::create(
                           *builder, *location, MemTokenType::get(context),
                           dependencies)
                           .getToken();
        if (!dependency)
          dependency = TokenOp::create(*builder, *location,
                                       MemTokenType::get(context))
                           .getToken();
        emitBarrier(dependency);
        values[barrier.getToken()] = memoryToken;
      } else if (xw::GlobalIdOp id = dyn_cast<xw::GlobalIdOp>(operation)) {
        if (failed(lowerId(id, id.getDim(), true)))
          return failure();
      } else if (xw::LocalIdOp id = dyn_cast<xw::LocalIdOp>(operation)) {
        if (failed(lowerId(id, id.getDim(), false)))
          return failure();
      } else if (xw::GroupIdOp id = dyn_cast<xw::GroupIdOp>(operation)) {
        if (failed(lowerUniformQuery(id, id.getDim(), 1 + id.getDim())))
          return failure();
      } else if (xw::LocalSizeOp size = dyn_cast<xw::LocalSizeOp>(operation)) {
        if (failed(lowerUniformQuery(size, size.getDim(), 3 + size.getDim())))
          return failure();
      } else if (isa<func::ReturnOp>(operation)) {
        emitEot();
      } else if (operation.getName().getDialectNamespace() == "xw") {
        return operation.emitOpError(
            "unsupported semantic XW operation during XeMachine selection");
      } else {
        return operation.emitOpError(
            "selector accepts only func, scf, and XW operations");
      }
    }
    return success();
  }
};

} // namespace
