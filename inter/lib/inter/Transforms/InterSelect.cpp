// Select closed XW semantic IR to XeMachine operations.

#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <array>
#include <optional>

namespace inter {
#define GEN_PASS_DEF_SELECTTOMACHINE
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;
using namespace inter::xemachine;

namespace {

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
  using SelectToMachineBase::SelectToMachineBase;

  void runOnOperation() override {
    SmallVector<StringRef> featureNames;
    for (const std::string &feature : targetFeatures)
      featureNames.push_back(feature);
    TargetAttr requestedTarget =
        getOperation()->getAttrOfType<TargetAttr>(kCompilationTargetAttrName);
    llvm::Expected<TargetConfig> resolvedTarget =
        requestedTarget ? TargetConfig::resolve(requestedTarget)
                        : TargetConfig::resolve(chip, featureNames);
    if (!resolvedTarget) {
      getOperation().emitError(llvm::toString(resolvedTarget.takeError()));
      return signalPassFailure();
    }
    target.emplace(*resolvedTarget);

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
    getOperation()->removeAttr(kCompilationTargetAttrName);
    getOperation()->removeAttr(kCompilationSimdWidthAttrName);
  }

private:
  std::optional<TargetConfig> target;
  MLIRContext *context = nullptr;
  std::optional<Location> location;
  std::optional<OpBuilder> builder;
  DenseMap<Value, Value> values;
  DenseMap<Value, WideValue> wideValues;
  DenseMap<Value, Value> localPointers;
  DenseMap<Value, WideValue> widePointers;
  ArrayAttr kernelArguments;
  Value memoryToken;
  SmallVector<Value> payloadTail;
  std::array<Value, 3> localIds;
  std::array<bool, 3> usedIdAxes{};
  std::array<bool, 3> subgroupIdAxes{};
  int64_t simdWidth = 0;
  bool prologueEmitted = false;

  Type i8() const { return IntegerType::get(context, 8); }
  Type i1() const { return IntegerType::get(context, 1); }
  Type i16() const { return IntegerType::get(context, 16); }
  Type i32() const { return IntegerType::get(context, 32); }
  Type i64() const { return IntegerType::get(context, 64); }
  Type reg(int64_t dwords) const { return RegType::get(context, dwords, -1); }
  TypeAttr typeAttr(Type type) const { return TypeAttr::get(type); }
  RegionAttr canonicalRegion() const {
    return RegionAttr::get(context, 1, 1, 0);
  }
  RegionAttr uniformRegion() const { return RegionAttr::get(context, 0, 1, 0); }
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
    if (xw::PtrType pointer = dyn_cast<xw::PtrType>(type)) {
      std::optional<KernelAddressSpace> addressSpace =
          getAddressSpace(pointer.getAddressSpace());
      if (!addressSpace)
        return owner->emitOpError("unsupported pointer address space"),
               failure();
      return KernelABI::get().getMachinePointerBitWidth(*addressSpace);
    }
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

  FailureOr<Value> materializeVectorSplat(int64_t bits, VectorType vector,
                                          int64_t cardinality,
                                          Operation *owner) {
    FailureOr<int64_t> footprint =
        getFootprint(xw::SimdType::get(context, vector, cardinality), owner);
    FailureOr<int64_t> scalarBits =
        getElementBits(vector.getElementType(), owner);
    if (failed(footprint) || failed(scalarBits))
      return failure();
    if (*footprint % 16 != 0 || *scalarBits == 0 || 512 % *scalarBits != 0 ||
        512 / *scalarBits > 32)
      return owner->emitOpError("vector splat has no whole-GRF packet form"),
             failure();
    Value scalar = immediate(bits, vector.getElementType());
    SmallVector<Value> pieces;
    for (int64_t offset = 0; offset < *footprint; offset += 16)
      pieces.push_back(emitMove(reg(16), vector.getElementType(),
                                512 / *scalarBits, scalar, RegionAttr(),
                                false));
    return TupleFromElementsOp::create(*builder, *location, reg(*footprint),
                                       pieces)
        .getTuple();
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
    if (xw::PtrType pointer = dyn_cast<xw::PtrType>(elementType))
      return !isa<xw::LocalAddressSpaceAttr>(pointer.getAddressSpace());
    return isa<IndexType>(elementType);
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
    return RegionAttr::get(context, 1, executionSize / shape->cardinality, 0);
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
    SyncOp operation =
        SyncOp::create(*builder, *location, MemTokenType::get(context),
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
      IntegerAttr alignment = descriptor.getAs<IntegerAttr>("alignment");
      if (!kind || !size || !offset || !alignment)
        return kernel.emitOpError("incomplete XW kernel argument descriptor"),
               failure();
      bool pointer = kind.getValue() == "pointer";
      StringRef addressSpace = "none";
      if (IntegerAttr space = descriptor.getAs<IntegerAttr>("address_space")) {
        std::optional<KernelAddressSpace> decoded =
            KernelABI::get().decodeAddressSpace(space.getInt());
        addressSpace = decoded ? KernelABI::get().getAddressSpaceName(*decoded)
                               : "unknown";
      }
      StringAttr access = descriptor.getAs<StringAttr>("access");
      if (pointer && !access)
        return kernel.emitOpError(
                   "pointer XW kernel argument is missing access metadata"),
               failure();
      machineDescriptors.push_back(KernelArgAttr::get(
          kernel.getContext(),
          pointer ? KernelArgKind::by_pointer : KernelArgKind::by_value,
          attrBuilder.getStringAttr(addressSpace),
          pointer ? access : attrBuilder.getStringAttr("none"), size.getInt(),
          alignment.getInt(), offset.getInt()));
    }
    return attrBuilder.getArrayAttr(machineDescriptors);
  }

  FailureOr<std::pair<Value, int64_t>>
  getPayloadLocation(BlockArgument argument, Operation *owner) {
    FailureOr<KernelArgAttr> descriptor = getKernelArgument(argument, owner);
    if (failed(descriptor))
      return failure();
    uint64_t offset = descriptor->getOffset();
    FailureOr<int64_t> bits = getElementBits(argument.getType(), owner);
    if (failed(bits))
      return failure();
    uint64_t storageBytes = (*bits + 7) / 8;
    const KernelABI &abi = KernelABI::get();
    if (offset < abi.getInlinePayloadSize())
      return std::pair<Value, int64_t>{getInlineDataRegister(),
                                       offset / storageBytes};
    uint64_t tailOffset = offset - abi.getInlinePayloadSize();
    uint64_t chunk = tailOffset / abi.getPayloadChunkSize();
    if (chunk >= payloadTail.size())
      return owner->emitOpError("kernel argument at offset ")
                 << offset << " is outside loaded payload with "
                 << payloadTail.size() << " chunks",
             failure();
    return std::pair<Value, int64_t>{payloadTail[chunk],
                                     tailOffset % abi.getPayloadChunkSize() /
                                         storageBytes};
  }

  LogicalResult validateKernelArguments(func::FuncOp kernel) {
    if (!kernelArguments || kernelArguments.size() != kernel.getNumArguments())
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
            descriptor->getSize() != KernelABI::get().getPointerArgumentSize())
          return kernel.emitOpError("pointer argument descriptor mismatch");
        StringRef expectedSpace =
            getAddressSpaceName(pointer.getAddressSpace());
        if (descriptor->getAddressSpace().getValue() != expectedSpace)
          return kernel.emitOpError(
              "pointer address-space descriptor mismatch");
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

  std::optional<KernelAddressSpace>
  getAddressSpace(Attribute addressSpace) const {
    if (isa<xw::PrivateAddressSpaceAttr>(addressSpace))
      return KernelAddressSpace::privateSpace;
    if (isa<xw::GlobalAddressSpaceAttr>(addressSpace))
      return KernelAddressSpace::global;
    if (isa<xw::ConstantAddressSpaceAttr>(addressSpace))
      return KernelAddressSpace::constant;
    if (isa<xw::LocalAddressSpaceAttr>(addressSpace))
      return KernelAddressSpace::local;
    if (isa<xw::GenericAddressSpaceAttr>(addressSpace))
      return KernelAddressSpace::generic;
    return std::nullopt;
  }

  StringRef getAddressSpaceName(Attribute addressSpace) const {
    std::optional<KernelAddressSpace> decoded = getAddressSpace(addressSpace);
    return decoded ? KernelABI::get().getAddressSpaceName(*decoded) : "";
  }

  FailureOr<uint64_t> getSlmSize(func::FuncOp kernel) const {
    uint64_t size = 0;
    WalkResult result = kernel.walk([&](xw::AllocOp allocation) {
      uint64_t offset = allocation.getOffset().value_or(0);
      uint64_t alignment = allocation.getAlign();
      if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        allocation.emitOpError(
            "SLM allocation alignment must be a power of two");
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
    payloadTail.clear();
    localIds.fill(Value());
    usedIdAxes.fill(false);
    subgroupIdAxes.fill(false);
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
    if (simdWidth < 0 ||
        !target->supportsSimdWidth(static_cast<uint32_t>(simdWidth)))
      return kernel.emitOpError("SIMD width is unsupported by target '")
             << target->getChipName() << "'";
    ArrayAttr workGroupSize =
        kernel->getAttrOfType<ArrayAttr>("xw.required_work_group_size");
    if (workGroupSize && (workGroupSize.size() != 3 ||
                          llvm::any_of(workGroupSize, [](Attribute value) {
                            IntegerAttr integer = dyn_cast<IntegerAttr>(value);
                            return !integer || integer.getInt() <= 0;
                          })))
      return kernel.emitOpError(
          "xw.required_work_group_size must contain three positive integers");

    WalkResult idWalk = kernel.walk([&](Operation *operation) {
      if (isa<xw::SubgroupIdOp>(operation)) {
        subgroupIdAxes.fill(true);
        if (workGroupSize) {
          subgroupIdAxes[1] = cast<IntegerAttr>(workGroupSize[1]).getInt() != 1;
          subgroupIdAxes[2] = cast<IntegerAttr>(workGroupSize[2]).getInt() != 1;
        }
        for (unsigned axis = 0; axis < subgroupIdAxes.size(); ++axis)
          usedIdAxes[axis] |= subgroupIdAxes[axis];
        return WalkResult::advance();
      }
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
    machineFunction->setAttr(kTargetAttrName, target->getAttr(context));
    machineFunction->setAttr(kKernelArgsAttrName, kernelArguments);
    machineFunction->setAttr(kGrfCountAttrName, moduleBuilder.getI32IntegerAttr(
                                                    target->getGrfCount()));
    machineFunction->setAttr(
        kReservedGrfCountAttrName,
        moduleBuilder.getI32IntegerAttr(
            KernelABI::get().getReservedPayloadGrfCount()));
    machineFunction->setAttr(kSimdSizeAttrName,
                             moduleBuilder.getI32IntegerAttr(simdWidth));
    if (workGroupSize)
      machineFunction->setAttr(kRequiredWorkGroupSizeAttrName, workGroupSize);
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
                               moduleBuilder.getI32IntegerAttr(
                                   KernelABI::get().getInlinePayloadSize()));
    }
    if (usesThreadIds)
      machineFunction->setAttr(
          kPerThreadPayloadSizeAttrName,
          moduleBuilder.getI32IntegerAttr(getPerThreadPayloadSize()));

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

  void emitLocalIdEntry() {
    const KernelABI &abi = KernelABI::get();
    Value r0 = architecturalRegister(0);
    Value r1 = architecturalRegister(1);
    int64_t inlineDataRegister =
        abi.getInlineDataRegister(getPerThreadPayloadSize());
    MovOp::create(*builder, *location,
                  RegType::get(context, 16, inlineDataRegister), i32(), 8,
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
                      immediate(abi.getLocalIdBlobOffset(), i32()))
            .getResult();
    Value threadSlot =
        AndOp::create(*builder, *location, RegType::get(context, 16, 7), i16(),
                      1, canonicalDestination(), uniformRegion(), RegionAttr(),
                      IntegerAttr(), builder->getI32IntegerAttr(4),
                      IntegerAttr(), TypeAttr(), TypeAttr(), true, 0, r0,
                      immediate(0xff, i16()))
            .getResult();
    MulOp offsetAccumulator = MulOp::create(
        *builder, *location, ARFType::get(context, ARFFile::acc, 16, 0), i32(),
        1, canonicalDestination(), uniformRegion(), RegionAttr(), IntegerAttr(),
        IntegerAttr(), IntegerAttr(), typeAttr(i16()), typeAttr(i16()), true, 0,
        threadSlot, immediate(getPerThreadPayloadSize(), i16()));
    Value threadOffset =
        emitMove(RegType::get(context, 16, 8), i32(), 1,
                 offsetAccumulator.getResult(), uniformRegion(), true);
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
                        TypeAttr(), true, 0, address,
                        immediate(2 * abi.getLocalIdAxisStride(), i32()))
              .getResult();
      LoadBlockA32Op z = LoadBlockA32Op::create(
          *builder, *location, RegType::get(context, 16, 3),
          MemTokenType::get(context), zAddress, memoryToken, 16);
      z->setAttr(kAllowFixedOverlapAttrName, builder->getUnitAttr());
      memoryToken = z.getToken();
    }
  }

  int64_t getPerThreadPayloadSize() const {
    for (int64_t axis = 2; axis >= 0; --axis)
      if (usedIdAxes[axis])
        return KernelABI::get().getPerThreadPayloadSize(axis);
    llvm_unreachable("thread payload size requires a used ID axis");
  }

  Value getInlineDataRegister() {
    int64_t index = 1;
    if (usedIdAxes[0] || usedIdAxes[1] || usedIdAxes[2])
      index = KernelABI::get().getInlineDataRegister(getPerThreadPayloadSize());
    return architecturalRegister(index);
  }

  LogicalResult emitPrologue() {
    if (prologueEmitted)
      return success();
    prologueEmitted = true;
    bool usesThreadIds = usedIdAxes[0] || usedIdAxes[1] || usedIdAxes[2];
    if (usesThreadIds) {
      PayloadPrologueOp prologue =
          PayloadPrologueOp::create(*builder, *location);
      Block &body = prologue.getBody().emplaceBlock();
      builder->setInsertionPointToStart(&body);
      emitLocalIdEntry();
      PayloadPrologueEndOp::create(*builder, *location);
      builder->setInsertionPointAfter(prologue);
      memoryToken = Value();
      for (unsigned index = 0; index < 4; ++index)
        emitSync(SyncKind::nop);
      emitSync(SyncKind::allwr);
    }

    Value r0 = architecturalRegister(0);
    Value base =
        AndOp::create(*builder, *location, reg(16), i32(), 1,
                      canonicalDestination(), uniformRegion(), RegionAttr(),
                      IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                      TypeAttr(), true, 0, r0, immediate(0xFFFFFFC0, i32()))
            .getResult();
    const KernelABI &abi = KernelABI::get();
    uint64_t payloadEnd = abi.getInlinePayloadSize();
    for (Attribute attribute : kernelArguments) {
      KernelArgAttr descriptor = cast<KernelArgAttr>(attribute);
      payloadEnd =
          std::max(payloadEnd, descriptor.getOffset() + descriptor.getSize());
    }
    for (uint64_t offset = abi.getInlinePayloadSize(); offset < payloadEnd;
         offset += abi.getPayloadChunkSize()) {
      Value address = base;
      if (offset != abi.getInlinePayloadSize())
        address =
            AddOp::create(*builder, *location, reg(16), i32(), 1,
                          canonicalDestination(), uniformRegion(), RegionAttr(),
                          IntegerAttr(), IntegerAttr(), IntegerAttr(),
                          TypeAttr(), TypeAttr(), true, 0, base,
                          immediate(offset - abi.getInlinePayloadSize(), i32()))
                .getResult();
      LoadBlockA32Op tail = LoadBlockA32Op::create(*builder, *location, reg(16),
                                                   MemTokenType::get(context),
                                                   address, memoryToken, 16);
      memoryToken = tail.getToken();
      payloadTail.push_back(tail.getDst());
    }
    for (unsigned axis = 0; axis < 3; ++axis)
      if (usedIdAxes[axis])
        localIds[axis] = architecturalRegister(1 + axis);
    return success();
  }

  FailureOr<Value> lowerBareArgument(BlockArgument argument, Operation *owner) {
    if (Value found = values.lookup(argument))
      return found;
    FailureOr<KernelArgAttr> descriptor = getKernelArgument(argument, owner);
    FailureOr<std::pair<Value, int64_t>> payload =
        getPayloadLocation(argument, owner);
    if (failed(descriptor) || failed(payload))
      return failure();
    Type elementType = argument.getType();
    if (isa<xw::PtrType>(elementType))
      elementType = isLocalPointer(elementType) ? i32() : i64();
    FailureOr<int64_t> bits = getElementBits(elementType, owner);
    if (failed(bits))
      return failure();
    Value result = emitMove(reg((*bits + 31) / 32), elementType, 1,
                            payload->first, uniformRegion(), true, 0,
                            builder->getI32IntegerAttr(payload->second));
    values[argument] = result;
    return result;
  }

  FailureOr<int64_t> getConstantBits(Attribute value, Operation *owner) const {
    if (IntegerAttr integer = dyn_cast<IntegerAttr>(value))
      return integer.getValue().getSExtValue();
    if (FloatAttr floating = dyn_cast<FloatAttr>(value))
      return static_cast<int64_t>(
          floating.getValue().bitcastToAPInt().getZExtValue());
    if (DenseElementsAttr dense = dyn_cast<DenseElementsAttr>(value)) {
      if (!dense.isSplat())
        return owner->emitOpError(
                   "non-splat SIMD constants have no machine immediate form"),
               failure();
      if (dense.getElementType().isIntOrIndex())
        return dense.getSplatValue<APInt>().getSExtValue();
      if (isa<FloatType>(dense.getElementType()))
        return static_cast<int64_t>(
            dense.getSplatValue<APFloat>().bitcastToAPInt().getZExtValue());
    }
    return owner->emitOpError("unsupported constant attribute"), failure();
  }

  FailureOr<int64_t> getConstantBits(xw::ConstantOp constant) const {
    return getConstantBits(constant.getValue(), constant);
  }

  LogicalResult lowerConstant(Value result, Attribute value, Operation *owner) {
    FailureOr<ValueShape> shape = getShape(result.getType(), owner);
    FailureOr<int64_t> bits = getConstantBits(value, owner);
    if (failed(shape) || failed(bits))
      return failure();
    if (VectorType vector = dyn_cast<VectorType>(shape->elementType)) {
      FailureOr<Value> splat =
          materializeVectorSplat(*bits, vector, shape->cardinality, owner);
      if (failed(splat))
        return failure();
      values[result] = *splat;
      return success();
    }
    Value selected = immediate(*bits, shape->elementType);
    if (isWideSimd(result.getType()))
      wideValues[result] = {selected, selected};
    else
      values[result] = selected;
    return success();
  }

  FailureOr<Value> getValue(Value source, Operation *owner) {
    if (Value found = values.lookup(source))
      return found;
    if (BlockArgument argument = dyn_cast<BlockArgument>(source))
      return lowerBareArgument(argument, owner);
    if (xw::ConstantOp constant = source.getDefiningOp<xw::ConstantOp>()) {
      if (failed(lowerConstant(source, constant.getValue(), constant)))
        return failure();
      return isWideSimd(source.getType()) ? wideValues.lookup(source).low
                                          : values.lookup(source);
    }
    if (arith::ConstantOp constant =
            source.getDefiningOp<arith::ConstantOp>()) {
      if (failed(lowerConstant(source, constant.getValue(), constant)))
        return failure();
      return isWideSimd(source.getType()) ? wideValues.lookup(source).low
                                          : values.lookup(source);
    }
    return owner->emitOpError("operand was not selected"), failure();
  }

  FailureOr<Value> getUniformCondition(Value source, Operation *owner) {
    FailureOr<Value> selected = getValue(source, owner);
    if (failed(selected))
      return failure();
    if (ARFType type = dyn_cast<ARFType>(selected->getType())) {
      if (type.getFile() == ARFFile::f)
        return *selected;
      return owner->emitOpError("uniform condition uses a non-flag ARF"),
             failure();
    }
    return CmpOp::create(
               *builder, *location, ARFType::get(context, ARFFile::f, 2, -1),
               CondModifierAttr::get(context, CondModifier::ne),
               typeAttr(i32()), builder->getI32IntegerAttr(1), uniformRegion(),
               RegionAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
               TypeAttr(), *selected, immediate(0, i32()))
        .getFlag();
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
    if (source.getType().isIntOrIndexOrFloat() ||
        isa<xw::PtrType>(source.getType()))
      return WideValue{*scalar, *scalar};
    return owner->emitOpError("SIMD32 i64 operand was not decomposed"),
           failure();
  }

  LogicalResult lowerPoison(ub::PoisonOp operation) {
    if (!isa<ub::PoisonAttr>(operation.getValue()))
      return operation.emitOpError(
          "selector accepts only fully poisoned #ub.poison values");

    Type type = operation.getType();
    FailureOr<ValueShape> shape = getShape(type, operation);
    if (failed(shape))
      return failure();
    Type elementType = shape->elementType;
    if (VectorType vector = dyn_cast<VectorType>(elementType)) {
      if (vector.getRank() != 1 || vector.isScalable() ||
          !vector.getElementType().isIntOrIndexOrFloat())
        return operation.emitOpError("unsupported UB poison result type ")
               << type;
      values[operation.getResult()] = immediate(0, vector.getElementType());
      return success();
    }
    if (!elementType.isIntOrIndexOrFloat() && !isa<xw::PtrType>(elementType))
      return operation.emitOpError("unsupported UB poison result type ")
             << type;

    Type machineType = elementType;
    if (xw::PtrType pointer = dyn_cast<xw::PtrType>(elementType))
      machineType = isa<xw::LocalAddressSpaceAttr>(pointer.getAddressSpace())
                        ? i32()
                        : i64();
    Value zero = immediate(0, machineType);
    if (isWideSimd(type)) {
      WideValue result{zero, zero};
      wideValues[operation.getResult()] = result;
      if (isa<xw::PtrType>(elementType))
        widePointers[operation.getResult()] = result;
    } else {
      values[operation.getResult()] = zero;
    }
    return success();
  }

  LogicalResult lowerFreeze(xw::FreezeOp operation) {
    Type type = operation.getType();
    if (isWideSimd(type)) {
      FailureOr<WideValue> source =
          isa<xw::PtrType>(cast<xw::SimdType>(type).getElementType())
              ? materializeWidePointer(operation.getSource(), operation, 32)
              : getWideValue(operation.getSource(), operation);
      if (failed(source))
        return failure();
      wideValues[operation.getResult()] = *source;
      if (isa<xw::PtrType>(cast<xw::SimdType>(type).getElementType()))
        widePointers[operation.getResult()] = *source;
      return success();
    }
    FailureOr<Value> source = getValue(operation.getSource(), operation);
    if (failed(source))
      return failure();
    values[operation.getResult()] = *source;
    return success();
  }

  FailureOr<Value> materialize(Value source, Operation *owner) {
    FailureOr<Value> value = getValue(source, owner);
    FailureOr<ValueShape> shape = getShape(source.getType(), owner);
    FailureOr<int64_t> footprint = getFootprint(source.getType(), owner);
    if (failed(value) || failed(shape) || failed(footprint))
      return failure();
    if (!value->getDefiningOp<ImmOp>())
      return *value;
    if (VectorType vector = dyn_cast<VectorType>(shape->elementType)) {
      ImmOp immediateValue = value->getDefiningOp<ImmOp>();
      FailureOr<Value> splat = materializeVectorSplat(
          immediateValue.getValue(), vector, shape->cardinality, owner);
      if (failed(splat))
        return failure();
      return *splat;
    }
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
    if (input->getDefiningOp<ImmOp>() &&
        isa<VectorType>(resultShape->elementType)) {
      VectorType vector = cast<VectorType>(resultShape->elementType);
      ImmOp immediateValue = input->getDefiningOp<ImmOp>();
      FailureOr<Value> splat =
          materializeVectorSplat(immediateValue.getValue(), vector,
                                 resultShape->cardinality, operation);
      if (failed(splat))
        return failure();
      values[operation->getResult(0)] = *splat;
      return success();
    }
    if (isWideSimd(operation->getResult(0).getType())) {
      auto moveHalf = [&](int64_t maskOffset) {
        RegionAttr region =
            sourceShape->cardinality == 1
                ? uniformRegion()
                : RegionAttr::get(context, 1, 16 / sourceShape->cardinality, 0);
        return emitMove(reg(32), resultShape->elementType, 16, *input, region,
                        false, maskOffset);
      };
      wideValues[operation->getResult(0)] = {moveHalf(0), moveHalf(16)};
      return success();
    }
    RegionAttr region =
        resultShape->cardinality == 1 && sourceShape->cardinality > 1
            ? uniformRegion()
            : sourceRegion(source, resultShape->cardinality, operation);
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
      result =
          AddOp::create(*builder, *location, reg(*footprint),
                        shape->elementType, shape->cardinality,
                        canonicalDestination(), lhsRegion, rhsRegion,
                        IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                        TypeAttr(), shape->cardinality == 1, 0, *lhs, *rhs)
              .getResult();
      break;
    case xw::BinaryKind::SubI:
      result =
          SubOp::create(*builder, *location, reg(*footprint),
                        shape->elementType, shape->cardinality,
                        canonicalDestination(), rhsRegion, lhsRegion,
                        IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                        TypeAttr(), shape->cardinality == 1, 0, *rhs, *lhs)
              .getResult();
      break;
    case xw::BinaryKind::ShLI:
    case xw::BinaryKind::ShRUI:
      if (operation.getKind() == xw::BinaryKind::ShLI)
        result =
            ShlOp::create(
                *builder, *location, reg(*footprint), shape->elementType,
                shape->cardinality, canonicalDestination(), lhsRegion,
                rhsRegion, IntegerAttr(), IntegerAttr(), IntegerAttr(),
                TypeAttr(),
                shape->elementType.isInteger(64) ? typeAttr(i16()) : TypeAttr(),
                shape->cardinality == 1, 0, *lhs, *rhs)
                .getResult();
      else
        result =
            ShrOp::create(
                *builder, *location, reg(*footprint), shape->elementType,
                shape->cardinality, canonicalDestination(), lhsRegion,
                rhsRegion, IntegerAttr(), IntegerAttr(), IntegerAttr(),
                TypeAttr(),
                shape->elementType.isInteger(64) ? typeAttr(i16()) : TypeAttr(),
                shape->cardinality == 1, 0, *lhs, *rhs)
                .getResult();
      break;
    case xw::BinaryKind::AndI:
    case xw::BinaryKind::OrI:
      if (operation.getKind() == xw::BinaryKind::AndI)
        result = AndOp::create(*builder, *location, reg(*footprint),
                               shape->elementType, shape->cardinality,
                               canonicalDestination(), lhsRegion, rhsRegion,
                               IntegerAttr(), IntegerAttr(), IntegerAttr(),
                               TypeAttr(), TypeAttr(), shape->cardinality == 1,
                               0, *lhs, *rhs)
                     .getResult();
      else
        result = OrOp::create(*builder, *location, reg(*footprint),
                              shape->elementType, shape->cardinality,
                              canonicalDestination(), lhsRegion, rhsRegion,
                              IntegerAttr(), IntegerAttr(), IntegerAttr(),
                              TypeAttr(), TypeAttr(), shape->cardinality == 1,
                              0, *lhs, *rhs)
                     .getResult();
      break;
    case xw::BinaryKind::XOrI: {
      Value joined =
          OrOp::create(*builder, *location, reg(*footprint), shape->elementType,
                       shape->cardinality, canonicalDestination(), lhsRegion,
                       rhsRegion, IntegerAttr(), IntegerAttr(), IntegerAttr(),
                       TypeAttr(), TypeAttr(), shape->cardinality == 1, 0, *lhs,
                       *rhs)
              .getResult();
      Value overlap =
          AndOp::create(*builder, *location, reg(*footprint),
                        shape->elementType, shape->cardinality,
                        canonicalDestination(), lhsRegion, rhsRegion,
                        IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                        TypeAttr(), shape->cardinality == 1, 0, *lhs, *rhs)
              .getResult();
      result = SubOp::create(*builder, *location, reg(*footprint),
                             shape->elementType, shape->cardinality,
                             canonicalDestination(), canonicalRegion(),
                             canonicalRegion(), IntegerAttr(), IntegerAttr(),
                             IntegerAttr(), TypeAttr(), TypeAttr(),
                             shape->cardinality == 1, 0, overlap, joined)
                   .getResult();
      break;
    }
    case xw::BinaryKind::MulI: {
      Value accumulator =
          MulOp::create(*builder, *location,
                        ARFType::get(context, ARFFile::acc, *footprint, 0),
                        shape->elementType, shape->cardinality,
                        canonicalDestination(), lhsRegion, rhsRegion,
                        IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                        TypeAttr(), shape->cardinality == 1, 0, *lhs, *rhs)
              .getResult();
      result =
          emitMove(reg(*footprint), shape->elementType, shape->cardinality,
                   accumulator, canonicalRegion(), shape->cardinality == 1);
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
    if (operation.getKind() == xw::BinaryKind::DivUI ||
        operation.getKind() == xw::BinaryKind::RemUI ||
        operation.getKind() == xw::BinaryKind::DivSI ||
        operation.getKind() == xw::BinaryKind::RemSI)
      return operation.emitOpError(
          "SIMD32 i64 division/remainder has no exact two-half flag "
          "selection");
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
      RegionAttr leftRegion =
          left.getDefiningOp<ImmOp>() ? RegionAttr() : canonicalRegion();
      RegionAttr rightRegion =
          right.getDefiningOp<ImmOp>() ? RegionAttr() : canonicalRegion();
      if (operation.getKind() == xw::BinaryKind::AddI)
        return AddOp::create(*builder, *location, reg(32), i64(), 16,
                             canonicalDestination(), leftRegion, rightRegion,
                             IntegerAttr(), IntegerAttr(), IntegerAttr(),
                             TypeAttr(), TypeAttr(), false, maskOffset, left,
                             right)
            .getResult();
      if (operation.getKind() == xw::BinaryKind::SubI)
        return SubOp::create(*builder, *location, reg(32), i64(), 16,
                             canonicalDestination(), rightRegion, leftRegion,
                             IntegerAttr(), IntegerAttr(), IntegerAttr(),
                             TypeAttr(), TypeAttr(), false, maskOffset, right,
                             left)
            .getResult();
      if (operation.getKind() == xw::BinaryKind::ShLI)
        return ShlOp::create(*builder, *location, reg(32), i64(), 16,
                             canonicalDestination(), leftRegion, rightRegion,
                             IntegerAttr(), IntegerAttr(), IntegerAttr(),
                             TypeAttr(), typeAttr(i16()), false, maskOffset,
                             left, right)
            .getResult();
      return ShrOp::create(*builder, *location, reg(32), i64(), 16,
                           canonicalDestination(), leftRegion, rightRegion,
                           IntegerAttr(), IntegerAttr(), IntegerAttr(),
                           TypeAttr(), typeAttr(i16()), false, maskOffset, left,
                           right)
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
      result = SubOp::create(*builder, *location, reg(*footprint),
                             shape->elementType, shape->cardinality,
                             canonicalDestination(), rhsRegion, lhsRegion,
                             IntegerAttr(), IntegerAttr(), IntegerAttr(),
                             TypeAttr(), TypeAttr(), false, 0, *rhs, *lhs)
                   .getResult();
    else
      result = AddOp::create(*builder, *location, reg(*footprint),
                             shape->elementType, shape->cardinality,
                             canonicalDestination(), lhsRegion, rhsRegion,
                             IntegerAttr(), IntegerAttr(), IntegerAttr(),
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
            IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(), TypeAttr(),
            false, 0, *lhs, *rhs)
            .getResult();
    values[operation.getResult()] =
        emitMove(reg(*footprint), shape->elementType, shape->cardinality,
                 accumulator, canonicalRegion());
    return success();
  }

  LogicalResult lowerDpas(xw::DpasOp operation) {
    FailureOr<Value> a = materialize(operation.getA(), operation);
    FailureOr<Value> b = materialize(operation.getB(), operation);
    FailureOr<Value> acc = materialize(operation.getAcc(), operation);
    if (failed(a) || failed(b) || failed(acc))
      return failure();
    FailureOr<int64_t> footprint = getFootprint(operation.getType(), operation);
    if (failed(footprint))
      return failure();
    auto precision = [](xw::DpasPrecision value) {
      return value == xw::DpasPrecision::F16 ? DpasPrecision::F16
                                             : DpasPrecision::BF16;
    };
    DpasOp dpas = DpasOp::create(
        *builder, *location, reg(*footprint), *a, *b, *acc,
        DpasPrecisionAttr::get(context, precision(operation.getAPrecision())),
        DpasPrecisionAttr::get(context, precision(operation.getBPrecision())),
        builder->getI32IntegerAttr(operation.getSystolicDepth()),
        builder->getI32IntegerAttr(operation.getRepeatCount()),
        typeAttr(Float32Type::get(context)));
    values[operation.getResult()] = dpas.getDst();
    return success();
  }

  LogicalResult lowerBitcast(xw::BitcastOp operation) {
    FailureOr<Value> source = getValue(operation.getSource(), operation);
    FailureOr<int64_t> sourceFootprint =
        getFootprint(operation.getSource().getType(), operation);
    FailureOr<int64_t> resultFootprint =
        getFootprint(operation.getType(), operation);
    if (failed(source) || failed(sourceFootprint) || failed(resultFootprint))
      return failure();
    if (*sourceFootprint != *resultFootprint)
      return operation.emitOpError("bitcast register footprints must match");
    values[operation.getResult()] = *source;
    return success();
  }

  LogicalResult lowerUnsupportedFloat(Operation *operation,
                                      StringRef primitive) {
    return operation->emitOpError()
           << primitive << " has no exact XeMachine primitive";
  }

  LogicalResult lowerMaskBinary(Operation *operation) {
    FailureOr<Value> lhs = getValue(operation->getOperand(0), operation);
    FailureOr<Value> rhs = getValue(operation->getOperand(1), operation);
    FailureOr<ValueShape> shape =
        getShape(operation->getResult(0).getType(), operation);
    if (failed(lhs) || failed(rhs) || failed(shape))
      return failure();
    Type flagType = ARFType::get(context, ARFFile::f, 2, -1);
    Value result;
    if (isa<xw::MaskAndOp>(operation))
      result = AndOp::create(*builder, *location, flagType, i32(), 1,
                             canonicalDestination(), uniformRegion(),
                             uniformRegion(), IntegerAttr(), IntegerAttr(),
                             IntegerAttr(), TypeAttr(), TypeAttr(), true, 0,
                             *lhs, *rhs)
                   .getResult();
    else if (isa<xw::MaskOrOp>(operation))
      result =
          OrOp::create(*builder, *location, flagType, i32(), 1,
                       canonicalDestination(), uniformRegion(), uniformRegion(),
                       IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                       TypeAttr(), true, 0, *lhs, *rhs)
              .getResult();
    else {
      Value joined =
          OrOp::create(*builder, *location, reg(1), i32(), 1,
                       canonicalDestination(), uniformRegion(), uniformRegion(),
                       IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                       TypeAttr(), true, 0, *lhs, *rhs)
              .getResult();
      Value overlap = AndOp::create(*builder, *location, reg(1), i32(), 1,
                                    canonicalDestination(), uniformRegion(),
                                    uniformRegion(), IntegerAttr(),
                                    IntegerAttr(), IntegerAttr(), TypeAttr(),
                                    TypeAttr(), true, 0, *lhs, *rhs)
                          .getResult();
      result = SubOp::create(*builder, *location, flagType, i32(), 1,
                             canonicalDestination(), uniformRegion(),
                             uniformRegion(), IntegerAttr(), IntegerAttr(),
                             IntegerAttr(), TypeAttr(), TypeAttr(), true, 0,
                             overlap, joined)
                   .getResult();
    }
    values[operation->getResult(0)] = result;
    return success();
  }

  LogicalResult lowerArithXor(arith::XOrIOp operation) {
    if (!operation.getType().isInteger(1))
      return operation.emitOpError("selector supports arith.xori only for i1");
    FailureOr<Value> lhs = getValue(operation.getLhs(), operation);
    FailureOr<Value> rhs = getValue(operation.getRhs(), operation);
    if (failed(lhs) || failed(rhs))
      return failure();
    RegionAttr lhsRegion =
        lhs->getDefiningOp<ImmOp>() ? RegionAttr() : uniformRegion();
    RegionAttr rhsRegion =
        rhs->getDefiningOp<ImmOp>() ? RegionAttr() : uniformRegion();
    Value joined = OrOp::create(*builder, *location, reg(1), i1(), 1,
                                canonicalDestination(), lhsRegion, rhsRegion,
                                IntegerAttr(), IntegerAttr(), IntegerAttr(),
                                TypeAttr(), TypeAttr(), true, 0, *lhs, *rhs)
                       .getResult();
    Value overlap = AndOp::create(*builder, *location, reg(1), i1(), 1,
                                  canonicalDestination(), lhsRegion, rhsRegion,
                                  IntegerAttr(), IntegerAttr(), IntegerAttr(),
                                  TypeAttr(), TypeAttr(), true, 0, *lhs, *rhs)
                        .getResult();
    values[operation.getResult()] =
        SubOp::create(*builder, *location, reg(1), i1(), 1,
                      canonicalDestination(), uniformRegion(), uniformRegion(),
                      IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                      TypeAttr(), true, 0, overlap, joined)
            .getResult();
    return success();
  }

  LogicalResult lowerArithExtUI(arith::ExtUIOp operation) {
    FailureOr<Value> source = getValue(operation.getIn(), operation);
    FailureOr<int64_t> footprint = getFootprint(operation.getType(), operation);
    if (failed(source) || failed(footprint))
      return failure();
    values[operation.getResult()] =
        MovOp::create(*builder, *location, reg(*footprint), operation.getType(),
                      1, canonicalDestination(),
                      source->getDefiningOp<ImmOp>() ? RegionAttr()
                                                     : uniformRegion(),
                      IntegerAttr(), IntegerAttr(),
                      typeAttr(operation.getIn().getType()), true, 0, *source)
            .getResult();
    return success();
  }

  LogicalResult lowerMaskNot(xw::MaskNotOp operation) {
    FailureOr<Value> input = getValue(operation.getInput(), operation);
    FailureOr<ValueShape> shape = getShape(operation.getType(), operation);
    if (failed(input) || failed(shape))
      return failure();
    uint64_t activeBits = shape->cardinality == 32
                              ? 0xffffffffULL
                              : (1ULL << shape->cardinality) - 1;
    values[operation.getResult()] =
        SubOp::create(*builder, *location,
                      ARFType::get(context, ARFFile::f, 2, -1), i32(), 1,
                      canonicalDestination(), uniformRegion(), RegionAttr(),
                      IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                      TypeAttr(), true, 0, *input, immediate(activeBits, i32()))
            .getResult();
    return success();
  }

  LogicalResult lowerBallot(xw::BallotOp operation) {
    FailureOr<Value> mask = getValue(operation.getMask(), operation);
    if (failed(mask))
      return failure();
    Type resultType = operation.getType();
    values[operation.getResult()] =
        emitMove(reg(1), resultType, 1, *mask, uniformRegion(), true);
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
            *builder, *location, reg(32), i64(), 16, canonicalDestination(),
            canonicalRegion(), IntegerAttr(),
            builder->getI32IntegerAttr(offset),
            typeAttr(sourceShape->elementType), false, offset, *source);
        if (operation.getKind() == xw::CastKind::IntConvert &&
            operation.getPolicy()) {
          DictionaryAttr policy = *operation.getPolicy();
          auto extension = dyn_cast_or_null<xw::CastExtensionPolicyAttr>(
              policy.get("extension"));
          if (operation.getKind() == xw::CastKind::IntConvert && extension &&
              extension.getValue() == xw::CastExtension::Sign)
            move->setAttr("signedSource", builder->getUnitAttr());
        }
        return move.getResult();
      };
      wideValues[operation.getResult()] = {half(0), half(16)};
      return success();
    }
    FailureOr<ValueShape> sourceShape =
        getShape(operation.getSource().getType(), operation);
    FailureOr<ValueShape> resultShape =
        getShape(operation.getType(), operation);
    FailureOr<Value> source = getValue(operation.getSource(), operation);
    FailureOr<int64_t> footprint = getFootprint(operation.getType(), operation);
    if (failed(sourceShape) || failed(resultShape) || failed(source) ||
        failed(footprint))
      return failure();
    MovOp move = MovOp::create(
        *builder, *location, reg(*footprint), resultShape->elementType,
        resultShape->cardinality, canonicalDestination(),
        sourceRegion(operation.getSource(), resultShape->cardinality,
                     operation),
        IntegerAttr(), IntegerAttr(), typeAttr(sourceShape->elementType),
        resultShape->cardinality == 1, 0, *source);
    if (operation.getPolicy()) {
      DictionaryAttr policy = *operation.getPolicy();
      auto extension = dyn_cast_or_null<xw::CastExtensionPolicyAttr>(
          policy.get("extension"));
      if (operation.getKind() == xw::CastKind::IntConvert && extension &&
          extension.getValue() == xw::CastExtension::Sign)
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
    case arith::CmpFPredicate::UNE:
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
    if (isWideSimd(operation.getLhs().getType()))
      return operation.emitOpError(
          "SIMD32 i64 comparison has no exact two-half flag selection");
    FailureOr<ValueShape> resultShape =
        getShape(operation.getResult().getType(), operation);
    FailureOr<ValueShape> operandShape =
        getShape(operation.getLhs().getType(), operation);
    FailureOr<Value> lhs = getValue(operation.getLhs(), operation);
    FailureOr<Value> rhs = getValue(operation.getRhs(), operation);
    if (failed(resultShape) || failed(operandShape) || failed(lhs) ||
        failed(rhs))
      return failure();
    std::optional<CondModifier> condition =
        mapPredicate(operation.getPredicate());
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
    if (failed(resultShape) || failed(operandShape) || failed(lhs) ||
        failed(rhs))
      return failure();
    std::optional<CondModifier> condition =
        mapPredicate(operation.getPredicate());
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
    if (isWideSimd(operation.getType()))
      return operation.emitOpError(
          "SIMD32 i64 or A64 pointer select has no exact two-half selection");
    FailureOr<Value> condition =
        isa<xw::MaskType>(operation.getCondition().getType())
            ? getValue(operation.getCondition(), operation)
            : getUniformCondition(operation.getCondition(), operation);
    FailureOr<Value> trueValue = getValue(operation.getTrueValue(), operation);
    FailureOr<Value> falseValue =
        getValue(operation.getFalseValue(), operation);
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
      Region &region =
          index == 0 ? machineIf->getRegion(0) : machineIf->getRegion(1);
      builder->setInsertionPointToStart(&region.emplaceBlock());
      Value yielded = arms[index];
      if (yielded.getType() != resultType) {
        if (!yielded.getDefiningOp<ImmOp>())
          return operation.emitOpError(
              "selected branch value has incompatible machine type");
        yielded = emitMove(resultType, shape->elementType, shape->cardinality,
                           yielded, RegionAttr());
      }
      YieldOp::create(*builder, *location, ValueRange{yielded});
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
        FailureOr<int64_t> footprint =
            getFootprint(result.getType(), operation);
        if (failed(footprint))
          return failure();
        resultTypes.push_back(reg(*footprint));
      }
    }
    ExecIfOp machineIf =
        ExecIfOp::create(*builder, *location, resultTypes, *condition);
    for (unsigned index = 0; index < operation.getNumResults(); ++index)
      values[operation.getResult(index)] = machineIf.getResult(index);
    if (failed(lowerBranchRegions(operation.getOperation(), machineIf,
                                  /*uniform=*/false)))
      return failure();
    return success();
  }

  LogicalResult lowerScfIf(scf::IfOp operation) {
    FailureOr<Value> condition =
        getUniformCondition(operation.getCondition(), operation);
    if (failed(condition))
      return failure();
    SmallVector<Type> resultTypes;
    for (Value result : operation.getResults()) {
      if (isa<xw::MemTokenType>(result.getType()))
        resultTypes.push_back(MemTokenType::get(context));
      else {
        FailureOr<int64_t> footprint =
            getFootprint(result.getType(), operation);
        if (failed(footprint))
          return failure();
        resultTypes.push_back(reg(*footprint));
      }
    }
    UniformIfOp machineIf =
        UniformIfOp::create(*builder, *location, resultTypes, *condition);
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
        if (isa<VectorType>(shape->elementType)) {
          if (value->getType() != machineIf->getResult(index).getType())
            return sourceYield->emitOpError(
                "vector packet yield register footprint mismatch");
        }
        if (value->getType() == machineIf->getResult(index).getType()) {
          yielded.push_back(*value);
          continue;
        }
        if (!value->getDefiningOp<ImmOp>())
          return sourceYield->emitOpError(
              "yielded value has incompatible machine type");
        yielded.push_back(emitMove(machineIf->getResult(index).getType(),
                                   shape->elementType, shape->cardinality,
                                   *value, RegionAttr()));
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
    if (operation.getLowerBound().getType() !=
            operation.getUpperBound().getType() ||
        operation.getLowerBound().getType() != operation.getStep().getType())
      return operation.emitOpError("loop bounds and step must have one type");
    FailureOr<Value> lower = materialize(operation.getLowerBound(), operation);
    FailureOr<Value> upper = getValue(operation.getUpperBound(), operation);
    FailureOr<Value> step = getValue(operation.getStep(), operation);
    if (failed(lower) || failed(upper) || failed(step))
      return failure();
    SmallVector<Value> initial{*lower};
    SmallVector<Type> resultTypes{lower->getType()};
    for (Value init : operation.getInitArgs()) {
      if (isa<xw::MemTokenType>(init.getType())) {
        Value selected = values.lookup(init);
        if (!selected)
          return operation.emitOpError(
              "loop token initializer was not selected");
        initial.push_back(selected);
        resultTypes.push_back(MemTokenType::get(context));
      } else {
        FailureOr<Value> selected = getValue(init, operation);
        if (failed(selected))
          return failure();
        initial.push_back(*selected);
        resultTypes.push_back(selected->getType());
      }
    }
    UniformLoopOp loop =
        UniformLoopOp::create(*builder, *location, resultTypes, initial);
    Block &body = loop.getBody().emplaceBlock();
    for (Type type : resultTypes)
      body.addArgument(type, operation.getLoc());
    values[operation.getInductionVar()] = body.getArgument(0);
    for (unsigned index = 0; index < operation.getNumRegionIterArgs();
         ++index) {
      values[operation.getRegionIterArg(index)] = body.getArgument(index + 1);
      if (isa<xw::MemTokenType>(operation.getRegionIterArg(index).getType()))
        memoryToken = body.getArgument(index + 1);
    }
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
        typeAttr(inductionType), builder->getI32IntegerAttr(1),
        canonicalRegion(), uniformRegion(), IntegerAttr(), IntegerAttr(),
        TypeAttr(), TypeAttr(), next, *upper);
    condition->setAttr("signed", builder->getUnitAttr());
    SmallVector<Value> carried{next};
    for (Value source : sourceYield.getOperands()) {
      if (isa<xw::MemTokenType>(source.getType())) {
        Value selected = values.lookup(source);
        if (!selected)
          return sourceYield.emitOpError("loop token yield was not selected");
        carried.push_back(selected);
      } else {
        FailureOr<Value> selected = getValue(source, sourceYield);
        if (failed(selected))
          return failure();
        carried.push_back(*selected);
      }
    }
    ContinueIfOp::create(*builder, *location, condition.getFlag(), carried);
    builder->setInsertionPointAfter(loop);
    for (unsigned index = 0; index < operation.getNumResults(); ++index) {
      values[operation.getResult(index)] = loop.getResult(index + 1);
      if (isa<xw::MemTokenType>(operation.getResult(index).getType()))
        memoryToken = loop.getResult(index + 1);
    }
    return success();
  }

  LogicalResult lowerWhile(scf::WhileOp operation) {
    SmallVector<Value> initial;
    for (Value operand : operation.getInits()) {
      if (isa<xw::MemTokenType>(operand.getType())) {
        Value selected = values.lookup(operand);
        if (!selected)
          return operation.emitOpError(
              "while token initializer was not selected");
        initial.push_back(selected);
      } else {
        FailureOr<Value> selected = getValue(operand, operation);
        if (failed(selected))
          return failure();
        initial.push_back(*selected);
      }
    }
    SmallVector<Type> resultTypes;
    for (Value result : operation.getResults()) {
      if (isa<xw::MemTokenType>(result.getType())) {
        resultTypes.push_back(MemTokenType::get(context));
        continue;
      }
      FailureOr<int64_t> footprint = getFootprint(result.getType(), operation);
      if (failed(footprint))
        return failure();
      resultTypes.push_back(reg(*footprint));
    }
    Block &after = operation.getAfter().front();
    scf::YieldOp sourceYield = cast<scf::YieldOp>(after.getTerminator());
    SmallVector<Value> stateInitial(resultTypes.size());
    SmallVector<unsigned> beforeStateIndices;
    beforeStateIndices.reserve(sourceYield.getNumOperands());
    if (initial.size() == resultTypes.size()) {
      stateInitial.assign(initial.begin(), initial.end());
      for (unsigned index = 0; index < initial.size(); ++index)
        beforeStateIndices.push_back(index);
    } else {
      for (auto [index, operand] : llvm::enumerate(sourceYield.getOperands())) {
        BlockArgument argument = dyn_cast<BlockArgument>(operand);
        if (!argument || argument.getOwner() != &after)
          return sourceYield.emitOpError(
              "asymmetric machine while backedge must directly forward body "
              "arguments");
        unsigned stateIndex = argument.getArgNumber();
        if (stateIndex >= stateInitial.size())
          return sourceYield.emitOpError(
              "machine while state index is invalid");
        if (stateInitial[stateIndex])
          return sourceYield.emitOpError(
              "asymmetric machine while backedge maps multiple initializers "
              "to one state");
        stateInitial[stateIndex] = initial[index];
        beforeStateIndices.push_back(stateIndex);
      }
    }
    for (unsigned index = 0; index < stateInitial.size(); ++index) {
      Type semanticType = operation.getResult(index).getType();
      if (stateInitial[index] &&
          stateInitial[index].getType() == resultTypes[index])
        continue;
      if (isa<xw::MemTokenType>(semanticType))
        return operation.emitOpError(
            "machine while cannot synthesize an initial memory token");
      FailureOr<ValueShape> shape = getShape(semanticType, operation);
      if (failed(shape))
        return failure();
      Value source = stateInitial[index] ? stateInitial[index]
                                         : immediate(0, shape->elementType);
      if (source.getDefiningOp<ImmOp>() &&
          isa<VectorType>(shape->elementType)) {
        VectorType vector = cast<VectorType>(shape->elementType);
        ImmOp vectorImmediate = source.getDefiningOp<ImmOp>();
        FailureOr<Value> splat = materializeVectorSplat(
            vectorImmediate.getValue(), vector, shape->cardinality, operation);
        if (failed(splat))
          return failure();
        stateInitial[index] = *splat;
        continue;
      }
      stateInitial[index] = emitMove(
          resultTypes[index], shape->elementType, shape->cardinality, source,
          source.getDefiningOp<ImmOp>() ? RegionAttr() : uniformRegion(),
          shape->cardinality == 1);
    }
    UniformLoopOp loop =
        UniformLoopOp::create(*builder, *location, resultTypes, stateInitial);
    Block &body = loop.getBody().emplaceBlock();
    for (Type type : resultTypes)
      body.addArgument(type, operation.getLoc());
    Block &before = operation.getBefore().front();
    for (unsigned index = 0; index < before.getNumArguments(); ++index) {
      unsigned stateIndex = beforeStateIndices[index];
      values[before.getArgument(index)] = body.getArgument(stateIndex);
      if (isa<xw::MemTokenType>(before.getArgument(index).getType()))
        memoryToken = body.getArgument(stateIndex);
    }
    builder->setInsertionPointToStart(&body);
    scf::ConditionOp condition = cast<scf::ConditionOp>(before.getTerminator());
    DenseSet<unsigned> backedgeStateSet(beforeStateIndices.begin(),
                                        beforeStateIndices.end());
    SmallVector<Value> earlyExitValues(resultTypes.size());
    for (unsigned index = 0; index < condition.getArgs().size(); ++index) {
      if (backedgeStateSet.contains(index) ||
          isa<xw::MemTokenType>(condition.getArgs()[index].getType()))
        continue;
      BlockArgument argument =
          dyn_cast<BlockArgument>(condition.getArgs()[index]);
      if (!argument || argument.getOwner() != &before)
        continue;
      Value source =
          body.getArgument(beforeStateIndices[argument.getArgNumber()]);
      RegType resultType = cast<RegType>(resultTypes[index]);
      SmallVector<Value, 4> pieces;
      for (uint32_t offset = 0; offset < resultType.getWidthDwords();) {
        uint32_t width =
            std::min<uint32_t>(simdWidth, resultType.getWidthDwords() - offset);
        IntegerAttr sourceSub =
            offset == 0 ? IntegerAttr() : builder->getI32IntegerAttr(offset);
        pieces.push_back(emitMove(reg(width), i32(), width, source,
                                  canonicalRegion(), false, 0, sourceSub));
        offset += width;
      }
      earlyExitValues[index] =
          pieces.size() == 1 ? pieces.front()
                             : TupleFromElementsOp::create(*builder, *location,
                                                           resultType, pieces)
                                   .getTuple();
    }
    if (failed(lowerBlock(before)))
      return failure();
    FailureOr<Value> selectedCondition =
        getUniformCondition(condition.getCondition(), condition);
    if (failed(selectedCondition))
      return failure();
    Value conditionSnapshot =
        emitMove(reg(1), i32(), 1, *selectedCondition, uniformRegion(), true);

    if (after.getNumArguments() != condition.getArgs().size())
      return operation.emitOpError("while region argument count mismatch");
    SmallVector<Value> conditionArguments;
    SmallVector<Type> conditionTypes;
    for (Value argument : condition.getArgs()) {
      if (isa<xw::MemTokenType>(argument.getType())) {
        Value selected = values.lookup(argument);
        if (!selected)
          return condition.emitOpError(
              "while condition token was not selected");
        conditionArguments.push_back(selected);
        conditionTypes.push_back(MemTokenType::get(context));
      } else {
        FailureOr<Value> selected = getValue(argument, condition);
        if (failed(selected))
          return failure();
        conditionArguments.push_back(*selected);
        conditionTypes.push_back(selected->getType());
      }
      if (earlyExitValues[conditionArguments.size() - 1]) {
        conditionArguments.back() =
            earlyExitValues[conditionArguments.size() - 1];
        conditionTypes.back() = conditionArguments.back().getType();
      }
    }
    DenseMap<unsigned, unsigned> backedgeResults;
    SmallVector<Type> backedgeTypes;
    for (unsigned stateIndex : beforeStateIndices) {
      backedgeResults.try_emplace(stateIndex, backedgeTypes.size());
      backedgeTypes.push_back(conditionTypes[stateIndex]);
    }
    Value bodyCondition =
        CmpOp::create(
            *builder, *location, ARFType::get(context, ARFFile::f, 2, -1),
            CondModifierAttr::get(context, CondModifier::ne), typeAttr(i32()),
            builder->getI32IntegerAttr(1), uniformRegion(), RegionAttr(),
            IntegerAttr(), IntegerAttr(), TypeAttr(), TypeAttr(),
            conditionSnapshot, immediate(0, i32()))
            .getFlag();
    UniformIfOp executeBody =
        UniformIfOp::create(*builder, *location, backedgeTypes, bodyCondition);
    builder->setInsertionPointToStart(
        &executeBody.getThenRegion().emplaceBlock());
    for (unsigned index = 0; index < after.getNumArguments(); ++index) {
      values[after.getArgument(index)] = conditionArguments[index];
      if (isa<xw::MemTokenType>(after.getArgument(index).getType()))
        memoryToken = conditionArguments[index];
    }
    if (failed(lowerBlock(after)))
      return failure();
    SmallVector<Value> thenValues(beforeStateIndices.size());
    for (auto [index, operand] : llvm::enumerate(sourceYield.getOperands())) {
      if (isa<xw::MemTokenType>(operand.getType())) {
        Value selected = values.lookup(operand);
        if (!selected)
          return sourceYield.emitOpError("while token yield was not selected");
        thenValues[index] = selected;
      } else {
        FailureOr<Value> selected = getValue(operand, sourceYield);
        if (failed(selected))
          return failure();
        thenValues[index] = *selected;
      }
    }
    YieldOp::create(*builder, *location, thenValues);
    builder->setInsertionPointToStart(
        &executeBody.getElseRegion().emplaceBlock());
    SmallVector<Value> elseValues;
    for (unsigned stateIndex : beforeStateIndices)
      elseValues.push_back(conditionArguments[stateIndex]);
    YieldOp::create(*builder, *location, elseValues);
    builder->setInsertionPointAfter(executeBody);
    Value continueCondition =
        CmpOp::create(
            *builder, *location, ARFType::get(context, ARFFile::f, 2, -1),
            CondModifierAttr::get(context, CondModifier::ne), typeAttr(i32()),
            builder->getI32IntegerAttr(1), uniformRegion(), RegionAttr(),
            IntegerAttr(), IntegerAttr(), TypeAttr(), TypeAttr(),
            conditionSnapshot, immediate(0, i32()))
            .getFlag();
    SmallVector<Value> carried;
    for (unsigned index = 0; index < conditionArguments.size(); ++index) {
      DenseMap<unsigned, unsigned>::iterator backedge =
          backedgeResults.find(index);
      carried.push_back(backedge == backedgeResults.end()
                            ? conditionArguments[index]
                            : executeBody.getResult(backedge->second));
    }
    ContinueIfOp::create(*builder, *location, continueCondition, carried);
    builder->setInsertionPointAfter(loop);
    for (unsigned index = 0; index < operation.getNumResults(); ++index) {
      values[operation.getResult(index)] = loop.getResult(index);
      if (isa<xw::MemTokenType>(operation.getResult(index).getType()))
        memoryToken = loop.getResult(index);
    }
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
      return operation.emitOpError(
          "machine extract requires a rank-one vector");
    FailureOr<int64_t> elementBits =
        getElementBits(vector.getElementType(), operation);
    if (failed(elementBits))
      return failure();
    int64_t elementDwords = (*elementBits * cardinality + 31) / 32;
    SmallVector<Type> parts(vector.getNumElements(), reg(elementDwords));
    TupleToElementsOp split =
        TupleToElementsOp::create(*builder, *location, parts, *source);
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
        return AddOp::create(*builder, *location, reg(32), i64(), 16,
                             canonicalDestination(),
                             lhs.getDefiningOp<ImmOp>() ? RegionAttr()
                                                        : canonicalRegion(),
                             rhs.getDefiningOp<ImmOp>() ? RegionAttr()
                                                        : canonicalRegion(),
                             IntegerAttr(), IntegerAttr(), IntegerAttr(),
                             TypeAttr(), TypeAttr(), false, maskOffset, lhs,
                             rhs)
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
    if (local) {
      FailureOr<ValueShape> offsetShape =
          getShape(operation.getOffset().getType(), operation);
      if (failed(offsetShape))
        return failure();
      FailureOr<int64_t> offsetBits =
          getElementBits(offsetShape->elementType, operation);
      if (failed(offsetBits))
        return failure();
      if (*offsetBits != addressBits)
        return operation.emitOpError("local pointer offset must be i32");
    }
    int64_t footprint = (shape->cardinality * addressBits + 31) / 32;
    Value lhs = *base;
    Value rhs = *offset;
    Value lhsSource = operation.getBase();
    Value rhsSource = operation.getOffset();
    if (lhs.getDefiningOp<ImmOp>() && !rhs.getDefiningOp<ImmOp>()) {
      std::swap(lhs, rhs);
      std::swap(lhsSource, rhsSource);
    }
    Value result =
        AddOp::create(*builder, *location, reg(footprint), addressType,
                      shape->cardinality, canonicalDestination(),
                      sourceRegion(lhsSource, shape->cardinality, operation),
                      sourceRegion(rhsSource, shape->cardinality, operation),
                      IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                      TypeAttr(), shape->cardinality == 1, 0, lhs, rhs)
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

  FailureOr<int64_t> getPointerBits(Type type, Operation *owner) const {
    if (isLocalPointer(type))
      return 32;
    if (isA64Pointer(type))
      return 64;
    return owner->emitOpError("pointer address space has no Xe representation"),
           failure();
  }

  FailureOr<int64_t> getIntegerBits(Type type, Operation *owner) const {
    Type payload = type;
    if (xw::SimdType simd = dyn_cast<xw::SimdType>(type))
      payload = simd.getElementType();
    IntegerType integer = dyn_cast<IntegerType>(payload);
    if (!integer)
      return owner->emitOpError("pointer cast requires an integer payload"),
             failure();
    return integer.getWidth();
  }

  LogicalResult lowerAddrspaceCast(xw::AddrspaceCastOp operation) {
    bool sourceLocal = isLocalPointer(operation.getSource().getType());
    bool resultLocal = isLocalPointer(operation.getType());
    xw::PtrType sourcePointer =
        getPointerElementType(operation.getSource().getType());
    xw::PtrType resultPointer = getPointerElementType(operation.getType());
    bool sourceGeneric = sourcePointer && isa<xw::GenericAddressSpaceAttr>(
                                              sourcePointer.getAddressSpace());
    bool resultGeneric = resultPointer && isa<xw::GenericAddressSpaceAttr>(
                                              resultPointer.getAddressSpace());
    if ((sourceLocal && resultGeneric) || (sourceGeneric && resultLocal))
      return operation.emitOpError("local and generic address-space casts lack "
                                   "provenance-preserving machine selection");
    FailureOr<int64_t> sourceBits =
        getPointerBits(operation.getSource().getType(), operation);
    FailureOr<int64_t> resultBits =
        getPointerBits(operation.getType(), operation);
    if (failed(sourceBits) || failed(resultBits))
      return failure();
    if (*sourceBits > *resultBits)
      return operation.emitOpError(
          "A64 to local address-space cast would lose pointer bits");
    if (*sourceBits == *resultBits) {
      if (isWideSimd(operation.getType())) {
        FailureOr<WideValue> source =
            getWideValue(operation.getSource(), operation);
        if (failed(source))
          return failure();
        wideValues[operation.getResult()] = *source;
        widePointers[operation.getResult()] = *source;
      } else {
        FailureOr<Value> source = getValue(operation.getSource(), operation);
        if (failed(source))
          return failure();
        values[operation.getResult()] = *source;
      }
      return success();
    }
    FailureOr<ValueShape> shape = getShape(operation.getType(), operation);
    FailureOr<Value> source = getValue(operation.getSource(), operation);
    if (failed(shape) || failed(source))
      return failure();
    if (shape->cardinality == 32) {
      auto extend = [&](int64_t offset) {
        return MovOp::create(*builder, *location, reg(32), i64(), 16,
                             canonicalDestination(), canonicalRegion(),
                             IntegerAttr(), builder->getI32IntegerAttr(offset),
                             typeAttr(i32()), false, offset, *source)
            .getResult();
      };
      WideValue result{extend(0), extend(16)};
      wideValues[operation.getResult()] = result;
      widePointers[operation.getResult()] = result;
      return success();
    }
    values[operation.getResult()] =
        MovOp::create(
            *builder, *location, reg(shape->cardinality * 2), i64(),
            shape->cardinality, canonicalDestination(),
            sourceRegion(operation.getSource(), shape->cardinality, operation),
            IntegerAttr(), IntegerAttr(), typeAttr(i32()),
            shape->cardinality == 1, 0, *source)
            .getResult();
    return success();
  }

  LogicalResult lowerPtrToInt(xw::PtrToIntOp operation) {
    FailureOr<int64_t> pointerBits =
        getPointerBits(operation.getSource().getType(), operation);
    FailureOr<int64_t> integerBits =
        getIntegerBits(operation.getType(), operation);
    if (failed(pointerBits) || failed(integerBits))
      return failure();
    if (*integerBits < *pointerBits)
      return operation.emitOpError("pointer-to-integer cast would lose bits");
    if (*integerBits > 64)
      return operation.emitOpError(
          "pointer-to-integer result wider than 64 bits has no machine move");
    if (*integerBits == *pointerBits) {
      if (isWideSimd(operation.getSource().getType())) {
        FailureOr<WideValue> source =
            getWideValue(operation.getSource(), operation);
        if (failed(source))
          return failure();
        wideValues[operation.getResult()] = *source;
      } else {
        FailureOr<Value> source = getValue(operation.getSource(), operation);
        if (failed(source))
          return failure();
        values[operation.getResult()] = *source;
      }
      return success();
    }
    FailureOr<ValueShape> shape = getShape(operation.getType(), operation);
    FailureOr<Value> source = getValue(operation.getSource(), operation);
    if (failed(shape) || failed(source))
      return failure();
    if (shape->cardinality == 32) {
      auto extend = [&](int64_t offset) {
        return MovOp::create(*builder, *location, reg(32), i64(), 16,
                             canonicalDestination(), canonicalRegion(),
                             IntegerAttr(), builder->getI32IntegerAttr(offset),
                             typeAttr(i32()), false, offset, *source)
            .getResult();
      };
      wideValues[operation.getResult()] = {extend(0), extend(16)};
      return success();
    }
    values[operation.getResult()] =
        MovOp::create(
            *builder, *location, reg(shape->cardinality * 2), i64(),
            shape->cardinality, canonicalDestination(),
            sourceRegion(operation.getSource(), shape->cardinality, operation),
            IntegerAttr(), IntegerAttr(), typeAttr(i32()),
            shape->cardinality == 1, 0, *source)
            .getResult();
    return success();
  }

  LogicalResult lowerIntToPtr(xw::IntToPtrOp operation) {
    FailureOr<int64_t> integerBits =
        getIntegerBits(operation.getSource().getType(), operation);
    FailureOr<int64_t> pointerBits =
        getPointerBits(operation.getType(), operation);
    if (failed(integerBits) || failed(pointerBits))
      return failure();
    if (*integerBits > *pointerBits)
      return operation.emitOpError("integer-to-pointer cast would lose bits");
    if (*integerBits == *pointerBits && isWideSimd(operation.getType())) {
      FailureOr<WideValue> source =
          getWideValue(operation.getSource(), operation);
      if (failed(source))
        return failure();
      wideValues[operation.getResult()] = *source;
      widePointers[operation.getResult()] = *source;
    } else if (*integerBits == *pointerBits) {
      FailureOr<Value> source = getValue(operation.getSource(), operation);
      if (failed(source))
        return failure();
      values[operation.getResult()] = *source;
    } else {
      FailureOr<ValueShape> shape = getShape(operation.getType(), operation);
      FailureOr<Value> source = getValue(operation.getSource(), operation);
      if (failed(shape) || failed(source))
        return failure();
      Type destinationType = *pointerBits == 32 ? i32() : i64();
      if (*pointerBits == 64 && shape->cardinality == 32) {
        auto extend = [&](int64_t offset) {
          return MovOp::create(
                     *builder, *location, reg(32), i64(), 16,
                     canonicalDestination(), canonicalRegion(), IntegerAttr(),
                     builder->getI32IntegerAttr(offset),
                     typeAttr(IntegerType::get(context, *integerBits)), false,
                     offset, *source)
              .getResult();
        };
        WideValue result{extend(0), extend(16)};
        wideValues[operation.getResult()] = result;
        widePointers[operation.getResult()] = result;
        return success();
      }
      int64_t footprint = (shape->cardinality * *pointerBits + 31) / 32;
      values[operation.getResult()] =
          MovOp::create(*builder, *location, reg(footprint), destinationType,
                        shape->cardinality, canonicalDestination(),
                        sourceRegion(operation.getSource(), shape->cardinality,
                                     operation),
                        IntegerAttr(), IntegerAttr(),
                        typeAttr(IntegerType::get(context, *integerBits)),
                        shape->cardinality == 1, 0, *source)
              .getResult();
    }
    return success();
  }

  LogicalResult lowerPointerCompare(xw::PtrCmpOp operation) {
    FailureOr<ValueShape> shape =
        getShape(operation.getLhs().getType(), operation);
    FailureOr<int64_t> bits =
        getPointerBits(operation.getLhs().getType(), operation);
    if (failed(shape) || failed(bits))
      return failure();
    if (*bits == 64 && shape->cardinality == 32)
      return operation.emitOpError(
          "SIMD32 A64 pointer comparison has no decomposed flag selection");
    FailureOr<Value> lhs = getValue(operation.getLhs(), operation);
    FailureOr<Value> rhs = getValue(operation.getRhs(), operation);
    if (failed(lhs) || failed(rhs))
      return failure();
    CmpOp compare = CmpOp::create(
        *builder, *location, ARFType::get(context, ARFFile::f, 2, -1),
        CondModifierAttr::get(context, operation.getPredicate() ==
                                               arith::CmpIPredicate::eq
                                           ? CondModifier::eq
                                           : CondModifier::ne),
        typeAttr(*bits == 32 ? i32() : i64()),
        builder->getI32IntegerAttr(shape->cardinality),
        sourceRegion(operation.getLhs(), shape->cardinality, operation),
        sourceRegion(operation.getRhs(), shape->cardinality, operation),
        IntegerAttr(), IntegerAttr(), TypeAttr(), TypeAttr(), *lhs, *rhs);
    values[operation.getResult()] = compare.getFlag();
    return success();
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

  FailureOr<Value> emitBlock2DScalar(Value source, int64_t dwords,
                                     Operation *operation) {
    FailureOr<ValueShape> shape = getShape(source.getType(), operation);
    FailureOr<Value> selected = getValue(source, operation);
    if (failed(shape) || failed(selected))
      return failure();
    if (!selected->getDefiningOp<ImmOp>())
      return *selected;
    return emitMove(reg(dwords), shape->elementType, 1, *selected, RegionAttr(),
                    true);
  }

  FailureOr<Value> buildBlock2DPayload(Operation *operation, Value base,
                                       Value surfaceWidth, Value surfaceHeight,
                                       Value surfacePitch, Value x, Value y,
                                       int64_t blockWidth, int64_t blockHeight,
                                       int64_t blocks) {
    FailureOr<Value> selectedBase = getValue(base, operation);
    if (failed(selectedBase))
      return failure();
    Value address =
        selectedBase->getDefiningOp<ImmOp>()
            ? emitMove(reg(2), i64(), 1, *selectedBase, RegionAttr(), true)
            : *selectedBase;
    auto subtractOne = [&](Value source) -> FailureOr<Value> {
      FailureOr<Value> selected = getValue(source, operation);
      if (failed(selected))
        return failure();
      if (ImmOp immediateValue = selected->getDefiningOp<ImmOp>())
        return emitMove(reg(1), i32(), 1,
                        immediate(immediateValue.getValue() - 1, i32()),
                        RegionAttr(), true);
      return SubOp::create(*builder, *location, reg(1), i32(), 1,
                           canonicalDestination(), RegionAttr(),
                           uniformRegion(), IntegerAttr(), IntegerAttr(),
                           IntegerAttr(), TypeAttr(), TypeAttr(), true, 0,
                           immediate(1, i32()), *selected)
          .getResult();
    };
    Value zero =
        emitMove(reg(16), i32(), 16, immediate(0, i32()), RegionAttr(), true);
    Value shape =
        emitMove(reg(1), i32(), 1,
                 immediate((blockWidth - 1) | ((blockHeight - 1) << 8) |
                               ((blocks - 1) << 16),
                           i32()),
                 RegionAttr(), true);
    FailureOr<Value> selectedX = emitBlock2DScalar(x, 1, operation);
    FailureOr<Value> selectedY = emitBlock2DScalar(y, 1, operation);
    FailureOr<Value> selectedWidth = subtractOne(surfaceWidth);
    FailureOr<Value> selectedHeight = subtractOne(surfaceHeight);
    FailureOr<Value> selectedPitch = subtractOne(surfacePitch);
    if (failed(selectedX) || failed(selectedY) || failed(selectedWidth) ||
        failed(selectedHeight) || failed(selectedPitch))
      return failure();
    std::array<Value, 7> updates = {
        address,    *selectedWidth, *selectedHeight, *selectedPitch, *selectedX,
        *selectedY, shape};
    SmallVector<Attribute> offsets;
    for (int64_t offset : {0, 2, 3, 4, 5, 6, 7})
      offsets.push_back(builder->getI64IntegerAttr(offset));
    UpdateTupleOp payload =
        UpdateTupleOp::create(*builder, *location, reg(16), zero, updates,
                              builder->getArrayAttr(offsets));
    return payload.getResult();
  }

  LogicalResult lowerBlock2D(Operation *operation, Value base,
                             Value surfaceWidth, Value surfaceHeight,
                             Value surfacePitch, Value x, Value y,
                             int64_t blockWidth, int64_t blockHeight,
                             int64_t blocks, Value data, Value dependency,
                             Value valueResult, Value tokenResult,
                             uint32_t descriptor) {
    FailureOr<Value> payload = buildBlock2DPayload(
        operation, base, surfaceWidth, surfaceHeight, surfacePitch, x, y,
        blockWidth, blockHeight, blocks);
    FailureOr<Value> selectedDependency = mapDependency(operation, dependency);
    if (failed(payload) || failed(selectedDependency))
      return failure();
    Value selectedData;
    if (data) {
      FailureOr<Value> materialized = materialize(data, operation);
      if (failed(materialized))
        return failure();
      selectedData = *materialized;
    }
    int64_t resultDwords = 0;
    if (valueResult) {
      FailureOr<int64_t> footprint =
          getFootprint(valueResult.getType(), operation);
      if (failed(footprint))
        return failure();
      resultDwords = *footprint;
    }
    SendOp send = SendOp::create(*builder, *location, reg(resultDwords),
                                 MemTokenType::get(context), SendFn::ugm, 0,
                                 descriptor, 0, 1, true, false, *payload,
                                 selectedData, Value(), *selectedDependency);
    if (valueResult)
      values[valueResult] = send.getDst();
    Value token = send.getToken();
    if (isa<xw::Block2DPrefetchOp>(operation))
      token = AfterOp::create(*builder, *location, MemTokenType::get(context),
                              token)
                  .getToken();
    values[tokenResult] = token;
    memoryToken = token;
    return success();
  }

  LogicalResult lowerLoad(xw::LoadOp operation) {
    FailureOr<ValueShape> shape =
        getShape(operation.getValue().getType(), operation);
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
      LoadSLMOp load =
          LoadSLMOp::create(*builder, *location, reg(shape->cardinality),
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
        *builder, *location, reg(shape->cardinality),
        MemTokenType::get(context), *address, *dependency, shape->cardinality);
    values[operation.getValue()] = load.getDst();
    values[operation.getToken()] = load.getToken();
    memoryToken = load.getToken();
    return success();
  }

  LogicalResult lowerStore(xw::StoreOp operation) {
    FailureOr<ValueShape> shape =
        getShape(operation.getValue().getType(), operation);
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
      StoreSLMOp store =
          StoreSLMOp::create(*builder, *location, MemTokenType::get(context),
                             *address, *data, *dependency, shape->cardinality);
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
    StoreA64Op store =
        StoreA64Op::create(*builder, *location, MemTokenType::get(context),
                           *address, *data, *dependency, shape->cardinality);
    values[operation.getToken()] = store.getToken();
    memoryToken = store.getToken();
    return success();
  }

  LogicalResult lowerAtomic(xw::AtomicRMWOp operation) {
    if (operation.getKind() != arith::AtomicRMWKind::addi)
      return operation.emitOpError("only atomic add has XeMachine support");
    FailureOr<ValueShape> shape =
        getShape(operation.getOld().getType(), operation);
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
    AtomicIAddA64Op atomic =
        AtomicIAddA64Op::create(*builder, *location, reg(shape->cardinality),
                                MemTokenType::get(context), *address, *data,
                                *dependency, shape->cardinality);
    values[operation.getOld()] = atomic.getDst();
    values[operation.getToken()] = atomic.getToken();
    memoryToken = atomic.getToken();
    return success();
  }

  void emitBarrier(Value dependency) {
    Value inlineData = getInlineDataRegister();
    FenceSLMOp fence =
        FenceSLMOp::create(*builder, *location, reg(16),
                           MemTokenType::get(context), inlineData, dependency);
    FenceAwaitOp await =
        FenceAwaitOp::create(*builder, *location, MemTokenType::get(context),
                             fence.getReadback(), fence.getToken());
    Value payload =
        emitMove(reg(16), i32(), 16, immediate(0, i32()), RegionAttr(), true);
    Value control = MovOp::create(*builder, *location, reg(16), i32(), 1,
                                  canonicalDestination(), RegionAttr(),
                                  builder->getI32IntegerAttr(2), IntegerAttr(),
                                  TypeAttr(), true, 0, immediate(0x100, i32()))
                        .getResult();
    payload = UpdateTupleOp::create(
                  *builder, *location, reg(16), payload, ValueRange{control},
                  builder->getArrayAttr({builder->getI64IntegerAttr(0)}))
                  .getResult();
    Value header = MovOp::create(*builder, *location, reg(16), i8(), 2,
                                 canonicalDestination(), uniformRegion(),
                                 builder->getI32IntegerAttr(10),
                                 builder->getI32IntegerAttr(11), TypeAttr(),
                                 true, 0, inlineData)
                       .getResult();
    payload = UpdateTupleOp::create(
                  *builder, *location, reg(16), payload, ValueRange{header},
                  builder->getArrayAttr({builder->getI64IntegerAttr(0)}))
                  .getResult();
    BarrierSignalOp signal =
        BarrierSignalOp::create(*builder, *location, MemTokenType::get(context),
                                payload, await.getToken());
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
      result = MovOp::create(*builder, *location, reg(simdWidth), i32(),
                             simdWidth, canonicalDestination(),
                             canonicalRegion(), IntegerAttr(), IntegerAttr(),
                             typeAttr(i16()), false, 0, local)
                   .getResult();
    } else {
      Value r0 = architecturalRegister(0);
      Value inlineData = getInlineDataRegister();
      Value accumulator =
          MulOp::create(
              *builder, *location, ARFType::get(context, ARFFile::acc, 16, 0),
              i32(), 1, canonicalDestination(), uniformRegion(),
              uniformRegion(), IntegerAttr(),
              builder->getI32IntegerAttr(
                  KernelABI::get().getGroupIdSubregister(dim)),
              builder->getI32IntegerAttr(
                  KernelABI::get().getImplicitArgumentDword(
                      ImplicitKernelArgument::enqueuedLocalSize, dim)),
              TypeAttr(), TypeAttr(), true, 0, r0, inlineData)
              .getResult();
      Value base =
          emitMove(reg(16), i32(), 1, accumulator, uniformRegion(), true);
      auto lowerHalf = [&](int64_t offset) {
        Value spacedLocal =
            MovOp::create(*builder, *location, reg(32), i16(), 16,
                          DstRegionAttr::get(context, 4), canonicalRegion(),
                          IntegerAttr(), builder->getI32IntegerAttr(offset),
                          TypeAttr(), false, offset, local)
                .getResult();
        if (elementType.isInteger(64)) {
          Value groupLocal =
              AddOp::create(*builder, *location, reg(32), i64(), 16,
                            canonicalDestination(), uniformRegion(),
                            RegionAttr::get(context, 4, 1, 0), IntegerAttr(),
                            IntegerAttr(), IntegerAttr(), typeAttr(i32()),
                            typeAttr(i16()), false, offset, base, spacedLocal)
                  .getResult();
          return AddOp::create(
                     *builder, *location, reg(32), i64(), 16,
                     canonicalDestination(), canonicalRegion(), uniformRegion(),
                     IntegerAttr(), IntegerAttr(),
                     builder->getI32IntegerAttr(
                         KernelABI::get().getImplicitArgumentDword(
                             ImplicitKernelArgument::globalIdOffset, dim)),
                     TypeAttr(), typeAttr(i32()), false, offset, groupLocal,
                     inlineData)
              .getResult();
        }
        Value groupLocal =
            AddOp::create(*builder, *location, reg(16), i32(), 16,
                          canonicalDestination(), uniformRegion(),
                          RegionAttr::get(context, 4, 1, 0), IntegerAttr(),
                          IntegerAttr(), IntegerAttr(), TypeAttr(),
                          typeAttr(i16()), false, offset, base, spacedLocal)
                .getResult();
        return AddOp::create(
                   *builder, *location, reg(16), i32(), 16,
                   canonicalDestination(), canonicalRegion(), uniformRegion(),
                   IntegerAttr(), IntegerAttr(),
                   builder->getI32IntegerAttr(
                       KernelABI::get().getImplicitArgumentDword(
                           ImplicitKernelArgument::globalIdOffset, dim)),
                   TypeAttr(), TypeAttr(), false, offset, groupLocal,
                   inlineData)
            .getResult();
      };
      result = lowerHalf(0);
      if (elementType.isInteger(64)) {
        if (simdWidth == 32)
          wideValues[operation->getResult(0)] = {result, lowerHalf(16)};
        else
          values[operation->getResult(0)] = result;
        return success();
      }
      if (simdWidth == 32)
        result = TupleFromElementsOp::create(*builder, *location, reg(32),
                                             ValueRange{result, lowerHalf(16)})
                     .getTuple();
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
      return MovOp::create(*builder, *location, reg(32), i64(), 16,
                           canonicalDestination(), canonicalRegion(),
                           IntegerAttr(), builder->getI32IntegerAttr(offset),
                           typeAttr(i32()), false, offset, result)
          .getResult();
    };
    wideValues[operation->getResult(0)] = {widen(0), widen(16)};
    return success();
  }

  LogicalResult lowerUniformQuery(Operation *operation, int64_t dim,
                                  int sourceSub, bool inlineData) {
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
    Value source =
        inlineData ? getInlineDataRegister() : architecturalRegister(0);
    values[operation->getResult(0)] =
        MovOp::create(*builder, *location, reg(*footprint), shape->elementType,
                      1, canonicalDestination(), uniformRegion(), IntegerAttr(),
                      builder->getI32IntegerAttr(sourceSub), typeAttr(i32()),
                      true, 0, source)
            .getResult();
    return success();
  }

  LogicalResult lowerLaneId(xw::LaneIdOp operation) {
    return operation.emitOpError(
        "lane ID has no XeMachine channel-index primitive");
  }

  LogicalResult lowerSubgroupId(xw::SubgroupIdOp operation) {
    FailureOr<ValueShape> shape = getShape(operation.getType(), operation);
    FailureOr<int64_t> footprint = getFootprint(operation.getType(), operation);
    if (failed(shape) || failed(footprint))
      return failure();
    if (shape->cardinality != 1 || !shape->elementType.isInteger(32))
      return operation.emitOpError("subgroup ID must be a bare i32 value");
    if (failed(emitPrologue()))
      return failure();
    Value inlineData = getInlineDataRegister();
    auto readLocalId = [&](int64_t axis) {
      return MovOp::create(*builder, *location, reg(1), i32(), 1,
                           canonicalDestination(), uniformRegion(),
                           IntegerAttr(), IntegerAttr(), typeAttr(i16()), true,
                           0, localIds[axis])
          .getResult();
    };
    auto readLocalSize = [&](int64_t axis) {
      return MovOp::create(*builder, *location, reg(1), i32(), 1,
                           canonicalDestination(), uniformRegion(),
                           IntegerAttr(), builder->getI32IntegerAttr(axis),
                           typeAttr(i32()), true, 0, inlineData)
          .getResult();
    };
    auto multiply = [&](Value lhs, Value rhs) {
      Value product = MulOp::create(*builder, *location,
                                    ARFType::get(context, ARFFile::acc, 1, 0),
                                    i32(), 1, canonicalDestination(),
                                    uniformRegion(), uniformRegion(),
                                    IntegerAttr(), IntegerAttr(), IntegerAttr(),
                                    TypeAttr(), TypeAttr(), true, 0, lhs, rhs)
                          .getResult();
      return emitMove(reg(1), i32(), 1, product, uniformRegion(), true);
    };
    auto add = [&](Value lhs, Value rhs) {
      return AddOp::create(
                 *builder, *location, reg(1), i32(), 1, canonicalDestination(),
                 uniformRegion(), uniformRegion(), IntegerAttr(), IntegerAttr(),
                 IntegerAttr(), TypeAttr(), TypeAttr(), true, 0, lhs, rhs)
          .getResult();
    };
    Value linear = readLocalId(0);
    if (subgroupIdAxes[1] || subgroupIdAxes[2]) {
      Value localSizeX = readLocalSize(0);
      if (subgroupIdAxes[1])
        linear = add(linear, multiply(readLocalId(1), localSizeX));
      if (subgroupIdAxes[2]) {
        Value localPlane = multiply(localSizeX, readLocalSize(1));
        linear = add(linear, multiply(readLocalId(2), localPlane));
      }
    }
    values[operation.getResult()] =
        ShrOp::create(*builder, *location, reg(*footprint), i32(), 1,
                      canonicalDestination(), uniformRegion(), RegionAttr(),
                      IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                      TypeAttr(), true, 0, linear,
                      immediate(llvm::Log2_64(simdWidth), i16()))
            .getResult();
    return success();
  }

  LogicalResult lowerShuffle(xw::ShuffleOp operation) {
    xw::ConstantOp lane =
        operation.getSourceLane().getDefiningOp<xw::ConstantOp>();
    if (!lane)
      return operation.emitOpError(
          "dynamic shuffle has no XeMachine indirect-region primitive");
    FailureOr<int64_t> sourceLane = getConstantBits(lane);
    FailureOr<ValueShape> shape = getShape(operation.getType(), operation);
    FailureOr<int64_t> bits = getElementBits(
        cast<xw::SimdType>(operation.getType()).getElementType(), operation);
    FailureOr<Value> source = getValue(operation.getSource(), operation);
    FailureOr<int64_t> footprint = getFootprint(operation.getType(), operation);
    if (failed(sourceLane) || failed(shape) || failed(bits) || failed(source) ||
        failed(footprint))
      return failure();
    if (*sourceLane < 0 || *sourceLane >= shape->cardinality)
      return operation.emitOpError("constant shuffle lane is out of range");
    if (*bits > 32)
      return operation.emitOpError(
          "shuffle payload wider than 32 bits has no direct region selection");
    values[operation.getResult()] = emitMove(
        reg(*footprint), shape->elementType, shape->cardinality, *source,
        uniformRegion(), false, 0, builder->getI32IntegerAttr(*sourceLane));
    return success();
  }

  LogicalResult lowerUnavailableQuery(Operation *operation) {
    return operation->emitOpError(
        "query is absent from the Xe payload contract");
  }

  LogicalResult lowerAllocRelease(xw::AllocReleaseOp operation) {
    if (!isLocalPointer(operation.getAllocation().getType()))
      return operation.emitOpError(
          "allocation release requires a local pointer");
    Value dependency = values.lookup(operation.getDependency());
    if (!dependency)
      return operation.emitOpError("allocation release token was not selected");
    values[operation.getToken()] = dependency;
    memoryToken = dependency;
    return success();
  }

  void emitEot() {
    EotOp::create(*builder, *location, architecturalRegister(0), memoryToken);
  }

  LogicalResult lowerBlock(Block &block) {
    for (Operation &operation : block) {
      if (ub::PoisonOp poison = dyn_cast<ub::PoisonOp>(operation)) {
        if (failed(lowerPoison(poison)))
          return failure();
      } else if (xw::ConstantOp constant =
                     dyn_cast<xw::ConstantOp>(operation)) {
        if (failed(lowerConstant(constant.getResult(), constant.getValue(),
                                 constant)))
          return failure();
      } else if (arith::ConstantOp constant =
                     dyn_cast<arith::ConstantOp>(operation)) {
        if (failed(lowerConstant(constant.getResult(), constant.getValue(),
                                 constant)))
          return failure();
      } else if (arith::XOrIOp xorOperation =
                     dyn_cast<arith::XOrIOp>(operation)) {
        if (failed(lowerArithXor(xorOperation)))
          return failure();
      } else if (arith::ExtUIOp extension =
                     dyn_cast<arith::ExtUIOp>(operation)) {
        if (failed(lowerArithExtUI(extension)))
          return failure();
      } else if (xw::SplatOp splat = dyn_cast<xw::SplatOp>(operation)) {
        if (failed(lowerView(splat, splat.getSource())))
          return failure();
      } else if (xw::ReadFirstOp read = dyn_cast<xw::ReadFirstOp>(operation)) {
        if (failed(lowerView(read, read.getSource())))
          return failure();
      } else if (xw::ExpandOp expand = dyn_cast<xw::ExpandOp>(operation)) {
        if (failed(lowerView(expand, expand.getSource())))
          return failure();
      } else if (xw::FreezeOp freeze = dyn_cast<xw::FreezeOp>(operation)) {
        if (failed(lowerFreeze(freeze)))
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
      } else if (xw::DpasOp dpas = dyn_cast<xw::DpasOp>(operation)) {
        if (failed(lowerDpas(dpas)))
          return failure();
      } else if (xw::BitcastOp bitcast = dyn_cast<xw::BitcastOp>(operation)) {
        if (failed(lowerBitcast(bitcast)))
          return failure();
      } else if (isa<xw::FMaxOp>(operation)) {
        if (failed(lowerUnsupportedFloat(&operation, "floating maximum")))
          return failure();
      } else if (isa<xw::FmaOp>(operation)) {
        if (failed(lowerUnsupportedFloat(&operation, "fused multiply-add")))
          return failure();
      } else if (isa<xw::FExp2Op>(operation)) {
        if (failed(lowerUnsupportedFloat(&operation, "base-two exponential")))
          return failure();
      } else if (isa<xw::FRcpOp>(operation)) {
        if (failed(lowerUnsupportedFloat(&operation, "reciprocal")))
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
      } else if (isa<xw::MaskAndOp, xw::MaskOrOp, xw::MaskXorOp>(operation)) {
        if (failed(lowerMaskBinary(&operation)))
          return failure();
      } else if (xw::MaskNotOp maskNot = dyn_cast<xw::MaskNotOp>(operation)) {
        if (failed(lowerMaskNot(maskNot)))
          return failure();
      } else if (xw::BallotOp ballot = dyn_cast<xw::BallotOp>(operation)) {
        if (failed(lowerBallot(ballot)))
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
      } else if (xw::AddrspaceCastOp pointer =
                     dyn_cast<xw::AddrspaceCastOp>(operation)) {
        if (failed(lowerAddrspaceCast(pointer)))
          return failure();
      } else if (xw::PtrToIntOp pointer = dyn_cast<xw::PtrToIntOp>(operation)) {
        if (failed(lowerPtrToInt(pointer)))
          return failure();
      } else if (xw::IntToPtrOp pointer = dyn_cast<xw::IntToPtrOp>(operation)) {
        if (failed(lowerIntToPtr(pointer)))
          return failure();
      } else if (xw::PtrCmpOp compare = dyn_cast<xw::PtrCmpOp>(operation)) {
        if (failed(lowerPointerCompare(compare)))
          return failure();
      } else if (xw::NullOp null = dyn_cast<xw::NullOp>(operation)) {
        FailureOr<int64_t> bits = getPointerBits(null.getType(), null);
        if (failed(bits))
          return failure();
        values[null.getResult()] = immediate(0, *bits == 32 ? i32() : i64());
      } else if (xw::LocalMemoryBaseOp local =
                     dyn_cast<xw::LocalMemoryBaseOp>(operation)) {
        values[local.getResult()] = immediate(local.getOffset(), i32());
      } else if (xw::AllocOp allocation = dyn_cast<xw::AllocOp>(operation)) {
        values[allocation.getResult()] =
            immediate(allocation.getOffset().value_or(0), i32());
      } else if (xw::AllocReleaseOp release =
                     dyn_cast<xw::AllocReleaseOp>(operation)) {
        if (failed(lowerAllocRelease(release)))
          return failure();
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
          memoryToken =
              AfterOp::create(*builder, *location, MemTokenType::get(context),
                              dependencies)
                  .getToken();
        else
          memoryToken =
              TokenJoinOp::create(*builder, *location,
                                  MemTokenType::get(context), dependencies)
                  .getToken();
        values[operation.getResult(0)] = memoryToken;
      } else if (xw::LoadOp load = dyn_cast<xw::LoadOp>(operation)) {
        if (failed(lowerLoad(load)))
          return failure();
      } else if (xw::StoreOp store = dyn_cast<xw::StoreOp>(operation)) {
        if (failed(lowerStore(store)))
          return failure();
      } else if (xw::Block2DPrefetchOp prefetch =
                     dyn_cast<xw::Block2DPrefetchOp>(operation)) {
        if (prefetch.getElementBits() != 16 || prefetch.getBlockWidth() != 16 ||
            prefetch.getBlockHeight() != 8 || prefetch.getBlocks() != 1 ||
            prefetch.getTranspose() || prefetch.getVnni())
          return prefetch.emitOpError(
              "BMG selection supports only an untransformed 16-bit 8x16 "
              "single-block prefetch");
        if (failed(lowerBlock2D(
                prefetch, prefetch.getBase(), prefetch.getSurfaceWidth(),
                prefetch.getSurfaceHeight(), prefetch.getSurfacePitch(),
                prefetch.getX(), prefetch.getY(), prefetch.getBlockWidth(),
                prefetch.getBlockHeight(), prefetch.getBlocks(), Value(),
                prefetch.getDependency(), Value(), prefetch.getToken(),
                0x02080203)))
          return failure();
      } else if (xw::Block2DReadOp read =
                     dyn_cast<xw::Block2DReadOp>(operation)) {
        bool ordinary = !read.getVnni() && read.getBlockHeight() == 8;
        bool transformed = read.getVnni() && read.getBlockHeight() == 16;
        if (read.getElementBits() != 16 || read.getBlockWidth() != 16 ||
            read.getBlocks() != 1 || read.getTranspose() ||
            (!ordinary && !transformed))
          return read.emitOpError(
              "BMG selection supports only 16-bit 8x16 ordinary or 16x16 "
              "VNNI single-block reads");
        FailureOr<int64_t> footprint =
            getFootprint(read.getValue().getType(), read);
        int64_t expectedDwords = read.getBlockWidth() * read.getBlockHeight() *
                                 read.getElementBits() / 32;
        if (failed(footprint) || *footprint != expectedDwords)
          return read.emitOpError(
              "result packet does not match the selected block2D read");
        uint32_t descriptor = read.getVnni() ? 0x02800283 : 0x02400203;
        if (failed(lowerBlock2D(read, read.getBase(), read.getSurfaceWidth(),
                                read.getSurfaceHeight(), read.getSurfacePitch(),
                                read.getX(), read.getY(), read.getBlockWidth(),
                                read.getBlockHeight(), read.getBlocks(),
                                Value(), read.getDependency(), read.getValue(),
                                read.getToken(), descriptor)))
          return failure();
      } else if (xw::Block2DWriteOp write =
                     dyn_cast<xw::Block2DWriteOp>(operation)) {
        if (write.getElementBits() != 32 || write.getBlockWidth() != 16 ||
            write.getBlockHeight() != 8 || write.getBlocks() != 1 ||
            write.getTranspose() || write.getVnni())
          return write.emitOpError(
              "BMG selection supports only an untransformed 32-bit 8x16 "
              "single-block write");
        FailureOr<int64_t> footprint =
            getFootprint(write.getValue().getType(), write);
        int64_t expectedDwords = write.getBlockWidth() *
                                 write.getBlockHeight() *
                                 write.getElementBits() / 32;
        if (failed(footprint) || *footprint != expectedDwords)
          return write.emitOpError(
              "data packet does not match the selected block2D write");
        if (failed(lowerBlock2D(
                write, write.getBase(), write.getSurfaceWidth(),
                write.getSurfaceHeight(), write.getSurfacePitch(), write.getX(),
                write.getY(), write.getBlockWidth(), write.getBlockHeight(),
                write.getBlocks(), write.getValue(), write.getDependency(),
                Value(), write.getToken(), 0x02000407)))
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
          dependency =
              TokenJoinOp::create(*builder, *location,
                                  MemTokenType::get(context), dependencies)
                  .getToken();
        if (!dependency)
          dependency =
              TokenOp::create(*builder, *location, MemTokenType::get(context))
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
        if (failed(lowerUniformQuery(
                id, id.getDim(),
                KernelABI::get().getGroupIdSubregister(id.getDim()), false)))
          return failure();
      } else if (xw::LocalSizeOp size = dyn_cast<xw::LocalSizeOp>(operation)) {
        if (failed(lowerUniformQuery(
                size, size.getDim(),
                KernelABI::get().getImplicitArgumentDword(
                    ImplicitKernelArgument::enqueuedLocalSize, size.getDim()),
                true)))
          return failure();
      } else if (xw::LaunchBlockSizeOp size =
                     dyn_cast<xw::LaunchBlockSizeOp>(operation)) {
        if (failed(lowerUniformQuery(
                size, size.getDim(),
                KernelABI::get().getImplicitArgumentDword(
                    ImplicitKernelArgument::enqueuedLocalSize, size.getDim()),
                true)))
          return failure();
      } else if (isa<xw::GlobalSizeOp, xw::NumGroupsOp, xw::LaunchGridSizeOp>(
                     operation)) {
        if (failed(lowerUnavailableQuery(&operation)))
          return failure();
      } else if (xw::LaneIdOp lane = dyn_cast<xw::LaneIdOp>(operation)) {
        if (failed(lowerLaneId(lane)))
          return failure();
      } else if (xw::SubgroupIdOp id = dyn_cast<xw::SubgroupIdOp>(operation)) {
        if (failed(lowerSubgroupId(id)))
          return failure();
      } else if (xw::ShuffleOp shuffle = dyn_cast<xw::ShuffleOp>(operation)) {
        if (failed(lowerShuffle(shuffle)))
          return failure();
      } else if (isa<func::ReturnOp>(operation)) {
        emitEot();
      } else if (operation.getName().getDialectNamespace() == "xw") {
        return operation.emitOpError(
            "unsupported semantic XW operation during XeMachine selection");
      } else if (operation.getName().getDialectNamespace() == "ub") {
        return operation.emitOpError(
            "selector accepts only fully poisoned ub.poison operations");
      } else {
        return operation.emitOpError(
            "selector accepts only func, scf, selected arith, and XW "
            "operations");
      }
    }
    return success();
  }
};

} // namespace
