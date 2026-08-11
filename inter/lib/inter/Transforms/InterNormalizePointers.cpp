// Normalize typed LLVM GEPs to opaque pointers plus byte offsets.

#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/Support/MathExtras.h"

namespace inter {
#define GEN_PASS_DEF_NORMALIZEPOINTERS
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;

namespace {

struct NormalizePointers
    : public inter::impl::NormalizePointersBase<NormalizePointers> {
  void runOnOperation() override {
    func::FuncOp kernel = getOperation();
    if (!kernel->hasAttr("xemachine.kernel"))
      return;

    SmallVector<LLVM::GEPOp> geps;
    kernel.walk([&](LLVM::GEPOp gep) { geps.push_back(gep); });
    for (LLVM::GEPOp gep : geps) {
      if (failed(convertGEP(gep)))
        return signalPassFailure();
    }
  }

  FailureOr<uint64_t> getFixedSize(LLVM::GEPOp gep, const DataLayout &layout,
                                   Type type, bool useABIAlignment = true) {
    llvm::TypeSize size = layout.getTypeSize(type);
    if (size.isScalable())
      return gep.emitOpError("cannot normalize a GEP over a scalable type"),
             failure();
    uint64_t fixedSize = size.getFixedValue();
    if (useABIAlignment)
      fixedSize = llvm::alignTo(fixedSize, layout.getTypeABIAlignment(type));
    return fixedSize;
  }

  FailureOr<uint64_t> getStructFieldOffset(LLVM::GEPOp gep,
                                           const DataLayout &layout,
                                           LLVM::LLVMStructType type,
                                           unsigned field) {
    ArrayRef<Type> body = type.getBody();
    if (field >= body.size())
      return gep.emitOpError("has an out-of-range struct index"), failure();

    uint64_t offset = 0;
    for (unsigned index : llvm::seq(field)) {
      Type element = body[index];
      if (!type.isPacked())
        offset = llvm::alignTo(offset, layout.getTypeABIAlignment(element));
      FailureOr<uint64_t> size = getFixedSize(gep, layout, element,
                                              /*useABIAlignment=*/false);
      if (failed(size))
        return failure();
      offset += *size;
    }
    if (!type.isPacked())
      offset = llvm::alignTo(offset, layout.getTypeABIAlignment(body[field]));
    return offset;
  }

  LLVM::IntegerOverflowFlags getIntegerFlags(LLVM::GEPOp gep) {
    LLVM::IntegerOverflowFlags flags = LLVM::IntegerOverflowFlags::none;
    LLVM::GEPNoWrapFlags gepFlags = gep.getNoWrapFlags();
    if (LLVM::bitEnumContainsAny(gepFlags, LLVM::GEPNoWrapFlags::nusw))
      flags = flags | LLVM::IntegerOverflowFlags::nsw;
    if (LLVM::bitEnumContainsAny(gepFlags, LLVM::GEPNoWrapFlags::nuw))
      flags = flags | LLVM::IntegerOverflowFlags::nuw;
    return flags;
  }

  Value getIndexValue(OpBuilder &builder, LLVM::GEPOp gep,
                      PointerUnion<IntegerAttr, Value> index,
                      IntegerType indexType, LLVM::IntegerOverflowFlags flags) {
    OpFoldResult foldResult = isa<IntegerAttr>(index)
                                  ? OpFoldResult(cast<IntegerAttr>(index))
                                  : OpFoldResult(cast<Value>(index));
    if (std::optional<std::pair<APInt, bool>> constant =
            getConstantAPIntValue(foldResult)) {
      APInt value = constant->first.sextOrTrunc(indexType.getWidth());
      return LLVM::ConstantOp::create(builder, gep.getLoc(), indexType,
                                      builder.getIntegerAttr(indexType, value));
    }

    Value value = cast<Value>(index);
    auto integerType = dyn_cast<IntegerType>(value.getType());
    if (!integerType)
      return nullptr;
    if (integerType.getWidth() < indexType.getWidth())
      return LLVM::SExtOp::create(builder, gep.getLoc(), indexType, value);
    if (integerType.getWidth() > indexType.getWidth())
      return LLVM::TruncOp::create(builder, gep.getLoc(), indexType, value,
                                   flags);
    return value;
  }

  Value getConstant(OpBuilder &builder, Location loc, IntegerType type,
                    uint64_t value) {
    return LLVM::ConstantOp::create(
        builder, loc, type,
        builder.getIntegerAttr(type, APInt(type.getWidth(), value)));
  }

  LogicalResult addScaledIndex(OpBuilder &builder, LLVM::GEPOp gep,
                               PointerUnion<IntegerAttr, Value> index,
                               uint64_t stride, IntegerType indexType,
                               LLVM::IntegerOverflowFlags flags,
                               Value &offset) {
    OpFoldResult foldResult = isa<IntegerAttr>(index)
                                  ? OpFoldResult(cast<IntegerAttr>(index))
                                  : OpFoldResult(cast<Value>(index));
    if (isConstantIntValue(foldResult, 0))
      return success();
    Value term = getIndexValue(builder, gep, index, indexType, flags);
    if (!term)
      return gep.emitOpError("requires integer GEP indices"), failure();
    if (stride != 1) {
      Value scale = getConstant(builder, gep.getLoc(), indexType, stride);
      term = LLVM::MulOp::create(builder, gep.getLoc(), term, scale, flags);
    }
    offset =
        offset ? LLVM::AddOp::create(builder, gep.getLoc(), offset, term, flags)
               : term;
    return success();
  }

  LogicalResult addConstantOffset(OpBuilder &builder, LLVM::GEPOp gep,
                                  uint64_t value, IntegerType indexType,
                                  LLVM::IntegerOverflowFlags flags,
                                  Value &offset) {
    if (value == 0)
      return success();
    Value term = getConstant(builder, gep.getLoc(), indexType, value);
    offset =
        offset ? LLVM::AddOp::create(builder, gep.getLoc(), offset, term, flags)
               : term;
    return success();
  }

  LogicalResult convertGEP(LLVM::GEPOp gep) {
    auto pointerType = dyn_cast<LLVM::LLVMPointerType>(gep.getBase().getType());
    if (!pointerType || !isa<LLVM::LLVMPointerType>(gep.getType()))
      return gep.emitOpError("requires scalar opaque pointers"), failure();

    DataLayout layout = DataLayout::closest(gep);
    std::optional<uint64_t> indexWidth =
        layout.getTypeIndexBitwidth(pointerType);
    if (!indexWidth || *indexWidth == 0)
      return gep.emitOpError("cannot determine pointer index width"), failure();
    IntegerType indexType =
        IntegerType::get(gep.getContext(), static_cast<unsigned>(*indexWidth));
    LLVM::IntegerOverflowFlags flags = getIntegerFlags(gep);
    OpBuilder builder(gep);
    Value offset;
    Type currentType = gep.getElemType();

    for (auto [position, index] : llvm::enumerate(gep.getIndices())) {
      if (position == 0) {
        FailureOr<uint64_t> stride = getFixedSize(gep, layout, currentType);
        if (failed(stride) ||
            failed(addScaledIndex(builder, gep, index, *stride, indexType,
                                  flags, offset)))
          return failure();
        continue;
      }

      if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(currentType)) {
        currentType = arrayType.getElementType();
        FailureOr<uint64_t> stride = getFixedSize(gep, layout, currentType);
        if (failed(stride) ||
            failed(addScaledIndex(builder, gep, index, *stride, indexType,
                                  flags, offset)))
          return failure();
        continue;
      }
      if (auto structType = dyn_cast<LLVM::LLVMStructType>(currentType)) {
        OpFoldResult foldResult = isa<IntegerAttr>(index)
                                      ? OpFoldResult(cast<IntegerAttr>(index))
                                      : OpFoldResult(cast<Value>(index));
        std::optional<int64_t> field = getConstantIntValue(foldResult);
        if (!field || *field < 0)
          return gep.emitOpError("requires a constant non-negative struct "
                                 "index"),
                 failure();
        FailureOr<uint64_t> fieldOffset = getStructFieldOffset(
            gep, layout, structType, static_cast<unsigned>(*field));
        if (failed(fieldOffset) ||
            failed(addConstantOffset(builder, gep, *fieldOffset, indexType,
                                     flags, offset)))
          return failure();
        currentType = structType.getBody()[*field];
        continue;
      }
      if (auto vectorType = dyn_cast<VectorType>(currentType)) {
        currentType = vectorType.getElementType();
        FailureOr<uint64_t> stride =
            getFixedSize(gep, layout, currentType, /*useABIAlignment=*/false);
        if (failed(stride) ||
            failed(addScaledIndex(builder, gep, index, *stride, indexType,
                                  flags, offset)))
          return failure();
        continue;
      }
      gep.emitOpError("cannot index into type ") << currentType;
      return failure();
    }

    if (!offset)
      offset = getConstant(builder, gep.getLoc(), indexType, 0);
    uint32_t gepFlags = static_cast<uint32_t>(gep.getNoWrapFlags());
    xw::PtrAddOp ptrAdd = xw::PtrAddOp::create(
        builder, gep.getLoc(), gep.getType(), gep.getBase(), offset,
        builder.getI32IntegerAttr(gepFlags));
    gep.replaceAllUsesWith(ptrAdd.getResult());
    gep.erase();
    return success();
  }
};

} // namespace
