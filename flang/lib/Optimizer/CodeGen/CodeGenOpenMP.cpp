//===-- CodeGenOpenMP.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/CodeGenOpenMP.h"

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/LowLevelIntrinsics.h"
#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace fir;

#define DEBUG_TYPE "flang-codegen-openmp"

// fir::LLVMTypeConverter for converting to LLVM IR dialect types.
#include "flang/Optimizer/CodeGen/TypeConverter.h"

namespace {
/// A pattern that converts the region arguments in a single-region OpenMP
/// operation to the LLVM dialect. The body of the region is not modified and is
/// expected to either be processed by the conversion infrastructure or already
/// contain ops compatible with LLVM dialect types.
template <typename OpType>
class OpenMPFIROpConversion : public mlir::ConvertOpToLLVMPattern<OpType> {
public:
  explicit OpenMPFIROpConversion(const fir::LLVMTypeConverter &lowering)
      : mlir::ConvertOpToLLVMPattern<OpType>(lowering) {}

  const fir::LLVMTypeConverter &lowerTy() const {
    return *static_cast<const fir::LLVMTypeConverter *>(
        this->getTypeConverter());
  }
};

// FIR Op specific conversion for MapInfoOp that overwrites the default OpenMP
// Dialect lowering, this allows FIR specific lowering of types, required for
// descriptors of allocatables currently.
struct MapInfoOpConversion
    : public OpenMPFIROpConversion<mlir::omp::MapInfoOp> {
  using OpenMPFIROpConversion::OpenMPFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::omp::MapInfoOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const mlir::TypeConverter *converter = getTypeConverter();
    llvm::SmallVector<mlir::Type> resTypes;
    if (failed(converter->convertTypes(curOp->getResultTypes(), resTypes)))
      return mlir::failure();

    llvm::SmallVector<mlir::NamedAttribute> newAttrs;
    mlir::omp::MapInfoOp newOp;
    for (mlir::NamedAttribute attr : curOp->getAttrs()) {
      if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr.getValue())) {
        mlir::Type newAttr;
        if (fir::isTypeWithDescriptor(typeAttr.getValue())) {
          newAttr = lowerTy().convertBoxTypeAsStruct(
              mlir::cast<fir::BaseBoxType>(typeAttr.getValue()));
        } else {
          newAttr = converter->convertType(typeAttr.getValue());
        }
        newAttrs.emplace_back(attr.getName(), mlir::TypeAttr::get(newAttr));
      } else {
        newAttrs.push_back(attr);
      }
    }

    rewriter.replaceOpWithNewOp<mlir::omp::MapInfoOp>(
        curOp, resTypes, adaptor.getOperands(), newAttrs);

    return mlir::success();
  }
};

// FIR op specific conversion for PrivateClauseOp that overwrites the default
// OpenMP Dialect lowering, this allows FIR-aware lowering of types, required
// for boxes because the OpenMP dialect conversion doesn't know anything about
// FIR types.
struct PrivateClauseOpConversion
    : public OpenMPFIROpConversion<mlir::omp::PrivateClauseOp> {
  using OpenMPFIROpConversion::OpenMPFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::omp::PrivateClauseOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const fir::LLVMTypeConverter &converter = lowerTy();
    mlir::Type convertedAllocType;
    if (auto box = mlir::dyn_cast<fir::BaseBoxType>(curOp.getType())) {
      // In LLVM codegen fir.box<> == fir.ref<fir.box<>> == llvm.ptr
      // Here we really do want the actual structure
      if (box.isAssumedRank())
        TODO(curOp->getLoc(), "Privatize an assumed rank array");
      unsigned rank = 0;
      if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(
              fir::unwrapRefType(box.getEleTy())))
        rank = seqTy.getShape().size();
      convertedAllocType = converter.convertBoxTypeAsStruct(box, rank);
    } else {
      convertedAllocType = converter.convertType(adaptor.getType());
    }
    if (!convertedAllocType)
      return mlir::failure();
    rewriter.startOpModification(curOp);
    curOp.setType(convertedAllocType);
    rewriter.finalizeOpModification(curOp);
    return mlir::success();
  }
};

static mlir::LLVM::LLVMFuncOp getOmpTargetAlloc(mlir::Operation *op) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  if (mlir::LLVM::LLVMFuncOp mallocFunc =
          module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("omp_target_alloc"))
    return mallocFunc;
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto i64Ty = mlir::IntegerType::get(module->getContext(), 64);
  auto i32Ty = mlir::IntegerType::get(module->getContext(), 32);
  return moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
      moduleBuilder.getUnknownLoc(), "omp_target_alloc",
      mlir::LLVM::LLVMFunctionType::get(
          mlir::LLVM::LLVMPointerType::get(module->getContext()),
          {i64Ty, i32Ty},
          /*isVarArg=*/false));
}

static mlir::Type convertObjectType(const fir::LLVMTypeConverter &converter,
                                    mlir::Type firType) {
  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(firType))
    return converter.convertBoxTypeAsStruct(boxTy);
  return converter.convertType(firType);
}

static llvm::SmallVector<mlir::NamedAttribute>
addLLVMOpBundleAttrs(mlir::ConversionPatternRewriter &rewriter,
                     llvm::ArrayRef<mlir::NamedAttribute> attrs,
                     int32_t numCallOperands) {
  llvm::SmallVector<mlir::NamedAttribute> newAttrs;
  newAttrs.reserve(attrs.size() + 2);

  for (mlir::NamedAttribute attr : attrs) {
    if (attr.getName() != "operandSegmentSizes")
      newAttrs.push_back(attr);
  }

  newAttrs.push_back(rewriter.getNamedAttr(
      "operandSegmentSizes",
      rewriter.getDenseI32ArrayAttr({numCallOperands, 0})));
  newAttrs.push_back(rewriter.getNamedAttr("op_bundle_sizes",
                                           rewriter.getDenseI32ArrayAttr({})));
  return newAttrs;
}

static mlir::LLVM::ConstantOp
genConstantIndex(mlir::Location loc, mlir::Type ity,
                 mlir::ConversionPatternRewriter &rewriter,
                 std::int64_t offset) {
  auto cattr = rewriter.getI64IntegerAttr(offset);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, ity, cattr);
}

static mlir::Value
computeElementDistance(mlir::Location loc, mlir::Type llvmObjectType,
                       mlir::Type idxTy,
                       mlir::ConversionPatternRewriter &rewriter,
                       const mlir::DataLayout &dataLayout) {
  llvm::TypeSize size = dataLayout.getTypeSize(llvmObjectType);
  unsigned short alignment = dataLayout.getTypeABIAlignment(llvmObjectType);
  std::int64_t distance = llvm::alignTo(size, alignment);
  return genConstantIndex(loc, idxTy, rewriter, distance);
}

static mlir::Value genTypeSizeInBytes(mlir::Location loc, mlir::Type idxTy,
                                      mlir::ConversionPatternRewriter &rewriter,
                                      mlir::Type llTy,
                                      const mlir::DataLayout &dataLayout) {
  return computeElementDistance(loc, llTy, idxTy, rewriter, dataLayout);
}

template <typename OP>
static mlir::Value
genAllocationScaleSize(OP op, mlir::Type ity,
                       mlir::ConversionPatternRewriter &rewriter) {
  mlir::Location loc = op.getLoc();
  mlir::Type dataTy = op.getInType();
  auto seqTy = mlir::dyn_cast<fir::SequenceType>(dataTy);
  fir::SequenceType::Extent constSize = 1;
  if (seqTy) {
    int constRows = seqTy.getConstantRows();
    const fir::SequenceType::ShapeRef &shape = seqTy.getShape();
    if (constRows != static_cast<int>(shape.size())) {
      for (auto extent : shape) {
        if (constRows-- > 0)
          continue;
        if (extent != fir::SequenceType::getUnknownExtent())
          constSize *= extent;
      }
    }
  }

  if (constSize != 1) {
    mlir::Value constVal{
        genConstantIndex(loc, ity, rewriter, constSize).getResult()};
    return constVal;
  }
  return nullptr;
}

static mlir::Value integerCast(const fir::LLVMTypeConverter &converter,
                               mlir::Location loc,
                               mlir::ConversionPatternRewriter &rewriter,
                               mlir::Type ty, mlir::Value val,
                               bool fold = false) {
  auto valTy = val.getType();
  // If the value was not yet lowered, lower its type so that it can
  // be used in getPrimitiveTypeSizeInBits.
  if (!mlir::isa<mlir::IntegerType>(valTy))
    valTy = converter.convertType(valTy);
  auto toSize = mlir::LLVM::getPrimitiveTypeSizeInBits(ty);
  auto fromSize = mlir::LLVM::getPrimitiveTypeSizeInBits(valTy);
  if (fold) {
    if (toSize < fromSize)
      return rewriter.createOrFold<mlir::LLVM::TruncOp>(loc, ty, val);
    if (toSize > fromSize)
      return rewriter.createOrFold<mlir::LLVM::SExtOp>(loc, ty, val);
  } else {
    if (toSize < fromSize)
      return rewriter.create<mlir::LLVM::TruncOp>(loc, ty, val);
    if (toSize > fromSize)
      return rewriter.create<mlir::LLVM::SExtOp>(loc, ty, val);
  }
  return val;
}

// FIR Op specific conversion for TargetAllocMemOp
struct TargetAllocMemOpConversion
    : public OpenMPFIROpConversion<mlir::omp::TargetAllocMemOp> {
  using OpenMPFIROpConversion::OpenMPFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::omp::TargetAllocMemOp allocmemOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type heapTy = allocmemOp.getAllocatedType();
    mlir::LLVM::LLVMFuncOp mallocFunc = getOmpTargetAlloc(allocmemOp);
    mlir::Location loc = allocmemOp.getLoc();
    auto ity = lowerTy().indexType();
    mlir::Type dataTy = fir::unwrapRefType(heapTy);
    mlir::Type llvmObjectTy = convertObjectType(lowerTy(), dataTy);
    mlir::Type llvmPtrTy =
        mlir::LLVM::LLVMPointerType::get(allocmemOp.getContext(), 0);
    if (fir::isRecordWithTypeParameters(fir::unwrapSequenceType(dataTy)))
      TODO(loc, "omp.target_allocmem codegen of derived type with length "
                "parameters");
    mlir::Value size = genTypeSizeInBytes(loc, ity, rewriter, llvmObjectTy,
                                          lowerTy().getDataLayout());
    if (auto scaleSize = genAllocationScaleSize(allocmemOp, ity, rewriter))
      size = rewriter.create<mlir::LLVM::MulOp>(loc, ity, size, scaleSize);
    for (mlir::Value opnd : adaptor.getOperands().drop_front())
      size = rewriter.create<mlir::LLVM::MulOp>(
          loc, ity, size, integerCast(lowerTy(), loc, rewriter, ity, opnd));
    auto mallocTyWidth = lowerTy().getIndexTypeBitwidth();
    auto mallocTy =
        mlir::IntegerType::get(rewriter.getContext(), mallocTyWidth);
    if (mallocTyWidth != ity.getIntOrFloatBitWidth())
      size = integerCast(lowerTy(), loc, rewriter, mallocTy, size);
    allocmemOp->setAttr("callee", mlir::SymbolRefAttr::get(mallocFunc));
    auto callOp = rewriter.create<mlir::LLVM::CallOp>(
        loc, llvmPtrTy,
        mlir::SmallVector<mlir::Value, 2>({size, allocmemOp.getDevice()}),
        addLLVMOpBundleAttrs(rewriter, allocmemOp->getAttrs(), 2));
    rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(
        allocmemOp, rewriter.getIntegerType(64), callOp.getResult());
    return mlir::success();
  }
};
} // namespace

void fir::populateOpenMPFIRToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<MapInfoOpConversion>(converter);
  patterns.add<PrivateClauseOpConversion>(converter);
  patterns.add<TargetAllocMemOpConversion>(converter);
}
