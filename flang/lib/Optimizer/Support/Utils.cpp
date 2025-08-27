//===-- Utils.cpp ---------------------------------------------------------===//
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

#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InternalNames.h"

fir::TypeInfoOp fir::lookupTypeInfoOp(fir::RecordType recordType,
                                      mlir::ModuleOp module,
                                      const mlir::SymbolTable *symbolTable) {
  // fir.type_info was created with the mangled name of the derived type.
  // It is the same as the name in the related fir.type, except when a pass
  // lowered the fir.type (e.g., when lowering fir.boxproc type if the type has
  // pointer procedure components), in which case suffix may have been added to
  // the fir.type name. Get rid of them when looking up for the fir.type_info.
  llvm::StringRef originalMangledTypeName =
      fir::NameUniquer::dropTypeConversionMarkers(recordType.getName());
  return fir::lookupTypeInfoOp(originalMangledTypeName, module, symbolTable);
}

fir::TypeInfoOp fir::lookupTypeInfoOp(llvm::StringRef name,
                                      mlir::ModuleOp module,
                                      const mlir::SymbolTable *symbolTable) {
  if (symbolTable)
    if (auto typeInfo = symbolTable->lookup<fir::TypeInfoOp>(name))
      return typeInfo;
  return module.lookupSymbol<fir::TypeInfoOp>(name);
}

std::optional<llvm::ArrayRef<int64_t>> fir::getComponentLowerBoundsIfNonDefault(
    fir::RecordType recordType, llvm::StringRef component,
    mlir::ModuleOp module, const mlir::SymbolTable *symbolTable) {
  fir::TypeInfoOp typeInfo =
      fir::lookupTypeInfoOp(recordType, module, symbolTable);
  if (!typeInfo || typeInfo.getComponentInfo().empty())
    return std::nullopt;
  for (auto componentInfo :
       typeInfo.getComponentInfo().getOps<fir::DTComponentOp>())
    if (componentInfo.getName() == component)
      return componentInfo.getLowerBounds();
  return std::nullopt;
}

mlir::LLVM::ConstantOp
fir::genConstantIndex(mlir::Location loc, mlir::Type ity,
                      mlir::ConversionPatternRewriter &rewriter,
                      std::int64_t offset) {
  auto cattr = rewriter.getI64IntegerAttr(offset);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, ity, cattr);
}

mlir::Value
fir::computeElementDistance(mlir::Location loc, mlir::Type llvmObjectType,
                            mlir::Type idxTy,
                            mlir::ConversionPatternRewriter &rewriter,
                            const mlir::DataLayout &dataLayout) {
  llvm::TypeSize size = dataLayout.getTypeSize(llvmObjectType);
  unsigned short alignment = dataLayout.getTypeABIAlignment(llvmObjectType);
  std::int64_t distance = llvm::alignTo(size, alignment);
  return fir::genConstantIndex(loc, idxTy, rewriter, distance);
}

mlir::Value
fir::genAllocationScaleSize(mlir::Location loc, mlir::Type dataTy,
                            mlir::Type ity,
                            mlir::ConversionPatternRewriter &rewriter) {
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
        fir::genConstantIndex(loc, ity, rewriter, constSize).getResult()};
    return constVal;
  }
  return nullptr;
}

mlir::Value fir::integerCast(const fir::LLVMTypeConverter &converter,
                             mlir::Location loc,
                             mlir::ConversionPatternRewriter &rewriter,
                             mlir::Type ty, mlir::Value val, bool fold) {
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
