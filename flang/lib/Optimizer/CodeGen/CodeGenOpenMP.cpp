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
} // namespace

void fir::populateOpenMPFIRToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<MapInfoOpConversion>(converter);
}
