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

  mlir::omp::MapBoundsOp
  createBoundsForCharString(mlir::ConversionPatternRewriter &rewriter,
                            unsigned int len, mlir::Location loc) const {
    mlir::Type i64Ty = rewriter.getIntegerType(64);
    auto lBound = mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);
    auto uBoundAndExt =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, len - 1);
    auto stride = mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1);
    auto baseLb = mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1);
    auto mapBoundType = rewriter.getType<mlir::omp::MapBoundsType>();
    return mlir::omp::MapBoundsOp::create(rewriter, loc, mapBoundType, lBound,
                                          uBoundAndExt, uBoundAndExt, stride,
                                          /*strideInBytes*/ false, baseLb);
  }

  llvm::LogicalResult
  matchAndRewrite(mlir::omp::MapInfoOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const mlir::TypeConverter *converter = getTypeConverter();
    llvm::SmallVector<mlir::Type> resTypes;
    if (failed(converter->convertTypes(curOp->getResultTypes(), resTypes)))
      return mlir::failure();

    llvm::SmallVector<mlir::NamedAttribute> newAttrs;
    mlir::omp::MapBoundsOp mapBoundsOp;
    for (mlir::NamedAttribute attr : curOp->getAttrs()) {
      if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr.getValue())) {
        mlir::Type newAttr;
        if (fir::isTypeWithDescriptor(typeAttr.getValue())) {
          newAttr = lowerTy().convertBoxTypeAsStruct(
              mlir::cast<fir::BaseBoxType>(typeAttr.getValue()));
        } else if (fir::isa_char_string(fir::unwrapSequenceType(
                       fir::unwrapPassByRefType(typeAttr.getValue()))) &&
                   !characterWithDynamicLen(
                       fir::unwrapPassByRefType(typeAttr.getValue()))) {
          // Characters with a LEN param are represented as char
          // arrays/strings, the initial lowering doesn't generate
          // bounds for these, however, we require them to map the
          // data appropriately in the later lowering stages. This
          // is to prevent the need for unecessary caveats
          // specific to Flang. We also strip the array from the
          // type so that all variations of strings are treated
          // identically and there's no caveats or specialisations
          // required in the later stages. As an example, Boxed
          // char strings will emit a single char array no matter
          // the number of dimensions caused by additional array
          // dimensions which needs specialised for, as it differs
          // from the non-box variation which will emit each array
          // wrapping the character array, e.g. given a type of
          // the same dimensions, if one is boxed, the types would
          // end up:
          //
          //     array<i8 x 16>
          //  vs
          //     array<10 x array< 10 x array<i8 x 16>>>
          //
          // This means we have to treat one specially in the
          // lowering. So we try to "canonicalize" it here.
          // TODO: Handle dynamic LEN characters.
          if (auto ct = mlir::dyn_cast_or_null<fir::CharacterType>(
                  fir::unwrapSequenceType(typeAttr.getValue()))) {
            newAttr = converter->convertType(
                fir::unwrapSequenceType(typeAttr.getValue()));
            if (auto type = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(newAttr))
              newAttr = type.getElementType();
            // We do not generate for device, as MapBoundsOps are
            // unsupported, as they're currently unused.
            auto offloadMod =
                llvm::dyn_cast_or_null<mlir::omp::OffloadModuleInterface>(
                    *curOp->getParentOfType<mlir::ModuleOp>());
            if (!offloadMod.getIsTargetDevice())
              mapBoundsOp = createBoundsForCharString(rewriter, ct.getLen(),
                                                      curOp.getLoc());
          } else {
            newAttr = converter->convertType(typeAttr.getValue());
          }
        } else {
          newAttr = converter->convertType(typeAttr.getValue());
        }
        newAttrs.emplace_back(attr.getName(), mlir::TypeAttr::get(newAttr));
      } else {
        newAttrs.push_back(attr);
      }
    }

    auto newOp = rewriter.replaceOpWithNewOp<mlir::omp::MapInfoOp>(
        curOp, resTypes, adaptor.getOperands(), newAttrs);
    if (mapBoundsOp) {
      rewriter.startOpModification(newOp);
      newOp.getBoundsMutable().append(mlir::ValueRange{mapBoundsOp});
      rewriter.finalizeOpModification(newOp);
    }

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
} // namespace

void fir::populateOpenMPFIRToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<MapInfoOpConversion>(converter);
  patterns.add<PrivateClauseOpConversion>(converter);
}
