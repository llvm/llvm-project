//===-- CodeGenOpenMP.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
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
#include "flang/Optimizer/Support/Utils.h"
#include "aiir/Conversion/LLVMCommon/ConversionTarget.h"
#include "aiir/Conversion/LLVMCommon/Pattern.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Transforms/DialectConversion.h"

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
class OpenMPFIROpConversion : public aiir::ConvertOpToLLVMPattern<OpType> {
public:
  explicit OpenMPFIROpConversion(const fir::LLVMTypeConverter &lowering)
      : aiir::ConvertOpToLLVMPattern<OpType>(lowering) {}

  const fir::LLVMTypeConverter &lowerTy() const {
    return *static_cast<const fir::LLVMTypeConverter *>(
        this->getTypeConverter());
  }
};

// FIR Op specific conversion for MapInfoOp that overwrites the default OpenMP
// Dialect lowering, this allows FIR specific lowering of types, required for
// descriptors of allocatables currently.
struct MapInfoOpConversion
    : public OpenMPFIROpConversion<aiir::omp::MapInfoOp> {
  using OpenMPFIROpConversion::OpenMPFIROpConversion;

  aiir::omp::MapBoundsOp
  createBoundsForCharString(aiir::ConversionPatternRewriter &rewriter,
                            unsigned int len, aiir::Location loc) const {
    aiir::Type i64Ty = rewriter.getIntegerType(64);
    auto lBound = aiir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);
    auto uBoundAndExt =
        aiir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, len - 1);
    auto stride = aiir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1);
    auto baseLb = aiir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1);
    auto mapBoundType = rewriter.getType<aiir::omp::MapBoundsType>();
    return aiir::omp::MapBoundsOp::create(rewriter, loc, mapBoundType, lBound,
                                          uBoundAndExt, uBoundAndExt, stride,
                                          /*strideInBytes*/ false, baseLb);
  }

  llvm::LogicalResult
  matchAndRewrite(aiir::omp::MapInfoOp curOp, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    const aiir::TypeConverter *converter = getTypeConverter();
    llvm::SmallVector<aiir::Type> resTypes;
    if (failed(converter->convertTypes(curOp->getResultTypes(), resTypes)))
      return aiir::failure();

    llvm::SmallVector<aiir::NamedAttribute> newAttrs;
    aiir::omp::MapBoundsOp mapBoundsOp;
    for (aiir::NamedAttribute attr : curOp->getAttrs()) {
      if (auto typeAttr = aiir::dyn_cast<aiir::TypeAttr>(attr.getValue())) {
        aiir::Type newAttr;
        if (fir::isTypeWithDescriptor(typeAttr.getValue())) {
          newAttr = lowerTy().convertBoxTypeAsStruct(
              aiir::cast<fir::BaseBoxType>(typeAttr.getValue()));
        } else if (fir::isa_char_string(fir::unwrapSequenceType(
                       fir::unwrapPassByRefType(typeAttr.getValue()))) &&
                   !characterWithDynamicLen(
                       fir::unwrapPassByRefType(typeAttr.getValue()))) {
          // Characters with a LEN param are represented as strings
          // (array of characters), the lowering to LLVM dialect
          // doesn't generate bounds for these (and this is not
          // done at the initial lowering either) and there is
          // minor inconsistencies in the variable types we
          // create for the map without this step when converting
          // to the LLVM dialect.
          //
          // For example, given the types:
          //
          //  1) CHARACTER(LEN=16), dimension(:,:), allocatable :: char_arr
          //  2) CHARACTER(LEN=16), dimension(10,10) :: char_arr
          //
          // We get the FIR types (note for 1: we already peeled off the
          // dynamic extents from the type at this stage, but the conversion
          // to llvm dialect does that in any case, so the final result
          // is the same):
          //
          //  1) !fir.char<1,16>
          //  2) !fir.array<10x10x!fir.char<1,16>>
          //
          // Which are converted to the LLVM dialect types:
          //
          // 1) !llvm.array<16 x i8>
          // 2) llvm.array<10 x array<10 x array<16 x i8>>
          //
          // And in both cases, we are missing the innermost bounds for
          // the !fir.char<1,16> which is expanded into a 16 x i8 array
          // in the conversion to LLVM dialect.
          //
          // The problem with this is that we would like to treat these
          // cases identically and not have to create specialised
          // lowerings for either of these in the lowering to LLVM-IR
          // and treat them like any other array that passes through.
          //
          // To do so below, we generate an extra bound for the
          // innermost array (the char type/string) using the LEN
          // parameter of the character type. And we "canonicalize"
          // the type, stripping it down to the base element type,
          // which in this case is an i8. This effectively allows
          // the lowering to treat this as a 1-D array with multiple
          // bounds which it is capable of handling without any special
          // casing.
          // TODO: Handle dynamic LEN characters.
          if (auto ct = aiir::dyn_cast_or_null<fir::CharacterType>(
                  fir::unwrapSequenceType(typeAttr.getValue()))) {
            newAttr = converter->convertType(
                fir::unwrapSequenceType(typeAttr.getValue()));
            if (auto type = aiir::dyn_cast<aiir::LLVM::LLVMArrayType>(newAttr))
              newAttr = type.getElementType();
            // We do not generate MapBoundsOps for the device pass, as
            // MapBoundsOps are not generated for the device pass, as
            // they're unused in the device lowering.
            auto offloadMod =
                llvm::dyn_cast_or_null<aiir::omp::OffloadModuleInterface>(
                    *curOp->getParentOfType<aiir::ModuleOp>());
            if (!offloadMod.getIsTargetDevice())
              mapBoundsOp = createBoundsForCharString(rewriter, ct.getLen(),
                                                      curOp.getLoc());
          } else {
            newAttr = converter->convertType(typeAttr.getValue());
          }
        } else {
          newAttr = converter->convertType(typeAttr.getValue());
        }
        newAttrs.emplace_back(attr.getName(), aiir::TypeAttr::get(newAttr));
      } else {
        newAttrs.push_back(attr);
      }
    }

    auto newOp = rewriter.replaceOpWithNewOp<aiir::omp::MapInfoOp>(
        curOp, resTypes, adaptor.getOperands(), newAttrs);
    if (mapBoundsOp) {
      rewriter.startOpModification(newOp);
      newOp.getBoundsMutable().append(aiir::ValueRange{mapBoundsOp});
      rewriter.finalizeOpModification(newOp);
    }

    return aiir::success();
  }
};

// FIR op specific conversion for PrivateClauseOp that overwrites the default
// OpenMP Dialect lowering, this allows FIR-aware lowering of types, required
// for boxes because the OpenMP dialect conversion doesn't know anything about
// FIR types.
struct PrivateClauseOpConversion
    : public OpenMPFIROpConversion<aiir::omp::PrivateClauseOp> {
  using OpenMPFIROpConversion::OpenMPFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(aiir::omp::PrivateClauseOp curOp, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    const fir::LLVMTypeConverter &converter = lowerTy();
    aiir::Type convertedAllocType;
    if (auto box = aiir::dyn_cast<fir::BaseBoxType>(curOp.getType())) {
      // In LLVM codegen fir.box<> == fir.ref<fir.box<>> == llvm.ptr
      // Here we really do want the actual structure
      if (box.isAssumedRank())
        TODO(curOp->getLoc(), "Privatize an assumed rank array");
      unsigned rank = 0;
      if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(
              fir::unwrapRefType(box.getEleTy())))
        rank = seqTy.getShape().size();
      convertedAllocType = converter.convertBoxTypeAsStruct(box, rank);
    } else {
      convertedAllocType = converter.convertType(adaptor.getType());
    }
    if (!convertedAllocType)
      return aiir::failure();
    rewriter.startOpModification(curOp);
    curOp.setType(convertedAllocType);
    rewriter.finalizeOpModification(curOp);
    return aiir::success();
  }
};

// Convert FIR type to LLVM without turning fir.box<T> into memory
// reference.
static aiir::Type convertObjectType(const fir::LLVMTypeConverter &converter,
                                    aiir::Type firType) {
  if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(firType))
    return converter.convertBoxTypeAsStruct(boxTy);
  return converter.convertType(firType);
}

// FIR Op specific conversion for TargetAllocMemOp
struct TargetAllocMemOpConversion
    : public OpenMPFIROpConversion<aiir::omp::TargetAllocMemOp> {
  using OpenMPFIROpConversion::OpenMPFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(aiir::omp::TargetAllocMemOp allocmemOp, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Type heapTy = allocmemOp.getAllocatedType();
    aiir::Location loc = allocmemOp.getLoc();
    auto ity = lowerTy().indexType();
    aiir::Type dataTy = fir::unwrapRefType(heapTy);
    aiir::Type llvmObjectTy = convertObjectType(lowerTy(), dataTy);
    if (fir::isRecordWithTypeParameters(fir::unwrapSequenceType(dataTy)))
      TODO(loc, "omp.target_allocmem codegen of derived type with length "
                "parameters");
    aiir::Value size = fir::computeElementDistance(
        loc, llvmObjectTy, ity, rewriter, lowerTy().getDataLayout());
    if (auto scaleSize = fir::genAllocationScaleSize(
            loc, allocmemOp.getInType(), ity, rewriter))
      size = aiir::LLVM::MulOp::create(rewriter, loc, ity, size, scaleSize);
    for (aiir::Value opnd : adaptor.getOperands().drop_front())
      size = aiir::LLVM::MulOp::create(
          rewriter, loc, ity, size,
          integerCast(lowerTy(), loc, rewriter, ity, opnd));
    auto mallocTyWidth = lowerTy().getIndexTypeBitwidth();
    auto mallocTy =
        aiir::IntegerType::get(rewriter.getContext(), mallocTyWidth);
    if (mallocTyWidth != ity.getIntOrFloatBitWidth())
      size = integerCast(lowerTy(), loc, rewriter, mallocTy, size);
    rewriter.modifyOpInPlace(allocmemOp, [&]() {
      allocmemOp.setInType(rewriter.getI8Type());
      allocmemOp.getTypeparamsMutable().clear();
      allocmemOp.getTypeparamsMutable().append(size);
    });
    return aiir::success();
  }
};

struct DeclareMapperOpConversion
    : public OpenMPFIROpConversion<aiir::omp::DeclareMapperOp> {
  using OpenMPFIROpConversion::OpenMPFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(aiir::omp::DeclareMapperOp curOp, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    rewriter.startOpModification(curOp);
    curOp.setType(convertObjectType(lowerTy(), curOp.getType()));
    rewriter.finalizeOpModification(curOp);
    return aiir::success();
  }
};

} // namespace

void fir::populateOpenMPFIRToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, aiir::RewritePatternSet &patterns) {
  patterns.add<MapInfoOpConversion>(converter);
  patterns.add<PrivateClauseOpConversion>(converter);
  patterns.add<TargetAllocMemOpConversion>(converter);
  patterns.add<DeclareMapperOpConversion>(converter);
}
