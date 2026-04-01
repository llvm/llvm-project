//===-- CodeGen.cpp -- bridge to lower to LLVM ----------------------------===//
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

#include "flang/Optimizer/CodeGen/FIROpPatterns.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/Support/Debug.h"

static inline aiir::Type getLlvmPtrType(aiir::AIIRContext *context,
                                        unsigned addressSpace = 0) {
  return aiir::LLVM::LLVMPointerType::get(context, addressSpace);
}

static unsigned getTypeDescFieldId(aiir::Type ty) {
  auto isArray = aiir::isa<fir::SequenceType>(fir::dyn_cast_ptrOrBoxEleTy(ty));
  return isArray ? kOptTypePtrPosInBox : kDimsPosInBox;
}

namespace fir {

ConvertFIRToLLVMPattern::ConvertFIRToLLVMPattern(
    llvm::StringRef rootOpName, aiir::AIIRContext *context,
    const fir::LLVMTypeConverter &typeConverter,
    const fir::FIRToLLVMPassOptions &options, aiir::PatternBenefit benefit)
    : ConvertToLLVMPattern(rootOpName, context, typeConverter, benefit),
      options(options) {}

// Convert FIR type to LLVM without turning fir.box<T> into memory
// reference.
aiir::Type
ConvertFIRToLLVMPattern::convertObjectType(aiir::Type firType) const {
  if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(firType))
    return lowerTy().convertBoxTypeAsStruct(boxTy);
  return lowerTy().convertType(firType);
}

aiir::LLVM::ConstantOp ConvertFIRToLLVMPattern::genI32Constant(
    aiir::Location loc, aiir::ConversionPatternRewriter &rewriter,
    int value) const {
  aiir::Type i32Ty = rewriter.getI32Type();
  aiir::IntegerAttr attr = rewriter.getI32IntegerAttr(value);
  return aiir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, attr);
}

aiir::LLVM::ConstantOp ConvertFIRToLLVMPattern::genConstantOffset(
    aiir::Location loc, aiir::ConversionPatternRewriter &rewriter,
    int offset) const {
  aiir::Type ity = lowerTy().offsetType();
  aiir::IntegerAttr cattr = rewriter.getI32IntegerAttr(offset);
  return aiir::LLVM::ConstantOp::create(rewriter, loc, ity, cattr);
}

/// Perform an extension or truncation as needed on an integer value. Lowering
/// to the specific target may involve some sign-extending or truncation of
/// values, particularly to fit them from abstract box types to the
/// appropriate reified structures.
aiir::Value ConvertFIRToLLVMPattern::integerCast(
    aiir::Location loc, aiir::ConversionPatternRewriter &rewriter,
    aiir::Type ty, aiir::Value val, bool fold) const {
  auto valTy = val.getType();
  // If the value was not yet lowered, lower its type so that it can
  // be used in getPrimitiveTypeSizeInBits.
  if (!aiir::isa<aiir::IntegerType>(valTy))
    valTy = convertType(valTy);
  auto toSize = aiir::LLVM::getPrimitiveTypeSizeInBits(ty);
  auto fromSize = aiir::LLVM::getPrimitiveTypeSizeInBits(valTy);
  if (fold) {
    if (toSize < fromSize)
      return rewriter.createOrFold<aiir::LLVM::TruncOp>(loc, ty, val);
    if (toSize > fromSize)
      return rewriter.createOrFold<aiir::LLVM::SExtOp>(loc, ty, val);
  } else {
    if (toSize < fromSize)
      return aiir::LLVM::TruncOp::create(rewriter, loc, ty, val);
    if (toSize > fromSize)
      return aiir::LLVM::SExtOp::create(rewriter, loc, ty, val);
  }
  return val;
}

fir::ConvertFIRToLLVMPattern::TypePair
ConvertFIRToLLVMPattern::getBoxTypePair(aiir::Type firBoxTy) const {
  aiir::Type llvmBoxTy =
      lowerTy().convertBoxTypeAsStruct(aiir::cast<fir::BaseBoxType>(firBoxTy));
  return TypePair{firBoxTy, llvmBoxTy};
}

/// Construct code sequence to extract the specific value from a `fir.box`.
aiir::Value ConvertFIRToLLVMPattern::getValueFromBox(
    aiir::Location loc, TypePair boxTy, aiir::Value box, aiir::Type resultTy,
    aiir::ConversionPatternRewriter &rewriter, int boxValue) const {
  if (aiir::isa<aiir::LLVM::LLVMPointerType>(box.getType())) {
    auto pty = getLlvmPtrType(resultTy.getContext());
    auto p = aiir::LLVM::GEPOp::create(
        rewriter, loc, pty, boxTy.llvm, box,
        llvm::ArrayRef<aiir::LLVM::GEPArg>{0, boxValue});
    auto fldTy = getBoxEleTy(boxTy.llvm, {boxValue});
    auto loadOp = aiir::LLVM::LoadOp::create(rewriter, loc, fldTy, p);
    auto castOp = integerCast(loc, rewriter, resultTy, loadOp);
    attachTBAATag(loadOp, boxTy.fir, nullptr, p);
    return castOp;
  }
  return aiir::LLVM::ExtractValueOp::create(rewriter, loc, box, boxValue);
}

/// Method to construct code sequence to get the triple for dimension `dim`
/// from a box.
llvm::SmallVector<aiir::Value, 3> ConvertFIRToLLVMPattern::getDimsFromBox(
    aiir::Location loc, llvm::ArrayRef<aiir::Type> retTys, TypePair boxTy,
    aiir::Value box, aiir::Value dim,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value l0 =
      loadDimFieldFromBox(loc, boxTy, box, dim, 0, retTys[0], rewriter);
  aiir::Value l1 =
      loadDimFieldFromBox(loc, boxTy, box, dim, 1, retTys[1], rewriter);
  aiir::Value l2 =
      loadDimFieldFromBox(loc, boxTy, box, dim, 2, retTys[2], rewriter);
  return {l0, l1, l2};
}

llvm::SmallVector<aiir::Value, 3> ConvertFIRToLLVMPattern::getDimsFromBox(
    aiir::Location loc, llvm::ArrayRef<aiir::Type> retTys, TypePair boxTy,
    aiir::Value box, int dim, aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value l0 =
      getDimFieldFromBox(loc, boxTy, box, dim, 0, retTys[0], rewriter);
  aiir::Value l1 =
      getDimFieldFromBox(loc, boxTy, box, dim, 1, retTys[1], rewriter);
  aiir::Value l2 =
      getDimFieldFromBox(loc, boxTy, box, dim, 2, retTys[2], rewriter);
  return {l0, l1, l2};
}

aiir::Value ConvertFIRToLLVMPattern::loadDimFieldFromBox(
    aiir::Location loc, TypePair boxTy, aiir::Value box, aiir::Value dim,
    int off, aiir::Type ty, aiir::ConversionPatternRewriter &rewriter) const {
  assert(aiir::isa<aiir::LLVM::LLVMPointerType>(box.getType()) &&
         "descriptor inquiry with runtime dim can only be done on descriptor "
         "in memory");
  aiir::LLVM::GEPOp p = genGEP(loc, boxTy.llvm, rewriter, box, 0,
                               static_cast<int>(kDimsPosInBox), dim, off);
  auto loadOp = aiir::LLVM::LoadOp::create(rewriter, loc, ty, p);
  attachTBAATag(loadOp, boxTy.fir, nullptr, p);
  return loadOp;
}

aiir::Value ConvertFIRToLLVMPattern::getDimFieldFromBox(
    aiir::Location loc, TypePair boxTy, aiir::Value box, int dim, int off,
    aiir::Type ty, aiir::ConversionPatternRewriter &rewriter) const {
  if (aiir::isa<aiir::LLVM::LLVMPointerType>(box.getType())) {
    aiir::LLVM::GEPOp p = genGEP(loc, boxTy.llvm, rewriter, box, 0,
                                 static_cast<int>(kDimsPosInBox), dim, off);
    auto loadOp = aiir::LLVM::LoadOp::create(rewriter, loc, ty, p);
    attachTBAATag(loadOp, boxTy.fir, nullptr, p);
    return loadOp;
  }
  return aiir::LLVM::ExtractValueOp::create(
      rewriter, loc, box,
      llvm::ArrayRef<std::int64_t>{kDimsPosInBox, dim, off});
}

aiir::Value ConvertFIRToLLVMPattern::getStrideFromBox(
    aiir::Location loc, TypePair boxTy, aiir::Value box, unsigned dim,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto idxTy = lowerTy().indexType();
  return getDimFieldFromBox(loc, boxTy, box, dim, kDimStridePos, idxTy,
                            rewriter);
}

/// Read base address from a fir.box. Returned address has type ty.
aiir::Value ConvertFIRToLLVMPattern::getBaseAddrFromBox(
    aiir::Location loc, TypePair boxTy, aiir::Value box,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resultTy = ::getLlvmPtrType(boxTy.llvm.getContext());
  return getValueFromBox(loc, boxTy, box, resultTy, rewriter, kAddrPosInBox);
}

aiir::Value ConvertFIRToLLVMPattern::getElementSizeFromBox(
    aiir::Location loc, aiir::Type resultTy, TypePair boxTy, aiir::Value box,
    aiir::ConversionPatternRewriter &rewriter) const {
  return getValueFromBox(loc, boxTy, box, resultTy, rewriter, kElemLenPosInBox);
}

/// Read base address from a fir.box. Returned address has type ty.
aiir::Value ConvertFIRToLLVMPattern::getRankFromBox(
    aiir::Location loc, TypePair boxTy, aiir::Value box,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resultTy = getBoxEleTy(boxTy.llvm, {kRankPosInBox});
  return getValueFromBox(loc, boxTy, box, resultTy, rewriter, kRankPosInBox);
}

/// Read the extra field from a fir.box.
aiir::Value ConvertFIRToLLVMPattern::getExtraFromBox(
    aiir::Location loc, TypePair boxTy, aiir::Value box,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resultTy = getBoxEleTy(boxTy.llvm, {kExtraPosInBox});
  return getValueFromBox(loc, boxTy, box, resultTy, rewriter, kExtraPosInBox);
}

// Get the element type given an LLVM type that is of the form
// (array|struct|vector)+ and the provided indexes.
aiir::Type ConvertFIRToLLVMPattern::getBoxEleTy(
    aiir::Type type, llvm::ArrayRef<std::int64_t> indexes) const {
  for (unsigned i : indexes) {
    if (auto t = aiir::dyn_cast<aiir::LLVM::LLVMStructType>(type)) {
      assert(!t.isOpaque() && i < t.getBody().size());
      type = t.getBody()[i];
    } else if (auto t = aiir::dyn_cast<aiir::LLVM::LLVMArrayType>(type)) {
      type = t.getElementType();
    } else if (auto t = aiir::dyn_cast<aiir::VectorType>(type)) {
      type = t.getElementType();
    } else {
      fir::emitFatalError(aiir::UnknownLoc::get(type.getContext()),
                          "request for invalid box element type");
    }
  }
  return type;
}

// Return LLVM type of the object described by a fir.box of \p boxType.
aiir::Type ConvertFIRToLLVMPattern::getLlvmObjectTypeFromBoxType(
    aiir::Type boxType) const {
  aiir::Type objectType = fir::dyn_cast_ptrOrBoxEleTy(boxType);
  assert(objectType && "boxType must be a box type");
  return this->convertType(objectType);
}

/// Read the address of the type descriptor from a box.
aiir::Value ConvertFIRToLLVMPattern::loadTypeDescAddress(
    aiir::Location loc, TypePair boxTy, aiir::Value box,
    aiir::ConversionPatternRewriter &rewriter) const {
  unsigned typeDescFieldId = getTypeDescFieldId(boxTy.fir);
  aiir::Type tdescType = lowerTy().convertTypeDescType(rewriter.getContext());
  return getValueFromBox(loc, boxTy, box, tdescType, rewriter, typeDescFieldId);
}

// Load the attribute from the \p box and perform a check against \p maskValue
// The final comparison is implemented as `(attribute & maskValue) != 0`.
aiir::Value ConvertFIRToLLVMPattern::genBoxAttributeCheck(
    aiir::Location loc, TypePair boxTy, aiir::Value box,
    aiir::ConversionPatternRewriter &rewriter, unsigned maskValue) const {
  aiir::Type attrTy = rewriter.getI32Type();
  aiir::Value attribute =
      getValueFromBox(loc, boxTy, box, attrTy, rewriter, kAttributePosInBox);
  aiir::LLVM::ConstantOp attrMask = genConstantOffset(loc, rewriter, maskValue);
  auto maskRes =
      aiir::LLVM::AndOp::create(rewriter, loc, attrTy, attribute, attrMask);
  aiir::LLVM::ConstantOp c0 = genConstantOffset(loc, rewriter, 0);
  return aiir::LLVM::ICmpOp::create(rewriter, loc,
                                    aiir::LLVM::ICmpPredicate::ne, maskRes, c0);
}

aiir::Value ConvertFIRToLLVMPattern::computeBoxSize(
    aiir::Location loc, TypePair boxTy, aiir::Value box,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto firBoxType = aiir::dyn_cast<fir::BaseBoxType>(boxTy.fir);
  assert(firBoxType && "must be a BaseBoxType");
  const aiir::DataLayout &dl = lowerTy().getDataLayout();
  if (!firBoxType.isAssumedRank())
    return genConstantOffset(loc, rewriter, dl.getTypeSize(boxTy.llvm));
  fir::BaseBoxType firScalarBoxType = firBoxType.getBoxTypeWithNewShape(0);
  aiir::Type llvmScalarBoxType =
      lowerTy().convertBoxTypeAsStruct(firScalarBoxType);
  llvm::TypeSize scalarBoxSizeCst = dl.getTypeSize(llvmScalarBoxType);
  aiir::Value scalarBoxSize =
      genConstantOffset(loc, rewriter, scalarBoxSizeCst);
  aiir::Value rawRank = getRankFromBox(loc, boxTy, box, rewriter);
  aiir::Value rank =
      integerCast(loc, rewriter, scalarBoxSize.getType(), rawRank);
  aiir::Type llvmDimsType = getBoxEleTy(boxTy.llvm, {kDimsPosInBox, 1});
  llvm::TypeSize sizePerDimCst = dl.getTypeSize(llvmDimsType);
  assert((scalarBoxSizeCst + sizePerDimCst ==
          dl.getTypeSize(lowerTy().convertBoxTypeAsStruct(
              firBoxType.getBoxTypeWithNewShape(1)))) &&
         "descriptor layout requires adding padding for dim field");
  aiir::Value sizePerDim = genConstantOffset(loc, rewriter, sizePerDimCst);
  aiir::Value dimsSize = aiir::LLVM::MulOp::create(
      rewriter, loc, sizePerDim.getType(), sizePerDim, rank);
  aiir::Value size = aiir::LLVM::AddOp::create(
      rewriter, loc, scalarBoxSize.getType(), scalarBoxSize, dimsSize);
  return size;
}

// Find the Block in which the alloca should be inserted.
// The order to recursively find the proper block:
// 1. An OpenMP Op that will be outlined.
// 2. An OpenMP or OpenACC Op with one or more regions holding executable code.
// 3. A LLVMFuncOp
// 4. The first ancestor that is one of the above.
aiir::Block *ConvertFIRToLLVMPattern::getBlockForAllocaInsert(
    aiir::Operation *op, aiir::Region *parentRegion) const {
  if (auto iface = aiir::dyn_cast<aiir::omp::OutlineableOpenMPOpInterface>(op))
    return iface.getAllocaBlock();
  if (auto recipeIface = aiir::dyn_cast<aiir::accomp::RecipeInterface>(op))
    return recipeIface.getAllocaBlock(*parentRegion);
  if (auto llvmFuncOp = aiir::dyn_cast<aiir::LLVM::LLVMFuncOp>(op))
    return &llvmFuncOp.front();

  return getBlockForAllocaInsert(op->getParentOp(), parentRegion);
}

// Generate an alloca of size 1 for an object of type \p llvmObjectTy in the
// allocation address space provided for the architecture in the DataLayout
// specification. If the address space is different from the devices
// program address space we perform a cast. In the case of most architectures
// the program and allocation address space will be the default of 0 and no
// cast will be emitted.
aiir::Value ConvertFIRToLLVMPattern::genAllocaAndAddrCastWithType(
    aiir::Location loc, aiir::Type llvmObjectTy, unsigned alignment,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto thisPt = rewriter.saveInsertionPoint();
  aiir::Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
  aiir::Region *parentRegion = rewriter.getInsertionBlock()->getParent();
  aiir::Block *insertBlock = getBlockForAllocaInsert(parentOp, parentRegion);
  rewriter.setInsertionPointToStart(insertBlock);
  auto size = genI32Constant(loc, rewriter, 1);
  unsigned allocaAs = getAllocaAddressSpace(rewriter);
  unsigned programAs = getProgramAddressSpace(rewriter);

  aiir::Value al = aiir::LLVM::AllocaOp::create(
      rewriter, loc, ::getLlvmPtrType(llvmObjectTy.getContext(), allocaAs),
      llvmObjectTy, size, alignment);

  // if our allocation address space, is not the same as the program address
  // space, then we must emit a cast to the program address space before use.
  // An example case would be on AMDGPU, where the allocation address space is
  // the numeric value 5 (private), and the program address space is 0
  // (generic).
  if (allocaAs != programAs) {
    al = aiir::LLVM::AddrSpaceCastOp::create(
        rewriter, loc, ::getLlvmPtrType(llvmObjectTy.getContext(), programAs),
        al);
  }

  rewriter.restoreInsertionPoint(thisPt);
  return al;
}

unsigned ConvertFIRToLLVMPattern::getAllocaAddressSpace(
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
  assert(parentOp != nullptr &&
         "expected insertion block to have parent operation");
  auto module = aiir::isa<aiir::ModuleOp>(parentOp)
                    ? aiir::cast<aiir::ModuleOp>(parentOp)
                    : parentOp->getParentOfType<aiir::ModuleOp>();
  if (module)
    if (aiir::Attribute addrSpace =
            aiir::DataLayout(module).getAllocaMemorySpace())
      return llvm::cast<aiir::IntegerAttr>(addrSpace).getUInt();
  return defaultAddressSpace;
}

unsigned ConvertFIRToLLVMPattern::getProgramAddressSpace(
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
  assert(parentOp != nullptr &&
         "expected insertion block to have parent operation");
  auto module = aiir::isa<aiir::ModuleOp>(parentOp)
                    ? aiir::cast<aiir::ModuleOp>(parentOp)
                    : parentOp->getParentOfType<aiir::ModuleOp>();
  if (module)
    if (aiir::Attribute addrSpace =
            aiir::DataLayout(module).getProgramMemorySpace())
      return llvm::cast<aiir::IntegerAttr>(addrSpace).getUInt();
  return defaultAddressSpace;
}

unsigned ConvertFIRToLLVMPattern::getGlobalAddressSpace(
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
  assert(parentOp != nullptr &&
         "expected insertion block to have parent operation");
  auto module = aiir::isa<aiir::ModuleOp>(parentOp)
                    ? aiir::cast<aiir::ModuleOp>(parentOp)
                    : parentOp->getParentOfType<aiir::ModuleOp>();
  if (module)
    if (aiir::Attribute addrSpace =
            aiir::DataLayout(module).getGlobalMemorySpace())
      return llvm::cast<aiir::IntegerAttr>(addrSpace).getUInt();
  return defaultAddressSpace;
}

} // namespace fir
