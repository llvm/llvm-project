//===-- CodeGen.cpp -- bridge to lower to LLVM ----------------------------===//
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

#include "flang/Optimizer/CodeGen/FIROpPatterns.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/Support/Debug.h"

static inline mlir::Type getLlvmPtrType(mlir::MLIRContext *context,
                                        unsigned addressSpace = 0) {
  return mlir::LLVM::LLVMPointerType::get(context, addressSpace);
}

static unsigned getTypeDescFieldId(mlir::Type ty) {
  auto isArray = mlir::isa<fir::SequenceType>(fir::dyn_cast_ptrOrBoxEleTy(ty));
  return isArray ? kOptTypePtrPosInBox : kDimsPosInBox;
}

namespace fir {

ConvertFIRToLLVMPattern::ConvertFIRToLLVMPattern(
    llvm::StringRef rootOpName, mlir::MLIRContext *context,
    const fir::LLVMTypeConverter &typeConverter,
    const fir::FIRToLLVMPassOptions &options, mlir::PatternBenefit benefit)
    : ConvertToLLVMPattern(rootOpName, context, typeConverter, benefit),
      options(options) {}

// Convert FIR type to LLVM without turning fir.box<T> into memory
// reference.
mlir::Type
ConvertFIRToLLVMPattern::convertObjectType(mlir::Type firType) const {
  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(firType))
    return lowerTy().convertBoxTypeAsStruct(boxTy);
  return lowerTy().convertType(firType);
}

mlir::LLVM::ConstantOp ConvertFIRToLLVMPattern::genI32Constant(
    mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
    int value) const {
  mlir::Type i32Ty = rewriter.getI32Type();
  mlir::IntegerAttr attr = rewriter.getI32IntegerAttr(value);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, i32Ty, attr);
}

mlir::LLVM::ConstantOp ConvertFIRToLLVMPattern::genConstantOffset(
    mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
    int offset) const {
  mlir::Type ity = lowerTy().offsetType();
  mlir::IntegerAttr cattr = rewriter.getI32IntegerAttr(offset);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, ity, cattr);
}

/// Perform an extension or truncation as needed on an integer value. Lowering
/// to the specific target may involve some sign-extending or truncation of
/// values, particularly to fit them from abstract box types to the
/// appropriate reified structures.
mlir::Value ConvertFIRToLLVMPattern::integerCast(
    mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
    mlir::Type ty, mlir::Value val, bool fold) const {
  auto valTy = val.getType();
  // If the value was not yet lowered, lower its type so that it can
  // be used in getPrimitiveTypeSizeInBits.
  if (!mlir::isa<mlir::IntegerType>(valTy))
    valTy = convertType(valTy);
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

fir::ConvertFIRToLLVMPattern::TypePair
ConvertFIRToLLVMPattern::getBoxTypePair(mlir::Type firBoxTy) const {
  mlir::Type llvmBoxTy =
      lowerTy().convertBoxTypeAsStruct(mlir::cast<fir::BaseBoxType>(firBoxTy));
  return TypePair{firBoxTy, llvmBoxTy};
}

/// Construct code sequence to extract the specific value from a `fir.box`.
mlir::Value ConvertFIRToLLVMPattern::getValueFromBox(
    mlir::Location loc, TypePair boxTy, mlir::Value box, mlir::Type resultTy,
    mlir::ConversionPatternRewriter &rewriter, int boxValue) const {
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(box.getType())) {
    auto pty = getLlvmPtrType(resultTy.getContext());
    auto p = rewriter.create<mlir::LLVM::GEPOp>(
        loc, pty, boxTy.llvm, box,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, boxValue});
    auto fldTy = getBoxEleTy(boxTy.llvm, {boxValue});
    auto loadOp = rewriter.create<mlir::LLVM::LoadOp>(loc, fldTy, p);
    auto castOp = integerCast(loc, rewriter, resultTy, loadOp);
    attachTBAATag(loadOp, boxTy.fir, nullptr, p);
    return castOp;
  }
  return rewriter.create<mlir::LLVM::ExtractValueOp>(loc, box, boxValue);
}

/// Method to construct code sequence to get the triple for dimension `dim`
/// from a box.
llvm::SmallVector<mlir::Value, 3> ConvertFIRToLLVMPattern::getDimsFromBox(
    mlir::Location loc, llvm::ArrayRef<mlir::Type> retTys, TypePair boxTy,
    mlir::Value box, mlir::Value dim,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value l0 =
      loadDimFieldFromBox(loc, boxTy, box, dim, 0, retTys[0], rewriter);
  mlir::Value l1 =
      loadDimFieldFromBox(loc, boxTy, box, dim, 1, retTys[1], rewriter);
  mlir::Value l2 =
      loadDimFieldFromBox(loc, boxTy, box, dim, 2, retTys[2], rewriter);
  return {l0, l1, l2};
}

llvm::SmallVector<mlir::Value, 3> ConvertFIRToLLVMPattern::getDimsFromBox(
    mlir::Location loc, llvm::ArrayRef<mlir::Type> retTys, TypePair boxTy,
    mlir::Value box, int dim, mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value l0 =
      getDimFieldFromBox(loc, boxTy, box, dim, 0, retTys[0], rewriter);
  mlir::Value l1 =
      getDimFieldFromBox(loc, boxTy, box, dim, 1, retTys[1], rewriter);
  mlir::Value l2 =
      getDimFieldFromBox(loc, boxTy, box, dim, 2, retTys[2], rewriter);
  return {l0, l1, l2};
}

mlir::Value ConvertFIRToLLVMPattern::loadDimFieldFromBox(
    mlir::Location loc, TypePair boxTy, mlir::Value box, mlir::Value dim,
    int off, mlir::Type ty, mlir::ConversionPatternRewriter &rewriter) const {
  assert(mlir::isa<mlir::LLVM::LLVMPointerType>(box.getType()) &&
         "descriptor inquiry with runtime dim can only be done on descriptor "
         "in memory");
  mlir::LLVM::GEPOp p = genGEP(loc, boxTy.llvm, rewriter, box, 0,
                               static_cast<int>(kDimsPosInBox), dim, off);
  auto loadOp = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
  attachTBAATag(loadOp, boxTy.fir, nullptr, p);
  return loadOp;
}

mlir::Value ConvertFIRToLLVMPattern::getDimFieldFromBox(
    mlir::Location loc, TypePair boxTy, mlir::Value box, int dim, int off,
    mlir::Type ty, mlir::ConversionPatternRewriter &rewriter) const {
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(box.getType())) {
    mlir::LLVM::GEPOp p = genGEP(loc, boxTy.llvm, rewriter, box, 0,
                                 static_cast<int>(kDimsPosInBox), dim, off);
    auto loadOp = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
    attachTBAATag(loadOp, boxTy.fir, nullptr, p);
    return loadOp;
  }
  return rewriter.create<mlir::LLVM::ExtractValueOp>(
      loc, box, llvm::ArrayRef<std::int64_t>{kDimsPosInBox, dim, off});
}

mlir::Value ConvertFIRToLLVMPattern::getStrideFromBox(
    mlir::Location loc, TypePair boxTy, mlir::Value box, unsigned dim,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto idxTy = lowerTy().indexType();
  return getDimFieldFromBox(loc, boxTy, box, dim, kDimStridePos, idxTy,
                            rewriter);
}

/// Read base address from a fir.box. Returned address has type ty.
mlir::Value ConvertFIRToLLVMPattern::getBaseAddrFromBox(
    mlir::Location loc, TypePair boxTy, mlir::Value box,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type resultTy = ::getLlvmPtrType(boxTy.llvm.getContext());
  return getValueFromBox(loc, boxTy, box, resultTy, rewriter, kAddrPosInBox);
}

mlir::Value ConvertFIRToLLVMPattern::getElementSizeFromBox(
    mlir::Location loc, mlir::Type resultTy, TypePair boxTy, mlir::Value box,
    mlir::ConversionPatternRewriter &rewriter) const {
  return getValueFromBox(loc, boxTy, box, resultTy, rewriter, kElemLenPosInBox);
}

/// Read base address from a fir.box. Returned address has type ty.
mlir::Value ConvertFIRToLLVMPattern::getRankFromBox(
    mlir::Location loc, TypePair boxTy, mlir::Value box,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type resultTy = getBoxEleTy(boxTy.llvm, {kRankPosInBox});
  return getValueFromBox(loc, boxTy, box, resultTy, rewriter, kRankPosInBox);
}

/// Read the extra field from a fir.box.
mlir::Value ConvertFIRToLLVMPattern::getExtraFromBox(
    mlir::Location loc, TypePair boxTy, mlir::Value box,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type resultTy = getBoxEleTy(boxTy.llvm, {kExtraPosInBox});
  return getValueFromBox(loc, boxTy, box, resultTy, rewriter, kExtraPosInBox);
}

// Get the element type given an LLVM type that is of the form
// (array|struct|vector)+ and the provided indexes.
mlir::Type ConvertFIRToLLVMPattern::getBoxEleTy(
    mlir::Type type, llvm::ArrayRef<std::int64_t> indexes) const {
  for (unsigned i : indexes) {
    if (auto t = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type)) {
      assert(!t.isOpaque() && i < t.getBody().size());
      type = t.getBody()[i];
    } else if (auto t = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(type)) {
      type = t.getElementType();
    } else if (auto t = mlir::dyn_cast<mlir::VectorType>(type)) {
      type = t.getElementType();
    } else {
      fir::emitFatalError(mlir::UnknownLoc::get(type.getContext()),
                          "request for invalid box element type");
    }
  }
  return type;
}

// Return LLVM type of the object described by a fir.box of \p boxType.
mlir::Type ConvertFIRToLLVMPattern::getLlvmObjectTypeFromBoxType(
    mlir::Type boxType) const {
  mlir::Type objectType = fir::dyn_cast_ptrOrBoxEleTy(boxType);
  assert(objectType && "boxType must be a box type");
  return this->convertType(objectType);
}

/// Read the address of the type descriptor from a box.
mlir::Value ConvertFIRToLLVMPattern::loadTypeDescAddress(
    mlir::Location loc, TypePair boxTy, mlir::Value box,
    mlir::ConversionPatternRewriter &rewriter) const {
  unsigned typeDescFieldId = getTypeDescFieldId(boxTy.fir);
  mlir::Type tdescType = lowerTy().convertTypeDescType(rewriter.getContext());
  return getValueFromBox(loc, boxTy, box, tdescType, rewriter, typeDescFieldId);
}

// Load the attribute from the \p box and perform a check against \p maskValue
// The final comparison is implemented as `(attribute & maskValue) != 0`.
mlir::Value ConvertFIRToLLVMPattern::genBoxAttributeCheck(
    mlir::Location loc, TypePair boxTy, mlir::Value box,
    mlir::ConversionPatternRewriter &rewriter, unsigned maskValue) const {
  mlir::Type attrTy = rewriter.getI32Type();
  mlir::Value attribute =
      getValueFromBox(loc, boxTy, box, attrTy, rewriter, kAttributePosInBox);
  mlir::LLVM::ConstantOp attrMask = genConstantOffset(loc, rewriter, maskValue);
  auto maskRes =
      rewriter.create<mlir::LLVM::AndOp>(loc, attrTy, attribute, attrMask);
  mlir::LLVM::ConstantOp c0 = genConstantOffset(loc, rewriter, 0);
  return rewriter.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::ne,
                                             maskRes, c0);
}

mlir::Value ConvertFIRToLLVMPattern::computeBoxSize(
    mlir::Location loc, TypePair boxTy, mlir::Value box,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto firBoxType = mlir::dyn_cast<fir::BaseBoxType>(boxTy.fir);
  assert(firBoxType && "must be a BaseBoxType");
  const mlir::DataLayout &dl = lowerTy().getDataLayout();
  if (!firBoxType.isAssumedRank())
    return genConstantOffset(loc, rewriter, dl.getTypeSize(boxTy.llvm));
  fir::BaseBoxType firScalarBoxType = firBoxType.getBoxTypeWithNewShape(0);
  mlir::Type llvmScalarBoxType =
      lowerTy().convertBoxTypeAsStruct(firScalarBoxType);
  llvm::TypeSize scalarBoxSizeCst = dl.getTypeSize(llvmScalarBoxType);
  mlir::Value scalarBoxSize =
      genConstantOffset(loc, rewriter, scalarBoxSizeCst);
  mlir::Value rawRank = getRankFromBox(loc, boxTy, box, rewriter);
  mlir::Value rank =
      integerCast(loc, rewriter, scalarBoxSize.getType(), rawRank);
  mlir::Type llvmDimsType = getBoxEleTy(boxTy.llvm, {kDimsPosInBox, 1});
  llvm::TypeSize sizePerDimCst = dl.getTypeSize(llvmDimsType);
  assert((scalarBoxSizeCst + sizePerDimCst ==
          dl.getTypeSize(lowerTy().convertBoxTypeAsStruct(
              firBoxType.getBoxTypeWithNewShape(1)))) &&
         "descriptor layout requires adding padding for dim field");
  mlir::Value sizePerDim = genConstantOffset(loc, rewriter, sizePerDimCst);
  mlir::Value dimsSize = rewriter.create<mlir::LLVM::MulOp>(
      loc, sizePerDim.getType(), sizePerDim, rank);
  mlir::Value size = rewriter.create<mlir::LLVM::AddOp>(
      loc, scalarBoxSize.getType(), scalarBoxSize, dimsSize);
  return size;
}

// Find the Block in which the alloca should be inserted.
// The order to recursively find the proper block:
// 1. An OpenMP Op that will be outlined.
// 2. An OpenMP or OpenACC Op with one or more regions holding executable code.
// 3. A LLVMFuncOp
// 4. The first ancestor that is one of the above.
mlir::Block *ConvertFIRToLLVMPattern::getBlockForAllocaInsert(
    mlir::Operation *op, mlir::Region *parentRegion) const {
  if (auto iface = mlir::dyn_cast<mlir::omp::OutlineableOpenMPOpInterface>(op))
    return iface.getAllocaBlock();
  if (auto recipeIface = mlir::dyn_cast<mlir::accomp::RecipeInterface>(op))
    return recipeIface.getAllocaBlock(*parentRegion);
  if (auto llvmFuncOp = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(op))
    return &llvmFuncOp.front();

  return getBlockForAllocaInsert(op->getParentOp(), parentRegion);
}

// Generate an alloca of size 1 for an object of type \p llvmObjectTy in the
// allocation address space provided for the architecture in the DataLayout
// specification. If the address space is different from the devices
// program address space we perform a cast. In the case of most architectures
// the program and allocation address space will be the default of 0 and no
// cast will be emitted.
mlir::Value ConvertFIRToLLVMPattern::genAllocaAndAddrCastWithType(
    mlir::Location loc, mlir::Type llvmObjectTy, unsigned alignment,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto thisPt = rewriter.saveInsertionPoint();
  mlir::Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
  mlir::Region *parentRegion = rewriter.getInsertionBlock()->getParent();
  mlir::Block *insertBlock = getBlockForAllocaInsert(parentOp, parentRegion);
  rewriter.setInsertionPointToStart(insertBlock);
  auto size = genI32Constant(loc, rewriter, 1);
  unsigned allocaAs = getAllocaAddressSpace(rewriter);
  unsigned programAs = getProgramAddressSpace(rewriter);

  mlir::Value al = rewriter.create<mlir::LLVM::AllocaOp>(
      loc, ::getLlvmPtrType(llvmObjectTy.getContext(), allocaAs), llvmObjectTy,
      size, alignment);

  // if our allocation address space, is not the same as the program address
  // space, then we must emit a cast to the program address space before use.
  // An example case would be on AMDGPU, where the allocation address space is
  // the numeric value 5 (private), and the program address space is 0
  // (generic).
  if (allocaAs != programAs) {
    al = rewriter.create<mlir::LLVM::AddrSpaceCastOp>(
        loc, ::getLlvmPtrType(llvmObjectTy.getContext(), programAs), al);
  }

  rewriter.restoreInsertionPoint(thisPt);
  return al;
}

unsigned ConvertFIRToLLVMPattern::getAllocaAddressSpace(
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
  assert(parentOp != nullptr &&
         "expected insertion block to have parent operation");
  auto module = mlir::isa<mlir::ModuleOp>(parentOp)
                    ? mlir::cast<mlir::ModuleOp>(parentOp)
                    : parentOp->getParentOfType<mlir::ModuleOp>();
  if (module)
    if (mlir::Attribute addrSpace =
            mlir::DataLayout(module).getAllocaMemorySpace())
      return llvm::cast<mlir::IntegerAttr>(addrSpace).getUInt();
  return defaultAddressSpace;
}

unsigned ConvertFIRToLLVMPattern::getProgramAddressSpace(
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
  assert(parentOp != nullptr &&
         "expected insertion block to have parent operation");
  auto module = mlir::isa<mlir::ModuleOp>(parentOp)
                    ? mlir::cast<mlir::ModuleOp>(parentOp)
                    : parentOp->getParentOfType<mlir::ModuleOp>();
  if (module)
    if (mlir::Attribute addrSpace =
            mlir::DataLayout(module).getProgramMemorySpace())
      return llvm::cast<mlir::IntegerAttr>(addrSpace).getUInt();
  return defaultAddressSpace;
}

unsigned ConvertFIRToLLVMPattern::getGlobalAddressSpace(
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
  assert(parentOp != nullptr &&
         "expected insertion block to have parent operation");
  auto module = mlir::isa<mlir::ModuleOp>(parentOp)
                    ? mlir::cast<mlir::ModuleOp>(parentOp)
                    : parentOp->getParentOfType<mlir::ModuleOp>();
  if (module)
    if (mlir::Attribute addrSpace =
            mlir::DataLayout(module).getGlobalMemorySpace())
      return llvm::cast<mlir::IntegerAttr>(addrSpace).getUInt();
  return defaultAddressSpace;
}

} // namespace fir
