//====- LoweringHelpers.cpp - Lowering helper functions -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains helper functions for lowering from CIR to LLVM or MLIR.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/LoweringHelpers.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/SymbolTable.h"

static unsigned getIntOrBoolBitWidth(mlir::Type ty) {
  if (auto intTy = mlir::dyn_cast<cir::IntType>(ty))
    return intTy.getWidth();
  assert(mlir::isa<cir::BoolType>(ty) &&
         "expected CIR integer or bool element type");
  return 1;
}

mlir::DenseElementsAttr
convertStringAttrToDenseElementsAttr(cir::ConstArrayAttr attr,
                                     mlir::Type type) {
  const auto stringAttr = mlir::cast<mlir::StringAttr>(attr.getElts());
  const auto arrayTy = mlir::cast<cir::ArrayType>(attr.getType());
  const unsigned totalSize = arrayTy.getSize();
  const unsigned trailingZeros = attr.getTrailingZerosNum();
  assert(stringAttr.size() + trailingZeros == totalSize &&
         "string const_array size must match explicit elements plus "
         "trailing_zeros");

  const unsigned bitWidth = getIntOrBoolBitWidth(arrayTy.getElementType());
  llvm::SmallVector<mlir::APInt> values;
  values.reserve(totalSize);

  // String bytes are raw values; interpret each as an unsigned byte so a
  // high-bit char (>= 0x80) does not sign-extend to a value that overflows
  // the element bit width when constructing the APInt.
  for (const char element : stringAttr)
    values.emplace_back(bitWidth, static_cast<unsigned char>(element));

  values.insert(values.end(), trailingZeros, mlir::APInt::getZero(bitWidth));

  return mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({totalSize}, type), llvm::ArrayRef(values));
}

template <> mlir::APInt getZeroInitFromType(mlir::Type ty) {
  if (mlir::isa<cir::BoolType>(ty))
    return mlir::APInt::getZero(1);
  const auto intTy = mlir::cast<cir::IntType>(ty);
  return mlir::APInt::getZero(intTy.getWidth());
}

template <> mlir::APFloat getZeroInitFromType(mlir::Type ty) {
  assert((mlir::isa<cir::SingleType, cir::DoubleType>(ty)) &&
         "only float and double supported");

  if (ty.isF32() || mlir::isa<cir::SingleType>(ty))
    return mlir::APFloat(0.f);

  if (ty.isF64() || mlir::isa<cir::DoubleType>(ty))
    return mlir::APFloat(0.0);

  llvm_unreachable("NYI");
}

/// \param attr the ConstArrayAttr to convert
/// \param values the output parameter, the values array to fill
/// \param currentDims the shpae of tensor we're going to convert to
/// \param dimIndex the current dimension we're processing
/// \param currentIndex the current index in the values array
template <typename AttrTy, typename StorageTy>
void convertToDenseElementsAttrImpl(
    cir::ConstArrayAttr attr, llvm::SmallVectorImpl<StorageTy> &values,
    const llvm::SmallVectorImpl<int64_t> &currentDims, int64_t dimIndex,
    int64_t currentIndex) {
  if (auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getElts())) {
    if (auto arrayType = mlir::dyn_cast<cir::ArrayType>(attr.getType())) {
      for (auto element : stringAttr) {
        auto intAttr = cir::IntAttr::get(arrayType.getElementType(), element);
        values[currentIndex++] = mlir::dyn_cast<AttrTy>(intAttr).getValue();
      }
      // Remaining slots are trailing zeros; values was zero-initialized.
      currentIndex += attr.getTrailingZerosNum();
      return;
    }
  }

  dimIndex++;
  std::size_t elementsSizeInCurrentDim = 1;
  for (std::size_t i = dimIndex; i < currentDims.size(); i++)
    elementsSizeInCurrentDim *= currentDims[i];

  auto arrayAttr = mlir::cast<mlir::ArrayAttr>(attr.getElts());
  for (auto eltAttr : arrayAttr) {
    if constexpr (std::is_same_v<StorageTy, mlir::APInt>) {
      if (auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(eltAttr)) {
        values[currentIndex++] =
            llvm::APInt(1, static_cast<uint64_t>(boolAttr.getValue()));
        continue;
      }
    }
    if (auto valueAttr = mlir::dyn_cast<AttrTy>(eltAttr)) {
      values[currentIndex++] = valueAttr.getValue();
      continue;
    }

    if (auto subArrayAttr = mlir::dyn_cast<cir::ConstArrayAttr>(eltAttr)) {
      convertToDenseElementsAttrImpl<AttrTy>(subArrayAttr, values, currentDims,
                                             dimIndex, currentIndex);
      currentIndex += elementsSizeInCurrentDim;
      continue;
    }

    if (mlir::isa<cir::ZeroAttr, cir::UndefAttr>(eltAttr)) {
      currentIndex += elementsSizeInCurrentDim;
      continue;
    }

    llvm_unreachable("unknown element in ConstArrayAttr");
  }
}

template <typename AttrTy, typename StorageTy>
mlir::DenseElementsAttr convertToDenseElementsAttr(
    cir::ConstArrayAttr attr, const llvm::SmallVectorImpl<int64_t> &dims,
    mlir::Type elementType, mlir::Type convertedElementType) {
  unsigned vectorSize = 1;
  for (auto dim : dims)
    vectorSize *= dim;
  auto values = llvm::SmallVector<StorageTy, 8>(
      vectorSize, getZeroInitFromType<StorageTy>(elementType));
  convertToDenseElementsAttrImpl<AttrTy>(attr, values, dims, /*currentDim=*/0,
                                         /*initialIndex=*/0);
  return mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get(dims, convertedElementType),
      llvm::ArrayRef(values));
}

/// Return true when \p gv can be lowered to a \c FlatSymbolRefAttr leaf without
/// addrspacecast or bitcast (mirrors \c CIRAttrToValue::visitCirAttr).
static bool globalViewMatchesPointerLeaf(cir::GlobalViewAttr gv,
                                         mlir::ModuleOp moduleOp,
                                         const mlir::TypeConverter *converter) {
  if (gv.getIndices() || mlir::isa<cir::IntType, cir::VPtrType>(gv.getType()))
    return false;

  auto ptrTy = mlir::dyn_cast<cir::PointerType>(gv.getType());
  if (!ptrTy)
    return false;

  unsigned sourceAddrSpace = 0;
  mlir::Type sourceType;
  auto sourceSymbol =
      mlir::SymbolTable::lookupSymbolIn(moduleOp, gv.getSymbol());
  if (auto llvmSymbol = mlir::dyn_cast<mlir::LLVM::GlobalOp>(sourceSymbol)) {
    sourceType = llvmSymbol.getType();
    sourceAddrSpace = llvmSymbol.getAddrSpace();
  } else if (auto cirSymbol = mlir::dyn_cast<cir::GlobalOp>(sourceSymbol)) {
    sourceType = converter->convertType(cirSymbol.getSymType());
    if (auto targetAS = mlir::dyn_cast_if_present<cir::TargetAddressSpaceAttr>(
            cirSymbol.getAddrSpaceAttr()))
      sourceAddrSpace = targetAS.getValue();
  } else {
    // cir.func and other symbols not yet lowered to globals cannot be used as
    // bulk constant leaves; those cases keep the insertvalue fallback.
    return false;
  }

  auto llvmDstTy = converter->convertType<mlir::LLVM::LLVMPointerType>(ptrTy);
  if (llvmDstTy.getAddressSpace() != sourceAddrSpace)
    return false;

  mlir::Type llvmEltTy = converter->convertType(ptrTy.getPointee());
  if (llvmEltTy == sourceType)
    return true;
  if (auto arrTy = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(sourceType))
    return llvmEltTy == arrTy.getElementType();
  return false;
}

/// Lower a single pointer-element of a \c cir.const_array to an LLVM-dialect
/// constant leaf suitable for a bulk \c llvm.mlir.constant.  Only handles
/// address-of-global without indices and null pointers; indexed global views
/// must use the per-element \c llvm.insertvalue fallback.
static std::optional<mlir::Attribute>
lowerPointerElementAttr(mlir::Attribute elt, mlir::MLIRContext *ctx,
                        mlir::ModuleOp moduleOp,
                        const mlir::TypeConverter *converter) {
  if (auto gv = mlir::dyn_cast<cir::GlobalViewAttr>(elt)) {
    if (!moduleOp || !globalViewMatchesPointerLeaf(gv, moduleOp, converter))
      return std::nullopt;
    return gv.getSymbol();
  }
  if (auto nullPtr = mlir::dyn_cast<cir::ConstPtrAttr>(elt)) {
    if (nullPtr.isNullValue())
      return mlir::LLVM::ZeroAttr::get(ctx);
    return std::nullopt;
  }
  return std::nullopt;
}

static bool containsPoison(mlir::Attribute attr) {
  if (mlir::isa<cir::PoisonAttr>(attr))
    return true;
  if (auto elts = mlir::dyn_cast<mlir::ArrayAttr>(attr))
    return llvm::any_of(elts, containsPoison);
  if (auto constArr = mlir::dyn_cast<cir::ConstArrayAttr>(attr)) {
    if (mlir::isa<mlir::StringAttr>(constArr.getElts()))
      return false;
    if (auto elts = mlir::dyn_cast<mlir::ArrayAttr>(constArr.getElts()))
      return llvm::any_of(elts, containsPoison);
  }
  return false;
}

std::optional<mlir::Attribute>
lowerConstArrayAttr(cir::ConstArrayAttr constArr,
                    const mlir::TypeConverter *converter,
                    mlir::ModuleOp moduleOp) {
  // Ensure ConstArrayAttr has a type.
  const auto typedConstArr = mlir::cast<mlir::TypedAttr>(constArr);

  // Ensure ConstArrayAttr type is a ArrayType.
  const auto cirArrayType = mlir::cast<cir::ArrayType>(typedConstArr.getType());

  // Is a ConstArrayAttr with an cir::ArrayType: fetch element type.
  mlir::Type type = cirArrayType;
  auto dims = llvm::SmallVector<int64_t, 2>{};
  while (auto arrayType = mlir::dyn_cast<cir::ArrayType>(type)) {
    dims.push_back(arrayType.getSize());
    type = arrayType.getElementType();
  }

  if (containsPoison(constArr))
    return std::nullopt;

  if (mlir::isa<mlir::StringAttr>(constArr.getElts()))
    return convertStringAttrToDenseElementsAttr(constArr,
                                                converter->convertType(type));
  if (mlir::isa<cir::IntType>(type))
    return convertToDenseElementsAttr<cir::IntAttr, mlir::APInt>(
        constArr, dims, type, converter->convertType(type));

  if (mlir::isa<cir::BoolType>(type))
    return convertToDenseElementsAttr<cir::IntAttr, mlir::APInt>(
        constArr, dims, type, converter->convertType(type));

  if (mlir::isa<cir::FPTypeInterface>(type))
    return convertToDenseElementsAttr<cir::FPAttr, mlir::APFloat>(
        constArr, dims, type, converter->convertType(type));

  if (mlir::isa<cir::PointerType>(type)) {
    // FIXME: Pointer arrays with trailing_zeros (null-sentinel tables) fall
    // through to the insertvalue path for now.
    if (constArr.getTrailingZerosNum() > 0)
      return std::nullopt;
    auto eltsArr = mlir::dyn_cast<mlir::ArrayAttr>(constArr.getElts());
    if (!eltsArr)
      return std::nullopt;
    llvm::SmallVector<mlir::Attribute> lowered;
    lowered.reserve(eltsArr.size());
    mlir::MLIRContext *ctx = constArr.getContext();
    for (mlir::Attribute elt : eltsArr) {
      std::optional<mlir::Attribute> llvmElt =
          lowerPointerElementAttr(elt, ctx, moduleOp, converter);
      if (!llvmElt)
        return std::nullopt;
      lowered.push_back(*llvmElt);
    }
    return mlir::ArrayAttr::get(ctx, lowered);
  }

  return std::nullopt;
}

mlir::Value getConstAPInt(mlir::OpBuilder &bld, mlir::Location loc,
                          mlir::Type typ, const llvm::APInt &val) {
  return mlir::LLVM::ConstantOp::create(bld, loc, typ, val);
}

mlir::Value getConst(mlir::OpBuilder &bld, mlir::Location loc, mlir::Type typ,
                     unsigned val) {
  return mlir::LLVM::ConstantOp::create(bld, loc, typ, val);
}

mlir::Value createShL(mlir::OpBuilder &bld, mlir::Value lhs, unsigned rhs) {
  if (!rhs)
    return lhs;
  mlir::Value rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return mlir::LLVM::ShlOp::create(bld, lhs.getLoc(), lhs, rhsVal);
}

mlir::Value createAShR(mlir::OpBuilder &bld, mlir::Value lhs, unsigned rhs) {
  if (!rhs)
    return lhs;
  mlir::Value rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return mlir::LLVM::AShrOp::create(bld, lhs.getLoc(), lhs, rhsVal);
}

mlir::Value createAnd(mlir::OpBuilder &bld, mlir::Value lhs,
                      const llvm::APInt &rhs) {
  mlir::Value rhsVal = getConstAPInt(bld, lhs.getLoc(), lhs.getType(), rhs);
  return mlir::LLVM::AndOp::create(bld, lhs.getLoc(), lhs, rhsVal);
}

mlir::Value createLShR(mlir::OpBuilder &bld, mlir::Value lhs, unsigned rhs) {
  if (!rhs)
    return lhs;
  mlir::Value rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return mlir::LLVM::LShrOp::create(bld, lhs.getLoc(), lhs, rhsVal);
}
