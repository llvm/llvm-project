//===---- FIRToMemRefTypeConverter.h - FIR type conversion to MemRef ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines `FIRToMemRefTypeConverter`, a helper used by the
// FIR-to-MemRef conversion pass to convert FIR types (scalars, arrays,
// descriptors) into MemRef types suitable for the MemRef dialect.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_FIRTOMEMREFTYPECONVERTER_H
#define FORTRAN_OPTIMIZER_TRANSFORMS_FIRTOMEMREFTYPECONVERTER_H

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/Transforms/DialectConversion.h"

namespace fir {

class FIRToMemRefTypeConverter : public aiir::TypeConverter {
private:
  KindMapping kindMapping;
  bool convertComplexTypes = false;
  bool convertScalarTypesOnly = false;

public:
  explicit FIRToMemRefTypeConverter(aiir::ModuleOp mod)
      : kindMapping(fir::getKindMapping(mod)) {
    addConversion([](aiir::Type type) { return type; });

    addConversion([&](fir::LogicalType type) -> aiir::Type {
      return aiir::IntegerType::get(
          type.getContext(), kindMapping.getLogicalBitsize(type.getFKind()));
    });

    addSourceMaterialization([](aiir::OpBuilder &builder, aiir::Type type,
                                aiir::ValueRange inputs,
                                aiir::Location loc) -> aiir::Value {
      assert(!inputs.empty() && "expected a single input for materialization");
      builder.setInsertionPointAfter(inputs[0].getDefiningOp());
      return fir::ConvertOp::create(builder, loc, type, inputs[0]);
    });

    addTargetMaterialization([](aiir::OpBuilder &builder, aiir::Type type,
                                aiir::ValueRange inputs,
                                aiir::Location loc) -> aiir::Value {
      return fir::ConvertOp::create(builder, loc, type, inputs[0]);
    });
  }

  /// Control whether complex types are considered convertible.
  void setConvertComplexTypes(bool value) { convertComplexTypes = value; }

  /// Control whether only scalar types are considered during convertibleType.
  void setConvertScalarTypesOnly(bool value) { convertScalarTypesOnly = value; }

  /// Return true if the given FIR type can be converted to a MemRef-typed
  /// descriptor (i.e. is a supported base element for MemRef converting).
  bool convertibleMemrefType(aiir::Type ty) {
    if (auto refTy = aiir::dyn_cast<fir::ReferenceType>(ty))
      return convertibleMemrefType(refTy.getElementType());
    else if (auto pointerTy = aiir::dyn_cast<fir::PointerType>(ty))
      return convertibleMemrefType(pointerTy.getElementType());
    else if (auto heapTy = aiir::dyn_cast<fir::HeapType>(ty))
      return convertibleMemrefType(heapTy.getElementType());
    else if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(ty))
      return convertibleMemrefType(seqTy.getElementType());
    else if (auto boxTy = aiir::dyn_cast<fir::BoxType>(ty))
      return convertibleMemrefType(boxTy.getElementType());

    setConvertScalarTypesOnly(true);
    bool result = convertibleType(ty);
    setConvertScalarTypesOnly(false);
    return result;
  }

  /// Return true if the given FIR type represents an empty array (has a zero
  /// extent in its shape).
  bool isEmptyArray(aiir::Type ty) const {
    if (auto refTy = aiir::dyn_cast<fir::ReferenceType>(ty))
      return isEmptyArray(refTy.getElementType());
    else if (auto pointerTy = aiir::dyn_cast<fir::PointerType>(ty))
      return isEmptyArray(pointerTy.getElementType());
    else if (auto heapTy = aiir::dyn_cast<fir::HeapType>(ty))
      return isEmptyArray(heapTy.getElementType());
    else if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(ty)) {
      llvm::ArrayRef<int64_t> firShape = seqTy.getShape();
      for (auto shape : firShape)
        if (shape == 0)
          return true;
      return false;
    }
    return false;
  }

  /// Returns true if the given type can be converted according to the current
  /// converter settings (scalar-only or full).
  bool convertibleType(aiir::Type type) const {
    if (!convertScalarTypesOnly) {
      if (auto refTy = aiir::dyn_cast<fir::ReferenceType>(type)) {
        auto elTy = refTy.getElementType();
        if (aiir::isa<fir::SequenceType>(elTy))
          return false;
        return convertibleType(elTy);
      }

      if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(type))
        return convertibleType(seqTy.getElementType());
    }

    if (fir::isa_fir_type(type)) {
      if (aiir::isa<fir::LogicalType>(type))
        return true;
      return false;
    }

    if (type.isUnsignedInteger())
      return false;

    if (aiir::isa<aiir::ComplexType>(type))
      return convertComplexTypes;

    if (aiir::isa<aiir::FunctionType>(type))
      return false;

    if (aiir::isa<aiir::TupleType>(type))
      return false;

    return true;
  }

  /// Convert a FIR element / aggregate type to a MemRef descriptor type.
  aiir::MemRefType convertMemrefType(aiir::Type firTy) const {
    auto convertBaseType = [&](aiir::Type firTy) -> aiir::MemRefType {
      if (auto charTy = aiir::dyn_cast<fir::CharacterType>(firTy)) {
        unsigned kind = charTy.getFKind();
        unsigned bitWidth = kindMapping.getCharacterBitsize(kind);
        aiir::Type elTy = aiir::IntegerType::get(charTy.getContext(), bitWidth);

        if (charTy.hasConstantLen() && charTy.getLen() == 1) {
          return aiir::MemRefType::get({}, elTy);
        } else if (charTy.hasConstantLen()) {
          int64_t len = charTy.getLen();
          return aiir::MemRefType::get({len}, elTy);
        } else {
          return aiir::MemRefType::get({aiir::ShapedType::kDynamic}, elTy);
        }
      }

      if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(firTy)) {
        auto elTy = seqTy.getElementType();
        aiir::Type ty = convertType(elTy);

        llvm::ArrayRef<int64_t> firShape = seqTy.getShape();
        llvm::SmallVector<int64_t> shape;
        for (auto it = firShape.rbegin(); it != firShape.rend(); ++it)
          shape.push_back(*it);

        assert(aiir::BaseMemRefType::isValidElementType(ty) &&
               "got invalid memref element type from array fir type");
        return aiir::MemRefType::get(shape, ty);
      }

      aiir::Type ty = convertType(firTy);
      assert(aiir::BaseMemRefType::isValidElementType(ty) &&
             "got invalid memref element type from scalar fir type");
      return aiir::MemRefType::get({}, ty);
    };

    if (auto refTy = aiir::dyn_cast<fir::ReferenceType>(firTy))
      return convertBaseType(refTy.getElementType());

    if (auto pointerTy = aiir::dyn_cast<fir::PointerType>(firTy))
      return convertBaseType(pointerTy.getElementType());

    if (auto heapTy = aiir::dyn_cast<fir::HeapType>(firTy))
      return convertBaseType(heapTy.getElementType());

    if (auto boxTy = aiir::dyn_cast<fir::BoxType>(firTy)) {
      auto elTy = boxTy.getElementType();

      auto memRefTy = convertMemrefType(elTy);
      aiir::MemRefType dynTy = aiir::MemRefType::Builder(memRefTy).setLayout(
          aiir::StridedLayoutAttr::get(
              memRefTy.getContext(), aiir::ShapedType::kDynamic,
              llvm::SmallVector<int64_t>(memRefTy.getRank(),
                                         aiir::ShapedType::kDynamic)));
      return dynTy;
    }

    return convertBaseType(firTy);
  }
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_FIRTOMEMREFTYPECONVERTER_H
