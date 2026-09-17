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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace fir {

class FIRToMemRefTypeConverter : public mlir::TypeConverter {
private:
  KindMapping kindMapping;
  bool convertComplexTypes = false;
  bool convertScalarTypesOnly = false;

  mlir::MemRefType convertMemrefBaseType(mlir::Type baseTy) const {
    if (auto charTy = mlir::dyn_cast<fir::CharacterType>(baseTy)) {
      unsigned kind = charTy.getFKind();
      unsigned bitWidth = kindMapping.getCharacterBitsize(kind);
      mlir::Type elTy = mlir::IntegerType::get(charTy.getContext(), bitWidth);

      if (charTy.hasConstantLen() && charTy.getLen() == 1)
        return mlir::MemRefType::get({}, elTy);
      if (charTy.hasConstantLen())
        return mlir::MemRefType::get({charTy.getLen()}, elTy);
      return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elTy);
    }

    if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(baseTy)) {
      mlir::Type ty = convertType(seqTy.getElementType());
      llvm::ArrayRef<int64_t> firShape = seqTy.getShape();
      llvm::SmallVector<int64_t> shape;
      for (auto it = firShape.rbegin(); it != firShape.rend(); ++it)
        shape.push_back(*it);
      assert(mlir::BaseMemRefType::isValidElementType(ty) &&
             "got invalid memref element type from array fir type");
      return mlir::MemRefType::get(shape, ty);
    }

    mlir::Type ty = convertType(baseTy);
    assert(mlir::BaseMemRefType::isValidElementType(ty) &&
           "got invalid memref element type from scalar fir type");
    return mlir::MemRefType::get({}, ty);
  }

public:
  explicit FIRToMemRefTypeConverter(mlir::ModuleOp mod)
      : kindMapping(fir::getKindMapping(mod)) {
    addConversion([](mlir::Type type) { return type; });

    addConversion([&](fir::LogicalType type) -> mlir::Type {
      return mlir::IntegerType::get(
          type.getContext(), kindMapping.getLogicalBitsize(type.getFKind()));
    });

    addSourceMaterialization([](mlir::OpBuilder &builder, mlir::Type type,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      assert(!inputs.empty() && "expected a single input for materialization");
      builder.setInsertionPointAfter(inputs[0].getDefiningOp());
      return fir::ConvertOp::create(builder, loc, type, inputs[0]);
    });

    addTargetMaterialization([](mlir::OpBuilder &builder, mlir::Type type,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      return fir::ConvertOp::create(builder, loc, type, inputs[0]);
    });
  }

  /// Control whether complex types are considered convertible.
  void setConvertComplexTypes(bool value) { convertComplexTypes = value; }

  /// Control whether only scalar types are considered during convertibleType.
  void setConvertScalarTypesOnly(bool value) { convertScalarTypesOnly = value; }

  /// Return true if the given FIR type can be converted to a MemRef-typed
  /// descriptor. Uses the same `dyn_cast_ptrEleTy` / `box` recursion as
  /// `convertMemrefType`. Nested pointers such as
  /// `!fir.ref<!fir.heap<!fir.array<?xf32>>>` hold a heap address, not the
  /// array, and are not convertible.
  bool convertibleMemrefType(mlir::Type ty) {
    if (mlir::Type pointee = fir::dyn_cast_ptrEleTy(ty))
      ty = pointee;
    else if (auto boxTy = mlir::dyn_cast<fir::BoxType>(ty))
      return convertibleMemrefType(boxTy.getElementType());

    // convertMemrefType peels only one pointer wrapper. A remaining pointer
    // or box is not a valid memref element.
    if (fir::isa_ref_type(ty) || fir::isa_box_type(ty))
      return false;

    if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty))
      ty = seqTy.getElementType();
    if (fir::isa_ref_type(ty) || fir::isa_box_type(ty))
      return false;

    setConvertScalarTypesOnly(true);
    bool result = convertibleType(ty);
    setConvertScalarTypesOnly(false);
    return result;
  }

  /// Return true if the given FIR type represents an empty array (has a zero
  /// extent in its shape).
  bool isEmptyArray(mlir::Type ty) const {
    if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(ty))
      return isEmptyArray(refTy.getElementType());
    else if (auto pointerTy = mlir::dyn_cast<fir::PointerType>(ty))
      return isEmptyArray(pointerTy.getElementType());
    else if (auto heapTy = mlir::dyn_cast<fir::HeapType>(ty))
      return isEmptyArray(heapTy.getElementType());
    else if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty)) {
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
  bool convertibleType(mlir::Type type) const {
    if (!convertScalarTypesOnly) {
      if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(type)) {
        auto elTy = refTy.getElementType();
        if (mlir::isa<fir::SequenceType>(elTy))
          return false;
        return convertibleType(elTy);
      }

      if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(type))
        return convertibleType(seqTy.getElementType());
    }

    if (fir::isa_fir_type(type)) {
      if (mlir::isa<fir::LogicalType>(type))
        return true;
      return false;
    }

    if (type.isUnsignedInteger())
      return false;

    if (mlir::isa<mlir::ComplexType>(type))
      return convertComplexTypes;

    if (mlir::isa<mlir::FunctionType>(type))
      return false;

    if (mlir::isa<mlir::TupleType>(type))
      return false;

    return true;
  }

  /// Convert a FIR element / aggregate type to a MemRef descriptor type.
  mlir::MemRefType convertMemrefType(mlir::Type firTy) const {
    if (mlir::Type pointee = fir::dyn_cast_ptrEleTy(firTy))
      return convertMemrefBaseType(pointee);

    if (auto boxTy = mlir::dyn_cast<fir::BoxType>(firTy)) {
      mlir::MemRefType memRefTy = convertMemrefType(boxTy.getElementType());
      return mlir::MemRefType::Builder(memRefTy).setLayout(
          mlir::StridedLayoutAttr::get(
              memRefTy.getContext(), mlir::ShapedType::kDynamic,
              llvm::SmallVector<int64_t>(memRefTy.getRank(),
                                         mlir::ShapedType::kDynamic)));
    }

    return convertMemrefBaseType(firTy);
  }
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_FIRTOMEMREFTYPECONVERTER_H
