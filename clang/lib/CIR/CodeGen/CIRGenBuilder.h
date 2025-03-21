//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENBUILDER_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENBUILDER_H

#include "CIRGenTypeCache.h"
#include "clang/CIR/MissingFeatures.h"

#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/MissingFeatures.h"

namespace clang::CIRGen {

class CIRGenBuilderTy : public cir::CIRBaseBuilderTy {
  const CIRGenTypeCache &typeCache;

public:
  CIRGenBuilderTy(mlir::MLIRContext &mlirContext, const CIRGenTypeCache &tc)
      : CIRBaseBuilderTy(mlirContext), typeCache(tc) {}

  cir::LongDoubleType getLongDoubleTy(const llvm::fltSemantics &format) const {
    if (&format == &llvm::APFloat::IEEEdouble())
      return cir::LongDoubleType::get(getContext(), typeCache.DoubleTy);
    if (&format == &llvm::APFloat::x87DoubleExtended())
      return cir::LongDoubleType::get(getContext(), typeCache.FP80Ty);
    if (&format == &llvm::APFloat::IEEEquad())
      return cir::LongDoubleType::get(getContext(), typeCache.FP128Ty);
    if (&format == &llvm::APFloat::PPCDoubleDouble())
      llvm_unreachable("NYI: PPC double-double format for long double");
    llvm_unreachable("Unsupported format for long double");
  }

  bool isSized(mlir::Type ty) {
    if (mlir::isa<cir::PointerType, cir::ArrayType, cir::BoolType,
                  cir::IntType>(ty))
      return true;

    assert(!cir::MissingFeatures::unsizedTypes());
    return false;
  }

  // Return true if the value is a null constant such as null pointer, (+0.0)
  // for floating-point or zero initializer
  bool isNullValue(mlir::Attribute attr) const {
    if (mlir::isa<cir::ZeroAttr>(attr))
      return true;

    if (const auto ptrVal = mlir::dyn_cast<cir::ConstPtrAttr>(attr))
      return ptrVal.isNullValue();

    if (const auto intVal = mlir::dyn_cast<cir::IntAttr>(attr))
      return intVal.isNullValue();

    if (const auto boolVal = mlir::dyn_cast<cir::BoolAttr>(attr))
      return !boolVal.getValue();

    if (auto fpAttr = mlir::dyn_cast<cir::FPAttr>(attr)) {
      auto fpVal = fpAttr.getValue();
      bool ignored;
      llvm::APFloat fv(+0.0);
      fv.convert(fpVal.getSemantics(), llvm::APFloat::rmNearestTiesToEven,
                 &ignored);
      return fv.bitwiseIsEqual(fpVal);
    }

    if (const auto arrayVal = mlir::dyn_cast<cir::ConstArrayAttr>(attr)) {
      if (mlir::isa<mlir::StringAttr>(arrayVal.getElts()))
        return false;
      for (const auto elt : mlir::cast<mlir::ArrayAttr>(arrayVal.getElts())) {
        if (!isNullValue(elt))
          return false;
      }
      return true;
    }
    return false;
  }

  bool isInt(mlir::Type i) { return mlir::isa<cir::IntType>(i); }

  // Creates constant nullptr for pointer type ty.
  cir::ConstantOp getNullPtr(mlir::Type ty, mlir::Location loc) {
    assert(!cir::MissingFeatures::targetCodeGenInfoGetNullPointer());
    return create<cir::ConstantOp>(loc, ty, getConstPtrAttr(ty, 0));
  }
};

} // namespace clang::CIRGen

#endif
