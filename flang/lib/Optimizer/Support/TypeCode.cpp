//===-- Optimizer/Support/TypeCode.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Support/TypeCode.h"
#include "flang/ISO_Fortran_binding.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "llvm/Support/ErrorHandling.h"

namespace fir {

/// Return the ISO_C_BINDING intrinsic module value of type \p ty.
int getTypeCode(mlir::Type ty, fir::KindMapping &kindMap) {
  unsigned width = 0;
  if (mlir::IntegerType intTy = ty.dyn_cast<mlir::IntegerType>()) {
    switch (intTy.getWidth()) {
    case 8:
      return CFI_type_int8_t;
    case 16:
      return CFI_type_int16_t;
    case 32:
      return CFI_type_int32_t;
    case 64:
      return CFI_type_int64_t;
    case 128:
      return CFI_type_int128_t;
    }
    llvm_unreachable("unsupported integer type");
  }
  if (fir::LogicalType logicalTy = ty.dyn_cast<fir::LogicalType>()) {
    switch (kindMap.getLogicalBitsize(logicalTy.getFKind())) {
    case 8:
      return CFI_type_Bool;
    case 16:
      return CFI_type_int_least16_t;
    case 32:
      return CFI_type_int_least32_t;
    case 64:
      return CFI_type_int_least64_t;
    }
    llvm_unreachable("unsupported logical type");
  }
  if (mlir::FloatType floatTy = ty.dyn_cast<mlir::FloatType>()) {
    switch (floatTy.getWidth()) {
    case 16:
      return floatTy.isBF16() ? CFI_type_bfloat : CFI_type_half_float;
    case 32:
      return CFI_type_float;
    case 64:
      return CFI_type_double;
    case 80:
      return CFI_type_extended_double;
    case 128:
      return CFI_type_float128;
    }
    llvm_unreachable("unsupported real type");
  }
  if (fir::isa_complex(ty)) {
    if (mlir::ComplexType complexTy = ty.dyn_cast<mlir::ComplexType>()) {
      mlir::FloatType floatTy =
          complexTy.getElementType().cast<mlir::FloatType>();
      if (floatTy.isBF16())
        return CFI_type_bfloat_Complex;
      width = floatTy.getWidth();
    } else if (fir::ComplexType complexTy = ty.dyn_cast<fir::ComplexType>()) {
      auto FKind = complexTy.getFKind();
      if (FKind == 3)
        return CFI_type_bfloat_Complex;
      width = kindMap.getRealBitsize(FKind);
    }
    switch (width) {
    case 16:
      return CFI_type_half_float_Complex;
    case 32:
      return CFI_type_float_Complex;
    case 64:
      return CFI_type_double_Complex;
    case 80:
      return CFI_type_extended_double_Complex;
    case 128:
      return CFI_type_float128_Complex;
    }
    llvm_unreachable("unsupported complex size");
  }
  if (fir::CharacterType charTy = ty.dyn_cast<fir::CharacterType>()) {
    switch (kindMap.getCharacterBitsize(charTy.getFKind())) {
    case 8:
      return CFI_type_char;
    case 16:
      return CFI_type_char16_t;
    case 32:
      return CFI_type_char32_t;
    }
    llvm_unreachable("unsupported character type");
  }
  if (fir::isa_ref_type(ty))
    return CFI_type_cptr;
  if (ty.isa<fir::RecordType>())
    return CFI_type_struct;
  llvm_unreachable("unsupported type");
}

} // namespace fir
