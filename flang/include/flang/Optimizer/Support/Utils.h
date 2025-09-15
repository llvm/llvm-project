//===-- Optimizer/Support/Utils.h -------------------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_SUPPORT_UTILS_H
#define FORTRAN_OPTIMIZER_SUPPORT_UTILS_H

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Support/default-kinds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include "flang/Optimizer/CodeGen/TypeConverter.h"

namespace fir {
/// Return the integer value of a arith::ConstantOp.
inline std::int64_t toInt(mlir::arith::ConstantOp cop) {
  return mlir::cast<mlir::IntegerAttr>(cop.getValue())
      .getValue()
      .getSExtValue();
}

// Translate front-end KINDs for use in the IR and code gen.
inline std::vector<fir::KindTy>
fromDefaultKinds(const Fortran::common::IntrinsicTypeDefaultKinds &defKinds) {
  return {static_cast<fir::KindTy>(defKinds.GetDefaultKind(
              Fortran::common::TypeCategory::Character)),
          static_cast<fir::KindTy>(
              defKinds.GetDefaultKind(Fortran::common::TypeCategory::Complex)),
          static_cast<fir::KindTy>(defKinds.doublePrecisionKind()),
          static_cast<fir::KindTy>(
              defKinds.GetDefaultKind(Fortran::common::TypeCategory::Integer)),
          static_cast<fir::KindTy>(
              defKinds.GetDefaultKind(Fortran::common::TypeCategory::Logical)),
          static_cast<fir::KindTy>(
              defKinds.GetDefaultKind(Fortran::common::TypeCategory::Real))};
}

inline std::string mlirTypeToString(mlir::Type type) {
  std::string result{};
  llvm::raw_string_ostream sstream(result);
  sstream << type;
  return result;
}

inline std::optional<int> mlirFloatTypeToKind(mlir::Type type) {
  if (type.isF16())
    return 2;
  else if (type.isBF16())
    return 3;
  else if (type.isF32())
    return 4;
  else if (type.isF64())
    return 8;
  else if (type.isF80())
    return 10;
  else if (type.isF128())
    return 16;
  return std::nullopt;
}

inline std::string mlirTypeToIntrinsicFortran(fir::FirOpBuilder &builder,
                                              mlir::Type type,
                                              mlir::Location loc,
                                              const llvm::Twine &name) {
  if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (std::optional<int> kind = mlirFloatTypeToKind(type))
      return "REAL(KIND="s + std::to_string(*kind) + ")";
  } else if (auto cplxTy = mlir::dyn_cast<mlir::ComplexType>(type)) {
    if (std::optional<int> kind = mlirFloatTypeToKind(cplxTy.getElementType()))
      return "COMPLEX(KIND+"s + std::to_string(*kind) + ")";
  } else if (type.isUnsignedInteger()) {
    if (type.isInteger(8))
      return "UNSIGNED(KIND=1)";
    else if (type.isInteger(16))
      return "UNSIGNED(KIND=2)";
    else if (type.isInteger(32))
      return "UNSIGNED(KIND=4)";
    else if (type.isInteger(64))
      return "UNSIGNED(KIND=8)";
    else if (type.isInteger(128))
      return "UNSIGNED(KIND=16)";
  } else if (type.isInteger(8))
    return "INTEGER(KIND=1)";
  else if (type.isInteger(16))
    return "INTEGER(KIND=2)";
  else if (type.isInteger(32))
    return "INTEGER(KIND=4)";
  else if (type.isInteger(64))
    return "INTEGER(KIND=8)";
  else if (type.isInteger(128))
    return "INTEGER(KIND=16)";
  else if (type == fir::LogicalType::get(builder.getContext(), 1))
    return "LOGICAL(KIND=1)";
  else if (type == fir::LogicalType::get(builder.getContext(), 2))
    return "LOGICAL(KIND=2)";
  else if (type == fir::LogicalType::get(builder.getContext(), 4))
    return "LOGICAL(KIND=4)";
  else if (type == fir::LogicalType::get(builder.getContext(), 8))
    return "LOGICAL(KIND=8)";

  fir::emitFatalError(loc, "unsupported type in " + name + ": " +
                               fir::mlirTypeToString(type));
}

inline void intrinsicTypeTODO(fir::FirOpBuilder &builder, mlir::Type type,
                              mlir::Location loc,
                              const llvm::Twine &intrinsicName) {
  TODO(loc,
       "intrinsic: " +
           fir::mlirTypeToIntrinsicFortran(builder, type, loc, intrinsicName) +
           " in " + intrinsicName);
}

inline void intrinsicTypeTODO2(fir::FirOpBuilder &builder, mlir::Type type1,
                               mlir::Type type2, mlir::Location loc,
                               const llvm::Twine &intrinsicName) {
  TODO(loc,
       "intrinsic: {" +
           fir::mlirTypeToIntrinsicFortran(builder, type2, loc, intrinsicName) +
           ", " +
           fir::mlirTypeToIntrinsicFortran(builder, type2, loc, intrinsicName) +
           "} in " + intrinsicName);
}

inline std::pair<Fortran::common::TypeCategory, KindMapping::KindTy>
mlirTypeToCategoryKind(mlir::Location loc, mlir::Type type) {
  if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (std::optional<int> kind = mlirFloatTypeToKind(type))
      return {Fortran::common::TypeCategory::Real, *kind};
  } else if (auto cplxTy = mlir::dyn_cast<mlir::ComplexType>(type)) {
    if (std::optional<int> kind = mlirFloatTypeToKind(cplxTy.getElementType()))
      return {Fortran::common::TypeCategory::Complex, *kind};
  } else if (type.isInteger(8))
    return {type.isUnsignedInteger() ? Fortran::common::TypeCategory::Unsigned
                                     : Fortran::common::TypeCategory::Integer,
            1};
  else if (type.isInteger(16))
    return {type.isUnsignedInteger() ? Fortran::common::TypeCategory::Unsigned
                                     : Fortran::common::TypeCategory::Integer,
            2};
  else if (type.isInteger(32))
    return {type.isUnsignedInteger() ? Fortran::common::TypeCategory::Unsigned
                                     : Fortran::common::TypeCategory::Integer,
            4};
  else if (type.isInteger(64))
    return {type.isUnsignedInteger() ? Fortran::common::TypeCategory::Unsigned
                                     : Fortran::common::TypeCategory::Integer,
            8};
  else if (type.isInteger(128))
    return {type.isUnsignedInteger() ? Fortran::common::TypeCategory::Unsigned
                                     : Fortran::common::TypeCategory::Integer,
            16};
  else if (auto logicalType = mlir::dyn_cast<fir::LogicalType>(type))
    return {Fortran::common::TypeCategory::Logical, logicalType.getFKind()};
  else if (auto charType = mlir::dyn_cast<fir::CharacterType>(type))
    return {Fortran::common::TypeCategory::Character, charType.getFKind()};
  else if (mlir::isa<fir::RecordType>(type))
    return {Fortran::common::TypeCategory::Derived, 0};
  fir::emitFatalError(loc, "unsupported type: " + fir::mlirTypeToString(type));
}

/// Find the fir.type_info that was created for this \p recordType in \p module,
/// if any. \p  symbolTable can be provided to speed-up the lookup. This tool
/// will match record type even if they have been "altered" in type conversion
/// passes.
fir::TypeInfoOp
lookupTypeInfoOp(fir::RecordType recordType, mlir::ModuleOp module,
                 const mlir::SymbolTable *symbolTable = nullptr);

/// Find the fir.type_info named \p name in \p module, if any. \p  symbolTable
/// can be provided to speed-up the lookup. Prefer using the equivalent with a
/// RecordType argument  unless it is certain \p name has not been altered by a
/// pass rewriting fir.type (see NameUniquer::dropTypeConversionMarkers).
fir::TypeInfoOp
lookupTypeInfoOp(llvm::StringRef name, mlir::ModuleOp module,
                 const mlir::SymbolTable *symbolTable = nullptr);

/// Returns all lower bounds of \p component if it is an array component of \p
/// recordType with non default lower bounds. Returns nullopt if this is not an
/// array componnet of \p recordType or if its lower bounds are all ones.
std::optional<llvm::ArrayRef<int64_t>> getComponentLowerBoundsIfNonDefault(
    fir::RecordType recordType, llvm::StringRef component,
    mlir::ModuleOp module, const mlir::SymbolTable *symbolTable = nullptr);

/// Generate a LLVM constant value of type `ity`, using the provided offset.
mlir::LLVM::ConstantOp
genConstantIndex(mlir::Location loc, mlir::Type ity,
                 mlir::ConversionPatternRewriter &rewriter,
                 std::int64_t offset);

/// Helper function for generating the LLVM IR that computes the distance
/// in bytes between adjacent elements pointed to by a pointer
/// of type \p ptrTy. The result is returned as a value of \p idxTy integer
/// type.
mlir::Value computeElementDistance(mlir::Location loc,
                                   mlir::Type llvmObjectType, mlir::Type idxTy,
                                   mlir::ConversionPatternRewriter &rewriter,
                                   const mlir::DataLayout &dataLayout);

// Compute the alloc scale size (constant factors encoded in the array type).
// We do this for arrays without a constant interior or arrays of character with
// dynamic length arrays, since those are the only ones that get decayed to a
// pointer to the element type.
mlir::Value genAllocationScaleSize(mlir::Location loc, mlir::Type dataTy,
                                   mlir::Type ity,
                                   mlir::ConversionPatternRewriter &rewriter);

/// Perform an extension or truncation as needed on an integer value. Lowering
/// to the specific target may involve some sign-extending or truncation of
/// values, particularly to fit them from abstract box types to the
/// appropriate reified structures.
mlir::Value integerCast(const fir::LLVMTypeConverter &converter,
                        mlir::Location loc,
                        mlir::ConversionPatternRewriter &rewriter,
                        mlir::Type ty, mlir::Value val, bool fold = false);
} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_UTILS_H
