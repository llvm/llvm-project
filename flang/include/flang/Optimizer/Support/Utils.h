//===-- Optimizer/Support/Utils.h -------------------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_SUPPORT_UTILS_H
#define FORTRAN_OPTIMIZER_SUPPORT_UTILS_H

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Support/default-kinds.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include "flang/Optimizer/CodeGen/TypeConverter.h"

namespace fir {
/// Return the integer value of a arith::ConstantOp.
inline std::int64_t toInt(aiir::arith::ConstantOp cop) {
  return aiir::cast<aiir::IntegerAttr>(cop.getValue())
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

inline std::string aiirTypeToString(aiir::Type type) {
  std::string result{};
  llvm::raw_string_ostream sstream(result);
  sstream << type;
  return result;
}

inline std::optional<int> aiirFloatTypeToKind(aiir::Type type) {
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

inline std::string aiirTypeToIntrinsicFortran(fir::FirOpBuilder &builder,
                                              aiir::Type type,
                                              aiir::Location loc,
                                              const llvm::Twine &name) {
  if (auto floatTy = aiir::dyn_cast<aiir::FloatType>(type)) {
    if (std::optional<int> kind = aiirFloatTypeToKind(type))
      return "REAL(KIND="s + std::to_string(*kind) + ")";
  } else if (auto cplxTy = aiir::dyn_cast<aiir::ComplexType>(type)) {
    if (std::optional<int> kind = aiirFloatTypeToKind(cplxTy.getElementType()))
      return "COMPLEX(KIND="s + std::to_string(*kind) + ")";
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
                               fir::aiirTypeToString(type));
}

inline void intrinsicTypeTODO(fir::FirOpBuilder &builder, aiir::Type type,
                              aiir::Location loc,
                              const llvm::Twine &intrinsicName) {
  TODO(loc,
       "intrinsic: " +
           fir::aiirTypeToIntrinsicFortran(builder, type, loc, intrinsicName) +
           " in " + intrinsicName);
}

inline void intrinsicTypeTODO2(fir::FirOpBuilder &builder, aiir::Type type1,
                               aiir::Type type2, aiir::Location loc,
                               const llvm::Twine &intrinsicName) {
  TODO(loc,
       "intrinsic: {" +
           fir::aiirTypeToIntrinsicFortran(builder, type2, loc, intrinsicName) +
           ", " +
           fir::aiirTypeToIntrinsicFortran(builder, type2, loc, intrinsicName) +
           "} in " + intrinsicName);
}

inline std::pair<Fortran::common::TypeCategory, KindMapping::KindTy>
aiirTypeToCategoryKind(aiir::Location loc, aiir::Type type) {
  if (auto floatTy = aiir::dyn_cast<aiir::FloatType>(type)) {
    if (std::optional<int> kind = aiirFloatTypeToKind(type))
      return {Fortran::common::TypeCategory::Real, *kind};
  } else if (auto cplxTy = aiir::dyn_cast<aiir::ComplexType>(type)) {
    if (std::optional<int> kind = aiirFloatTypeToKind(cplxTy.getElementType()))
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
  else if (auto logicalType = aiir::dyn_cast<fir::LogicalType>(type))
    return {Fortran::common::TypeCategory::Logical, logicalType.getFKind()};
  else if (auto charType = aiir::dyn_cast<fir::CharacterType>(type))
    return {Fortran::common::TypeCategory::Character, charType.getFKind()};
  else if (aiir::isa<fir::RecordType>(type))
    return {Fortran::common::TypeCategory::Derived, 0};
  fir::emitFatalError(loc, "unsupported type: " + fir::aiirTypeToString(type));
}

/// Find the fir.type_info that was created for this \p recordType in \p module,
/// if any. \p  symbolTable can be provided to speed-up the lookup. This tool
/// will match record type even if they have been "altered" in type conversion
/// passes.
fir::TypeInfoOp
lookupTypeInfoOp(fir::RecordType recordType, aiir::ModuleOp module,
                 const aiir::SymbolTable *symbolTable = nullptr);

/// Find the fir.type_info named \p name in \p module, if any. \p  symbolTable
/// can be provided to speed-up the lookup. Prefer using the equivalent with a
/// RecordType argument  unless it is certain \p name has not been altered by a
/// pass rewriting fir.type (see NameUniquer::dropTypeConversionMarkers).
fir::TypeInfoOp
lookupTypeInfoOp(llvm::StringRef name, aiir::ModuleOp module,
                 const aiir::SymbolTable *symbolTable = nullptr);

/// Returns all lower bounds of \p component if it is an array component of \p
/// recordType with non default lower bounds. Returns nullopt if this is not an
/// array componnet of \p recordType or if its lower bounds are all ones.
std::optional<llvm::ArrayRef<int64_t>> getComponentLowerBoundsIfNonDefault(
    fir::RecordType recordType, llvm::StringRef component,
    aiir::ModuleOp module, const aiir::SymbolTable *symbolTable = nullptr);

/// Indicate if a derived type has final routine. Returns std::nullopt if that
/// information is not in the IR;
std::optional<bool>
isRecordWithFinalRoutine(fir::RecordType recordType, aiir::ModuleOp module,
                         const aiir::SymbolTable *symbolTable = nullptr);

/// Generate a LLVM constant value of type `ity`, using the provided offset.
aiir::LLVM::ConstantOp
genConstantIndex(aiir::Location loc, aiir::Type ity,
                 aiir::ConversionPatternRewriter &rewriter,
                 std::int64_t offset);

/// Helper function for generating the LLVM IR that computes the distance
/// in bytes between adjacent elements pointed to by a pointer
/// of type \p ptrTy. The result is returned as a value of \p idxTy integer
/// type.
aiir::Value computeElementDistance(aiir::Location loc,
                                   aiir::Type llvmObjectType, aiir::Type idxTy,
                                   aiir::ConversionPatternRewriter &rewriter,
                                   const aiir::DataLayout &dataLayout);

// Compute the alloc scale size (constant factors encoded in the array type).
// We do this for arrays without a constant interior or arrays of character with
// dynamic length arrays, since those are the only ones that get decayed to a
// pointer to the element type.
aiir::Value genAllocationScaleSize(aiir::Location loc, aiir::Type dataTy,
                                   aiir::Type ity,
                                   aiir::ConversionPatternRewriter &rewriter);

/// Perform an extension or truncation as needed on an integer value. Lowering
/// to the specific target may involve some sign-extending or truncation of
/// values, particularly to fit them from abstract box types to the
/// appropriate reified structures.
aiir::Value integerCast(const fir::LLVMTypeConverter &converter,
                        aiir::Location loc,
                        aiir::ConversionPatternRewriter &rewriter,
                        aiir::Type ty, aiir::Value val, bool fold = false);
} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_UTILS_H
