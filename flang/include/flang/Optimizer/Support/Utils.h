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

#include "flang/Common/default-kinds.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace fir {
/// Return the integer value of a arith::ConstantOp.
inline std::int64_t toInt(mlir::arith::ConstantOp cop) {
  return mlir::cast<mlir::IntegerAttr>(cop.getValue())
      .getValue()
      .getSExtValue();
}

// Reconstruct binding tables for dynamic dispatch.
using BindingTable = llvm::DenseMap<llvm::StringRef, unsigned>;
using BindingTables = llvm::DenseMap<llvm::StringRef, BindingTable>;

inline void buildBindingTables(BindingTables &bindingTables,
                               mlir::ModuleOp mod) {

  // The binding tables are defined in FIR after lowering inside fir.type_info
  // operations. Go through each binding tables and store the procedure name and
  // binding index for later use by the fir.dispatch conversion pattern.
  for (auto typeInfo : mod.getOps<fir::TypeInfoOp>()) {
    unsigned bindingIdx = 0;
    BindingTable bindings;
    if (typeInfo.getDispatchTable().empty()) {
      bindingTables[typeInfo.getSymName()] = bindings;
      continue;
    }
    for (auto dtEntry :
         typeInfo.getDispatchTable().front().getOps<fir::DTEntryOp>()) {
      bindings[dtEntry.getMethod()] = bindingIdx;
      ++bindingIdx;
    }
    bindingTables[typeInfo.getSymName()] = bindings;
  }
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

inline std::string numericMlirTypeToFortran(fir::FirOpBuilder &builder,
                                            mlir::Type type, mlir::Location loc,
                                            const llvm::Twine &name) {
  if (type.isF16())
    return "REAL(KIND=2)";
  else if (type.isBF16())
    return "REAL(KIND=3)";
  else if (type.isTF32())
    return "REAL(KIND=unknown)";
  else if (type.isF32())
    return "REAL(KIND=4)";
  else if (type.isF64())
    return "REAL(KIND=8)";
  else if (type.isF80())
    return "REAL(KIND=10)";
  else if (type.isF128())
    return "REAL(KIND=16)";
  else if (type.isInteger(8))
    return "INTEGER(KIND=1)";
  else if (type.isInteger(16))
    return "INTEGER(KIND=2)";
  else if (type.isInteger(32))
    return "INTEGER(KIND=4)";
  else if (type.isInteger(64))
    return "INTEGER(KIND=8)";
  else if (type.isInteger(128))
    return "INTEGER(KIND=16)";
  else if (type == fir::ComplexType::get(builder.getContext(), 2))
    return "COMPLEX(KIND=2)";
  else if (type == fir::ComplexType::get(builder.getContext(), 3))
    return "COMPLEX(KIND=3)";
  else if (type == fir::ComplexType::get(builder.getContext(), 4))
    return "COMPLEX(KIND=4)";
  else if (type == fir::ComplexType::get(builder.getContext(), 8))
    return "COMPLEX(KIND=8)";
  else if (type == fir::ComplexType::get(builder.getContext(), 10))
    return "COMPLEX(KIND=10)";
  else if (type == fir::ComplexType::get(builder.getContext(), 16))
    return "COMPLEX(KIND=16)";
  else
    fir::emitFatalError(loc, "unsupported type in " + name + ": " +
                                 fir::mlirTypeToString(type));
}

inline void intrinsicTypeTODO(fir::FirOpBuilder &builder, mlir::Type type,
                              mlir::Location loc,
                              const llvm::Twine &intrinsicName) {
  TODO(loc,
       "intrinsic: " +
           fir::numericMlirTypeToFortran(builder, type, loc, intrinsicName) +
           " in " + intrinsicName);
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

} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_UTILS_H
