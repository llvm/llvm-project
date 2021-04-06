//===-- Lower/CharacterRuntime.h -- lower CHARACTER operations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CHARACTERRUNTIME_H
#define FORTRAN_LOWER_CHARACTERRUNTIME_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace fir {
class ExtendedValue;
}

namespace Fortran {
namespace lower {
class FirOpBuilder;

/// Generate call to a character comparison for two ssa-values of type
/// `boxchar`.
mlir::Value genCharCompare(FirOpBuilder &builder, mlir::Location loc,
                           mlir::CmpIPredicate cmp,
                           const fir::ExtendedValue &lhs,
                           const fir::ExtendedValue &rhs);

/// Generate call to a character comparison op for two unboxed variables. There
/// are 4 arguments, 2 for the lhs and 2 for the rhs. Each CHARACTER must pass a
/// reference to its buffer (`ref<char<K>>`) and its LEN type parameter (some
/// integral type).
mlir::Value genRawCharCompare(FirOpBuilder &builder, mlir::Location loc,
                              mlir::CmpIPredicate cmp, mlir::Value lhsBuff,
                              mlir::Value lhsLen, mlir::Value rhsBuff,
                              mlir::Value rhsLen);

/// Generate call to INDEX runtime.
/// This calls the simple runtime entry points based on the KIND of the string.
/// No descriptors are used.
mlir::Value genIndex(FirOpBuilder &builder, mlir::Location loc, int kind,
                     mlir::Value stringBase, mlir::Value stringLen,
                     mlir::Value substringBase, mlir::Value substringLen,
                     mlir::Value back);

/// Generate call to INDEX runtime.
/// This calls the descriptor based runtime call implementation for the index
/// intrinsic.
void genIndexDescriptor(FirOpBuilder &builder, mlir::Location loc,
                        mlir::Value resultBox, mlir::Value stringBox,
                        mlir::Value substringBox, mlir::Value backOpt,
                        mlir::Value kind);

/// Generate call to trim runtime.
///   \p resultBox must be an unallocated allocatable used for the temporary
///   result. \p stringBox must be a fir.box describing trim string argument.
/// The runtime will always allocate the resultBox.
void genTrim(Fortran::lower::FirOpBuilder &builder, mlir::Location loc,
             mlir::Value resultBox, mlir::Value stringBox);

/// Generate call to scan runtime.
/// This calls the descriptor based runtime call implementation of the scan
/// intrinsics.
void genScanDescriptor(Fortran::lower::FirOpBuilder &builder, 
                       mlir::Location loc,
                       mlir::Value resultBox, mlir::Value stringBox,
                       mlir::Value setBox, mlir::Value backBox, 
                       mlir::Value kind);

/// Generate call to the scan runtime routine that is specialized on 
/// \param kind.
/// The \param kind represents the kind of the elements in the strings.
mlir::Value genScan(Fortran::lower::FirOpBuilder &builder,
                    mlir::Location loc, int kind, mlir::Value stringBase,
                    mlir::Value stringLen, mlir::Value setBase,
                    mlir::Value setLen, mlir::Value back);

/// Generate call to verify runtime.
/// This calls the descriptor based runtime call implementation of the scan
/// intrinsics.
void genVerifyDescriptor(Fortran::lower::FirOpBuilder &builder, 
                         mlir::Location loc,
                         mlir::Value resultBox, mlir::Value stringBox,
                         mlir::Value setBox, mlir::Value backBox, 
                       mlir::Value kind);

/// Generate call to the verify runtime routine that is specialized on 
/// \param kind.
/// The \param kind represents the kind of the elements in the strings.
mlir::Value genVerify(Fortran::lower::FirOpBuilder &builder,
                      mlir::Location loc, int kind, mlir::Value stringBase,
                      mlir::Value stringLen, mlir::Value setBase,
                      mlir::Value setLen, mlir::Value back);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CHARACTERRUNTIME_H
