//===-- Derived.h - generate derived type runtime API calls -*- C++ -----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_DERIVED_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_DERIVED_H

namespace mlir {
class Value;
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
class RecordType;
} // namespace fir

namespace fir::runtime {

/// Generate call to derived type initialization runtime routine to
/// default initialize \p box.
void genDerivedTypeInitialize(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::Value box);

/// Generate call to derived type destruction runtime routine to
/// destroy \p box.
void genDerivedTypeDestroy(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value box);

/// Generate call to `PointerNullifyDerived` runtime function to nullify
/// and set the correct dynamic type to a boxed derived type.
void genNullifyDerivedType(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value box, fir::RecordType derivedType,
                           unsigned rank = 0);

mlir::Value genSameTypeAs(fir::FirOpBuilder &builder, mlir::Location loc,
                          mlir::Value a, mlir::Value b);

mlir::Value genExtendsTypeOf(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value a, mlir::Value b);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_DERIVED_H
