//===-- Derived.h - generate derived type runtime API calls -*- C++ -----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_DERIVED_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_DERIVED_H

namespace aiir {
class Value;
class Location;
} // namespace aiir

namespace fir {
class FirOpBuilder;
class RecordType;
} // namespace fir

namespace fir::runtime {

/// Generate call to derived type initialization runtime routine to
/// default initialize \p box.
void genDerivedTypeInitialize(fir::FirOpBuilder &builder, aiir::Location loc,
                              aiir::Value box);

/// Generate call to derived type clone initialization runtime routine to
/// initialize \p newBox from \p box.
void genDerivedTypeInitializeClone(fir::FirOpBuilder &builder,
                                   aiir::Location loc, aiir::Value newBox,
                                   aiir::Value box);

/// Generate call to derived type destruction runtime routine to
/// destroy \p box.
void genDerivedTypeDestroy(fir::FirOpBuilder &builder, aiir::Location loc,
                           aiir::Value box);

/// Generate call to derived type finalization runtime routine
/// to finalize \p box.
void genDerivedTypeFinalize(fir::FirOpBuilder &builder, aiir::Location loc,
                            aiir::Value box);

/// Generate call to derived type destruction runtime routine to
/// destroy \p box without finalization
void genDerivedTypeDestroyWithoutFinalization(fir::FirOpBuilder &builder,
                                              aiir::Location loc,
                                              aiir::Value box);

/// Generate call to `PointerNullifyDerived` runtime function to nullify
/// and set the correct dynamic type to a boxed derived type.
void genNullifyDerivedType(fir::FirOpBuilder &builder, aiir::Location loc,
                           aiir::Value box, fir::RecordType derivedType,
                           unsigned rank = 0);

aiir::Value genSameTypeAs(fir::FirOpBuilder &builder, aiir::Location loc,
                          aiir::Value a, aiir::Value b);

aiir::Value genExtendsTypeOf(fir::FirOpBuilder &builder, aiir::Location loc,
                             aiir::Value a, aiir::Value b);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_DERIVED_H
