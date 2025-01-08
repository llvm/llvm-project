//===-- ConvertConstant.h -- lowering of constants --------------*- C++ -*-===//
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
///
/// Implements the conversion from evaluate::Constant to FIR.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CONVERTCONSTANT_H
#define FORTRAN_LOWER_CONVERTCONSTANT_H

#include "flang/Evaluate/constant.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"

namespace Fortran::lower {
class AbstractConverter;

/// Class to lower evaluate::Constant to fir::ExtendedValue.
template <typename T>
class ConstantBuilder {
public:
  /// Lower \p constant into a fir::ExtendedValue.
  /// If \p outlineBigConstantsInReadOnlyMemory is set, character, derived
  /// type, and array constants will be lowered into read only memory
  /// fir.global, and the resulting fir::ExtendedValue will contain the address
  /// of the fir.global. This option should not be set if the constant is being
  /// lowered while the builder is already in a fir.global body because
  /// fir.global initialization body cannot contain code manipulating memory
  /// (e.g.  fir.load/fir.store...).
  static fir::ExtendedValue gen(Fortran::lower::AbstractConverter &converter,
                                mlir::Location loc,
                                const evaluate::Constant<T> &constant,
                                bool outlineBigConstantsInReadOnlyMemory);
};
using namespace evaluate;
FOR_EACH_SPECIFIC_TYPE(extern template class ConstantBuilder, )

template <typename T>
fir::ExtendedValue convertConstant(Fortran::lower::AbstractConverter &converter,
                                   mlir::Location loc,
                                   const evaluate::Constant<T> &constant,
                                   bool outlineBigConstantsInReadOnlyMemory) {
  return ConstantBuilder<T>::gen(converter, loc, constant,
                                 outlineBigConstantsInReadOnlyMemory);
}

/// Create a fir.global array with a dense attribute containing the value of
/// \p initExpr.
/// Using a dense attribute allows faster MLIR compilation times compared to
/// creating an initialization body for the initial value. However, a dense
/// attribute can only be created if initExpr is a non-empty rank 1 numerical or
/// logical Constant<T>. Otherwise, the value returned will be null.
fir::GlobalOp tryCreatingDenseGlobal(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Type symTy,
                                     llvm::StringRef globalName,
                                     mlir::StringAttr linkage, bool isConst,
                                     const Fortran::lower::SomeExpr &initExpr,
                                     cuf::DataAttributeAttr dataAttr = {});

/// Lower a StructureConstructor that must be lowered in read only data although
/// it may not be wrapped into a Constant<T> (this may be the case for derived
/// type descriptor compiler generated data that is not fully compliant with
/// Fortran constant expression but can and must still be lowered into read only
/// memory).
fir::ExtendedValue
genInlinedStructureCtorLit(Fortran::lower::AbstractConverter &converter,
                           mlir::Location loc,
                           const Fortran::evaluate::StructureConstructor &ctor);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_CONVERTCONSTANT_H
