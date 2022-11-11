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
template <typename T>
class ConstantBuilder {};

/// Class to lower intrinsic evaluate::Constant to fir::ExtendedValue.
template <common::TypeCategory TC, int KIND>
class ConstantBuilder<evaluate::Type<TC, KIND>> {
public:
  /// Lower \p constant into a fir::ExtendedValue.
  /// If \p outlineBigConstantsInReadOnlyMemory is set, character and array
  /// constants will be lowered into read only memory fir.global, and the
  /// resulting fir::ExtendedValue will contain the address of the fir.global.
  /// This option should not be set if the constant is being lowered while the
  /// builder is already in a fir.global body because fir.global initialization
  /// body cannot contain code manipulating memory (e.g. fir.load/fir.store...).
  static fir::ExtendedValue
  gen(fir::FirOpBuilder &builder, mlir::Location loc,
      const evaluate::Constant<evaluate::Type<TC, KIND>> &constant,
      bool outlineBigConstantsInReadOnlyMemory);
};

template <common::TypeCategory TC, int KIND>
using IntrinsicConstantBuilder = ConstantBuilder<evaluate::Type<TC, KIND>>;

using namespace evaluate;
FOR_EACH_INTRINSIC_KIND(extern template class ConstantBuilder, )

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
                                     const Fortran::lower::SomeExpr &initExpr);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_CONVERTCONSTANT_H
