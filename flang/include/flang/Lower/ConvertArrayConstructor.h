//===-- ConvertArrayConstructor.h -- Array constructor lowering -*- C++ -*-===//
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
/// Implements the conversion from evaluate::ArrayConstructor to HLFIR.
///
//===----------------------------------------------------------------------===//
#ifndef FORTRAN_LOWER_CONVERTARRAYCONSTRUCTOR_H
#define FORTRAN_LOWER_CONVERTARRAYCONSTRUCTOR_H

#include "flang/Evaluate/type.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"

namespace Fortran::evaluate {
template <typename T>
class ArrayConstructor;
}

namespace Fortran::lower {
class AbstractConverter;
class SymMap;
class StatementContext;

/// Class to lower evaluate::ArrayConstructor<T> to hlfir::EntityWithAttributes.
template <typename T>
class ArrayConstructorBuilder {
public:
  static hlfir::EntityWithAttributes
  gen(mlir::Location loc, Fortran::lower::AbstractConverter &converter,
      const Fortran::evaluate::ArrayConstructor<T> &expr,
      Fortran::lower::SymMap &symMap,
      Fortran::lower::StatementContext &stmtCtx);
};
using namespace evaluate;
FOR_EACH_SPECIFIC_TYPE(extern template class ArrayConstructorBuilder, )
} // namespace Fortran::lower

#endif // FORTRAN_LOWER_CONVERTARRAYCONSTRUCTOR_H
