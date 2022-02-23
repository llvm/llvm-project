//===- Lower/ConvertVariable.h -- lowering of variables to FIR --*- C++ -*-===//
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
/// Instantiation of pft::Variable in FIR/MLIR.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CONVERT_VARIABLE_H
#define FORTRAN_LOWER_CONVERT_VARIABLE_H

namespace Fortran ::lower {
class AbstractConverter;
class CallerInterface;
class SymMap;
namespace pft {
struct Variable;
}

/// Instantiate variable \p var and add it to \p symMap.
/// The AbstractConverter builder must be set.
/// The AbstractConverter own symbol mapping is not used during the
/// instantiation and can be different form \p symMap.
void instantiateVariable(AbstractConverter &, const pft::Variable &var,
                         SymMap &symMap);

/// Instantiate the variables that appear in the specification expressions
/// of the result of a function call. The instantiated variables are added
/// to \p symMap.
void mapCallInterfaceSymbols(AbstractConverter &,
                             const Fortran::lower::CallerInterface &caller,
                             SymMap &symMap);

} // namespace Fortran::lower
#endif // FORTRAN_LOWER_CONVERT_VARIABLE_H
