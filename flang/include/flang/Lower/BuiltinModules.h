//===-- BuiltinModules.h --------------------------------------*- C++ -*-===//
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
/// Define information about builtin derived types from flang/module/xxx.f90
/// files so that these types can be manipulated by lowering.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_BUILTINMODULES_H
#define FORTRAN_LOWER_BUILTINMODULES_H

namespace Fortran::lower::builtin {
/// Address field name of __builtin_c_f_pointer and __builtin_c_ptr types.
constexpr char cptrFieldName[] = "__address";
} // namespace Fortran::lower::builtin

#endif // FORTRAN_LOWER_BUILTINMODULES_H
