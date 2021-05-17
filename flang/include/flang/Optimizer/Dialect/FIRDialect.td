//===-- FIRDialect.td - FIR dialect definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the FIR dialect
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_FIR_DIALECT
#define FORTRAN_DIALECT_FIR_DIALECT

def fir_Dialect : Dialect {
  let name = "fir";
  let cppNamespace = "::fir";
}

#endif // FORTRAN_DIALECT_FIR_DIALECT
