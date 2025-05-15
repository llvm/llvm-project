//===-- PRIFDialect.td - PRIF dialect base definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the PRIF (Parallel Runtime Interface for Fortran) dialect
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_PRIF_PRIFDIALECT
#define FORTRAN_DIALECT_PRIF_PRIFDIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def PRIFDialect : Dialect {
  let name = "prif";

  let summary = "PRIF (Parallel Runtime Interface for Fortran) dialect";

  let description = [{
    The "prif" dialect is designed to contain the basic coarray operations
    in Fortran as descibed in the PRIF specificatin.
    This includes synchronization operations, atomic operations,
    image queries, teams, criticals, etc. The PRIF dialect operations use 
    the FIR types and are tightly coupled with FIR and HLFIR.
  }];

  let usePropertiesForAttributes = 1;
  let cppNamespace = "::prif";
  let dependentDialects = ["fir::FIROpsDialect"];
}

#endif // FORTRAN_DIALECT_PRIF_PRIFDIALECT
