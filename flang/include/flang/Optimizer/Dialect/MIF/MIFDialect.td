//===-- MIFDialect.td - MIF dialect base definitions - tablegen ---------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the Multi-Image Fortran (MIF)  dialect
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_MIF_MIFDIALECT
#define FORTRAN_DIALECT_MIF_MIFDIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def MIFDialect : Dialect {
  let name = "mif";

  let summary = "Multi-Image Fortran dialect";

  let description = [{
    The "MIF" dialect is designed to contain the basic coarray operations
    in Fortran and all multi image operations as descibed in the standard.
    This includes synchronization operations, atomic operations,
    image queries, teams, criticals, etc. The MIF dialect operations use 
    the FIR types and are tightly coupled with FIR and HLFIR.
  }];

  let cppNamespace = "::mif";
  let dependentDialects = ["fir::FIROpsDialect"];
}

#endif // FORTRAN_DIALECT_MIF_MIFDIALECT
