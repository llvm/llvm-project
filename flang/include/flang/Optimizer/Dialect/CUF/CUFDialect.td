//===-- CUFDialect.td - CUF dialect base definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the CUDA Fortran dialect
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_CUF_CUFDIALECT
#define FORTRAN_DIALECT_CUF_CUFDIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def CUFDialect : Dialect {
  let name = "cuf";

  let summary = "CUDA Fortran dialect";

  let description = [{
    This dialect models CUDA Fortran operations. The CUF dialect operations use
    the FIR types and are tightly coupled with FIR and HLFIR.
  }];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
  let cppNamespace = "::cuf";
  let dependentDialects = ["fir::FIROpsDialect"];

  let extraClassDeclaration = [{
  private:
    // Register the CUF Attributes.
    void registerAttributes();
  }];
}

#endif // FORTRAN_DIALECT_CUF_CUFDIALECT
