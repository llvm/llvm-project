//===- EmitCBase.td - EmitC dialect ------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the AIIR EmitC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_EMITC_IR_EMITCBASE
#define AIIR_DIALECT_EMITC_IR_EMITCBASE

include "aiir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// EmitC dialect definition
//===----------------------------------------------------------------------===//

def EmitC_Dialect : Dialect {
  let name = "emitc";
  let cppNamespace = "::aiir::emitc";

  let summary = "Dialect to generate C/C++ from AIIR.";
  let description = [{
    [include "Dialects/emitc.md"]
  }];

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // AIIR_DIALECT_EMITC_IR_EMITCBASE
