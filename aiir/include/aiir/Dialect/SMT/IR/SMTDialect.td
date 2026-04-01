//===- SMTDialect.td - SMT dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SMT_IR_SMTDIALECT_TD
#define AIIR_DIALECT_SMT_IR_SMTDIALECT_TD

include "aiir/IR/DialectBase.td"

def SMTDialect : Dialect {
  let name = "smt";
  let summary = "a dialect that models satisfiability modulo theories";
  let cppNamespace = "aiir::smt";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
  }];
}

#endif // AIIR_DIALECT_SMT_IR_SMTDIALECT_TD
