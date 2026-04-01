//===- PDLDialect.td - PDL dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the AIIR PDL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_PDL_IR_PDLDIALECT
#define AIIR_DIALECT_PDL_IR_PDLDIALECT

include "aiir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PDL Dialect
//===----------------------------------------------------------------------===//

def PDL_Dialect : Dialect {
  let summary = "High level pattern definition dialect";
  let description = [{
    PDL presents a high level abstraction for the rewrite pattern infrastructure
    available in AIIR. This abstraction allows for representing patterns
    transforming AIIR, as AIIR. This allows for applying all of the benefits
    that the general AIIR infrastructure provides, to the infrastructure itself.
    This means that pattern matching can be more easily verified for
    correctness, targeted by frontends, and optimized.

    PDL abstracts over various different aspects of patterns and core AIIR data
    structures. Patterns are specified via a `pdl.pattern` operation. These
    operations contain a region body for the "matcher" code, and terminate with
    a `pdl.rewrite` that either dispatches to an external rewriter or contains
    a region for the rewrite specified via `pdl`. The types of values in `pdl`
    are handle types to AIIR C++ types, with `!pdl.attribute`, `!pdl.operation`,
    `!pdl.value`, and `!pdl.type` directly mapping to `aiir::Attribute`,
    `aiir::Operation*`, `aiir::Value`, and `aiir::Type` respectively.

    An example pattern is shown below:

    ```aiir
    // pdl.pattern contains metadata similarly to a `RewritePattern`.
    pdl.pattern : benefit(1) {
      // External input operand values are specified via `pdl.operand` operations.
      // Result types are constrainted via `pdl.type` operations.

      %resultType = pdl.type
      %inputOperand = pdl.operand
      %root = pdl.operation "foo.op"(%inputOperand) -> %resultType
      pdl.rewrite %root {
        pdl.replace %root with (%inputOperand)
      }
    }
    ```

    The above pattern simply replaces an operation with its first operand. Note
    how the input operation is specified structurally, similarly to how it would
    look in memory. This is a simple example and pdl provides support for many
    other features such as applying external constraints or external generator
    methods. These features and more are detailed below.
  }];

  let name = "pdl";
  let cppNamespace = "::aiir::pdl";

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // AIIR_DIALECT_PDL_IR_PDLDIALECT
