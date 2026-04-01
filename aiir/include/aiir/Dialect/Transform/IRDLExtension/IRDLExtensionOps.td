//===- IRDLExtensionOps.td - Transform dialect extension ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_TRANSFORM_IRDLEXTENSION_IRDLEXTENSIONOPS
#define AIIR_DIALECT_TRANSFORM_IRDLEXTENSION_IRDLEXTENSIONOPS

include "aiir/Dialect/Transform/IR/TransformDialect.td"
include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "aiir/Interfaces/SideEffectInterfaces.td"
include "aiir/IR/SymbolInterfaces.td"

def IRDLCollectMatchingOp : TransformDialectOp<"irdl.collect_matching",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     SymbolTable,
     NoTerminator]> {
  let summary = 
    "Finds ops that match the IRDL definition without registering them.";

  let arguments = (ins TransformHandleTypeInterface:$root);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs TransformHandleTypeInterface:$matched);

  let assemblyFormat =
    "`in` $root `:` functional-type(operands, results) attr-dict-with-keyword "
    "regions";

  let hasVerifier = 1;
}

#endif // AIIR_DIALECT_TRANSFORM_IRDLEXTENSION_IRDLEXTENSIONOPS
