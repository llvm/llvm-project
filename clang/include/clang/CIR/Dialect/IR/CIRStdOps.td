//===-- CIRStdOps.td - CIR standard library ops ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Defines ops representing standard library calls, raised from plain calls
/// by the cir-idiom-recognizer pass and lowered back by LoweringPrepare.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRSTDOPS_TD
#define CLANG_CIR_DIALECT_IR_CIRSTDOPS_TD

// knownKind names the CIR_KnownFuncKind entry whose tagged calls raise
// to this operation.
class CIR_StdOp<string functionName, dag args, dag res,
                list<Trait> traits = [], string knownKind = "">
    : CIR_Op<"std." # functionName, traits> {
  let arguments = !con(args, (ins FlatSymbolRefAttr:$original_fn));
  let results = res;
  let hasLLVMLowering = false;

  let extraClassDeclaration = [{
    static constexpr unsigned getNumArgs() {
      return }] # !size(args) # [{;
    }
    static llvm::StringLiteral getFunctionName() {
      return "}] # functionName # [{";
    }
  }] # !if(!empty(knownKind), "", [{
    static cir::KnownFuncKind getFuncKind() {
      return cir::KnownFuncKind::}] # knownKind # [{;
    }
  }]);
}

def CIR_StdFindOp : CIR_StdOp<"find",
  (ins CIR_AnyType:$first, CIR_AnyType:$last, CIR_AnyType:$pattern),
  (outs CIR_AnyType:$result),
  [AllTypesMatch<["first", "last", "result"]>], "StdFind"> {
  let summary = "std::find()";
  let assemblyFormat = [{
    `(` $first `:` qualified(type($first)) `,`
        $last `:` qualified(type($last)) `,`
        $pattern `:` qualified(type($pattern)) `,`
        $original_fn `)` `->` qualified(type($result)) attr-dict
  }];
}

#endif // CLANG_CIR_DIALECT_IR_CIRSTDOPS_TD
