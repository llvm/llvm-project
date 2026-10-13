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

// knownKind names the CIR_KnownFuncKind entry this operation raises, or is
// empty for a C library function that has no such entry.
class CIR_StdOp<string functionName, dag args, dag res,
                list<Trait> traits = [], string knownKind = "">
    : CIR_Op<"std." # functionName, traits # [CIR_StdOpInterface]> {
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
    // True when matched by its KnownFuncKind tag, not the callee symbol.
    static constexpr bool hasKnownFuncKind() {
      return }] # !if(!empty(knownKind), "false", "true") # [{;
    }
    static bool signatureMatches(mlir::TypeRange operands,
                                 mlir::TypeRange results);
  }] # !if(!empty(knownKind), "", [{
    static cir::KnownFuncKind getFuncKind() {
      return cir::KnownFuncKind::}] # knownKind # [{;
    }
  }]);
}

// A C library op whose signatureMatches is generated from its type constraints.
class CIR_CStdLibOp<string functionName, dag args, dag res,
                    list<Trait> traits = []>
    : CIR_StdOp<functionName, args, res, traits> {
  assert !foldl(true, !listconcat(
                          !foreach(i, !range(!size(args)),
                                   !getdagarg<TypeConstraint>(args, i)),
                          !foreach(i, !range(!size(res)),
                                   !getdagarg<TypeConstraint>(res, i))),
                acc, c, !and(acc, !ne(c.cppFunctionName, ""))),
      "CIR_CStdLibOp operand and result types must set cppFunctionName";

  let extraClassDefinition = [{
    bool $cppClass::signatureMatches(mlir::TypeRange operands,
                                     mlir::TypeRange results) {
      return }] # !interleave(
          !listconcat(
              ["operands.size() == " # !size(args),
               "results.size() == " # !size(res)],
              !foreach(i, !range(!size(args)),
                       "::cir::" #
                       !getdagarg<TypeConstraint>(args, i).cppFunctionName #
                       "(operands[" # i # "])"),
              !foreach(i, !range(!size(res)),
                       "::cir::" #
                       !getdagarg<TypeConstraint>(res, i).cppFunctionName #
                       "(results[" # i # "])")),
          " && ") # [{;
    }
  }];
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

def CIR_StrLenOp : CIR_CStdLibOp<"strlen",
  (ins CIR_PtrToChar8Type:$string),
  (outs CIR_AnyFundamentalUIntType:$result)> {
  let summary = "C standard library strlen()";
  let assemblyFormat = [{
    `(` $string `:` qualified(type($string)) `,` $original_fn `)`
    `->` qualified(type($result)) attr-dict
  }];
}

#endif // CLANG_CIR_DIALECT_IR_CIRSTDOPS_TD
