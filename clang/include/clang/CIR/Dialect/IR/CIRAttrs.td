//===- CIRAttrs.td - CIR dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_DIALECT_CIR_ATTRS
#define MLIR_CIR_DIALECT_CIR_ATTRS

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "clang/CIR/Dialect/IR/CIRDialect.td"

//===----------------------------------------------------------------------===//
// CIR Attrs
//===----------------------------------------------------------------------===//

class CIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// NullAttr
//===----------------------------------------------------------------------===//

def NullAttr : CIR_Attr<"Null", "null", [TypedAttrInterface]> {
  let summary = "A simple attr to represent nullptr";
  let description = [{
    The NullAttr represents the value of nullptr within cir.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);

  let assemblyFormat = [{}];
}

//===----------------------------------------------------------------------===//
// ZeroAttr
//===----------------------------------------------------------------------===//

def ZeroAttr : CIR_Attr<"Zero", "zero", [TypedAttrInterface]> {
  let summary = "Attribute to represent zero initialization";
  let description = [{
    The ZeroAttr is used to indicate zero initialization on structs.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = [{}];
}

//===----------------------------------------------------------------------===//
// ConstArrayAttr
//===----------------------------------------------------------------------===//

def ConstArrayAttr : CIR_Attr<"ConstArray", "const_array", [TypedAttrInterface]> {
  let summary = "A constant array from ArrayAttr or StringRefAttr";
  let description = [{
    An CIR array attribute is an array of literals of the specified attr types.
  }];

  // `$type` is the `self` type of the attribute (i.e. the type of the
  // Attribute itself).
  //
  // `arrayAttr` is the actual attribute array with elements for this constant
  // array, there's yet no need to own these elements.
  //
  // TODO: create a trait for ArrayAttrOrStringAttr value instead of relying
  // on verifier.
  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "Attribute":$value);

  // Define a custom builder for the type; that removes the need to pass
  // in an MLIRContext instance, as it can be infered from the `type`.
  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::cir::ArrayType":$type,
                                        "Attribute":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  // Printing and parsing available in CIRDialect.cpp
  let hasCustomAssemblyFormat = 1;

  // Enable verifier.
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// SignedOverflowBehaviorAttr
//===----------------------------------------------------------------------===//

def SignedOverflowBehaviorAttr : AttrDef<CIR_Dialect, "SignedOverflowBehavior"> {
  let mnemonic = "signed_overflow_behavior";
  let parameters = (ins
    "sob::SignedOverflowBehavior":$behavior
  );
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// AST Wrappers
//===----------------------------------------------------------------------===//

class ASTDecl<string name, string prefix, list<Trait> traits = []>
  : CIR_Attr<!strconcat("AST", name), !strconcat(prefix, ".ast"), traits> {
  string clang_name = !strconcat("const clang::", name, " *");

  let summary = !strconcat("Wraps a '", clang_name, "' AST node.");
  let description = [{
    Operations optionally refer to this node, they could be available depending
    on the CIR lowering stage. Whether it's attached to the appropriated
    CIR operation is delegated to the operation verifier.

    This always implies a non-null AST reference (verified).
  }];
  let parameters = (ins clang_name:$astDecl);

  // Printing and parsing available in CIRDialect.cpp
  let hasCustomAssemblyFormat = 1;

  // Enable verifier.
  let genVerifyDecl = 1;
}

def ASTFunctionDeclAttr : ASTDecl<"FunctionDecl", "function.decl">;
def ASTVarDeclAttr : ASTDecl<"VarDecl", "var.decl">;
def ASTRecordDeclAttr : ASTDecl<"RecordDecl", "record.decl">;

#endif // MLIR_CIR_DIALECT_CIR_ATTRS
