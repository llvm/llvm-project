//===- FIRAttr.td - FIR Attributes -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the FIR dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef FIR_DIALECT_FIR_ATTRS
#define FIR_DIALECT_FIR_ATTRS

include "flang/Optimizer/Dialect/FIRDialect.td"
include "mlir/IR/EnumAttr.td"

class fir_Attr<string name> : AttrDef<FIROpsDialect, name>;

def FIRnoAttributes  : I32BitEnumAttrCaseNone<"None">;
def FIRallocatable  : I32BitEnumAttrCaseBit<"allocatable", 0>;
def FIRasynchronous : I32BitEnumAttrCaseBit<"asynchronous", 1>;
def FIRbind_c       : I32BitEnumAttrCaseBit<"bind_c", 2>;
def FIRcontiguous   : I32BitEnumAttrCaseBit<"contiguous", 3>;
def FIRintent_in    : I32BitEnumAttrCaseBit<"intent_in", 4>;
def FIRintent_inout : I32BitEnumAttrCaseBit<"intent_inout", 5>;
def FIRintent_out   : I32BitEnumAttrCaseBit<"intent_out", 6>;
def FIRoptional     : I32BitEnumAttrCaseBit<"optional", 7>;
def FIRparameter    : I32BitEnumAttrCaseBit<"parameter", 8>;
def FIRpointer      : I32BitEnumAttrCaseBit<"pointer", 9>;
def FIRtarget       : I32BitEnumAttrCaseBit<"target", 10>;
def FIRvalue        : I32BitEnumAttrCaseBit<"value", 11>;
def FIRvolatile     : I32BitEnumAttrCaseBit<"fortran_volatile", 12, "volatile">;
def FIRHostAssoc    : I32BitEnumAttrCaseBit<"host_assoc", 13>;

def fir_FortranVariableFlagsEnum : I32BitEnumAttr<
    "FortranVariableFlagsEnum",
    "Fortran variable attributes",
    [FIRnoAttributes, FIRallocatable, FIRasynchronous, FIRbind_c, FIRcontiguous,
     FIRintent_in, FIRintent_inout, FIRintent_out, FIRoptional, FIRparameter,
     FIRpointer, FIRtarget, FIRvalue, FIRvolatile, FIRHostAssoc]> {
  let separator = ", ";
  let cppNamespace = "::fir";
  let printBitEnumPrimaryGroups = 1;
}

def fir_FortranVariableFlagsAttr : fir_Attr<"FortranVariableFlags"> {
  let mnemonic = "var_attrs";

  let parameters = (ins
    "FortranVariableFlagsEnum":$flags
  );
  let hasCustomAssemblyFormat = 1;
  let returnType = "::fir::FortranVariableFlagsEnum";
  let convertFromStorage = "$_self.getFlags()";
  let constBuilderCall =
        "::fir::FortranVariableFlagsAttr::get($_builder.getContext(), $0)";
}

def fir_BoxFieldAttr : I32EnumAttr<
    "BoxFieldAttr", "",
    [
      I32EnumAttrCase<"base_addr", 0>,
      I32EnumAttrCase<"derived_type", 1>
    ]> {
  let cppNamespace = "fir";
}

def fir_ReduceOperationEnum : I32BitEnumAttr<"ReduceOperationEnum",
    "intrinsic operations and functions supported by DO CONCURRENT REDUCE",
    [
      I32BitEnumAttrCaseBit<"Add", 0, "add">,
      I32BitEnumAttrCaseBit<"Multiply", 1, "multiply">,
      I32BitEnumAttrCaseBit<"AND", 2, "and">,
      I32BitEnumAttrCaseBit<"OR", 3, "or">,
      I32BitEnumAttrCaseBit<"EQV", 4, "eqv">,
      I32BitEnumAttrCaseBit<"NEQV", 5, "neqv">,
      I32BitEnumAttrCaseBit<"MAX", 6, "max">,
      I32BitEnumAttrCaseBit<"MIN", 7, "min">,
      I32BitEnumAttrCaseBit<"IAND", 8, "iand">,
      I32BitEnumAttrCaseBit<"IOR", 9, "ior">,
      I32BitEnumAttrCaseBit<"EIOR", 10, "eior">
    ]> {
  let separator = ", ";
  let cppNamespace = "::fir";
  let printBitEnumPrimaryGroups = 1;
}

def fir_ReduceAttr : fir_Attr<"Reduce"> {
  let mnemonic = "reduce_attr";

  let parameters = (ins
    "ReduceOperationEnum":$reduce_operation
  );

  let assemblyFormat = "`<` $reduce_operation `>`";
}

// mlir::SideEffects::Resource for modelling operations which add debugging information
def DebuggingResource : Resource<"::fir::DebuggingResource">;

def fir_LowerBoundModifierAttribute : I32EnumAttr<
    "LowerBoundModifierAttribute",
    "Describes how to modify lower bounds",
    [
      I32EnumAttrCase<"Preserve", 0, "preserve">,
      I32EnumAttrCase<"SetToOnes", 1, "ones">,
      I32EnumAttrCase<"SetToZeroes", 2, "zeroes">,
    ]> {
  let cppNamespace = "::fir";
}

#endif // FIR_DIALECT_FIR_ATTRS
