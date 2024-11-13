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


/// Fortran procedure attributes (F2023 15.6.2.1). BIND attribute (18.3.7)
/// is also tracked in the same enum. Recursive (resp. Impure) attribute
/// is implied by the absence of opposite NonRecursive (resp. Pure) attribute.
/// Beware that "elemental" does not implicitly imply "pure" as it does in
/// Fortran, "pure" must be made explicit when generating the FIR attribute.
def FIRfuncNoAttributes  : I32BitEnumAttrCaseNone<"none">;
def FIRfuncElemental     : I32BitEnumAttrCaseBit<"elemental", 0>;
def FIRfuncPure          : I32BitEnumAttrCaseBit<"pure", 1>;
def FIRfuncNonRecursive  : I32BitEnumAttrCaseBit<"non_recursive", 2>;
def FIRfuncSimple        : I32BitEnumAttrCaseBit<"simple", 3>;
def FIRfuncBind_c        : I32BitEnumAttrCaseBit<"bind_c", 4>;

def fir_FortranProcedureFlagsEnum : I32BitEnumAttr<
    "FortranProcedureFlagsEnum",
    "Fortran procedure attributes",
    [FIRfuncNoAttributes, FIRfuncElemental, FIRfuncPure, FIRfuncNonRecursive,
     FIRfuncSimple, FIRfuncBind_c]> {
  let separator = ", ";
  let cppNamespace = "::fir";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def fir_FortranProcedureFlagsAttr :
    EnumAttr<FIROpsDialect, fir_FortranProcedureFlagsEnum, "proc_attrs"> {
  let assemblyFormat = "`<` $value `>`";
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

def fir_LocationKind : I32EnumAttr<"LocationKind", "Flang location kind",
  [
    I32EnumAttrCase<"Base", 0, "base">,
    I32EnumAttrCase<"Inclusion", 1, "inclusion">,
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::fir";
}
def fir_LocationKindAttr : EnumAttr<FIROpsDialect, fir_LocationKind, "loc_kind">;

def LocationKindArrayAttr : ArrayOfAttr<FIROpsDialect, "LocationKindArray",
    "loc_kind_array", "LocationKindAttr">;

#endif // FIR_DIALECT_FIR_ATTRS
