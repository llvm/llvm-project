//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CIR dialect enum base classes
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRENUMATTR_TD
#define CLANG_CIR_DIALECT_IR_CIRENUMATTR_TD

include "mlir/IR/EnumAttr.td"

class CIR_I32EnumAttr<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::cir";
}

class CIR_I64EnumAttr<string name, string summary, list<I64EnumAttrCase> cases>
    : I64EnumAttr<name, summary, cases> {
  let cppNamespace = "::cir";
}

class CIR_EnumAttr<EnumAttrInfo info, string name = "", list<Trait> traits = []>
    : EnumAttr<CIR_Dialect, info, name, traits> {
  let assemblyFormat = "`<` $value `>`";
}

class CIR_DefaultValuedEnumParameter<EnumAttrInfo info, string value = "">
    : EnumParameter<info> {
  let defaultValue = value;
}

def CIR_AddressSpace
    : CIR_I32EnumAttr<
          "AddressSpace", "address space kind",
          [I32EnumAttrCase<"Default", 0, "default">,
           I32EnumAttrCase<"OffloadPrivate", 1, "offload_private">,
           I32EnumAttrCase<"OffloadLocal", 2, "offload_local">,
           I32EnumAttrCase<"OffloadGlobal", 3, "offload_global">,
           I32EnumAttrCase<"OffloadConstant", 4, "offload_constant">,
           I32EnumAttrCase<"OffloadGeneric", 5, "offload_generic">,
           I32EnumAttrCase<"Target", 6, "target">]> {
  let description = [{
    The `address_space` attribute is used to represent address spaces for
    pointer types in CIR. It provides a unified model on top of `clang::LangAS`
    and simplifies the representation of address spaces.

    The `value` parameter is an extensible enum, which encodes target address
    space as an offset to the last language address space. For that reason, the
    attribute is implemented as custom AddressSpaceAttr, which provides custom
    printer and parser for the `value` parameter.
  }];

  let genSpecializedAttr = 0;
}

#endif // CLANG_CIR_DIALECT_IR_CIRENUMATTR_TD
