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
include "clang/CIR/Dialect/IR/CIRDialect.td"

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

def CIR_LangAddressSpace : CIR_I32EnumAttr<
  "LangAddressSpace", "language address space kind", [
  I32EnumAttrCase<"Default", 0, "default">,
  I32EnumAttrCase<"OffloadPrivate", 1, "offload_private">,
  I32EnumAttrCase<"OffloadLocal", 2, "offload_local">,
  I32EnumAttrCase<"OffloadGlobal", 3, "offload_global">,
  I32EnumAttrCase<"OffloadConstant", 4, "offload_constant">,
  I32EnumAttrCase<"OffloadGeneric", 5, "offload_generic">
]> {
  let description = [{
    Enumerates language-specific address spaces used by CIR. These represent
    semantic qualifiers from source languages (e.g., CUDA `__shared__`,
    OpenCL `__local`) before target lowering.
  }];

  let genSpecializedAttr = 0;
}

#endif // CLANG_CIR_DIALECT_IR_CIRENUMATTR_TD
