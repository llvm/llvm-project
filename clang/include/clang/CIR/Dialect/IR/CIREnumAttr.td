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

// A CIR enum describes a C++ enum and nothing else. It either gets a
// `CIR_EnumAttr` wrapper below, or it is used as a raw parameter of some other
// attribute, so it derives from `EnumInfo` rather than the older
// `EnumAttrInfo`, which doubles as an `IntegerAttr` constraint and would need
// its `genSpecializedAttr` cleared on every enum.
class CIR_I32Enum<string name, string summary, list<EnumCase> cases>
    : I32Enum<name, summary, cases> {
  let cppNamespace = "::cir";
}

class CIR_I64Enum<string name, string summary, list<EnumCase> cases>
    : I64Enum<name, summary, cases> {
  let cppNamespace = "::cir";
}

class CIR_I32BitEnum<string name, string summary,
                         list<BitEnumCaseBase> cases>
    : I32BitEnum<name, summary, cases> {
  let cppNamespace = "::cir";
}

// Upstream `EnumAttr` defaults to a bare `$value`, so a standalone attribute
// falls back to the generic `#cir<cleanup all>` form. Keeping the `<` `>`
// delimiters gets the dialect's own `#cir.cleanup<all>` instead. Operations
// that want the bare keyword wrap the argument in the `enum` directive, as in
// `enum($cleanupKind)`. Naming the argument directly prints `<all>`.
class CIR_EnumAttr<EnumInfo info, string name = "", list<Trait> traits = []>
    : EnumAttr<CIR_Dialect, info, name, traits> {
  let assemblyFormat = "`<` $value `>`";
}

def CIR_LangAddressSpace : CIR_I32Enum<
  "LangAddressSpace", "language address space kind", [
  I32EnumCase<"Default", 0, "default">,
  I32EnumCase<"OffloadPrivate", 1, "offload_private">,
  I32EnumCase<"OffloadLocal", 2, "offload_local">,
  I32EnumCase<"OffloadGlobal", 3, "offload_global">,
  I32EnumCase<"OffloadConstant", 4, "offload_constant">,
  I32EnumCase<"OffloadGeneric", 5, "offload_generic">,
  I32EnumCase<"OffloadGlobalDevice", 6, "offload_global_device">,
  I32EnumCase<"OffloadGlobalHost", 7, "offload_global_host">
]> {
  let description = [{
    Enumerates language-specific address spaces used by CIR. These represent
    semantic qualifiers from source languages (e.g., CUDA `__shared__`,
    OpenCL `__local`) before target lowering.
  }];

}

#endif // CLANG_CIR_DIALECT_IR_CIRENUMATTR_TD
