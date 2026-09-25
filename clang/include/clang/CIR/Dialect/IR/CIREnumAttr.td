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

// Bit positions match `mlir::LLVM::FastmathFlags`. Only `contract` is set by
// CIRGen today; the other bits exist so the attribute can round-trip.
def CIR_FastMathFlags : CIR_I32BitEnum<
    "FastMathFlags", "fast-math flags", [
  I32BitEnumCaseNone<"none">,
  I32BitEnumCaseBit<"nnan", 0>,
  I32BitEnumCaseBit<"ninf", 1>,
  I32BitEnumCaseBit<"nsz", 2>,
  I32BitEnumCaseBit<"arcp", 3>,
  I32BitEnumCaseBit<"contract", 4>,
  I32BitEnumCaseBit<"afn", 5>,
  I32BitEnumCaseBit<"reassoc", 6>
]> {
  let description = [{
    Per-operation fast-math flags. These are the LLVM fast-math bits, carried
    on CIR floating-point operations and lowered onto the corresponding LLVM
    dialect operation's `fastmathFlags`.

    `contract` allows the backend to fuse this operation with another
    floating-point operation, including across statements. It is what
    `-ffp-contract=fast` records. `-ffp-contract=on` is represented separately
    by `cir.fmuladd`.
  }];
  let separator = ", ";
}

def CIR_FastMathFlagsAttr : CIR_EnumAttr<CIR_FastMathFlags, "fastmath"> {
  let summary = "Fast-math flags for a floating-point operation";
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
