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

#endif // CLANG_CIR_DIALECT_IR_CIRENUMATTR_TD
