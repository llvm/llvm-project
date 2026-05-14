//===- ASTAttrInterfaces.td - CIR AST Interface Definitions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_INTERFACES_ASTATTRINTERFACES_TD
#define MLIR_CIR_INTERFACES_ASTATTRINTERFACES_TD

include "mlir/IR/OpBase.td"

let cppNamespace = "::cir" in {
  def ASTVarDeclInterface : AttrInterface<"ASTVarDeclInterface"> {
    let methods = [
      InterfaceMethod<"", "bool", "isLocalVarDecl", (ins), [{}],
                      /*defaultImplementation=*/[{
        return $_attr.getAst()->isLocalVarDecl();
      }]>,
      InterfaceMethod<"", "clang::VarDecl::TLSKind", "getTLSKind",
                      (ins), [{}],
                      /*defaultImplementation=*/[{
        return $_attr.getAst()->getTLSKind();
      }]>,
      InterfaceMethod<"", "bool", "isInline", (ins), [{}],
                      /*defaultImplementation=*/[{
        return $_attr.getAst()->isInline();
      }]>,
      InterfaceMethod<"", "clang::TemplateSpecializationKind",
                      "getTemplateSpecializationKind", (ins), [{}],
                      /*defaultImplementation=*/[{
        return $_attr.getAst()->getTemplateSpecializationKind();
      }]>
    ];
  }
} // namespace cir

#endif // MLIR_CIR_INTERFACES_ASTATTRINTERFACES_TD
