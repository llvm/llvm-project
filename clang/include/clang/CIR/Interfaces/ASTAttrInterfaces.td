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

  def ASTFunctionDeclInterface : AttrInterface<"ASTFunctionDeclInterface"> {
    let methods = [
      InterfaceMethod<"", "bool", "hasSizeParameter", (ins), [{}],
                      /*defaultImplementation=*/[{
        const clang::FunctionDecl *FD = $_attr.getAst();
        return FD && FD->getUsualDeleteParams().Size;
      }]>,
      InterfaceMethod<"", "bool", "hasTypeAwareDelete", (ins), [{}],
                      /*defaultImplementation=*/[{
        const clang::FunctionDecl *FD = $_attr.getAst();
        return FD && ::clang::isTypeAwareAllocation(
            FD->getUsualDeleteParams().TypeAwareDelete);
      }]>,
      InterfaceMethod<"", "bool", "hasDestroyingDelete", (ins), [{}],
                      /*defaultImplementation=*/[{
        const clang::FunctionDecl *FD = $_attr.getAst();
        return FD && FD->getUsualDeleteParams().DestroyingDelete;
      }]>,
      InterfaceMethod<"", "bool", "hasAlignmentParameter", (ins), [{}],
                      /*defaultImplementation=*/[{
        const clang::FunctionDecl *FD = $_attr.getAst();
        return FD && ::clang::isAlignedAllocation(
            FD->getUsualDeleteParams().Alignment);
      }]>,
    ];
  }

  def ASTCXXDeleteExprInterface : AttrInterface<"ASTCXXDeleteExprInterface"> {
    let methods = [
      InterfaceMethod<"", "cir::ASTFunctionDeclInterface", "getOperatorDelete",
                      (ins), [{}],
                      /*defaultImplementation=*/[{
        const clang::CXXDeleteExpr *E = $_attr.getAst();
        if (!E || !E->getOperatorDelete())
          return {};
        return mlir::cast<cir::ASTFunctionDeclInterface>(
            makeFuncDeclAttr(E->getOperatorDelete(), $_attr.getContext()));
      }]>,
      InterfaceMethod<"", "bool", "doesUsualArrayDeleteWantSize", (ins), [{}],
                      /*defaultImplementation=*/[{
        const clang::CXXDeleteExpr *E = $_attr.getAst();
        return E && E->doesUsualArrayDeleteWantSize();
      }]>,
      InterfaceMethod<"", "bool", "isElementDestructedType", (ins), [{}],
                      /*defaultImplementation=*/[{
        const clang::CXXDeleteExpr *E = $_attr.getAst();
        if (!E) return false;
        clang::QualType eltTy = E->getOperatorDelete()->getASTContext()
            .getBaseElementType(E->getDestroyedType());
        return eltTy.isDestructedType();
      }]>,
    ];
  }

} // namespace cir

#endif // MLIR_CIR_INTERFACES_ASTATTRINTERFACES_TD
