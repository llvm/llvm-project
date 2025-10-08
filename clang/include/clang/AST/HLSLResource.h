//===- HLSLResource.h - Routines for HLSL resources and bindings ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides shared routines to help analyze HLSL resources and
// theirs bindings during Sema and CodeGen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_HLSLRESOURCE_H
#define LLVM_CLANG_AST_HLSLRESOURCE_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {

class HLSLResourceBindingAttr;
class HLSLRVkBindingAttr;

namespace hlsl {

struct ResourceBindingAttrs {
  HLSLResourceBindingAttr *RegBinding;
  HLSLVkBindingAttr *VkBinding;

  ResourceBindingAttrs(const Decl *D) {
    RegBinding = D->getAttr<HLSLResourceBindingAttr>();
    bool IsSpirv = D->getASTContext().getTargetInfo().getTriple().isSPIRV();
    VkBinding = IsSpirv ? D->getAttr<HLSLVkBindingAttr>() : nullptr;
  }

  bool hasBinding() const { return RegBinding || VkBinding; }
  bool isExplicit() const {
    return (RegBinding && RegBinding->hasRegisterSlot()) || VkBinding;
  }

  unsigned getSlot() const {
    assert(isExplicit() && "no explicit binding");
    if (VkBinding)
      return VkBinding->getBinding();
    if (RegBinding && RegBinding->hasRegisterSlot())
      return RegBinding->getSlotNumber();
    llvm_unreachable("no explicit binding");
  }

  unsigned getSpace() const {
    if (VkBinding)
      return VkBinding->getSet();
    if (RegBinding)
      return RegBinding->getSpaceNumber();
    return 0;
  }

  bool hasImplicitOrderID() const {
    return RegBinding && RegBinding->hasImplicitBindingOrderID();
  }

  unsigned getImplicitOrderID() const {
    assert(hasImplicitOrderID());
    return RegBinding->getImplicitBindingOrderID();
  }

  void setImplicitOrderID(unsigned Value) const {
    assert(hasBinding() && !isExplicit() && !hasImplicitOrderID());
    RegBinding->setImplicitBindingOrderID(Value);
  }
};

} // namespace hlsl

} // namespace clang

#endif // LLVM_CLANG_AST_HLSLRESOURCE_H
