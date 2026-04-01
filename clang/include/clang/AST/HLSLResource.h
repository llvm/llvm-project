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
#include "clang/AST/DeclBase.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Support/Compiler.h"
#include "llvm/Frontend/HLSL/HLSLResource.h"
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
  void setCounterImplicitOrderID(unsigned Value) const {
    assert(hasBinding() && !hasCounterImplicitOrderID());
    RegBinding->setImplicitCounterBindingOrderID(Value);
  }

  bool hasCounterImplicitOrderID() const {
    return RegBinding && RegBinding->hasImplicitCounterBindingOrderID();
  }

  unsigned getCounterImplicitOrderID() const {
    assert(hasCounterImplicitOrderID());
    return RegBinding->getImplicitCounterBindingOrderID();
  }
};

inline uint32_t getResourceDimensions(llvm::dxil::ResourceDimension Dim) {
  switch (Dim) {
  case llvm::dxil::ResourceDimension::Dim1D:
    return 1;
    break;
  case llvm::dxil::ResourceDimension::Dim2D:
    return 2;
    break;
  case llvm::dxil::ResourceDimension::Dim3D:
  case llvm::dxil::ResourceDimension::Cube:
    return 3;
    break;
  case llvm::dxil::ResourceDimension::Unknown:
    llvm_unreachable(
        "We cannot get the dimension of a resource with unknown dimension.");
  }
  llvm_unreachable("Unhandled llvm::dxil::ResourceDimension enum.");
}

// Helper class for building a name of a global resource variable that
// gets created for a resource embedded in a struct or class. This will
// also be used from CodeGen to build a name that matches the resource
// access with the corresponding declaration.
class EmbeddedResourceNameBuilder {
  llvm::SmallString<64> Name;
  llvm::SmallVector<unsigned> Offsets;

  inline static constexpr std::string_view BaseClassDelim = "::";
  inline static constexpr std::string_view FieldDelim = ".";
  inline static constexpr std::string_view ArrayIndexDelim = FieldDelim;

public:
  EmbeddedResourceNameBuilder(llvm::StringRef BaseName) : Name(BaseName) {}
  EmbeddedResourceNameBuilder() : Name("") {}

  void pushName(llvm::StringRef N) { pushName(N, FieldDelim); }
  void pushBaseName(llvm::StringRef N);
  void pushArrayIndex(uint64_t Index);

  void pop() {
    assert(!Offsets.empty() && "no name to pop");
    Name.resize(Offsets.pop_back_val());
  }

  IdentifierInfo *getNameAsIdentifier(ASTContext &AST) const {
    return &AST.Idents.get(Name);
  }

private:
  void pushName(llvm::StringRef N, llvm::StringRef Delim);
};

} // namespace hlsl

} // namespace clang

#endif // LLVM_CLANG_AST_HLSLRESOURCE_H
