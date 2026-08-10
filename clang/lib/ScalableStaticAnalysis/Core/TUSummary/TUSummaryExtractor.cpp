//===- TUSummaryExtractor.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/ScalableStaticAnalysis/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "llvm/Support/Casting.h"
#include <optional>

using namespace clang;
using namespace ssaf;

static EntityLinkageType mapLinkage(const NamedDecl *ND) {
  switch (ND->getFormalLinkage()) {
  case Linkage::Invalid: {
    llvm_unreachable("Shouldn't be invalid");
  }
  case Linkage::None:
    return EntityLinkageType::None;
  case Linkage::Internal:
  case Linkage::UniqueExternal:
  case Linkage::VisibleNone:
    return EntityLinkageType::Internal;
  case Linkage::Module:
  case Linkage::External:
    return EntityLinkageType::External;
  }
  llvm_unreachable("Unhandled clang::Linkage kind");
}

static EntityVisibility mapVisibility(const NamedDecl *ND) {
  switch (ND->getVisibility()) {
  case HiddenVisibility:
    return EntityVisibility::Hidden;
  case ProtectedVisibility:
    return EntityVisibility::Protected;
  case DefaultVisibility:
    return EntityVisibility::Default;
  }
  llvm_unreachable("Unhandled clang::Visibility kind");
}

static EntityDefinitionKind mapDefinitionKind(const NamedDecl *ND,
                                              ASTContext &Ctx) {
  if (const auto *FD = dyn_cast<FunctionDecl>(ND)) {
    return FD->isThisDeclarationADefinition()
               ? EntityDefinitionKind::Definition
               : EntityDefinitionKind::Declaration;
  }
  if (const auto *VD = dyn_cast<VarDecl>(ND)) {
    return VD->isThisDeclarationADefinition(Ctx) == VarDecl::DeclarationOnly
               ? EntityDefinitionKind::Declaration
               : EntityDefinitionKind::Definition;
  }
  return EntityDefinitionKind::Definition;
}

static EntityBinding mapBinding(const NamedDecl *ND, ASTContext &Ctx) {
  if (ND->hasAttr<WeakAttr>() || ND->hasAttr<WeakRefAttr>()) {
    return EntityBinding::Weak;
  }
  if (const auto *FD = dyn_cast<FunctionDecl>(ND)) {
    if (!FD->isThisDeclarationADefinition()) {
      return EntityBinding::Undefined;
    }
    switch (Ctx.GetGVALinkageForFunction(FD)) {
    case GVA_AvailableExternally:
      return EntityBinding::Undefined;
    case GVA_DiscardableODR:
    case GVA_StrongODR:
      // An ODR definition binds strongly; that its duplicates are permitted is
      // recorded separately by mapCoalescing(). Object formats express this
      // differently — ELF and Mach-O lower it to a weak symbol, COFF keeps it
      // strong and uses a COMDAT — which is LinkageRules' concern, not ours.
      return EntityBinding::Strong;
    case GVA_Internal:
    case GVA_StrongExternal:
      return EntityBinding::Strong;
    }
    return EntityBinding::Strong;
  }
  if (const auto *VD = dyn_cast<VarDecl>(ND)) {
    switch (VD->isThisDeclarationADefinition(Ctx)) {
    case VarDecl::DeclarationOnly:
      return EntityBinding::Undefined;
    case VarDecl::TentativeDefinition:
      return EntityBinding::Common;
    case VarDecl::Definition:
      break;
    }
    return EntityBinding::Strong;
  }
  return EntityBinding::Strong;
}

/// Returns whether every definition of \p ND is required to be identical, as
/// the One Definition Rule guarantees for inline functions, templates and
/// vtables.
static EntityCoalescing mapCoalescing(const NamedDecl *ND, ASTContext &Ctx) {
  // An explicitly weak symbol may be replaced by an unrelated definition, so
  // its copies carry no identity guarantee even if it is also inline.
  if (ND->hasAttr<WeakAttr>() || ND->hasAttr<WeakRefAttr>()) {
    return EntityCoalescing::None;
  }
  if (const auto *FD = dyn_cast<FunctionDecl>(ND)) {
    if (!FD->isThisDeclarationADefinition()) {
      return EntityCoalescing::None;
    }
    switch (Ctx.GetGVALinkageForFunction(FD)) {
    case GVA_DiscardableODR:
    case GVA_StrongODR:
      return EntityCoalescing::ODR;
    default:
      return EntityCoalescing::None;
    }
  }
  if (const auto *VD = dyn_cast<VarDecl>(ND)) {
    if (VD->isThisDeclarationADefinition(Ctx) != VarDecl::Definition) {
      return EntityCoalescing::None;
    }
    switch (Ctx.GetGVALinkageForVariable(VD)) {
    case GVA_DiscardableODR:
    case GVA_StrongODR:
      return EntityCoalescing::ODR;
    default:
      return EntityCoalescing::None;
    }
  }
  return EntityCoalescing::None;
}

static EntityLinkage getEntityLinkageForDecl(const Decl *D) {
  const auto *ND = dyn_cast<NamedDecl>(D);
  if (!ND) {
    return EntityLinkage(EntityLinkageType::None, EntityBinding::Undefined,
                         EntityCoalescing::None, EntityVisibility::Default,
                         EntityDefinitionKind::Declaration);
  }

  // Parameters have no linkage in C++, but SSAF needs them to inherit
  // the linker properties from their parent functions.
  // Here is why:
  //   SSAF treats parameters as entities and may not always associate them back
  //   to their parent functions. Therefore, it needs to identify parameters of
  //   functions with external linkage across different TUs. Treating them as
  //   having no linkage (as in C++) causes the same parameter in different TUs
  //   to be assigned different EntityIDs. As a result, the behavior of the
  //   parameter across multiple TUs cannot be correlated.
  if (const auto *PVD = dyn_cast<ParmVarDecl>(D)) {
    if (const auto *FD = llvm::dyn_cast_or_null<FunctionDecl>(
            PVD->getParentFunctionOrMethod())) {
      return getEntityLinkageForDecl(FD);
    }
  }

  ASTContext &Ctx = ND->getASTContext();
  return EntityLinkage(mapLinkage(ND), mapBinding(ND, Ctx),
                       mapCoalescing(ND, Ctx), mapVisibility(ND),
                       mapDefinitionKind(ND, Ctx));
}

std::optional<EntityId> TUSummaryExtractor::addEntity(const NamedDecl *D) {
  auto Name = getEntityName(D);
  if (!Name)
    return std::nullopt;
  return SummaryBuilder.addEntity(*Name, getEntityLinkageForDecl(D));
}

std::optional<EntityId>
TUSummaryExtractor::addEntityForReturn(const FunctionDecl *FD) {
  auto Name = getEntityNameForReturn(FD);
  if (!Name)
    return std::nullopt;
  return SummaryBuilder.addEntity(*Name, getEntityLinkageForDecl(FD));
}

const SSAFOptions &TUSummaryExtractor::getOptions() const {
  return SummaryBuilder.getOptions();
}
