//===- Visitor.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/InstallAPI/Visitor.h"
#include "clang/AST/Availability.h"
#include "clang/Basic/Linkage.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"

using namespace llvm;
using namespace llvm::MachO;

namespace clang::installapi {

// Exported NamedDecl needs to have external linkage and
// default visibility from LinkageComputer.
static bool isExported(const NamedDecl *D) {
  auto LV = D->getLinkageAndVisibility();
  return isExternallyVisible(LV.getLinkage()) &&
         (LV.getVisibility() == DefaultVisibility);
}

static SymbolFlags getFlags(bool WeakDef, bool ThreadLocal) {
  SymbolFlags Result = SymbolFlags::None;
  if (WeakDef)
    Result |= SymbolFlags::WeakDefined;
  if (ThreadLocal)
    Result |= SymbolFlags::ThreadLocalValue;

  return Result;
}

void InstallAPIVisitor::HandleTranslationUnit(ASTContext &ASTCtx) {
  if (ASTCtx.getDiagnostics().hasErrorOccurred())
    return;

  auto *D = ASTCtx.getTranslationUnitDecl();
  TraverseDecl(D);
}

std::string InstallAPIVisitor::getMangledName(const NamedDecl *D) const {
  SmallString<256> Name;
  if (MC->shouldMangleDeclName(D)) {
    raw_svector_ostream NStream(Name);
    MC->mangleName(D, NStream);
  } else
    Name += D->getNameAsString();

  return getBackendMangledName(Name);
}

std::string InstallAPIVisitor::getBackendMangledName(Twine Name) const {
  SmallString<256> FinalName;
  Mangler::getNameWithPrefix(FinalName, Name, DataLayout(Layout));
  return std::string(FinalName);
}

/// Collect all global variables.
bool InstallAPIVisitor::VisitVarDecl(const VarDecl *D) {
  // Skip function parameters.
  if (isa<ParmVarDecl>(D))
    return true;

  // Skip variables in records. They are handled seperately for C++.
  if (D->getDeclContext()->isRecord())
    return true;

  // Skip anything inside functions or methods.
  if (!D->isDefinedOutsideFunctionOrMethod())
    return true;

  // If this is a template but not specialization or instantiation, skip.
  if (D->getASTContext().getTemplateOrSpecializationInfo(D) &&
      D->getTemplateSpecializationKind() == TSK_Undeclared)
    return true;

  // TODO: Capture SourceLocation & Availability for Decls.
  const RecordLinkage Linkage =
      isExported(D) ? RecordLinkage::Exported : RecordLinkage::Internal;
  const bool WeakDef = D->hasAttr<WeakAttr>();
  const bool ThreadLocal = D->getTLSKind() != VarDecl::TLS_None;
  Slice.addGlobal(getMangledName(D), Linkage, GlobalRecord::Kind::Variable,
                  getFlags(WeakDef, ThreadLocal));
  return true;
}

} // namespace clang::installapi
