//===--- CIRGenVTables.cpp - Emit CIR Code for C++ vtables ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of virtual tables.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/CodeGen/ConstantInitBuilder.h"
#include "llvm/Support/Format.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>
#include <cstdio>

using namespace clang;
using namespace cir;

CIRGenVTables::CIRGenVTables(CIRGenModule &CGM)
    : CGM(CGM), VTContext(CGM.getASTContext().getVTableContext()) {}

static bool UseRelativeLayout(const CIRGenModule &CGM) {
  return CGM.getTarget().getCXXABI().isItaniumFamily() &&
         CGM.getItaniumVTableContext().isRelativeLayout();
}

mlir::Type CIRGenModule::getVTableComponentType() {
  mlir::Type ptrTy = builder.getInt8PtrTy();
  if (UseRelativeLayout(*this))
    ptrTy = builder.getInt32PtrTy();
  return ptrTy;
}

mlir::Type CIRGenVTables::getVTableComponentType() {
  return CGM.getVTableComponentType();
}

mlir::Type CIRGenVTables::getVTableType(const VTableLayout &layout) {
  SmallVector<mlir::Type, 4> tys;
  auto ctx = CGM.getBuilder().getContext();
  auto componentType = getVTableComponentType();
  for (unsigned i = 0, e = layout.getNumVTables(); i != e; ++i)
    tys.push_back(
        mlir::cir::ArrayType::get(ctx, componentType, layout.getVTableSize(i)));

  // FIXME(cir): should VTableLayout be encoded like we do for some
  // AST nodes?
  return mlir::cir::StructType::get(ctx, tys, "vtable",
                                    /*body=*/true);
}

/// At this point in the translation unit, does it appear that can we
/// rely on the vtable being defined elsewhere in the program?
///
/// The response is really only definitive when called at the end of
/// the translation unit.
///
/// The only semantic restriction here is that the object file should
/// not contain a vtable definition when that vtable is defined
/// strongly elsewhere.  Otherwise, we'd just like to avoid emitting
/// vtables when unnecessary.
/// TODO(cir): this should be merged into common AST helper for codegen.
bool CIRGenVTables::isVTableExternal(const CXXRecordDecl *RD) {
  assert(RD->isDynamicClass() && "Non-dynamic classes have no VTable.");

  // We always synthesize vtables if they are needed in the MS ABI. MSVC doesn't
  // emit them even if there is an explicit template instantiation.
  if (CGM.getTarget().getCXXABI().isMicrosoft())
    return false;

  // If we have an explicit instantiation declaration (and not a
  // definition), the vtable is defined elsewhere.
  TemplateSpecializationKind TSK = RD->getTemplateSpecializationKind();
  if (TSK == TSK_ExplicitInstantiationDeclaration)
    return true;

  // Otherwise, if the class is an instantiated template, the
  // vtable must be defined here.
  if (TSK == TSK_ImplicitInstantiation ||
      TSK == TSK_ExplicitInstantiationDefinition)
    return false;

  // Otherwise, if the class doesn't have a key function (possibly
  // anymore), the vtable must be defined here.
  const CXXMethodDecl *keyFunction =
      CGM.getASTContext().getCurrentKeyFunction(RD);
  if (!keyFunction)
    return false;

  // Otherwise, if we don't have a definition of the key function, the
  // vtable must be defined somewhere else.
  return !keyFunction->hasBody();
}

static bool shouldEmitAvailableExternallyVTable(const CIRGenModule &CGM,
                                                const CXXRecordDecl *RD) {
  assert(CGM.getCodeGenOpts().OptimizationLevel == 0 && "NYI");
  return CGM.getCodeGenOpts().OptimizationLevel > 0 &&
         CGM.getCXXABI().canSpeculativelyEmitVTable(RD);
}

/// Given that we're currently at the end of the translation unit, and
/// we've emitted a reference to the vtable for this class, should
/// we define that vtable?
static bool shouldEmitVTableAtEndOfTranslationUnit(CIRGenModule &CGM,
                                                   const CXXRecordDecl *RD) {
  // If vtable is internal then it has to be done.
  if (!CGM.getVTables().isVTableExternal(RD))
    return true;

  // If it's external then maybe we will need it as available_externally.
  return shouldEmitAvailableExternallyVTable(CGM, RD);
}

/// Given that at some point we emitted a reference to one or more
/// vtables, and that we are now at the end of the translation unit,
/// decide whether we should emit them.
void CIRGenModule::buildDeferredVTables() {
#ifndef NDEBUG
  // Remember the size of DeferredVTables, because we're going to assume
  // that this entire operation doesn't modify it.
  size_t savedSize = DeferredVTables.size();
#endif

  for (const CXXRecordDecl *RD : DeferredVTables)
    if (shouldEmitVTableAtEndOfTranslationUnit(*this, RD)) {
      llvm_unreachable("NYI");
      // VTables.GenerateClassData(RD);
    } else if (shouldOpportunisticallyEmitVTables()) {
      llvm_unreachable("NYI");
      // OpportunisticVTables.push_back(RD);
    }

  assert(savedSize == DeferredVTables.size() &&
         "deferred extra vtables during vtable emission?");
  DeferredVTables.clear();
}