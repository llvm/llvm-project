//===----------------------------------------------------------------------===//
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

#include "CIRGenVTables.h"

#include "CIRGenCXXABI.h"
#include "CIRGenModule.h"
#include "mlir/IR/Types.h"
#include "clang/AST/VTableBuilder.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;
using namespace clang;
using namespace clang::CIRGen;

CIRGenVTables::CIRGenVTables(CIRGenModule &cgm)
    : cgm(cgm), vtContext(cgm.getASTContext().getVTableContext()) {}

mlir::Type CIRGenModule::getVTableComponentType() {
  mlir::Type ptrTy = builder.getUInt8PtrTy();
  assert(!cir::MissingFeatures::vtableRelativeLayout());
  return ptrTy;
}

mlir::Type CIRGenVTables::getVTableComponentType() {
  return cgm.getVTableComponentType();
}

cir::RecordType CIRGenVTables::getVTableType(const VTableLayout &layout) {
  SmallVector<mlir::Type, 4> tys;
  mlir::Type componentType = getVTableComponentType();
  for (unsigned i = 0, e = layout.getNumVTables(); i != e; ++i)
    tys.push_back(cir::ArrayType::get(componentType, layout.getVTableSize(i)));

  // FIXME(cir): should VTableLayout be encoded like we do for some
  // AST nodes?
  return cgm.getBuilder().getAnonRecordTy(tys, /*incomplete=*/false);
}

/// This is a callback from Sema to tell us that a particular vtable is
/// required to be emitted in this translation unit.
///
/// This is only called for vtables that _must_ be emitted (mainly due to key
/// functions).  For weak vtables, CodeGen tracks when they are needed and
/// emits them as-needed.
void CIRGenModule::emitVTable(const CXXRecordDecl *rd) {
  vtables.generateClassData(rd);
}

void CIRGenVTables::generateClassData(const CXXRecordDecl *rd) {
  assert(!cir::MissingFeatures::generateDebugInfo());

  if (rd->getNumVBases())
    cgm.errorNYI(rd->getSourceRange(), "emitVirtualInheritanceTables");

  cgm.getCXXABI().emitVTableDefinitions(*this, rd);
}

mlir::Attribute CIRGenVTables::getVTableComponent(
    const VTableLayout &layout, unsigned componentIndex, mlir::Attribute rtti,
    unsigned &nextVTableThunkIndex, unsigned vtableAddressPoint,
    bool vtableHasLocalLinkage) {
  const VTableComponent &component = layout.vtable_components()[componentIndex];

  CIRGenBuilderTy builder = cgm.getBuilder();

  assert(!cir::MissingFeatures::vtableRelativeLayout());

  switch (component.getKind()) {
  case VTableComponent::CK_VCallOffset:
    cgm.errorNYI("getVTableComponent: VCallOffset");
    return mlir::Attribute();
  case VTableComponent::CK_VBaseOffset:
    cgm.errorNYI("getVTableComponent: VBaseOffset");
    return mlir::Attribute();
  case VTableComponent::CK_CompleteDtorPointer:
    cgm.errorNYI("getVTableComponent: CompleteDtorPointer");
    return mlir::Attribute();
  case VTableComponent::CK_DeletingDtorPointer:
    cgm.errorNYI("getVTableComponent: DeletingDtorPointer");
    return mlir::Attribute();
  case VTableComponent::CK_UnusedFunctionPointer:
    cgm.errorNYI("getVTableComponent: UnusedFunctionPointer");
    return mlir::Attribute();

  case VTableComponent::CK_OffsetToTop:
    return builder.getConstPtrAttr(builder.getUInt8PtrTy(),
                                   component.getOffsetToTop().getQuantity());

  case VTableComponent::CK_RTTI:
    assert((mlir::isa<cir::GlobalViewAttr>(rtti) ||
            mlir::isa<cir::ConstPtrAttr>(rtti)) &&
           "expected GlobalViewAttr or ConstPtrAttr");
    return rtti;

  case VTableComponent::CK_FunctionPointer: {
    GlobalDecl gd = component.getGlobalDecl();

    assert(!cir::MissingFeatures::cudaSupport());

    cir::FuncOp fnPtr;
    if (cast<CXXMethodDecl>(gd.getDecl())->isPureVirtual()) {
      cgm.errorNYI("getVTableComponent: CK_FunctionPointer: pure virtual");
      return mlir::Attribute();
    } else if (cast<CXXMethodDecl>(gd.getDecl())->isDeleted()) {
      cgm.errorNYI("getVTableComponent: CK_FunctionPointer: deleted virtual");
      return mlir::Attribute();
    } else if (nextVTableThunkIndex < layout.vtable_thunks().size() &&
               layout.vtable_thunks()[nextVTableThunkIndex].first ==
                   componentIndex) {
      cgm.errorNYI("getVTableComponent: CK_FunctionPointer: thunk");
      return mlir::Attribute();
    } else {
      // Otherwise we can use the method definition directly.
      cir::FuncType fnTy = cgm.getTypes().getFunctionTypeForVTable(gd);
      fnPtr = cgm.getAddrOfFunction(gd, fnTy, /*ForVTable=*/true);
    }

    return cir::GlobalViewAttr::get(
        builder.getUInt8PtrTy(),
        mlir::FlatSymbolRefAttr::get(fnPtr.getSymNameAttr()));
  }
  }

  llvm_unreachable("Unexpected vtable component kind");
}

void CIRGenVTables::createVTableInitializer(cir::GlobalOp &vtableOp,
                                            const clang::VTableLayout &layout,
                                            mlir::Attribute rtti,
                                            bool vtableHasLocalLinkage) {
  mlir::Type componentType = getVTableComponentType();

  const llvm::SmallVectorImpl<unsigned> &addressPoints =
      layout.getAddressPointIndices();
  unsigned nextVTableThunkIndex = 0;

  if (layout.getNumVTables() > 1)
    cgm.errorNYI("emitVTableDefinitions: multiple vtables");

  // We'll need a loop here to handle multiple vtables, but for now we only
  // support one.
  unsigned vtableIndex = 0;
  size_t vtableStart = layout.getVTableOffset(vtableIndex);
  size_t vtableEnd = vtableStart + layout.getVTableSize(vtableIndex);

  // Build a ConstArrayAttr of the vtable components.
  llvm::SmallVector<mlir::Attribute> components;
  for (size_t componentIndex = vtableStart; componentIndex < vtableEnd;
       ++componentIndex) {
    components.push_back(
        getVTableComponent(layout, componentIndex, rtti, nextVTableThunkIndex,
                           addressPoints[vtableIndex], vtableHasLocalLinkage));
  }

  mlir::MLIRContext *mlirContext = &cgm.getMLIRContext();

  // Create a ConstArrayAttr to hold the components.
  auto arr = cir::ConstArrayAttr::get(
      cir::ArrayType::get(componentType, components.size()),
      mlir::ArrayAttr::get(mlirContext, components));

  // Create a ConstRecordAttr to hold the component array.
  const auto members = mlir::ArrayAttr::get(mlirContext, {arr});
  cir::ConstRecordAttr record = cgm.getBuilder().getAnonConstRecord(members);

  // Create a VTableAttr
  auto vtableAttr = cir::VTableAttr::get(record.getType(), record.getMembers());

  // Add the vtable initializer to the vtable global op.
  cgm.setInitializer(vtableOp, vtableAttr);
}

/// Compute the required linkage of the vtable for the given class.
///
/// Note that we only call this at the end of the translation unit.
cir::GlobalLinkageKind CIRGenModule::getVTableLinkage(const CXXRecordDecl *rd) {
  if (!rd->isExternallyVisible())
    return cir::GlobalLinkageKind::InternalLinkage;

  // We're at the end of the translation unit, so the current key
  // function is fully correct.
  const CXXMethodDecl *keyFunction = astContext.getCurrentKeyFunction(rd);
  if (keyFunction && !rd->hasAttr<DLLImportAttr>()) {
    // If this class has a key function, use that to determine the
    // linkage of the vtable.
    const FunctionDecl *def = nullptr;
    if (keyFunction->hasBody(def))
      keyFunction = cast<CXXMethodDecl>(def);

    // All of the cases below do something different with AppleKext enabled.
    assert(!cir::MissingFeatures::appleKext());
    switch (keyFunction->getTemplateSpecializationKind()) {
    case TSK_Undeclared:
    case TSK_ExplicitSpecialization:
      assert(
          (def || codeGenOpts.OptimizationLevel > 0 ||
           codeGenOpts.getDebugInfo() != llvm::codegenoptions::NoDebugInfo) &&
          "Shouldn't query vtable linkage without key function, "
          "optimizations, or debug info");
      if (!def && codeGenOpts.OptimizationLevel > 0)
        return cir::GlobalLinkageKind::AvailableExternallyLinkage;

      if (keyFunction->isInlined())
        return !astContext.getLangOpts().AppleKext
                   ? cir::GlobalLinkageKind::LinkOnceODRLinkage
                   : cir::GlobalLinkageKind::InternalLinkage;
      return cir::GlobalLinkageKind::ExternalLinkage;

    case TSK_ImplicitInstantiation:
      return cir::GlobalLinkageKind::LinkOnceODRLinkage;

    case TSK_ExplicitInstantiationDefinition:
      return cir::GlobalLinkageKind::WeakODRLinkage;

    case TSK_ExplicitInstantiationDeclaration:
      llvm_unreachable("Should not have been asked to emit this");
    }
  }

  errorNYI(rd->getSourceRange(), "getVTableLinkage: no key function");
  return cir::GlobalLinkageKind::ExternalLinkage;
}

void CIRGenVTables::emitThunks(GlobalDecl gd) {
  const CXXMethodDecl *md =
      cast<CXXMethodDecl>(gd.getDecl())->getCanonicalDecl();

  // We don't need to generate thunks for the base destructor.
  if (isa<CXXDestructorDecl>(md) && gd.getDtorType() == Dtor_Base)
    return;

  const VTableContextBase::ThunkInfoVectorTy *thunkInfoVector =
      vtContext->getThunkInfo(gd);

  if (!thunkInfoVector)
    return;

  cgm.errorNYI(md->getSourceRange(), "emitThunks");
}
