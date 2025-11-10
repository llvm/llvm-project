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
#include "clang/AST/VTTBuilder.h"
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
bool CIRGenVTables::isVTableExternal(const CXXRecordDecl *rd) {
  assert(rd->isDynamicClass() && "Non-dynamic classes have no VTable.");

  // We always synthesize vtables if they are needed in the MS ABI. MSVC doesn't
  // emit them even if there is an explicit template instantiation.
  if (cgm.getTarget().getCXXABI().isMicrosoft())
    return false;

  // If we have an explicit instantiation declaration (and not a
  // definition), the vtable is defined elsewhere.
  TemplateSpecializationKind tsk = rd->getTemplateSpecializationKind();
  if (tsk == TSK_ExplicitInstantiationDeclaration)
    return true;

  // Otherwise, if the class is an instantiated template, the
  // vtable must be defined here.
  if (tsk == TSK_ImplicitInstantiation ||
      tsk == TSK_ExplicitInstantiationDefinition)
    return false;

  // Otherwise, if the class doesn't have a key function (possibly
  // anymore), the vtable must be defined here.
  const CXXMethodDecl *keyFunction =
      cgm.getASTContext().getCurrentKeyFunction(rd);
  if (!keyFunction)
    return false;

  // Otherwise, if we don't have a definition of the key function, the
  // vtable must be defined somewhere else.
  return !keyFunction->hasBody();
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
    cgm.getCXXABI().emitVirtualInheritanceTables(rd);

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
  case VTableComponent::CK_UnusedFunctionPointer:
    cgm.errorNYI("getVTableComponent: UnusedFunctionPointer");
    return mlir::Attribute();

  case VTableComponent::CK_VCallOffset:
    return builder.getConstPtrAttr(builder.getUInt8PtrTy(),
                                   component.getVCallOffset().getQuantity());

  case VTableComponent::CK_VBaseOffset:
    return builder.getConstPtrAttr(builder.getUInt8PtrTy(),
                                   component.getVBaseOffset().getQuantity());

  case VTableComponent::CK_OffsetToTop:
    return builder.getConstPtrAttr(builder.getUInt8PtrTy(),
                                   component.getOffsetToTop().getQuantity());

  case VTableComponent::CK_RTTI:
    assert((mlir::isa<cir::GlobalViewAttr>(rtti) ||
            mlir::isa<cir::ConstPtrAttr>(rtti)) &&
           "expected GlobalViewAttr or ConstPtrAttr");
    return rtti;

  case VTableComponent::CK_FunctionPointer:
  case VTableComponent::CK_CompleteDtorPointer:
  case VTableComponent::CK_DeletingDtorPointer: {
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

  mlir::MLIRContext *mlirContext = &cgm.getMLIRContext();

  SmallVector<mlir::Attribute> vtables;
  for (auto [vtableIndex, addressPoint] : llvm::enumerate(addressPoints)) {
    // Build a ConstArrayAttr of the vtable components.
    size_t vtableStart = layout.getVTableOffset(vtableIndex);
    size_t vtableEnd = vtableStart + layout.getVTableSize(vtableIndex);
    llvm::SmallVector<mlir::Attribute> components;
    components.reserve(vtableEnd - vtableStart);
    for (size_t componentIndex : llvm::seq(vtableStart, vtableEnd))
      components.push_back(
          getVTableComponent(layout, componentIndex, rtti, nextVTableThunkIndex,
                             addressPoint, vtableHasLocalLinkage));
    // Create a ConstArrayAttr to hold the components.
    auto arr = cir::ConstArrayAttr::get(
        cir::ArrayType::get(componentType, components.size()),
        mlir::ArrayAttr::get(mlirContext, components));
    vtables.push_back(arr);
  }

  // Create a ConstRecordAttr to hold the component array.
  const auto members = mlir::ArrayAttr::get(mlirContext, vtables);
  cir::ConstRecordAttr record = cgm.getBuilder().getAnonConstRecord(members);

  // Create a VTableAttr
  auto vtableAttr = cir::VTableAttr::get(record.getType(), record.getMembers());

  // Add the vtable initializer to the vtable global op.
  cgm.setInitializer(vtableOp, vtableAttr);
}

cir::GlobalOp CIRGenVTables::generateConstructionVTable(
    const CXXRecordDecl *rd, const BaseSubobject &base, bool baseIsVirtual,
    cir::GlobalLinkageKind linkage, VTableAddressPointsMapTy &addressPoints) {
  assert(!cir::MissingFeatures::generateDebugInfo());

  std::unique_ptr<VTableLayout> vtLayout(
      getItaniumVTableContext().createConstructionVTableLayout(
          base.getBase(), base.getBaseOffset(), baseIsVirtual, rd));

  // Add the address points.
  addressPoints = vtLayout->getAddressPoints();

  // Get the mangled construction vtable name.
  SmallString<256> outName;
  llvm::raw_svector_ostream out(outName);
  cast<ItaniumMangleContext>(cgm.getCXXABI().getMangleContext())
      .mangleCXXCtorVTable(rd, base.getBaseOffset().getQuantity(),
                           base.getBase(), out);
  SmallString<256> name(outName);

  assert(!cir::MissingFeatures::vtableRelativeLayout());

  cir::RecordType vtType = getVTableType(*vtLayout);

  // Construction vtable symbols are not part of the Itanium ABI, so we cannot
  // guarantee that they actually will be available externally. Instead, when
  // emitting an available_externally VTT, we provide references to an internal
  // linkage construction vtable. The ABI only requires complete-object vtables
  // to be the same for all instances of a type, not construction vtables.
  if (linkage == cir::GlobalLinkageKind::AvailableExternallyLinkage)
    linkage = cir::GlobalLinkageKind::InternalLinkage;

  llvm::Align align = cgm.getDataLayout().getABITypeAlign(vtType);
  mlir::Location loc = cgm.getLoc(rd->getSourceRange());

  // Create the variable that will hold the construction vtable.
  cir::GlobalOp vtable = cgm.createOrReplaceCXXRuntimeVariable(
      loc, name, vtType, linkage, CharUnits::fromQuantity(align));

  // V-tables are always unnamed_addr.
  assert(!cir::MissingFeatures::opGlobalUnnamedAddr());

  mlir::Attribute rtti = cgm.getAddrOfRTTIDescriptor(
      loc, cgm.getASTContext().getCanonicalTagType(base.getBase()));

  // Create and set the initializer.
  createVTableInitializer(vtable, *vtLayout, rtti,
                          cir::isLocalLinkage(vtable.getLinkage()));

  // Set properties only after the initializer has been set to ensure that the
  // GV is treated as definition and not declaration.
  assert(!vtable.isDeclaration() && "Shouldn't set properties on declaration");
  cgm.setGVProperties(vtable, rd);

  assert(!cir::MissingFeatures::vtableEmitMetadata());
  assert(!cir::MissingFeatures::vtableRelativeLayout());

  return vtable;
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
  // -fapple-kext mode does not support weak linkage, so we must use
  // internal linkage.
  if (astContext.getLangOpts().AppleKext)
    return cir::GlobalLinkageKind::InternalLinkage;

  auto discardableODRLinkage = cir::GlobalLinkageKind::LinkOnceODRLinkage;
  auto nonDiscardableODRLinkage = cir::GlobalLinkageKind::WeakODRLinkage;
  if (rd->hasAttr<DLLExportAttr>()) {
    // Cannot discard exported vtables.
    discardableODRLinkage = nonDiscardableODRLinkage;
  } else if (rd->hasAttr<DLLImportAttr>()) {
    // Imported vtables are available externally.
    discardableODRLinkage = cir::GlobalLinkageKind::AvailableExternallyLinkage;
    nonDiscardableODRLinkage =
        cir::GlobalLinkageKind::AvailableExternallyLinkage;
  }

  switch (rd->getTemplateSpecializationKind()) {
  case TSK_Undeclared:
  case TSK_ExplicitSpecialization:
  case TSK_ImplicitInstantiation:
    return discardableODRLinkage;

  case TSK_ExplicitInstantiationDeclaration: {
    errorNYI(rd->getSourceRange(),
             "getVTableLinkage: explicit instantiation declaration");
    return cir::GlobalLinkageKind::ExternalLinkage;
  }

  case TSK_ExplicitInstantiationDefinition:
    return nonDiscardableODRLinkage;
  }

  llvm_unreachable("Invalid TemplateSpecializationKind!");
}

cir::GlobalOp CIRGenVTables::getAddrOfVTT(const CXXRecordDecl *rd) {
  assert(rd->getNumVBases() && "Only classes with virtual bases need a VTT");

  SmallString<256> outName;
  llvm::raw_svector_ostream out(outName);
  cast<ItaniumMangleContext>(cgm.getCXXABI().getMangleContext())
      .mangleCXXVTT(rd, out);
  StringRef name = outName.str();

  // This will also defer the definition of the VTT.
  (void)cgm.getCXXABI().getAddrOfVTable(rd, CharUnits());

  VTTBuilder builder(cgm.getASTContext(), rd, /*GenerateDefinition=*/false);

  auto arrayType = cir::ArrayType::get(cgm.getBuilder().getUInt8PtrTy(),
                                       builder.getVTTComponents().size());
  llvm::Align align =
      cgm.getDataLayout().getABITypeAlign(cgm.getBuilder().getUInt8PtrTy());
  cir::GlobalOp vtt = cgm.createOrReplaceCXXRuntimeVariable(
      cgm.getLoc(rd->getSourceRange()), name, arrayType,
      cir::GlobalLinkageKind::ExternalLinkage, CharUnits::fromQuantity(align));
  cgm.setGVProperties(vtt, rd);
  return vtt;
}

static cir::GlobalOp
getAddrOfVTTVTable(CIRGenVTables &cgvt, CIRGenModule &cgm,
                   const CXXRecordDecl *mostDerivedClass,
                   const VTTVTable &vtable, cir::GlobalLinkageKind linkage,
                   VTableLayout::AddressPointsMapTy &addressPoints) {
  if (vtable.getBase() == mostDerivedClass) {
    assert(vtable.getBaseOffset().isZero() &&
           "Most derived class vtable must have a zero offset!");
    // This is a regular vtable.
    return cgm.getCXXABI().getAddrOfVTable(mostDerivedClass, CharUnits());
  }
  return cgvt.generateConstructionVTable(
      mostDerivedClass, vtable.getBaseSubobject(), vtable.isVirtual(), linkage,
      addressPoints);
}

/// Emit the definition of the given vtable.
void CIRGenVTables::emitVTTDefinition(cir::GlobalOp vttOp,
                                      cir::GlobalLinkageKind linkage,
                                      const CXXRecordDecl *rd) {
  VTTBuilder builder(cgm.getASTContext(), rd, /*GenerateDefinition=*/true);

  mlir::MLIRContext *mlirContext = &cgm.getMLIRContext();

  auto arrayType = cir::ArrayType::get(cgm.getBuilder().getUInt8PtrTy(),
                                       builder.getVTTComponents().size());

  SmallVector<cir::GlobalOp> vtables;
  SmallVector<VTableAddressPointsMapTy> vtableAddressPoints;
  for (const VTTVTable &vtt : builder.getVTTVTables()) {
    vtableAddressPoints.push_back(VTableAddressPointsMapTy());
    vtables.push_back(getAddrOfVTTVTable(*this, cgm, rd, vtt, linkage,
                                         vtableAddressPoints.back()));
  }

  SmallVector<mlir::Attribute> vttComponents;
  for (const VTTComponent &vttComponent : builder.getVTTComponents()) {
    const VTTVTable &vttVT = builder.getVTTVTables()[vttComponent.VTableIndex];
    cir::GlobalOp vtable = vtables[vttComponent.VTableIndex];
    VTableLayout::AddressPointLocation addressPoint;
    if (vttVT.getBase() == rd) {
      // Just get the address point for the regular vtable.
      addressPoint =
          getItaniumVTableContext().getVTableLayout(rd).getAddressPoint(
              vttComponent.VTableBase);
    } else {
      addressPoint = vtableAddressPoints[vttComponent.VTableIndex].lookup(
          vttComponent.VTableBase);
      assert(addressPoint.AddressPointIndex != 0 &&
             "Did not find ctor vtable address point!");
    }

    mlir::Attribute indices[2] = {
        cgm.getBuilder().getI32IntegerAttr(addressPoint.VTableIndex),
        cgm.getBuilder().getI32IntegerAttr(addressPoint.AddressPointIndex),
    };

    auto indicesAttr = mlir::ArrayAttr::get(mlirContext, indices);
    cir::GlobalViewAttr init = cgm.getBuilder().getGlobalViewAttr(
        cgm.getBuilder().getUInt8PtrTy(), vtable, indicesAttr);

    vttComponents.push_back(init);
  }

  auto init = cir::ConstArrayAttr::get(
      arrayType, mlir::ArrayAttr::get(mlirContext, vttComponents));

  vttOp.setInitialValueAttr(init);

  // Set the correct linkage.
  vttOp.setLinkage(linkage);
  mlir::SymbolTable::setSymbolVisibility(
      vttOp, CIRGenModule::getMLIRVisibility(vttOp));

  if (cgm.supportsCOMDAT() && vttOp.isWeakForLinker())
    vttOp.setComdat(true);
}

uint64_t CIRGenVTables::getSubVTTIndex(const CXXRecordDecl *rd,
                                       BaseSubobject base) {
  BaseSubobjectPairTy classSubobjectPair(rd, base);

  SubVTTIndiciesMapTy::iterator it = subVTTIndicies.find(classSubobjectPair);
  if (it != subVTTIndicies.end())
    return it->second;

  VTTBuilder builder(cgm.getASTContext(), rd, /*GenerateDefinition=*/false);

  for (const auto &entry : builder.getSubVTTIndices()) {
    // Insert all indices.
    BaseSubobjectPairTy subclassSubobjectPair(rd, entry.first);

    subVTTIndicies.insert(std::make_pair(subclassSubobjectPair, entry.second));
  }

  it = subVTTIndicies.find(classSubobjectPair);
  assert(it != subVTTIndicies.end() && "Did not find index!");

  return it->second;
}

uint64_t CIRGenVTables::getSecondaryVirtualPointerIndex(const CXXRecordDecl *rd,
                                                        BaseSubobject base) {
  auto it = secondaryVirtualPointerIndices.find(std::make_pair(rd, base));

  if (it != secondaryVirtualPointerIndices.end())
    return it->second;

  VTTBuilder builder(cgm.getASTContext(), rd, /*GenerateDefinition=*/false);

  // Insert all secondary vpointer indices.
  for (const auto &entry : builder.getSecondaryVirtualPointerIndices()) {
    std::pair<const CXXRecordDecl *, BaseSubobject> pair =
        std::make_pair(rd, entry.first);

    secondaryVirtualPointerIndices.insert(std::make_pair(pair, entry.second));
  }

  it = secondaryVirtualPointerIndices.find(std::make_pair(rd, base));
  assert(it != secondaryVirtualPointerIndices.end() && "Did not find index!");

  return it->second;
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
