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

cir::FuncOp CIRGenModule::getAddrOfThunk(StringRef name, mlir::Type fnTy,
                                         GlobalDecl gd) {
  return getOrCreateCIRFunction(name, fnTy, gd, /*forVTable=*/true,
                                /*dontDefer=*/true, /*isThunk=*/true);
}

static void setThunkProperties(CIRGenModule &cgm, const ThunkInfo &thunk,
                               cir::FuncOp thunkFn, bool forVTable,
                               GlobalDecl gd) {
  cgm.setFunctionLinkage(gd, thunkFn);
  cgm.getCXXABI().setThunkLinkage(thunkFn, forVTable, gd,
                                  !thunk.Return.isEmpty());

  // Set the right visibility.
  cgm.setGVProperties(thunkFn, cast<NamedDecl>(gd.getDecl()));

  if (!cgm.getCXXABI().exportThunk()) {
    assert(!cir::MissingFeatures::setDLLStorageClass());
    cgm.setDSOLocal(static_cast<mlir::Operation *>(thunkFn));
  }

  if (cgm.supportsCOMDAT() && thunkFn.isWeakForLinker())
    thunkFn.setComdat(true);
}

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
    return builder.getConstNullPtrAttr(builder.getUInt8PtrTy());

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
    GlobalDecl gd = component.getGlobalDecl(
        cgm.getASTContext().getTargetInfo().emitVectorDeletingDtors(
            cgm.getASTContext().getLangOpts()));

    assert(!cir::MissingFeatures::cudaSupport());

    auto getSpecialVirtFn = [&](StringRef name) -> cir::FuncOp {
      assert(!cir::MissingFeatures::vtableRelativeLayout());

      if (cgm.getLangOpts().OpenMP && cgm.getLangOpts().OpenMPIsTargetDevice &&
          cgm.getTriple().isNVPTX())
        cgm.errorNYI(gd.getDecl()->getSourceRange(),
                     "getVTableComponent for OMP Device NVPTX");

      cir::FuncType fnTy =
          cgm.getBuilder().getFuncType({}, cgm.getBuilder().getVoidTy());
      cir::FuncOp fnPtr = cgm.createRuntimeFunction(fnTy, name);

      assert(!cir::MissingFeatures::opGlobalUnnamedAddr());
      return fnPtr;
    };

    cir::FuncOp fnPtr;
    if (cast<CXXMethodDecl>(gd.getDecl())->isPureVirtual()) {
      if (!pureVirtualFn)
        pureVirtualFn =
            getSpecialVirtFn(cgm.getCXXABI().getPureVirtualCallName());
      fnPtr = pureVirtualFn;
    } else if (cast<CXXMethodDecl>(gd.getDecl())->isDeleted()) {
      if (!deletedVirtualFn)
        deletedVirtualFn =
            getSpecialVirtFn(cgm.getCXXABI().getDeletedVirtualCallName());
      fnPtr = deletedVirtualFn;
    } else if (nextVTableThunkIndex < layout.vtable_thunks().size() &&
               layout.vtable_thunks()[nextVTableThunkIndex].first ==
                   componentIndex) {
      const ThunkInfo &thunkInfo =
          layout.vtable_thunks()[nextVTableThunkIndex].second;
      nextVTableThunkIndex++;
      fnPtr = maybeEmitThunk(gd, thunkInfo, /*forVTable=*/true);
      assert(!cir::MissingFeatures::pointerAuthentication());
    } else {
      // Otherwise we can use the method definition directly.
      cir::FuncType fnTy = cgm.getTypes().getFunctionType(gd);
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

static RValue performReturnAdjustment(CIRGenFunction &cgf, QualType resultType,
                                      RValue rv, const ThunkInfo &thunk) {
  // Emit the return adjustment.  For non-reference pointer returns, match
  // classic codegen: skip the adjustment when the returned pointer is null.
  bool nullCheckValue = !resultType->isReferenceType();
  mlir::Value returnValue = rv.getValue();

  const CXXRecordDecl *classDecl =
      resultType->getPointeeType()->getAsCXXRecordDecl();
  CharUnits classAlign = cgf.cgm.getClassPointerAlignment(classDecl);
  mlir::Type pointeeType = cgf.convertTypeForMem(resultType->getPointeeType());
  CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Location loc = returnValue.getLoc();

  if (!nullCheckValue) {
    returnValue = cgf.cgm.getCXXABI().performReturnAdjustment(
        cgf, Address(returnValue, pointeeType, classAlign), classDecl,
        thunk.Return);
    return RValue::get(returnValue);
  }

  mlir::Value isNotNull = builder.createPtrIsNotNull(returnValue);
  returnValue =
      cir::TernaryOp::create(
          builder, loc, isNotNull,
          [&](mlir::OpBuilder &, mlir::Location) {
            mlir::Value adjusted = cgf.cgm.getCXXABI().performReturnAdjustment(
                cgf, Address(returnValue, pointeeType, classAlign), classDecl,
                thunk.Return);
            builder.createYield(loc, adjusted);
          },
          [&](mlir::OpBuilder &, mlir::Location) {
            mlir::Value nullVal =
                builder.getNullPtr(returnValue.getType(), loc).getResult();
            builder.createYield(loc, nullVal);
          })
          .getResult();

  return RValue::get(returnValue);
}

void CIRGenFunction::startThunk(cir::FuncOp fn, GlobalDecl gd,
                                const CIRGenFunctionInfo &fnInfo,
                                bool isUnprototyped) {
  assert(!curGD.getDecl() && "curGD was already set!");
  curGD = gd;
  curFuncIsThunk = true;

  // Build FunctionArgs.
  const CXXMethodDecl *md = cast<CXXMethodDecl>(gd.getDecl());
  QualType thisType = md->getThisType();
  QualType resultType;
  if (isUnprototyped)
    resultType = cgm.getASTContext().VoidTy;
  else if (cgm.getCXXABI().hasThisReturn(gd))
    resultType = thisType;
  else if (cgm.getCXXABI().hasMostDerivedReturn(gd))
    resultType = cgm.getASTContext().VoidPtrTy;
  else
    resultType = md->getType()->castAs<FunctionProtoType>()->getReturnType();
  FunctionArgList functionArgs;

  // Create the implicit 'this' parameter declaration.
  cgm.getCXXABI().buildThisParam(*this, functionArgs);

  // Add the rest of the parameters, if we have a prototype to work with.
  if (!isUnprototyped) {
    functionArgs.append(md->param_begin(), md->param_end());

    if (isa<CXXDestructorDecl>(md))
      cgm.getCXXABI().addImplicitStructorParams(*this, resultType,
                                                functionArgs);
  }

  assert(!cir::MissingFeatures::generateDebugInfo());

  // Start defining the function.
  cir::FuncType funcType = cgm.getTypes().getFunctionType(fnInfo);
  startFunction(GlobalDecl(), resultType, fn, funcType, functionArgs,
                md->getLocation(), md->getLocation());
  // TODO(cir): Move this into startFunction.
  curFnInfo = &fnInfo;
  assert(!cir::MissingFeatures::generateDebugInfo());

  // Since we didn't pass a GlobalDecl to startFunction, do this ourselves.
  cgm.getCXXABI().emitInstanceFunctionProlog(md->getLocation(), *this);
  cxxThisValue = cxxabiThisValue;
  curCodeDecl = md;
  curFuncDecl = md;
}

void CIRGenFunction::finishThunk() {
  // Clear these to restore the invariants expected by
  // startFunction/finishFunction.
  curCodeDecl = nullptr;
  curFuncDecl = nullptr;

  finishFunction(SourceLocation());
}

void CIRGenFunction::emitCallAndReturnForThunk(cir::FuncOp callee,
                                               const ThunkInfo *thunk,
                                               bool isUnprototyped) {
  assert(isa<CXXMethodDecl>(curGD.getDecl()) &&
         "Please use a new CGF for this thunk");
  const CXXMethodDecl *md = cast<CXXMethodDecl>(curGD.getDecl());

  // Determine the this pointer class (may differ from md's class for thunks).
  const CXXRecordDecl *thisValueClass =
      md->getThisType()->getPointeeCXXRecordDecl();
  if (thunk)
    thisValueClass = thunk->ThisType->getPointeeCXXRecordDecl();

  mlir::Value adjustedThisPtr =
      thunk ? cgm.getCXXABI().performThisAdjustment(*this, loadCXXThisAddress(),
                                                    thisValueClass, *thunk)
            : loadCXXThis();

  // If perfect forwarding is required a variadic method, a method using
  // inalloca, or an unprototyped thunk, use musttail. Emit an error if this
  // thunk requires a return adjustment, since that is impossible with musttail.
  assert(!cir::MissingFeatures::opCallInAlloca());
  if ((curFnInfo && curFnInfo->isVariadic()) || isUnprototyped) {
    // Error if return adjustment is needed (can't do with musttail).
    if (thunk && !thunk->Return.isEmpty()) {
      if (isUnprototyped)
        cgm.errorUnsupported(
            md, "return-adjusting thunk with incomplete parameter type");
      else if (curFnInfo && curFnInfo->isVariadic())
        llvm_unreachable("shouldn't try to emit musttail return-adjusting "
                         "thunks for variadic functions");
      else
        cgm.errorUnsupported(
            md, "non-trivial argument copy for return-adjusting thunk");
    }
    emitMustTailThunk(curGD, adjustedThisPtr, callee);
    return;
  }

  // Build the call argument list.
  CallArgList callArgs;
  QualType thisType = md->getThisType();
  callArgs.add(RValue::get(adjustedThisPtr), thisType);

  if (isa<CXXDestructorDecl>(md))
    cgm.getCXXABI().adjustCallArgsForDestructorThunk(*this, curGD, callArgs);

#ifndef NDEBUG
  unsigned prefixArgs = callArgs.size() - 1;
#endif

  // Add the rest of the method parameters.
  for (const ParmVarDecl *pd : md->parameters())
    emitDelegateCallArg(callArgs, pd, SourceLocation());

  const FunctionProtoType *fpt = md->getType()->castAs<FunctionProtoType>();

#ifndef NDEBUG
  const CIRGenFunctionInfo &callFnInfo = cgm.getTypes().arrangeCXXMethodCall(
      callArgs, fpt, RequiredArgs::getFromProtoWithExtraSlots(fpt, 1),
      prefixArgs);
  assert(callFnInfo.argTypeSize() == curFnInfo->argTypeSize());
#endif

  // Determine whether we have a return value slot to use.
  QualType resultType = cgm.getCXXABI().hasThisReturn(curGD) ? thisType
                        : cgm.getCXXABI().hasMostDerivedReturn(curGD)
                            ? cgm.getASTContext().VoidPtrTy
                            : fpt->getReturnType();

  ReturnValueSlot slot;
  // This should also be tracking volatile, unused, and externally destructed.
  assert(!cir::MissingFeatures::returnValueSlotFeatures());
  if (!resultType->isVoidType() && hasAggregateEvaluationKind(resultType))
    slot = ReturnValueSlot(returnValue);

  // Now emit our call.
  CIRGenCallee cirCallee = CIRGenCallee::forDirect(callee, curGD);
  mlir::Location loc = builder.getUnknownLoc();
  RValue rv = emitCall(*curFnInfo, cirCallee, slot, callArgs,
                       /*callOrTryCall=*/nullptr, loc);

  // Consider return adjustment if we have ThunkInfo.
  if (thunk && !thunk->Return.isEmpty())
    rv = performReturnAdjustment(*this, resultType, rv, *thunk);
  else
    assert(!cir::MissingFeatures::opCallMustTail());

  // Emit return.
  if (!resultType->isVoidType() && slot.isNull())
    cgm.getCXXABI().emitReturnFromThunk(*this, rv, resultType);

  // Disable final ARC autorelease.
  assert(!cir::MissingFeatures::objCLifetime());

  finishThunk();
}

void CIRGenFunction::emitMustTailThunk(GlobalDecl gd,
                                       mlir::Value adjustedThisPtr,
                                       cir::FuncOp callee) {
  // Forward all function arguments, replacing 'this' with the adjusted pointer.
  // The call is marked musttail so varargs are forwarded correctly.
  mlir::Block *entryBlock = getCurFunctionEntryBlock();
  SmallVector<mlir::Value> args;
  for (mlir::BlockArgument arg : entryBlock->getArguments())
    args.push_back(arg);

  // Replace the 'this' argument (first arg) with the adjusted pointer.
  assert(!args.empty() && "thunk must have at least 'this' argument");
  if (adjustedThisPtr.getType() != args[0].getType())
    adjustedThisPtr = builder.createBitcast(adjustedThisPtr, args[0].getType());
  args[0] = adjustedThisPtr;

  mlir::Location loc = curFn->getLoc();
  cir::FuncType calleeTy = callee.getFunctionType();
  mlir::Type retTy = calleeTy.getReturnType();

  cir::CallOp call = builder.createCallOp(loc, callee, args);
  call->setAttr(cir::CIRDialect::getMustTailAttrName(),
                mlir::UnitAttr::get(builder.getContext()));

  if (isa<cir::VoidType>(retTy))
    cir::ReturnOp::create(builder, loc);
  else
    cir::ReturnOp::create(builder, loc, call->getResult(0));

  finishThunk();
}

void CIRGenFunction::generateThunk(cir::FuncOp fn,
                                   const CIRGenFunctionInfo &fnInfo,
                                   GlobalDecl gd, const ThunkInfo &thunk,
                                   bool isUnprototyped) {
  // Create entry block and set up the builder's insertion point.
  // This must be done before calling startThunk() which calls startFunction().
  assert(fn.isDeclaration() && "Function already has body?");
  mlir::Block *entryBb = fn.addEntryBlock();
  builder.setInsertionPointToStart(entryBb);

  // Create a scope in the symbol table to hold variable declarations.
  // This is required before startFunction processes parameters, as it will
  // insert them into the symbolTable (ScopedHashTable) which requires an
  // active scope.
  SymTableScopeTy varScope(symbolTable);

  // Create lexical scope - must stay alive for entire thunk generation.
  // startFunction() requires currLexScope to be set.
  SourceLocRAIIObject locRAII(*this, fn.getLoc());
  LexicalScope lexScope{*this, fn.getLoc(), entryBb};

  startThunk(fn, gd, fnInfo, isUnprototyped);
  assert(!cir::MissingFeatures::generateDebugInfo());

  // Get our callee. Use a placeholder type if this method is unprototyped so
  // that CIRGenModule doesn't try to set attributes.
  mlir::Type ty;
  if (isUnprototyped)
    cgm.errorNYI("unprototyped thunk placeholder type");
  else
    ty = cgm.getTypes().getFunctionType(fnInfo);

  cir::FuncOp calleeOp = cgm.getAddrOfFunction(gd, ty, /*forVTable=*/true);

  // Make the call and return the result.
  emitCallAndReturnForThunk(calleeOp, &thunk, isUnprototyped);
}

static bool shouldEmitVTableThunk(CIRGenModule &cgm, const CXXMethodDecl *md,
                                  bool isUnprototyped, bool forVTable) {
  // Always emit thunks in the MS C++ ABI. We cannot rely on other TUs to
  // provide thunks for us.
  if (cgm.getTarget().getCXXABI().isMicrosoft())
    return true;

  // In the Itanium C++ ABI, vtable thunks are provided by TUs that provide
  // definitions of the main method. Therefore, emitting thunks with the vtable
  // is purely an optimization. Emit the thunk if optimizations are enabled and
  // all of the parameter types are complete.
  if (forVTable)
    return cgm.getCodeGenOpts().OptimizationLevel && !isUnprototyped;

  // Always emit thunks along with the method definition.
  return true;
}

cir::FuncOp CIRGenVTables::maybeEmitThunk(GlobalDecl gd,
                                          const ThunkInfo &thunkAdjustments,
                                          bool forVTable) {
  const CXXMethodDecl *md = cast<CXXMethodDecl>(gd.getDecl());
  SmallString<256> name;
  MangleContext &mCtx = cgm.getCXXABI().getMangleContext();

  llvm::raw_svector_ostream out(name);
  if (const CXXDestructorDecl *dd = dyn_cast<CXXDestructorDecl>(md)) {
    mCtx.mangleCXXDtorThunk(dd, gd.getDtorType(), thunkAdjustments,
                            /*elideOverrideInfo=*/false, out);
  } else {
    mCtx.mangleThunk(md, thunkAdjustments, /*elideOverrideInfo=*/false, out);
  }

  if (cgm.getASTContext().useAbbreviatedThunkName(gd, name.str())) {
    name = "";
    if (const CXXDestructorDecl *dd = dyn_cast<CXXDestructorDecl>(md))
      mCtx.mangleCXXDtorThunk(dd, gd.getDtorType(), thunkAdjustments,
                              /*elideOverrideInfo=*/true, out);
    else
      mCtx.mangleThunk(md, thunkAdjustments, /*elideOverrideInfo=*/true, out);
  }

  cir::FuncType thunkVTableTy = cgm.getTypes().getFunctionType(gd);
  cir::FuncOp thunk = cgm.getAddrOfThunk(name, thunkVTableTy, gd);

  // If we don't need to emit a definition, return this declaration as is.
  bool isUnprototyped = !cgm.getTypes().isFuncTypeConvertible(
      md->getType()->castAs<FunctionType>());
  if (!shouldEmitVTableThunk(cgm, md, isUnprototyped, forVTable))
    return thunk;

  // Arrange a function prototype appropriate for a function definition. In some
  // cases in the MS ABI, we may need to build an unprototyped musttail thunk.
  const CIRGenFunctionInfo &fnInfo =
      isUnprototyped ? (cgm.errorNYI("unprototyped must-tail thunk"),
                        cgm.getTypes().arrangeGlobalDeclaration(gd))
                     : cgm.getTypes().arrangeGlobalDeclaration(gd);
  cir::FuncType thunkFnTy = cgm.getTypes().getFunctionType(fnInfo);

  // This is to replace OG's casting to a function, keeping it here to
  // streamline the 1-to-1 mapping from OG starting below.
  cir::FuncOp thunkFn = thunk;
  if (thunk.getFunctionType() != thunkFnTy) {
    cir::FuncOp oldThunkFn = thunkFn;

    assert(oldThunkFn.isDeclaration() && "Shouldn't replace non-declaration");

    // Remove the name from the old thunk function and get a new thunk.
    oldThunkFn.setName(StringRef());
    thunkFn =
        cir::FuncOp::create(cgm.getBuilder(), thunk->getLoc(), name.str(),
                            thunkFnTy, cir::GlobalLinkageKind::ExternalLinkage);
    cgm.setCIRFunctionAttributes(md, fnInfo, thunkFn, /*isThunk=*/false);

    if (!oldThunkFn->use_empty())
      oldThunkFn->replaceAllUsesWith(thunkFn);

    // Remove the old thunk.
    oldThunkFn->erase();
  }

  bool abiHasKeyFunctions = cgm.getTarget().getCXXABI().hasKeyFunctions();
  bool useAvailableExternallyLinkage = forVTable && abiHasKeyFunctions;

  // If the type of the underlying GlobalValue is wrong, we'll have to replace
  // it. It should be a declaration.
  if (!thunkFn.isDeclaration()) {
    if (!abiHasKeyFunctions || useAvailableExternallyLinkage) {
      // There is already a thunk emitted for this function, do nothing.
      return thunkFn;
    }

    setThunkProperties(cgm, thunkAdjustments, thunkFn, forVTable, gd);
    return thunkFn;
  }

  // TODO(cir): Add "thunk" attribute if unprototyped.

  cgm.setCIRFunctionAttributesForDefinition(cast<FunctionDecl>(gd.getDecl()),
                                            thunkFn);

  // Thunks for variadic methods are special because in general variadic
  // arguments cannot be perfectly forwarded. In the general case, clang
  // implements such thunks by cloning the original function body. However, for
  // thunks with no return adjustment on targets that support musttail, we can
  // use musttail to perfectly forward the variadic arguments.
  bool shouldCloneVarArgs = false;
  if (!isUnprototyped && thunkFn.getFunctionType().isVarArg()) {
    shouldCloneVarArgs = true;
    if (thunkAdjustments.Return.isEmpty()) {
      switch (cgm.getTriple().getArch()) {
      case llvm::Triple::x86_64:
      case llvm::Triple::x86:
      case llvm::Triple::aarch64:
        shouldCloneVarArgs = false;
        break;
      default:
        break;
      }
    }
  }

  if (shouldCloneVarArgs) {
    if (useAvailableExternallyLinkage)
      return thunkFn;
    cgm.errorNYI("varargs thunk cloning");
  } else {
    // Normal thunk body generation.
    mlir::OpBuilder::InsertionGuard guard(cgm.getBuilder());
    CIRGenFunction cgf(cgm, cgm.getBuilder());
    cgf.generateThunk(thunkFn, fnInfo, gd, thunkAdjustments, isUnprototyped);
  }

  setThunkProperties(cgm, thunkAdjustments, thunkFn, forVTable, gd);
  return thunkFn;
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

  for (const ThunkInfo &thunk : *thunkInfoVector)
    maybeEmitThunk(gd, thunk, /*forVTable=*/false);
}

static bool shouldEmitAvailableExternallyVTable(const CIRGenModule &cgm,
                                                const CXXRecordDecl *rd) {
  return cgm.getCodeGenOpts().OptimizationLevel > 0 &&
         cgm.getCXXABI().canSpeculativelyEmitVTable(rd);
}

/// Given that we're currently at the end of the translation unit, and
/// we've emitted a reference to the vtable for this class, should
/// we define that vtable?
static bool shouldEmitVTableAtEndOfTranslationUnit(CIRGenModule &cgm,
                                                   const CXXRecordDecl *rd) {
  // If vtable is internal then it has to be done.
  if (!cgm.getVTables().isVTableExternal(rd))
    return true;

  // If it's external then maybe we will need it as available_externally.
  return shouldEmitAvailableExternallyVTable(cgm, rd);
}

/// Given that at some point we emitted a reference to one or more
/// vtables, and that we are now at the end of the translation unit,
/// decide whether we should emit them.
void CIRGenModule::emitDeferredVTables() {
#ifndef NDEBUG
  // Remember the size of DeferredVTables, because we're going to assume
  // that this entire operation doesn't modify it.
  size_t savedSize = deferredVTables.size();
#endif
  for (const CXXRecordDecl *rd : deferredVTables) {
    if (shouldEmitVTableAtEndOfTranslationUnit(*this, rd))
      vtables.generateClassData(rd);
    else if (shouldOpportunisticallyEmitVTables())
      opportunisticVTables.push_back(rd);
  }

  assert(savedSize == deferredVTables.size() &&
         "deferred extra vtables during vtable emission?");
  deferredVTables.clear();
}

void CIRGenModule::emitVTablesOpportunistically() {
  // Try to emit external vtables as available_externally if they have emitted
  // all inlined virtual functions.  It runs after EmitDeferred() and therefore
  // is not allowed to create new references to things that need to be emitted
  // lazily. Note that it also uses fact that we eagerly emitting RTTI.

  assert(
      (opportunisticVTables.empty() || shouldOpportunisticallyEmitVTables()) &&
      "Only emit opportunistic vtables with optimizations");

  for (const CXXRecordDecl *rd : opportunisticVTables) {
    assert(getVTables().isVTableExternal(rd) &&
           "This queue should only contain external vtables");
    if (getCXXABI().canSpeculativelyEmitVTable(rd))
      vtables.generateClassData(rd);
  }
  opportunisticVTables.clear();
}

bool CIRGenModule::shouldOpportunisticallyEmitVTables() {
  return codeGenOpts.OptimizationLevel > 0;
}
