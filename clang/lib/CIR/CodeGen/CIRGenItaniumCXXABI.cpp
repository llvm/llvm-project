//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides C++ code generation targeting the Itanium C++ ABI.  The class
// in this file generates structures that follow the Itanium C++ ABI, which is
// documented at:
//  https://itanium-cxx-abi.github.io/cxx-abi/abi.html
//  https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html
//
// It also supports the closely-related ARM ABI, documented at:
// https://developer.arm.com/documentation/ihi0041/g/
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"

#include "clang/AST/ExprCXX.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/VTableBuilder.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {

class CIRGenItaniumCXXABI : public CIRGenCXXABI {
protected:
  /// All the vtables which have been defined.
  llvm::DenseMap<const CXXRecordDecl *, cir::GlobalOp> vtables;

public:
  CIRGenItaniumCXXABI(CIRGenModule &cgm) : CIRGenCXXABI(cgm) {
    assert(!cir::MissingFeatures::cxxabiUseARMMethodPtrABI());
    assert(!cir::MissingFeatures::cxxabiUseARMGuardVarABI());
  }

  bool needsVTTParameter(clang::GlobalDecl gd) override;

  void emitInstanceFunctionProlog(SourceLocation loc,
                                  CIRGenFunction &cgf) override;

  void emitCXXConstructors(const clang::CXXConstructorDecl *d) override;
  void emitCXXDestructors(const clang::CXXDestructorDecl *d) override;
  void emitCXXStructor(clang::GlobalDecl gd) override;

  void emitDestructorCall(CIRGenFunction &cgf, const CXXDestructorDecl *dd,
                          CXXDtorType type, bool forVirtualBase,
                          bool delegating, Address thisAddr,
                          QualType thisTy) override;

  bool useThunkForDtorVariant(const CXXDestructorDecl *dtor,
                              CXXDtorType dt) const override {
    // Itanium does not emit any destructor variant as an inline thunk.
    // Delegating may occur as an optimization, but all variants are either
    // emitted with external linkage or as linkonce if they are inline and used.
    return false;
  }

  bool isVirtualOffsetNeededForVTableField(CIRGenFunction &cgf,
                                           CIRGenFunction::VPtr vptr) override;

  cir::GlobalOp getAddrOfVTable(const CXXRecordDecl *rd,
                                CharUnits vptrOffset) override;

  mlir::Value getVTableAddressPoint(BaseSubobject base,
                                    const CXXRecordDecl *vtableClass) override;

  mlir::Value getVTableAddressPointInStructor(
      CIRGenFunction &cgf, const clang::CXXRecordDecl *vtableClass,
      clang::BaseSubobject base,
      const clang::CXXRecordDecl *nearestVBase) override;

  bool doStructorsInitializeVPtrs(const CXXRecordDecl *vtableClass) override {
    return true;
  }
};

} // namespace

void CIRGenItaniumCXXABI::emitInstanceFunctionProlog(SourceLocation loc,
                                                     CIRGenFunction &cgf) {
  // Naked functions have no prolog.
  if (cgf.curFuncDecl && cgf.curFuncDecl->hasAttr<NakedAttr>()) {
    cgf.cgm.errorNYI(cgf.curFuncDecl->getLocation(),
                     "emitInstanceFunctionProlog: Naked");
  }

  /// Initialize the 'this' slot. In the Itanium C++ ABI, no prologue
  /// adjustments are required, because they are all handled by thunks.
  setCXXABIThisValue(cgf, loadIncomingCXXThis(cgf));

  /// Classic codegen has code here to initialize the 'vtt' slot if
  // getStructorImplicitParamDecl(cgf) returns a non-null value, but in the
  // current implementation (of classic codegen) it never does.
  assert(!cir::MissingFeatures::cxxabiStructorImplicitParam());

  /// If this is a function that the ABI specifies returns 'this', initialize
  /// the return slot to this' at the start of the function.
  ///
  /// Unlike the setting of return types, this is done within the ABI
  /// implementation instead of by clients of CIRGenCXXBI because:
  /// 1) getThisValue is currently protected
  /// 2) in theory, an ABI could implement 'this' returns some other way;
  ///    HasThisReturn only specifies a contract, not the implementation
  if (hasThisReturn(cgf.curGD)) {
    cgf.cgm.errorNYI(cgf.curFuncDecl->getLocation(),
                     "emitInstanceFunctionProlog: hasThisReturn");
  }
}

// Find out how to cirgen the complete destructor and constructor
namespace {
enum class StructorCIRGen { Emit, RAUW, Alias, COMDAT };
}

static StructorCIRGen getCIRGenToUse(CIRGenModule &cgm,
                                     const CXXMethodDecl *md) {
  if (!cgm.getCodeGenOpts().CXXCtorDtorAliases)
    return StructorCIRGen::Emit;

  // The complete and base structors are not equivalent if there are any virtual
  // bases, so emit separate functions.
  if (md->getParent()->getNumVBases()) {
    // The return value is correct here, but other support for this is NYI.
    cgm.errorNYI(md->getSourceRange(), "getCIRGenToUse: virtual bases");
    return StructorCIRGen::Emit;
  }

  GlobalDecl aliasDecl;
  if (const auto *dd = dyn_cast<CXXDestructorDecl>(md)) {
    aliasDecl = GlobalDecl(dd, Dtor_Complete);
  } else {
    const auto *cd = cast<CXXConstructorDecl>(md);
    aliasDecl = GlobalDecl(cd, Ctor_Complete);
  }

  cir::GlobalLinkageKind linkage = cgm.getFunctionLinkage(aliasDecl);

  if (cir::isDiscardableIfUnused(linkage))
    return StructorCIRGen::RAUW;

  // FIXME: Should we allow available_externally aliases?
  if (!cir::isValidLinkage(linkage))
    return StructorCIRGen::RAUW;

  if (cir::isWeakForLinker(linkage)) {
    // Only ELF and wasm support COMDATs with arbitrary names (C5/D5).
    if (cgm.getTarget().getTriple().isOSBinFormatELF() ||
        cgm.getTarget().getTriple().isOSBinFormatWasm())
      return StructorCIRGen::COMDAT;
    return StructorCIRGen::Emit;
  }

  return StructorCIRGen::Alias;
}

static void emitConstructorDestructorAlias(CIRGenModule &cgm,
                                           GlobalDecl aliasDecl,
                                           GlobalDecl targetDecl) {
  cir::GlobalLinkageKind linkage = cgm.getFunctionLinkage(aliasDecl);

  // Does this function alias already exists?
  StringRef mangledName = cgm.getMangledName(aliasDecl);
  auto globalValue = dyn_cast_or_null<cir::CIRGlobalValueInterface>(
      cgm.getGlobalValue(mangledName));
  if (globalValue && !globalValue.isDeclaration())
    return;

  auto entry = cast_or_null<cir::FuncOp>(cgm.getGlobalValue(mangledName));

  // Retrieve aliasee info.
  auto aliasee = cast<cir::FuncOp>(cgm.getAddrOfGlobal(targetDecl));

  // Populate actual alias.
  cgm.emitAliasForGlobal(mangledName, entry, aliasDecl, aliasee, linkage);
}

void CIRGenItaniumCXXABI::emitCXXStructor(GlobalDecl gd) {
  auto *md = cast<CXXMethodDecl>(gd.getDecl());
  StructorCIRGen cirGenType = getCIRGenToUse(cgm, md);
  const auto *cd = dyn_cast<CXXConstructorDecl>(md);

  if (cd ? gd.getCtorType() == Ctor_Complete
         : gd.getDtorType() == Dtor_Complete) {
    GlobalDecl baseDecl =
        cd ? gd.getWithCtorType(Ctor_Base) : gd.getWithDtorType(Dtor_Base);
    ;

    if (cirGenType == StructorCIRGen::Alias ||
        cirGenType == StructorCIRGen::COMDAT) {
      emitConstructorDestructorAlias(cgm, gd, baseDecl);
      return;
    }

    if (cirGenType == StructorCIRGen::RAUW) {
      StringRef mangledName = cgm.getMangledName(gd);
      mlir::Operation *aliasee = cgm.getAddrOfGlobal(baseDecl);
      cgm.addReplacement(mangledName, aliasee);
      return;
    }
  }

  auto fn = cgm.codegenCXXStructor(gd);

  cgm.maybeSetTrivialComdat(*md, fn);
}

void CIRGenItaniumCXXABI::emitCXXConstructors(const CXXConstructorDecl *d) {
  // Just make sure we're in sync with TargetCXXABI.
  assert(cgm.getTarget().getCXXABI().hasConstructorVariants());

  // The constructor used for constructing this as a base class;
  // ignores virtual bases.
  cgm.emitGlobal(GlobalDecl(d, Ctor_Base));

  // The constructor used for constructing this as a complete class;
  // constructs the virtual bases, then calls the base constructor.
  if (!d->getParent()->isAbstract()) {
    // We don't need to emit the complete ctro if the class is abstract.
    cgm.emitGlobal(GlobalDecl(d, Ctor_Complete));
  }
}

void CIRGenItaniumCXXABI::emitCXXDestructors(const CXXDestructorDecl *d) {
  // The destructor used for destructing this as a base class; ignores
  // virtual bases.
  cgm.emitGlobal(GlobalDecl(d, Dtor_Base));

  // The destructor used for destructing this as a most-derived class;
  // call the base destructor and then destructs any virtual bases.
  cgm.emitGlobal(GlobalDecl(d, Dtor_Complete));

  // The destructor in a virtual table is always a 'deleting'
  // destructor, which calls the complete destructor and then uses the
  // appropriate operator delete.
  if (d->isVirtual())
    cgm.emitGlobal(GlobalDecl(d, Dtor_Deleting));
}

/// Return whether the given global decl needs a VTT (virtual table table)
/// parameter, which it does if it's a base constructor or destructor with
/// virtual bases.
bool CIRGenItaniumCXXABI::needsVTTParameter(GlobalDecl gd) {
  auto *md = cast<CXXMethodDecl>(gd.getDecl());

  // We don't have any virtual bases, just return early.
  if (!md->getParent()->getNumVBases())
    return false;

  // Check if we have a base constructor.
  if (isa<CXXConstructorDecl>(md) && gd.getCtorType() == Ctor_Base)
    return true;

  // Check if we have a base destructor.
  if (isa<CXXDestructorDecl>(md) && gd.getDtorType() == Dtor_Base)
    return true;

  return false;
}

void CIRGenItaniumCXXABI::emitDestructorCall(
    CIRGenFunction &cgf, const CXXDestructorDecl *dd, CXXDtorType type,
    bool forVirtualBase, bool delegating, Address thisAddr, QualType thisTy) {
  GlobalDecl gd(dd, type);
  if (needsVTTParameter(gd)) {
    cgm.errorNYI(dd->getSourceRange(), "emitDestructorCall: VTT");
  }

  mlir::Value vtt = nullptr;
  ASTContext &astContext = cgm.getASTContext();
  QualType vttTy = astContext.getPointerType(astContext.VoidPtrTy);
  assert(!cir::MissingFeatures::appleKext());
  CIRGenCallee callee =
      CIRGenCallee::forDirect(cgm.getAddrOfCXXStructor(gd), gd);

  cgf.emitCXXDestructorCall(gd, callee, thisAddr.getPointer(), thisTy, vtt,
                            vttTy, nullptr);
}

CIRGenCXXABI *clang::CIRGen::CreateCIRGenItaniumCXXABI(CIRGenModule &cgm) {
  switch (cgm.getASTContext().getCXXABIKind()) {
  case TargetCXXABI::GenericItanium:
  case TargetCXXABI::GenericAArch64:
    return new CIRGenItaniumCXXABI(cgm);

  case TargetCXXABI::AppleARM64:
    // The general Itanium ABI will do until we implement something that
    // requires special handling.
    assert(!cir::MissingFeatures::cxxabiAppleARM64CXXABI());
    return new CIRGenItaniumCXXABI(cgm);

  default:
    llvm_unreachable("bad or NYI ABI kind");
  }
}

cir::GlobalOp CIRGenItaniumCXXABI::getAddrOfVTable(const CXXRecordDecl *rd,
                                                   CharUnits vptrOffset) {
  assert(vptrOffset.isZero() && "Itanium ABI only supports zero vptr offsets");
  cir::GlobalOp &vtable = vtables[rd];
  if (vtable)
    return vtable;

  // Queue up this vtable for possible deferred emission.
  assert(!cir::MissingFeatures::deferredVtables());

  SmallString<256> name;
  llvm::raw_svector_ostream out(name);
  getMangleContext().mangleCXXVTable(rd, out);

  const VTableLayout &vtLayout =
      cgm.getItaniumVTableContext().getVTableLayout(rd);
  mlir::Type vtableType = cgm.getVTables().getVTableType(vtLayout);

  // Use pointer alignment for the vtable. Otherwise we would align them based
  // on the size of the initializer which doesn't make sense as only single
  // values are read.
  unsigned ptrAlign = cgm.getItaniumVTableContext().isRelativeLayout()
                          ? 32
                          : cgm.getTarget().getPointerAlign(LangAS::Default);

  vtable = cgm.createOrReplaceCXXRuntimeVariable(
      cgm.getLoc(rd->getSourceRange()), name, vtableType,
      cir::GlobalLinkageKind::ExternalLinkage,
      cgm.getASTContext().toCharUnitsFromBits(ptrAlign));
  // LLVM codegen handles unnamedAddr
  assert(!cir::MissingFeatures::opGlobalUnnamedAddr());

  // In MS C++ if you have a class with virtual functions in which you are using
  // selective member import/export, then all virtual functions must be exported
  // unless they are inline, otherwise a link error will result. To match this
  // behavior, for such classes, we dllimport the vtable if it is defined
  // externally and all the non-inline virtual methods are marked dllimport, and
  // we dllexport the vtable if it is defined in this TU and all the non-inline
  // virtual methods are marked dllexport.
  if (cgm.getTarget().hasPS4DLLImportExport())
    cgm.errorNYI(rd->getSourceRange(),
                 "getAddrOfVTable: PS4 DLL import/export");

  cgm.setGVProperties(vtable, rd);
  return vtable;
}

mlir::Value
CIRGenItaniumCXXABI::getVTableAddressPoint(BaseSubobject base,
                                           const CXXRecordDecl *vtableClass) {
  cir::GlobalOp vtable = getAddrOfVTable(vtableClass, CharUnits());

  // Find the appropriate vtable within the vtable group, and the address point
  // within that vtable.
  VTableLayout::AddressPointLocation addressPoint =
      cgm.getItaniumVTableContext()
          .getVTableLayout(vtableClass)
          .getAddressPoint(base);

  mlir::OpBuilder &builder = cgm.getBuilder();
  auto vtablePtrTy = cir::VPtrType::get(builder.getContext());

  return builder.create<cir::VTableAddrPointOp>(
      cgm.getLoc(vtableClass->getSourceRange()), vtablePtrTy,
      mlir::FlatSymbolRefAttr::get(vtable.getSymNameAttr()),
      cir::AddressPointAttr::get(cgm.getBuilder().getContext(),
                                 addressPoint.VTableIndex,
                                 addressPoint.AddressPointIndex));
}

mlir::Value CIRGenItaniumCXXABI::getVTableAddressPointInStructor(
    CIRGenFunction &cgf, const clang::CXXRecordDecl *vtableClass,
    clang::BaseSubobject base, const clang::CXXRecordDecl *nearestVBase) {

  if (base.getBase()->getNumVBases()) {
    cgm.errorNYI(cgf.curFuncDecl->getLocation(),
                 "getVTableAddressPointInStructor: virtual base");
  }
  return getVTableAddressPoint(base, vtableClass);
}

bool CIRGenItaniumCXXABI::isVirtualOffsetNeededForVTableField(
    CIRGenFunction &cgf, CIRGenFunction::VPtr vptr) {
  if (vptr.nearestVBase == nullptr)
    return false;
  return needsVTTParameter(cgf.curGD);
}
