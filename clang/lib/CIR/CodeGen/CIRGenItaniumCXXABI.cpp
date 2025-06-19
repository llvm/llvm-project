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
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {

class CIRGenItaniumCXXABI : public CIRGenCXXABI {
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

  bool useThunkForDtorVariant(const CXXDestructorDecl *dtor,
                              CXXDtorType dt) const override {
    // Itanium does not emit any destructor variant as an inline thunk.
    // Delegating may occur as an optimization, but all variants are either
    // emitted with external linkage or as linkonce if they are inline and used.
    return false;
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

void CIRGenItaniumCXXABI::emitCXXStructor(GlobalDecl gd) {
  auto *md = cast<CXXMethodDecl>(gd.getDecl());

  if (cgm.getCodeGenOpts().CXXCtorDtorAliases)
    cgm.errorNYI(md->getSourceRange(), "Ctor/Dtor aliases");

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
