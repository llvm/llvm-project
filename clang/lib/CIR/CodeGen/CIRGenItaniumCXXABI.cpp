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
  void emitCXXStructor(clang::GlobalDecl gd) override;
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
    // The assignment is correct here, but other support for this is NYI.
    cgm.errorNYI(md->getSourceRange(), "getCIRGenToUse: dtor");
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
  auto *cd = dyn_cast<CXXConstructorDecl>(md);

  StructorCIRGen cirGenType = getCIRGenToUse(cgm, md);

  if (!cd) {
    cgm.errorNYI(md->getSourceRange(), "CXCABI emit destructor");
    return;
  }

  if (gd.getCtorType() == Ctor_Complete) {
    GlobalDecl baseDecl = gd.getWithCtorType(Ctor_Base);

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
