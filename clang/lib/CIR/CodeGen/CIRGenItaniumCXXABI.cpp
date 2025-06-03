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

#include "clang/AST/GlobalDecl.h"
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

  void emitInstanceFunctionProlog(SourceLocation Loc,
                                  CIRGenFunction &CGF) override;
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
