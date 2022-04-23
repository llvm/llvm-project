//===----- CIRGenItaniumCXXABI.cpp - Emit CIR from ASTs for a Module ------===//
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
#include "CIRGenFunctionInfo.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/TargetInfo.h"

using namespace cir;
using namespace clang;

namespace {
class CIRGenItaniumCXXABI : public cir::CIRGenCXXABI {
protected:
  bool UseARMMethodPtrABI;
  bool UseARMGuardVarABI;
  bool Use32BitVTableOffsetABI;

public:
  CIRGenItaniumCXXABI(CIRGenModule &CGM, bool UseARMMethodPtrABI = false,
                      bool UseARMGuardVarABI = false)
      : CIRGenCXXABI(CGM), UseARMMethodPtrABI{UseARMMethodPtrABI},
        UseARMGuardVarABI{UseARMGuardVarABI}, Use32BitVTableOffsetABI{false} {
    assert(!UseARMMethodPtrABI && "NYI");
    assert(!UseARMGuardVarABI && "NYI");
  }
  AddedStructorArgs getImplicitConstructorArgs(CIRGenFunction &CGF,
                                               const CXXConstructorDecl *D,
                                               CXXCtorType Type,
                                               bool ForVirtualBase,
                                               bool Delegating) override;

  bool NeedsVTTParameter(GlobalDecl GD) override;

  RecordArgABI getRecordArgABI(const clang::CXXRecordDecl *RD) const override {
    // If C++ prohibits us from making a copy, pass by address.
    if (!RD->canPassInRegisters())
      return RecordArgABI::Indirect;
    else
      return RecordArgABI::Default;
  }

  bool classifyReturnType(CIRGenFunctionInfo &FI) const override;

  AddedStructorArgCounts
  buildStructorSignature(GlobalDecl GD,
                         llvm::SmallVectorImpl<CanQualType> &ArgTys) override;

  bool isThisCompleteObject(GlobalDecl GD) const override {
    // The Itanium ABI has separate complete-object vs. base-object variants of
    // both constructors and destructors.
    if (isa<CXXDestructorDecl>(GD.getDecl())) {
      llvm_unreachable("NYI");
    }
    if (isa<CXXConstructorDecl>(GD.getDecl())) {
      switch (GD.getCtorType()) {
      case Ctor_Complete:
        return true;

      case Ctor_Base:
        return false;

      case Ctor_CopyingClosure:
      case Ctor_DefaultClosure:
        llvm_unreachable("closure ctors in Itanium ABI?");

      case Ctor_Comdat:
        llvm_unreachable("emitting ctor comdat as function?");
      }
      llvm_unreachable("bad dtor kind");
    }

    // No other kinds.
    return false;
  }

  void buildInstanceFunctionProlog(CIRGenFunction &CGF) override;

  void addImplicitStructorParams(CIRGenFunction &CGF, QualType &ResTy,
                                 FunctionArgList &Params) override;

  void buildCXXConstructors(const clang::CXXConstructorDecl *D) override;

  void buildCXXStructor(clang::GlobalDecl GD) override;

  bool doStructorsInitializeVPtrs(const CXXRecordDecl *VTableClass) override {
    return true;
  }
};
} // namespace

CIRGenCXXABI::AddedStructorArgs CIRGenItaniumCXXABI::getImplicitConstructorArgs(
    CIRGenFunction &CGF, const CXXConstructorDecl *D, CXXCtorType Type,
    bool ForVirtualBase, bool Delegating) {
  assert(!NeedsVTTParameter(GlobalDecl(D, Type)) && "VTT NYI");

  return {};
}

/// Return whether the given global decl needs a VTT parameter, which it does if
/// it's a base constructor or destructor with virtual bases.
bool CIRGenItaniumCXXABI::NeedsVTTParameter(GlobalDecl GD) {
  auto *MD = cast<CXXMethodDecl>(GD.getDecl());

  // We don't have any virtual bases, just return early.
  if (!MD->getParent()->getNumVBases())
    return false;

  // Check if we have a base constructor.
  if (isa<CXXConstructorDecl>(MD) && GD.getCtorType() == Ctor_Base)
    return true;

  // Check if we have a base destructor.
  if (isa<CXXDestructorDecl>(MD) && GD.getDtorType() == Dtor_Base)
    llvm_unreachable("NYI");

  return false;
}

CIRGenCXXABI *cir::CreateCIRGenItaniumCXXABI(CIRGenModule &CGM) {
  switch (CGM.getASTContext().getCXXABIKind()) {
  case TargetCXXABI::GenericItanium:
    return new CIRGenItaniumCXXABI(CGM);

  default:
    llvm_unreachable("bad or NYI ABI kind");
  }
}

bool CIRGenItaniumCXXABI::classifyReturnType(CIRGenFunctionInfo &FI) const {
  auto *RD = FI.getReturnType()->getAsCXXRecordDecl();
  assert(!RD && "RecordDecl return types NYI");
  return false;
}

CIRGenCXXABI::AddedStructorArgCounts
CIRGenItaniumCXXABI::buildStructorSignature(
    GlobalDecl GD, llvm::SmallVectorImpl<CanQualType> &ArgTys) {
  auto &Context = getContext();

  // All parameters are already in place except VTT, which goes after 'this'.
  // These are clang types, so we don't need to worry about sret yet.

  // Check if we need to add a VTT parameter (which has type void **).
  if ((isa<CXXConstructorDecl>(GD.getDecl()) ? GD.getCtorType() == Ctor_Base
                                             : GD.getDtorType() == Dtor_Base) &&
      cast<CXXMethodDecl>(GD.getDecl())->getParent()->getNumVBases() != 0) {
    llvm_unreachable("NYI");
    (void)Context;
  }

  return AddedStructorArgCounts{};
}

// Find out how to cirgen the complete destructor and constructor
namespace {
enum class StructorCIRGen { Emit, RAUW, Alias, COMDAT };
}

static StructorCIRGen getCIRGenToUse(CIRGenModule &CGM,
                                     const CXXMethodDecl *MD) {
  if (!CGM.getCodeGenOpts().CXXCtorDtorAliases)
    return StructorCIRGen::Emit;

  llvm_unreachable("Nothing else implemented yet");
}

void CIRGenItaniumCXXABI::buildCXXStructor(GlobalDecl GD) {
  auto *MD = cast<CXXMethodDecl>(GD.getDecl());
  auto *CD = dyn_cast<CXXConstructorDecl>(MD);
  const CXXDestructorDecl *DD = CD ? nullptr : cast<CXXDestructorDecl>(MD);

  StructorCIRGen CIRGenType = getCIRGenToUse(CGM, MD);

  if (CD ? GD.getCtorType() == Ctor_Complete
         : GD.getDtorType() == Dtor_Complete) {
    GlobalDecl BaseDecl;
    if (CD)
      BaseDecl = GD.getWithCtorType(Ctor_Base);
    else
      BaseDecl = GD.getWithDtorType(Dtor_Base);

    if (CIRGenType == StructorCIRGen::Alias ||
        CIRGenType == StructorCIRGen::COMDAT) {
      llvm_unreachable("NYI");
    }

    if (CIRGenType == StructorCIRGen::RAUW) {
      llvm_unreachable("NYI");
    }
  }

  // The base destructor is equivalent to the base destructor of its base class
  // if there is exactly one non-virtual base class with a non-trivial
  // destructor, there are no fields with a non-trivial destructor, and the body
  // of the destructor is trivial.
  if (DD && GD.getDtorType() == Dtor_Base &&
      CIRGenType != StructorCIRGen::COMDAT)
    llvm_unreachable("NYI");

  // FIXME: The deleting destructor is equivalent to the selected operator
  // delete if:
  //  * either the delete is a destroying operator delete or the destructor
  //    would be trivial if it weren't virtual.
  //  * the conversion from the 'this' parameter to the first parameter of the
  //    destructor is equivalent to a bitcast,
  //  * the destructor does not have an implicit "this" return, and
  //  * the operator delete has the same calling convention and CIR function
  //    type as the destructor.
  // In such cases we should try to emit the deleting dtor as an alias to the
  // selected 'operator delete'.

  mlir::FuncOp Fn = CGM.codegenCXXStructor(GD);

  if (CIRGenType == StructorCIRGen::COMDAT) {
    llvm_unreachable("NYI");
  } else {
    CGM.maybeSetTrivialComdat(*MD, Fn);
  }
}

void CIRGenItaniumCXXABI::addImplicitStructorParams(CIRGenFunction &CGF,
                                                    QualType &ResTY,
                                                    FunctionArgList &Params) {
  const auto *MD = cast<CXXMethodDecl>(CGF.CurGD.getDecl());
  assert(isa<CXXConstructorDecl>(MD) || isa<CXXDestructorDecl>(MD));

  // Check if we need a VTT parameter as well.
  if (NeedsVTTParameter(CGF.CurGD)) {
    llvm_unreachable("NYI");
  }
}

mlir::Operation *CIRGenCXXABI::loadIncomingCXXThis(CIRGenFunction &CGF) {
  return CGF.createLoad(getThisDecl(CGF), "this");
}

void CIRGenCXXABI::setCXXABIThisValue(CIRGenFunction &CGF,
                                      mlir::Operation *ThisPtr) {
  /// Initialize the 'this' slot.
  assert(getThisDecl(CGF) && "no 'this' variable for function");
  CGF.CXXABIThisValue = ThisPtr;
}

void CIRGenItaniumCXXABI::buildInstanceFunctionProlog(CIRGenFunction &CGF) {
  // Naked functions have no prolog.
  if (CGF.CurFuncDecl && CGF.CurFuncDecl->hasAttr<NakedAttr>())
    llvm_unreachable("NYI");

  /// Initialize the 'this' slot. In the Itanium C++ ABI, no prologue
  /// adjustments are required, because they are all handled by thunks.
  setCXXABIThisValue(CGF, loadIncomingCXXThis(CGF));

  /// Initialize the 'vtt' slot if needed.
  if (getStructorImplicitParamDecl(CGF)) {
    llvm_unreachable("NYI");
  }

  /// If this is a function that the ABI specifies returns 'this', initialize
  /// the return slot to this' at the start of the function.
  ///
  /// Unlike the setting of return types, this is done within the ABI
  /// implementation instead of by clients of CIRGenCXXBI because:
  /// 1) getThisValue is currently protected
  /// 2) in theory, an ABI could implement 'this' returns some other way;
  ///    HasThisReturn only specifies a contract, not the implementation
  if (HasThisReturn(CGF.CurGD))
    llvm_unreachable("NYI");
}

void CIRGenItaniumCXXABI::buildCXXConstructors(const CXXConstructorDecl *D) {
  // Just make sure we're in sync with TargetCXXABI.
  assert(CGM.getTarget().getCXXABI().hasConstructorVariants());

  // The constructor used for constructing this as a base class;
  // ignores virtual bases.
  CGM.buildGlobal(GlobalDecl(D, Ctor_Base));

  // The constructor used for constructing this as a complete class;
  // constructs the virtual bases, then calls the base constructor.
  if (!D->getParent()->isAbstract()) {
    // We don't need to emit the complete ctro if the class is abstract.
    CGM.buildGlobal(GlobalDecl(D, Ctor_Complete));
  }
}
