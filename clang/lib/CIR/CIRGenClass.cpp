//===--- CIRGenClass.cpp - Emit CIR Code for C++ classes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of classes
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"

#include "clang/AST/RecordLayout.h"

using namespace clang;
using namespace cir;

/// Checks whether the given constructor is a valid subject for the
/// complete-to-base constructor delgation optimization, i.e. emitting the
/// complete constructor as a simple call to the base constructor.
bool CIRGenFunction::IsConstructorDelegationValid(
    const CXXConstructorDecl *Ctor) {

  // Currently we disable the optimization for classes with virtual bases
  // because (1) the address of parameter variables need to be consistent across
  // all initializers but (2) the delegate function call necessarily creates a
  // second copy of the parameter variable.
  //
  // The limiting example (purely theoretical AFAIK):
  //   struct A { A(int &c) { c++; } };
  //   struct A : virtual A {
  //     B(int count) : A(count) { printf("%d\n", count); }
  //   };
  // ...although even this example could in principle be emitted as a delegation
  // since the address of the parameter doesn't escape.
  if (Ctor->getParent()->getNumVBases()) {
    llvm_unreachable("NYI");
  }

  // We also disable the optimization for variadic functions because it's
  // impossible to "re-pass" varargs.
  if (Ctor->getType()->castAs<FunctionProtoType>()->isVariadic())
    return false;

  // FIXME: Decide if we can do a delegation of a delegating constructor.
  if (Ctor->isDelegatingConstructor())
    llvm_unreachable("NYI");

  return true;
}

void CIRGenFunction::initializeVTablePointers(const CXXRecordDecl *RD) {
  // Ignore classes without a vtable.
  if (!RD->isDynamicClass())
    return;

  // Initialize the vtable pointers for this class and all of its bases.
  if (CGM.getCXXABI().doStructorsInitializeVPtrs(RD))
    for (const auto &Vptr : getVTablePointers(RD)) {
      llvm_unreachable("NYI");
      (void)Vptr;
    }

  if (RD->getNumVBases())
    llvm_unreachable("NYI");
}

CIRGenFunction::VPtrsVector
CIRGenFunction::getVTablePointers(const CXXRecordDecl *VTableClass) {
  CIRGenFunction::VPtrsVector VPtrsResult;
  VisitedVirtualBasesSetTy VBases;
  getVTablePointers(BaseSubobject(VTableClass, CharUnits::Zero()),
                    /*NearestVBase=*/nullptr,
                    /*OffsetFromNearestVBase=*/CharUnits::Zero(),
                    /*BaseIsNonVirtualPrimaryBase=*/false, VTableClass, VBases,
                    VPtrsResult);
  return VPtrsResult;
}

void CIRGenFunction::getVTablePointers(BaseSubobject Base,
                                       const CXXRecordDecl *NearestVBase,
                                       CharUnits OffsetFromNearestVBase,
                                       bool BaseIsNonVirtualPrimaryBase,
                                       const CXXRecordDecl *VTableClass,
                                       VisitedVirtualBasesSetTy &VBases,
                                       VPtrsVector &Vptrs) {
  // If this base is a non-virtual primary base the address point has already
  // been set.
  if (!BaseIsNonVirtualPrimaryBase) {
    // Initialize the vtable pointer for this base.
    VPtr Vptr = {Base, NearestVBase, OffsetFromNearestVBase, VTableClass};
    Vptrs.push_back(Vptr);
  }

  const CXXRecordDecl *RD = Base.getBase();

  // Traverse bases.
  for (const auto &I : RD->bases()) {
    (void)I;
    llvm_unreachable("NYI");
  }
}

Address CIRGenFunction::LoadCXXThisAddress() {
  assert(CurFuncDecl && "loading 'this' without a func declaration?");
  assert(isa<CXXMethodDecl>(CurFuncDecl));

  // Lazily compute CXXThisAlignment.
  if (CXXThisAlignment.isZero()) {
    // Just use the best known alignment for the parent.
    // TODO: if we're currently emitting a complete-object ctor/dtor, we can
    // always use the complete-object alignment.
    auto RD = cast<CXXMethodDecl>(CurFuncDecl)->getParent();
    CXXThisAlignment = CGM.getClassPointerAlignment(RD);
  }

  // Consider how to do this if we ever have multiple returns
  auto Result = LoadCXXThis()->getOpResult(0);
  return Address(Result, CXXThisAlignment);
}

void CIRGenFunction::buildDelegateCXXConstructorCall(
    const CXXConstructorDecl *Ctor, CXXCtorType CtorType,
    const FunctionArgList &Args, SourceLocation Loc) {
  CallArgList DelegateArgs;

  FunctionArgList::const_iterator I = Args.begin(), E = Args.end();
  assert(I != E && "no parameters to constructor");

  // this
  Address This = LoadCXXThisAddress();
  DelegateArgs.add(RValue::get(This.getPointer()), (*I)->getType());
  ++I;

  // FIXME: The location of the VTT parameter in the parameter list is specific
  // to the Itanium ABI and shouldn't be hardcoded here.
  if (CGM.getCXXABI().NeedsVTTParameter(CurGD)) {
    llvm_unreachable("NYI");
  }

  // Explicit arguments.
  for (; I != E; ++I) {
    const VarDecl *param = *I;
    // FIXME: per-argument source location
    buildDelegateCallArg(DelegateArgs, param, Loc);
  }

  buildCXXConstructorCall(Ctor, CtorType, /*ForVirtualBase=*/false,
                          /*Delegating=*/true, This, DelegateArgs,
                          AggValueSlot::MayOverlap, Loc,
                          /*NewPointerIsChecked=*/true);
}
