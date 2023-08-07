//===--- CIRGenDeclCXX.cpp - Build CIR Code for C++ declarations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with code generation of C++ declarations
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/Attr.h"
#include "clang/Basic/LangOptions.h"

using namespace clang;
using namespace mlir::cir;
using namespace cir;

void CIRGenModule::buildCXXGlobalInitFunc() {
  while (!CXXGlobalInits.empty() && !CXXGlobalInits.back())
    CXXGlobalInits.pop_back();

  if (CXXGlobalInits.empty()) // TODO(cir): &&
                              // PrioritizedCXXGlobalInits.empty())
    return;

  assert(0 && "NYE");
}

void CIRGenModule::buildGlobalVarDeclInit(const VarDecl *D,
                                          mlir::cir::GlobalOp Addr,
                                          bool PerformInit) {
  // According to E.2.3.1 in CUDA-7.5 Programming guide: __device__,
  // __constant__ and __shared__ variables defined in namespace scope,
  // that are of class type, cannot have a non-empty constructor. All
  // the checks have been done in Sema by now. Whatever initializers
  // are allowed are empty and we just need to ignore them here.
  if (getLangOpts().CUDAIsDevice && !getLangOpts().GPUAllowDeviceInit &&
      (D->hasAttr<CUDADeviceAttr>() || D->hasAttr<CUDAConstantAttr>() ||
       D->hasAttr<CUDASharedAttr>()))
    return;

  assert(!getLangOpts().OpenMP && "OpenMP global var init not implemented");

  // Check if we've already initialized this decl.
  auto I = DelayedCXXInitPosition.find(D);
  if (I != DelayedCXXInitPosition.end() && I->second == ~0U)
    return;

  if (PerformInit) {
    QualType T = D->getType();

    // TODO: handle address space
    // The address space of a static local variable (DeclPtr) may be different
    // from the address space of the "this" argument of the constructor. In that
    // case, we need an addrspacecast before calling the constructor.
    //
    // struct StructWithCtor {
    //   __device__ StructWithCtor() {...}
    // };
    // __device__ void foo() {
    //   __shared__ StructWithCtor s;
    //   ...
    // }
    //
    // For example, in the above CUDA code, the static local variable s has a
    // "shared" address space qualifier, but the constructor of StructWithCtor
    // expects "this" in the "generic" address space.
    assert(!UnimplementedFeature::addressSpace());

    if (!T->isReferenceType()) {
      codegenGlobalInitCxxStructor(D, Addr);
      return;
    }
  }
}
