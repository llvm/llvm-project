//===--- GlobalDecl.cpp - Global declaration holder -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Out-of-line members of GlobalDecl. GlobalDecl is a small key type that is
// included very widely, so everything here is kept out of the header to spare
// its users the OpenMP and OpenACC declaration hierarchies, the generated
// attribute classes, and the template declaration hierarchy.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclOpenACC.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/DeclTemplate.h"

using namespace clang;

GlobalDecl::GlobalDecl(const OMPDeclareReductionDecl *D) { Init(D); }
GlobalDecl::GlobalDecl(const OMPDeclareMapperDecl *D) { Init(D); }
GlobalDecl::GlobalDecl(const OpenACCRoutineDecl *D) { Init(D); }
GlobalDecl::GlobalDecl(const OpenACCDeclareDecl *D) { Init(D); }

bool GlobalDecl::hasCUDAGlobalAttr(const Decl *D) {
  return D->hasAttr<CUDAGlobalAttr>();
}

bool GlobalDecl::isKernelReference(const Decl *D) {
  if (const auto *FD = dyn_cast<FunctionDecl>(D))
    return FD->isReferenceableKernel();
  if (const auto *FTD = dyn_cast<FunctionTemplateDecl>(D))
    return FTD->getTemplatedDecl()->hasAttr<CUDAGlobalAttr>();
  return false;
}

KernelReferenceKind
GlobalDecl::getDefaultKernelReference(const FunctionDecl *D) {
  return (D->hasAttr<DeviceKernelAttr>() || D->getLangOpts().CUDAIsDevice)
             ? KernelReferenceKind::Kernel
             : KernelReferenceKind::Stub;
}
