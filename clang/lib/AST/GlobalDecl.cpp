//===--- GlobalDecl.cpp - Global declaration holder -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// GlobalDecl.h is included very widely, so these members are defined here to
// keep the OpenMP and OpenACC declaration hierarchies, the generated attribute
// classes, and the template declaration hierarchy out of it.
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
