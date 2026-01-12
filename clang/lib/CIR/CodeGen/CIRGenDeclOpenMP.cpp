//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Decl nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "clang/AST/DeclOpenMP.h"

using namespace clang;
using namespace clang::CIRGen;

void CIRGenModule::emitOMPThreadPrivateDecl(const OMPThreadPrivateDecl *d) {
  // TODO(OpenMP): We don't properly differentiate between 'emitDecl' and
  // 'emitGlobal' and 'emitTopLevelDecl' in CIRGenDecl.cpp/CIRGenModule.cpp, so
  // if this decl requires we differentiate those, we probably need to split
  // this function into multiples.
  errorNYI(d->getSourceRange(), "OpenMP OMPThreadPrivateDecl");
}

void CIRGenFunction::emitOMPThreadPrivateDecl(const OMPThreadPrivateDecl &d) {
  // TODO(OpenMP): We don't properly differentiate between 'emitDecl' and
  // 'emitGlobal' and 'emitTopLevelDecl' in CIRGenDecl.cpp/CIRGenModule.cpp, so
  // if this decl requires we differentiate those, we probably need to split
  // this function into multiples.
  getCIRGenModule().errorNYI(d.getSourceRange(), "OpenMP OMPThreadPrivateDecl");
}

void CIRGenModule::emitOMPGroupPrivateDecl(const OMPGroupPrivateDecl *d) {
  // TODO(OpenMP): We don't properly differentiate between 'emitDecl' and
  // 'emitGlobal' and 'emitTopLevelDecl' in CIRGenDecl.cpp/CIRGenModule.cpp, so
  // if this decl requires we differentiate those, we probably need to split
  // this function into multiples.
  errorNYI(d->getSourceRange(), "OpenMP OMPGroupPrivateDecl");
}

void CIRGenFunction::emitOMPGroupPrivateDecl(const OMPGroupPrivateDecl &d) {
  // TODO(OpenMP): We don't properly differentiate between 'emitDecl' and
  // 'emitGlobal' and 'emitTopLevelDecl' in CIRGenDecl.cpp/CIRGenModule.cpp, so
  // if this decl requires we differentiate those, we probably need to split
  // this function into multiples.
  getCIRGenModule().errorNYI(d.getSourceRange(), "OpenMP OMPGroupPrivateDecl");
}

void CIRGenModule::emitOMPCapturedExpr(const OMPCapturedExprDecl *d) {
  // TODO(OpenMP): We don't properly differentiate between 'emitDecl' and
  // 'emitGlobal' and 'emitTopLevelDecl' in CIRGenDecl.cpp/CIRGenModule.cpp, so
  // if this decl requires we differentiate those, we probably need to split
  // this function into multiples.
  errorNYI(d->getSourceRange(), "OpenMP OMPCapturedExpr");
}

void CIRGenFunction::emitOMPCapturedExpr(const OMPCapturedExprDecl &d) {
  // TODO(OpenMP): We don't properly differentiate between 'emitDecl' and
  // 'emitGlobal' and 'emitTopLevelDecl' in CIRGenDecl.cpp/CIRGenModule.cpp, so
  // if this decl requires we differentiate those, we probably need to split
  // this function into multiples.
  getCIRGenModule().errorNYI(d.getSourceRange(), "OpenMP OMPCapturedExpr");
}

void CIRGenModule::emitOMPAllocateDecl(const OMPAllocateDecl *d) {
  // TODO(OpenMP): We don't properly differentiate between 'emitDecl' and
  // 'emitGlobal' and 'emitTopLevelDecl' in CIRGenDecl.cpp/CIRGenModule.cpp, so
  // if this decl requires we differentiate those, we probably need to split
  // this function into multiples.
  errorNYI(d->getSourceRange(), "OpenMP OMPAllocateDecl");
}

void CIRGenFunction::emitOMPAllocateDecl(const OMPAllocateDecl &d) {
  // TODO(OpenMP): We don't properly differentiate between 'emitDecl' and
  // 'emitGlobal' and 'emitTopLevelDecl' in CIRGenDecl.cpp/CIRGenModule.cpp, so
  // if this decl requires we differentiate those, we probably need to split
  // this function into multiples.
  getCIRGenModule().errorNYI(d.getSourceRange(), "OpenMP OMPAllocateDecl");
}

void CIRGenModule::emitOMPDeclareReduction(const OMPDeclareReductionDecl *d) {
  // TODO(OpenMP): We don't properly differentiate between 'emitDecl' and
  // 'emitGlobal' and 'emitTopLevelDecl' in CIRGenDecl.cpp/CIRGenModule.cpp, so
  // if this decl requires we differentiate those, we probably need to split
  // this function into multiples.
  errorNYI(d->getSourceRange(), "OpenMP OMPDeclareReduction");
}

void CIRGenFunction::emitOMPDeclareReduction(const OMPDeclareReductionDecl &d) {
  // TODO(OpenMP): We don't properly differentiate between 'emitDecl' and
  // 'emitGlobal' and 'emitTopLevelDecl' in CIRGenDecl.cpp/CIRGenModule.cpp, so
  // if this decl requires we differentiate those, we probably need to split
  // this function into multiples.
  getCIRGenModule().errorNYI(d.getSourceRange(), "OpenMP OMPDeclareReduction");
}

void CIRGenModule::emitOMPDeclareMapper(const OMPDeclareMapperDecl *d) {
  // TODO(OpenMP): We don't properly differentiate between 'emitDecl' and
  // 'emitGlobal' and 'emitTopLevelDecl' in CIRGenDecl.cpp/CIRGenModule.cpp, so
  // if this decl requires we differentiate those, we probably need to split
  // this function into multiples.
  errorNYI(d->getSourceRange(), "OpenMP OMPDeclareMapper");
}

void CIRGenFunction::emitOMPDeclareMapper(const OMPDeclareMapperDecl &d) {
  // TODO(OpenMP): We don't properly differentiate between 'emitDecl' and
  // 'emitGlobal' and 'emitTopLevelDecl' in CIRGenDecl.cpp/CIRGenModule.cpp, so
  // if this decl requires we differentiate those, we probably need to split
  // this function into multiples.
  getCIRGenModule().errorNYI(d.getSourceRange(), "OpenMP OMPDeclareMapper");
}

void CIRGenModule::emitOMPRequiresDecl(const OMPRequiresDecl *d) {
  // TODO(OpenMP): We don't properly differentiate between 'emitDecl' and
  // 'emitGlobal' and 'emitTopLevelDecl' in CIRGenDecl.cpp/CIRGenModule.cpp, so
  // if this decl requires we differentiate those, we probably need to split
  // this function into multiples.
  errorNYI(d->getSourceRange(), "OpenMP OMPRequiresDecl");
}

void CIRGenFunction::emitOMPRequiresDecl(const OMPRequiresDecl &d) {
  // TODO(OpenMP): We don't properly differentiate between 'emitDecl' and
  // 'emitGlobal' and 'emitTopLevelDecl' in CIRGenDecl.cpp/CIRGenModule.cpp, so
  // if this decl requires we differentiate those, we probably need to split
  // this function into multiples.
  getCIRGenModule().errorNYI(d.getSourceRange(), "OpenMP OMPRequiresDecl");
}
