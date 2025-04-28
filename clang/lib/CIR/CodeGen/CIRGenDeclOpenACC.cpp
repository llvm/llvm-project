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
#include "clang/AST/DeclOpenACC.h"

using namespace clang;
using namespace clang::CIRGen;

void CIRGenFunction::emitOpenACCDeclare(const OpenACCDeclareDecl &d) {
  getCIRGenModule().errorNYI(d.getSourceRange(), "OpenACC Declare Construct");
}

void CIRGenFunction::emitOpenACCRoutine(const OpenACCRoutineDecl &d) {
  getCIRGenModule().errorNYI(d.getSourceRange(), "OpenACC Routine Construct");
}

void CIRGenModule::emitGlobalOpenACCDecl(const OpenACCConstructDecl *d) {
  if (isa<OpenACCRoutineDecl>(d))
    errorNYI(d->getSourceRange(), "OpenACC Routine Construct");
  else if (isa<OpenACCDeclareDecl>(d))
    errorNYI(d->getSourceRange(), "OpenACC Declare Construct");
  else
    llvm_unreachable("unknown OpenACC declaration kind?");
}
