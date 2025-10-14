//===----------------------------------------------------------------------===//
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

#include "CIRGenModule.h"
#include "clang/AST/Attr.h"
#include "clang/Basic/LangOptions.h"

using namespace clang;
using namespace clang::CIRGen;

void CIRGenModule::emitCXXGlobalVarDeclInitFunc(const VarDecl *vd,
                                                cir::GlobalOp addr,
                                                bool performInit) {
  assert(!cir::MissingFeatures::cudaSupport());

  assert(!cir::MissingFeatures::deferredCXXGlobalInit());

  emitCXXGlobalVarDeclInit(vd, addr, performInit);
}
