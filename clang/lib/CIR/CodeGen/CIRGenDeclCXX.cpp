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

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/LangOptions.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"

using namespace clang;
using namespace clang::CIRGen;

void CIRGenFunction::emitCXXGuardedInit(const VarDecl &varDecl,
                                        cir::GlobalOp globalOp,
                                        bool performInit) {
  // If we've been asked to forbid guard variables, emit an error now.
  // This diagnostic is hard-coded for Darwin's use case; we can find
  // better phrasing if someone else needs it.
  if (cgm.getCodeGenOpts().ForbidGuardVariables)
    cgm.error(varDecl.getLocation(), "guard variables are forbidden");

  // Compute the mangled guard variable name and set the static_local attribute
  // BEFORE emitting initialization. This ensures that GetGlobalOps created
  // during initialization (e.g., in the ctor region) will see the attribute
  // and be marked with static_local accordingly.
  llvm::SmallString<256> guardName;
  {
    llvm::raw_svector_ostream out(guardName);
    cgm.getCXXABI().getMangleContext().mangleStaticGuardVariable(&varDecl, out);
  }

  // Mark the global as static local with the guard name. The emission of the
  // guard/acquire is done during LoweringPrepare.
  auto guardAttr = mlir::StringAttr::get(&cgm.getMLIRContext(), guardName);
  globalOp.setStaticLocalGuardAttr(
      cir::StaticLocalGuardAttr::get(&cgm.getMLIRContext(), guardAttr));

  // Emit the initializer and add a global destructor if appropriate.
  cgm.emitCXXGlobalVarDeclInit(&varDecl, globalOp, performInit);
}

void CIRGenModule::emitCXXGlobalVarDeclInitFunc(const VarDecl *vd,
                                                cir::GlobalOp addr,
                                                bool performInit) {
  assert(!cir::MissingFeatures::cudaSupport());

  assert(!cir::MissingFeatures::deferredCXXGlobalInit());

  emitCXXGlobalVarDeclInit(vd, addr, performInit);
}
