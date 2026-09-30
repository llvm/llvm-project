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

  // Compute the mangled guard variable name and set the dynamic_init_guard
  // attribute BEFORE emitting initialization. This ensures that GetGlobalOps
  // created during initialization (e.g., in the ctor region) will see the
  // attribute and be marked with static_local accordingly.
  llvm::SmallString<256> guardName;
  {
    llvm::raw_svector_ostream out(guardName);
    cgm.getCXXABI().getMangleContext().mangleStaticGuardVariable(&varDecl, out);
  }

  // Mark the global as requiring guarded dynamic initialization, with the
  // guard name. The emission of the guard/acquire is done during
  // LoweringPrepare.
  auto guardAttr = mlir::StringAttr::get(&cgm.getMLIRContext(), guardName);

  globalOp.setDynamicInitGuardAttr(
      cir::DynamicInitGuardAttr::get(&cgm.getMLIRContext(), guardAttr));

  if (varDecl.isStaticLocal())
    cgm.emitCXXStaticLocalVarDeclInit(&varDecl, globalOp, performInit);
  else
    cgm.emitCXXGlobalVarDeclInit(&varDecl, globalOp, performInit);
}

void CIRGenModule::setGlobalTlsReferences(const VarDecl &vd,
                                          cir::GlobalOp globalOp) {
  assert(!vd.isStaticLocal() && vd.getTLSKind());

  // C doesn't need guarded thread-local init, because it can't have
  // non-constant init.
  if (!getLangOpts().CPlusPlus)
    return;

  // TLS Static doesn't need a wrapper.
  if (vd.getTLSKind() != VarDecl::TLS_Dynamic)
    return;

  llvm::SmallString<256> wrapperFuncName;
  llvm::SmallString<256> initFuncName;
  llvm::SmallString<256> guardName;

  if (getCXXABI().getMangleContext().getKind() == MangleContext::MK_Itanium) {
    llvm::raw_svector_ostream wrapperOut(wrapperFuncName);
    llvm::raw_svector_ostream initOut(initFuncName);
    llvm::raw_svector_ostream guardStream(guardName);

    auto &mc = cast<ItaniumMangleContext>(getCXXABI().getMangleContext());
    mc.mangleItaniumThreadLocalWrapper(&vd, wrapperOut);
    mc.mangleItaniumThreadLocalInit(&vd, initOut);
    if (globalOp.hasWeakLinkage() || globalOp.hasLinkOnceLinkage() ||
        isTemplateInstantiation(vd.getTemplateSpecializationKind())) {
      getCXXABI().getMangleContext().mangleStaticGuardVariable(&vd,
                                                               guardStream);
    }

  } else {
    errorNYI(vd.getSourceRange(),
             "setGlobalTlsReferences: non-itanium mangler");
    return;
  }
  globalOp.setTlsRefsAttr(cir::ThreadLocalGlobalWrapperInitAttr::get(
      &getMLIRContext(), wrapperFuncName, initFuncName, guardName));
}

void CIRGenModule::emitCXXGlobalVarDeclInitFunc(const VarDecl *vd,
                                                cir::GlobalOp addr,
                                                bool performInit) {
  assert(!cir::MissingFeatures::cudaSupport());

  if (addr.hasWeakLinkage() || addr.hasLinkOnceLinkage() ||
      (vd->getTLSKind() == VarDecl::TLS_Dynamic &&
       isTemplateInstantiation(vd->getTemplateSpecializationKind()))) {
    CIRGenFunction(*this, builder).emitCXXGuardedInit(*vd, addr, performInit);
  } else {
    emitCXXGlobalVarDeclInit(vd, addr, performInit);
  }
}
