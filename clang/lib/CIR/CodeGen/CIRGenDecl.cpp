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
#include "clang/AST/Expr.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

/// Emit code and set up symbol table for a variable declaration with auto,
/// register, or no storage class specifier. These turn into simple stack
/// objects, globals depending on target.
void CIRGenFunction::emitAutoVarDecl(const VarDecl &d) {
  QualType ty = d.getType();
  assert(ty.getAddressSpace() == LangAS::Default);

  auto loc = getLoc(d.getSourceRange());

  if (d.isEscapingByref())
    cgm.errorNYI(d.getSourceRange(),
                 "emitAutoVarDecl: decl escaping by reference");

  CharUnits alignment = getContext().getDeclAlign(&d);

  // If the type is variably-modified, emit all the VLA sizes for it.
  if (ty->isVariablyModifiedType())
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarDecl: variably modified type");

  Address address = Address::invalid();
  if (!ty->isConstantSizeType())
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarDecl: non-constant size type");

  // A normal fixed sized variable becomes an alloca in the entry block,
  mlir::Type allocaTy = convertTypeForMem(ty);
  // Create the temp alloca and declare variable using it.
  mlir::Value addrVal;
  address = createTempAlloca(allocaTy, alignment, loc, d.getName(),
                             /*ArraySize=*/nullptr);
  setAddrOfLocalVar(&d, address);
  // TODO: emit var init and cleanup
}

void CIRGenFunction::emitVarDecl(const VarDecl &d) {
  if (d.hasExternalStorage()) {
    // Don't emit it now, allow it to be emitted lazily on its first use.
    return;
  }

  if (d.getStorageDuration() != SD_Automatic)
    cgm.errorNYI(d.getSourceRange(), "emitVarDecl automatic storage duration");
  if (d.getType().getAddressSpace() == LangAS::opencl_local)
    cgm.errorNYI(d.getSourceRange(), "emitVarDecl openCL address space");

  assert(d.hasLocalStorage());

  assert(!cir::MissingFeatures::opAllocaVarDeclContext());
  return emitAutoVarDecl(d);
}

void CIRGenFunction::emitDecl(const Decl &d) {
  switch (d.getKind()) {
  case Decl::Var: {
    const VarDecl &vd = cast<VarDecl>(d);
    assert(vd.isLocalVarDecl() &&
           "Should not see file-scope variables inside a function!");
    emitVarDecl(vd);
    return;
  }
  default:
    cgm.errorNYI(d.getSourceRange(), "emitDecl: unhandled decl type");
  }
}
