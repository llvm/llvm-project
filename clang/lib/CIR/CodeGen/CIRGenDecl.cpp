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
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

void CIRGenFunction::emitAutoVarAlloca(const VarDecl &d) {
  QualType ty = d.getType();
  if (ty.getAddressSpace() != LangAS::Default)
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarAlloca: address space");

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
  address = createTempAlloca(allocaTy, alignment, loc, d.getName());
  declare(address, &d, ty, getLoc(d.getSourceRange()), alignment);
    
  setAddrOfLocalVar(&d, address);
}

void CIRGenFunction::emitAutoVarInit(const clang::VarDecl &d) {
  QualType type = d.getType();

  // If this local has an initializer, emit it now.
  const Expr *init = d.getInit();

  if (init || !type.isPODType(getContext())) {
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarInit");
  }
}

void CIRGenFunction::emitAutoVarCleanups(const clang::VarDecl &d) {
  // Check the type for a cleanup.
  if (QualType::DestructionKind dtorKind = d.needsDestruction(getContext()))
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarCleanups: type cleanup");

  assert(!cir::MissingFeatures::opAllocaPreciseLifetime());

  // Handle the cleanup attribute.
  if (d.hasAttr<CleanupAttr>())
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarCleanups: CleanupAttr");
}


/// Emit code and set up symbol table for a variable declaration with auto,
/// register, or no storage class specifier. These turn into simple stack
/// objects, globals depending on target.
void CIRGenFunction::emitAutoVarDecl(const VarDecl &d) {
  emitAutoVarAlloca(d);
  emitAutoVarInit(d);
  emitAutoVarCleanups(d);
}

void CIRGenFunction::emitVarDecl(const VarDecl &d) {
  // If the declaration has external storage, don't emit it now, allow it to be
  // emitted lazily on its first use.
  if (d.hasExternalStorage())
    return;

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
