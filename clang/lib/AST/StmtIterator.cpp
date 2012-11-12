//===--- StmtIterator.cpp - Iterators for Statements ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines internal methods for StmtIterator.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtIterator.h"
#include "clang/AST/Decl.h"

using namespace clang;

// FIXME: Add support for dependent-sized array types in C++?
// Does it even make sense to build a CFG for an uninstantiated template?
static inline const VariableArrayType *FindVA(const Type* t) {
  while (const ArrayType *vt = dyn_cast<ArrayType>(t)) {
    if (const VariableArrayType *vat = dyn_cast<VariableArrayType>(vt))
      if (vat->getSizeExpr())
        return vat;

    t = vt->getElementType().getTypePtr();
  }

  return NULL;
}

void StmtIteratorBase::NextVA() {
  assert (getVAPtr());

  const VariableArrayType *p = getVAPtr();
  p = FindVA(p->getElementType().getTypePtr());
  setVAPtr(p);

  if (p)
    return;

  if (inDecl()) {
    if (VarDecl* VD = dyn_cast<VarDecl>(decl))
      if (VD->Init)
        return;

    NextDecl();
  }
  else if (inDeclGroup()) {
    if (VarDecl* VD = dyn_cast<VarDecl>(*DGI))
      if (VD->Init)
        return;

    NextDecl();
  }
  else {
    assert (inSizeOfTypeVA());
    assert(!decl);
    RawVAPtr = 0;
  }
}

void StmtIteratorBase::NextDecl(bool ImmediateAdvance) {
  assert (getVAPtr() == NULL);

  if (inDecl()) {
    assert(decl);

    // FIXME: SIMPLIFY AWAY.
    if (ImmediateAdvance)
      decl = 0;
    else if (HandleDecl(decl))
      return;
  }
  else {
    assert(inDeclGroup());

    if (ImmediateAdvance)
      ++DGI;

    for ( ; DGI != DGE; ++DGI)
      if (HandleDecl(*DGI))
        return;
  }

  RawVAPtr = 0;
}

bool StmtIteratorBase::HandleDecl(Decl* D) {

  if (VarDecl* VD = dyn_cast<VarDecl>(D)) {
    if (const VariableArrayType* VAPtr = FindVA(VD->getType().getTypePtr())) {
      setVAPtr(VAPtr);
      return true;
    }

    if (VD->getInit())
      return true;
  }
  else if (TypedefNameDecl* TD = dyn_cast<TypedefNameDecl>(D)) {
    if (const VariableArrayType* VAPtr =
        FindVA(TD->getUnderlyingType().getTypePtr())) {
      setVAPtr(VAPtr);
      return true;
    }
  }
  else if (EnumConstantDecl* ECD = dyn_cast<EnumConstantDecl>(D)) {
    if (ECD->getInitExpr())
      return true;
  }

  return false;
}

StmtIteratorBase::StmtIteratorBase(Decl *d, Stmt **s)
  : stmt(s), decl(d), RawVAPtr(d ? DeclMode : 0) {
  if (decl)
    NextDecl(false);
}

StmtIteratorBase::StmtIteratorBase(Decl** dgi, Decl** dge)
  : stmt(0), DGI(dgi), RawVAPtr(DeclGroupMode), DGE(dge) {
  NextDecl(false);
}

StmtIteratorBase::StmtIteratorBase(const VariableArrayType* t)
  : stmt(0), decl(0), RawVAPtr(SizeOfTypeVAMode) {
  RawVAPtr |= reinterpret_cast<uintptr_t>(t);
}

Stmt*& StmtIteratorBase::GetDeclExpr() const {

  if (const VariableArrayType* VAPtr = getVAPtr()) {
    assert (VAPtr->SizeExpr);
    return const_cast<Stmt*&>(VAPtr->SizeExpr);
  }

  assert (inDecl() || inDeclGroup());

  if (inDeclGroup()) {
    VarDecl* VD = cast<VarDecl>(*DGI);
    return *VD->getInitAddress();
  }

  assert (inDecl());

  if (VarDecl* VD = dyn_cast<VarDecl>(decl)) {
    assert (VD->Init);
    return *VD->getInitAddress();
  }

  EnumConstantDecl* ECD = cast<EnumConstantDecl>(decl);
  return ECD->Init;
}
