//===- Scope.cpp - Lexical scope information --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Scope class, which is used for recording
// information about a lexical scope.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/Scope.h"
#include "clang/AST/Decl.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

void Scope::setFlags(Scope *parent, unsigned flags) {
  AnyParent = parent;
  Flags = flags;

  if (parent && !(flags & FnScope)) {
    BreakParent    = parent->BreakParent;
    ContinueParent = parent->ContinueParent;
  } else {
    // Control scopes do not contain the contents of nested function scopes for
    // control flow purposes.
    BreakParent = ContinueParent = nullptr;
  }

  if (parent) {
    Depth = parent->Depth + 1;
    PrototypeDepth = parent->PrototypeDepth;
    PrototypeIndex = 0;
    FnParent       = parent->FnParent;
    BlockParent    = parent->BlockParent;
    TemplateParamParent = parent->TemplateParamParent;
    MSLastManglingParent = parent->MSLastManglingParent;
    MSCurManglingNumber = getMSLastManglingNumber();
    if ((Flags & (FnScope | ClassScope | BlockScope | TemplateParamScope |
                  FunctionPrototypeScope | AtCatchScope | ObjCMethodScope)) ==
        0)
      Flags |= parent->getFlags() & OpenMPSimdDirectiveScope;
  } else {
    Depth = 0;
    PrototypeDepth = 0;
    PrototypeIndex = 0;
    MSLastManglingParent = FnParent = BlockParent = nullptr;
    TemplateParamParent = nullptr;
    MSLastManglingNumber = 1;
    MSCurManglingNumber = 1;
  }

  // If this scope is a function or contains breaks/continues, remember it.
  if (flags & FnScope)            FnParent = this;
  // The MS mangler uses the number of scopes that can hold declarations as
  // part of an external name.
  if (Flags & (ClassScope | FnScope)) {
    MSLastManglingNumber = getMSLastManglingNumber();
    MSLastManglingParent = this;
    MSCurManglingNumber = 1;
  }
  if (flags & BreakScope)         BreakParent = this;
  if (flags & ContinueScope)      ContinueParent = this;
  if (flags & BlockScope)         BlockParent = this;
  if (flags & TemplateParamScope) TemplateParamParent = this;

  // If this is a prototype scope, record that.
  if (flags & FunctionPrototypeScope) PrototypeDepth++;

  if (flags & DeclScope) {
    if (flags & FunctionPrototypeScope)
      ; // Prototype scopes are uninteresting.
    else if ((flags & ClassScope) && getParent()->isClassScope())
      ; // Nested class scopes aren't ambiguous.
    else if ((flags & ClassScope) && getParent()->getFlags() == DeclScope)
      ; // Classes inside of namespaces aren't ambiguous.
    else if ((flags & EnumScope))
      ; // Don't increment for enum scopes.
    else
      incrementMSManglingNumber();
  }
}

void Scope::Init(Scope *parent, unsigned flags) {
  setFlags(parent, flags);

  DeclsInScope.clear();
  UsingDirectives.clear();
  Entity = nullptr;
  ErrorTrap.reset();
  NRVO.setPointerAndInt(nullptr, 0);
}

bool Scope::containedInPrototypeScope() const {
  const Scope *S = this;
  while (S) {
    if (S->isFunctionPrototypeScope())
      return true;
    S = S->getParent();
  }
  return false;
}

void Scope::AddFlags(unsigned FlagsToSet) {
  assert((FlagsToSet & ~(BreakScope | ContinueScope)) == 0 &&
         "Unsupported scope flags");
  if (FlagsToSet & BreakScope) {
    assert((Flags & BreakScope) == 0 && "Already set");
    BreakParent = this;
  }
  if (FlagsToSet & ContinueScope) {
    assert((Flags & ContinueScope) == 0 && "Already set");
    ContinueParent = this;
  }
  Flags |= FlagsToSet;
}

void Scope::mergeNRVOIntoParent() {
  if (VarDecl *Candidate = NRVO.getPointer()) {
    if (isDeclScope(Candidate))
      Candidate->setNRVOVariable(true);
  }

  if (getEntity())
    return;

  if (NRVO.getInt())
    getParent()->setNoNRVO();
  else if (NRVO.getPointer())
    getParent()->addNRVOCandidate(NRVO.getPointer());
}

LLVM_DUMP_METHOD void Scope::dump() const { dumpImpl(llvm::errs()); }

void Scope::dumpImpl(raw_ostream &OS) const {
  unsigned Flags = getFlags();
  bool HasFlags = Flags != 0;

  if (HasFlags)
    OS << "Flags: ";

  while (Flags) {
    if (Flags & FnScope) {
      OS << "FnScope";
      Flags &= ~FnScope;
    } else if (Flags & BreakScope) {
      OS << "BreakScope";
      Flags &= ~BreakScope;
    } else if (Flags & ContinueScope) {
      OS << "ContinueScope";
      Flags &= ~ContinueScope;
    } else if (Flags & DeclScope) {
      OS << "DeclScope";
      Flags &= ~DeclScope;
    } else if (Flags & ControlScope) {
      OS << "ControlScope";
      Flags &= ~ControlScope;
    } else if (Flags & ClassScope) {
      OS << "ClassScope";
      Flags &= ~ClassScope;
    } else if (Flags & BlockScope) {
      OS << "BlockScope";
      Flags &= ~BlockScope;
    } else if (Flags & TemplateParamScope) {
      OS << "TemplateParamScope";
      Flags &= ~TemplateParamScope;
    } else if (Flags & FunctionPrototypeScope) {
      OS << "FunctionPrototypeScope";
      Flags &= ~FunctionPrototypeScope;
    } else if (Flags & FunctionDeclarationScope) {
      OS << "FunctionDeclarationScope";
      Flags &= ~FunctionDeclarationScope;
    } else if (Flags & AtCatchScope) {
      OS << "AtCatchScope";
      Flags &= ~AtCatchScope;
    } else if (Flags & ObjCMethodScope) {
      OS << "ObjCMethodScope";
      Flags &= ~ObjCMethodScope;
    } else if (Flags & SwitchScope) {
      OS << "SwitchScope";
      Flags &= ~SwitchScope;
    } else if (Flags & TryScope) {
      OS << "TryScope";
      Flags &= ~TryScope;
    } else if (Flags & FnTryCatchScope) {
      OS << "FnTryCatchScope";
      Flags &= ~FnTryCatchScope;
    } else if (Flags & SEHTryScope) {
      OS << "SEHTryScope";
      Flags &= ~SEHTryScope;
    } else if (Flags & SEHExceptScope) {
      OS << "SEHExceptScope";
      Flags &= ~SEHExceptScope;
    } else if (Flags & OpenMPDirectiveScope) {
      OS << "OpenMPDirectiveScope";
      Flags &= ~OpenMPDirectiveScope;
    } else if (Flags & OpenMPLoopDirectiveScope) {
      OS << "OpenMPLoopDirectiveScope";
      Flags &= ~OpenMPLoopDirectiveScope;
    } else if (Flags & OpenMPSimdDirectiveScope) {
      OS << "OpenMPSimdDirectiveScope";
      Flags &= ~OpenMPSimdDirectiveScope;
    }

    if (Flags)
      OS << " | ";
  }
  if (HasFlags)
    OS << '\n';

  if (const Scope *Parent = getParent())
    OS << "Parent: (clang::Scope*)" << Parent << '\n';

  OS << "Depth: " << Depth << '\n';
  OS << "MSLastManglingNumber: " << getMSLastManglingNumber() << '\n';
  OS << "MSCurManglingNumber: " << getMSCurManglingNumber() << '\n';
  if (const DeclContext *DC = getEntity())
    OS << "Entity : (clang::DeclContext*)" << DC << '\n';

  if (NRVO.getInt())
    OS << "NRVO not allowed\n";
  else if (NRVO.getPointer())
    OS << "NRVO candidate : (clang::VarDecl*)" << NRVO.getPointer() << '\n';
}
