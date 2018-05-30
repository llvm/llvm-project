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

void Scope::setNRVOCandidate(VarDecl *Candidate) {
  for (Decl *D : DeclsInScope) {
    VarDecl *VD = dyn_cast<VarDecl>(D);
    if (VD && VD != Candidate && VD->isNRVOCandidate())
      VD->setNRVOVariable(false);
  }

  if (Scope *parent = getParent())
    parent->setNRVOCandidate(Candidate);
}

LLVM_DUMP_METHOD void Scope::dump() const { dumpImpl(llvm::errs()); }

void Scope::dumpImpl(raw_ostream &OS) const {
  unsigned Flags = getFlags();
  bool HasFlags = Flags != 0;

  if (HasFlags)
    OS << "Flags: ";

  std::pair<unsigned, const char *> FlagInfo[] = {
      {FnScope, "FnScope"},
      {BreakScope, "BreakScope"},
      {ContinueScope, "ContinueScope"},
      {DeclScope, "DeclScope"},
      {ControlScope, "ControlScope"},
      {ClassScope, "ClassScope"},
      {BlockScope, "BlockScope"},
      {TemplateParamScope, "TemplateParamScope"},
      {FunctionPrototypeScope, "FunctionPrototypeScope"},
      {FunctionDeclarationScope, "FunctionDeclarationScope"},
      {AtCatchScope, "AtCatchScope"},
      {ObjCMethodScope, "ObjCMethodScope"},
      {SwitchScope, "SwitchScope"},
      {TryScope, "TryScope"},
      {FnTryCatchScope, "FnTryCatchScope"},
      {OpenMPDirectiveScope, "OpenMPDirectiveScope"},
      {OpenMPLoopDirectiveScope, "OpenMPLoopDirectiveScope"},
      {OpenMPSimdDirectiveScope, "OpenMPSimdDirectiveScope"},
      {EnumScope, "EnumScope"},
      {SEHTryScope, "SEHTryScope"},
      {SEHExceptScope, "SEHExceptScope"},
      {SEHFilterScope, "SEHFilterScope"},
      {CompoundStmtScope, "CompoundStmtScope"},
      {ClassInheritanceScope, "ClassInheritanceScope"}};

  for (auto Info : FlagInfo) {
    if (Flags & Info.first) {
      OS << Info.second;
      Flags &= ~Info.first;
      if (Flags)
        OS << " | ";
    }
  }

  assert(Flags == 0 && "Unknown scope flags");

  if (HasFlags)
    OS << '\n';

  if (const Scope *Parent = getParent())
    OS << "Parent: (clang::Scope*)" << Parent << '\n';

  OS << "Depth: " << Depth << '\n';
  OS << "MSLastManglingNumber: " << getMSLastManglingNumber() << '\n';
  OS << "MSCurManglingNumber: " << getMSCurManglingNumber() << '\n';
  if (const DeclContext *DC = getEntity())
    OS << "Entity : (clang::DeclContext*)" << DC << '\n';
}
