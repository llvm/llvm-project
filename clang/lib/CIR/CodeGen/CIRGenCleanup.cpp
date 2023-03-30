//===--- CIRGenCleanup.cpp - Bookkeeping and code emission for cleanups ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code dealing with the IR generation for cleanups
// and related information.
//
// A "cleanup" is a piece of code which needs to be executed whenever
// control transfers out of a particular scope.  This can be
// conditionalized to occur only on exceptional control flow, only on
// normal control flow, or both.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCleanup.h"
#include "CIRGenFunction.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;

//===----------------------------------------------------------------------===//
// CIRGenFunction cleanup related
//===----------------------------------------------------------------------===//

/// Build a unconditional branch to the lexical scope cleanup block
/// or with the labeled blocked if already solved.
///
/// Track on scope basis, goto's we need to fix later.
mlir::cir::BrOp CIRGenFunction::buildBranchThroughCleanup(mlir::Location Loc,
                                                          JumpDest Dest) {
  // Remove this once we go for making sure unreachable code is
  // well modeled (or not).
  assert(builder.getInsertionBlock() && "not yet implemented");
  assert(!UnimplementedFeature::ehStack());

  // Insert a branch: to the cleanup block (unsolved) or to the already
  // materialized label. Keep track of unsolved goto's.
  return builder.create<BrOp>(Loc, Dest.isValid() ? Dest.getBlock()
                                                  : ReturnBlock().getBlock());
}

/// Emits all the code to cause the given temporary to be cleaned up.
void CIRGenFunction::buildCXXTemporary(const CXXTemporary *Temporary,
                                       QualType TempType, Address Ptr) {
  pushDestroy(NormalAndEHCleanup, Ptr, TempType, destroyCXXObject,
              /*useEHCleanup*/ true);
}

//===----------------------------------------------------------------------===//
// EHScopeStack
//===----------------------------------------------------------------------===//

void EHScopeStack::Cleanup::anchor() {}

/// Push an entry of the given size onto this protected-scope stack.
char *EHScopeStack::allocate(size_t Size) {
  Size = llvm::alignTo(Size, ScopeStackAlignment);
  if (!StartOfBuffer) {
    unsigned Capacity = 1024;
    while (Capacity < Size)
      Capacity *= 2;
    StartOfBuffer = new char[Capacity];
    StartOfData = EndOfBuffer = StartOfBuffer + Capacity;
  } else if (static_cast<size_t>(StartOfData - StartOfBuffer) < Size) {
    unsigned CurrentCapacity = EndOfBuffer - StartOfBuffer;
    unsigned UsedCapacity = CurrentCapacity - (StartOfData - StartOfBuffer);

    unsigned NewCapacity = CurrentCapacity;
    do {
      NewCapacity *= 2;
    } while (NewCapacity < UsedCapacity + Size);

    char *NewStartOfBuffer = new char[NewCapacity];
    char *NewEndOfBuffer = NewStartOfBuffer + NewCapacity;
    char *NewStartOfData = NewEndOfBuffer - UsedCapacity;
    memcpy(NewStartOfData, StartOfData, UsedCapacity);
    delete[] StartOfBuffer;
    StartOfBuffer = NewStartOfBuffer;
    EndOfBuffer = NewEndOfBuffer;
    StartOfData = NewStartOfData;
  }

  assert(StartOfBuffer + Size <= StartOfData);
  StartOfData -= Size;
  return StartOfData;
}

void *EHScopeStack::pushCleanup(CleanupKind Kind, size_t Size) {
  char *Buffer = allocate(EHCleanupScope::getSizeForCleanupSize(Size));
  bool IsNormalCleanup = Kind & NormalCleanup;
  bool IsEHCleanup = Kind & EHCleanup;
  bool IsLifetimeMarker = Kind & LifetimeMarker;

  // Per C++ [except.terminate], it is implementation-defined whether none,
  // some, or all cleanups are called before std::terminate. Thus, when
  // terminate is the current EH scope, we may skip adding any EH cleanup
  // scopes.
  if (InnermostEHScope != stable_end() &&
      find(InnermostEHScope)->getKind() == EHScope::Terminate)
    IsEHCleanup = false;

  EHCleanupScope *Scope = new (Buffer)
      EHCleanupScope(IsNormalCleanup, IsEHCleanup, Size, BranchFixups.size(),
                     InnermostNormalCleanup, InnermostEHScope);
  if (IsNormalCleanup)
    InnermostNormalCleanup = stable_begin();
  if (IsEHCleanup)
    InnermostEHScope = stable_begin();
  if (IsLifetimeMarker)
    llvm_unreachable("NYI");

  // With Windows -EHa, Invoke llvm.seh.scope.begin() for EHCleanup
  if (CGF->getLangOpts().EHAsynch && IsEHCleanup && !IsLifetimeMarker &&
      CGF->getTarget().getCXXABI().isMicrosoft())
    llvm_unreachable("NYI");

  return Scope->getCleanupBuffer();
}