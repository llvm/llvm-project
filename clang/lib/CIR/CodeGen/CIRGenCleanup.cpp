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

#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

//===----------------------------------------------------------------------===//
// CIRGenFunction cleanup related
//===----------------------------------------------------------------------===//

/// Emits all the code to cause the given temporary to be cleaned up.
void CIRGenFunction::emitCXXTemporary(const CXXTemporary *temporary,
                                      QualType tempType, Address ptr) {
  pushDestroy(NormalAndEHCleanup, ptr, tempType, destroyCXXObject);
}

//===----------------------------------------------------------------------===//
// EHScopeStack
//===----------------------------------------------------------------------===//

void EHScopeStack::Cleanup::anchor() {}

EHScopeStack::stable_iterator
EHScopeStack::getInnermostActiveNormalCleanup() const {
  stable_iterator si = getInnermostNormalCleanup();
  stable_iterator se = stable_end();
  while (si != se) {
    EHCleanupScope &cleanup = llvm::cast<EHCleanupScope>(*find(si));
    if (cleanup.isActive())
      return si;
    si = cleanup.getEnclosingNormalCleanup();
  }
  return stable_end();
}

/// Push an entry of the given size onto this protected-scope stack.
char *EHScopeStack::allocate(size_t size) {
  size = llvm::alignTo(size, ScopeStackAlignment);
  if (!startOfBuffer) {
    unsigned capacity = llvm::PowerOf2Ceil(std::max<size_t>(size, 1024ul));
    startOfBuffer = std::make_unique<char[]>(capacity);
    startOfData = endOfBuffer = startOfBuffer.get() + capacity;
  } else if (static_cast<size_t>(startOfData - startOfBuffer.get()) < size) {
    unsigned currentCapacity = endOfBuffer - startOfBuffer.get();
    unsigned usedCapacity =
        currentCapacity - (startOfData - startOfBuffer.get());
    unsigned requiredCapacity = usedCapacity + size;
    // We know from the 'else if' condition that requiredCapacity is greater
    // than currentCapacity.
    unsigned newCapacity = llvm::PowerOf2Ceil(requiredCapacity);

    std::unique_ptr<char[]> newStartOfBuffer =
        std::make_unique<char[]>(newCapacity);
    char *newEndOfBuffer = newStartOfBuffer.get() + newCapacity;
    char *newStartOfData = newEndOfBuffer - usedCapacity;
    memcpy(newStartOfData, startOfData, usedCapacity);
    startOfBuffer.swap(newStartOfBuffer);
    endOfBuffer = newEndOfBuffer;
    startOfData = newStartOfData;
  }

  assert(startOfBuffer.get() + size <= startOfData);
  startOfData -= size;
  return startOfData;
}

void EHScopeStack::deallocate(size_t size) {
  startOfData += llvm::alignTo(size, ScopeStackAlignment);
}

void *EHScopeStack::pushCleanup(CleanupKind kind, size_t size) {
  char *buffer = allocate(EHCleanupScope::getSizeForCleanupSize(size));
  bool isNormalCleanup = kind & NormalCleanup;
  bool isEHCleanup = kind & EHCleanup;
  bool isLifetimeMarker = kind & LifetimeMarker;
  bool skipCleanupScope = false;

  cir::CleanupKind cleanupKind = cir::CleanupKind::All;
  if (isEHCleanup && cgf->getLangOpts().Exceptions) {
    cleanupKind =
        isNormalCleanup ? cir::CleanupKind::All : cir::CleanupKind::EH;
  } else {
    if (isNormalCleanup)
      cleanupKind = cir::CleanupKind::Normal;
    else
      skipCleanupScope = true;
  }

  cir::CleanupScopeOp cleanupScope = nullptr;
  if (!skipCleanupScope) {
    CIRGenBuilderTy &builder = cgf->getBuilder();
    mlir::Location loc = builder.getUnknownLoc();
    cleanupScope = cir::CleanupScopeOp::create(
        builder, loc, cleanupKind,
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // Terminations will be handled in popCleanup
        },
        /*cleanupBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // Terminations will be handled after emiting cleanup
        });

    builder.setInsertionPointToEnd(&cleanupScope.getBodyRegion().back());
  }

  // Per C++ [except.terminate], it is implementation-defined whether none,
  // some, or all cleanups are called before std::terminate. Thus, when
  // terminate is the current EH scope, we may skip adding any EH cleanup
  // scopes.
  if (innermostEHScope != stable_end() &&
      find(innermostEHScope)->getKind() == EHScope::Terminate)
    isEHCleanup = false;

  EHCleanupScope *scope = new (buffer)
      EHCleanupScope(isNormalCleanup, isEHCleanup, size, cleanupScope,
                     innermostNormalCleanup, innermostEHScope);

  if (isNormalCleanup)
    innermostNormalCleanup = stable_begin();

  if (isEHCleanup)
    innermostEHScope = stable_begin();

  if (isLifetimeMarker)
    cgf->cgm.errorNYI("push lifetime marker cleanup");

  // With Windows -EHa, Invoke llvm.seh.scope.begin() for EHCleanup
  if (cgf->getLangOpts().EHAsynch && isEHCleanup && !isLifetimeMarker &&
      cgf->getTarget().getCXXABI().isMicrosoft())
    cgf->cgm.errorNYI("push seh cleanup");

  return scope->getCleanupBuffer();
}

void EHScopeStack::popCleanup() {
  assert(!empty() && "popping exception stack when not empty");

  assert(isa<EHCleanupScope>(*begin()));
  EHCleanupScope &cleanup = cast<EHCleanupScope>(*begin());
  innermostNormalCleanup = cleanup.getEnclosingNormalCleanup();
  innermostEHScope = cleanup.getEnclosingEHScope();
  deallocate(cleanup.getAllocatedSize());

  cir::CleanupScopeOp cleanupScope = cleanup.getCleanupScopeOp();
  if (cleanupScope) {
    auto *block = &cleanupScope.getBodyRegion().back();
    if (!block->mightHaveTerminator()) {
      mlir::OpBuilder::InsertionGuard guard(cgf->getBuilder());
      cgf->getBuilder().setInsertionPointToEnd(block);
      cir::YieldOp::create(cgf->getBuilder(),
                           cgf->getBuilder().getUnknownLoc());
    }
    cgf->getBuilder().setInsertionPointAfter(cleanupScope);
  }

  // Destroy the cleanup.
  cleanup.destroy();
}

bool EHScopeStack::requiresCatchOrCleanup() const {
  for (stable_iterator si = getInnermostEHScope(); si != stable_end();) {
    if (auto *cleanup = dyn_cast<EHCleanupScope>(&*find(si))) {
      if (cleanup->isLifetimeMarker()) {
        // Skip lifetime markers and continue from the enclosing EH scope
        assert(!cir::MissingFeatures::emitLifetimeMarkers());
        continue;
      }
    }
    return true;
  }
  return false;
}

/// The given cleanup block is being deactivated. Configure a cleanup variable
/// if necessary.
static void setupCleanupBlockDeactivation(CIRGenFunction &cgf,
                                          EHScopeStack::stable_iterator c,
                                          mlir::Operation *dominatingIP) {
  EHCleanupScope &scope = cast<EHCleanupScope>(*cgf.ehStack.find(c));

  assert((scope.isNormalCleanup() || scope.isEHCleanup()) &&
         "cleanup block is neither normal nor EH?");

  if (scope.isNormalCleanup())
    scope.setTestFlagInNormalCleanup();

  if (scope.isEHCleanup())
    scope.setTestFlagInEHCleanup();

  CIRGenBuilderTy &builder = cgf.getBuilder();

  // If the cleanup block doesn't exist yet, create it and set its initial
  // value to `true`. If we are inside a conditional branch, the value must be
  // initialized before the conditional branch begins.
  Address var = scope.getActiveFlag();
  if (!var.isValid()) {
    mlir::Location loc = builder.getUnknownLoc();

    var = cgf.createTempAllocaWithoutCast(builder.getBoolTy(), CharUnits::One(),
                                          loc, "cleanup.isactive");
    scope.setActiveFlag(var);

    assert(dominatingIP && "no existing variable and no dominating IP!");

    if (cgf.isInConditionalBranch()) {
      mlir::Value val = builder.getBool(true, loc);
      cgf.setBeforeOutermostConditional(val, var);
    } else {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(dominatingIP);
      builder.createFlagStore(loc, true, var.getPointer());
    }
  }

  // The code above sets the `isActive` flag to `true` as its initial state
  // at the point where the variable is created. The code below sets it to
  // `false` at the point where the cleanup is deactivated.
  mlir::Location loc = builder.getUnknownLoc();
  builder.createFlagStore(loc, false, var.getPointer());
}

/// Deactive a cleanup that was created in an active state.
void CIRGenFunction::deactivateCleanupBlock(EHScopeStack::stable_iterator c,
                                            mlir::Operation *dominatingIP) {
  assert(c != ehStack.stable_end() && "deactivating bottom of stack?");
  EHCleanupScope &scope = cast<EHCleanupScope>(*ehStack.find(c));
  assert(scope.isActive() && "double deactivation");

  // If it's the top of the stack, just pop it, but do so only if it belongs
  // to the current RunCleanupsScope.
  if (c == ehStack.stable_begin() &&
      currentCleanupStackDepth.strictlyEncloses(c)) {
    popCleanupBlock();
    return;
  }

  // Otherwise, follow the general case.
  setupCleanupBlockDeactivation(*this, c, dominatingIP);

  scope.setActive(false);
}

static void emitCleanup(CIRGenFunction &cgf, cir::CleanupScopeOp cleanupScope,
                        EHScopeStack::Cleanup *cleanup,
                        EHScopeStack::Cleanup::Flags flags,
                        Address activeFlag) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Block &block = cleanupScope.getCleanupRegion().back();

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&block);

  // Ask the cleanup to emit itself.
  assert(cgf.haveInsertPoint() && "expected insertion point");

  if (activeFlag.isValid()) {
    mlir::Location loc = cleanupScope.getLoc();
    mlir::Value isActive = builder.createFlagLoad(loc, activeFlag.getPointer());
    cir::IfOp::create(builder, loc, isActive,
                      /*withElseRegion=*/false,
                      /*thenBuilder=*/
                      [&](mlir::OpBuilder &, mlir::Location) {
                        cleanup->emit(cgf, flags);
                        assert(cgf.haveInsertPoint() &&
                               "cleanup ended with no insertion point?");
                        builder.createYield(loc);
                      });
  } else {
    cleanup->emit(cgf, flags);
    assert(cgf.haveInsertPoint() && "cleanup ended with no insertion point?");
  }

  mlir::Block &cleanupRegionLastBlock = cleanupScope.getCleanupRegion().back();
  if (cleanupRegionLastBlock.empty() ||
      !cleanupRegionLastBlock.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPointToEnd(&cleanupRegionLastBlock);
    builder.createYield(cleanupScope.getLoc());
  }
}

void CIRGenFunction::popCleanupBlock() {
  assert(!ehStack.empty() && "cleanup stack is empty!");
  assert(isa<EHCleanupScope>(*ehStack.begin()) && "top not a cleanup!");
  EHCleanupScope &scope = cast<EHCleanupScope>(*ehStack.begin());

  cir::CleanupScopeOp cleanupScope = scope.getCleanupScopeOp();
  assert(cleanupScope && "CleanupScopeOp is nullptr");

  // Remember activation information.
  Address normalActiveFlag = scope.shouldTestFlagInNormalCleanup()
                                 ? scope.getActiveFlag()
                                 : Address::invalid();
  Address ehActiveFlag = scope.shouldTestFlagInEHCleanup()
                             ? scope.getActiveFlag()
                             : Address::invalid();

  bool requiresNormalCleanup = scope.isNormalCleanup();
  bool requiresEHCleanup = scope.isEHCleanup();

  // If we don't need the cleanup at all, we're done.
  if (!requiresNormalCleanup && !requiresEHCleanup) {
    ehStack.popCleanup();
    return;
  }

  // Copy the cleanup emission data out.  This uses either a stack
  // array or malloc'd memory, depending on the size, which is
  // behavior that SmallVector would provide, if we could use it
  // here. Unfortunately, if you ask for a SmallVector<char>, the
  // alignment isn't sufficient.
  auto *cleanupSource = reinterpret_cast<char *>(scope.getCleanupBuffer());
  alignas(EHScopeStack::ScopeStackAlignment) char
      cleanupBufferStack[8 * sizeof(void *)];
  std::unique_ptr<char[]> cleanupBufferHeap;
  size_t cleanupSize = scope.getCleanupSize();
  EHScopeStack::Cleanup *cleanup;

  // This is necessary because we are going to deallocate the cleanup
  // (in popCleanup) before we emit it.
  if (cleanupSize <= sizeof(cleanupBufferStack)) {
    memcpy(cleanupBufferStack, cleanupSource, cleanupSize);
    cleanup = reinterpret_cast<EHScopeStack::Cleanup *>(cleanupBufferStack);
  } else {
    cleanupBufferHeap.reset(new char[cleanupSize]);
    memcpy(cleanupBufferHeap.get(), cleanupSource, cleanupSize);
    cleanup =
        reinterpret_cast<EHScopeStack::Cleanup *>(cleanupBufferHeap.get());
  }

  EHScopeStack::Cleanup::Flags cleanupFlags;
  if (scope.isNormalCleanup())
    cleanupFlags.setIsNormalCleanupKind();
  if (scope.isEHCleanup())
    cleanupFlags.setIsEHCleanupKind();

  // Determine the active flag for the cleanup handler.
  Address cleanupActiveFlag = normalActiveFlag.isValid() ? normalActiveFlag
                              : ehActiveFlag.isValid()   ? ehActiveFlag
                                                         : Address::invalid();

  // In CIR, the cleanup code is emitted into the cleanup region of the
  // cir.cleanup.scope op. There is no CFG threading needed — the FlattenCFG
  // pass handles lowering the structured cleanup scope.
  ehStack.popCleanup();
  scope.markEmitted();
  emitCleanup(*this, cleanupScope, cleanup, cleanupFlags, cleanupActiveFlag);
}

/// Pops cleanup blocks until the given savepoint is reached.
void CIRGenFunction::popCleanupBlocks(
    EHScopeStack::stable_iterator oldCleanupStackDepth,
    ArrayRef<mlir::Value *> valuesToReload) {
  // If the current stack depth is the same as the cleanup stack depth,
  // we won't be exiting any cleanup scopes, so we don't need to reload
  // any values.
  bool requiresCleanup = false;
  for (auto it = ehStack.begin(), ie = ehStack.find(oldCleanupStackDepth);
       it != ie; ++it) {
    if (isa<EHCleanupScope>(&*it)) {
      requiresCleanup = true;
      break;
    }
  }

  // If there are values that we need to keep live, spill them now before
  // we pop the cleanup blocks. These are passed as pointers to mlir::Value
  // because we're going to replace them with the reloaded value.
  SmallVector<Address> tempAllocas;
  if (requiresCleanup) {
    for (mlir::Value *valPtr : valuesToReload) {
      mlir::Value val = *valPtr;
      if (!val)
        continue;

      // TODO(cir): Check for static allocas.

      Address temp = createDefaultAlignTempAlloca(val.getType(), val.getLoc(),
                                                  "tmp.exprcleanup");
      tempAllocas.push_back(temp);
      builder.createStore(val.getLoc(), val, temp);
    }
  }

  // Pop cleanup blocks until we reach the base stack depth for the
  // current scope.
  while (ehStack.stable_begin() != oldCleanupStackDepth)
    popCleanupBlock();

  // Reload the values that we spilled, if necessary.
  if (requiresCleanup) {
    for (auto [addr, valPtr] : llvm::zip(tempAllocas, valuesToReload)) {
      mlir::Location loc = valPtr->getLoc();
      *valPtr = builder.createLoad(loc, addr);
    }
  }
}

/// Pops cleanup blocks until the given savepoint is reached, then add the
/// cleanups from the given savepoint in the lifetime-extended cleanups stack.
void CIRGenFunction::popCleanupBlocks(
    EHScopeStack::stable_iterator oldCleanupStackDepth,
    size_t oldLifetimeExtendedSize, ArrayRef<mlir::Value *> valuesToReload) {
  popCleanupBlocks(oldCleanupStackDepth, valuesToReload);

  // Promote deferred lifetime-extended cleanups onto the EH scope stack.
  for (const LifetimeExtendedCleanupEntry &cleanup : llvm::make_range(
           lifetimeExtendedCleanupStack.begin() + oldLifetimeExtendedSize,
           lifetimeExtendedCleanupStack.end()))
    pushLifetimeExtendedCleanupToEHStack(cleanup);
  lifetimeExtendedCleanupStack.truncate(oldLifetimeExtendedSize);
}
