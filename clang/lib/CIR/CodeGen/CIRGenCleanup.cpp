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

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {
/// Return true if the expression tree contains an AbstractConditionalOperator
/// (ternary ?:), which is the only construct whose CIR codegen calls
/// ConditionalEvaluation::beginEvaluation() and thus causes cleanups to be
/// deferred via pushFullExprCleanup.  Logical &&/|| do NOT call
/// beginEvaluation(); their branch-local cleanups are handled by LexicalScope.
class ConditionalEvaluationFinder
    : public RecursiveASTVisitor<ConditionalEvaluationFinder> {
  bool foundConditional = false;

public:
  bool found() const { return foundConditional; }

  bool VisitAbstractConditionalOperator(AbstractConditionalOperator *) {
    foundConditional = true;
    return false;
  }

  // Don't cross evaluation-context boundaries.
  bool TraverseLambdaExpr(LambdaExpr *) { return true; }
  bool TraverseBlockExpr(BlockExpr *) { return true; }
  bool TraverseStmtExpr(StmtExpr *) { return true; }
};
} // namespace

//===----------------------------------------------------------------------===//
// CIRGenFunction cleanup related
//===----------------------------------------------------------------------===//

/// Emits all the code to cause the given temporary to be cleaned up.
void CIRGenFunction::emitCXXTemporary(const CXXTemporary *temporary,
                                      QualType tempType, Address ptr) {
  pushDestroy(NormalAndEHCleanup, ptr, tempType, destroyCXXObject);
}

Address CIRGenFunction::createCleanupActiveFlag() {
  assert(isInConditionalBranch());
  mlir::Location loc = builder.getUnknownLoc();

  // Place the alloca in the function entry block so it dominates everything,
  // including both regions of any enclosing cir.cleanup.scope.  We can't rely
  // on the default curLexScope path because we may be inside a ternary branch
  // whose LexicalScope would capture the alloca.
  Address active = createTempAllocaWithoutCast(
      builder.getBoolTy(), CharUnits::One(), loc, "cleanup.cond",
      /*arraySize=*/nullptr,
      builder.getBestAllocaInsertPoint(getCurFunctionEntryBlock()));

  // Initialize to false before the outermost conditional.
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.restoreInsertionPoint(outermostConditional->getInsertPoint());
    builder.createFlagStore(loc, false, active.getPointer());
  }

  // Set to true at the current location (inside the conditional branch).
  builder.createFlagStore(loc, true, active.getPointer());

  return active;
}

void CIRGenFunction::initFullExprCleanup() {
  initFullExprCleanupWithFlag(createCleanupActiveFlag());
}

void CIRGenFunction::initFullExprCleanupWithFlag(Address activeFlag) {
  EHCleanupScope &cleanup = cast<EHCleanupScope>(*ehStack.begin());
  assert(!cleanup.hasActiveFlag() && "cleanup already has active flag?");
  cleanup.setActiveFlag(activeFlag);

  cleanup.setTestFlagInNormalCleanup(cleanup.isNormalCleanup());
  cleanup.setTestFlagInEHCleanup(cleanup.isEHCleanup());
}

CIRGenFunction::FullExprCleanupScope::FullExprCleanupScope(CIRGenFunction &cgf,
                                                           const Expr *subExpr)
    : cgf(cgf), cleanups(cgf), scope(nullptr),
      deferredCleanupStackSize(cgf.deferredConditionalCleanupStack.size()) {

  assert(subExpr && "ExprWithCleanups always has a sub-expression");
  ConditionalEvaluationFinder finder;
  finder.TraverseStmt(const_cast<Expr *>(subExpr));
  if (finder.found()) {
    mlir::Location loc = cgf.builder.getUnknownLoc();
    cir::CleanupKind cleanupKind = cgf.getLangOpts().Exceptions
                                       ? cir::CleanupKind::All
                                       : cir::CleanupKind::Normal;
    scope = cir::CleanupScopeOp::create(
        cgf.builder, loc, cleanupKind,
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {},
        /*cleanupBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {});
    cgf.builder.setInsertionPointToEnd(&scope.getBodyRegion().front());
  }
}

void CIRGenFunction::FullExprCleanupScope::exit(
    ArrayRef<mlir::Value *> valuesToReload) {
  assert(!exited && "FullExprCleanupScope::exit called twice");
  exited = true;

  size_t oldSize = deferredCleanupStackSize;
  bool hasDeferredCleanups =
      cgf.deferredConditionalCleanupStack.size() > oldSize;

  if (!scope) {
    cgf.deferredConditionalCleanupStack.truncate(oldSize);
    cleanups.forceCleanup(valuesToReload);
    return;
  }

  // Spill any values that callers need after the scope is closed.
  SmallVector<Address> tempAllocas;
  for (mlir::Value *valPtr : valuesToReload) {
    mlir::Value val = *valPtr;
    if (!val) {
      tempAllocas.push_back(Address::invalid());
      continue;
    }
    Address temp = cgf.createDefaultAlignTempAlloca(val.getType(), val.getLoc(),
                                                    "tmp.exprcleanup");
    tempAllocas.push_back(temp);
    cgf.builder.createStore(val.getLoc(), val, temp);
  }

  // Pop any EH and lifetime-extended cleanups that were pushed during
  // the expression (e.g. temporary destructors).
  cleanups.forceCleanup();

  // Make sure the cleanup scope body region has a terminator.
  {
    mlir::OpBuilder::InsertionGuard guard(cgf.builder);
    mlir::Block &lastBodyBlock = scope.getBodyRegion().back();
    cgf.builder.setInsertionPointToEnd(&lastBodyBlock);
    if (lastBodyBlock.empty() ||
        !lastBodyBlock.back().hasTrait<mlir::OpTrait::IsTerminator>())
      cgf.builder.createYield(scope.getLoc());
  }

  // Emit any deferred cleanups.
  {
    mlir::OpBuilder::InsertionGuard guard(cgf.builder);
    mlir::Block &cleanupBlock = scope.getCleanupRegion().front();
    cgf.builder.setInsertionPointToEnd(&cleanupBlock);

    if (hasDeferredCleanups) {
      for (const PendingCleanupEntry &entry : llvm::reverse(llvm::make_range(
               cgf.deferredConditionalCleanupStack.begin() + oldSize,
               cgf.deferredConditionalCleanupStack.end()))) {
        if (entry.activeFlag.isValid()) {
          mlir::Value flag =
              cgf.builder.createLoad(scope.getLoc(), entry.activeFlag);
          cir::IfOp::create(
              cgf.builder, scope.getLoc(), flag, /*withElseRegion=*/false,
              [&](mlir::OpBuilder &b, mlir::Location loc) {
                cgf.emitDestroy(entry.addr, entry.type, entry.destroyer);
                cgf.builder.createYield(loc);
              });
        } else {
          cgf.emitDestroy(entry.addr, entry.type, entry.destroyer);
        }
      }
    }
    cgf.builder.createYield(scope.getLoc());
  }

  cgf.deferredConditionalCleanupStack.truncate(oldSize);
  cgf.builder.setInsertionPointAfter(scope);

  // Reload spilled values now that the builder is after the closed scope.
  for (auto [addr, valPtr] : llvm::zip(tempAllocas, valuesToReload)) {
    if (!addr.isValid())
      continue;
    *valPtr = cgf.builder.createLoad(valPtr->getLoc(), addr);
  }
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
    // Exceptions are disabled (or no EH flag was requested). Drop the EH
    // flag so the scope entry stays consistent with the op's cleanup kind.
    isEHCleanup = false;
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

  scope.setTestFlagInNormalCleanup(scope.isNormalCleanup());
  scope.setTestFlagInEHCleanup(scope.isEHCleanup());

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
    popCleanupBlock(/*forDeactivation=*/true);
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

/// Check whether a cleanup scope body contains any non-yield exits that branch
/// through the cleanup. These exits branch through the cleanup and require
/// the normal cleanup to be executed even when the cleanup has been
/// deactivated.
static bool bodyHasBranchThroughExits(mlir::Region &bodyRegion) {
  return bodyRegion
      .walk([&](mlir::Operation *op) {
        if (isa<cir::ReturnOp, cir::GotoOp>(op))
          return mlir::WalkResult::interrupt();
        return mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

/// Pop a cleanup block from the stack.
///
/// \param forDeactivation - When true, this indicates that the cleanup block
/// is being popped because it was deactivated while at the top of the stack.
void CIRGenFunction::popCleanupBlock(bool forDeactivation) {
  assert(!ehStack.empty() && "cleanup stack is empty!");
  assert(isa<EHCleanupScope>(*ehStack.begin()) && "top not a cleanup!");
  EHCleanupScope &scope = cast<EHCleanupScope>(*ehStack.begin());

  cir::CleanupScopeOp cleanupScope = scope.getCleanupScopeOp();
  assert(cleanupScope && "CleanupScopeOp is nullptr");

  bool requiresNormalCleanup = scope.isNormalCleanup();
  bool requiresEHCleanup = scope.isEHCleanup();

  // When we're popping a cleanup to deactivate it, we need to know if anything
  // in the cleanup scope body region branches through the cleanup handler
  // before the entire cleanup scope body has executed. If the cleanup scope
  // body falls through, we don't want to emit normal cleanup code. However,
  // if the cleanup body region contains early exits (return or goto), we do
  // need to execute the normal cleanup when the early exit is taken. To handle
  // that case, we guard the cleanup with an "active" flag so that it executes
  // conditionally and set the flag to false when the cleanup body falls
  // through. Classic codegen tracks this state with "hasBranches" and
  // "getFixupDepth" on the cleanup scope, but because CIR uses structured
  // control flow, we need to check for early exits and insert the active
  // flag handling here. Note that when a cleanup is deactivated while not at
  // the top of the stack, the active flag gets created in
  // setupCleanupBlockDeactivation.
  if (forDeactivation && requiresNormalCleanup) {
    if (bodyHasBranchThroughExits(cleanupScope.getBodyRegion())) {
      // The active flag shouldn't exist if the scope was at the top of the
      // stack when it was deactivated.
      assert(!scope.getActiveFlag().isValid() && "active flag already set");

      // Create the flag.
      mlir::Location loc = builder.getUnknownLoc();
      Address activeFlag = createTempAllocaWithoutCast(
          builder.getBoolTy(), CharUnits::One(), loc, "cleanup.isactive");

      // Initialize the flag to true before the cleanup scope (the point where
      // the cleanup becomes active).
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(cleanupScope);
        builder.createFlagStore(loc, true, activeFlag.getPointer());
      }

      // Set the flag to false at the end of the cleanup scope body region.
      assert(builder.getInsertionBlock() ==
                 &cleanupScope.getBodyRegion().back() &&
             "expected insertion point in cleanup body");
      builder.createFlagStore(loc, false, activeFlag.getPointer());

      scope.setActiveFlag(activeFlag);
      scope.setTestFlagInNormalCleanup(true);
    } else {
      // If the cleanup was pushed on the stack as normal+eh, downgrade it to
      // eh-only.
      if (requiresEHCleanup)
        cleanupScope.setCleanupKind(cir::CleanupKind::EH);
      requiresNormalCleanup = false;
    }
  }

  Address normalActiveFlag = scope.shouldTestFlagInNormalCleanup()
                                 ? scope.getActiveFlag()
                                 : Address::invalid();
  Address ehActiveFlag = scope.shouldTestFlagInEHCleanup()
                             ? scope.getActiveFlag()
                             : Address::invalid();

  // If we don't need the cleanup at all, we're done.
  if (!requiresNormalCleanup && !requiresEHCleanup) {
    // If we get here, the cleanup scope isn't needed. Rather than try to move
    // the contents of its body region out of the cleanup and erase it, we just
    // add a yield to the cleanup region to make it valid but no-op. It will be
    // erased during canonicalization.
    mlir::Block &cleanupBlock = cleanupScope.getCleanupRegion().back();
    if (!cleanupBlock.mightHaveTerminator()) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&cleanupBlock);
      cir::YieldOp::create(builder, builder.getUnknownLoc());
    }
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
  for (const PendingCleanupEntry &cleanup : llvm::make_range(
           lifetimeExtendedCleanupStack.begin() + oldLifetimeExtendedSize,
           lifetimeExtendedCleanupStack.end()))
    pushPendingCleanupToEHStack(cleanup);
  lifetimeExtendedCleanupStack.truncate(oldLifetimeExtendedSize);
}
