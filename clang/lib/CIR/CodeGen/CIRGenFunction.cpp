//===- CIRGenFunction.cpp - Emit CIR from ASTs for a Function -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This coordinates the per-function state used while generating code
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "CIRGenCXXABI.h"
#include "CIRGenModule.h"
#include "CIRGenOpenMPRuntime.h"
#include "clang/AST/Attrs.inc"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/CIR/MissingFeatures.h"

#include "clang/AST/ASTLambda.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/DiagnosticCategories.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/FPEnv.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/ADT/PointerIntPair.h"

#include "CIRGenTBAA.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

CIRGenFunction::CIRGenFunction(CIRGenModule &cgm, CIRGenBuilderTy &builder,
                               bool suppressNewContext)
    : CIRGenTypeCache(cgm), CGM{cgm}, builder(builder),
      SanOpts(cgm.getLangOpts().Sanitize), CurFPFeatures(cgm.getLangOpts()),
      ShouldEmitLifetimeMarkers(false) {
  if (!suppressNewContext)
    cgm.getCXXABI().getMangleContext().startNewFunction();
  EHStack.setCGF(this);

  // TODO(CIR): SetFastMathFlags(CurFPFeatures);
}

CIRGenFunction::~CIRGenFunction() {
  assert(LifetimeExtendedCleanupStack.empty() && "failed to emit a cleanup");
  assert(DeferredDeactivationCleanupStack.empty() &&
         "missed to deactivate a cleanup");

  // TODO(cir): set function is finished.
  assert(!cir::MissingFeatures::openMPRuntime());

  // If we have an OpenMPIRBuilder we want to finalize functions (incl.
  // outlining etc) at some point. Doing it once the function codegen is done
  // seems to be a reasonable spot. We do it here, as opposed to the deletion
  // time of the CodeGenModule, because we have to ensure the IR has not yet
  // been "emitted" to the outside, thus, modifications are still sensible.
  assert(!cir::MissingFeatures::openMPRuntime());
}

clang::ASTContext &CIRGenFunction::getContext() const {
  return CGM.getASTContext();
}

cir::TypeEvaluationKind CIRGenFunction::getEvaluationKind(QualType type) {
  type = type.getCanonicalType();
  while (true) {
    switch (type->getTypeClass()) {
#define TYPE(name, parent)
#define ABSTRACT_TYPE(name, parent)
#define NON_CANONICAL_TYPE(name, parent) case Type::name:
#define DEPENDENT_TYPE(name, parent) case Type::name:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(name, parent) case Type::name:
#include "clang/AST/TypeNodes.inc"
      llvm_unreachable("non-canonical or dependent type in IR-generation");

    case Type::ArrayParameter:
    case Type::HLSLAttributedResource:
      llvm_unreachable("NYI");

    case Type::Auto:
    case Type::DeducedTemplateSpecialization:
      llvm_unreachable("undeduced type in IR-generation");

    // Various scalar types.
    case Type::Builtin:
    case Type::Pointer:
    case Type::BlockPointer:
    case Type::LValueReference:
    case Type::RValueReference:
    case Type::MemberPointer:
    case Type::Vector:
    case Type::ExtVector:
    case Type::ConstantMatrix:
    case Type::FunctionProto:
    case Type::FunctionNoProto:
    case Type::Enum:
    case Type::ObjCObjectPointer:
    case Type::Pipe:
    case Type::BitInt:
      return cir::TEK_Scalar;

    // Complexes.
    case Type::Complex:
      return cir::TEK_Complex;

    // Arrays, records, and Objective-C objects.
    case Type::ConstantArray:
    case Type::IncompleteArray:
    case Type::VariableArray:
    case Type::Record:
    case Type::ObjCObject:
    case Type::ObjCInterface:
      return cir::TEK_Aggregate;

    // We operate on atomic values according to their underlying type.
    case Type::Atomic:
      type = cast<AtomicType>(type)->getValueType();
      continue;
    }
    llvm_unreachable("unknown type kind!");
  }
}

mlir::Type CIRGenFunction::convertTypeForMem(QualType t) {
  return CGM.getTypes().convertTypeForMem(t);
}

mlir::Type CIRGenFunction::convertType(QualType t) {
  return CGM.getTypes().convertType(t);
}

mlir::Location CIRGenFunction::getLoc(SourceLocation sLoc) {
  // Some AST nodes might contain invalid source locations (e.g.
  // CXXDefaultArgExpr), workaround that to still get something out.
  if (sLoc.isValid()) {
    const SourceManager &sm = getContext().getSourceManager();
    PresumedLoc pLoc = sm.getPresumedLoc(sLoc);
    StringRef filename = pLoc.getFilename();
    return mlir::FileLineColLoc::get(builder.getStringAttr(filename),
                                     pLoc.getLine(), pLoc.getColumn());
  }
  // Do our best...
  assert(currSrcLoc && "expected to inherit some source location");
  return *currSrcLoc;
}

mlir::Location CIRGenFunction::getLoc(SourceRange sLoc) {
  // Some AST nodes might contain invalid source locations (e.g.
  // CXXDefaultArgExpr), workaround that to still get something out.
  if (sLoc.isValid()) {
    mlir::Location b = getLoc(sLoc.getBegin());
    mlir::Location e = getLoc(sLoc.getEnd());
    SmallVector<mlir::Location, 2> locs = {b, e};
    mlir::Attribute metadata;
    return mlir::FusedLoc::get(locs, metadata, &getMLIRContext());
  }
  if (currSrcLoc) {
    return *currSrcLoc;
  }

  // We're brave, but time to give up.
  return builder.getUnknownLoc();
}

mlir::Location CIRGenFunction::getLoc(mlir::Location lhs, mlir::Location rhs) {
  SmallVector<mlir::Location, 2> locs = {lhs, rhs};
  mlir::Attribute metadata;
  return mlir::FusedLoc::get(locs, metadata, &getMLIRContext());
}

/// Return true if the statement contains a label in it.  If
/// this statement is not executed normally, it not containing a label means
/// that we can just remove the code.
bool CIRGenFunction::ContainsLabel(const Stmt *s, bool ignoreCaseStmts) {
  // Null statement, not a label!
  if (!s)
    return false;

  // If this is a label, we have to emit the code, consider something like:
  // if (0) {  ...  foo:  bar(); }  goto foo;
  //
  // TODO: If anyone cared, we could track __label__'s, since we know that you
  // can't jump to one from outside their declared region.
  if (isa<LabelStmt>(s))
    return true;

  // If this is a case/default statement, and we haven't seen a switch, we
  // have to emit the code.
  if (isa<SwitchCase>(s) && !ignoreCaseStmts)
    return true;

  // If this is a switch statement, we want to ignore cases below it.
  if (isa<SwitchStmt>(s))
    ignoreCaseStmts = true;

  // Scan subexpressions for verboten labels.
  return std::any_of(s->child_begin(), s->child_end(),
                     [=](const Stmt *subStmt) {
                       return ContainsLabel(subStmt, ignoreCaseStmts);
                     });
}

bool CIRGenFunction::sanitizePerformTypeCheck() const {
  return SanOpts.has(SanitizerKind::Null) ||
         SanOpts.has(SanitizerKind::Alignment) ||
         SanOpts.has(SanitizerKind::ObjectSize) ||
         SanOpts.has(SanitizerKind::Vptr);
}

void CIRGenFunction::emitTypeCheck(TypeCheckKind tck, clang::SourceLocation loc,
                                   mlir::Value v, clang::QualType type,
                                   clang::CharUnits alignment,
                                   clang::SanitizerSet skippedChecks,
                                   std::optional<mlir::Value> arraySize) {
  if (!sanitizePerformTypeCheck())
    return;

  assert(false && "type check NYI");
}

/// If the specified expression does not fold
/// to a constant, or if it does but contains a label, return false.  If it
/// constant folds return true and set the folded value.
bool CIRGenFunction::ConstantFoldsToSimpleInteger(const Expr *cond,
                                                  llvm::APSInt &resultInt,
                                                  bool allowLabels) {
  // FIXME: Rename and handle conversion of other evaluatable things
  // to bool.
  Expr::EvalResult result;
  if (!cond->EvaluateAsInt(result, getContext()))
    return false; // Not foldable, not integer or not fully evaluatable.

  llvm::APSInt intValue = result.Val.getInt();
  if (!allowLabels && ContainsLabel(cond))
    return false; // Contains a label.

  resultInt = intValue;
  return true;
}

/// Determine whether the function F ends with a return stmt.
static bool endsWithReturn(const Decl *f) {
  const Stmt *body = nullptr;
  if (auto *fd = dyn_cast_or_null<FunctionDecl>(f))
    body = fd->getBody();
  else if (auto *omd = dyn_cast_or_null<ObjCMethodDecl>(f))
    llvm_unreachable("NYI");

  if (auto *cs = dyn_cast_or_null<CompoundStmt>(body)) {
    auto lastStmt = cs->body_rbegin();
    if (lastStmt != cs->body_rend())
      return isa<ReturnStmt>(*lastStmt);
  }
  return false;
}

void CIRGenFunction::emitAndUpdateRetAlloca(QualType ty, mlir::Location loc,
                                            CharUnits alignment) {

  if (ty->isVoidType()) {
    // Void type; nothing to return.
    ReturnValue = Address::invalid();

    // Count the implicit return.
    if (!endsWithReturn(CurFuncDecl))
      ++NumReturnExprs;
  } else if (CurFnInfo->getReturnInfo().getKind() ==
             cir::ABIArgInfo::Indirect) {
    // TODO(CIR): Consider this implementation in CIRtoLLVM
    llvm_unreachable("NYI");
    // TODO(CIR): Consider this implementation in CIRtoLLVM
  } else if (CurFnInfo->getReturnInfo().getKind() ==
             cir::ABIArgInfo::InAlloca) {
    llvm_unreachable("NYI");
  } else {
    auto addr = emitAlloca("__retval", ty, loc, alignment);
    FnRetAlloca = addr;
    ReturnValue = Address(addr, alignment);

    // Tell the epilog emitter to autorelease the result. We do this now so
    // that various specialized functions can suppress it during their IR -
    // generation
    if (getLangOpts().ObjCAutoRefCount)
      llvm_unreachable("NYI");
  }
}

mlir::LogicalResult CIRGenFunction::declare(const Decl *var, QualType ty,
                                            mlir::Location loc,
                                            CharUnits alignment,
                                            mlir::Value &addr, bool isParam) {
  const auto *namedVar = dyn_cast_or_null<NamedDecl>(var);
  assert(namedVar && "Needs a named decl");
  assert(!symbolTable.count(var) && "not supposed to be available just yet");

  addr = emitAlloca(namedVar->getName(), ty, loc, alignment);
  auto allocaOp = cast<cir::AllocaOp>(addr.getDefiningOp());
  if (isParam)
    allocaOp.setInitAttr(mlir::UnitAttr::get(&getMLIRContext()));
  if (ty->isReferenceType() || ty.isConstQualified())
    allocaOp.setConstantAttr(mlir::UnitAttr::get(&getMLIRContext()));

  symbolTable.insert(var, addr);
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::declare(Address addr, const Decl *var,
                                            QualType ty, mlir::Location loc,
                                            CharUnits alignment,
                                            mlir::Value &addrVal,
                                            bool isParam) {
  const auto *namedVar = dyn_cast_or_null<NamedDecl>(var);
  assert(namedVar && "Needs a named decl");
  assert(!symbolTable.count(var) && "not supposed to be available just yet");

  addrVal = addr.getPointer();
  auto allocaOp = cast<cir::AllocaOp>(addrVal.getDefiningOp());
  if (isParam)
    allocaOp.setInitAttr(mlir::UnitAttr::get(&getMLIRContext()));
  if (ty->isReferenceType() || ty.isConstQualified())
    allocaOp.setConstantAttr(mlir::UnitAttr::get(&getMLIRContext()));

  symbolTable.insert(var, addrVal);
  return mlir::success();
}

/// All scope related cleanup needed:
/// - Patching up unsolved goto's.
/// - Build all cleanup code and insert yield/returns.
void CIRGenFunction::LexicalScope::cleanup() {
  auto &builder = CGF.builder;
  auto *localScope = CGF.currLexScope;

  auto applyCleanup = [&]() {
    if (PerformCleanup) {
      // ApplyDebugLocation
      assert(!cir::MissingFeatures::generateDebugInfo());
      ForceCleanup();
    }
  };

  // Cleanup are done right before codegen resume a scope. This is where
  // objects are destroyed.
  SmallVector<mlir::Block *> retBlocks;
  for (auto *retBlock : localScope->getRetBlocks()) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(retBlock);
    retBlocks.push_back(retBlock);
    mlir::Location retLoc = localScope->getRetLoc(retBlock);
    (void)emitReturn(retLoc);
  }

  auto removeUnusedRetBlocks = [&]() {
    for (mlir::Block *retBlock : retBlocks) {
      if (!retBlock->getUses().empty())
        continue;
      retBlock->erase();
    }
  };

  auto insertCleanupAndLeave = [&](mlir::Block *insPt) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(insPt);

    // If we still don't have a cleanup block, it means that `applyCleanup`
    // below might be able to get us one.
    mlir::Block *cleanupBlock = localScope->getCleanupBlock(builder);

    // Leverage and defers to RunCleanupsScope's dtor and scope handling.
    applyCleanup();

    // If we now have one after `applyCleanup`, hook it up properly.
    if (!cleanupBlock && localScope->getCleanupBlock(builder)) {
      cleanupBlock = localScope->getCleanupBlock(builder);
      builder.create<BrOp>(insPt->back().getLoc(), cleanupBlock);
      if (!cleanupBlock->mightHaveTerminator()) {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(cleanupBlock);
        builder.create<YieldOp>(localScope->EndLoc);
      }
    }

    if (localScope->Depth == 0) {
      // TODO(cir): get rid of all this special cases once cleanups are properly
      // implemented.
      // TODO(cir): most of this code should move into emitBranchThroughCleanup
      if (localScope->getRetBlocks().size() == 1) {
        mlir::Block *retBlock = localScope->getRetBlocks()[0];
        mlir::Location loc = localScope->getRetLoc(retBlock);
        if (retBlock->getUses().empty())
          retBlock->erase();
        else {
          // Thread return block via cleanup block.
          if (cleanupBlock) {
            for (auto &blockUse : retBlock->getUses()) {
              auto brOp = dyn_cast<cir::BrOp>(blockUse.getOwner());
              brOp.setSuccessor(cleanupBlock);
            }
          }
          builder.create<BrOp>(loc, retBlock);
          return;
        }
      }
      emitImplicitReturn();
      return;
    }

    // End of any local scope != function
    // Ternary ops have to deal with matching arms for yielding types
    // and do return a value, it must do its own cir.yield insertion.
    if (!localScope->isTernary() && !insPt->mightHaveTerminator()) {
      !retVal ? builder.create<YieldOp>(localScope->EndLoc)
              : builder.create<YieldOp>(localScope->EndLoc, retVal);
    }
  };

  // If a cleanup block has been created at some point, branch to it
  // and set the insertion point to continue at the cleanup block.
  // Terminators are then inserted either in the cleanup block or
  // inline in this current block.
  auto *cleanupBlock = localScope->getCleanupBlock(builder);
  if (cleanupBlock)
    insertCleanupAndLeave(cleanupBlock);

  // Now deal with any pending block wrap up like implicit end of
  // scope.

  // If a terminator is already present in the current block, nothing
  // else to do here.
  auto *currBlock = builder.getBlock();
  if (isGlobalInit() && !currBlock)
    return;
  if (currBlock->mightHaveTerminator() && currBlock->getTerminator())
    return;

  // An empty non-entry block has nothing to offer, and since this is
  // synthetic, losing information does not affect anything.
  bool entryBlock = builder.getInsertionBlock()->isEntryBlock();
  if (!entryBlock && currBlock->empty()) {
    currBlock->erase();
    // Remove unused cleanup blocks.
    if (cleanupBlock && cleanupBlock->hasNoPredecessors())
      cleanupBlock->erase();
    // FIXME(cir): ideally we should call applyCleanup() before we
    // get into this condition and emit the proper cleanup. This is
    // needed to get nrvo to interop with dtor logic.
    PerformCleanup = false;
    removeUnusedRetBlocks();
    return;
  }

  // If there's a cleanup block, branch to it, nothing else to do.
  if (cleanupBlock) {
    builder.create<BrOp>(currBlock->back().getLoc(), cleanupBlock);
    return;
  }

  // No pre-existent cleanup block, emit cleanup code and yield/return.
  insertCleanupAndLeave(currBlock);
}

cir::ReturnOp CIRGenFunction::LexicalScope::emitReturn(mlir::Location loc) {
  auto &builder = CGF.getBuilder();

  // If we are on a coroutine, add the coro_end builtin call.
  auto fn = dyn_cast<cir::FuncOp>(CGF.CurFn);
  assert(fn && "other callables NYI");
  if (fn.getCoroutine())
    CGF.emitCoroEndBuiltinCall(loc,
                               builder.getNullPtr(builder.getVoidPtrTy(), loc));

  if (CGF.FnRetCIRTy.has_value()) {
    // If there's anything to return, load it first.
    auto val = builder.create<LoadOp>(loc, *CGF.FnRetCIRTy, *CGF.FnRetAlloca);
    return builder.create<ReturnOp>(loc, llvm::ArrayRef(val.getResult()));
  }
  return builder.create<ReturnOp>(loc);
}

void CIRGenFunction::LexicalScope::emitImplicitReturn() {
  auto &builder = CGF.getBuilder();
  auto *localScope = CGF.currLexScope;

  const auto *fd = cast<clang::FunctionDecl>(CGF.CurGD.getDecl());

  // C++11 [stmt.return]p2:
  //   Flowing off the end of a function [...] results in undefined behavior
  //   in a value-returning function.
  // C11 6.9.1p12:
  //   If the '}' that terminates a function is reached, and the value of the
  //   function call is used by the caller, the behavior is undefined.
  if (CGF.getLangOpts().CPlusPlus && !fd->hasImplicitReturnZero() &&
      !CGF.SawAsmBlock && !fd->getReturnType()->isVoidType() &&
      builder.getInsertionBlock()) {
    bool shouldEmitUnreachable = CGF.CGM.getCodeGenOpts().StrictReturn ||
                                 !CGF.CGM.MayDropFunctionReturn(
                                     fd->getASTContext(), fd->getReturnType());

    if (CGF.SanOpts.has(SanitizerKind::Return)) {
      assert(!cir::MissingFeatures::sanitizerReturn());
      llvm_unreachable("NYI");
    } else if (shouldEmitUnreachable) {
      if (CGF.CGM.getCodeGenOpts().OptimizationLevel == 0) {
        builder.create<cir::TrapOp>(localScope->EndLoc);
        builder.clearInsertionPoint();
        return;
      }
    }

    if (CGF.SanOpts.has(SanitizerKind::Return) || shouldEmitUnreachable) {
      builder.create<cir::UnreachableOp>(localScope->EndLoc);
      builder.clearInsertionPoint();
      return;
    }
  }

  (void)emitReturn(localScope->EndLoc);
}

cir::TryOp CIRGenFunction::LexicalScope::getClosestTryParent() {
  auto *scope = this;
  while (scope) {
    if (scope->isTry())
      return scope->getTry();
    scope = scope->ParentScope;
  }
  return nullptr;
}

void CIRGenFunction::finishFunction(SourceLocation endLoc) {
  // CIRGen doesn't use a BreakContinueStack or evaluates OnlySimpleReturnStmts.

  // Usually the return expression is evaluated before the cleanup
  // code.  If the function contains only a simple return statement,
  // such as a constant, the location before the cleanup code becomes
  // the last useful breakpoint in the function, because the simple
  // return expression will be evaluated after the cleanup code. To be
  // safe, set the debug location for cleanup code to the location of
  // the return statement.  Otherwise the cleanup code should be at the
  // end of the function's lexical scope.
  //
  // If there are multiple branches to the return block, the branch
  // instructions will get the location of the return statements and
  // all will be fine.
  if (auto *di = getDebugInfo())
    assert(!cir::MissingFeatures::generateDebugInfo() && "NYI");

  // Pop any cleanups that might have been associated with the
  // parameters.  Do this in whatever block we're currently in; it's
  // important to do this before we enter the return block or return
  // edges will be *really* confused.
  bool hasCleanups = EHStack.stable_begin() != PrologueCleanupDepth;
  if (hasCleanups) {
    // Make sure the line table doesn't jump back into the body for
    // the ret after it's been at EndLoc.
    if (auto *di = getDebugInfo())
      assert(!cir::MissingFeatures::generateDebugInfo() && "NYI");
    // FIXME(cir): should we clearInsertionPoint? breaks many testcases
    PopCleanupBlocks(PrologueCleanupDepth);
  }

  // Emit function epilog (to return).

  // Original LLVM codegen does EmitReturnBlock() here, CIRGen handles
  // this as part of LexicalScope instead, given CIR might have multiple
  // blocks with `cir.return`.
  if (ShouldInstrumentFunction()) {
    assert(!cir::MissingFeatures::shouldInstrumentFunction() && "NYI");
  }

  // Emit debug descriptor for function end.
  if (auto *di = getDebugInfo())
    assert(!cir::MissingFeatures::generateDebugInfo() && "NYI");

  // Reset the debug location to that of the simple 'return' expression, if any
  // rather than that of the end of the function's scope '}'.
  assert(!cir::MissingFeatures::generateDebugInfo() && "NYI");

  assert(!cir::MissingFeatures::emitFunctionEpilog() && "NYI");
  assert(!cir::MissingFeatures::emitEndEHSpec() && "NYI");

  assert(EHStack.empty() && "did not remove all scopes from cleanup stack!");

  // If someone did an indirect goto, emit the indirect goto block at the end of
  // the function.
  assert(!cir::MissingFeatures::indirectBranch() && "NYI");

  // If some of our locals escaped, insert a call to llvm.localescape in the
  // entry block.
  assert(!cir::MissingFeatures::escapedLocals() && "NYI");

  // If someone took the address of a label but never did an indirect goto, we
  // made a zero entry PHI node, which is illegal, zap it now.
  assert(!cir::MissingFeatures::indirectBranch() && "NYI");

  // CIRGen doesn't need to emit EHResumeBlock, TerminateLandingPad,
  // TerminateHandler, UnreachableBlock, TerminateFunclets, NormalCleanupDest
  // here because the basic blocks aren't shared.

  assert(!cir::MissingFeatures::emitDeclMetadata() && "NYI");
  assert(!cir::MissingFeatures::deferredReplacements() && "NYI");

  // Add the min-legal-vector-width attribute. This contains the max width from:
  // 1. min-vector-width attribute used in the source program.
  // 2. Any builtins used that have a vector width specified.
  // 3. Values passed in and out of inline assembly.
  // 4. Width of vector arguments and return types for this function.
  // 5. Width of vector arguments and return types for functions called by
  // this function.
  assert(!cir::MissingFeatures::minLegalVectorWidthAttr() && "NYI");

  // Add vscale_range attribute if appropriate.
  assert(!cir::MissingFeatures::vscaleRangeAttr() && "NYI");

  // In traditional LLVM codegen, if clang generated an unreachable return
  // block, it'd be deleted now. Same for unused ret allocas from ReturnValue
}

static void eraseEmptyAndUnusedBlocks(cir::FuncOp fnOp) {
  // Remove any left over blocks that are unrecheable and empty, since they do
  // not represent unrecheable code useful for warnings nor anything deemed
  // useful in general.
  SmallVector<mlir::Block *> blocksToDelete;
  for (auto &blk : fnOp.getBlocks()) {
    if (!blk.empty() || !blk.getUses().empty())
      continue;
    blocksToDelete.push_back(&blk);
  }
  for (auto *b : blocksToDelete)
    b->erase();
}

cir::FuncOp CIRGenFunction::generateCode(clang::GlobalDecl gd, cir::FuncOp fn,
                                         const CIRGenFunctionInfo &fnInfo) {
  assert(fn && "generating code for a null function");
  const auto *const fd = cast<FunctionDecl>(gd.getDecl());
  CurGD = gd;

  FnRetQualTy = fd->getReturnType();
  if (!FnRetQualTy->isVoidType())
    FnRetCIRTy = convertType(FnRetQualTy);

  FunctionArgList args;
  QualType resTy = buildFunctionArgList(gd, args);

  if (fd->isInlineBuiltinDeclaration()) {
    llvm_unreachable("NYI");
  } else {
    // Detect the unusual situation where an inline version is shadowed by a
    // non-inline version. In that case we should pick the external one
    // everywhere. That's GCC behavior too. Unfortunately, I cannot find a way
    // to detect that situation before we reach codegen, so do some late
    // replacement.
    for (const auto *pd = fd->getPreviousDecl(); pd;
         pd = pd->getPreviousDecl()) {
      if (LLVM_UNLIKELY(pd->isInlineBuiltinDeclaration())) {
        llvm_unreachable("NYI");
      }
    }
  }

  // Check if we should generate debug info for this function.
  if (fd->hasAttr<NoDebugAttr>()) {
    assert(!cir::MissingFeatures::noDebugInfo());
  }

  // The function might not have a body if we're generating thunks for a
  // function declaration.
  SourceRange bodyRange;
  if (Stmt *body = fd->getBody())
    bodyRange = body->getSourceRange();
  else
    bodyRange = fd->getLocation();
  // TODO: CurEHLocation

  // Use the location of the start of the function to determine where the
  // function definition is located. By default we use the location of the
  // declaration as the location for the subprogram. A function may lack a
  // declaration in the source code if it is created by code gen. (examples:
  // _GLOBAL__I_a, __cxx_global_array_dtor, thunk).
  SourceLocation loc = fd->getLocation();

  // If this is a function specialization then use the pattern body as the
  // location for the function.
  if (const auto *specDecl = fd->getTemplateInstantiationPattern())
    if (specDecl->hasBody(specDecl))
      loc = specDecl->getLocation();

  Stmt *body = fd->getBody();

  if (body) {
    // LLVM codegen: Coroutines always emit lifetime markers
    // Hide this under request for lifetime emission so that we can write
    // tests when the time comes, but CIR should be intrinsically scope
    // accurate, so no need to tie coroutines to such markers.
    if (isa<CoroutineBodyStmt>(body))
      assert(!cir::MissingFeatures::shouldEmitLifetimeMarkers() && "NYI");

    // Initialize helper which will detect jumps which can cause invalid
    // lifetime markers.
    if (ShouldEmitLifetimeMarkers)
      assert(!cir::MissingFeatures::shouldEmitLifetimeMarkers() && "NYI");
  }

  // Create a scope in the symbol table to hold variable declarations.
  SymTableScopeTy varScope(symbolTable);
  // Compiler synthetized functions might have invalid slocs...
  auto bSrcLoc = fd->getBody()->getBeginLoc();
  auto eSrcLoc = fd->getBody()->getEndLoc();
  auto unknownLoc = builder.getUnknownLoc();

  auto fnBeginLoc = bSrcLoc.isValid() ? getLoc(bSrcLoc) : unknownLoc;
  auto fnEndLoc = eSrcLoc.isValid() ? getLoc(eSrcLoc) : unknownLoc;
  const auto fusedLoc =
      mlir::FusedLoc::get(&getMLIRContext(), {fnBeginLoc, fnEndLoc});
  SourceLocRAIIObject fnLoc{*this, loc.isValid() ? getLoc(loc) : unknownLoc};

  assert(fn.isDeclaration() && "Function already has body?");
  mlir::Block *entryBb = fn.addEntryBlock();
  builder.setInsertionPointToStart(entryBb);
  {
    // Initialize lexical scope information.
    LexicalScope lexScope{*this, fusedLoc, entryBb};

    // Emit the standard function prologue.
    StartFunction(gd, resTy, fn, fnInfo, args, loc, bodyRange.getBegin());

    // Save parameters for coroutine function.
    if (body && isa_and_nonnull<CoroutineBodyStmt>(body))
      llvm::append_range(FnArgs, fd->parameters());

    // Ensure that the function adheres to the forward progress guarantee, which
    // is required by certain optimizations.
    // In C++11 and up, the attribute will be removed if the body contains a
    // trivial empty loop.
    if (cir::MissingFeatures::mustProgress())
      llvm_unreachable("NYI");

    // Generate the body of the function.
    // TODO: PGO.assignRegionCounters
    assert(!cir::MissingFeatures::shouldInstrumentFunction());
    if (isa<CXXDestructorDecl>(fd))
      emitDestructorBody(args);
    else if (isa<CXXConstructorDecl>(fd))
      emitConstructorBody(args);
    else if (getLangOpts().CUDA && !getLangOpts().CUDAIsDevice &&
             fd->hasAttr<CUDAGlobalAttr>())
      CGM.getCUDARuntime().emitDeviceStub(*this, fn, args);
    else if (isa<CXXMethodDecl>(fd) &&
             cast<CXXMethodDecl>(fd)->isLambdaStaticInvoker()) {
      // The lambda static invoker function is special, because it forwards or
      // clones the body of the function call operator (but is actually
      // static).
      emitLambdaStaticInvokeBody(cast<CXXMethodDecl>(fd));
    } else if (fd->isDefaulted() && isa<CXXMethodDecl>(fd) &&
               (cast<CXXMethodDecl>(fd)->isCopyAssignmentOperator() ||
                cast<CXXMethodDecl>(fd)->isMoveAssignmentOperator())) {
      // Implicit copy-assignment gets the same special treatment as implicit
      // copy-constructors.
      emitImplicitAssignmentOperatorBody(args);
    } else if (body) {
      if (mlir::failed(emitFunctionBody(body))) {
        fn.erase();
        return nullptr;
      }
    } else
      llvm_unreachable("no definition for emitted function");

    assert(builder.getInsertionBlock() && "Should be valid");

    if (mlir::failed(fn.verifyBody()))
      return nullptr;

    // Emit the standard function epilogue.
    finishFunction(bodyRange.getEnd());

    // If we haven't marked the function nothrow through other means, do a quick
    // pass now to see if we can.
    assert(!cir::MissingFeatures::tryMarkNoThrow());
  }

  eraseEmptyAndUnusedBlocks(fn);
  return fn;
}

mlir::Value CIRGenFunction::createLoad(const VarDecl *vd, const char *name) {
  auto addr = GetAddrOfLocalVar(vd);
  return builder.create<LoadOp>(getLoc(vd->getLocation()),
                                addr.getElementType(), addr.getPointer());
}

void CIRGenFunction::emitConstructorBody(FunctionArgList &args) {
  assert(!cir::MissingFeatures::emitAsanPrologueOrEpilogue());
  const auto *ctor = cast<CXXConstructorDecl>(CurGD.getDecl());
  auto ctorType = CurGD.getCtorType();

  assert((CGM.getTarget().getCXXABI().hasConstructorVariants() ||
          ctorType == Ctor_Complete) &&
         "can only generate complete ctor for this ABI");

  // Before we go any further, try the complete->base constructor delegation
  // optimization.
  if (ctorType == Ctor_Complete && IsConstructorDelegationValid(ctor) &&
      CGM.getTarget().getCXXABI().hasConstructorVariants()) {
    emitDelegateCXXConstructorCall(ctor, Ctor_Base, args, ctor->getEndLoc());
    return;
  }

  const FunctionDecl *definition = nullptr;
  Stmt *body = ctor->getBody(definition);
  assert(definition == ctor && "emitting wrong constructor body");

  // Enter the function-try-block before the constructor prologue if
  // applicable.
  bool isTryBody = (isa_and_nonnull<CXXTryStmt>(body));
  if (isTryBody)
    llvm_unreachable("NYI");

  // TODO: incrementProfileCounter

  // TODO: RunClenaupCcope RunCleanups(*this);

  // TODO: in restricted cases, we can emit the vbase initializers of a
  // complete ctor and then delegate to the base ctor.

  // Emit the constructor prologue, i.e. the base and member initializers.
  emitCtorPrologue(ctor, ctorType, args);

  // Emit the body of the statement.
  if (isTryBody)
    llvm_unreachable("NYI");
  else {
    // TODO: propagate this result via mlir::logical result. Just unreachable
    // now just to have it handled.
    if (mlir::failed(emitStmt(body, true)))
      llvm_unreachable("NYI");
  }

  // Emit any cleanup blocks associated with the member or base initializers,
  // which inlcudes (along the exceptional path) the destructors for those
  // members and bases that were fully constructed.
  /// TODO: RunCleanups.ForceCleanup();

  if (isTryBody)
    llvm_unreachable("NYI");
}

/// Given a value of type T* that may not be to a complete object, construct
/// an l-vlaue withi the natural pointee alignment of T.
LValue CIRGenFunction::MakeNaturalAlignPointeeAddrLValue(mlir::Value val,
                                                         QualType ty) {
  // FIXME(cir): is it safe to assume Op->getResult(0) is valid? Perhaps
  // assert on the result type first.
  LValueBaseInfo baseInfo;
  TBAAAccessInfo tbaaInfo;
  CharUnits align = CGM.getNaturalTypeAlignment(ty, &baseInfo, &tbaaInfo,
                                                /* for PointeeType= */ true);
  return makeAddrLValue(Address(val, align), ty, baseInfo, tbaaInfo);
}

LValue CIRGenFunction::MakeNaturalAlignAddrLValue(mlir::Value val,
                                                  QualType ty) {
  LValueBaseInfo baseInfo;
  TBAAAccessInfo tbaaInfo;
  CharUnits alignment = CGM.getNaturalTypeAlignment(ty, &baseInfo, &tbaaInfo);
  Address addr(val, convertTypeForMem(ty), alignment);
  return LValue::makeAddr(addr, ty, getContext(), baseInfo, tbaaInfo);
}

// Map the LangOption for exception behavior into the corresponding enum in
// the IR.
static cir::fp::ExceptionBehavior
toConstrainedExceptMd(LangOptions::FPExceptionModeKind kind) {
  switch (kind) {
  case LangOptions::FPE_Ignore:
    return cir::fp::ebIgnore;
  case LangOptions::FPE_MayTrap:
    return cir::fp::ebMayTrap;
  case LangOptions::FPE_Strict:
    return cir::fp::ebStrict;
  default:
    llvm_unreachable("Unsupported FP Exception Behavior");
  }
}

bool CIRGenFunction::ShouldSkipSanitizerInstrumentation() {
  if (!CurFuncDecl)
    return false;
  return CurFuncDecl->hasAttr<DisableSanitizerInstrumentationAttr>();
}

/// Return true if the current function should be instrumented with XRay nop
/// sleds.
bool CIRGenFunction::ShouldXRayInstrumentFunction() const {
  return CGM.getCodeGenOpts().XRayInstrumentFunctions;
}

static bool matchesStlAllocatorFn(const Decl *d, const ASTContext &astContext) {
  auto *md = dyn_cast_or_null<CXXMethodDecl>(d);
  if (!md || !md->getDeclName().getAsIdentifierInfo() ||
      !md->getDeclName().getAsIdentifierInfo()->isStr("allocate") ||
      (md->getNumParams() != 1 && md->getNumParams() != 2))
    return false;

  if (md->parameters()[0]->getType().getCanonicalType() !=
      astContext.getSizeType())
    return false;

  if (md->getNumParams() == 2) {
    auto *pt = md->parameters()[1]->getType()->getAs<clang::PointerType>();
    if (!pt || !pt->isVoidPointerType() ||
        !pt->getPointeeType().isConstQualified())
      return false;
  }

  return true;
}

/// TODO: this should live in `emitFunctionProlog`
/// An argument came in as a promoted argument; demote it back to its
/// declared type.
static mlir::Value emitArgumentDemotion(CIRGenFunction &cgf, const VarDecl *var,
                                        mlir::Value value) {
  mlir::Type ty = cgf.convertType(var->getType());

  // This can happen with promotions that actually don't change the
  // underlying type, like the enum promotions.
  if (value.getType() == ty)
    return value;

  assert((isa<cir::IntType>(ty) || cir::isAnyFloatingPointType(ty)) &&
         "unexpected promotion type");

  if (isa<cir::IntType>(ty))
    return cgf.getBuilder().CIRBaseBuilderTy::createIntCast(value, ty);

  return cgf.getBuilder().CIRBaseBuilderTy::createCast(cir::CastKind::floating,
                                                       value, ty);
}

void CIRGenFunction::StartFunction(GlobalDecl gd, QualType retTy,
                                   cir::FuncOp Fn,
                                   const CIRGenFunctionInfo &fnInfo,
                                   const FunctionArgList &args,
                                   SourceLocation Loc,
                                   SourceLocation startLoc) {
  assert(!CurFn &&
         "Do not use a CIRGenFunction object for more than one function");

  const auto *d = gd.getDecl();

  DidCallStackSave = false;
  CurCodeDecl = d;
  const auto *fd = dyn_cast_or_null<FunctionDecl>(d);
  if (fd && fd->usesSEHTry())
    CurSEHParent = gd;
  CurFuncDecl = (d ? d->getNonClosureContext() : nullptr);
  FnRetTy = retTy;
  CurFn = Fn;
  CurFnInfo = &fnInfo;

  // If this function is ignored for any of the enabled sanitizers, disable
  // the sanitizer for the function.
  do {
#define SANITIZER(NAME, ID)                                                    \
  if (SanOpts.empty())                                                         \
    break;                                                                     \
  if (SanOpts.has(SanitizerKind::ID))                                          \
    if (CGM.isInNoSanitizeList(SanitizerKind::ID, Fn, Loc))                    \
      SanOpts.set(SanitizerKind::ID, false);

#include "clang/Basic/Sanitizers.def"
#undef SANITIZER
  } while (false);

  if (d) {
    const bool sanitizeBounds = SanOpts.hasOneOf(SanitizerKind::Bounds);
    SanitizerMask noSanitizeMask;
    bool noSanitizeCoverage = false;

    for (auto *attr : d->specific_attrs<NoSanitizeAttr>()) {
      noSanitizeMask |= attr->getMask();
      // SanitizeCoverage is not handled by SanOpts.
      if (attr->hasCoverage())
        noSanitizeCoverage = true;
    }

    // Apply the no_sanitize* attributes to SanOpts.
    SanOpts.Mask &= ~noSanitizeMask;
    if (noSanitizeMask & SanitizerKind::Address)
      SanOpts.set(SanitizerKind::KernelAddress, false);
    if (noSanitizeMask & SanitizerKind::KernelAddress)
      SanOpts.set(SanitizerKind::Address, false);
    if (noSanitizeMask & SanitizerKind::HWAddress)
      SanOpts.set(SanitizerKind::KernelHWAddress, false);
    if (noSanitizeMask & SanitizerKind::KernelHWAddress)
      SanOpts.set(SanitizerKind::HWAddress, false);

    // TODO(cir): set llvm::Attribute::NoSanitizeBounds
    if (sanitizeBounds && !SanOpts.hasOneOf(SanitizerKind::Bounds))
      assert(!cir::MissingFeatures::sanitizeOther());

    // TODO(cir): set llvm::Attribute::NoSanitizeCoverage
    if (noSanitizeCoverage && CGM.getCodeGenOpts().hasSanitizeCoverage())
      assert(!cir::MissingFeatures::sanitizeOther());

    // Some passes need the non-negated no_sanitize attribute. Pass them on.
    if (CGM.getCodeGenOpts().hasSanitizeBinaryMetadata()) {
      // TODO(cir): set no_sanitize_thread
      if (noSanitizeMask & SanitizerKind::Thread)
        assert(!cir::MissingFeatures::sanitizeOther());
    }
  }

  if (ShouldSkipSanitizerInstrumentation()) {
    assert(!cir::MissingFeatures::sanitizeOther());
  } else {
    // Apply sanitizer attributes to the function.
    if (SanOpts.hasOneOf(SanitizerKind::Address | SanitizerKind::KernelAddress))
      assert(!cir::MissingFeatures::sanitizeOther());
    if (SanOpts.hasOneOf(SanitizerKind::HWAddress |
                         SanitizerKind::KernelHWAddress))
      assert(!cir::MissingFeatures::sanitizeOther());
    if (SanOpts.has(SanitizerKind::MemtagStack))
      assert(!cir::MissingFeatures::sanitizeOther());
    if (SanOpts.has(SanitizerKind::Thread))
      assert(!cir::MissingFeatures::sanitizeOther());
    if (SanOpts.has(SanitizerKind::NumericalStability))
      assert(!cir::MissingFeatures::sanitizeOther());
    if (SanOpts.hasOneOf(SanitizerKind::Memory | SanitizerKind::KernelMemory))
      assert(!cir::MissingFeatures::sanitizeOther());
  }
  if (SanOpts.has(SanitizerKind::SafeStack))
    assert(!cir::MissingFeatures::sanitizeOther());
  if (SanOpts.has(SanitizerKind::ShadowCallStack))
    assert(!cir::MissingFeatures::sanitizeOther());

  if (SanOpts.has(SanitizerKind::Realtime))
    llvm_unreachable("NYI");

  // Apply fuzzing attribute to the function.
  if (SanOpts.hasOneOf(SanitizerKind::Fuzzer | SanitizerKind::FuzzerNoLink))
    assert(!cir::MissingFeatures::sanitizeOther());

  // Ignore TSan memory acesses from within ObjC/ObjC++ dealloc, initialize,
  // .cxx_destruct, __destroy_helper_block_ and all of their calees at run time.
  if (SanOpts.has(SanitizerKind::Thread)) {
    if (const auto *omd = dyn_cast_or_null<ObjCMethodDecl>(d)) {
      llvm_unreachable("NYI");
    }
  }

  // Ignore unrelated casts in STL allocate() since the allocator must cast
  // from void* to T* before object initialization completes. Don't match on the
  // namespace because not all allocators are in std::
  if (d && SanOpts.has(SanitizerKind::CFIUnrelatedCast)) {
    if (matchesStlAllocatorFn(d, getContext()))
      SanOpts.Mask &= ~SanitizerKind::CFIUnrelatedCast;
  }

  // Ignore null checks in coroutine functions since the coroutines passes
  // are not aware of how to move the extra UBSan instructions across the split
  // coroutine boundaries.
  if (d && SanOpts.has(SanitizerKind::Null))
    if (fd && fd->getBody() &&
        fd->getBody()->getStmtClass() == Stmt::CoroutineBodyStmtClass)
      SanOpts.Mask &= ~SanitizerKind::Null;

  // Add pointer authentication attriburtes.
  const CodeGenOptions &codeGenOptions = CGM.getCodeGenOpts();
  if (codeGenOptions.PointerAuth.ReturnAddresses)
    llvm_unreachable("NYI");
  if (codeGenOptions.PointerAuth.FunctionPointers)
    llvm_unreachable("NYI");
  if (codeGenOptions.PointerAuth.AuthTraps)
    llvm_unreachable("NYI");
  if (codeGenOptions.PointerAuth.IndirectGotos)
    llvm_unreachable("NYI");

  // Apply xray attributes to the function (as a string, for now)
  if (const auto *xRayAttr = d ? d->getAttr<XRayInstrumentAttr>() : nullptr) {
    assert(!cir::MissingFeatures::xray());
  } else {
    assert(!cir::MissingFeatures::xray());
  }

  if (ShouldXRayInstrumentFunction()) {
    assert(!cir::MissingFeatures::xray());
  }

  if (CGM.getCodeGenOpts().getProfileInstr() != CodeGenOptions::ProfileNone) {
    assert(!cir::MissingFeatures::getProfileCount());
  }

  unsigned count, offset;
  if (const auto *attr =
          d ? d->getAttr<PatchableFunctionEntryAttr>() : nullptr) {
    llvm_unreachable("NYI");
  } else {
    count = CGM.getCodeGenOpts().PatchableFunctionEntryCount;
    offset = CGM.getCodeGenOpts().PatchableFunctionEntryOffset;
  }
  if (count && offset <= count) {
    llvm_unreachable("NYI");
  }
  // Instruct that functions for COFF/CodeView targets should start with a
  // pathable instruction, but only on x86/x64. Don't forward this to ARM/ARM64
  // backends as they don't need it -- instructions on these architectures are
  // always automatically patachable at runtime.
  if (CGM.getCodeGenOpts().HotPatch &&
      getContext().getTargetInfo().getTriple().isX86() &&
      getContext().getTargetInfo().getTriple().getEnvironment() !=
          llvm::Triple::CODE16)
    llvm_unreachable("NYI");

  // Add no-jump-tables value.
  if (CGM.getCodeGenOpts().NoUseJumpTables)
    llvm_unreachable("NYI");

  // Add no-inline-line-tables value.
  if (CGM.getCodeGenOpts().NoInlineLineTables)
    llvm_unreachable("NYI");

  // Add profile-sample-accurate value.
  if (CGM.getCodeGenOpts().ProfileSampleAccurate)
    llvm_unreachable("NYI");

  if (!CGM.getCodeGenOpts().SampleProfileFile.empty())
    llvm_unreachable("NYI");

  if (d && d->hasAttr<CFICanonicalJumpTableAttr>())
    llvm_unreachable("NYI");

  if (d && d->hasAttr<NoProfileFunctionAttr>())
    llvm_unreachable("NYI");

  if (d && d->hasAttr<HybridPatchableAttr>())
    llvm_unreachable("NYI");

  if (d) {
    // Funciton attribiutes take precedence over command line flags.
    if ([[maybe_unused]] auto *a = d->getAttr<FunctionReturnThunksAttr>()) {
      llvm_unreachable("NYI");
    } else if (CGM.getCodeGenOpts().FunctionReturnThunks)
      llvm_unreachable("NYI");
  }

  if (fd && (getLangOpts().OpenCL ||
             ((getLangOpts().HIP || getLangOpts().OffloadViaLLVM) &&
              getLangOpts().CUDAIsDevice))) {
    // Add metadata for a kernel function.
    emitKernelMetadata(fd, Fn);
  }

  if (fd && fd->hasAttr<ClspvLibclcBuiltinAttr>()) {
    llvm_unreachable("NYI");
  }

  // If we are checking function types, emit a function type signature as
  // prologue data.
  if (fd && getLangOpts().CPlusPlus && SanOpts.has(SanitizerKind::Function)) {
    llvm_unreachable("NYI");
  }

  // If we're checking nullability, we need to know whether we can check the
  // return value. Initialize the falg to 'true' and refine it in
  // emitParmDecl.
  if (SanOpts.has(SanitizerKind::NullabilityReturn)) {
    llvm_unreachable("NYI");
  }

  // If we're in C++ mode and the function name is "main", it is guaranteed to
  // be norecurse by the standard (3.6.1.3 "The function main shall not be
  // used within a program").
  //
  // OpenCL C 2.0 v2.2-11 s6.9.i:
  //     Recursion is not supported.
  //
  // SYCL v1.2.1 s3.10:
  //     kernels cannot include RTTI information, exception cases, recursive
  //     code, virtual functions or make use of C++ libraries that are not
  //     compiled for the device.
  if (fd &&
      ((getLangOpts().CPlusPlus && fd->isMain()) || getLangOpts().OpenCL ||
       getLangOpts().SYCLIsDevice |
           (getLangOpts().CUDA && fd->hasAttr<CUDAGlobalAttr>()))) {
    // TODO: support norecurse attr
  }

  llvm::RoundingMode rm = getLangOpts().getDefaultRoundingMode();
  cir::fp::ExceptionBehavior fpExceptionBehavior =
      toConstrainedExceptMd(getLangOpts().getDefaultExceptionMode());
  builder.setDefaultConstrainedRounding(rm);
  builder.setDefaultConstrainedExcept(fpExceptionBehavior);
  if ((fd && (fd->UsesFPIntrin() || fd->hasAttr<StrictFPAttr>())) ||
      (!fd && (fpExceptionBehavior != cir::fp::ebIgnore ||
               rm != llvm::RoundingMode::NearestTiesToEven))) {
    llvm_unreachable("NYI");
  }

  if (cir::MissingFeatures::stackrealign())
    llvm_unreachable("NYI");

  if (fd && fd->isMain() && cir::MissingFeatures::zerocallusedregs())
    llvm_unreachable("NYI");

  // CIRGen has its own logic for entry blocks, usually per operation region.
  mlir::Block *retBlock = currLexScope->getOrCreateRetBlock(*this, getLoc(Loc));
  // returnBlock handles per region getJumpDestInCurrentScope LLVM traditional
  // codegen logic.
  (void)returnBlock(retBlock);

  mlir::Block *entryBb = &Fn.getBlocks().front();

  if (cir::MissingFeatures::requiresReturnValueCheck())
    llvm_unreachable("NYI");

  if (getDebugInfo()) {
    llvm_unreachable("NYI");
  }

  if (ShouldInstrumentFunction()) {
    llvm_unreachable("NYI");
  }

  // Since emitting the mcount call here impacts optimizations such as
  // function inlining, we just add an attribute to insert a mcount call in
  // backend. The attribute "counting-function" is set to mcount function name
  // which is architecture dependent.
  if (CGM.getCodeGenOpts().InstrumentForProfiling) {
    llvm_unreachable("NYI");
  }

  if (CGM.getCodeGenOpts().PackedStack) {
    llvm_unreachable("NYI");
  }

  if (CGM.getCodeGenOpts().WarnStackSize != UINT_MAX) {
    llvm_unreachable("NYI");
  }

  assert(!cir::MissingFeatures::emitStartEHSpec() && "NYI");
  PrologueCleanupDepth = EHStack.stable_begin();

  // Emit OpenMP specific initialization of the device functions.
  if (getLangOpts().OpenMP && CurCodeDecl)
    CGM.getOpenMPRuntime().emitFunctionProlog(*this, CurCodeDecl);

  if (fd && getLangOpts().HLSL) {
    // Handle emitting HLSL entry functions.
    if (fd->hasAttr<HLSLShaderAttr>()) {
      llvm_unreachable("NYI");
    }
    llvm_unreachable("NYI");
  }

  // TODO: emitFunctionProlog

  {
    // Set the insertion point in the builder to the beginning of the
    // function body, it will be used throughout the codegen to create
    // operations in this function.
    builder.setInsertionPointToStart(entryBb);

    // TODO: this should live in `emitFunctionProlog
    // Declare all the function arguments in the symbol table.
    for (const auto nameValue : llvm::zip(args, entryBb->getArguments())) {
      auto *paramVar = std::get<0>(nameValue);
      mlir::Value paramVal = std::get<1>(nameValue);
      auto alignment = getContext().getDeclAlign(paramVar);
      auto paramLoc = getLoc(paramVar->getSourceRange());
      paramVal.setLoc(paramLoc);

      mlir::Value addr;
      if (failed(declare(paramVar, paramVar->getType(), paramLoc, alignment,
                         addr, true /*param*/)))
        return;

      auto address = Address(addr, alignment);
      setAddrOfLocalVar(paramVar, address);

      // TODO: this should live in `emitFunctionProlog`
      bool isPromoted = isa<ParmVarDecl>(paramVar) &&
                        cast<ParmVarDecl>(paramVar)->isKNRPromoted();
      assert(!cir::MissingFeatures::constructABIArgDirectExtend());
      if (isPromoted)
        paramVal = emitArgumentDemotion(*this, paramVar, paramVal);

      // Location of the store to the param storage tracked as beginning of
      // the function body.
      auto fnBodyBegin = getLoc(fd->getBody()->getBeginLoc());
      builder.CIRBaseBuilderTy::createStore(fnBodyBegin, paramVal, addr);
    }
    assert(builder.getInsertionBlock() && "Should be valid");

    auto fnEndLoc = getLoc(fd->getBody()->getEndLoc());

    // When the current function is not void, create an address to store the
    // result value.
    if (FnRetCIRTy.has_value())
      emitAndUpdateRetAlloca(FnRetQualTy, fnEndLoc,
                             CGM.getNaturalTypeAlignment(FnRetQualTy));
  }

  if (isa_and_nonnull<CXXMethodDecl>(d) &&
      cast<CXXMethodDecl>(d)->isInstance()) {
    CGM.getCXXABI().emitInstanceFunctionProlog(Loc, *this);

    const auto *md = cast<CXXMethodDecl>(d);
    if (md->getParent()->isLambda() && md->getOverloadedOperator() == OO_Call) {
      // We're in a lambda.
      auto fn = dyn_cast<cir::FuncOp>(CurFn);
      assert(fn && "other callables NYI");
      fn.setLambdaAttr(mlir::UnitAttr::get(&getMLIRContext()));

      // Figure out the captures.
      md->getParent()->getCaptureFields(LambdaCaptureFields,
                                        LambdaThisCaptureField);
      if (LambdaThisCaptureField) {
        // If the lambda captures the object referred to by '*this' - either by
        // value or by reference, make sure CXXThisValue points to the correct
        // object.

        // Get the lvalue for the field (which is a copy of the enclosing object
        // or contains the address of the enclosing object).
        LValue thisFieldLValue =
            emitLValueForLambdaField(LambdaThisCaptureField);
        if (!LambdaThisCaptureField->getType()->isPointerType()) {
          // If the enclosing object was captured by value, just use its
          // address. Sign this pointer.
          CXXThisValue = thisFieldLValue.getPointer();
        } else {
          // Load the lvalue pointed to by the field, since '*this' was captured
          // by reference.
          CXXThisValue = emitLoadOfLValue(thisFieldLValue, SourceLocation())
                             .getScalarVal();
        }
      }
      for (auto *fd : md->getParent()->fields()) {
        if (fd->hasCapturedVLAType()) {
          llvm_unreachable("NYI");
        }
      }

    } else {
      // Not in a lambda; just use 'this' from the method.
      // FIXME: Should we generate a new load for each use of 'this'? The fast
      // register allocator would be happier...
      CXXThisValue = CXXABIThisValue;
    }

    // Check the 'this' pointer once per function, if it's available
    if (CXXABIThisValue) {
      SanitizerSet skippedChecks;
      skippedChecks.set(SanitizerKind::ObjectSize, true);
      QualType thisTy = md->getThisType();
      (void)thisTy;

      // If this is the call operator of a lambda with no capture-default, it
      // may have a staic invoker function, which may call this operator with
      // a null 'this' pointer.
      if (isLambdaCallOperator(md) &&
          md->getParent()->getLambdaCaptureDefault() == LCD_None)
        skippedChecks.set(SanitizerKind::Null, true);

      assert(!cir::MissingFeatures::emitTypeCheck() && "NYI");
    }
  }

  // If any of the arguments have a variably modified type, make sure to emit
  // the type size, but only if the function is not naked. Naked functions have
  // no prolog to run this evaluation.
  if (!fd || !fd->hasAttr<NakedAttr>()) {
    for (const VarDecl *vd : args) {
      // Dig out the type as written from ParmVarDecls; it's unclear whether the
      // standard (C99 6.9.1p10) requires this, but we're following the
      // precedent set by gcc.
      QualType ty;
      if (const auto *pvd = dyn_cast<ParmVarDecl>(vd))
        ty = pvd->getOriginalType();
      else
        ty = vd->getType();

      if (ty->isVariablyModifiedType())
        emitVariablyModifiedType(ty);
    }
  }
  // Emit a location at the end of the prologue.
  if (getDebugInfo())
    llvm_unreachable("NYI");
  // TODO: Do we need to handle this in two places like we do with
  // target-features/target-cpu?
  if (CurFuncDecl)
    if ([[maybe_unused]] const auto *vecWidth =
            CurFuncDecl->getAttr<MinVectorWidthAttr>())
      llvm_unreachable("NYI");

  if (CGM.shouldEmitConvergenceTokens())
    llvm_unreachable("NYI");
}

/// Return true if the current function should be instrumented with
/// __cyg_profile_func_* calls
bool CIRGenFunction::ShouldInstrumentFunction() {
  if (!CGM.getCodeGenOpts().InstrumentFunctions &&
      !CGM.getCodeGenOpts().InstrumentFunctionsAfterInlining &&
      !CGM.getCodeGenOpts().InstrumentFunctionEntryBare)
    return false;

  llvm_unreachable("NYI");
}

mlir::LogicalResult CIRGenFunction::emitFunctionBody(const clang::Stmt *body) {
  // TODO: incrementProfileCounter(Body);

  // We start with function level scope for variables.
  SymTableScopeTy varScope(symbolTable);

  auto result = mlir::LogicalResult::success();
  if (const CompoundStmt *s = dyn_cast<CompoundStmt>(body))
    emitCompoundStmtWithoutScope(*s);
  else
    result = emitStmt(body, /*useCurrentScope*/ true);

  // This is checked after emitting the function body so we know if there are
  // any permitted infinite loops.
  // TODO: if (checkIfFunctionMustProgress())
  // CurFn->addFnAttr(llvm::Attribute::MustProgress);
  return result;
}

clang::QualType CIRGenFunction::buildFunctionArgList(clang::GlobalDecl gd,
                                                     FunctionArgList &args) {
  const auto *fd = cast<FunctionDecl>(gd.getDecl());
  QualType resTy = fd->getReturnType();

  const auto *md = dyn_cast<CXXMethodDecl>(fd);
  if (md && md->isInstance()) {
    if (CGM.getCXXABI().HasThisReturn(gd))
      llvm_unreachable("NYI");
    else if (CGM.getCXXABI().hasMostDerivedReturn(gd))
      llvm_unreachable("NYI");
    CGM.getCXXABI().buildThisParam(*this, args);
  }

  // The base version of an inheriting constructor whose constructed base is a
  // virtual base is not passed any arguments (because it doesn't actually
  // call the inherited constructor).
  bool passedParams = true;
  if (const auto *cd = dyn_cast<CXXConstructorDecl>(fd))
    if (auto inherited = cd->getInheritedConstructor())
      passedParams =
          getTypes().inheritingCtorHasParams(inherited, gd.getCtorType());

  if (passedParams) {
    for (auto *param : fd->parameters()) {
      args.push_back(param);
      if (!param->hasAttr<PassObjectSizeAttr>())
        continue;

      auto *implicit = ImplicitParamDecl::Create(
          getContext(), param->getDeclContext(), param->getLocation(),
          /*Id=*/nullptr, getContext().getSizeType(), ImplicitParamKind::Other);
      SizeArguments[param] = implicit;
      args.push_back(implicit);
    }
  }

  if (md && (isa<CXXConstructorDecl>(md) || isa<CXXDestructorDecl>(md)))
    CGM.getCXXABI().addImplicitStructorParams(*this, resTy, args);

  return resTy;
}

static std::string getVersionedTmpName(llvm::StringRef name, unsigned cnt) {
  SmallString<256> buffer;
  llvm::raw_svector_ostream out(buffer);
  out << name << cnt;
  return std::string(out.str());
}

std::string CIRGenFunction::getCounterAggTmpAsString() {
  return getVersionedTmpName("agg.tmp", CounterAggTmp++);
}

std::string CIRGenFunction::getCounterRefTmpAsString() {
  return getVersionedTmpName("ref.tmp", CounterRefTmp++);
}

void CIRGenFunction::emitNullInitialization(mlir::Location loc, Address destPtr,
                                            QualType ty) {
  // Ignore empty classes in C++.
  if (getLangOpts().CPlusPlus) {
    if (const RecordType *rt = ty->getAs<RecordType>()) {
      if (cast<CXXRecordDecl>(rt->getDecl())->isEmpty())
        return;
    }
  }

  // Cast the dest ptr to the appropriate i8 pointer type.
  if (builder.isInt8Ty(destPtr.getElementType())) {
    llvm_unreachable("NYI");
  }

  // Get size and alignment info for this aggregate.
  CharUnits size = getContext().getTypeSizeInChars(ty);
  [[maybe_unused]] mlir::Attribute sizeVal{};
  [[maybe_unused]] const VariableArrayType *vla = nullptr;

  // Don't bother emitting a zero-byte memset.
  if (size.isZero()) {
    // But note that getTypeInfo returns 0 for a VLA.
    if (const VariableArrayType *vlaType = dyn_cast_or_null<VariableArrayType>(
            getContext().getAsArrayType(ty))) {
      llvm_unreachable("NYI");
    } else {
      return;
    }
  } else {
    sizeVal = CGM.getSize(size);
  }

  // If the type contains a pointer to data member we can't memset it to zero.
  // Instead, create a null constant and copy it to the destination.
  // TODO: there are other patterns besides zero that we can usefully memset,
  // like -1, which happens to be the pattern used by member-pointers.
  if (!CGM.getTypes().isZeroInitializable(ty)) {
    llvm_unreachable("NYI");
  }

  // In LLVM Codegen: otherwise, just memset the whole thing to zero using
  // Builder.CreateMemSet. In CIR just emit a store of #cir.zero to the
  // respective address.
  // Builder.CreateMemSet(DestPtr, Builder.getInt8(0), SizeVal, false);
  builder.createStore(loc, builder.getZero(loc, convertType(ty)), destPtr);
}

CIRGenFunction::CIRGenFPOptionsRAII::CIRGenFPOptionsRAII(CIRGenFunction &cgf,
                                                         const clang::Expr *e)
    : CGF(cgf) {
  ConstructorHelper(e->getFPFeaturesInEffect(cgf.getLangOpts()));
}

CIRGenFunction::CIRGenFPOptionsRAII::CIRGenFPOptionsRAII(CIRGenFunction &cgf,
                                                         FPOptions fpFeatures)
    : CGF(cgf) {
  ConstructorHelper(fpFeatures);
}

void CIRGenFunction::CIRGenFPOptionsRAII::ConstructorHelper(
    FPOptions fpFeatures) {
  OldFPFeatures = CGF.CurFPFeatures;
  CGF.CurFPFeatures = fpFeatures;

  OldExcept = CGF.builder.getDefaultConstrainedExcept();
  OldRounding = CGF.builder.getDefaultConstrainedRounding();

  if (OldFPFeatures == fpFeatures)
    return;

  // TODO(cir): create guard to restore fast math configurations.
  assert(!cir::MissingFeatures::fastMathGuard());

  llvm::RoundingMode newRoundingBehavior = fpFeatures.getRoundingMode();
  // TODO(cir): override rounding behaviour once FM configs are guarded.
  auto newExceptionBehavior =
      toConstrainedExceptMd(static_cast<LangOptions::FPExceptionModeKind>(
          fpFeatures.getExceptionMode()));
  // TODO(cir): override exception behaviour once FM configs are guarded.

  // TODO(cir): override FP flags once FM configs are guarded.
  assert(!cir::MissingFeatures::fastMathFlags());

  assert((CGF.CurFuncDecl == nullptr || CGF.builder.getIsFPConstrained() ||
          isa<CXXConstructorDecl>(CGF.CurFuncDecl) ||
          isa<CXXDestructorDecl>(CGF.CurFuncDecl) ||
          (newExceptionBehavior == cir::fp::ebIgnore &&
           newRoundingBehavior == llvm::RoundingMode::NearestTiesToEven)) &&
         "FPConstrained should be enabled on entire function");

  // TODO(cir): mark CIR function with fast math attributes.
  assert(!cir::MissingFeatures::fastMathFuncAttributes());
}

CIRGenFunction::CIRGenFPOptionsRAII::~CIRGenFPOptionsRAII() {
  CGF.CurFPFeatures = OldFPFeatures;
  CGF.builder.setDefaultConstrainedExcept(OldExcept);
  CGF.builder.setDefaultConstrainedRounding(OldRounding);
}

// TODO(cir): should be shared with LLVM codegen.
bool CIRGenFunction::shouldNullCheckClassCastValue(const CastExpr *ce) {
  const Expr *e = ce->getSubExpr();

  if (ce->getCastKind() == CK_UncheckedDerivedToBase)
    return false;

  if (isa<CXXThisExpr>(e->IgnoreParens())) {
    // We always assume that 'this' is never null.
    return false;
  }

  if (const ImplicitCastExpr *ice = dyn_cast<ImplicitCastExpr>(ce)) {
    // And that glvalue casts are never null.
    if (ice->isGLValue())
      return false;
  }

  return true;
}

void CIRGenFunction::emitDeclRefExprDbgValue(const DeclRefExpr *e,
                                             const APValue &init) {
  assert(!cir::MissingFeatures::generateDebugInfo());
}

Address CIRGenFunction::emitVAListRef(const Expr *e) {
  if (getContext().getBuiltinVaListType()->isArrayType())
    return emitPointerWithAlignment(e);
  return emitLValue(e).getAddress();
}

// Emits an error if we don't have a valid set of target features for the
// called function.
void CIRGenFunction::checkTargetFeatures(const CallExpr *e,
                                         const FunctionDecl *targetDecl) {
  return checkTargetFeatures(e->getBeginLoc(), targetDecl);
}

// Emits an error if we don't have a valid set of target features for the
// called function.
void CIRGenFunction::checkTargetFeatures(SourceLocation loc,
                                         const FunctionDecl *targetDecl) {
  // Early exit if this is an indirect call.
  if (!targetDecl)
    return;

  // Get the current enclosing function if it exists. If it doesn't
  // we can't check the target features anyhow.
  const FunctionDecl *fd = dyn_cast_or_null<FunctionDecl>(CurCodeDecl);
  if (!fd)
    return;

  // Grab the required features for the call. For a builtin this is listed in
  // the td file with the default cpu, for an always_inline function this is any
  // listed cpu and any listed features.
  unsigned builtinId = targetDecl->getBuiltinID();
  std::string missingFeature;
  llvm::StringMap<bool> callerFeatureMap;
  CGM.getASTContext().getFunctionFeatureMap(callerFeatureMap, fd);
  if (builtinId) {
    StringRef featureList(
        getContext().BuiltinInfo.getRequiredFeatures(builtinId));
    if (!Builtin::evaluateRequiredTargetFeatures(featureList,
                                                 callerFeatureMap)) {
      CGM.getDiags().Report(loc, diag::err_builtin_needs_feature)
          << targetDecl->getDeclName() << featureList;
    }
  } else if (!targetDecl->isMultiVersion() &&
             targetDecl->hasAttr<TargetAttr>()) {
    // Get the required features for the callee.

    const TargetAttr *td = targetDecl->getAttr<TargetAttr>();
    ParsedTargetAttr parsedAttr = getContext().filterFunctionTargetAttrs(td);

    SmallVector<StringRef, 1> reqFeatures;
    llvm::StringMap<bool> calleeFeatureMap;
    getContext().getFunctionFeatureMap(calleeFeatureMap, targetDecl);

    for (const auto &f : parsedAttr.Features) {
      if (f[0] == '+' && calleeFeatureMap.lookup(f.substr(1)))
        reqFeatures.push_back(StringRef(f).substr(1));
    }

    for (const auto &f : calleeFeatureMap) {
      // Only positive features are "required".
      if (f.getValue())
        reqFeatures.push_back(f.getKey());
    }
    if (!llvm::all_of(reqFeatures, [&](StringRef feature) {
          if (!callerFeatureMap.lookup(feature)) {
            missingFeature = feature.str();
            return false;
          }
          return true;
        }))
      CGM.getDiags().Report(loc, diag::err_function_needs_feature)
          << fd->getDeclName() << targetDecl->getDeclName() << missingFeature;
  } else if (!fd->isMultiVersion() && fd->hasAttr<TargetAttr>()) {
    llvm::StringMap<bool> calleeFeatureMap;
    getContext().getFunctionFeatureMap(calleeFeatureMap, targetDecl);

    for (const auto &f : calleeFeatureMap) {
      if (f.getValue() && (!callerFeatureMap.lookup(f.getKey()) ||
                           !callerFeatureMap.find(f.getKey())->getValue()))
        CGM.getDiags().Report(loc, diag::err_function_needs_feature)
            << fd->getDeclName() << targetDecl->getDeclName() << f.getKey();
    }
  }
}

CIRGenFunction::VlaSizePair CIRGenFunction::getVLASize(QualType type) {
  const VariableArrayType *vla =
      CGM.getASTContext().getAsVariableArrayType(type);
  assert(vla && "type was not a variable array type!");
  return getVLASize(vla);
}

CIRGenFunction::VlaSizePair
CIRGenFunction::getVLASize(const VariableArrayType *type) {
  // The number of elements so far; always size_t.
  mlir::Value numElements;

  QualType elementType;
  do {
    elementType = type->getElementType();
    mlir::Value vlaSize = VLASizeMap[type->getSizeExpr()];
    assert(vlaSize && "no size for VLA!");
    assert(vlaSize.getType() == SizeTy);

    if (!numElements) {
      numElements = vlaSize;
    } else {
      // It's undefined behavior if this wraps around, so mark it that way.
      // FIXME: Teach -fsanitize=undefined to trap this.

      numElements = builder.createMul(numElements, vlaSize);
    }
  } while ((type = getContext().getAsVariableArrayType(elementType)));

  assert(numElements && "Undefined elements number");
  return {numElements, elementType};
}

// TODO(cir): most part of this function can be shared between CIRGen
// and traditional LLVM codegen
void CIRGenFunction::emitVariablyModifiedType(QualType type) {
  assert(type->isVariablyModifiedType() &&
         "Must pass variably modified type to EmitVLASizes!");

  // We're going to walk down into the type and look for VLA
  // expressions.
  do {
    assert(type->isVariablyModifiedType());

    const Type *ty = type.getTypePtr();
    switch (ty->getTypeClass()) {
    case clang::Type::CountAttributed:
    case clang::Type::PackIndexing:
    case clang::Type::ArrayParameter:
    case clang::Type::HLSLAttributedResource:
      llvm_unreachable("NYI");

#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.inc"
      llvm_unreachable("unexpected dependent type!");

    // These types are never variably-modified.
    case Type::Builtin:
    case Type::Complex:
    case Type::Vector:
    case Type::ExtVector:
    case Type::ConstantMatrix:
    case Type::Record:
    case Type::Enum:
    case Type::Using:
    case Type::TemplateSpecialization:
    case Type::ObjCTypeParam:
    case Type::ObjCObject:
    case Type::ObjCInterface:
    case Type::ObjCObjectPointer:
    case Type::BitInt:
      llvm_unreachable("type class is never variably-modified!");

    case Type::Elaborated:
      type = cast<clang::ElaboratedType>(ty)->getNamedType();
      break;

    case Type::Adjusted:
      type = cast<clang::AdjustedType>(ty)->getAdjustedType();
      break;

    case Type::Decayed:
      type = cast<clang::DecayedType>(ty)->getPointeeType();
      break;

    case Type::Pointer:
      type = cast<clang::PointerType>(ty)->getPointeeType();
      break;

    case Type::BlockPointer:
      type = cast<clang::BlockPointerType>(ty)->getPointeeType();
      break;

    case Type::LValueReference:
    case Type::RValueReference:
      type = cast<clang::ReferenceType>(ty)->getPointeeType();
      break;

    case Type::MemberPointer:
      type = cast<clang::MemberPointerType>(ty)->getPointeeType();
      break;

    case Type::ConstantArray:
    case Type::IncompleteArray:
      // Losing element qualification here is fine.
      type = cast<clang::ArrayType>(ty)->getElementType();
      break;

    case Type::VariableArray: {
      // Losing element qualification here is fine.
      const VariableArrayType *vat = cast<clang::VariableArrayType>(ty);

      // Unknown size indication requires no size computation.
      // Otherwise, evaluate and record it.
      if (const Expr *sizeExpr = vat->getSizeExpr()) {
        // It's possible that we might have emitted this already,
        // e.g. with a typedef and a pointer to it.
        mlir::Value &entry = VLASizeMap[sizeExpr];
        if (!entry) {
          mlir::Value size = emitScalarExpr(sizeExpr);
          assert(!cir::MissingFeatures::sanitizeVLABound());

          // Always zexting here would be wrong if it weren't
          // undefined behavior to have a negative bound.
          // FIXME: What about when size's type is larger than size_t?
          entry = builder.createIntCast(size, SizeTy);
        }
      }
      type = vat->getElementType();
      break;
    }

    case Type::FunctionProto:
    case Type::FunctionNoProto:
      type = cast<clang::FunctionType>(ty)->getReturnType();
      break;

    case Type::Paren:
    case Type::TypeOf:
    case Type::UnaryTransform:
    case Type::Attributed:
    case Type::BTFTagAttributed:
    case Type::SubstTemplateTypeParm:
    case Type::MacroQualified:
      // Keep walking after single level desugaring.
      type = type.getSingleStepDesugaredType(getContext());
      break;

    case Type::Typedef:
    case Type::Decltype:
    case Type::Auto:
    case Type::DeducedTemplateSpecialization:
      // Stop walking: nothing to do.
      return;

    case Type::TypeOfExpr:
      // Stop walking: emit typeof expression.
      emitIgnoredExpr(cast<clang::TypeOfExprType>(ty)->getUnderlyingExpr());
      return;

    case Type::Atomic:
      type = cast<clang::AtomicType>(ty)->getValueType();
      break;

    case Type::Pipe:
      type = cast<clang::PipeType>(ty)->getElementType();
      break;
    }
  } while (type->isVariablyModifiedType());
}

/// Computes the length of an array in elements, as well as the base
/// element type and a properly-typed first element pointer.
mlir::Value
CIRGenFunction::emitArrayLength(const clang::ArrayType *origArrayType,
                                QualType &baseType, Address &addr) {
  const auto *arrayType = origArrayType;

  // If it's a VLA, we have to load the stored size.  Note that
  // this is the size of the VLA in bytes, not its size in elements.
  mlir::Value numVLAElements{};
  if (isa<VariableArrayType>(arrayType)) {
    llvm_unreachable("NYI");
  }

  uint64_t countFromCLAs = 1;
  QualType eltType;

  // llvm::ArrayType *llvmArrayType =
  //     dyn_cast<llvm::ArrayType>(addr.getElementType());
  auto cirArrayType = mlir::dyn_cast<cir::ArrayType>(addr.getElementType());

  while (cirArrayType) {
    assert(isa<ConstantArrayType>(arrayType));
    countFromCLAs *= cirArrayType.getSize();
    eltType = arrayType->getElementType();

    cirArrayType = mlir::dyn_cast<cir::ArrayType>(cirArrayType.getEltType());

    arrayType = getContext().getAsArrayType(arrayType->getElementType());
    assert((!cirArrayType || arrayType) &&
           "CIR and Clang types are out-of-synch");
  }

  if (arrayType) {
    // From this point onwards, the Clang array type has been emitted
    // as some other type (probably a packed struct). Compute the array
    // size, and just emit the 'begin' expression as a bitcast.
    llvm_unreachable("NYI");
  }

  baseType = eltType;
  auto numElements = builder.getConstInt(*currSrcLoc, SizeTy, countFromCLAs);

  // If we had any VLA dimensions, factor them in.
  if (numVLAElements)
    llvm_unreachable("NYI");

  return numElements;
}

mlir::Value CIRGenFunction::emitAlignmentAssumption(
    mlir::Value ptrValue, QualType ty, SourceLocation loc,
    SourceLocation assumptionLoc, mlir::IntegerAttr alignment,
    mlir::Value offsetValue) {
  if (SanOpts.has(SanitizerKind::Alignment))
    llvm_unreachable("NYI");
  return builder.create<cir::AssumeAlignedOp>(getLoc(assumptionLoc), ptrValue,
                                              alignment, offsetValue);
}

mlir::Value CIRGenFunction::emitAlignmentAssumption(
    mlir::Value ptrValue, const Expr *expr, SourceLocation assumptionLoc,
    mlir::IntegerAttr alignment, mlir::Value offsetValue) {
  QualType ty = expr->getType();
  SourceLocation loc = expr->getExprLoc();
  return emitAlignmentAssumption(ptrValue, ty, loc, assumptionLoc, alignment,
                                 offsetValue);
}

void CIRGenFunction::emitVarAnnotations(const VarDecl *decl, mlir::Value val) {
  assert(decl->hasAttr<AnnotateAttr>() && "no annotate attribute");
  llvm::SmallVector<mlir::Attribute, 4> annotations;
  for (const auto *annot : decl->specific_attrs<AnnotateAttr>()) {
    annotations.push_back(CGM.emitAnnotateAttr(annot));
  }
  auto allocaOp = dyn_cast_or_null<cir::AllocaOp>(val.getDefiningOp());
  assert(allocaOp && "expects available alloca");
  allocaOp.setAnnotationsAttr(builder.getArrayAttr(annotations));
}
