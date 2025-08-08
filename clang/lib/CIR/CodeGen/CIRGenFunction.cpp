//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal per-function state used for AST-to-ClangIR code gen
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"

#include "CIRGenCXXABI.h"
#include "CIRGenCall.h"
#include "CIRGenValue.h"
#include "mlir/IR/Location.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/CIR/MissingFeatures.h"

#include <cassert>

namespace clang::CIRGen {

CIRGenFunction::CIRGenFunction(CIRGenModule &cgm, CIRGenBuilderTy &builder,
                               bool suppressNewContext)
    : CIRGenTypeCache(cgm), cgm{cgm}, builder(builder) {
  ehStack.setCGF(this);
}

CIRGenFunction::~CIRGenFunction() {}

// This is copied from clang/lib/CodeGen/CodeGenFunction.cpp
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
    case Type::HLSLAttributedResource:
    case Type::HLSLInlineSpirv:
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
    case Type::ArrayParameter:
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
  return cgm.getTypes().convertTypeForMem(t);
}

mlir::Type CIRGenFunction::convertType(QualType t) {
  return cgm.getTypes().convertType(t);
}

mlir::Location CIRGenFunction::getLoc(SourceLocation srcLoc) {
  // Some AST nodes might contain invalid source locations (e.g.
  // CXXDefaultArgExpr), workaround that to still get something out.
  if (srcLoc.isValid()) {
    const SourceManager &sm = getContext().getSourceManager();
    PresumedLoc pLoc = sm.getPresumedLoc(srcLoc);
    StringRef filename = pLoc.getFilename();
    return mlir::FileLineColLoc::get(builder.getStringAttr(filename),
                                     pLoc.getLine(), pLoc.getColumn());
  }
  // Do our best...
  assert(currSrcLoc && "expected to inherit some source location");
  return *currSrcLoc;
}

mlir::Location CIRGenFunction::getLoc(SourceRange srcLoc) {
  // Some AST nodes might contain invalid source locations (e.g.
  // CXXDefaultArgExpr), workaround that to still get something out.
  if (srcLoc.isValid()) {
    mlir::Location beg = getLoc(srcLoc.getBegin());
    mlir::Location end = getLoc(srcLoc.getEnd());
    SmallVector<mlir::Location, 2> locs = {beg, end};
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

bool CIRGenFunction::containsLabel(const Stmt *s, bool ignoreCaseStmts) {
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

  // If this is a switch statement, we want to ignore case statements when we
  // recursively process the sub-statements of the switch. If we haven't
  // encountered a switch statement, we treat case statements like labels, but
  // if we are processing a switch statement, case statements are expected.
  if (isa<SwitchStmt>(s))
    ignoreCaseStmts = true;

  // Scan subexpressions for verboten labels.
  return std::any_of(s->child_begin(), s->child_end(),
                     [=](const Stmt *subStmt) {
                       return containsLabel(subStmt, ignoreCaseStmts);
                     });
}

/// If the specified expression does not fold to a constant, or if it does but
/// contains a label, return false.  If it constant folds return true and set
/// the boolean result in Result.
bool CIRGenFunction::constantFoldsToBool(const Expr *cond, bool &resultBool,
                                         bool allowLabels) {
  llvm::APSInt resultInt;
  if (!constantFoldsToSimpleInteger(cond, resultInt, allowLabels))
    return false;

  resultBool = resultInt.getBoolValue();
  return true;
}

/// If the specified expression does not fold to a constant, or if it does
/// fold but contains a label, return false. If it constant folds, return
/// true and set the folded value.
bool CIRGenFunction::constantFoldsToSimpleInteger(const Expr *cond,
                                                  llvm::APSInt &resultInt,
                                                  bool allowLabels) {
  // FIXME: Rename and handle conversion of other evaluatable things
  // to bool.
  Expr::EvalResult result;
  if (!cond->EvaluateAsInt(result, getContext()))
    return false; // Not foldable, not integer or not fully evaluatable.

  llvm::APSInt intValue = result.Val.getInt();
  if (!allowLabels && containsLabel(cond))
    return false; // Contains a label.

  resultInt = intValue;
  return true;
}

void CIRGenFunction::emitAndUpdateRetAlloca(QualType type, mlir::Location loc,
                                            CharUnits alignment) {
  if (!type->isVoidType()) {
    fnRetAlloca = emitAlloca("__retval", convertType(type), loc, alignment,
                             /*insertIntoFnEntryBlock=*/false);
  }
}

void CIRGenFunction::declare(mlir::Value addrVal, const Decl *var, QualType ty,
                             mlir::Location loc, CharUnits alignment,
                             bool isParam) {
  assert(isa<NamedDecl>(var) && "Needs a named decl");
  assert(!cir::MissingFeatures::cgfSymbolTable());

  auto allocaOp = addrVal.getDefiningOp<cir::AllocaOp>();
  assert(allocaOp && "expected cir::AllocaOp");

  if (isParam)
    allocaOp.setInitAttr(mlir::UnitAttr::get(&getMLIRContext()));
  if (ty->isReferenceType() || ty.isConstQualified())
    allocaOp.setConstantAttr(mlir::UnitAttr::get(&getMLIRContext()));
}

void CIRGenFunction::LexicalScope::cleanup() {
  CIRGenBuilderTy &builder = cgf.builder;
  LexicalScope *localScope = cgf.curLexScope;

  auto applyCleanup = [&]() {
    if (performCleanup) {
      // ApplyDebugLocation
      assert(!cir::MissingFeatures::generateDebugInfo());
      forceCleanup();
    }
  };

  if (returnBlock != nullptr) {
    // Write out the return block, which loads the value from `__retval` and
    // issues the `cir.return`.
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(returnBlock);
    (void)emitReturn(*returnLoc);
  }

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
      builder.create<cir::BrOp>(insPt->back().getLoc(), cleanupBlock);
      if (!cleanupBlock->mightHaveTerminator()) {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(cleanupBlock);
        builder.create<cir::YieldOp>(localScope->endLoc);
      }
    }

    if (localScope->depth == 0) {
      // Reached the end of the function.
      if (returnBlock != nullptr) {
        if (returnBlock->getUses().empty()) {
          returnBlock->erase();
        } else {
          // Thread return block via cleanup block.
          if (cleanupBlock) {
            for (mlir::BlockOperand &blockUse : returnBlock->getUses()) {
              cir::BrOp brOp = mlir::cast<cir::BrOp>(blockUse.getOwner());
              brOp.setSuccessor(cleanupBlock);
            }
          }

          builder.create<cir::BrOp>(*returnLoc, returnBlock);
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
      !retVal ? builder.create<cir::YieldOp>(localScope->endLoc)
              : builder.create<cir::YieldOp>(localScope->endLoc, retVal);
    }
  };

  // If a cleanup block has been created at some point, branch to it
  // and set the insertion point to continue at the cleanup block.
  // Terminators are then inserted either in the cleanup block or
  // inline in this current block.
  mlir::Block *cleanupBlock = localScope->getCleanupBlock(builder);
  if (cleanupBlock)
    insertCleanupAndLeave(cleanupBlock);

  // Now deal with any pending block wrap up like implicit end of
  // scope.

  mlir::Block *curBlock = builder.getBlock();
  if (isGlobalInit() && !curBlock)
    return;
  if (curBlock->mightHaveTerminator() && curBlock->getTerminator())
    return;

  // Get rid of any empty block at the end of the scope.
  bool entryBlock = builder.getInsertionBlock()->isEntryBlock();
  if (!entryBlock && curBlock->empty()) {
    curBlock->erase();
    if (returnBlock != nullptr && returnBlock->getUses().empty())
      returnBlock->erase();
    return;
  }

  // If there's a cleanup block, branch to it, nothing else to do.
  if (cleanupBlock) {
    builder.create<cir::BrOp>(curBlock->back().getLoc(), cleanupBlock);
    return;
  }

  // No pre-existent cleanup block, emit cleanup code and yield/return.
  insertCleanupAndLeave(curBlock);
}

cir::ReturnOp CIRGenFunction::LexicalScope::emitReturn(mlir::Location loc) {
  CIRGenBuilderTy &builder = cgf.getBuilder();

  if (!cgf.curFn.getFunctionType().hasVoidReturn()) {
    // Load the value from `__retval` and return it via the `cir.return` op.
    auto value = builder.create<cir::LoadOp>(
        loc, cgf.curFn.getFunctionType().getReturnType(), *cgf.fnRetAlloca);
    return builder.create<cir::ReturnOp>(loc,
                                         llvm::ArrayRef(value.getResult()));
  }
  return builder.create<cir::ReturnOp>(loc);
}

// This is copied from CodeGenModule::MayDropFunctionReturn.  This is a
// candidate for sharing between CIRGen and CodeGen.
static bool mayDropFunctionReturn(const ASTContext &astContext,
                                  QualType returnType) {
  // We can't just discard the return value for a record type with a complex
  // destructor or a non-trivially copyable type.
  if (const RecordType *recordType =
          returnType.getCanonicalType()->getAs<RecordType>()) {
    if (const auto *classDecl = dyn_cast<CXXRecordDecl>(recordType->getDecl()))
      return classDecl->hasTrivialDestructor();
  }
  return returnType.isTriviallyCopyableType(astContext);
}

void CIRGenFunction::LexicalScope::emitImplicitReturn() {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  LexicalScope *localScope = cgf.curLexScope;

  const auto *fd = cast<clang::FunctionDecl>(cgf.curGD.getDecl());

  // In C++, flowing off the end of a non-void function is always undefined
  // behavior. In C, flowing off the end of a non-void function is undefined
  // behavior only if the non-existent return value is used by the caller.
  // That influences whether the terminating op is trap, unreachable, or
  // return.
  if (cgf.getLangOpts().CPlusPlus && !fd->hasImplicitReturnZero() &&
      !cgf.sawAsmBlock && !fd->getReturnType()->isVoidType() &&
      builder.getInsertionBlock()) {
    bool shouldEmitUnreachable =
        cgf.cgm.getCodeGenOpts().StrictReturn ||
        !mayDropFunctionReturn(fd->getASTContext(), fd->getReturnType());

    if (shouldEmitUnreachable) {
      assert(!cir::MissingFeatures::sanitizers());
      if (cgf.cgm.getCodeGenOpts().OptimizationLevel == 0)
        builder.create<cir::TrapOp>(localScope->endLoc);
      else
        builder.create<cir::UnreachableOp>(localScope->endLoc);
      builder.clearInsertionPoint();
      return;
    }
  }

  (void)emitReturn(localScope->endLoc);
}

void CIRGenFunction::startFunction(GlobalDecl gd, QualType returnType,
                                   cir::FuncOp fn, cir::FuncType funcType,
                                   FunctionArgList args, SourceLocation loc,
                                   SourceLocation startLoc) {
  assert(!curFn &&
         "CIRGenFunction can only be used for one function at a time");

  curFn = fn;

  const Decl *d = gd.getDecl();
  const auto *fd = dyn_cast_or_null<FunctionDecl>(d);
  curFuncDecl = d->getNonClosureContext();

  prologueCleanupDepth = ehStack.stable_begin();

  mlir::Block *entryBB = &fn.getBlocks().front();
  builder.setInsertionPointToStart(entryBB);

  // TODO(cir): this should live in `emitFunctionProlog
  // Declare all the function arguments in the symbol table.
  for (const auto nameValue : llvm::zip(args, entryBB->getArguments())) {
    const VarDecl *paramVar = std::get<0>(nameValue);
    mlir::Value paramVal = std::get<1>(nameValue);
    CharUnits alignment = getContext().getDeclAlign(paramVar);
    mlir::Location paramLoc = getLoc(paramVar->getSourceRange());
    paramVal.setLoc(paramLoc);

    mlir::Value addrVal =
        emitAlloca(cast<NamedDecl>(paramVar)->getName(),
                   convertType(paramVar->getType()), paramLoc, alignment,
                   /*insertIntoFnEntryBlock=*/true);

    declare(addrVal, paramVar, paramVar->getType(), paramLoc, alignment,
            /*isParam=*/true);

    setAddrOfLocalVar(paramVar, Address(addrVal, alignment));

    bool isPromoted = isa<ParmVarDecl>(paramVar) &&
                      cast<ParmVarDecl>(paramVar)->isKNRPromoted();
    assert(!cir::MissingFeatures::constructABIArgDirectExtend());
    if (isPromoted)
      cgm.errorNYI(fd->getSourceRange(), "Function argument demotion");

    // Location of the store to the param storage tracked as beginning of
    // the function body.
    mlir::Location fnBodyBegin = getLoc(fd->getBody()->getBeginLoc());
    builder.CIRBaseBuilderTy::createStore(fnBodyBegin, paramVal, addrVal);
  }
  assert(builder.getInsertionBlock() && "Should be valid");

  // When the current function is not void, create an address to store the
  // result value.
  if (!returnType->isVoidType())
    emitAndUpdateRetAlloca(returnType, getLoc(fd->getBody()->getEndLoc()),
                           getContext().getTypeAlignInChars(returnType));

  if (isa_and_nonnull<CXXMethodDecl>(d) &&
      cast<CXXMethodDecl>(d)->isInstance()) {
    cgm.getCXXABI().emitInstanceFunctionProlog(loc, *this);

    const auto *md = cast<CXXMethodDecl>(d);
    if (md->getParent()->isLambda() && md->getOverloadedOperator() == OO_Call) {
      cgm.errorNYI(loc, "lambda call operator");
    } else {
      // Not in a lambda; just use 'this' from the method.
      // FIXME: Should we generate a new load for each use of 'this'? The fast
      // register allocator would be happier...
      cxxThisValue = cxxabiThisValue;
    }

    assert(!cir::MissingFeatures::sanitizers());
    assert(!cir::MissingFeatures::emitTypeCheck());
  }
}

void CIRGenFunction::finishFunction(SourceLocation endLoc) {
  // Pop any cleanups that might have been associated with the
  // parameters.  Do this in whatever block we're currently in; it's
  // important to do this before we enter the return block or return
  // edges will be *really* confused.
  // TODO(cir): Use prologueCleanupDepth here.
  bool hasCleanups = ehStack.stable_begin() != prologueCleanupDepth;
  if (hasCleanups) {
    assert(!cir::MissingFeatures::generateDebugInfo());
    // FIXME(cir): should we clearInsertionPoint? breaks many testcases
    popCleanupBlocks(prologueCleanupDepth);
  }
}

mlir::LogicalResult CIRGenFunction::emitFunctionBody(const clang::Stmt *body) {
  auto result = mlir::LogicalResult::success();
  if (const CompoundStmt *block = dyn_cast<CompoundStmt>(body))
    emitCompoundStmtWithoutScope(*block);
  else
    result = emitStmt(body, /*useCurrentScope=*/true);

  return result;
}

static void eraseEmptyAndUnusedBlocks(cir::FuncOp func) {
  // Remove any leftover blocks that are unreachable and empty, since they do
  // not represent unreachable code useful for warnings nor anything deemed
  // useful in general.
  SmallVector<mlir::Block *> blocksToDelete;
  for (mlir::Block &block : func.getBlocks()) {
    if (block.empty() && block.getUses().empty())
      blocksToDelete.push_back(&block);
  }
  for (mlir::Block *block : blocksToDelete)
    block->erase();
}

cir::FuncOp CIRGenFunction::generateCode(clang::GlobalDecl gd, cir::FuncOp fn,
                                         cir::FuncType funcType) {
  const auto funcDecl = cast<FunctionDecl>(gd.getDecl());
  curGD = gd;

  SourceLocation loc = funcDecl->getLocation();
  Stmt *body = funcDecl->getBody();
  SourceRange bodyRange =
      body ? body->getSourceRange() : funcDecl->getLocation();

  SourceLocRAIIObject fnLoc{*this, loc.isValid() ? getLoc(loc)
                                                 : builder.getUnknownLoc()};

  auto validMLIRLoc = [&](clang::SourceLocation clangLoc) {
    return clangLoc.isValid() ? getLoc(clangLoc) : builder.getUnknownLoc();
  };
  const mlir::Location fusedLoc = mlir::FusedLoc::get(
      &getMLIRContext(),
      {validMLIRLoc(bodyRange.getBegin()), validMLIRLoc(bodyRange.getEnd())});
  mlir::Block *entryBB = fn.addEntryBlock();

  FunctionArgList args;
  QualType retTy = buildFunctionArgList(gd, args);

  {
    LexicalScope lexScope(*this, fusedLoc, entryBB);

    startFunction(gd, retTy, fn, funcType, args, loc, bodyRange.getBegin());

    if (isa<CXXDestructorDecl>(funcDecl)) {
      emitDestructorBody(args);
    } else if (isa<CXXConstructorDecl>(funcDecl)) {
      emitConstructorBody(args);
    } else if (getLangOpts().CUDA && !getLangOpts().CUDAIsDevice &&
               funcDecl->hasAttr<CUDAGlobalAttr>()) {
      getCIRGenModule().errorNYI(bodyRange, "CUDA kernel");
    } else if (isa<CXXMethodDecl>(funcDecl) &&
               cast<CXXMethodDecl>(funcDecl)->isLambdaStaticInvoker()) {
      getCIRGenModule().errorNYI(bodyRange, "Lambda static invoker");
    } else if (funcDecl->isDefaulted() && isa<CXXMethodDecl>(funcDecl) &&
               (cast<CXXMethodDecl>(funcDecl)->isCopyAssignmentOperator() ||
                cast<CXXMethodDecl>(funcDecl)->isMoveAssignmentOperator())) {
      // Implicit copy-assignment gets the same special treatment as implicit
      // copy-constructors.
      emitImplicitAssignmentOperatorBody(args);
    } else if (body) {
      if (mlir::failed(emitFunctionBody(body))) {
        fn.erase();
        return nullptr;
      }
    } else {
      // Anything without a body should have been handled above.
      llvm_unreachable("no definition for normal function");
    }

    if (mlir::failed(fn.verifyBody()))
      return nullptr;

    finishFunction(bodyRange.getEnd());
  }

  eraseEmptyAndUnusedBlocks(fn);
  return fn;
}

void CIRGenFunction::emitConstructorBody(FunctionArgList &args) {
  assert(!cir::MissingFeatures::sanitizers());
  const auto *ctor = cast<CXXConstructorDecl>(curGD.getDecl());
  CXXCtorType ctorType = curGD.getCtorType();

  assert((cgm.getTarget().getCXXABI().hasConstructorVariants() ||
          ctorType == Ctor_Complete) &&
         "can only generate complete ctor for this ABI");

  if (ctorType == Ctor_Complete && isConstructorDelegationValid(ctor) &&
      cgm.getTarget().getCXXABI().hasConstructorVariants()) {
    emitDelegateCXXConstructorCall(ctor, Ctor_Base, args, ctor->getEndLoc());
    return;
  }

  const FunctionDecl *definition = nullptr;
  Stmt *body = ctor->getBody(definition);
  assert(definition == ctor && "emitting wrong constructor body");

  if (isa_and_nonnull<CXXTryStmt>(body)) {
    cgm.errorNYI(ctor->getSourceRange(), "emitConstructorBody: try body");
    return;
  }

  assert(!cir::MissingFeatures::incrementProfileCounter());
  assert(!cir::MissingFeatures::runCleanupsScope());

  // TODO: in restricted cases, we can emit the vbase initializers of a
  // complete ctor and then delegate to the base ctor.

  // Emit the constructor prologue, i.e. the base and member initializers.
  emitCtorPrologue(ctor, ctorType, args);

  // TODO(cir): propagate this result via mlir::logical result. Just unreachable
  // now just to have it handled.
  if (mlir::failed(emitStmt(body, true))) {
    cgm.errorNYI(ctor->getSourceRange(),
                 "emitConstructorBody: emit body statement failed.");
    return;
  }
}

/// Emits the body of the current destructor.
void CIRGenFunction::emitDestructorBody(FunctionArgList &args) {
  const CXXDestructorDecl *dtor = cast<CXXDestructorDecl>(curGD.getDecl());
  CXXDtorType dtorType = curGD.getDtorType();

  // For an abstract class, non-base destructors are never used (and can't
  // be emitted in general, because vbase dtors may not have been validated
  // by Sema), but the Itanium ABI doesn't make them optional and Clang may
  // in fact emit references to them from other compilations, so emit them
  // as functions containing a trap instruction.
  if (dtorType != Dtor_Base && dtor->getParent()->isAbstract()) {
    cgm.errorNYI(dtor->getSourceRange(), "abstract base class destructors");
    return;
  }

  Stmt *body = dtor->getBody();
  assert(body && !cir::MissingFeatures::incrementProfileCounter());

  // The call to operator delete in a deleting destructor happens
  // outside of the function-try-block, which means it's always
  // possible to delegate the destructor body to the complete
  // destructor.  Do so.
  if (dtorType == Dtor_Deleting) {
    cgm.errorNYI(dtor->getSourceRange(), "deleting destructor");
    return;
  }

  // If the body is a function-try-block, enter the try before
  // anything else.
  const bool isTryBody = isa_and_nonnull<CXXTryStmt>(body);
  if (isTryBody)
    cgm.errorNYI(dtor->getSourceRange(), "function-try-block destructor");

  assert(!cir::MissingFeatures::sanitizers());
  assert(!cir::MissingFeatures::dtorCleanups());

  // If this is the complete variant, just invoke the base variant;
  // the epilogue will destruct the virtual bases.  But we can't do
  // this optimization if the body is a function-try-block, because
  // we'd introduce *two* handler blocks.  In the Microsoft ABI, we
  // always delegate because we might not have a definition in this TU.
  switch (dtorType) {
  case Dtor_Comdat:
    llvm_unreachable("not expecting a COMDAT");
  case Dtor_Deleting:
    llvm_unreachable("already handled deleting case");

  case Dtor_Complete:
    assert((body || getTarget().getCXXABI().isMicrosoft()) &&
           "can't emit a dtor without a body for non-Microsoft ABIs");

    assert(!cir::MissingFeatures::dtorCleanups());

    if (!isTryBody) {
      QualType thisTy = dtor->getFunctionObjectParameterType();
      emitCXXDestructorCall(dtor, Dtor_Base, /*forVirtualBase=*/false,
                            /*delegating=*/false, loadCXXThisAddress(), thisTy);
      break;
    }

    // Fallthrough: act like we're in the base variant.
    [[fallthrough]];

  case Dtor_Base:
    assert(body);

    assert(!cir::MissingFeatures::dtorCleanups());
    assert(!cir::MissingFeatures::vtableInitialization());

    if (isTryBody) {
      cgm.errorNYI(dtor->getSourceRange(), "function-try-block destructor");
    } else if (body) {
      (void)emitStmt(body, /*useCurrentScope=*/true);
    } else {
      assert(dtor->isImplicit() && "bodyless dtor not implicit");
      // nothing to do besides what's in the epilogue
    }
    // -fapple-kext must inline any call to this dtor into
    // the caller's body.
    assert(!cir::MissingFeatures::appleKext());

    break;
  }

  assert(!cir::MissingFeatures::dtorCleanups());

  // Exit the try if applicable.
  if (isTryBody)
    cgm.errorNYI(dtor->getSourceRange(), "function-try-block destructor");
}

/// Given a value of type T* that may not be to a complete object, construct
/// an l-vlaue withi the natural pointee alignment of T.
LValue CIRGenFunction::makeNaturalAlignPointeeAddrLValue(mlir::Value val,
                                                         QualType ty) {
  // FIXME(cir): is it safe to assume Op->getResult(0) is valid? Perhaps
  // assert on the result type first.
  LValueBaseInfo baseInfo;
  assert(!cir::MissingFeatures::opTBAA());
  CharUnits align = cgm.getNaturalTypeAlignment(ty, &baseInfo);
  return makeAddrLValue(Address(val, align), ty, baseInfo);
}

LValue CIRGenFunction::makeNaturalAlignAddrLValue(mlir::Value val,
                                                  QualType ty) {
  LValueBaseInfo baseInfo;
  CharUnits alignment = cgm.getNaturalTypeAlignment(ty, &baseInfo);
  Address addr(val, convertTypeForMem(ty), alignment);
  assert(!cir::MissingFeatures::opTBAA());
  return makeAddrLValue(addr, ty, baseInfo);
}

clang::QualType CIRGenFunction::buildFunctionArgList(clang::GlobalDecl gd,
                                                     FunctionArgList &args) {
  const auto *fd = cast<FunctionDecl>(gd.getDecl());
  QualType retTy = fd->getReturnType();

  const auto *md = dyn_cast<CXXMethodDecl>(fd);
  if (md && md->isInstance()) {
    if (cgm.getCXXABI().hasThisReturn(gd))
      cgm.errorNYI(fd->getSourceRange(), "this return");
    else if (cgm.getCXXABI().hasMostDerivedReturn(gd))
      cgm.errorNYI(fd->getSourceRange(), "most derived return");
    cgm.getCXXABI().buildThisParam(*this, args);
  }

  if (const auto *cd = dyn_cast<CXXConstructorDecl>(fd))
    if (cd->getInheritedConstructor())
      cgm.errorNYI(fd->getSourceRange(),
                   "buildFunctionArgList: inherited constructor");

  for (auto *param : fd->parameters())
    args.push_back(param);

  if (md && (isa<CXXConstructorDecl>(md) || isa<CXXDestructorDecl>(md)))
    assert(!cir::MissingFeatures::cxxabiStructorImplicitParam());

  return retTy;
}

/// Emit code to compute a designator that specifies the location
/// of the expression.
/// FIXME: document this function better.
LValue CIRGenFunction::emitLValue(const Expr *e) {
  // FIXME: ApplyDebugLocation DL(*this, e);
  switch (e->getStmtClass()) {
  default:
    getCIRGenModule().errorNYI(e->getSourceRange(),
                               std::string("l-value not implemented for '") +
                                   e->getStmtClassName() + "'");
    return LValue();
  case Expr::ArraySubscriptExprClass:
    return emitArraySubscriptExpr(cast<ArraySubscriptExpr>(e));
  case Expr::UnaryOperatorClass:
    return emitUnaryOpLValue(cast<UnaryOperator>(e));
  case Expr::StringLiteralClass:
    return emitStringLiteralLValue(cast<StringLiteral>(e));
  case Expr::MemberExprClass:
    return emitMemberExpr(cast<MemberExpr>(e));
  case Expr::CompoundLiteralExprClass:
    return emitCompoundLiteralLValue(cast<CompoundLiteralExpr>(e));
  case Expr::BinaryOperatorClass:
    return emitBinaryOperatorLValue(cast<BinaryOperator>(e));
  case Expr::CompoundAssignOperatorClass: {
    QualType ty = e->getType();
    if (ty->getAs<AtomicType>()) {
      cgm.errorNYI(e->getSourceRange(),
                   "CompoundAssignOperator with AtomicType");
      return LValue();
    }
    if (!ty->isAnyComplexType())
      return emitCompoundAssignmentLValue(cast<CompoundAssignOperator>(e));

    return emitComplexCompoundAssignmentLValue(cast<CompoundAssignOperator>(e));
  }
  case Expr::CallExprClass:
  case Expr::CXXMemberCallExprClass:
  case Expr::CXXOperatorCallExprClass:
  case Expr::UserDefinedLiteralClass:
    return emitCallExprLValue(cast<CallExpr>(e));
  case Expr::ParenExprClass:
    return emitLValue(cast<ParenExpr>(e)->getSubExpr());
  case Expr::DeclRefExprClass:
    return emitDeclRefLValue(cast<DeclRefExpr>(e));
  case Expr::CStyleCastExprClass:
  case Expr::CXXStaticCastExprClass:
  case Expr::CXXDynamicCastExprClass:
  case Expr::ImplicitCastExprClass:
    return emitCastLValue(cast<CastExpr>(e));
  case Expr::MaterializeTemporaryExprClass:
    return emitMaterializeTemporaryExpr(cast<MaterializeTemporaryExpr>(e));
  }
}

static std::string getVersionedTmpName(llvm::StringRef name, unsigned cnt) {
  SmallString<256> buffer;
  llvm::raw_svector_ostream out(buffer);
  out << name << cnt;
  return std::string(out.str());
}

std::string CIRGenFunction::getCounterRefTmpAsString() {
  return getVersionedTmpName("ref.tmp", counterRefTmp++);
}

std::string CIRGenFunction::getCounterAggTmpAsString() {
  return getVersionedTmpName("agg.tmp", counterAggTmp++);
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
    cgm.errorNYI(loc, "Cast the dest ptr to the appropriate i8 pointer type");
  }

  // Get size and alignment info for this aggregate.
  const CharUnits size = getContext().getTypeSizeInChars(ty);
  if (size.isZero()) {
    // But note that getTypeInfo returns 0 for a VLA.
    if (isa<VariableArrayType>(getContext().getAsArrayType(ty))) {
      cgm.errorNYI(loc,
                   "emitNullInitialization for zero size VariableArrayType");
    } else {
      return;
    }
  }

  // If the type contains a pointer to data member we can't memset it to zero.
  // Instead, create a null constant and copy it to the destination.
  // TODO: there are other patterns besides zero that we can usefully memset,
  // like -1, which happens to be the pattern used by member-pointers.
  if (!cgm.getTypes().isZeroInitializable(ty)) {
    cgm.errorNYI(loc, "type is not zero initializable");
  }

  // In LLVM Codegen: otherwise, just memset the whole thing to zero using
  // Builder.CreateMemSet. In CIR just emit a store of #cir.zero to the
  // respective address.
  // Builder.CreateMemSet(DestPtr, Builder.getInt8(0), SizeVal, false);
  const mlir::Value zeroValue = builder.getNullValue(convertType(ty), loc);
  builder.createStore(loc, zeroValue, destPtr);
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

/// Computes the length of an array in elements, as well as the base
/// element type and a properly-typed first element pointer.
mlir::Value
CIRGenFunction::emitArrayLength(const clang::ArrayType *origArrayType,
                                QualType &baseType, Address &addr) {
  const clang::ArrayType *arrayType = origArrayType;

  // If it's a VLA, we have to load the stored size.  Note that
  // this is the size of the VLA in bytes, not its size in elements.
  if (isa<VariableArrayType>(arrayType)) {
    assert(cir::MissingFeatures::vlas());
    cgm.errorNYI(*currSrcLoc, "VLAs");
    return builder.getConstInt(*currSrcLoc, SizeTy, 0);
  }

  uint64_t countFromCLAs = 1;
  QualType eltType;

  auto cirArrayType = mlir::dyn_cast<cir::ArrayType>(addr.getElementType());

  while (cirArrayType) {
    assert(isa<ConstantArrayType>(arrayType));
    countFromCLAs *= cirArrayType.getSize();
    eltType = arrayType->getElementType();

    cirArrayType =
        mlir::dyn_cast<cir::ArrayType>(cirArrayType.getElementType());

    arrayType = getContext().getAsArrayType(arrayType->getElementType());
    assert((!cirArrayType || arrayType) &&
           "CIR and Clang types are out-of-sync");
  }

  if (arrayType) {
    // From this point onwards, the Clang array type has been emitted
    // as some other type (probably a packed struct). Compute the array
    // size, and just emit the 'begin' expression as a bitcast.
    cgm.errorNYI(*currSrcLoc, "length for non-array underlying types");
  }

  baseType = eltType;
  return builder.getConstInt(*currSrcLoc, SizeTy, countFromCLAs);
}

// TODO(cir): Most of this function can be shared between CIRGen
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
    case Type::CountAttributed:
    case Type::PackIndexing:
    case Type::ArrayParameter:
    case Type::HLSLAttributedResource:
    case Type::HLSLInlineSpirv:
    case Type::PredefinedSugar:
      cgm.errorNYI("CIRGenFunction::emitVariablyModifiedType");
      break;

#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.inc"
      llvm_unreachable(
          "dependent type must be resolved before the CIR codegen");

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
      cgm.errorNYI("CIRGenFunction::emitVariablyModifiedType VLA");
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

} // namespace clang::CIRGen
