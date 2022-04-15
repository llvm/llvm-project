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

#include "clang/AST/ExprObjC.h"
#include "clang/Basic/TargetInfo.h"

#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;

CIRGenFunction::CIRGenFunction(CIRGenModule &CGM, mlir::OpBuilder &builder)
    : CGM{CGM}, builder(builder), CurFuncDecl(nullptr),
      SanOpts(CGM.getLangOpts().Sanitize) {}

clang::ASTContext &CIRGenFunction::getContext() const {
  return CGM.getASTContext();
}

TypeEvaluationKind CIRGenFunction::getEvaluationKind(QualType type) {
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
      return TEK_Scalar;

    // Complexes.
    case Type::Complex:
      return TEK_Complex;

    // Arrays, records, and Objective-C objects.
    case Type::ConstantArray:
    case Type::IncompleteArray:
    case Type::VariableArray:
    case Type::Record:
    case Type::ObjCObject:
    case Type::ObjCInterface:
      return TEK_Aggregate;

    // We operate on atomic values according to their underlying type.
    case Type::Atomic:
      type = cast<AtomicType>(type)->getValueType();
      continue;
    }
    llvm_unreachable("unknown type kind!");
  }
}

mlir::Type CIRGenFunction::convertType(QualType T) {
  return CGM.getTypes().ConvertType(T);
}

mlir::Location CIRGenFunction::getLoc(SourceLocation SLoc) {
  const SourceManager &SM = getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(SLoc);
  StringRef Filename = PLoc.getFilename();
  return mlir::FileLineColLoc::get(builder.getStringAttr(Filename),
                                   PLoc.getLine(), PLoc.getColumn());
}

mlir::Location CIRGenFunction::getLoc(SourceRange SLoc) {
  mlir::Location B = getLoc(SLoc.getBegin());
  mlir::Location E = getLoc(SLoc.getEnd());
  SmallVector<mlir::Location, 2> locs = {B, E};
  mlir::Attribute metadata;
  return mlir::FusedLoc::get(locs, metadata, builder.getContext());
}

mlir::Location CIRGenFunction::getLoc(mlir::Location lhs, mlir::Location rhs) {
  SmallVector<mlir::Location, 2> locs = {lhs, rhs};
  mlir::Attribute metadata;
  return mlir::FusedLoc::get(locs, metadata, builder.getContext());
}

/// Return true if the statement contains a label in it.  If
/// this statement is not executed normally, it not containing a label means
/// that we can just remove the code.
bool CIRGenFunction::ContainsLabel(const Stmt *S, bool IgnoreCaseStmts) {
  // Null statement, not a label!
  if (!S)
    return false;

  // If this is a label, we have to emit the code, consider something like:
  // if (0) {  ...  foo:  bar(); }  goto foo;
  //
  // TODO: If anyone cared, we could track __label__'s, since we know that you
  // can't jump to one from outside their declared region.
  if (isa<LabelStmt>(S))
    return true;

  // If this is a case/default statement, and we haven't seen a switch, we
  // have to emit the code.
  if (isa<SwitchCase>(S) && !IgnoreCaseStmts)
    return true;

  // If this is a switch statement, we want to ignore cases below it.
  if (isa<SwitchStmt>(S))
    IgnoreCaseStmts = true;

  // Scan subexpressions for verboten labels.
  for (const Stmt *SubStmt : S->children())
    if (ContainsLabel(SubStmt, IgnoreCaseStmts))
      return true;

  return false;
}

/// If the specified expression does not fold
/// to a constant, or if it does but contains a label, return false.  If it
/// constant folds return true and set the folded value.
bool CIRGenFunction::ConstantFoldsToSimpleInteger(const Expr *Cond,
                                                  llvm::APSInt &ResultInt,
                                                  bool AllowLabels) {
  // FIXME: Rename and handle conversion of other evaluatable things
  // to bool.
  Expr::EvalResult Result;
  if (!Cond->EvaluateAsInt(Result, getContext()))
    return false; // Not foldable, not integer or not fully evaluatable.

  llvm::APSInt Int = Result.Val.getInt();
  if (!AllowLabels && ContainsLabel(Cond))
    return false; // Contains a label.

  ResultInt = Int;
  return true;
}

mlir::Type CIRGenFunction::getCIRType(const QualType &type) {
  return CGM.getCIRType(type);
}

void CIRGenFunction::buildAndUpdateRetAlloca(QualType ty, mlir::Location loc,
                                             CharUnits alignment) {
  auto addr =
      buildAlloca("__retval", InitStyle::uninitialized, ty, loc, alignment);
  FnRetAlloca = addr;
}

mlir::LogicalResult CIRGenFunction::declare(const Decl *var, QualType ty,
                                            mlir::Location loc,
                                            CharUnits alignment,
                                            mlir::Value &addr, bool isParam) {
  const auto *namedVar = dyn_cast_or_null<NamedDecl>(var);
  assert(namedVar && "Needs a named decl");
  assert(!symbolTable.count(var) && "not supposed to be available just yet");

  addr = buildAlloca(namedVar->getName(),
                     isParam ? InitStyle::paraminit : InitStyle::uninitialized,
                     ty, loc, alignment);

  symbolTable.insert(var, addr);
  return mlir::success();
}

/// All scope related cleanup needed:
/// - Patching up unsolved goto's.
/// - Build all cleanup code and insert yield/returns.
void CIRGenFunction::LexicalScopeGuard::cleanup() {
  auto &builder = CGF.builder;
  auto *localScope = CGF.currLexScope;

  // Handle pending gotos and the solved labels in this scope.
  while (!localScope->PendingGotos.empty()) {
    auto gotoInfo = localScope->PendingGotos.back();
    // FIXME: Currently only support resolving goto labels inside the
    // same lexical ecope.
    assert(localScope->SolvedLabels.count(gotoInfo.second) &&
           "goto across scopes not yet supported");

    // The goto in this lexical context actually maps to a basic
    // block.
    auto g = cast<mlir::cir::BrOp>(gotoInfo.first);
    g.setSuccessor(CGF.LabelMap[gotoInfo.second].getBlock());
    localScope->PendingGotos.pop_back();
  }
  localScope->SolvedLabels.clear();

  // Cleanup are done right before codegen resume a scope. This is where
  // objects are destroyed.
  unsigned curLoc = 0;
  for (auto *retBlock : localScope->getRetBlocks()) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(retBlock);
    mlir::Location retLoc = *localScope->getRetLocs()[curLoc];
    curLoc++;

    // TODO: insert actual scope cleanup HERE (dtors and etc)

    // If there's anything to return, load it first.
    if (CGF.FnRetCIRTy.has_value()) {
      auto val =
          builder.create<LoadOp>(retLoc, *CGF.FnRetCIRTy, *CGF.FnRetAlloca);
      builder.create<ReturnOp>(retLoc, llvm::ArrayRef(val.getResult()));
    } else {
      builder.create<ReturnOp>(retLoc);
    }
  }

  auto insertCleanupAndLeave = [&](mlir::Block *InsPt) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(InsPt);
    // TODO: insert actual scope cleanup (dtors and etc)
    if (localScope->Depth != 0) // end of any local scope != function
      builder.create<YieldOp>(localScope->EndLoc);
    else
      builder.create<ReturnOp>(localScope->EndLoc);
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
  bool entryBlock = builder.getInsertionBlock()->isEntryBlock();
  auto *currBlock = builder.getBlock();
  bool hasTerminator =
      !currBlock->empty() &&
      currBlock->back().hasTrait<mlir::OpTrait::IsTerminator>();
  if (hasTerminator)
    return;

  // An empty non-entry block has nothing to offer.
  if (!entryBlock && currBlock->empty()) {
    currBlock->erase();
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

mlir::FuncOp CIRGenFunction::generateCode(clang::GlobalDecl GD, mlir::FuncOp Fn,
                                          const CIRGenFunctionInfo &FnInfo) {
  assert(Fn && "generating code for a null function");
  const auto FD = cast<FunctionDecl>(GD.getDecl());
  if (FD->isInlineBuiltinDeclaration()) {
    llvm_unreachable("NYI");
  } else {
    // Detect the unusual situation where an inline version is shadowed by a
    // non-inline version. In that case we should pick the external one
    // everywhere. That's GCC behavior too. Unfortunately, I cannot find a way
    // to detect that situation before we reach codegen, so do some late
    // replacement.
    for (const auto *PD = FD->getPreviousDecl(); PD;
         PD = PD->getPreviousDecl()) {
      if (LLVM_UNLIKELY(PD->isInlineBuiltinDeclaration())) {
        llvm_unreachable("NYI");
      }
    }
  }

  // Check if we should generate debug info for this function.
  if (FD->hasAttr<NoDebugAttr>()) {
    llvm_unreachable("NYI");
  }

  // If this is a function specialization then use the pattern body as the
  // location for the function.
  if (const auto *SpecDecl = FD->getTemplateInstantiationPattern())
    llvm_unreachable("NYI");

  Stmt *Body = FD->getBody();

  if (Body) {
    // Coroutines always emit lifetime markers
    if (isa<CoroutineBodyStmt>(Body))
      llvm_unreachable("Coroutines NYI");
  }

  // Create a scope in the symbol table to hold variable declarations.
  SymTableScopeTy varScope(symbolTable);

  const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD);
  assert(!MD && "methods not implemented");

  FnRetQualTy = FD->getReturnType();
  mlir::TypeRange FnTyRange = {};
  if (!FnRetQualTy->isVoidType()) {
    FnRetCIRTy = getCIRType(FnRetQualTy);
  }

  // In MLIR the entry block of the function is special: it must have the
  // same argument list as the function itself.
  mlir::Block *entryBlock = Fn.addEntryBlock();

  // Set the insertion point in the builder to the beginning of the
  // function body, it will be used throughout the codegen to create
  // operations in this function.
  builder.setInsertionPointToStart(entryBlock);
  auto FnBeginLoc = getLoc(FD->getBody()->getEndLoc());
  auto FnEndLoc = getLoc(FD->getBody()->getEndLoc());

  // Initialize lexical scope information.
  {
    LexicalScopeContext lexScope{FnBeginLoc, FnEndLoc,
                                 builder.getInsertionBlock()};
    LexicalScopeGuard scopeGuard{*this, &lexScope};

    // Declare all the function arguments in the symbol table.
    for (const auto nameValue :
         llvm::zip(FD->parameters(), entryBlock->getArguments())) {
      auto *paramVar = std::get<0>(nameValue);
      auto paramVal = std::get<1>(nameValue);
      auto alignment = getContext().getDeclAlign(paramVar);
      auto paramLoc = getLoc(paramVar->getSourceRange());
      paramVal.setLoc(paramLoc);

      mlir::Value addr;
      if (failed(declare(paramVar, paramVar->getType(), paramLoc, alignment,
                         addr, true /*param*/)))
        return nullptr;
      // Location of the store to the param storage tracked as beginning of
      // the function body.
      auto fnBodyBegin = getLoc(FD->getBody()->getBeginLoc());
      builder.create<mlir::cir::StoreOp>(fnBodyBegin, paramVal, addr);
    }
    assert(builder.getInsertionBlock() && "Should be valid");

    // When the current function is not void, create an address to store the
    // result value.
    if (FnRetCIRTy.has_value())
      buildAndUpdateRetAlloca(FnRetQualTy, FnEndLoc,
                              CGM.getNaturalTypeAlignment(FnRetQualTy));

    // Emit the body of the function.
    if (mlir::failed(buildFunctionBody(FD->getBody()))) {
      Fn.erase();
      return nullptr;
    }
    assert(builder.getInsertionBlock() && "Should be valid");
  }

  if (mlir::failed(Fn.verifyBody()))
    return nullptr;

  return Fn;
}

/// ShouldInstrumentFunction - Return true if the current function should be
/// instrumented with __cyg_profile_func_* calls
bool CIRGenFunction::ShouldInstrumentFunction() {
  if (!CGM.getCodeGenOpts().InstrumentFunctions &&
      !CGM.getCodeGenOpts().InstrumentFunctionsAfterInlining &&
      !CGM.getCodeGenOpts().InstrumentFunctionEntryBare)
    return false;

  llvm_unreachable("NYI");
}

mlir::LogicalResult CIRGenFunction::buildFunctionBody(const clang::Stmt *Body) {
  // TODO: incrementProfileCounter(Body);

  // We start with function level scope for variables.
  SymTableScopeTy varScope(symbolTable);

  auto result = mlir::LogicalResult::success();
  if (const CompoundStmt *S = dyn_cast<CompoundStmt>(Body))
    result = buildCompoundStmtWithoutScope(*S);
  else
    result = buildStmt(Body, /*useCurrentScope*/ true);

  // This is checked after emitting the function body so we know if there are
  // any permitted infinite loops.
  // TODO: if (checkIfFunctionMustProgress())
  // CurFn->addFnAttr(llvm::Attribute::MustProgress);
  return result;
}

clang::QualType CIRGenFunction::buildFunctionArgList(clang::GlobalDecl GD,
                                                     FunctionArgList &Args) {
  const auto *FD = cast<FunctionDecl>(GD.getDecl());
  QualType ResTy = FD->getReturnType();

  const auto *MD = dyn_cast<CXXMethodDecl>(FD);
  if (MD && MD->isInstance()) {
    llvm_unreachable("NYI");
  }

  // The base version of an inheriting constructor whose constructed base is a
  // virtual base is not passed any arguments (because it doesn't actually
  // call the inherited constructor).
  bool PassedParams = true;
  if (const auto *CD = dyn_cast<CXXConstructorDecl>(FD))
    llvm_unreachable("NYI");

  if (PassedParams) {
    for (auto *Param : FD->parameters()) {
      Args.push_back(Param);
      if (!Param->hasAttr<PassObjectSizeAttr>())
        continue;

      llvm_unreachable("PassObjectSizeAttr NYI");
    }
  }

  if (MD && (isa<CXXConstructorDecl>(MD) || isa<CXXDestructorDecl>(MD)))
    llvm_unreachable("NYI");

  return ResTy;
}
