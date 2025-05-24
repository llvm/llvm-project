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
#include "clang/AST/GlobalDecl.h"
#include "clang/CIR/MissingFeatures.h"

#include <cassert>

namespace clang::CIRGen {

CIRGenFunction::CIRGenFunction(CIRGenModule &cgm, CIRGenBuilderTy &builder,
                               bool suppressNewContext)
    : CIRGenTypeCache(cgm), cgm{cgm}, builder(builder) {}

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
  const auto *namedVar = dyn_cast_or_null<NamedDecl>(var);
  assert(namedVar && "Needs a named decl");
  assert(!cir::MissingFeatures::cgfSymbolTable());

  auto allocaOp = cast<cir::AllocaOp>(addrVal.getDefiningOp());
  if (isParam)
    allocaOp.setInitAttr(mlir::UnitAttr::get(&getMLIRContext()));
  if (ty->isReferenceType() || ty.isConstQualified())
    allocaOp.setConstantAttr(mlir::UnitAttr::get(&getMLIRContext()));
}

void CIRGenFunction::LexicalScope::cleanup() {
  CIRGenBuilderTy &builder = cgf.builder;
  LexicalScope *localScope = cgf.curLexScope;

  if (returnBlock != nullptr) {
    // Write out the return block, which loads the value from `__retval` and
    // issues the `cir.return`.
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(returnBlock);
    (void)emitReturn(*returnLoc);
  }

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

  // Reached the end of the scope.
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(curBlock);

    if (localScope->depth == 0) {
      // Reached the end of the function.
      if (returnBlock != nullptr) {
        if (returnBlock->getUses().empty())
          returnBlock->erase();
        else {
          builder.create<cir::BrOp>(*returnLoc, returnBlock);
          return;
        }
      }
      emitImplicitReturn();
      return;
    }
    // Reached the end of a non-function scope.  Some scopes, such as those
    // used with the ?: operator, can return a value.
    if (!localScope->isTernary() && !curBlock->mightHaveTerminator()) {
      !retVal ? builder.create<cir::YieldOp>(localScope->endLoc)
              : builder.create<cir::YieldOp>(localScope->endLoc, retVal);
    }
  }
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

  const auto *fd = dyn_cast_or_null<FunctionDecl>(gd.getDecl());

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
}

void CIRGenFunction::finishFunction(SourceLocation endLoc) {}

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

    if (isa<CXXDestructorDecl>(funcDecl))
      getCIRGenModule().errorNYI(bodyRange, "C++ destructor definition");
    else if (isa<CXXConstructorDecl>(funcDecl))
      getCIRGenModule().errorNYI(bodyRange, "C++ constructor definition");
    else if (getLangOpts().CUDA && !getLangOpts().CUDAIsDevice &&
             funcDecl->hasAttr<CUDAGlobalAttr>())
      getCIRGenModule().errorNYI(bodyRange, "CUDA kernel");
    else if (isa<CXXMethodDecl>(funcDecl) &&
             cast<CXXMethodDecl>(funcDecl)->isLambdaStaticInvoker())
      getCIRGenModule().errorNYI(bodyRange, "Lambda static invoker");
    else if (funcDecl->isDefaulted() && isa<CXXMethodDecl>(funcDecl) &&
             (cast<CXXMethodDecl>(funcDecl)->isCopyAssignmentOperator() ||
              cast<CXXMethodDecl>(funcDecl)->isMoveAssignmentOperator()))
      getCIRGenModule().errorNYI(bodyRange, "Default assignment operator");
    else if (body) {
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

  if (isa<CXXConstructorDecl>(fd))
    cgm.errorNYI(fd->getSourceRange(),
                 "buildFunctionArgList: CXXConstructorDecl");

  for (auto *param : fd->parameters())
    args.push_back(param);

  if (md && (isa<CXXConstructorDecl>(md) || isa<CXXDestructorDecl>(md)))
    cgm.errorNYI(fd->getSourceRange(),
                 "buildFunctionArgList: implicit structor params");

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
    cgm.errorNYI(e->getSourceRange(),
                 "CompoundAssignOperator with ComplexType");
    return LValue();
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
  }
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

} // namespace clang::CIRGen
