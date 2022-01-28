//===- CIRGenModule.cpp - Per-Module state for CIR generation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-translation-unit state used for CIR translation.
//
//===----------------------------------------------------------------------===//

#include "CIRGenModule.h"

#include "CIRGenFunction.h"
#include "CIRGenTypes.h"
#include "CIRGenValue.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace mlir::cir;
using namespace cir;
using namespace clang;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;

CIRGenModule::CIRGenModule(mlir::MLIRContext &context,
                           clang::ASTContext &astctx)
    : builder(&context), astCtx(astctx) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  genTypes = std::make_unique<CIRGenTypes>(astCtx, this->getBuilder());
}

mlir::Location CIRGenModule::getLoc(SourceLocation SLoc) {
  const SourceManager &SM = astCtx.getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(SLoc);
  StringRef Filename = PLoc.getFilename();
  return mlir::FileLineColLoc::get(builder.getStringAttr(Filename),
                                   PLoc.getLine(), PLoc.getColumn());
}

mlir::Location CIRGenModule::getLoc(SourceRange SLoc) {
  mlir::Location B = getLoc(SLoc.getBegin());
  mlir::Location E = getLoc(SLoc.getEnd());
  SmallVector<mlir::Location, 2> locs = {B, E};
  mlir::Attribute metadata;
  return mlir::FusedLoc::get(locs, metadata, builder.getContext());
}

mlir::Location CIRGenModule::getLoc(mlir::Location lhs, mlir::Location rhs) {
  SmallVector<mlir::Location, 2> locs = {lhs, rhs};
  mlir::Attribute metadata;
  return mlir::FusedLoc::get(locs, metadata, builder.getContext());
}

mlir::LogicalResult CIRGenModule::declare(const Decl *var, QualType T,
                                          mlir::Location loc,
                                          CharUnits alignment,
                                          mlir::Value &addr, bool IsParam) {
  const auto *namedVar = dyn_cast_or_null<NamedDecl>(var);
  assert(namedVar && "Needs a named decl");

  if (symbolTable.count(var))
    return mlir::failure();

  auto localVarTy = getCIRType(T);
  auto localVarPtrTy =
      mlir::cir::PointerType::get(builder.getContext(), localVarTy);

  auto alignIntAttr =
      mlir::IntegerAttr::get(mlir::IntegerType::get(builder.getContext(), 64),
                             alignment.getQuantity());

  auto localVarAddr = builder.create<mlir::cir::AllocaOp>(
      loc, /*addr type*/ localVarPtrTy, /*var type*/ localVarTy,
      IsParam ? InitStyle::paraminit : InitStyle::uninitialized, alignIntAttr);

  auto *parentBlock = localVarAddr->getBlock();
  localVarAddr->moveBefore(&parentBlock->front());

  // Insert into the symbol table, allocate some stack space in the
  // function entry block.
  symbolTable.insert(var, localVarAddr);
  addr = localVarAddr;

  return mlir::success();
}

bool CIRGenModule::isTypeConstant(QualType Ty, bool ExcludeCtor) {
  if (!Ty.isConstant(astCtx) && !Ty->isReferenceType())
    return false;

  if (astCtx.getLangOpts().CPlusPlus) {
    if (const CXXRecordDecl *Record =
            astCtx.getBaseElementType(Ty)->getAsCXXRecordDecl())
      return ExcludeCtor && !Record->hasMutableFields() &&
             Record->hasTrivialDestructor();
  }

  return true;
}

CIRGenModule::AutoVarEmission
CIRGenModule::buildAutoVarAlloca(const VarDecl &D) {
  QualType Ty = D.getType();
  // TODO: (|| Ty.getAddressSpace() == LangAS::opencl_private &&
  //        getLangOpts().OpenCL))
  assert(Ty.getAddressSpace() == LangAS::Default);

  assert(!D.isEscapingByref() && "not implemented");
  assert(!Ty->isVariablyModifiedType() && "not implemented");
  assert(!astCtx.getLangOpts().OpenMP && // !CGM.getLangOpts().OpenMPIRBuilder
         "not implemented");
  bool NRVO = astCtx.getLangOpts().ElideConstructors && D.isNRVOVariable();
  assert(!NRVO && "not implemented");
  assert(Ty->isConstantSizeType() && "not implemented");
  assert(!D.hasAttr<AnnotateAttr>() && "not implemented");

  AutoVarEmission emission(D);
  CharUnits alignment = astCtx.getDeclAlign(&D);
  // TODO: debug info
  // TODO: use CXXABI

  // If this value is an array or struct with a statically determinable
  // constant initializer, there are optimizations we can do.
  //
  // TODO: We should constant-evaluate the initializer of any variable,
  // as long as it is initialized by a constant expression. Currently,
  // isConstantInitializer produces wrong answers for structs with
  // reference or bitfield members, and a few other cases, and checking
  // for POD-ness protects us from some of these.
  if (D.getInit() && (Ty->isArrayType() || Ty->isRecordType()) &&
      (D.isConstexpr() ||
       ((Ty.isPODType(astCtx) ||
         astCtx.getBaseElementType(Ty)->isObjCObjectPointerType()) &&
        D.getInit()->isConstantInitializer(astCtx, false)))) {

    // If the variable's a const type, and it's neither an NRVO
    // candidate nor a __block variable and has no mutable members,
    // emit it as a global instead.
    // Exception is if a variable is located in non-constant address space
    // in OpenCL.
    // TODO: deal with CGM.getCodeGenOpts().MergeAllConstants
    // TODO: perhaps we don't need this at all at CIR since this can
    // be done as part of lowering down to LLVM.
    if ((!astCtx.getLangOpts().OpenCL ||
         Ty.getAddressSpace() == LangAS::opencl_constant) &&
        (!NRVO && !D.isEscapingByref() && isTypeConstant(Ty, true)))
      assert(0 && "not implemented");

    // Otherwise, tell the initialization code that we're in this case.
    emission.IsConstantAggregate = true;
  }

  // TODO: track source location range...
  mlir::Value addr;
  if (failed(declare(&D, Ty, getLoc(D.getSourceRange()), alignment, addr))) {
    theModule.emitError("Cannot declare variable");
    return emission;
  }

  // TODO: what about emitting lifetime markers for MSVC catch parameters?
  // TODO: something like @llvm.lifetime.start/end here? revisit this later.
  emission.Addr = RawAddress{addr, alignment};
  return emission;
}

/// Determine whether the given initializer is trivial in the sense
/// that it requires no code to be generated.
bool CIRGenModule::isTrivialInitializer(const Expr *Init) {
  if (!Init)
    return true;

  if (const CXXConstructExpr *Construct = dyn_cast<CXXConstructExpr>(Init))
    if (CXXConstructorDecl *Constructor = Construct->getConstructor())
      if (Constructor->isTrivial() && Constructor->isDefaultConstructor() &&
          !Construct->requiresZeroInitialization())
        return true;

  return false;
}

// TODO: this can also be abstrated into common AST helpers
bool CIRGenModule::hasBooleanRepresentation(QualType Ty) {

  if (Ty->isBooleanType())
    return true;

  if (const EnumType *ET = Ty->getAs<EnumType>())
    return ET->getDecl()->getIntegerType()->isBooleanType();

  if (const AtomicType *AT = Ty->getAs<AtomicType>())
    return hasBooleanRepresentation(AT->getValueType());

  return false;
}

mlir::Value CIRGenModule::buildToMemory(mlir::Value Value, QualType Ty) {
  // Bool has a different representation in memory than in registers.
  return Value;
}

void CIRGenModule::buildStoreOfScalar(mlir::Value value, LValue lvalue,
                                      const Decl *InitDecl) {
  // TODO: constant matrix type, volatile, non temporal, TBAA
  buildStoreOfScalar(value, lvalue.getAddress(), false, lvalue.getType(),
                     lvalue.getBaseInfo(), InitDecl, false);
}

void CIRGenModule::buildStoreOfScalar(mlir::Value Value, RawAddress Addr,
                                      bool Volatile, QualType Ty,
                                      LValueBaseInfo BaseInfo,
                                      const Decl *InitDecl,
                                      bool isNontemporal) {
  // TODO: PreserveVec3Type
  // TODO: LValueIsSuitableForInlineAtomic ?
  // TODO: TBAA
  Value = buildToMemory(Value, Ty);
  if (Ty->isAtomicType() || isNontemporal) {
    assert(0 && "not implemented");
  }

  // Update the alloca with more info on initialization.
  auto SrcAlloca =
      dyn_cast_or_null<mlir::cir::AllocaOp>(Addr.getPointer().getDefiningOp());
  if (InitDecl) {
    InitStyle IS;
    const VarDecl *VD = dyn_cast_or_null<VarDecl>(InitDecl);
    assert(VD && "VarDecl expected");
    if (VD->hasInit()) {
      switch (VD->getInitStyle()) {
      case VarDecl::ParenListInit:
        llvm_unreachable("NYI");
      case VarDecl::CInit:
        IS = InitStyle::cinit;
        break;
      case VarDecl::CallInit:
        IS = InitStyle::callinit;
        break;
      case VarDecl::ListInit:
        IS = InitStyle::listinit;
        break;
      }
      SrcAlloca.setInitAttr(InitStyleAttr::get(builder.getContext(), IS));
    }
  }
  assert(currSrcLoc && "must pass in source location");
  builder.create<mlir::cir::StoreOp>(*currSrcLoc, Value, Addr.getPointer());
}

void CIRGenModule::buldStoreThroughLValue(RValue Src, LValue Dst,
                                          const Decl *InitDecl) {
  assert(Dst.isSimple() && "only implemented simple");
  // TODO: ObjC lifetime.
  assert(Src.isScalar() && "Can't emit an agg store with this method");
  buildStoreOfScalar(Src.getScalarVal(), Dst, InitDecl);
}

void CIRGenModule::buildScalarInit(const Expr *init, const ValueDecl *D,
                                   LValue lvalue) {
  // TODO: this is where a lot of ObjC lifetime stuff would be done.
  mlir::Value value = buildScalarExpr(init);
  SourceLocRAIIObject Loc{*this, getLoc(D->getSourceRange())};
  buldStoreThroughLValue(RValue::get(value), lvalue, D);
  return;
}

void CIRGenModule::buildExprAsInit(const Expr *init, const ValueDecl *D,
                                   LValue lvalue) {
  QualType type = D->getType();

  if (type->isReferenceType()) {
    assert(0 && "not implemented");
    return;
  }
  switch (CIRGenFunction::getEvaluationKind(type)) {
  case TEK_Scalar:
    buildScalarInit(init, D, lvalue);
    return;
  case TEK_Complex: {
    assert(0 && "not implemented");
    return;
  }
  case TEK_Aggregate:
    assert(0 && "not implemented");
    return;
  }
  llvm_unreachable("bad evaluation kind");
}

void CIRGenModule::buildAutoVarInit(const AutoVarEmission &emission) {
  assert(emission.Variable && "emission was not valid!");

  const VarDecl &D = *emission.Variable;
  QualType type = D.getType();

  // If this local has an initializer, emit it now.
  const Expr *Init = D.getInit();

  // TODO: in LLVM codegen if we are at an unreachable point, the initializer
  // isn't emitted unless it contains a label. What we want for CIR?
  assert(builder.getInsertionBlock());

  // Initialize the variable here if it doesn't have a initializer and it is a
  // C struct that is non-trivial to initialize or an array containing such a
  // struct.
  if (!Init && type.isNonTrivialToPrimitiveDefaultInitialize() ==
                   QualType::PDIK_Struct) {
    assert(0 && "not implemented");
    return;
  }

  const RawAddress Loc = emission.Addr;

  // Note: constexpr already initializes everything correctly.
  LangOptions::TrivialAutoVarInitKind trivialAutoVarInit =
      (D.isConstexpr()
           ? LangOptions::TrivialAutoVarInitKind::Uninitialized
           : (D.getAttr<UninitializedAttr>()
                  ? LangOptions::TrivialAutoVarInitKind::Uninitialized
                  : astCtx.getLangOpts().getTrivialAutoVarInit()));

  auto initializeWhatIsTechnicallyUninitialized = [&](RawAddress Loc) {
    if (trivialAutoVarInit ==
        LangOptions::TrivialAutoVarInitKind::Uninitialized)
      return;

    assert(0 && "unimplemented");
  };

  if (isTrivialInitializer(Init))
    return initializeWhatIsTechnicallyUninitialized(Loc);

  if (emission.IsConstantAggregate ||
      D.mightBeUsableInConstantExpressions(astCtx)) {
    assert(0 && "not implemented");
  }

  initializeWhatIsTechnicallyUninitialized(Loc);
  LValue lv = LValue::makeAddr(Loc, type, AlignmentSource::Decl);
  return buildExprAsInit(Init, &D, lv);
}

void CIRGenModule::buildAutoVarCleanups(const AutoVarEmission &emission) {
  assert(emission.Variable && "emission was not valid!");

  // TODO: in LLVM codegen if we are at an unreachable point codgen
  // is ignored. What we want for CIR?
  assert(builder.getInsertionBlock());
  const VarDecl &D = *emission.Variable;

  // Check the type for a cleanup.
  // TODO: something like emitAutoVarTypeCleanup
  if (QualType::DestructionKind dtorKind = D.needsDestruction(astCtx))
    assert(0 && "not implemented");

  // In GC mode, honor objc_precise_lifetime.
  if (astCtx.getLangOpts().getGC() != LangOptions::NonGC &&
      D.hasAttr<ObjCPreciseLifetimeAttr>())
    assert(0 && "not implemented");

  // Handle the cleanup attribute.
  if (const CleanupAttr *CA = D.getAttr<CleanupAttr>())
    assert(0 && "not implemented");

  // TODO: handle block variable
}

/// Emit code and set up symbol table for a variable declaration with auto,
/// register, or no storage class specifier. These turn into simple stack
/// objects, globals depending on target.
void CIRGenModule::buildAutoVarDecl(const VarDecl &D) {
  AutoVarEmission emission = buildAutoVarAlloca(D);
  buildAutoVarInit(emission);
  buildAutoVarCleanups(emission);
}

void CIRGenModule::buildVarDecl(const VarDecl &D) {
  if (D.hasExternalStorage()) {
    assert(0 && "should we just returns is there something to track?");
    // Don't emit it now, allow it to be emitted lazily on its first use.
    return;
  }

  // Some function-scope variable does not have static storage but still
  // needs to be emitted like a static variable, e.g. a function-scope
  // variable in constant address space in OpenCL.
  if (D.getStorageDuration() != SD_Automatic)
    assert(0 && "not implemented");

  if (D.getType().getAddressSpace() == LangAS::opencl_local)
    assert(0 && "not implemented");

  assert(D.hasLocalStorage());
  return buildAutoVarDecl(D);
}

void CIRGenModule::buildDecl(const Decl &D) {
  switch (D.getKind()) {
  case Decl::ImplicitConceptSpecialization:
  case Decl::TopLevelStmt:
  case Decl::HLSLBuffer:
  case Decl::UnnamedGlobalConstant:
    llvm_unreachable("NYI");
  case Decl::BuiltinTemplate:
  case Decl::TranslationUnit:
  case Decl::ExternCContext:
  case Decl::Namespace:
  case Decl::UnresolvedUsingTypename:
  case Decl::ClassTemplateSpecialization:
  case Decl::ClassTemplatePartialSpecialization:
  case Decl::VarTemplateSpecialization:
  case Decl::VarTemplatePartialSpecialization:
  case Decl::TemplateTypeParm:
  case Decl::UnresolvedUsingValue:
  case Decl::NonTypeTemplateParm:
  case Decl::CXXDeductionGuide:
  case Decl::CXXMethod:
  case Decl::CXXConstructor:
  case Decl::CXXDestructor:
  case Decl::CXXConversion:
  case Decl::Field:
  case Decl::MSProperty:
  case Decl::IndirectField:
  case Decl::ObjCIvar:
  case Decl::ObjCAtDefsField:
  case Decl::ParmVar:
  case Decl::ImplicitParam:
  case Decl::ClassTemplate:
  case Decl::VarTemplate:
  case Decl::FunctionTemplate:
  case Decl::TypeAliasTemplate:
  case Decl::TemplateTemplateParm:
  case Decl::ObjCMethod:
  case Decl::ObjCCategory:
  case Decl::ObjCProtocol:
  case Decl::ObjCInterface:
  case Decl::ObjCCategoryImpl:
  case Decl::ObjCImplementation:
  case Decl::ObjCProperty:
  case Decl::ObjCCompatibleAlias:
  case Decl::PragmaComment:
  case Decl::PragmaDetectMismatch:
  case Decl::AccessSpec:
  case Decl::LinkageSpec:
  case Decl::Export:
  case Decl::ObjCPropertyImpl:
  case Decl::FileScopeAsm:
  case Decl::Friend:
  case Decl::FriendTemplate:
  case Decl::Block:
  case Decl::Captured:
  case Decl::UsingShadow:
  case Decl::ConstructorUsingShadow:
  case Decl::ObjCTypeParam:
  case Decl::Binding:
  case Decl::UnresolvedUsingIfExists:
    llvm_unreachable("Declaration should not be in declstmts!");
  case Decl::Record:    // struct/union/class X;
  case Decl::CXXRecord: // struct/union/class X; [C++]
    assert(0 && "Not implemented");
    return;
  case Decl::Enum: // enum X;
    assert(0 && "Not implemented");
    return;
  case Decl::Function:     // void X();
  case Decl::EnumConstant: // enum ? { X = ? }
  case Decl::StaticAssert: // static_assert(X, ""); [C++0x]
  case Decl::Label:        // __label__ x;
  case Decl::Import:
  case Decl::MSGuid: // __declspec(uuid("..."))
  case Decl::TemplateParamObject:
  case Decl::OMPThreadPrivate:
  case Decl::OMPAllocate:
  case Decl::OMPCapturedExpr:
  case Decl::OMPRequires:
  case Decl::Empty:
  case Decl::Concept:
  case Decl::LifetimeExtendedTemporary:
  case Decl::RequiresExprBody:
    // None of these decls require codegen support.
    return;

  case Decl::NamespaceAlias:
    assert(0 && "Not implemented");
    return;
  case Decl::Using: // using X; [C++]
    assert(0 && "Not implemented");
    return;
  case Decl::UsingEnum: // using enum X; [C++]
    assert(0 && "Not implemented");
    return;
  case Decl::UsingPack:
    assert(0 && "Not implemented");
    return;
  case Decl::UsingDirective: // using namespace X; [C++]
    assert(0 && "Not implemented");
    return;
  case Decl::Var:
  case Decl::Decomposition: {
    const VarDecl &VD = cast<VarDecl>(D);
    assert(VD.isLocalVarDecl() &&
           "Should not see file-scope variables inside a function!");
    buildVarDecl(VD);
    if (auto *DD = dyn_cast<DecompositionDecl>(&VD))
      assert(0 && "Not implemented");

    // FIXME: add this
    // if (auto *DD = dyn_cast<DecompositionDecl>(&VD))
    //   for (auto *B : DD->bindings())
    //     if (auto *HD = B->getHoldingVar())
    //       EmitVarDecl(*HD);
    return;
  }

  case Decl::OMPDeclareReduction:
  case Decl::OMPDeclareMapper:
    assert(0 && "Not implemented");

  case Decl::Typedef:     // typedef int X;
  case Decl::TypeAlias: { // using X = int; [C++0x]
    assert(0 && "Not implemented");
  }
  }
}

/// Emit the computation of the specified expression of scalar type,
/// ignoring the result.
mlir::Value CIRGenModule::buildScalarExpr(const Expr *E) {
  assert(E && CIRGenFunction::hasScalarEvaluationKind(E->getType()) &&
         "Invalid scalar expression to emit");

  return ScalarExprEmitter(*CurCGF, *this).Visit(const_cast<Expr *>(E));
}

/// Emit a conversion from the specified type to the specified destination
/// type, both of which are CIR scalar types.
mlir::Value CIRGenModule::buildScalarConversion(mlir::Value Src, QualType SrcTy,
                                                QualType DstTy,
                                                SourceLocation Loc) {
  assert(CIRGenFunction::hasScalarEvaluationKind(SrcTy) &&
         CIRGenFunction::hasScalarEvaluationKind(DstTy) &&
         "Invalid scalar expression to emit");
  return ScalarExprEmitter(*CurCGF, *this)
      .buildScalarConversion(Src, SrcTy, DstTy, Loc);
}

mlir::LogicalResult CIRGenModule::buildReturnStmt(const ReturnStmt &S) {
  assert(!(astCtx.getLangOpts().ElideConstructors && S.getNRVOCandidate() &&
           S.getNRVOCandidate()->isNRVOVariable()) &&
         "unimplemented");
  assert(!CurCGF->FnRetQualTy->isReferenceType() && "unimplemented");

  // Emit the result value, even if unused, to evaluate the side effects.
  const Expr *RV = S.getRetValue();
  if (!RV) // Do nothing (return value is left uninitialized)
    return mlir::success();
  assert(!isa<ExprWithCleanups>(RV) && "unimplemented");

  mlir::Value V = nullptr;
  switch (CIRGenFunction::getEvaluationKind(RV->getType())) {
  case TEK_Scalar:
    V = buildScalarExpr(RV);
    // Builder.CreateStore(EmitScalarExpr(RV), ReturnValue);
    break;
  case TEK_Complex:
  case TEK_Aggregate:
    llvm::errs() << "ReturnStmt EvaluationKind not implemented\n";
    return mlir::failure();
  }

  CurCGF->RetValue = V;
  // Otherwise, this return operation has zero operands.
  if (!V || (RV && RV->getType()->isVoidType())) {
    // FIXME: evaluate for side effects.
  }

  builder.create<ReturnOp>(getLoc(S.getSourceRange()),
                           V ? ArrayRef(V) : ArrayRef<mlir::Value>());
  return mlir::success();
}

mlir::LogicalResult CIRGenModule::buildDeclStmt(const DeclStmt &S) {
  if (!builder.getInsertionBlock())
    theModule.emitError(
        "Seems like this is unreachable code, what should we do?");

  for (const auto *I : S.decls()) {
    buildDecl(*I);
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenModule::buildSimpleStmt(const Stmt *S,
                                                  bool useCurrentScope) {
  switch (S->getStmtClass()) {
  default:
    return mlir::failure();
  case Stmt::DeclStmtClass:
    return buildDeclStmt(cast<DeclStmt>(*S));
  case Stmt::CompoundStmtClass:
    return useCurrentScope
               ? buildCompoundStmtWithoutScope(cast<CompoundStmt>(*S))
               : buildCompoundStmt(cast<CompoundStmt>(*S));
  case Stmt::ReturnStmtClass:
    return buildReturnStmt(cast<ReturnStmt>(*S));
  case Stmt::NullStmtClass:
    break;

  case Stmt::LabelStmtClass:
  case Stmt::AttributedStmtClass:
  case Stmt::GotoStmtClass:
  case Stmt::BreakStmtClass:
  case Stmt::ContinueStmtClass:
  case Stmt::DefaultStmtClass:
  case Stmt::CaseStmtClass:
  case Stmt::SEHLeaveStmtClass:
    llvm::errs() << "CIR codegen for '" << S->getStmtClassName()
                 << "' not implemented\n";
    assert(0 && "not implemented");
  }

  return mlir::success();
}

LValue CIRGenModule::buildDeclRefLValue(const DeclRefExpr *E) {
  const NamedDecl *ND = E->getDecl();

  assert(E->isNonOdrUse() != NOUR_Unevaluated &&
         "should not emit an unevaluated operand");

  if (const auto *VD = dyn_cast<VarDecl>(ND)) {
    // Global Named registers access via intrinsics only
    assert(VD->getStorageClass() != SC_Register && "not implemented");
    assert(E->isNonOdrUse() != NOUR_Constant && "not implemented");
    assert(!E->refersToEnclosingVariableOrCapture() && "not implemented");
    assert(!(VD->hasLinkage() || VD->isStaticDataMember()) &&
           "not implemented");
    assert(!VD->isEscapingByref() && "not implemented");
    assert(!VD->getType()->isReferenceType() && "not implemented");
    assert(symbolTable.count(VD) && "should be already mapped");

    mlir::Value V = symbolTable.lookup(VD);
    assert(V && "Name lookup must succeed");

    LValue LV = LValue::makeAddr(RawAddress(V, CharUnits::fromQuantity(4)),
                                 VD->getType(), AlignmentSource::Decl);
    return LV;
  }

  llvm_unreachable("Unhandled DeclRefExpr?");
}

/// Emit code to compute the specified expression which
/// can have any type.  The result is returned as an RValue struct.
/// TODO: if this is an aggregate expression, add a AggValueSlot to indicate
/// where the result should be returned.
RValue CIRGenModule::buildAnyExpr(const Expr *E) {
  switch (CIRGenFunction::getEvaluationKind(E->getType())) {
  case TEK_Scalar:
    return RValue::get(buildScalarExpr(E));
  case TEK_Complex:
    assert(0 && "not implemented");
  case TEK_Aggregate:
    assert(0 && "not implemented");
  }
  llvm_unreachable("bad evaluation kind");
}

LValue CIRGenModule::buildBinaryOperatorLValue(const BinaryOperator *E) {
  // Comma expressions just emit their LHS then their RHS as an l-value.
  if (E->getOpcode() == BO_Comma) {
    assert(0 && "not implemented");
  }

  if (E->getOpcode() == BO_PtrMemD || E->getOpcode() == BO_PtrMemI)
    assert(0 && "not implemented");

  assert(E->getOpcode() == BO_Assign && "unexpected binary l-value");

  // Note that in all of these cases, __block variables need the RHS
  // evaluated first just in case the variable gets moved by the RHS.

  switch (CIRGenFunction::getEvaluationKind(E->getType())) {
  case TEK_Scalar: {
    assert(E->getLHS()->getType().getObjCLifetime() ==
               clang::Qualifiers::ObjCLifetime::OCL_None &&
           "not implemented");

    RValue RV = buildAnyExpr(E->getRHS());
    LValue LV = buildLValue(E->getLHS());

    SourceLocRAIIObject Loc{*this, getLoc(E->getSourceRange())};
    buldStoreThroughLValue(RV, LV, nullptr /*InitDecl*/);
    assert(!astCtx.getLangOpts().OpenMP && "last priv cond not implemented");
    return LV;
  }

  case TEK_Complex:
    assert(0 && "not implemented");
  case TEK_Aggregate:
    assert(0 && "not implemented");
  }
  llvm_unreachable("bad evaluation kind");
}

/// FIXME: this could likely be a common helper and not necessarily related
/// with codegen.
/// Return the best known alignment for an unknown pointer to a
/// particular class.
CharUnits CIRGenModule::getClassPointerAlignment(const CXXRecordDecl *RD) {
  if (!RD->hasDefinition())
    return CharUnits::One(); // Hopefully won't be used anywhere.

  auto &layout = astCtx.getASTRecordLayout(RD);

  // If the class is final, then we know that the pointer points to an
  // object of that type and can use the full alignment.
  if (RD->isEffectivelyFinal())
    return layout.getAlignment();

  // Otherwise, we have to assume it could be a subclass.
  return layout.getNonVirtualAlignment();
}

/// FIXME: this could likely be a common helper and not necessarily related
/// with codegen.
/// TODO: Add TBAAAccessInfo
CharUnits
CIRGenModule::getNaturalPointeeTypeAlignment(QualType T,
                                             LValueBaseInfo *BaseInfo) {
  return getNaturalTypeAlignment(T->getPointeeType(), BaseInfo,
                                 /* forPointeeType= */ true);
}

/// FIXME: this could likely be a common helper and not necessarily related
/// with codegen.
/// TODO: Add TBAAAccessInfo
CharUnits CIRGenModule::getNaturalTypeAlignment(QualType T,
                                                LValueBaseInfo *BaseInfo,
                                                bool forPointeeType) {
  // FIXME: This duplicates logic in ASTContext::getTypeAlignIfKnown. But
  // that doesn't return the information we need to compute BaseInfo.

  // Honor alignment typedef attributes even on incomplete types.
  // We also honor them straight for C++ class types, even as pointees;
  // there's an expressivity gap here.
  if (auto TT = T->getAs<TypedefType>()) {
    if (auto Align = TT->getDecl()->getMaxAlignment()) {
      if (BaseInfo)
        *BaseInfo = LValueBaseInfo(AlignmentSource::AttributedType);
      return astCtx.toCharUnitsFromBits(Align);
    }
  }

  bool AlignForArray = T->isArrayType();

  // Analyze the base element type, so we don't get confused by incomplete
  // array types.
  T = astCtx.getBaseElementType(T);

  if (T->isIncompleteType()) {
    // We could try to replicate the logic from
    // ASTContext::getTypeAlignIfKnown, but nothing uses the alignment if the
    // type is incomplete, so it's impossible to test. We could try to reuse
    // getTypeAlignIfKnown, but that doesn't return the information we need
    // to set BaseInfo.  So just ignore the possibility that the alignment is
    // greater than one.
    if (BaseInfo)
      *BaseInfo = LValueBaseInfo(AlignmentSource::Type);
    return CharUnits::One();
  }

  if (BaseInfo)
    *BaseInfo = LValueBaseInfo(AlignmentSource::Type);

  CharUnits Alignment;
  const CXXRecordDecl *RD;
  if (T.getQualifiers().hasUnaligned()) {
    Alignment = CharUnits::One();
  } else if (forPointeeType && !AlignForArray &&
             (RD = T->getAsCXXRecordDecl())) {
    // For C++ class pointees, we don't know whether we're pointing at a
    // base or a complete object, so we generally need to use the
    // non-virtual alignment.
    Alignment = getClassPointerAlignment(RD);
  } else {
    Alignment = astCtx.getTypeAlignInChars(T);
  }

  // Cap to the global maximum type alignment unless the alignment
  // was somehow explicit on the type.
  if (unsigned MaxAlign = astCtx.getLangOpts().MaxTypeAlign) {
    if (Alignment.getQuantity() > MaxAlign && !astCtx.isAlignmentRequired(T))
      Alignment = CharUnits::fromQuantity(MaxAlign);
  }
  return Alignment;
}

/// Given an expression of pointer type, try to
/// derive a more accurate bound on the alignment of the pointer.
RawAddress CIRGenModule::buildPointerWithAlignment(const Expr *E,
                                                   LValueBaseInfo *BaseInfo) {
  // We allow this with ObjC object pointers because of fragile ABIs.
  assert(E->getType()->isPointerType() ||
         E->getType()->isObjCObjectPointerType());
  E = E->IgnoreParens();

  // Casts:
  if (const CastExpr *CE = dyn_cast<CastExpr>(E)) {
    if (const auto *ECE = dyn_cast<ExplicitCastExpr>(CE))
      assert(0 && "not implemented");

    switch (CE->getCastKind()) {
    default:
      assert(0 && "not implemented");
    // Nothing to do here...
    case CK_LValueToRValue:
      break;
    }
  }

  // Unary &.
  if (const UnaryOperator *UO = dyn_cast<UnaryOperator>(E)) {
    assert(0 && "not implemented");
    // if (UO->getOpcode() == UO_AddrOf) {
    //   LValue LV = buildLValue(UO->getSubExpr());
    //   if (BaseInfo)
    //     *BaseInfo = LV.getBaseInfo();
    //   // TODO: TBBA info
    //   return LV.getAddress();
    // }
  }

  // TODO: conditional operators, comma.
  // Otherwise, use the alignment of the type.
  CharUnits Align = getNaturalPointeeTypeAlignment(E->getType(), BaseInfo);
  return RawAddress(buildScalarExpr(E), Align);
}

LValue CIRGenModule::buildUnaryOpLValue(const UnaryOperator *E) {
  // __extension__ doesn't affect lvalue-ness.
  assert(E->getOpcode() != UO_Extension && "not implemented");

  switch (E->getOpcode()) {
  default:
    llvm_unreachable("Unknown unary operator lvalue!");
  case UO_Deref: {
    QualType T = E->getSubExpr()->getType()->getPointeeType();
    assert(!T.isNull() && "CodeGenFunction::EmitUnaryOpLValue: Illegal type");

    LValueBaseInfo BaseInfo;
    // TODO: add TBAAInfo
    RawAddress Addr = buildPointerWithAlignment(E->getSubExpr(), &BaseInfo);
    LValue LV = LValue::makeAddr(Addr, T, BaseInfo);
    // TODO: set addr space
    // TODO: ObjC/GC/__weak write barrier stuff.
    return LV;
  }
  case UO_Real:
  case UO_Imag: {
    assert(0 && "not implemented");
  }
  case UO_PreInc:
  case UO_PreDec: {
    assert(0 && "not implemented");
  }
  }
}

/// Emit code to compute a designator that specifies the location
/// of the expression.
/// FIXME: document this function better.
LValue CIRGenModule::buildLValue(const Expr *E) {
  // FIXME: ApplyDebugLocation DL(*this, E);
  switch (E->getStmtClass()) {
  default: {
    emitError(getLoc(E->getExprLoc()), "l-value not implemented for '")
        << E->getStmtClassName() << "'";
    assert(0 && "not implemented");
  }
  case Expr::BinaryOperatorClass:
    return buildBinaryOperatorLValue(cast<BinaryOperator>(E));
  case Expr::DeclRefExprClass:
    return buildDeclRefLValue(cast<DeclRefExpr>(E));
  case Expr::UnaryOperatorClass:
    return buildUnaryOpLValue(cast<UnaryOperator>(E));
  case Expr::ObjCPropertyRefExprClass:
    llvm_unreachable("cannot emit a property reference directly");
  }

  return LValue::makeAddr(RawAddress::invalid(), E->getType());
}

/// EmitIgnoredExpr - Emit code to compute the specified expression,
/// ignoring the result.
void CIRGenModule::buildIgnoredExpr(const Expr *E) {
  assert(!E->isPRValue() && "not implemented");

  // Just emit it as an l-value and drop the result.
  buildLValue(E);
}

/// If the specified expression does not fold
/// to a constant, or if it does but contains a label, return false.  If it
/// constant folds return true and set the boolean result in Result.
bool CIRGenModule::ConstantFoldsToSimpleInteger(const Expr *Cond,
                                                bool &ResultBool,
                                                bool AllowLabels) {
  llvm::APSInt ResultInt;
  if (!ConstantFoldsToSimpleInteger(Cond, ResultInt, AllowLabels))
    return false;

  ResultBool = ResultInt.getBoolValue();
  return true;
}

/// Return true if the statement contains a label in it.  If
/// this statement is not executed normally, it not containing a label means
/// that we can just remove the code.
bool CIRGenModule::ContainsLabel(const Stmt *S, bool IgnoreCaseStmts) {
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
bool CIRGenModule::ConstantFoldsToSimpleInteger(const Expr *Cond,
                                                llvm::APSInt &ResultInt,
                                                bool AllowLabels) {
  // FIXME: Rename and handle conversion of other evaluatable things
  // to bool.
  Expr::EvalResult Result;
  if (!Cond->EvaluateAsInt(Result, astCtx))
    return false; // Not foldable, not integer or not fully evaluatable.

  llvm::APSInt Int = Result.Val.getInt();
  if (!AllowLabels && ContainsLabel(Cond))
    return false; // Contains a label.

  ResultInt = Int;
  return true;
}

/// Perform the usual unary conversions on the specified
/// expression and compare the result against zero, returning an Int1Ty value.
mlir::Value CIRGenModule::evaluateExprAsBool(const Expr *E) {
  // TODO: PGO
  if (const MemberPointerType *MPT = E->getType()->getAs<MemberPointerType>()) {
    assert(0 && "not implemented");
  }

  QualType BoolTy = astCtx.BoolTy;
  SourceLocation Loc = E->getExprLoc();
  // TODO: CGFPOptionsRAII for FP stuff.
  assert(!E->getType()->isAnyComplexType() &&
         "complex to scalar not implemented");
  return buildScalarConversion(buildScalarExpr(E), E->getType(), BoolTy, Loc);
}

/// Emit an if on a boolean condition to the specified blocks.
/// FIXME: Based on the condition, this might try to simplify the codegen of
/// the conditional based on the branch. TrueCount should be the number of
/// times we expect the condition to evaluate to true based on PGO data. We
/// might decide to leave this as a separate pass (see EmitBranchOnBoolExpr
/// for extra ideas).
mlir::LogicalResult CIRGenModule::buildIfOnBoolExpr(const Expr *cond,
                                                    mlir::Location loc,
                                                    const Stmt *thenS,
                                                    const Stmt *elseS) {
  // TODO: scoped ApplyDebugLocation DL(*this, Cond);
  // TODO: __builtin_unpredictable and profile counts?
  cond = cond->IgnoreParens();
  mlir::Value condV = evaluateExprAsBool(cond);
  mlir::LogicalResult resThen = mlir::success(), resElse = mlir::success();

  builder.create<mlir::cir::IfOp>(
      loc, condV, elseS,
      /*thenBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        resThen = buildStmt(thenS, /*useCurrentScope=*/true);
        builder.create<YieldOp>(getLoc(thenS->getSourceRange().getEnd()));
      },
      /*elseBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        resElse = buildStmt(elseS, /*useCurrentScope=*/true);
        builder.create<YieldOp>(getLoc(elseS->getSourceRange().getEnd()));
      });

  return mlir::LogicalResult::success(resThen.succeeded() &&
                                      resElse.succeeded());
}

mlir::LogicalResult CIRGenModule::buildIfStmt(const IfStmt &S) {
  // The else branch of a consteval if statement is always the only branch
  // that can be runtime evaluated.
  assert(!S.isConsteval() && "not implemented");
  mlir::LogicalResult res = mlir::success();

  // C99 6.8.4.1: The first substatement is executed if the expression
  // compares unequal to 0.  The condition must be a scalar type.
  auto ifStmtBuilder = [&]() -> mlir::LogicalResult {
    if (S.getInit())
      if (buildStmt(S.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();

    if (S.getConditionVariable())
      buildDecl(*S.getConditionVariable());

    // If the condition constant folds and can be elided, try to avoid
    // emitting the condition and the dead arm of the if/else.
    // FIXME: should this be done as part of a constant folder pass instead?
    bool CondConstant;
    if (ConstantFoldsToSimpleInteger(S.getCond(), CondConstant,
                                     S.isConstexpr())) {
      assert(0 && "not implemented");
    }

    // TODO: PGO and likelihood.
    // The mlir::Location for cir.if skips the init/cond part of IfStmt,
    // and effectively spans from "then-begin" to "else-end||then-end".
    auto ifLocStart = getLoc(S.getThen()->getSourceRange().getBegin());
    auto ifLocEnd = getLoc(S.getSourceRange().getEnd());
    return buildIfOnBoolExpr(S.getCond(), getLoc(ifLocStart, ifLocEnd),
                             S.getThen(), S.getElse());
  };

  // TODO: Add a new scoped symbol table.
  // LexicalScope ConditionScope(*this, S.getCond()->getSourceRange());
  // The if scope contains the full source range for IfStmt.
  auto scopeLoc = getLoc(S.getSourceRange());
  auto scopeLocEnd = getLoc(S.getSourceRange().getEnd());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, mlir::TypeRange(), /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        res = ifStmtBuilder();
        builder.create<YieldOp>(scopeLocEnd);
      });

  return res;
}

// Build CIR for a statement. useCurrentScope should be true if no
// new scopes need be created when finding a compound statement.
mlir::LogicalResult CIRGenModule::buildStmt(const Stmt *S,
                                            bool useCurrentScope) {
  if (mlir::succeeded(buildSimpleStmt(S, useCurrentScope)))
    return mlir::success();

  if (astCtx.getLangOpts().OpenMP && astCtx.getLangOpts().OpenMPSimd)
    assert(0 && "not implemented");

  switch (S->getStmtClass()) {
  case Stmt::OpenACCComputeConstructClass:
  case Stmt::OMPScopeDirectiveClass:
  case Stmt::OMPParallelMaskedDirectiveClass:
  case Stmt::OMPTargetTeamsGenericLoopDirectiveClass:
  case Stmt::OMPTeamsGenericLoopDirectiveClass:
  case Stmt::OMPTargetParallelGenericLoopDirectiveClass:
  case Stmt::OMPParallelGenericLoopDirectiveClass:
  case Stmt::OMPParallelMaskedTaskLoopDirectiveClass:
  case Stmt::OMPParallelMaskedTaskLoopSimdDirectiveClass:
  case Stmt::OMPErrorDirectiveClass:
  case Stmt::OMPMaskedTaskLoopDirectiveClass:
  case Stmt::OMPMaskedTaskLoopSimdDirectiveClass:
    llvm_unreachable("NYI");
  case Stmt::NoStmtClass:
  case Stmt::CXXCatchStmtClass:
  case Stmt::SEHExceptStmtClass:
  case Stmt::SEHFinallyStmtClass:
  case Stmt::MSDependentExistsStmtClass:
    llvm_unreachable("invalid statement class to emit generically");
  case Stmt::NullStmtClass:
  case Stmt::CompoundStmtClass:
  case Stmt::DeclStmtClass:
  case Stmt::LabelStmtClass:
  case Stmt::AttributedStmtClass:
  case Stmt::GotoStmtClass:
  case Stmt::BreakStmtClass:
  case Stmt::ContinueStmtClass:
  case Stmt::DefaultStmtClass:
  case Stmt::CaseStmtClass:
  case Stmt::SEHLeaveStmtClass:
    llvm_unreachable("should have emitted these statements as simple");

#define STMT(Type, Base)
#define ABSTRACT_STMT(Op)
#define EXPR(Type, Base) case Stmt::Type##Class:
#include "clang/AST/StmtNodes.inc"
    {
      // Remember the block we came in on.
      mlir::Block *incoming = builder.getInsertionBlock();
      assert(incoming && "expression emission must have an insertion point");

      buildIgnoredExpr(cast<Expr>(S));

      mlir::Block *outgoing = builder.getInsertionBlock();
      assert(outgoing && "expression emission cleared block!");

      // FIXME: Should we mimic LLVM emission here?
      // The expression emitters assume (reasonably!) that the insertion
      // point is always set.  To maintain that, the call-emission code
      // for noreturn functions has to enter a new block with no
      // predecessors.  We want to kill that block and mark the current
      // insertion point unreachable in the common case of a call like
      // "exit();".  Since expression emission doesn't otherwise create
      // blocks with no predecessors, we can just test for that.
      // However, we must be careful not to do this to our incoming
      // block, because *statement* emission does sometimes create
      // reachable blocks which will have no predecessors until later in
      // the function.  This occurs with, e.g., labels that are not
      // reachable by fallthrough.
      if (incoming != outgoing && outgoing->use_empty())
        assert(0 && "not implemented");
      break;
    }

  case Stmt::IfStmtClass:
    if (buildIfStmt(cast<IfStmt>(*S)).failed())
      return mlir::failure();
    break;
  case Stmt::IndirectGotoStmtClass:
  case Stmt::WhileStmtClass:
  case Stmt::DoStmtClass:
  case Stmt::ForStmtClass:
  case Stmt::ReturnStmtClass:
  case Stmt::SwitchStmtClass:
  // When implemented, GCCAsmStmtClass should fall-through to MSAsmStmtClass.
  case Stmt::GCCAsmStmtClass:
  case Stmt::MSAsmStmtClass:
  case Stmt::CoroutineBodyStmtClass:
  case Stmt::CoreturnStmtClass:
  case Stmt::CapturedStmtClass:
  case Stmt::ObjCAtTryStmtClass:
  case Stmt::ObjCAtThrowStmtClass:
  case Stmt::ObjCAtSynchronizedStmtClass:
  case Stmt::ObjCForCollectionStmtClass:
  case Stmt::ObjCAutoreleasePoolStmtClass:
  case Stmt::CXXTryStmtClass:
  case Stmt::CXXForRangeStmtClass:
  case Stmt::SEHTryStmtClass:
  case Stmt::OMPMetaDirectiveClass:
  case Stmt::OMPCanonicalLoopClass:
  case Stmt::OMPParallelDirectiveClass:
  case Stmt::OMPSimdDirectiveClass:
  case Stmt::OMPTileDirectiveClass:
  case Stmt::OMPUnrollDirectiveClass:
  case Stmt::OMPForDirectiveClass:
  case Stmt::OMPForSimdDirectiveClass:
  case Stmt::OMPSectionsDirectiveClass:
  case Stmt::OMPSectionDirectiveClass:
  case Stmt::OMPSingleDirectiveClass:
  case Stmt::OMPMasterDirectiveClass:
  case Stmt::OMPCriticalDirectiveClass:
  case Stmt::OMPParallelForDirectiveClass:
  case Stmt::OMPParallelForSimdDirectiveClass:
  case Stmt::OMPParallelMasterDirectiveClass:
  case Stmt::OMPParallelSectionsDirectiveClass:
  case Stmt::OMPTaskDirectiveClass:
  case Stmt::OMPTaskyieldDirectiveClass:
  case Stmt::OMPBarrierDirectiveClass:
  case Stmt::OMPTaskwaitDirectiveClass:
  case Stmt::OMPTaskgroupDirectiveClass:
  case Stmt::OMPFlushDirectiveClass:
  case Stmt::OMPDepobjDirectiveClass:
  case Stmt::OMPScanDirectiveClass:
  case Stmt::OMPOrderedDirectiveClass:
  case Stmt::OMPAtomicDirectiveClass:
  case Stmt::OMPTargetDirectiveClass:
  case Stmt::OMPTeamsDirectiveClass:
  case Stmt::OMPCancellationPointDirectiveClass:
  case Stmt::OMPCancelDirectiveClass:
  case Stmt::OMPTargetDataDirectiveClass:
  case Stmt::OMPTargetEnterDataDirectiveClass:
  case Stmt::OMPTargetExitDataDirectiveClass:
  case Stmt::OMPTargetParallelDirectiveClass:
  case Stmt::OMPTargetParallelForDirectiveClass:
  case Stmt::OMPTaskLoopDirectiveClass:
  case Stmt::OMPTaskLoopSimdDirectiveClass:
  case Stmt::OMPMasterTaskLoopDirectiveClass:
  case Stmt::OMPMasterTaskLoopSimdDirectiveClass:
  case Stmt::OMPParallelMasterTaskLoopDirectiveClass:
  case Stmt::OMPParallelMasterTaskLoopSimdDirectiveClass:
  case Stmt::OMPDistributeDirectiveClass:
  case Stmt::OMPTargetUpdateDirectiveClass:
  case Stmt::OMPDistributeParallelForDirectiveClass:
  case Stmt::OMPDistributeParallelForSimdDirectiveClass:
  case Stmt::OMPDistributeSimdDirectiveClass:
  case Stmt::OMPTargetParallelForSimdDirectiveClass:
  case Stmt::OMPTargetSimdDirectiveClass:
  case Stmt::OMPTeamsDistributeDirectiveClass:
  case Stmt::OMPTeamsDistributeSimdDirectiveClass:
  case Stmt::OMPTeamsDistributeParallelForSimdDirectiveClass:
  case Stmt::OMPTeamsDistributeParallelForDirectiveClass:
  case Stmt::OMPTargetTeamsDirectiveClass:
  case Stmt::OMPTargetTeamsDistributeDirectiveClass:
  case Stmt::OMPTargetTeamsDistributeParallelForDirectiveClass:
  case Stmt::OMPTargetTeamsDistributeParallelForSimdDirectiveClass:
  case Stmt::OMPTargetTeamsDistributeSimdDirectiveClass:
  case Stmt::OMPInteropDirectiveClass:
  case Stmt::OMPDispatchDirectiveClass:
  case Stmt::OMPGenericLoopDirectiveClass:
  case Stmt::OMPMaskedDirectiveClass: {
    llvm::errs() << "CIR codegen for '" << S->getStmtClassName()
                 << "' not implemented\n";
    assert(0 && "not implemented");
    break;
  }
  case Stmt::ObjCAtCatchStmtClass:
    llvm_unreachable(
        "@catch statements should be handled by EmitObjCAtTryStmt");
  case Stmt::ObjCAtFinallyStmtClass:
    llvm_unreachable(
        "@finally statements should be handled by EmitObjCAtTryStmt");
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenModule::buildFunctionBody(const Stmt *Body) {
  const CompoundStmt *S = dyn_cast<CompoundStmt>(Body);
  assert(S && "expected compound stmt");

  // We start with function level scope for variables.
  SymTableScopeTy varScope(symbolTable);
  return buildCompoundStmtWithoutScope(*S);
}

mlir::LogicalResult CIRGenModule::buildCompoundStmt(const CompoundStmt &S) {
  mlir::LogicalResult res = mlir::success();

  auto compoundStmtBuilder = [&]() -> mlir::LogicalResult {
    if (buildCompoundStmtWithoutScope(S).failed())
      return mlir::failure();

    return mlir::success();
  };

  // Add local scope to track new declared variables.
  SymTableScopeTy varScope(symbolTable);
  auto locBegin = getLoc(S.getSourceRange().getBegin());
  auto locEnd = getLoc(S.getSourceRange().getEnd());
  builder.create<mlir::cir::ScopeOp>(
      locBegin, mlir::TypeRange(), /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        res = compoundStmtBuilder();
        builder.create<YieldOp>(locEnd);
      });

  return res;
}

mlir::LogicalResult
CIRGenModule::buildCompoundStmtWithoutScope(const CompoundStmt &S) {
  for (auto *CurStmt : S.body())
    if (buildStmt(CurStmt, /*useCurrentScope=*/false).failed())
      return mlir::failure();

  return mlir::success();
}

void CIRGenModule::buildTopLevelDecl(Decl *decl) {
  switch (decl->getKind()) {
  default:
    assert(false && "Not yet implemented");
  case Decl::Function:
    buildFunction(cast<FunctionDecl>(decl));
    break;
  case Decl::CXXRecord: {
    CXXRecordDecl *crd = cast<CXXRecordDecl>(decl);
    // TODO: Handle debug info as CodeGenModule.cpp does
    for (auto *childDecl : crd->decls())
      if (isa<VarDecl>(childDecl) || isa<CXXRecordDecl>(childDecl))
        buildTopLevelDecl(childDecl);
    break;
  }
  case Decl::Record:
    // There's nothing to do here, we emit everything pertaining to `Record`s
    // lazily.
    // TODO: handle debug info here? See clang's
    // CodeGenModule::EmitTopLevelDecl
    break;
  }
}

mlir::FuncOp CIRGenModule::buildFunction(const FunctionDecl *FD) {
  CIRGenFunction CGF;
  CurCGF = &CGF;

  // Create a scope in the symbol table to hold variable declarations.
  SymTableScopeTy varScope(symbolTable);

  const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD);
  assert(!MD && "methods not implemented");
  auto fnLoc = getLoc(FD->getSourceRange());

  // Create an MLIR function for the given prototype.
  llvm::SmallVector<mlir::Type, 4> argTypes;

  for (auto *Param : FD->parameters())
    argTypes.push_back(getCIRType(Param->getType()));

  CurCGF->FnRetQualTy = FD->getReturnType();
  auto funcType =
      builder.getFunctionType(argTypes, CurCGF->FnRetQualTy->isVoidType()
                                            ? mlir::TypeRange()
                                            : getCIRType(CurCGF->FnRetQualTy));
  mlir::FuncOp function = mlir::FuncOp::create(fnLoc, FD->getName(), funcType);
  if (!function)
    return nullptr;

  // In MLIR the entry block of the function is special: it must have the
  // same argument list as the function itself.
  auto &entryBlock = *function.addEntryBlock();

  // Set the insertion point in the builder to the beginning of the
  // function body, it will be used throughout the codegen to create
  // operations in this function.
  builder.setInsertionPointToStart(&entryBlock);

  // Declare all the function arguments in the symbol table.
  for (const auto nameValue :
       llvm::zip(FD->parameters(), entryBlock.getArguments())) {
    auto *paramVar = std::get<0>(nameValue);
    auto paramVal = std::get<1>(nameValue);
    auto alignment = astCtx.getDeclAlign(paramVar);
    auto paramLoc = getLoc(paramVar->getSourceRange());
    paramVal.setLoc(paramLoc);

    mlir::Value addr;
    if (failed(declare(paramVar, paramVar->getType(), paramLoc, alignment, addr,
                       true /*param*/)))
      return nullptr;
    // Location of the store to the param storage tracked as beginning of
    // the function body.
    auto fnBodyBegin = getLoc(FD->getBody()->getBeginLoc());
    builder.create<mlir::cir::StoreOp>(fnBodyBegin, paramVal, addr);
  }

  // Emit the body of the function.
  if (mlir::failed(buildFunctionBody(FD->getBody()))) {
    function.erase();
    return nullptr;
  }

  ReturnOp returnOp;
  if (!entryBlock.empty())
    returnOp = dyn_cast<ReturnOp>(entryBlock.back());
  if (!returnOp)
    builder.create<ReturnOp>(getLoc(FD->getBody()->getEndLoc()));

  if (mlir::failed(function.verifyBody()))
    return nullptr;
  theModule.push_back(function);
  return function;
}

mlir::Type CIRGenModule::getCIRType(const QualType &type) {
  return genTypes->ConvertType(type);
}

void CIRGenModule::verifyModule() {
  // Verify the module after we have finished constructing it, this will
  // check the structural properties of the IR and invoke any specific
  // verifiers we have on the CIR operations.
  if (failed(mlir::verify(theModule)))
    theModule.emitError("module verification error");
}
