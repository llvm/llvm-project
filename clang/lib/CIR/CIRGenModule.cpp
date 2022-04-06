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

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenTypes.h"
#include "CIRGenValue.h"
#include "TargetInfo.h"

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
#include "clang/AST/GlobalDecl.h"
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
#include <iterator>
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

static CIRGenCXXABI *createCXXABI(CIRGenModule &CGM) {
  switch (CGM.getASTContext().getCXXABIKind()) {
  case TargetCXXABI::GenericItanium:
    return CreateItaniumCXXABI(CGM);
  default:
    llvm_unreachable("invalid C++ ABI kind");
  }
}

CIRGenModule::CIRGenModule(mlir::MLIRContext &context,
                           clang::ASTContext &astctx,
                           const clang::CodeGenOptions &CGO)
    : builder(&context), astCtx(astctx), langOpts(astctx.getLangOpts()),
      codeGenOpts(CGO),
      theModule{mlir::ModuleOp::create(builder.getUnknownLoc())},
      target(astCtx.getTargetInfo()), ABI(createCXXABI(*this)),
      genTypes{*this} {}

CIRGenModule::~CIRGenModule() {}

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

mlir::Value CIRGenModule::buildAlloca(StringRef name, InitStyle initStyle,
                                      QualType ty, mlir::Location loc,
                                      CharUnits alignment) {
  // Allocas are expected to be in the beginning of the entry block
  // for most of the regions.
  // FIXME: for non-scoped C/C++ switch case regions, alloca's should
  // go to the entry block of the switch scope, not of the case region.
  auto getAllocaInsertPositionOp =
      [&](mlir::Block **insertBlock) -> mlir::Operation * {
    auto *parentBlock = builder.getInsertionBlock();
    mlir::Region *r = parentBlock->getParent();
    assert(r->getBlocks().size() > 0 && "assume at least one block exists");
    mlir::Block &entryBlock = *r->begin();

    if (parentBlock != &entryBlock)
      parentBlock = &entryBlock;

    auto lastAlloca = std::find_if(
        parentBlock->rbegin(), parentBlock->rend(),
        [](mlir::Operation &op) { return isa<mlir::cir::AllocaOp>(&op); });

    *insertBlock = parentBlock;
    if (lastAlloca == parentBlock->rend())
      return nullptr;
    return &*lastAlloca;
  };

  auto localVarTy = getCIRType(ty);
  auto localVarPtrTy =
      mlir::cir::PointerType::get(builder.getContext(), localVarTy);

  auto alignIntAttr =
      mlir::IntegerAttr::get(mlir::IntegerType::get(builder.getContext(), 64),
                             alignment.getQuantity());

  mlir::Value addr;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *insertBlock = nullptr;
    mlir::Operation *insertOp = getAllocaInsertPositionOp(&insertBlock);

    if (insertOp)
      builder.setInsertionPointAfter(insertOp);
    else {
      assert(insertBlock && "expected valid insertion block");
      // No previous alloca found, place this one in the beginning
      // of the block.
      builder.setInsertionPointToStart(insertBlock);
    }

    addr = builder.create<mlir::cir::AllocaOp>(loc, /*addr type*/ localVarPtrTy,
                                               /*var type*/ localVarTy, name,
                                               initStyle, alignIntAttr);
  }
  return addr;
}

void CIRGenModule::buildAndUpdateRetAlloca(QualType ty, mlir::Location loc,
                                           CharUnits alignment) {
  auto addr =
      buildAlloca("__retval", InitStyle::uninitialized, ty, loc, alignment);
  CurCGF->FnRetAlloca = addr;
}

mlir::LogicalResult CIRGenModule::declare(const Decl *var, QualType ty,
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
  emission.Addr = Address{addr, alignment};
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

void CIRGenModule::buildStoreOfScalar(mlir::Value Value, Address Addr,
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

  const Address Loc = emission.Addr;

  // Note: constexpr already initializes everything correctly.
  LangOptions::TrivialAutoVarInitKind trivialAutoVarInit =
      (D.isConstexpr()
           ? LangOptions::TrivialAutoVarInitKind::Uninitialized
           : (D.getAttr<UninitializedAttr>()
                  ? LangOptions::TrivialAutoVarInitKind::Uninitialized
                  : astCtx.getLangOpts().getTrivialAutoVarInit()));

  auto initializeWhatIsTechnicallyUninitialized = [&](Address Loc) {
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

mlir::LogicalResult CIRGenModule::buildReturnStmt(const ReturnStmt &S) {
  assert(!(astCtx.getLangOpts().ElideConstructors && S.getNRVOCandidate() &&
           S.getNRVOCandidate()->isNRVOVariable()) &&
         "unimplemented");
  assert(!CurCGF->FnRetQualTy->isReferenceType() && "unimplemented");
  auto loc = getLoc(S.getSourceRange());

  // Emit the result value, even if unused, to evaluate the side effects.
  const Expr *RV = S.getRetValue();
  if (RV) {
    assert(!isa<ExprWithCleanups>(RV) && "unimplemented");

    mlir::Value V = nullptr;
    switch (CIRGenFunction::getEvaluationKind(RV->getType())) {
    case TEK_Scalar:
      V = buildScalarExpr(RV);
      builder.create<mlir::cir::StoreOp>(loc, V, *CurCGF->FnRetAlloca);
      break;
    case TEK_Complex:
    case TEK_Aggregate:
      llvm::errs() << "ReturnStmt EvaluationKind not implemented\n";
      return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    if (!V || (RV && RV->getType()->isVoidType())) {
      // FIXME: evaluate for side effects.
    }
  } else {
    // Do nothing (return value is left uninitialized), this is also
    // the path when returning from void functions.
  }

  // Create a new return block (if not existent) and add a branch to
  // it. The actual return instruction is only inserted during current
  // scope cleanup handling.
  auto *retBlock = currLexScope->getOrCreateRetBlock(*this, loc);
  builder.create<BrOp>(loc, retBlock);

  // Insert the new block to continue codegen after branch to ret block.
  builder.createBlock(builder.getBlock()->getParent());
  return mlir::success();
}

mlir::LogicalResult CIRGenModule::buildDeclStmt(const DeclStmt &S) {
  if (!builder.getInsertionBlock()) {
    theModule.emitError(
        "Seems like this is unreachable code, what should we do?");
    return mlir::failure();
  }

  for (const auto *I : S.decls()) {
    buildDecl(*I);
  }

  return mlir::success();
}

/// Build a unconditional branch to the lexical scope cleanup block
/// or with the labeled blocked if already solved.
///
/// Track on scope basis, goto's we need to fix later.
mlir::LogicalResult
CIRGenModule::buildBranchThroughCleanup(JumpDest &Dest, LabelDecl *L,
                                        mlir::Location Loc) {
  // Remove this once we go for making sure unreachable code is
  // well modeled (or not).
  assert(builder.getInsertionBlock() && "not yet implemented");

  // Insert a branch: to the cleanup block (unsolved) or to the already
  // materialized label. Keep track of unsolved goto's.
  mlir::Block *DstBlock = Dest.getBlock();
  auto G = builder.create<BrOp>(
      Loc, Dest.isValid() ? DstBlock
                          : currLexScope->getOrCreateCleanupBlock(builder));
  if (!Dest.isValid())
    currLexScope->PendingGotos.push_back(std::make_pair(G, L));

  return mlir::success();
}

/// All scope related cleanup needed:
/// - Patching up unsolved goto's.
/// - Build all cleanup code and insert yield/returns.
void CIRGenModule::LexicalScopeGuard::cleanup() {
  auto &builder = CGM.builder;
  auto *localScope = CGM.currLexScope;

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
    g.setSuccessor(CGM.LabelMap[gotoInfo.second].getBlock());
    localScope->PendingGotos.pop_back();
  }
  localScope->SolvedLabels.clear();

  // Cleanup are done right before codegen resume a scope. This is where
  // objects are destroyed.
  if (localScope->RetBlock) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(localScope->RetBlock);

    // TODO: insert actual scope cleanup HERE (dtors and etc)

    // If there's anything to return, load it first.
    if (CGM.CurCGF->FnRetTy.has_value()) {
      auto val = builder.create<LoadOp>(
          *localScope->RetLoc, *CGM.CurCGF->FnRetTy, *CGM.CurCGF->FnRetAlloca);
      builder.create<ReturnOp>(*localScope->RetLoc, ArrayRef(val.getResult()));
    } else {
      builder.create<ReturnOp>(*localScope->RetLoc);
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

mlir::LogicalResult CIRGenModule::buildGotoStmt(const GotoStmt &S) {
  // FIXME: LLVM codegen inserts emit stop point here for debug info
  // sake when the insertion point is available, but doesn't do
  // anything special when there isn't. We haven't implemented debug
  // info support just yet, look at this again once we have it.
  assert(builder.getInsertionBlock() && "not yet implemented");

  // A goto marks the end of a block, create a new one for codegen after
  // buildGotoStmt can resume building in that block.

  // Build a cir.br to the target label.
  auto &JD = LabelMap[S.getLabel()];
  if (buildBranchThroughCleanup(JD, S.getLabel(), getLoc(S.getSourceRange()))
          .failed())
    return mlir::failure();

  // Insert the new block to continue codegen after goto.
  builder.createBlock(builder.getBlock()->getParent());

  // What here...
  return mlir::success();
}

mlir::LogicalResult CIRGenModule::buildLabel(const LabelDecl *D) {
  JumpDest &Dest = LabelMap[D];

  // Create a new block to tag with a label and add a branch from
  // the current one to it. If the block is empty just call attach it
  // to this label.
  mlir::Block *currBlock = builder.getBlock();
  mlir::Block *labelBlock = currBlock;
  if (!currBlock->empty()) {

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      labelBlock = builder.createBlock(builder.getBlock()->getParent());
    }

    builder.create<BrOp>(getLoc(D->getSourceRange()), labelBlock);
    builder.setInsertionPointToEnd(labelBlock);
  }

  if (!Dest.isValid()) {
    Dest.Block = labelBlock;
    currLexScope->SolvedLabels.insert(D);
    // FIXME: add a label attribute to block...
  } else {
    assert(0 && "unimplemented");
  }

  //  FIXME: emit debug info for labels, incrementProfileCounter
  return mlir::success();
}

mlir::LogicalResult CIRGenModule::buildLabelStmt(const clang::LabelStmt &S) {
  if (buildLabel(S.getDecl()).failed())
    return mlir::failure();

  // IsEHa: not implemented.
  assert(!(astCtx.getLangOpts().EHAsynch && S.isSideEntry()));

  return buildStmt(S.getSubStmt(), /* useCurrentScope */ true);
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
  case Stmt::GotoStmtClass:
    return buildGotoStmt(cast<GotoStmt>(*S));

  case Stmt::NullStmtClass:
    break;

  case Stmt::LabelStmtClass:
    return buildLabelStmt(cast<LabelStmt>(*S));

  case Stmt::CaseStmtClass:
    assert(0 &&
           "Should not get here, currently handled directly from SwitchStmt");
    break;

  case Stmt::BreakStmtClass:
    return buildBreakStmt(cast<BreakStmt>(*S));

  case Stmt::AttributedStmtClass:
  case Stmt::ContinueStmtClass:
  case Stmt::DefaultStmtClass:
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

    LValue LV = LValue::makeAddr(Address(V, CharUnits::fromQuantity(4)),
                                 VD->getType(), AlignmentSource::Decl);
    return LV;
  }

  llvm_unreachable("Unhandled DeclRefExpr?");
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

    RValue RV = CurCGF->buildAnyExpr(E->getRHS());
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
Address CIRGenModule::buildPointerWithAlignment(const Expr *E,
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
  return Address(buildScalarExpr(E), Align);
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
    Address Addr = buildPointerWithAlignment(E->getSubExpr(), &BaseInfo);

    // Tag 'load' with deref attribute.
    if (auto loadOp =
            dyn_cast<::mlir::cir::LoadOp>(Addr.getPointer().getDefiningOp())) {
      loadOp.setIsDerefAttr(mlir::UnitAttr::get(builder.getContext()));
    }

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

  return LValue::makeAddr(Address::invalid(), E->getType());
}

/// EmitIgnoredExpr - Emit code to compute the specified expression,
/// ignoring the result.
void CIRGenModule::buildIgnoredExpr(const Expr *E) {
  if (E->isPRValue())
    return (void)CurCGF->buildAnyExpr(E);

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
        // FIXME: abstract all this massive location handling elsewhere.
        SmallVector<mlir::Location, 2> locs;
        if (loc.isa<mlir::FileLineColLoc>()) {
          locs.push_back(loc);
          locs.push_back(loc);
        } else if (loc.isa<mlir::FusedLoc>()) {
          auto fusedLoc = loc.cast<mlir::FusedLoc>();
          locs.push_back(fusedLoc.getLocations()[0]);
          locs.push_back(fusedLoc.getLocations()[1]);
        }
        LexicalScopeContext lexScope{locs[0], locs[1],
                                     builder.getInsertionBlock()};
        LexicalScopeGuard lexThenGuard{*this, &lexScope};
        resThen = buildStmt(thenS, /*useCurrentScope=*/true);
      },
      /*elseBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto fusedLoc = loc.cast<mlir::FusedLoc>();
        auto locBegin = fusedLoc.getLocations()[2];
        auto locEnd = fusedLoc.getLocations()[3];
        LexicalScopeContext lexScope{locBegin, locEnd,
                                     builder.getInsertionBlock()};
        LexicalScopeGuard lexElseGuard{*this, &lexScope};
        resElse = buildStmt(elseS, /*useCurrentScope=*/true);
      });

  return mlir::LogicalResult::success(resThen.succeeded() &&
                                      resElse.succeeded());
}

static mlir::Location getIfLocs(CIRGenModule &CGM, const clang::Stmt *thenS,
                                const clang::Stmt *elseS) {
  // Attempt to be more accurate as possible with IfOp location, generate
  // one fused location that has either 2 or 4 total locations, depending
  // on else's availability.
  SmallVector<mlir::Location, 4> ifLocs;
  mlir::Attribute metadata;

  clang::SourceRange t = thenS->getSourceRange();
  ifLocs.push_back(CGM.getLoc(t.getBegin()));
  ifLocs.push_back(CGM.getLoc(t.getEnd()));
  if (elseS) {
    clang::SourceRange e = elseS->getSourceRange();
    ifLocs.push_back(CGM.getLoc(e.getBegin()));
    ifLocs.push_back(CGM.getLoc(e.getEnd()));
  }

  return mlir::FusedLoc::get(ifLocs, metadata, CGM.getBuilder().getContext());
}

mlir::LogicalResult CIRGenModule::buildBreakStmt(const clang::BreakStmt &S) {
  builder.create<YieldOp>(
      getLoc(S.getBreakLoc()),
      mlir::cir::YieldOpKindAttr::get(builder.getContext(),
                                      mlir::cir::YieldOpKind::Break),
      mlir::ValueRange({}));
  return mlir::success();
}

mlir::LogicalResult CIRGenModule::buildCaseStmt(const CaseStmt &S,
                                                mlir::Type condType,
                                                CaseAttr &caseEntry) {
  assert((!S.getRHS() || !S.caseStmtIsGNURange()) &&
         "case ranges not implemented");
  auto res = mlir::success();

  auto intVal = S.getLHS()->EvaluateKnownConstInt(getASTContext());
  auto *ctx = builder.getContext();
  caseEntry = mlir::cir::CaseAttr::get(
      ctx, builder.getArrayAttr({}),
      CaseOpKindAttr::get(ctx, mlir::cir::CaseOpKind::Equal));
  {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    res = buildStmt(S.getSubStmt(),
                    /*useCurrentScope=*/!isa<CompoundStmt>(S.getSubStmt()));
  }

  // TODO: likelihood
  return res;
}

mlir::LogicalResult CIRGenModule::buildSwitchStmt(const SwitchStmt &S) {
  // TODO: LLVM codegen does some early optimization to fold the condition and
  // only emit live cases. CIR should use MLIR to achieve similar things,
  // nothing to be done here.
  // if (ConstantFoldsToSimpleInteger(S.getCond(), ConstantCondValue))...

  auto res = mlir::success();
  auto switchStmtBuilder = [&]() -> mlir::LogicalResult {
    if (S.getInit())
      if (buildStmt(S.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();

    if (S.getConditionVariable())
      buildDecl(*S.getConditionVariable());

    mlir::Value condV = buildScalarExpr(S.getCond());

    // TODO: PGO and likelihood (e.g. PGO.haveRegionCounts())
    // TODO: if the switch has a condition wrapped by __builtin_unpredictable?

    auto terminateCaseRegion = [&](mlir::Region &r, mlir::Location loc) {
      assert(r.getBlocks().size() <= 1 && "not implemented");
      if (r.empty())
        return;

      auto &block = r.back();

      if (block.empty() ||
          !block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        mlir::OpBuilder::InsertionGuard guardCase(builder);
        builder.setInsertionPointToEnd(&block);
        builder.create<YieldOp>(
            loc,
            mlir::cir::YieldOpKindAttr::get(
                builder.getContext(), mlir::cir::YieldOpKind::Fallthrough),
            mlir::ValueRange({}));
      }
    };

    // FIXME: track switch to handle nested stmts.
    auto swop = builder.create<SwitchOp>(
        getLoc(S.getBeginLoc()), condV,
        /*switchBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::OperationState &os) {
          auto *cs = dyn_cast<CompoundStmt>(S.getBody());
          assert(cs && "expected compound stmt");
          SmallVector<mlir::Attribute, 4> caseAttrs;

          mlir::Block *lastCaseBlock = nullptr;
          for (auto *c : cs->body()) {
            auto *newCase = dyn_cast<CaseStmt>(c);
            if (!newCase) {
              // This means it's a random stmt following up a case, just
              // emit it as part of previous known case.
              assert(lastCaseBlock && "expects pre-existing case block");
              mlir::OpBuilder::InsertionGuard guardCase(builder);
              builder.setInsertionPointToEnd(lastCaseBlock);
              res = buildStmt(c, /*useCurrentScope=*/!isa<CompoundStmt>(c));
              continue;
            }
            assert(newCase && "expected case stmt");
            const CaseStmt *nestedCase =
                dyn_cast<CaseStmt>(newCase->getSubStmt());
            assert(!nestedCase && "empty case fallthrough NYI");

            CaseAttr caseAttr;
            {
              mlir::OpBuilder::InsertionGuard guardCase(builder);
              mlir::Region *caseRegion = os.addRegion();
              lastCaseBlock = builder.createBlock(caseRegion);
              res = buildCaseStmt(*newCase, condV.getType(), caseAttr);
              if (res.failed())
                break;
            }
            caseAttrs.push_back(caseAttr);
          }

          os.addAttribute("cases", builder.getArrayAttr(caseAttrs));
        });

    // Make sure all case regions are terminated by inserting
    // fallthroughs when necessary.
    // FIXME: find a better way to get accurante with location here.
    for (auto &r : swop.getRegions())
      terminateCaseRegion(r, swop.getLoc());
    return res;
  };

  // The switch scope contains the full source range for SwitchStmt.
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, mlir::TypeRange(), /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto fusedLoc = loc.cast<mlir::FusedLoc>();
        auto scopeLocBegin = fusedLoc.getLocations()[0];
        auto scopeLocEnd = fusedLoc.getLocations()[1];
        LexicalScopeContext lexScope{scopeLocBegin, scopeLocEnd,
                                     builder.getInsertionBlock()};
        LexicalScopeGuard lexIfScopeGuard{*this, &lexScope};
        res = switchStmtBuilder();
      });

  return res;
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
    auto ifLoc = getIfLocs(*this, S.getThen(), S.getElse());
    return buildIfOnBoolExpr(S.getCond(), ifLoc, S.getThen(), S.getElse());
  };

  // TODO: Add a new scoped symbol table.
  // LexicalScope ConditionScope(*this, S.getCond()->getSourceRange());
  // The if scope contains the full source range for IfStmt.
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, mlir::TypeRange(), /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto fusedLoc = loc.cast<mlir::FusedLoc>();
        auto scopeLocBegin = fusedLoc.getLocations()[0];
        auto scopeLocEnd = fusedLoc.getLocations()[1];
        LexicalScopeContext lexScope{scopeLocBegin, scopeLocEnd,
                                     builder.getInsertionBlock()};
        LexicalScopeGuard lexIfScopeGuard{*this, &lexScope};
        res = ifStmtBuilder();
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
  case Stmt::SwitchStmtClass:
    if (buildSwitchStmt(cast<SwitchStmt>(*S)).failed())
      return mlir::failure();
    break;
  case Stmt::IndirectGotoStmtClass:
  case Stmt::WhileStmtClass:
  case Stmt::DoStmtClass:
  case Stmt::ForStmtClass:
  case Stmt::ReturnStmtClass:
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
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, mlir::TypeRange(), /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto fusedLoc = loc.cast<mlir::FusedLoc>();
        auto locBegin = fusedLoc.getLocations()[0];
        auto locEnd = fusedLoc.getLocations()[1];
        LexicalScopeContext lexScope{locBegin, locEnd,
                                     builder.getInsertionBlock()};
        LexicalScopeGuard lexScopeGuard{*this, &lexScope};
        res = compoundStmtBuilder();
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

bool CIRGenModule::MustBeEmitted(const ValueDecl *Global) {
  // Never defer when EmitAllDecls is specified.
  assert(!langOpts.EmitAllDecls && "EmitAllDecls NYI");
  assert(!codeGenOpts.KeepStaticConsts && "KeepStaticConsts NYI");

  return getASTContext().DeclMustBeEmitted(Global);
}

bool CIRGenModule::MayBeEmittedEagerly(const ValueDecl *Global) {
  assert(!langOpts.OpenMP && "NYI");
  auto const *FD = dyn_cast<FunctionDecl>(Global);
  assert(FD && "Only FunctionDecl should hit this path so far.");
  assert(!FD->isTemplated() && "Templates NYI");

  return true;
}

void CIRGenModule::buildGlobal(GlobalDecl GD) {
  const auto *Global = cast<ValueDecl>(GD.getDecl());

  assert(!Global->hasAttr<WeakRefAttr>() && "NYI");
  assert(!Global->hasAttr<IFuncAttr>() && "NYI");
  assert(!Global->hasAttr<CPUDispatchAttr>() && "NYI");
  assert(!langOpts.CUDA && "NYI");
  assert(!langOpts.OpenMP && "NYI");

  const auto *FD = dyn_cast<FunctionDecl>(Global);
  assert(FD && "Only FunctionDecl supported as of here");
  if (!FD->doesThisDeclarationHaveABody()) {
    assert(!FD->doesDeclarationForceExternallyVisibleDefinition() && "NYI");
    return;
  }

  assert(MustBeEmitted(Global) ||
         MayBeEmittedEagerly(Global) && "Delayed emission NYI");

  buildFunction(cast<FunctionDecl>(GD.getDecl()));
}
void CIRGenModule::buildTopLevelDecl(Decl *decl) {
  switch (decl->getKind()) {
  default:
    assert(false && "Not yet implemented");
  case Decl::Function:
    buildGlobal(cast<FunctionDecl>(decl));
    assert(!codeGenOpts.CoverageMapping && "Coverage Mapping NYI");
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
  CIRGenFunction CGF{*this};
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
  mlir::TypeRange FnTyRange = {};
  if (!CurCGF->FnRetQualTy->isVoidType()) {
    CurCGF->FnRetTy = getCIRType(CurCGF->FnRetQualTy);
    FnTyRange = mlir::TypeRange{*CurCGF->FnRetTy};
  }

  auto funcType = builder.getFunctionType(argTypes, FnTyRange);
  mlir::FuncOp function = mlir::FuncOp::create(fnLoc, FD->getName(), funcType);
  if (!function)
    return nullptr;

  // In MLIR the entry block of the function is special: it must have the
  // same argument list as the function itself.
  mlir::Block *entryBlock = function.addEntryBlock();

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
      auto alignment = astCtx.getDeclAlign(paramVar);
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
    if (CurCGF->FnRetTy.has_value())
      buildAndUpdateRetAlloca(CurCGF->FnRetQualTy, FnEndLoc,
                              getNaturalTypeAlignment(CurCGF->FnRetQualTy));

    // Emit the body of the function.
    if (mlir::failed(buildFunctionBody(FD->getBody()))) {
      function.erase();
      return nullptr;
    }
    assert(builder.getInsertionBlock() && "Should be valid");
  }

  if (mlir::failed(function.verifyBody()))
    return nullptr;
  theModule.push_back(function);

  CurCGF = nullptr;
  return function;
}

mlir::Type CIRGenModule::getCIRType(const QualType &type) {
  return genTypes.ConvertType(type);
}

void CIRGenModule::verifyModule() {
  // Verify the module after we have finished constructing it, this will
  // check the structural properties of the IR and invoke any specific
  // verifiers we have on the CIR operations.
  if (failed(mlir::verify(theModule)))
    theModule.emitError("module verification error");
}

mlir::FuncOp CIRGenModule::GetAddrOfFunction(clang::GlobalDecl GD,
                                             mlir::Type Ty, bool ForVTable,
                                             bool DontDefer,
                                             ForDefinition_t IsForDefinition) {
  assert(!ForVTable && "NYI");
  assert(!DontDefer && "NYI");

  assert(!cast<FunctionDecl>(GD.getDecl())->isConsteval() &&
         "consteval function should never be emitted");

  assert(!Ty && "No code paths implemented that have this set yet");
  const auto *FD = cast<FunctionDecl>(GD.getDecl());
  Ty = getTypes().ConvertType(FD->getType());

  assert(!dyn_cast<CXXDestructorDecl>(GD.getDecl()) && "NYI");

  StringRef MangledName = getMangledName(GD);
  auto F = GetOrCreateCIRFunction(MangledName, Ty, GD, ForVTable, DontDefer,
                                  /*IsThunk=*/false, IsForDefinition);

  assert(!langOpts.CUDA && "NYI");

  return F;
}

static std::string getMangledNameImpl(CIRGenModule &CGM, GlobalDecl GD,
                                      const NamedDecl *ND,
                                      bool OmitMultiVersionMangling = false) {
  assert(!OmitMultiVersionMangling && "NYI");

  SmallString<256> Buffer;

  llvm::raw_svector_ostream Out(Buffer);
  MangleContext &MC = CGM.getCXXABI().getMangleContext();

  // TODO: support the module name hash
  auto ShouldMangle = MC.shouldMangleDeclName(ND);
  assert(!ShouldMangle && "Mangling not actually implemented yet.");

  auto *II = ND->getIdentifier();
  assert(II && "Attempt to mangle unnamed decl.");

  const auto *FD = dyn_cast<FunctionDecl>(ND);
  assert(FD && "Only FunctionDecl supported");
  assert(FD->getType()->castAs<FunctionType>()->getCallConv() !=
             CC_X86RegCall &&
         "NYI");
  assert(!FD->hasAttr<CUDAGlobalAttr>() && "NYI");

  Out << II->getName();

  assert(!ShouldMangle && "Mangling not actually implemented yet.");

  if (const auto *FD = dyn_cast<FunctionDecl>(ND)) {
    assert(!FD->isMultiVersion() && "NYI");
  }
  assert(!CGM.getLangOpts().GPURelocatableDeviceCode && "NYI");

  return std::string(Out.str());
}

StringRef CIRGenModule::getMangledName(GlobalDecl GD) {
  auto CanonicalGD = GD.getCanonicalDecl();
  assert(!dyn_cast<CXXConstructorDecl>(CanonicalGD.getDecl()) && "NYI");
  assert(!langOpts.CUDAIsDevice && "NYI");

  // Keep the first result in the case of a mangling collision.
  const auto *ND = cast<NamedDecl>(GD.getDecl());
  std::string MangledName = getMangledNameImpl(*this, GD, ND);

  auto Result = Manglings.insert(std::make_pair(MangledName, GD));
  return MangledDeclNames[CanonicalGD] = Result.first->first();
}

/// GetOrCreateCIRFunction - If the specified mangled name is not in the module,
/// create and return a CIR Function with the specified type. If there is
/// something in the module with the specified name, return it potentially
/// bitcasted to the right type.
///
/// If D is non-null, it specifies a decl that corresponded to this. This is
/// used to set the attributes on the function when it is first created.
mlir::FuncOp CIRGenModule::GetOrCreateCIRFunction(
    StringRef MangledName, mlir::Type Ty, GlobalDecl GD, bool ForVTable,
    bool DontDefer, bool IsThunk, ForDefinition_t IsForDefinition) {
  assert(!ForVTable && "NYI");
  assert(!IsThunk && "NYI");

  const auto *D = GD.getDecl();

  // Any attempts to use a MultiVersion function should result in retrieving the
  // iFunc instead. Name mangling will handle the rest of the changes.
  auto const *FD = cast_or_null<FunctionDecl>(D);
  assert(FD && "Only FD supported so far");

  if (getLangOpts().OpenMP)
    llvm_unreachable("NYI");
  if (FD->isMultiVersion())
    llvm_unreachable("NYI");

  mlir::Value Entry = GetGlobalValue(GD.getDecl());

  if (Entry)
    assert(false && "Code path NYI since we're not yet using this for "
                    "generating fucntion decls");

  // This function doesn't have a complete type (for example, the return type is
  // an incompmlete struct). Use a fake type instead, and make sure not to try
  // to set attributes.
  bool IsIncompleteFunction = false;

  mlir::FunctionType FTy;
  if (Ty.isa<mlir::FunctionType>()) {
    FTy = Ty.cast<mlir::FunctionType>();
  } else {
    assert(false && "NYI");
    // FTy = mlir::FunctionType::get(VoidTy, false);
    IsIncompleteFunction = true;
  }

  auto fnLoc = getLoc(FD->getSourceRange());
  // TODO: CodeGen includeds the linkage (ExternalLinkage) and only passes the
  // mangledname if Entry is nullptr
  mlir::FuncOp F = mlir::FuncOp::create(fnLoc, MangledName, FTy);

  assert(!Entry && "NYI");

  // TODO: This might not be valid, seems the uniqueing system doesn't make
  // sense for MLIR
  // assert(F->getName().getStringRef() == MangledName && "name was uniqued!");

  // TODO: set function attributes from the declaration
  // TODO: set function attributes from the missing attributes param

  // TODO: Handle extra attributes

  assert(!DontDefer && "Only not DontDefer supported so far");

  if (!IsIncompleteFunction) {
    assert(F.getFunctionType() == Ty);
    return F;
  }

  assert(false && "Incompmlete functions NYI");
}

mlir::Value CIRGenModule::GetGlobalValue(const Decl *D) {
  return symbolTable.lookup(D);
}
