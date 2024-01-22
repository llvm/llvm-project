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
#include "CIRGenCstEmitter.h"
#include "CIRGenFunction.h"
#include "CIRGenOpenMPRuntime.h"
#include "CIRGenTypes.h"
#include "CIRGenValue.h"
#include "TargetInfo.h"

#include "UnimplementedFeatureGuarding.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "clang/AST/ASTConsumer.h"
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
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/NoSanitizeList.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/Frontend/FrontendDiagnostic.h"
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

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::SmallVector;
using llvm::StringRef;

static CIRGenCXXABI *createCXXABI(CIRGenModule &CGM) {
  switch (CGM.getASTContext().getCXXABIKind()) {
  case TargetCXXABI::GenericItanium:
  case TargetCXXABI::GenericAArch64:
  case TargetCXXABI::AppleARM64:
    return CreateCIRGenItaniumCXXABI(CGM);
  default:
    llvm_unreachable("invalid C++ ABI kind");
  }
}

CIRGenModule::CIRGenModule(mlir::MLIRContext &context,
                           clang::ASTContext &astctx,
                           const clang::CodeGenOptions &CGO,
                           DiagnosticsEngine &Diags)
    : builder(context, *this), astCtx(astctx), langOpts(astctx.getLangOpts()),
      codeGenOpts(CGO),
      theModule{mlir::ModuleOp::create(builder.getUnknownLoc())}, Diags(Diags),
      target(astCtx.getTargetInfo()), ABI(createCXXABI(*this)), genTypes{*this},
      VTables{*this}, openMPRuntime(new CIRGenOpenMPRuntime(*this)) {

  // Initialize CIR signed integer types cache.
  SInt8Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 8, /*isSigned=*/true);
  SInt16Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 16, /*isSigned=*/true);
  SInt32Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 32, /*isSigned=*/true);
  SInt64Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 64, /*isSigned=*/true);

  // Initialize CIR unsigned integer types cache.
  UInt8Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 8, /*isSigned=*/false);
  UInt16Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 16, /*isSigned=*/false);
  UInt32Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 32, /*isSigned=*/false);
  UInt64Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 64, /*isSigned=*/false);

  VoidTy = ::mlir::cir::VoidType::get(builder.getContext());

  // Initialize CIR pointer types cache.
  VoidPtrTy = ::mlir::cir::PointerType::get(builder.getContext(), VoidTy);

  // TODO: HalfTy
  // TODO: BFloatTy
  FloatTy = builder.getF32Type();
  DoubleTy = builder.getF64Type();
  // TODO(cir): perhaps we should abstract long double variations into a custom
  // cir.long_double type. Said type would also hold the semantics for lowering.
  LongDouble80BitsTy = builder.getF80Type();

  // TODO: PointerWidthInBits
  PointerAlignInBytes =
      astctx
          .toCharUnitsFromBits(
              astctx.getTargetInfo().getPointerAlign(LangAS::Default))
          .getQuantity();
  // TODO: SizeSizeInBytes
  // TODO: IntAlignInBytes
  UCharTy = ::mlir::cir::IntType::get(builder.getContext(),
                                      astCtx.getTargetInfo().getCharWidth(),
                                      /*isSigned=*/false);
  UIntTy = ::mlir::cir::IntType::get(builder.getContext(),
                                     astCtx.getTargetInfo().getIntWidth(),
                                     /*isSigned=*/false);
  UIntPtrTy = ::mlir::cir::IntType::get(
      builder.getContext(), astCtx.getTargetInfo().getMaxPointerWidth(),
      /*isSigned=*/false);
  UInt8PtrTy = builder.getPointerTo(UInt8Ty);
  UInt8PtrPtrTy = builder.getPointerTo(UInt8PtrTy);
  // TODO: AllocaInt8PtrTy
  // TODO: GlobalsInt8PtrTy
  // TODO: ConstGlobalsPtrTy
  // TODO: ASTAllocaAddressSpace

  PtrDiffTy = ::mlir::cir::IntType::get(
      builder.getContext(), astCtx.getTargetInfo().getMaxPointerWidth(),
      /*isSigned=*/true);

  mlir::cir::sob::SignedOverflowBehavior sob;
  switch (langOpts.getSignedOverflowBehavior()) {
  case clang::LangOptions::SignedOverflowBehaviorTy::SOB_Defined:
    sob = sob::SignedOverflowBehavior::defined;
    break;
  case clang::LangOptions::SignedOverflowBehaviorTy::SOB_Undefined:
    sob = sob::SignedOverflowBehavior::undefined;
    break;
  case clang::LangOptions::SignedOverflowBehaviorTy::SOB_Trapping:
    sob = sob::SignedOverflowBehavior::trapping;
    break;
  }
  theModule->setAttr("cir.sob",
                     mlir::cir::SignedOverflowBehaviorAttr::get(&context, sob));
  theModule->setAttr(
      "cir.lang", mlir::cir::LangAttr::get(&context, getCIRSourceLanguage()));
  // Set the module name to be the name of the main file. TranslationUnitDecl
  // often contains invalid source locations and isn't a reliable source for the
  // module location.
  auto MainFileID = astctx.getSourceManager().getMainFileID();
  const FileEntry &MainFile =
      *astctx.getSourceManager().getFileEntryForID(MainFileID);
  auto Path = MainFile.tryGetRealPathName();
  if (!Path.empty()) {
    theModule.setSymName(Path);
    theModule->setLoc(mlir::FileLineColLoc::get(&context, Path,
                                                /*line=*/0,
                                                /*col=*/0));
  }
}

CIRGenModule::~CIRGenModule() {}

bool CIRGenModule::isTypeConstant(QualType Ty, bool ExcludeCtor,
                                  bool ExcludeDtor) {
  if (!Ty.isConstant(astCtx) && !Ty->isReferenceType())
    return false;

  if (astCtx.getLangOpts().CPlusPlus) {
    if (const CXXRecordDecl *Record =
            astCtx.getBaseElementType(Ty)->getAsCXXRecordDecl())
      return ExcludeCtor && !Record->hasMutableFields() &&
             (Record->hasTrivialDestructor() || ExcludeDtor);
  }

  return true;
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

bool CIRGenModule::MustBeEmitted(const ValueDecl *Global) {
  // Never defer when EmitAllDecls is specified.
  assert(!langOpts.EmitAllDecls && "EmitAllDecls NYI");
  assert(!codeGenOpts.KeepStaticConsts && "KeepStaticConsts NYI");

  return getASTContext().DeclMustBeEmitted(Global);
}

bool CIRGenModule::MayBeEmittedEagerly(const ValueDecl *Global) {
  // In OpenMP 5.0 variables and function may be marked as
  // device_type(host/nohost) and we should not emit them eagerly unless we sure
  // that they must be emitted on the host/device. To be sure we need to have
  // seen a declare target with an explicit mentioning of the function, we know
  // we have if the level of the declare target attribute is -1. Note that we
  // check somewhere else if we should emit this at all.
  if (langOpts.OpenMP >= 50 && !langOpts.OpenMPSimd) {
    std::optional<OMPDeclareTargetDeclAttr *> ActiveAttr =
        OMPDeclareTargetDeclAttr::getActiveAttr(Global);
    if (!ActiveAttr || (*ActiveAttr)->getLevel() != (unsigned)-1)
      return false;
  }

  const auto *FD = dyn_cast<FunctionDecl>(Global);
  if (FD) {
    // Implicit template instantiations may change linkage if they are later
    // explicitly instantiated, so they should not be emitted eagerly.
    // TODO(cir): do we care?
    assert(FD->getTemplateSpecializationKind() != TSK_ImplicitInstantiation &&
           "not implemented");
    assert(!FD->isTemplated() && "Templates NYI");
  }
  const auto *VD = dyn_cast<VarDecl>(Global);
  if (VD)
    // A definition of an inline constexpr static data member may change
    // linkage later if it's redeclared outside the class.
    // TODO(cir): do we care?
    assert(astCtx.getInlineVariableDefinitionKind(VD) !=
               ASTContext::InlineVariableDefinitionKind::WeakUnknown &&
           "not implemented");

  // If OpenMP is enabled and threadprivates must be generated like TLS, delay
  // codegen for global variables, because they may be marked as threadprivate.
  if (langOpts.OpenMP && langOpts.OpenMPUseTLS &&
      getASTContext().getTargetInfo().isTLSSupported() &&
      isa<VarDecl>(Global) &&
      !Global->getType().isConstantStorage(getASTContext(), false, false) &&
      !OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(Global))
    return false;

  assert((FD || VD) &&
         "Only FunctionDecl and VarDecl should hit this path so far.");
  return true;
}

void CIRGenModule::buildGlobal(GlobalDecl GD) {
  const auto *Global = cast<ValueDecl>(GD.getDecl());

  assert(!Global->hasAttr<IFuncAttr>() && "NYI");
  assert(!Global->hasAttr<CPUDispatchAttr>() && "NYI");
  assert(!langOpts.CUDA && "NYI");

  if (langOpts.OpenMP) {
    // If this is OpenMP, check if it is legal to emit this global normally.
    if (openMPRuntime && openMPRuntime->emitTargetGlobal(GD)) {
      assert(!UnimplementedFeature::openMPRuntime());
      return;
    }
    if (auto *DRD = dyn_cast<OMPDeclareReductionDecl>(Global)) {
      assert(!UnimplementedFeature::openMP());
      return;
    }
    if (auto *DMD = dyn_cast<OMPDeclareMapperDecl>(Global)) {
      assert(!UnimplementedFeature::openMP());
      return;
    }
  }

  // Ignore declarations, they will be emitted on their first use.
  if (const auto *FD = dyn_cast<FunctionDecl>(Global)) {
    // Forward declarations are emitted lazily on first use.
    if (!FD->doesThisDeclarationHaveABody()) {
      if (!FD->doesDeclarationForceExternallyVisibleDefinition())
        return;

      llvm::StringRef MangledName = getMangledName(GD);

      // Compute the function info and CIR type.
      const auto &FI = getTypes().arrangeGlobalDeclaration(GD);
      mlir::Type Ty = getTypes().GetFunctionType(FI);

      GetOrCreateCIRFunction(MangledName, Ty, GD, /*ForVTable=*/false,
                             /*DontDefer=*/false);
      return;
    }
  } else {
    const auto *VD = cast<VarDecl>(Global);
    assert(VD->isFileVarDecl() && "Cannot emit local var decl as global.");
    if (VD->isThisDeclarationADefinition() != VarDecl::Definition &&
        !astCtx.isMSStaticDataMemberInlineDefinition(VD)) {
      if (langOpts.OpenMP) {
        // Emit declaration of the must-be-emitted declare target variable.
        if (std::optional<OMPDeclareTargetDeclAttr::MapTypeTy> Res =
                OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(VD)) {
          assert(0 && "OMPDeclareTargetDeclAttr NYI");
        }
      }
      // If this declaration may have caused an inline variable definition
      // to change linkage, make sure that it's emitted.
      // TODO(cir): probably use GetAddrOfGlobalVar(VD) below?
      assert((astCtx.getInlineVariableDefinitionKind(VD) !=
              ASTContext::InlineVariableDefinitionKind::Strong) &&
             "not implemented");
      return;
    }
  }

  // Defer code generation to first use when possible, e.g. if this is an inline
  // function. If the global mjust always be emitted, do it eagerly if possible
  // to benefit from cache locality.
  if (MustBeEmitted(Global) && MayBeEmittedEagerly(Global)) {
    // Emit the definition if it can't be deferred.
    buildGlobalDefinition(GD);
    return;
  }

  // If we're deferring emission of a C++ variable with an initializer, remember
  // the order in which it appeared on the file.
  if (getLangOpts().CPlusPlus && isa<VarDecl>(Global) &&
      cast<VarDecl>(Global)->hasInit()) {
    DelayedCXXInitPosition[Global] = CXXGlobalInits.size();
    CXXGlobalInits.push_back(nullptr);
  }

  llvm::StringRef MangledName = getMangledName(GD);
  if (getGlobalValue(MangledName) != nullptr) {
    // The value has already been used and should therefore be emitted.
    addDeferredDeclToEmit(GD);
  } else if (MustBeEmitted(Global)) {
    // The value must be emitted, but cannot be emitted eagerly.
    assert(!MayBeEmittedEagerly(Global));
    addDeferredDeclToEmit(GD);
  } else {
    // Otherwise, remember that we saw a deferred decl with this name. The first
    // use of the mangled name will cause it to move into DeferredDeclsToEmit.
    DeferredDecls[MangledName] = GD;
  }
}

void CIRGenModule::buildGlobalFunctionDefinition(GlobalDecl GD,
                                                 mlir::Operation *Op) {
  auto const *D = cast<FunctionDecl>(GD.getDecl());

  // Compute the function info and CIR type.
  const CIRGenFunctionInfo &FI = getTypes().arrangeGlobalDeclaration(GD);
  auto Ty = getTypes().GetFunctionType(FI);

  // Get or create the prototype for the function.
  // if (!V || (V.getValueType() != Ty))
  // TODO(cir): Figure out what to do here? llvm uses a GlobalValue for the
  // FuncOp in mlir
  Op = GetAddrOfFunction(GD, Ty, /*ForVTable=*/false, /*DontDefer=*/true,
                         ForDefinition);

  auto Fn = cast<mlir::cir::FuncOp>(Op);
  // Already emitted.
  if (!Fn.isDeclaration())
    return;

  setFunctionLinkage(GD, Fn);
  setGVProperties(Op, D);
  // TODO(cir): MaubeHandleStaticInExternC
  // TODO(cir): maybeSetTrivialComdat
  // TODO(cir): setLLVMFunctionFEnvAttributes

  CIRGenFunction CGF{*this, builder};
  CurCGF = &CGF;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    CGF.generateCode(GD, Fn, FI);
  }
  CurCGF = nullptr;

  // TODO: setNonAliasAttributes
  // TODO: SetLLVMFunctionAttributesForDeclaration

  assert(!D->getAttr<ConstructorAttr>() && "NYI");
  assert(!D->getAttr<DestructorAttr>() && "NYI");
  assert(!D->getAttr<AnnotateAttr>() && "NYI");
}

mlir::Operation *CIRGenModule::getGlobalValue(StringRef Name) {
  auto global = mlir::SymbolTable::lookupSymbolIn(theModule, Name);
  if (!global)
    return {};
  return global;
}

mlir::Value CIRGenModule::getGlobalValue(const Decl *D) {
  assert(CurCGF);
  return CurCGF->symbolTable.lookup(D);
}

mlir::cir::GlobalOp CIRGenModule::createGlobalOp(CIRGenModule &CGM,
                                                 mlir::Location loc,
                                                 StringRef name, mlir::Type t,
                                                 bool isCst,
                                                 mlir::Operation *insertPoint) {
  mlir::cir::GlobalOp g;
  auto &builder = CGM.getBuilder();
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Some global emissions are triggered while emitting a function, e.g.
    // void s() { const char *s = "yolo"; ... }
    //
    // Be sure to insert global before the current function
    auto *curCGF = CGM.getCurrCIRGenFun();
    if (curCGF)
      builder.setInsertionPoint(curCGF->CurFn);

    g = builder.create<mlir::cir::GlobalOp>(loc, name, t, isCst);
    if (!curCGF) {
      if (insertPoint)
        CGM.getModule().insert(insertPoint, g);
      else
        CGM.getModule().push_back(g);
    }

    // Default to private until we can judge based on the initializer,
    // since MLIR doesn't allow public declarations.
    mlir::SymbolTable::setSymbolVisibility(
        g, mlir::SymbolTable::Visibility::Private);
  }
  return g;
}

void CIRGenModule::setCommonAttributes(GlobalDecl GD, mlir::Operation *GV) {
  assert(!UnimplementedFeature::setCommonAttributes());
}

void CIRGenModule::replaceGlobal(mlir::cir::GlobalOp Old,
                                 mlir::cir::GlobalOp New) {
  assert(Old.getSymName() == New.getSymName() && "symbol names must match");

  // If the types does not match, update all references to Old to the new type.
  auto OldTy = Old.getSymType();
  auto NewTy = New.getSymType();
  if (OldTy != NewTy) {
    auto OldSymUses = Old.getSymbolUses(theModule.getOperation());
    if (OldSymUses.has_value()) {
      for (auto Use : *OldSymUses) {
        auto *UserOp = Use.getUser();
        assert((isa<mlir::cir::GetGlobalOp>(UserOp) ||
                isa<mlir::cir::GlobalOp>(UserOp)) &&
               "GlobalOp symbol user is neither a GetGlobalOp nor a GlobalOp");

        if (auto GGO = dyn_cast<mlir::cir::GetGlobalOp>(Use.getUser())) {
          auto UseOpResultValue = GGO.getAddr();
          UseOpResultValue.setType(
              mlir::cir::PointerType::get(builder.getContext(), NewTy));
        }
      }
    }
  }

  // Remove old global from the module.
  Old.erase();
}

/// If the specified mangled name is not in the module,
/// create and return an mlir GlobalOp with the specified type (TODO(cir):
/// address space).
///
/// TODO(cir):
/// 1. If there is something in the module with the specified name, return
/// it potentially bitcasted to the right type.
///
/// 2. If D is non-null, it specifies a decl that correspond to this.  This is
/// used to set the attributes on the global when it is first created.
///
/// 3. If IsForDefinition is true, it is guaranteed that an actual global with
/// type Ty will be returned, not conversion of a variable with the same
/// mangled name but some other type.
mlir::cir::GlobalOp
CIRGenModule::getOrCreateCIRGlobal(StringRef MangledName, mlir::Type Ty,
                                   LangAS AddrSpace, const VarDecl *D,
                                   ForDefinition_t IsForDefinition) {
  // Lookup the entry, lazily creating it if necessary.
  mlir::cir::GlobalOp Entry;
  if (auto *V = getGlobalValue(MangledName)) {
    assert(isa<mlir::cir::GlobalOp>(V) && "only supports GlobalOp for now");
    Entry = dyn_cast_or_null<mlir::cir::GlobalOp>(V);
  }

  // unsigned TargetAS = astCtx.getTargetAddressSpace(AddrSpace);
  if (Entry) {
    if (WeakRefReferences.erase(Entry)) {
      if (D && !D->hasAttr<WeakAttr>()) {
        auto LT = mlir::cir::GlobalLinkageKind::ExternalLinkage;
        Entry.setLinkageAttr(
            mlir::cir::GlobalLinkageKindAttr::get(builder.getContext(), LT));
        mlir::SymbolTable::setSymbolVisibility(Entry, getMLIRVisibility(Entry));
      }
    }

    // Handle dropped DLL attributes.
    if (D && !D->hasAttr<clang::DLLImportAttr>() &&
        !D->hasAttr<clang::DLLExportAttr>())
      assert(!UnimplementedFeature::setDLLStorageClass() && "NYI");

    if (langOpts.OpenMP && !langOpts.OpenMPSimd && D)
      getOpenMPRuntime().registerTargetGlobalVariable(D, Entry);

    // TODO(cir): check TargetAS matches Entry address space
    if (Entry.getSymType() == Ty &&
        !UnimplementedFeature::addressSpaceInGlobalVar())
      return Entry;

    // If there are two attempts to define the same mangled name, issue an
    // error.
    //
    // TODO(cir): look at mlir::GlobalValue::isDeclaration for all aspects of
    // recognizing the global as a declaration, for now only check if
    // initializer is present.
    if (IsForDefinition && !Entry.isDeclaration()) {
      GlobalDecl OtherGD;
      const VarDecl *OtherD;

      // Check that D is not yet in DiagnosedConflictingDefinitions is required
      // to make sure that we issue an error only once.
      if (D && lookupRepresentativeDecl(MangledName, OtherGD) &&
          (D->getCanonicalDecl() != OtherGD.getCanonicalDecl().getDecl()) &&
          (OtherD = dyn_cast<VarDecl>(OtherGD.getDecl())) &&
          OtherD->hasInit() &&
          DiagnosedConflictingDefinitions.insert(D).second) {
        getDiags().Report(D->getLocation(), diag::err_duplicate_mangled_name)
            << MangledName;
        getDiags().Report(OtherGD.getDecl()->getLocation(),
                          diag::note_previous_definition);
      }
    }

    // TODO(cir): LLVM codegen makes sure the result is of the correct type
    // by issuing a address space cast.

    // TODO(cir):
    // (In LLVM codgen, if global is requested for a definition, we always need
    // to create a new global, otherwise return a bitcast.)
    if (!IsForDefinition)
      assert(0 && "not implemented");
  }

  // TODO(cir): auto DAddrSpace = GetGlobalVarAddressSpace(D);
  // TODO(cir): do we need to strip pointer casts for Entry?

  auto loc = getLoc(D->getSourceRange());

  // mlir::SymbolTable::Visibility::Public is the default, no need to explicitly
  // mark it as such.
  auto GV = CIRGenModule::createGlobalOp(*this, loc, MangledName, Ty,
                                         /*isConstant=*/false,
                                         /*insertPoint=*/Entry.getOperation());

  // If we already created a global with the same mangled name (but different
  // type) before, replace it with the new global.
  if (Entry) {
    replaceGlobal(Entry, GV);
  }

  // This is the first use or definition of a mangled name.  If there is a
  // deferred decl with this name, remember that we need to emit it at the end
  // of the file.
  auto DDI = DeferredDecls.find(MangledName);
  if (DDI != DeferredDecls.end()) {
    // Move the potentially referenced deferred decl to the DeferredDeclsToEmit
    // list, and remove it from DeferredDecls (since we don't need it anymore).
    addDeferredDeclToEmit(DDI->second);
    DeferredDecls.erase(DDI);
  }

  // Handle things which are present even on external declarations.
  if (D) {
    if (langOpts.OpenMP && !langOpts.OpenMPSimd && D)
      getOpenMPRuntime().registerTargetGlobalVariable(D, Entry);

    // FIXME: This code is overly simple and should be merged with other global
    // handling.

    // TODO(cir):
    //   GV->setConstant(isTypeConstant(D->getType(), false));
    //   GV->setAlignment(getContext().getDeclAlign(D).getAsAlign());
    //   setLinkageForGV(GV, D);

    if (D->getTLSKind()) {
      assert(0 && "not implemented");
    }

    setGVProperties(GV, D);

    // If required by the ABI, treat declarations of static data members with
    // inline initializers as definitions.
    if (astCtx.isMSStaticDataMemberInlineDefinition(D)) {
      assert(0 && "not implemented");
    }

    // Emit section information for extern variables.
    if (D->hasExternalStorage()) {
      if (const SectionAttr *SA = D->getAttr<SectionAttr>()) {
        assert(!UnimplementedFeature::setGlobalVarSection());
        llvm_unreachable("section info for extern vars is NYI");
      }
    }

    // Handle XCore specific ABI requirements.
    if (getTriple().getArch() == llvm::Triple::xcore)
      assert(0 && "not implemented");

    // Check if we a have a const declaration with an initializer, we maybe
    // able to emit it as available_externally to expose it's value to the
    // optimizer.
    if (getLangOpts().CPlusPlus && GV.isPublic() &&
        D->getType().isConstQualified() && GV.isDeclaration() &&
        !D->hasDefinition() && D->hasInit() && !D->hasAttr<DLLImportAttr>()) {
      assert(0 && "not implemented");
    }
  }

  // TODO(cir): if this method is used to handle functions we must have
  // something closer to GlobalValue::isDeclaration instead of checking for
  // initializer.
  if (GV.isDeclaration()) {
    // TODO(cir): set target attributes

    // External HIP managed variables needed to be recorded for transformation
    // in both device and host compilations.
    if (getLangOpts().CUDA)
      assert(0 && "not implemented");
  }

  // TODO(cir): address space cast when needed for DAddrSpace.
  return GV;
}

mlir::cir::GlobalOp CIRGenModule::buildGlobal(const VarDecl *D, mlir::Type Ty,
                                              ForDefinition_t IsForDefinition) {
  assert(D->hasGlobalStorage() && "Not a global variable");
  QualType ASTTy = D->getType();
  if (!Ty)
    Ty = getTypes().convertTypeForMem(ASTTy);

  StringRef MangledName = getMangledName(D);
  return getOrCreateCIRGlobal(MangledName, Ty, ASTTy.getAddressSpace(), D,
                              IsForDefinition);
}

/// Return the mlir::Value for the address of the given global variable. If Ty
/// is non-null and if the global doesn't exist, then it will be created with
/// the specified type instead of whatever the normal requested type would be.
/// If IsForDefinition is true, it is guaranteed that an actual global with type
/// Ty will be returned, not conversion of a variable with the same mangled name
/// but some other type.
mlir::Value CIRGenModule::getAddrOfGlobalVar(const VarDecl *D, mlir::Type Ty,
                                             ForDefinition_t IsForDefinition) {
  assert(D->hasGlobalStorage() && "Not a global variable");
  QualType ASTTy = D->getType();
  if (!Ty)
    Ty = getTypes().convertTypeForMem(ASTTy);

  auto g = buildGlobal(D, Ty, IsForDefinition);
  auto ptrTy =
      mlir::cir::PointerType::get(builder.getContext(), g.getSymType());
  return builder.create<mlir::cir::GetGlobalOp>(getLoc(D->getSourceRange()),
                                                ptrTy, g.getSymName());
}

mlir::cir::GlobalViewAttr
CIRGenModule::getAddrOfGlobalVarAttr(const VarDecl *D, mlir::Type Ty,
                                     ForDefinition_t IsForDefinition) {
  assert(D->hasGlobalStorage() && "Not a global variable");
  QualType ASTTy = D->getType();
  if (!Ty)
    Ty = getTypes().convertTypeForMem(ASTTy);

  auto globalOp = buildGlobal(D, Ty, IsForDefinition);
  return builder.getGlobalViewAttr(builder.getPointerTo(Ty), globalOp);
}

mlir::Operation *CIRGenModule::getWeakRefReference(const ValueDecl *VD) {
  const AliasAttr *AA = VD->getAttr<AliasAttr>();
  assert(AA && "No alias?");

  // See if there is already something with the target's name in the module.
  mlir::Operation *Entry = getGlobalValue(AA->getAliasee());
  if (Entry) {
    assert((isa<mlir::cir::GlobalOp>(Entry) || isa<mlir::cir::FuncOp>(Entry)) &&
           "weak ref should be against a global variable or function");
    return Entry;
  }

  mlir::Type DeclTy = getTypes().convertTypeForMem(VD->getType());
  if (DeclTy.isa<mlir::cir::FuncType>()) {
    auto F = GetOrCreateCIRFunction(AA->getAliasee(), DeclTy,
                                    GlobalDecl(cast<FunctionDecl>(VD)),
                                    /*ForVtable=*/false);
    F.setLinkage(mlir::cir::GlobalLinkageKind::ExternalWeakLinkage);
    WeakRefReferences.insert(F);
    return F;
  }

  llvm_unreachable("GlobalOp NYI");
}

/// TODO(cir): looks like part of this code can be part of a common AST
/// helper betweem CIR and LLVM codegen.
template <typename SomeDecl>
void CIRGenModule::maybeHandleStaticInExternC(const SomeDecl *D,
                                              mlir::cir::GlobalOp GV) {
  if (!getLangOpts().CPlusPlus)
    return;

  // Must have 'used' attribute, or else inline assembly can't rely on
  // the name existing.
  if (!D->template hasAttr<UsedAttr>())
    return;

  // Must have internal linkage and an ordinary name.
  if (!D->getIdentifier() || D->getFormalLinkage() != Linkage::Internal)
    return;

  // Must be in an extern "C" context. Entities declared directly within
  // a record are not extern "C" even if the record is in such a context.
  const SomeDecl *First = D->getFirstDecl();
  if (First->getDeclContext()->isRecord() || !First->isInExternCContext())
    return;

  // TODO(cir):
  // OK, this is an internal linkage entity inside an extern "C" linkage
  // specification. Make a note of that so we can give it the "expected"
  // mangled name if nothing else is using that name.
  //
  // If we have multiple internal linkage entities with the same name
  // in extern "C" regions, none of them gets that name.
  assert(0 && "not implemented");
}

void CIRGenModule::buildGlobalVarDefinition(const clang::VarDecl *D,
                                            bool IsTentative) {
  // TODO(cir):
  // OpenCL global variables of sampler type are translated to function calls,
  // therefore no need to be translated.
  // If this is OpenMP device, check if it is legal to emit this global
  // normally.
  QualType ASTTy = D->getType();
  if (getLangOpts().OpenCL || getLangOpts().OpenMPIsTargetDevice)
    llvm_unreachable("not implemented");

  // TODO(cir): LLVM's codegen uses a llvm::TrackingVH here. Is that
  // necessary here for CIR gen?
  mlir::Attribute Init;
  bool NeedsGlobalCtor = false;
  // Whether the definition of the variable is available externally.
  // If yes, we shouldn't emit the GloablCtor and GlobalDtor for the variable
  // since this is the job for its original source.
  bool IsDefinitionAvailableExternally =
      astCtx.GetGVALinkageForVariable(D) == GVA_AvailableExternally;
  bool NeedsGlobalDtor =
      !IsDefinitionAvailableExternally &&
      D->needsDestruction(astCtx) == QualType::DK_cxx_destructor;

  const VarDecl *InitDecl;
  const Expr *InitExpr = D->getAnyInitializer(InitDecl);

  std::optional<ConstantEmitter> emitter;

  // CUDA E.2.4.1 "__shared__ variables cannot have an initialization
  // as part of their declaration."  Sema has already checked for
  // error cases, so we just need to set Init to UndefValue.
  bool IsCUDASharedVar =
      getLangOpts().CUDAIsDevice && D->hasAttr<CUDASharedAttr>();
  // Shadows of initialized device-side global variables are also left
  // undefined.
  // Managed Variables should be initialized on both host side and device side.
  bool IsCUDAShadowVar =
      !getLangOpts().CUDAIsDevice && !D->hasAttr<HIPManagedAttr>() &&
      (D->hasAttr<CUDAConstantAttr>() || D->hasAttr<CUDADeviceAttr>() ||
       D->hasAttr<CUDASharedAttr>());
  bool IsCUDADeviceShadowVar =
      getLangOpts().CUDAIsDevice && !D->hasAttr<HIPManagedAttr>() &&
      (D->getType()->isCUDADeviceBuiltinSurfaceType() ||
       D->getType()->isCUDADeviceBuiltinTextureType());
  if (getLangOpts().CUDA &&
      (IsCUDASharedVar || IsCUDAShadowVar || IsCUDADeviceShadowVar))
    assert(0 && "not implemented");
  else if (D->hasAttr<LoaderUninitializedAttr>())
    assert(0 && "not implemented");
  else if (!InitExpr) {
    // This is a tentative definition; tentative definitions are
    // implicitly initialized with { 0 }.
    //
    // Note that tentative definitions are only emitted at the end of
    // a translation unit, so they should never have incomplete
    // type. In addition, EmitTentativeDefinition makes sure that we
    // never attempt to emit a tentative definition if a real one
    // exists. A use may still exists, however, so we still may need
    // to do a RAUW.
    assert(!ASTTy->isIncompleteType() && "Unexpected incomplete type");
    Init = builder.getZeroInitAttr(getCIRType(D->getType()));
  } else {
    initializedGlobalDecl = GlobalDecl(D);
    emitter.emplace(*this);
    auto Initializer = emitter->tryEmitForInitializer(*InitDecl);
    if (!Initializer) {
      QualType T = InitExpr->getType();
      if (D->getType()->isReferenceType())
        T = D->getType();

      if (getLangOpts().CPlusPlus) {
        if (InitDecl->hasFlexibleArrayInit(astCtx))
          ErrorUnsupported(D, "flexible array initializer");
        Init = builder.getZeroInitAttr(getCIRType(T));
        if (!IsDefinitionAvailableExternally)
          NeedsGlobalCtor = true;
      } else {
        ErrorUnsupported(D, "static initializer");
      }
    } else {
      Init = Initializer;
      // We don't need an initializer, so remove the entry for the delayed
      // initializer position (just in case this entry was delayed) if we
      // also don't need to register a destructor.
      if (getLangOpts().CPlusPlus && !NeedsGlobalDtor)
        DelayedCXXInitPosition.erase(D);
    }
  }

  mlir::Type InitType;
  // If the initializer attribute is a SymbolRefAttr it means we are
  // initializing the global based on a global constant.
  //
  // TODO(cir): create another attribute to contain the final type and abstract
  // away SymbolRefAttr.
  if (auto symAttr = Init.dyn_cast<mlir::SymbolRefAttr>()) {
    auto cstGlobal = mlir::SymbolTable::lookupSymbolIn(theModule, symAttr);
    assert(isa<mlir::cir::GlobalOp>(cstGlobal) &&
           "unaware of other symbol providers");
    auto g = cast<mlir::cir::GlobalOp>(cstGlobal);
    auto arrayTy = g.getSymType().dyn_cast<mlir::cir::ArrayType>();
    // TODO(cir): pointer to array decay. Should this be modeled explicitly in
    // CIR?
    if (arrayTy)
      InitType = mlir::cir::PointerType::get(builder.getContext(),
                                             arrayTy.getEltType());
  } else {
    assert(Init.isa<mlir::TypedAttr>() && "This should have a type");
    auto TypedInitAttr = Init.cast<mlir::TypedAttr>();
    InitType = TypedInitAttr.getType();
  }
  assert(!InitType.isa<mlir::NoneType>() && "Should have a type by now");

  auto Entry = buildGlobal(D, InitType, ForDefinition_t(!IsTentative));
  // TODO(cir): Strip off pointer casts from Entry if we get them?

  // TODO(cir): LLVM codegen used GlobalValue to handle both Function or
  // GlobalVariable here. We currently only support GlobalOp, should this be
  // used for FuncOp?
  assert(dyn_cast<GlobalOp>(&Entry) && "FuncOp not supported here");
  auto GV = Entry;

  // We have a definition after a declaration with the wrong type.
  // We must make a new GlobalVariable* and update everything that used OldGV
  // (a declaration or tentative definition) with the new GlobalVariable*
  // (which will be a definition).
  //
  // This happens if there is a prototype for a global (e.g.
  // "extern int x[];") and then a definition of a different type (e.g.
  // "int x[10];"). This also happens when an initializer has a different type
  // from the type of the global (this happens with unions).
  if (!GV || GV.getSymType() != InitType) {
    // TODO(cir): this should include an address space check as well.
    assert(0 && "not implemented");
  }

  maybeHandleStaticInExternC(D, GV);

  if (D->hasAttr<AnnotateAttr>())
    assert(0 && "not implemented");

  // Set CIR's linkage type as appropriate.
  mlir::cir::GlobalLinkageKind Linkage =
      getCIRLinkageVarDefinition(D, /*IsConstant=*/false);

  // TODO(cir):
  // CUDA B.2.1 "The __device__ qualifier declares a variable that resides on
  // the device. [...]"
  // CUDA B.2.2 "The __constant__ qualifier, optionally used together with
  // __device__, declares a variable that: [...]
  if (GV && getLangOpts().CUDA) {
    assert(0 && "not implemented");
  }

  // Set initializer and finalize emission
  CIRGenModule::setInitializer(GV, Init);
  if (emitter)
    emitter->finalize(GV);

  // TODO(cir): If it is safe to mark the global 'constant', do so now.
  // GV->setConstant(!NeedsGlobalCtor && !NeedsGlobalDtor &&
  //                 isTypeConstant(D->getType(), true));

  // If it is in a read-only section, mark it 'constant'.
  if (const SectionAttr *SA = D->getAttr<SectionAttr>()) {
    assert(0 && "not implemented");
  }

  // TODO(cir):
  // GV->setAlignment(getContext().getDeclAlign(D).getAsAlign());

  // On Darwin, unlike other Itanium C++ ABI platforms, the thread-wrapper
  // function is only defined alongside the variable, not also alongside
  // callers. Normally, all accesses to a thread_local go through the
  // thread-wrapper in order to ensure initialization has occurred, underlying
  // variable will never be used other than the thread-wrapper, so it can be
  // converted to internal linkage.
  //
  // However, if the variable has the 'constinit' attribute, it _can_ be
  // referenced directly, without calling the thread-wrapper, so the linkage
  // must not be changed.
  //
  // Additionally, if the variable isn't plain external linkage, e.g. if it's
  // weak or linkonce, the de-duplication semantics are important to preserve,
  // so we don't change the linkage.
  if (D->getTLSKind() == VarDecl::TLS_Dynamic && GV.isPublic() &&
      astCtx.getTargetInfo().getTriple().isOSDarwin() &&
      !D->hasAttr<ConstInitAttr>()) {
    // TODO(cir): set to mlir::SymbolTable::Visibility::Private once we have
    // testcases.
    assert(0 && "not implemented");
  }

  // Set CIR linkage and DLL storage class.
  GV.setLinkage(Linkage);
  // FIXME(cir): setLinkage should likely set MLIR's visibility automatically.
  GV.setVisibility(getMLIRVisibilityFromCIRLinkage(Linkage));
  // TODO(cir): handle DLL storage classes in CIR?
  if (D->hasAttr<DLLImportAttr>())
    assert(!UnimplementedFeature::setDLLStorageClass());
  else if (D->hasAttr<DLLExportAttr>())
    assert(!UnimplementedFeature::setDLLStorageClass());
  else
    assert(!UnimplementedFeature::setDLLStorageClass());

  if (Linkage == mlir::cir::GlobalLinkageKind::CommonLinkage) {
    // common vars aren't constant even if declared const.
    GV.setConstant(false);
    // Tentative definition of global variables may be initialized with
    // non-zero null pointers. In this case they should have weak linkage
    // since common linkage must have zero initializer and must not have
    // explicit section therefore cannot have non-zero initial value.
    auto Initializer = GV.getInitialValue();
    if (Initializer && !getBuilder().isNullValue(*Initializer))
      GV.setLinkage(mlir::cir::GlobalLinkageKind::WeakAnyLinkage);
  }

  // TODO(cir): setNonAliasAttributes(D, GV);

  // TODO(cir): handle TLSKind if GV is not thread local
  if (D->getTLSKind()) { // && !GV->isThreadLocal())
    assert(0 && "not implemented");
  }

  // TODO(cir): maybeSetTrivialComdat(*D, *GV);

  // TODO(cir):
  // Emit the initializer function if necessary.
  if (NeedsGlobalCtor || NeedsGlobalDtor)
    buildGlobalVarDeclInit(D, GV, NeedsGlobalCtor);

  // TODO(cir): sanitizers (reportGlobalToASan) and global variable debug
  // information.
}

void CIRGenModule::buildGlobalDefinition(GlobalDecl GD, mlir::Operation *Op) {
  const auto *D = cast<ValueDecl>(GD.getDecl());

  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    // At -O0, don't generate CIR for functions with available_externally
    // linkage.
    if (!shouldEmitFunction(GD))
      return;

    if (const auto *Method = dyn_cast<CXXMethodDecl>(D)) {
      // Make sure to emit the definition(s) before we emit the thunks. This is
      // necessary for the generation of certain thunks.
      if (isa<CXXConstructorDecl>(Method) || isa<CXXDestructorDecl>(Method))
        ABI->buildCXXStructor(GD);
      else if (FD->isMultiVersion())
        llvm_unreachable("NYI");
      else
        buildGlobalFunctionDefinition(GD, Op);

      if (Method->isVirtual())
        getVTables().buildThunks(GD);

      return;
    }

    if (FD->isMultiVersion())
      llvm_unreachable("NYI");
    buildGlobalFunctionDefinition(GD, Op);
    return;
  }

  if (const auto *VD = dyn_cast<VarDecl>(D))
    return buildGlobalVarDefinition(VD, !VD->hasDefinition());

  llvm_unreachable("Invalid argument to buildGlobalDefinition()");
}

mlir::Attribute
CIRGenModule::getConstantArrayFromStringLiteral(const StringLiteral *E) {
  assert(!E->getType()->isPointerType() && "Strings are always arrays");

  // Don't emit it as the address of the string, emit the string data itself
  // as an inline array.
  if (E->getCharByteWidth() == 1) {
    SmallString<64> Str(E->getString());

    // Resize the string to the right size, which is indicated by its type.
    const ConstantArrayType *CAT = astCtx.getAsConstantArrayType(E->getType());
    auto finalSize = CAT->getSize().getZExtValue();
    Str.resize(finalSize);

    auto eltTy = getTypes().ConvertType(CAT->getElementType());
    return builder.getString(Str, eltTy, finalSize);
  }

  auto arrayTy =
      getTypes().ConvertType(E->getType()).dyn_cast<mlir::cir::ArrayType>();
  assert(arrayTy && "string literals must be emitted as an array type");

  auto arrayEltTy = arrayTy.getEltType().dyn_cast<mlir::cir::IntType>();
  assert(arrayEltTy &&
         "string literal elements must be emitted as integral type");

  auto arraySize = arrayTy.getSize();
  auto literalSize = E->getLength();

  // Collect the code units.
  SmallVector<uint32_t, 32> elementValues;
  elementValues.reserve(arraySize);
  for (unsigned i = 0; i < literalSize; ++i)
    elementValues.push_back(E->getCodeUnit(i));
  elementValues.resize(arraySize);

  // If the string is full of null bytes, emit a #cir.zero instead.
  if (std::all_of(elementValues.begin(), elementValues.end(),
                  [](uint32_t x) { return x == 0; }))
    return builder.getZeroAttr(arrayTy);

  // Otherwise emit a constant array holding the characters.
  SmallVector<mlir::Attribute, 32> elements;
  elements.reserve(arraySize);
  for (uint64_t i = 0; i < arraySize; ++i)
    elements.push_back(mlir::cir::IntAttr::get(arrayEltTy, elementValues[i]));

  auto elementsAttr = mlir::ArrayAttr::get(builder.getContext(), elements);
  return builder.getConstArray(elementsAttr, arrayTy);
}

// TODO(cir): this could be a common AST helper for both CIR and LLVM codegen.
LangAS CIRGenModule::getGlobalConstantAddressSpace() const {
  // OpenCL v1.2 s6.5.3: a string literal is in the constant address space.
  if (getLangOpts().OpenCL)
    return LangAS::opencl_constant;
  if (getLangOpts().SYCLIsDevice)
    return LangAS::sycl_global;
  if (auto AS = getTarget().getConstantAddressSpace())
    return AS.value();
  return LangAS::Default;
}

static mlir::cir::GlobalOp
generateStringLiteral(mlir::Location loc, mlir::TypedAttr C,
                      mlir::cir::GlobalLinkageKind LT, CIRGenModule &CGM,
                      StringRef GlobalName, CharUnits Alignment) {
  unsigned AddrSpace = CGM.getASTContext().getTargetAddressSpace(
      CGM.getGlobalConstantAddressSpace());
  assert((AddrSpace == 0 &&
          !cir::UnimplementedFeature::addressSpaceInGlobalVar()) &&
         "NYI");

  // Create a global variable for this string
  // FIXME(cir): check for insertion point in module level.
  auto GV = CIRGenModule::createGlobalOp(CGM, loc, GlobalName, C.getType(),
                                         !CGM.getLangOpts().WritableStrings);

  // Set up extra information and add to the module
  GV.setAlignmentAttr(CGM.getSize(Alignment));
  GV.setLinkageAttr(
      mlir::cir::GlobalLinkageKindAttr::get(CGM.getBuilder().getContext(), LT));
  CIRGenModule::setInitializer(GV, C);

  // TODO(cir)
  assert(!cir::UnimplementedFeature::threadLocal() && "NYI");
  assert(!cir::UnimplementedFeature::unnamedAddr() && "NYI");
  assert(!mlir::cir::isWeakForLinker(LT) && "NYI");
  assert(!cir::UnimplementedFeature::setDSOLocal() && "NYI");
  return GV;
}

/// Return a pointer to a constant array for the given string literal.
mlir::cir::GlobalViewAttr
CIRGenModule::getAddrOfConstantStringFromLiteral(const StringLiteral *S,
                                                 StringRef Name) {
  CharUnits Alignment =
      astCtx.getAlignOfGlobalVarInChars(S->getType(), /*VD=*/nullptr);

  mlir::Attribute C = getConstantArrayFromStringLiteral(S);

  mlir::cir::GlobalOp GV;
  if (!getLangOpts().WritableStrings && ConstantStringMap.count(C)) {
    GV = ConstantStringMap[C];
    // The bigger alignment always wins.
    if (!GV.getAlignment() ||
        uint64_t(Alignment.getQuantity()) > *GV.getAlignment())
      GV.setAlignmentAttr(getSize(Alignment));
  } else {
    SmallString<256> StringNameBuffer = Name;
    llvm::raw_svector_ostream Out(StringNameBuffer);
    if (StringLiteralCnt)
      Out << StringLiteralCnt;
    Name = Out.str();
    StringLiteralCnt++;

    SmallString<256> MangledNameBuffer;
    StringRef GlobalVariableName;
    auto LT = mlir::cir::GlobalLinkageKind::ExternalLinkage;

    // Mangle the string literal if that's how the ABI merges duplicate strings.
    // Don't do it if they are writable, since we don't want writes in one TU to
    // affect strings in another.
    if (getCXXABI().getMangleContext().shouldMangleStringLiteral(S) &&
        !getLangOpts().WritableStrings) {
      assert(0 && "not implemented");
    } else {
      LT = mlir::cir::GlobalLinkageKind::InternalLinkage;
      GlobalVariableName = Name;
    }

    auto loc = getLoc(S->getSourceRange());
    auto typedC = llvm::dyn_cast<mlir::TypedAttr>(C);
    if (!typedC)
      llvm_unreachable("this should never be untyped at this point");
    GV = generateStringLiteral(loc, typedC, LT, *this, GlobalVariableName,
                               Alignment);
    ConstantStringMap[C] = GV;

    assert(!cir::UnimplementedFeature::reportGlobalToASan() && "NYI");
  }

  auto ArrayTy = GV.getSymType().dyn_cast<mlir::cir::ArrayType>();
  assert(ArrayTy && "String literal must be array");
  auto PtrTy =
      mlir::cir::PointerType::get(builder.getContext(), ArrayTy.getEltType());

  return builder.getGlobalViewAttr(PtrTy, GV);
}

void CIRGenModule::buildDeclContext(const DeclContext *DC) {
  for (auto *I : DC->decls()) {
    // Unlike other DeclContexts, the contents of an ObjCImplDecl at TU scope
    // are themselves considered "top-level", so EmitTopLevelDecl on an
    // ObjCImplDecl does not recursively visit them. We need to do that in
    // case they're nested inside another construct (LinkageSpecDecl /
    // ExportDecl) that does stop them from being considered "top-level".
    if (auto *OID = dyn_cast<ObjCImplDecl>(I))
      llvm_unreachable("NYI");

    buildTopLevelDecl(I);
  }
}

void CIRGenModule::buildLinkageSpec(const LinkageSpecDecl *LSD) {
  if (LSD->getLanguage() != LinkageSpecLanguageIDs::C &&
      LSD->getLanguage() != LinkageSpecLanguageIDs::CXX) {
    llvm_unreachable("unsupported linkage spec");
    return;
  }
  buildDeclContext(LSD);
}

// Emit code for a single top level declaration.
void CIRGenModule::buildTopLevelDecl(Decl *decl) {
  // Ignore dependent declarations
  if (decl->isTemplated())
    return;

  // Consteval function shouldn't be emitted.
  if (auto *FD = dyn_cast<FunctionDecl>(decl))
    if (FD->isConsteval())
      return;

  switch (decl->getKind()) {
  default:
    llvm::errs() << "buildTopLevelDecl codegen for decl kind '"
                 << decl->getDeclKindName() << "' not implemented\n";
    assert(false && "Not yet implemented");

  case Decl::TranslationUnit: {
    // This path is CIR only - CIRGen handles TUDecls because
    // of clang-tidy checks, that operate on TU granularity.
    TranslationUnitDecl *TU = cast<TranslationUnitDecl>(decl);
    for (DeclContext::decl_iterator D = TU->decls_begin(),
                                    DEnd = TU->decls_end();
         D != DEnd; ++D)
      buildTopLevelDecl(*D);
    return;
  }
  case Decl::Var:
  case Decl::Decomposition:
  case Decl::VarTemplateSpecialization:
    buildGlobal(cast<VarDecl>(decl));
    assert(!isa<DecompositionDecl>(decl) && "not implemented");
    // if (auto *DD = dyn_cast<DecompositionDecl>(decl))
    //   for (auto *B : DD->bindings())
    //     if (auto *HD = B->getHoldingVar())
    //       EmitGlobal(HD);
    break;

  case Decl::CXXConversion:
  case Decl::CXXMethod:
  case Decl::Function:
    buildGlobal(cast<FunctionDecl>(decl));
    assert(!codeGenOpts.CoverageMapping && "Coverage Mapping NYI");
    break;
  // C++ Decls
  case Decl::Namespace:
    buildDeclContext(cast<NamespaceDecl>(decl));
    break;
  case Decl::ClassTemplateSpecialization: {
    // const auto *Spec = cast<ClassTemplateSpecializationDecl>(decl);
    assert(!UnimplementedFeature::generateDebugInfo() && "NYI");
  }
    [[fallthrough]];
  case Decl::CXXRecord: {
    CXXRecordDecl *crd = cast<CXXRecordDecl>(decl);
    // TODO: Handle debug info as CodeGenModule.cpp does
    for (auto *childDecl : crd->decls())
      if (isa<VarDecl>(childDecl) || isa<CXXRecordDecl>(childDecl))
        buildTopLevelDecl(childDecl);
    break;
  }
  // No code generation needed.
  case Decl::UsingShadow:
  case Decl::ClassTemplate:
  case Decl::VarTemplate:
  case Decl::Concept:
  case Decl::VarTemplatePartialSpecialization:
  case Decl::FunctionTemplate:
  case Decl::TypeAliasTemplate:
  case Decl::Block:
  case Decl::Empty:
  case Decl::Binding:
    break;
  case Decl::Using:     // using X; [C++]
  case Decl::UsingEnum: // using enum X; [C++]
  case Decl::NamespaceAlias:
  case Decl::UsingDirective: // using namespace X; [C++]
    assert(!UnimplementedFeature::generateDebugInfo() && "NYI");
    break;
  case Decl::CXXConstructor:
    getCXXABI().buildCXXConstructors(cast<CXXConstructorDecl>(decl));
    break;
  case Decl::CXXDestructor:
    getCXXABI().buildCXXDestructors(cast<CXXDestructorDecl>(decl));
    break;

  case Decl::StaticAssert:
    // Nothing to do.
    break;

  case Decl::LinkageSpec:
    buildLinkageSpec(cast<LinkageSpecDecl>(decl));
    break;

  case Decl::Typedef:
  case Decl::TypeAlias: // using foo = bar; [C++11]
  case Decl::Record:
  case Decl::Enum:
    assert(!UnimplementedFeature::generateDebugInfo() && "NYI");
    break;
  }
}

static bool shouldBeInCOMDAT(CIRGenModule &CGM, const Decl &D) {
  if (!CGM.supportsCOMDAT())
    return false;

  if (D.hasAttr<SelectAnyAttr>())
    return true;

  GVALinkage Linkage;
  if (auto *VD = dyn_cast<VarDecl>(&D))
    Linkage = CGM.getASTContext().GetGVALinkageForVariable(VD);
  else
    Linkage =
        CGM.getASTContext().GetGVALinkageForFunction(cast<FunctionDecl>(&D));

  switch (Linkage) {
  case clang::GVA_Internal:
  case clang::GVA_AvailableExternally:
  case clang::GVA_StrongExternal:
    return false;
  case clang::GVA_DiscardableODR:
  case clang::GVA_StrongODR:
    return true;
  }
  llvm_unreachable("No such linkage");
}

// TODO(cir): this could be a common method between LLVM codegen.
static bool isVarDeclStrongDefinition(const ASTContext &Context,
                                      CIRGenModule &CGM, const VarDecl *D,
                                      bool NoCommon) {
  // Don't give variables common linkage if -fno-common was specified unless it
  // was overridden by a NoCommon attribute.
  if ((NoCommon || D->hasAttr<NoCommonAttr>()) && !D->hasAttr<CommonAttr>())
    return true;

  // C11 6.9.2/2:
  //   A declaration of an identifier for an object that has file scope without
  //   an initializer, and without a storage-class specifier or with the
  //   storage-class specifier static, constitutes a tentative definition.
  if (D->getInit() || D->hasExternalStorage())
    return true;

  // A variable cannot be both common and exist in a section.
  if (D->hasAttr<SectionAttr>())
    return true;

  // A variable cannot be both common and exist in a section.
  // We don't try to determine which is the right section in the front-end.
  // If no specialized section name is applicable, it will resort to default.
  if (D->hasAttr<PragmaClangBSSSectionAttr>() ||
      D->hasAttr<PragmaClangDataSectionAttr>() ||
      D->hasAttr<PragmaClangRelroSectionAttr>() ||
      D->hasAttr<PragmaClangRodataSectionAttr>())
    return true;

  // Thread local vars aren't considered common linkage.
  if (D->getTLSKind())
    return true;

  // Tentative definitions marked with WeakImportAttr are true definitions.
  if (D->hasAttr<WeakImportAttr>())
    return true;

  // A variable cannot be both common and exist in a comdat.
  if (shouldBeInCOMDAT(CGM, *D))
    return true;

  // Declarations with a required alignment do not have common linkage in MSVC
  // mode.
  if (Context.getTargetInfo().getCXXABI().isMicrosoft()) {
    if (D->hasAttr<AlignedAttr>())
      return true;
    QualType VarType = D->getType();
    if (Context.isAlignmentRequired(VarType))
      return true;

    if (const auto *RT = VarType->getAs<RecordType>()) {
      const RecordDecl *RD = RT->getDecl();
      for (const FieldDecl *FD : RD->fields()) {
        if (FD->isBitField())
          continue;
        if (FD->hasAttr<AlignedAttr>())
          return true;
        if (Context.isAlignmentRequired(FD->getType()))
          return true;
      }
    }
  }

  // Microsoft's link.exe doesn't support alignments greater than 32 bytes for
  // common symbols, so symbols with greater alignment requirements cannot be
  // common.
  // Other COFF linkers (ld.bfd and LLD) support arbitrary power-of-two
  // alignments for common symbols via the aligncomm directive, so this
  // restriction only applies to MSVC environments.
  if (Context.getTargetInfo().getTriple().isKnownWindowsMSVCEnvironment() &&
      Context.getTypeAlignIfKnown(D->getType()) >
          Context.toBits(CharUnits::fromQuantity(32)))
    return true;

  return false;
}

void CIRGenModule::setInitializer(mlir::cir::GlobalOp &global,
                                  mlir::Attribute value) {
  // Recompute visibility when updating initializer.
  global.setInitialValueAttr(value);
  mlir::SymbolTable::setSymbolVisibility(
      global, CIRGenModule::getMLIRVisibility(global));
}

mlir::SymbolTable::Visibility
CIRGenModule::getMLIRVisibility(mlir::cir::GlobalOp op) {
  // MLIR doesn't accept public symbols declarations (only
  // definitions).
  if (op.isDeclaration())
    return mlir::SymbolTable::Visibility::Private;
  return getMLIRVisibilityFromCIRLinkage(op.getLinkage());
}

mlir::SymbolTable::Visibility CIRGenModule::getMLIRVisibilityFromCIRLinkage(
    mlir::cir::GlobalLinkageKind GLK) {
  switch (GLK) {
  case mlir::cir::GlobalLinkageKind::InternalLinkage:
  case mlir::cir::GlobalLinkageKind::PrivateLinkage:
    return mlir::SymbolTable::Visibility::Private;
  case mlir::cir::GlobalLinkageKind::ExternalLinkage:
  case mlir::cir::GlobalLinkageKind::ExternalWeakLinkage:
  case mlir::cir::GlobalLinkageKind::LinkOnceODRLinkage:
  case mlir::cir::GlobalLinkageKind::AvailableExternallyLinkage:
  case mlir::cir::GlobalLinkageKind::CommonLinkage:
    return mlir::SymbolTable::Visibility::Public;
  default: {
    llvm::errs() << "visibility not implemented for '"
                 << stringifyGlobalLinkageKind(GLK) << "'\n";
    assert(0 && "not implemented");
  }
  }
  llvm_unreachable("linkage should be handled above!");
}

mlir::cir::GlobalLinkageKind CIRGenModule::getCIRLinkageForDeclarator(
    const DeclaratorDecl *D, GVALinkage Linkage, bool IsConstantVariable) {
  if (Linkage == GVA_Internal)
    return mlir::cir::GlobalLinkageKind::InternalLinkage;

  if (D->hasAttr<WeakAttr>()) {
    if (IsConstantVariable)
      return mlir::cir::GlobalLinkageKind::WeakODRLinkage;
    else
      return mlir::cir::GlobalLinkageKind::WeakAnyLinkage;
  }

  if (const auto *FD = D->getAsFunction())
    if (FD->isMultiVersion() && Linkage == GVA_AvailableExternally)
      return mlir::cir::GlobalLinkageKind::LinkOnceAnyLinkage;

  // We are guaranteed to have a strong definition somewhere else,
  // so we can use available_externally linkage.
  if (Linkage == GVA_AvailableExternally)
    return mlir::cir::GlobalLinkageKind::AvailableExternallyLinkage;

  // Note that Apple's kernel linker doesn't support symbol
  // coalescing, so we need to avoid linkonce and weak linkages there.
  // Normally, this means we just map to internal, but for explicit
  // instantiations we'll map to external.

  // In C++, the compiler has to emit a definition in every translation unit
  // that references the function.  We should use linkonce_odr because
  // a) if all references in this translation unit are optimized away, we
  // don't need to codegen it.  b) if the function persists, it needs to be
  // merged with other definitions. c) C++ has the ODR, so we know the
  // definition is dependable.
  if (Linkage == GVA_DiscardableODR)
    return !astCtx.getLangOpts().AppleKext
               ? mlir::cir::GlobalLinkageKind::LinkOnceODRLinkage
               : mlir::cir::GlobalLinkageKind::InternalLinkage;

  // An explicit instantiation of a template has weak linkage, since
  // explicit instantiations can occur in multiple translation units
  // and must all be equivalent. However, we are not allowed to
  // throw away these explicit instantiations.
  //
  // CUDA/HIP: For -fno-gpu-rdc case, device code is limited to one TU,
  // so say that CUDA templates are either external (for kernels) or internal.
  // This lets llvm perform aggressive inter-procedural optimizations. For
  // -fgpu-rdc case, device function calls across multiple TU's are allowed,
  // therefore we need to follow the normal linkage paradigm.
  if (Linkage == GVA_StrongODR) {
    if (getLangOpts().AppleKext)
      return mlir::cir::GlobalLinkageKind::ExternalLinkage;
    if (getLangOpts().CUDA && getLangOpts().CUDAIsDevice &&
        !getLangOpts().GPURelocatableDeviceCode)
      return D->hasAttr<CUDAGlobalAttr>()
                 ? mlir::cir::GlobalLinkageKind::ExternalLinkage
                 : mlir::cir::GlobalLinkageKind::InternalLinkage;
    return mlir::cir::GlobalLinkageKind::WeakODRLinkage;
  }

  // C++ doesn't have tentative definitions and thus cannot have common
  // linkage.
  if (!getLangOpts().CPlusPlus && isa<VarDecl>(D) &&
      !isVarDeclStrongDefinition(astCtx, *this, cast<VarDecl>(D),
                                 getCodeGenOpts().NoCommon))
    return mlir::cir::GlobalLinkageKind::CommonLinkage;

  // selectany symbols are externally visible, so use weak instead of
  // linkonce.  MSVC optimizes away references to const selectany globals, so
  // all definitions should be the same and ODR linkage should be used.
  // http://msdn.microsoft.com/en-us/library/5tkz6s71.aspx
  if (D->hasAttr<SelectAnyAttr>())
    return mlir::cir::GlobalLinkageKind::WeakODRLinkage;

  // Otherwise, we have strong external linkage.
  assert(Linkage == GVA_StrongExternal);
  return mlir::cir::GlobalLinkageKind::ExternalLinkage;
}

/// This function is called when we implement a function with no prototype, e.g.
/// "int foo() {}". If there are existing call uses of the old function in the
/// module, this adjusts them to call the new function directly.
///
/// This is not just a cleanup: the always_inline pass requires direct calls to
/// functions to be able to inline them.  If there is a bitcast in the way, it
/// won't inline them. Instcombine normally deletes these calls, but it isn't
/// run at -O0.
void CIRGenModule::ReplaceUsesOfNonProtoTypeWithRealFunction(
    mlir::Operation *Old, mlir::cir::FuncOp NewFn) {

  // If we're redefining a global as a function, don't transform it.
  auto OldFn = dyn_cast<mlir::cir::FuncOp>(Old);
  if (!OldFn)
    return;

  // TODO(cir): this RAUW ignores the features below.
  assert(!UnimplementedFeature::exceptions() && "Call vs Invoke NYI");
  assert(!UnimplementedFeature::parameterAttributes());
  assert(!UnimplementedFeature::operandBundles());
  assert(OldFn->getAttrs().size() > 1 && "Attribute forwarding NYI");

  // Mark new function as originated from a no-proto declaration.
  NewFn.setNoProtoAttr(OldFn.getNoProtoAttr());

  // Iterate through all calls of the no-proto function.
  auto Calls = OldFn.getSymbolUses(OldFn->getParentOp());
  for (auto Call : Calls.value()) {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Fetch no-proto call to be replaced.
    auto noProtoCallOp = dyn_cast<mlir::cir::CallOp>(Call.getUser());
    assert(noProtoCallOp && "unexpected use of no-proto function");
    builder.setInsertionPoint(noProtoCallOp);

    // Patch call type with the real function type.
    auto realCallOp = builder.create<mlir::cir::CallOp>(
        noProtoCallOp.getLoc(), NewFn, noProtoCallOp.getOperands());

    // Replace old no proto call with fixed call.
    noProtoCallOp.replaceAllUsesWith(realCallOp);
    noProtoCallOp.erase();
  }
}

mlir::cir::GlobalLinkageKind
CIRGenModule::getCIRLinkageVarDefinition(const VarDecl *VD, bool IsConstant) {
  assert(!IsConstant && "constant variables NYI");
  GVALinkage Linkage = astCtx.GetGVALinkageForVariable(VD);
  return getCIRLinkageForDeclarator(VD, Linkage, IsConstant);
}

mlir::cir::GlobalLinkageKind CIRGenModule::getFunctionLinkage(GlobalDecl GD) {
  const auto *D = cast<FunctionDecl>(GD.getDecl());

  GVALinkage Linkage = astCtx.GetGVALinkageForFunction(D);

  if (const auto *Dtor = dyn_cast<CXXDestructorDecl>(D))
    return getCXXABI().getCXXDestructorLinkage(Linkage, Dtor, GD.getDtorType());

  if (isa<CXXConstructorDecl>(D) &&
      cast<CXXConstructorDecl>(D)->isInheritingConstructor() &&
      astCtx.getTargetInfo().getCXXABI().isMicrosoft()) {
    // Just like in LLVM codegen:
    // Our approach to inheriting constructors is fundamentally different from
    // that used by the MS ABI, so keep our inheriting constructor thunks
    // internal rather than trying to pick an unambiguous mangling for them.
    return mlir::cir::GlobalLinkageKind::InternalLinkage;
  }

  return getCIRLinkageForDeclarator(D, Linkage, /*IsConstantVariable=*/false);
}

void CIRGenModule::buildAliasForGlobal(StringRef mangledName,
                                       mlir::Operation *op, GlobalDecl aliasGD,
                                       mlir::cir::FuncOp aliasee,
                                       mlir::cir::GlobalLinkageKind linkage) {
  auto *aliasFD = dyn_cast<FunctionDecl>(aliasGD.getDecl());
  assert(aliasFD && "expected FunctionDecl");
  auto alias =
      createCIRFunction(getLoc(aliasGD.getDecl()->getSourceRange()),
                        mangledName, aliasee.getFunctionType(), aliasFD);
  alias.setAliasee(aliasee.getName());
  alias.setLinkage(linkage);
  mlir::SymbolTable::setSymbolVisibility(
      alias, getMLIRVisibilityFromCIRLinkage(linkage));

  // Alias constructors and destructors are always unnamed_addr.
  assert(!UnimplementedFeature::unnamedAddr());

  // Switch any previous uses to the alias.
  if (op) {
    llvm_unreachable("NYI");
  } else {
    // Name already set by createCIRFunction
  }

  // Finally, set up the alias with its proper name and attributes.
  setCommonAttributes(aliasGD, alias);
}

mlir::Type CIRGenModule::getCIRType(const QualType &type) {
  return genTypes.ConvertType(type);
}

bool CIRGenModule::verifyModule() {
  // Verify the module after we have finished constructing it, this will
  // check the structural properties of the IR and invoke any specific
  // verifiers we have on the CIR operations.
  return mlir::verify(theModule).succeeded();
}

std::pair<mlir::cir::FuncType, mlir::cir::FuncOp>
CIRGenModule::getAddrAndTypeOfCXXStructor(GlobalDecl GD,
                                          const CIRGenFunctionInfo *FnInfo,
                                          mlir::cir::FuncType FnType,
                                          bool Dontdefer,
                                          ForDefinition_t IsForDefinition) {
  auto *MD = cast<CXXMethodDecl>(GD.getDecl());

  if (isa<CXXDestructorDecl>(MD)) {
    // Always alias equivalent complete destructors to base destructors in the
    // MS ABI.
    if (getTarget().getCXXABI().isMicrosoft() &&
        GD.getDtorType() == Dtor_Complete &&
        MD->getParent()->getNumVBases() == 0)
      llvm_unreachable("NYI");
  }

  if (!FnType) {
    if (!FnInfo)
      FnInfo = &getTypes().arrangeCXXStructorDeclaration(GD);
    FnType = getTypes().GetFunctionType(*FnInfo);
  }

  auto Fn = GetOrCreateCIRFunction(getMangledName(GD), FnType, GD,
                                   /*ForVtable=*/false, Dontdefer,
                                   /*IsThunk=*/false, IsForDefinition);

  return {FnType, Fn};
}

mlir::cir::FuncOp
CIRGenModule::GetAddrOfFunction(clang::GlobalDecl GD, mlir::Type Ty,
                                bool ForVTable, bool DontDefer,
                                ForDefinition_t IsForDefinition) {
  assert(!cast<FunctionDecl>(GD.getDecl())->isConsteval() &&
         "consteval function should never be emitted");

  if (!Ty) {
    const auto *FD = cast<FunctionDecl>(GD.getDecl());
    Ty = getTypes().ConvertType(FD->getType());
  }

  // Devirtualized destructor calls may come through here instead of via
  // getAddrOfCXXStructor. Make sure we use the MS ABI base destructor instead
  // of the complete destructor when necessary.
  if (const auto *DD = dyn_cast<CXXDestructorDecl>(GD.getDecl())) {
    if (getTarget().getCXXABI().isMicrosoft() &&
        GD.getDtorType() == Dtor_Complete &&
        DD->getParent()->getNumVBases() == 0)
      llvm_unreachable("NYI");
  }

  StringRef MangledName = getMangledName(GD);
  auto F = GetOrCreateCIRFunction(MangledName, Ty, GD, ForVTable, DontDefer,
                                  /*IsThunk=*/false, IsForDefinition);

  assert(!langOpts.CUDA && "NYI");

  return F;
}

// Returns true if GD is a function decl with internal linkage and needs a
// unique suffix after the mangled name.
static bool isUniqueInternalLinkageDecl(GlobalDecl GD, CIRGenModule &CGM) {
  assert(CGM.getModuleNameHash().empty() &&
         "Unique internal linkage names NYI");

  return false;
}

static std::string getMangledNameImpl(CIRGenModule &CGM, GlobalDecl GD,
                                      const NamedDecl *ND,
                                      bool OmitMultiVersionMangling = false) {
  assert(!OmitMultiVersionMangling && "NYI");

  SmallString<256> Buffer;

  llvm::raw_svector_ostream Out(Buffer);
  MangleContext &MC = CGM.getCXXABI().getMangleContext();

  assert(CGM.getModuleNameHash().empty() && "NYI");
  auto ShouldMangle = MC.shouldMangleDeclName(ND);

  if (ShouldMangle) {
    MC.mangleName(GD.getWithDecl(ND), Out);
  } else {
    auto *II = ND->getIdentifier();
    assert(II && "Attempt to mangle unnamed decl.");

    const auto *FD = dyn_cast<FunctionDecl>(ND);

    if (FD &&
        FD->getType()->castAs<FunctionType>()->getCallConv() == CC_X86RegCall) {
      assert(0 && "NYI");
    } else if (FD && FD->hasAttr<CUDAGlobalAttr>() &&
               GD.getKernelReferenceKind() == KernelReferenceKind::Stub) {
      assert(0 && "NYI");
    } else {
      Out << II->getName();
    }
  }

  // Check if the module name hash should be appended for internal linkage
  // symbols. This should come before multi-version target suffixes are
  // appendded. This is to keep the name and module hash suffix of the internal
  // linkage function together. The unique suffix should only be added when name
  // mangling is done to make sure that the final name can be properly
  // demangled. For example, for C functions without prototypes, name mangling
  // is not done and the unique suffix should not be appended then.
  assert(!isUniqueInternalLinkageDecl(GD, CGM) && "NYI");

  if (const auto *FD = dyn_cast<FunctionDecl>(ND)) {
    assert(!FD->isMultiVersion() && "NYI");
  }
  assert(!CGM.getLangOpts().GPURelocatableDeviceCode && "NYI");

  return std::string(Out.str());
}

StringRef CIRGenModule::getMangledName(GlobalDecl GD) {
  auto CanonicalGD = GD.getCanonicalDecl();

  // Some ABIs don't have constructor variants. Make sure that base and complete
  // constructors get mangled the same.
  if (const auto *CD = dyn_cast<CXXConstructorDecl>(CanonicalGD.getDecl())) {
    if (!getTarget().getCXXABI().hasConstructorVariants()) {
      assert(false && "NYI");
    }
  }

  assert(!langOpts.CUDAIsDevice && "NYI");

  // Keep the first result in the case of a mangling collision.
  const auto *ND = cast<NamedDecl>(GD.getDecl());
  std::string MangledName = getMangledNameImpl(*this, GD, ND);

  auto Result = Manglings.insert(std::make_pair(MangledName, GD));
  return MangledDeclNames[CanonicalGD] = Result.first->first();
}

void CIRGenModule::buildTentativeDefinition(const VarDecl *D) {
  assert(!D->getInit() && "Cannot emit definite definitions here!");

  StringRef MangledName = getMangledName(D);
  auto *GV = getGlobalValue(MangledName);

  // TODO(cir): can a tentative definition come from something other than a
  // global op? If not, the assertion below is wrong and should be removed. If
  // so, getGlobalValue might be better of returining a global value interface
  // that alows use to manage different globals value types transparently.
  if (GV)
    assert(isa<mlir::cir::GlobalOp>(GV) &&
           "tentative definition can only be built from a cir.global_op");

  // We already have a definition, not declaration, with the same mangled name.
  // Emitting of declaration is not required (and actually overwrites emitted
  // definition).
  if (GV && !dyn_cast<mlir::cir::GlobalOp>(GV).isDeclaration())
    return;

  // If we have not seen a reference to this variable yet, place it into the
  // deferred declarations table to be emitted if needed later.
  if (!MustBeEmitted(D) && !GV) {
    DeferredDecls[MangledName] = D;
    return;
  }

  // The tentative definition is the only definition.
  buildGlobalVarDefinition(D);
}

void CIRGenModule::setGlobalVisibility(mlir::Operation *GV,
                                       const NamedDecl *D) const {
  assert(!UnimplementedFeature::setGlobalVisibility());
}

void CIRGenModule::setDSOLocal(mlir::Operation *Op) const {
  assert(!UnimplementedFeature::setDSOLocal());
}

void CIRGenModule::setGVProperties(mlir::Operation *Op,
                                   const NamedDecl *D) const {
  assert(!UnimplementedFeature::setDLLImportDLLExport());
  setGVPropertiesAux(Op, D);
}

void CIRGenModule::setGVPropertiesAux(mlir::Operation *Op,
                                      const NamedDecl *D) const {
  setGlobalVisibility(Op, D);
  setDSOLocal(Op);
  assert(!UnimplementedFeature::setPartition());
}

bool CIRGenModule::lookupRepresentativeDecl(StringRef MangledName,
                                            GlobalDecl &Result) const {
  auto Res = Manglings.find(MangledName);
  if (Res == Manglings.end())
    return false;
  Result = Res->getValue();
  return true;
}

mlir::cir::FuncOp
CIRGenModule::createCIRFunction(mlir::Location loc, StringRef name,
                                mlir::cir::FuncType Ty,
                                const clang::FunctionDecl *FD) {
  // At the point we need to create the function, the insertion point
  // could be anywhere (e.g. callsite). Do not rely on whatever it might
  // be, properly save, find the appropriate place and restore.
  FuncOp f;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Some global emissions are triggered while emitting a function, e.g.
    // void s() { x.method() }
    //
    // Be sure to insert a new function before a current one.
    auto *curCGF = getCurrCIRGenFun();
    if (curCGF)
      builder.setInsertionPoint(curCGF->CurFn);

    f = builder.create<mlir::cir::FuncOp>(loc, name, Ty);

    if (FD)
      f.setAstAttr(makeFuncDeclAttr(FD, builder.getContext()));

    if (FD && !FD->hasPrototype())
      f.setNoProtoAttr(builder.getUnitAttr());

    assert(f.isDeclaration() && "expected empty body");

    // A declaration gets private visibility by default, but external linkage
    // as the default linkage.
    f.setLinkageAttr(mlir::cir::GlobalLinkageKindAttr::get(
        builder.getContext(), mlir::cir::GlobalLinkageKind::ExternalLinkage));
    mlir::SymbolTable::setSymbolVisibility(
        f, mlir::SymbolTable::Visibility::Private);

    setExtraAttributesForFunc(f, FD);

    if (!curCGF)
      theModule.push_back(f);
  }
  return f;
}

bool isDefaultedMethod(const clang::FunctionDecl *FD) {
  if (FD->isDefaulted() && isa<CXXMethodDecl>(FD) &&
      (cast<CXXMethodDecl>(FD)->isCopyAssignmentOperator() ||
       cast<CXXMethodDecl>(FD)->isMoveAssignmentOperator()))
    return true;
  return false;
}

mlir::Location CIRGenModule::getLocForFunction(const clang::FunctionDecl *FD) {
  assert(FD && "Not sure which location to use yet");
  bool invalidLoc = (FD->getSourceRange().getBegin().isInvalid() ||
                     FD->getSourceRange().getEnd().isInvalid());
  if (!invalidLoc)
    return getLoc(FD->getSourceRange());

  // Use the module location
  return theModule->getLoc();
}

void CIRGenModule::setExtraAttributesForFunc(FuncOp f,
                                             const clang::FunctionDecl *FD) {
  mlir::NamedAttrList attrs;

  if (!FD) {
    // If we don't have a declaration to control inlining, the function isn't
    // explicitly marked as alwaysinline for semantic reasons, and inlining is
    // disabled, mark the function as noinline.
    if (codeGenOpts.getInlining() == CodeGenOptions::OnlyAlwaysInlining) {
      auto attr = mlir::cir::InlineAttr::get(
          builder.getContext(), mlir::cir::InlineKind::AlwaysInline);
      attrs.set(attr.getMnemonic(), attr);
    }
  } else if (FD->hasAttr<NoInlineAttr>()) {
    // Add noinline if the function isn't always_inline.
    auto attr = mlir::cir::InlineAttr::get(builder.getContext(),
                                           mlir::cir::InlineKind::NoInline);
    attrs.set(attr.getMnemonic(), attr);
  } else if (FD->hasAttr<AlwaysInlineAttr>()) {
    // (noinline wins over always_inline, and we can't specify both in IR)
    auto attr = mlir::cir::InlineAttr::get(builder.getContext(),
                                           mlir::cir::InlineKind::AlwaysInline);
    attrs.set(attr.getMnemonic(), attr);
  } else if (codeGenOpts.getInlining() == CodeGenOptions::OnlyAlwaysInlining) {
    // If we're not inlining, then force everything that isn't always_inline
    // to carry an explicit noinline attribute.
    auto attr = mlir::cir::InlineAttr::get(builder.getContext(),
                                           mlir::cir::InlineKind::NoInline);
    attrs.set(attr.getMnemonic(), attr);
  } else {
    // Otherwise, propagate the inline hint attribute and potentially use its
    // absence to mark things as noinline.
    // Search function and template pattern redeclarations for inline.
    auto CheckForInline = [](const FunctionDecl *FD) {
      auto CheckRedeclForInline = [](const FunctionDecl *Redecl) {
        return Redecl->isInlineSpecified();
      };
      if (any_of(FD->redecls(), CheckRedeclForInline))
        return true;
      const FunctionDecl *Pattern = FD->getTemplateInstantiationPattern();
      if (!Pattern)
        return false;
      return any_of(Pattern->redecls(), CheckRedeclForInline);
    };
    if (CheckForInline(FD)) {
      auto attr = mlir::cir::InlineAttr::get(builder.getContext(),
                                             mlir::cir::InlineKind::InlineHint);
      attrs.set(attr.getMnemonic(), attr);
    } else if (codeGenOpts.getInlining() == CodeGenOptions::OnlyHintInlining) {
      auto attr = mlir::cir::InlineAttr::get(builder.getContext(),
                                             mlir::cir::InlineKind::NoInline);
      attrs.set(attr.getMnemonic(), attr);
    }
  }

  // Track whether we need to add the optnone attribute,
  // starting with the default for this optimization level.
  bool ShouldAddOptNone =
      !codeGenOpts.DisableO0ImplyOptNone && codeGenOpts.OptimizationLevel == 0;
  if (FD) {
    ShouldAddOptNone &= !FD->hasAttr<MinSizeAttr>();
    ShouldAddOptNone &= !FD->hasAttr<AlwaysInlineAttr>();
    ShouldAddOptNone |= FD->hasAttr<OptimizeNoneAttr>();
  }

  if (ShouldAddOptNone) {
    auto optNoneAttr = mlir::cir::OptNoneAttr::get(builder.getContext());
    attrs.set(optNoneAttr.getMnemonic(), optNoneAttr);

    // OptimizeNone implies noinline; we should not be inlining such functions.
    auto noInlineAttr = mlir::cir::InlineAttr::get(
        builder.getContext(), mlir::cir::InlineKind::NoInline);
    attrs.set(noInlineAttr.getMnemonic(), noInlineAttr);
  }

  f.setExtraAttrsAttr(mlir::cir::ExtraFuncAttributesAttr::get(
      builder.getContext(), attrs.getDictionary(builder.getContext())));
}

/// If the specified mangled name is not in the module,
/// create and return a CIR Function with the specified type. If there is
/// something in the module with the specified name, return it potentially
/// bitcasted to the right type.
///
/// If D is non-null, it specifies a decl that corresponded to this. This is
/// used to set the attributes on the function when it is first created.
mlir::cir::FuncOp CIRGenModule::GetOrCreateCIRFunction(
    StringRef MangledName, mlir::Type Ty, GlobalDecl GD, bool ForVTable,
    bool DontDefer, bool IsThunk, ForDefinition_t IsForDefinition) {
  assert(!IsThunk && "NYI");

  const auto *D = GD.getDecl();

  // Any attempts to use a MultiVersion function should result in retrieving the
  // iFunc instead. Name mangling will handle the rest of the changes.
  if (const auto *FD = cast_or_null<FunctionDecl>(D)) {
    // For the device mark the function as one that should be emitted.
    if (getLangOpts().OpenMPIsTargetDevice && FD->isDefined() && !DontDefer &&
        !IsForDefinition) {
      assert(0 && "OpenMP target functions NYI");
    }
    if (FD->isMultiVersion())
      llvm_unreachable("NYI");
  }

  // Lookup the entry, lazily creating it if necessary.
  mlir::Operation *Entry = getGlobalValue(MangledName);
  if (Entry) {
    assert(isa<mlir::cir::FuncOp>(Entry) &&
           "not implemented, only supports FuncOp for now");

    if (WeakRefReferences.erase(Entry)) {
      llvm_unreachable("NYI");
    }

    // Handle dropped DLL attributes.
    if (D && !D->hasAttr<DLLImportAttr>() && !D->hasAttr<DLLExportAttr>()) {
      // TODO(CIR): Entry->setDLLStorageClass
      setDSOLocal(Entry);
    }

    // If there are two attempts to define the same mangled name, issue an
    // error.
    auto Fn = cast<mlir::cir::FuncOp>(Entry);
    if (IsForDefinition && Fn && !Fn.isDeclaration()) {
      GlobalDecl OtherGD;
      // CHeck that GD is not yet in DiagnosedConflictingDefinitions is required
      // to make sure that we issue and error only once.
      if (lookupRepresentativeDecl(MangledName, OtherGD) &&
          (GD.getCanonicalDecl().getDecl()) &&
          DiagnosedConflictingDefinitions.insert(GD).second) {
        getDiags().Report(D->getLocation(), diag::err_duplicate_mangled_name)
            << MangledName;
        getDiags().Report(OtherGD.getDecl()->getLocation(),
                          diag::note_previous_definition);
      }
    }

    if (Fn && Fn.getFunctionType() == Ty) {
      return Fn;
    }

    if (!IsForDefinition) {
      return Fn;
    }

    // TODO: clang checks here if this is a llvm::GlobalAlias... how will we
    // support this?
  }

  // This function doesn't have a complete type (for example, the return type is
  // an incomplete struct). Use a fake type instead, and make sure not to try to
  // set attributes.
  bool IsIncompleteFunction = false;

  mlir::cir::FuncType FTy;
  if (Ty.isa<mlir::cir::FuncType>()) {
    FTy = Ty.cast<mlir::cir::FuncType>();
  } else {
    assert(false && "NYI");
    // FTy = mlir::FunctionType::get(VoidTy, false);
    IsIncompleteFunction = true;
  }

  auto *FD = llvm::cast<FunctionDecl>(D);
  assert(FD && "Only FunctionDecl supported so far.");

  // TODO: CodeGen includeds the linkage (ExternalLinkage) and only passes the
  // mangledname if Entry is nullptr
  auto F = createCIRFunction(getLocForFunction(FD), MangledName, FTy, FD);

  // If we already created a function with the same mangled name (but different
  // type) before, take its name and add it to the list of functions to be
  // replaced with F at the end of CodeGen.
  //
  // This happens if there is a prototype for a function (e.g. "int f()") and
  // then a definition of a different type (e.g. "int f(int x)").
  if (Entry) {

    // Fetch a generic symbol-defining operation and its uses.
    auto SymbolOp = dyn_cast<mlir::SymbolOpInterface>(Entry);
    assert(SymbolOp && "Expected a symbol-defining operation");

    // TODO(cir): When can this symbol be something other than a function?
    assert(isa<mlir::cir::FuncOp>(Entry) && "NYI");

    // This might be an implementation of a function without a prototype, in
    // which case, try to do special replacement of calls which match the new
    // prototype. The really key thing here is that we also potentially drop
    // arguments from the call site so as to make a direct call, which makes the
    // inliner happier and suppresses a number of optimizer warnings (!) about
    // dropping arguments.
    if (SymbolOp.getSymbolUses(SymbolOp->getParentOp())) {
      ReplaceUsesOfNonProtoTypeWithRealFunction(Entry, F);
    }

    // Obliterate no-proto declaration.
    Entry->erase();
  }

  // TODO: This might not be valid, seems the uniqueing system doesn't make
  // sense for MLIR
  // assert(F->getName().getStringRef() == MangledName && "name was uniqued!");

  if (D)
    ; // TODO: set function attributes from the declaration

  // TODO: set function attributes from the missing attributes param

  // TODO: Handle extra attributes

  if (!DontDefer) {
    // All MSVC dtors other than the base dtor are linkonce_odr and delegate to
    // each other bottoming out wiht the base dtor. Therefore we emit non-base
    // dtors on usage, even if there is no dtor definition in the TU.
    if (isa_and_nonnull<CXXDestructorDecl>(D) &&
        getCXXABI().useThunkForDtorVariant(cast<CXXDestructorDecl>(D),
                                           GD.getDtorType())) {
      llvm_unreachable("NYI"); // addDeferredDeclToEmit(GD);
    }

    // This is the first use or definition of a mangled name. If there is a
    // deferred decl with this name, remember that we need to emit it at the end
    // of the file.
    auto DDI = DeferredDecls.find(MangledName);
    if (DDI != DeferredDecls.end()) {
      // Move the potentially referenced deferred decl to the
      // DeferredDeclsToEmit list, and remove it from DeferredDecls (since we
      // don't need it anymore).
      addDeferredDeclToEmit(DDI->second);
      DeferredDecls.erase(DDI);

      // Otherwise, there are cases we have to worry about where we're using a
      // declaration for which we must emit a definition but where we might not
      // find a top-level definition.
      //   - member functions defined inline in their classes
      //   - friend functions defined inline in some class
      //   - special member functions with implicit definitions
      // If we ever change our AST traversal to walk into class methods, this
      // will be unnecessary.
      //
      // We also don't emit a definition for a function if it's going to be an
      // entry in a vtable, unless it's already marked as used.
    } else if (getLangOpts().CPlusPlus && D) {
      // Look for a declaration that's lexically in a record.
      for (const auto *FD = cast<FunctionDecl>(D)->getMostRecentDecl(); FD;
           FD = FD->getPreviousDecl()) {
        if (isa<CXXRecordDecl>(FD->getLexicalDeclContext())) {
          if (FD->doesThisDeclarationHaveABody()) {
            if (isDefaultedMethod(FD))
              addDefaultMethodsToEmit(GD.getWithDecl(FD));
            else
              addDeferredDeclToEmit(GD.getWithDecl(FD));
            break;
          }
        }
      }
    }
  }

  if (!IsIncompleteFunction) {
    assert(F.getFunctionType() == Ty);
    return F;
  }

  // TODO(cir): Might need bitcast to different address space.
  assert(!UnimplementedFeature::addressSpace());
  return F;
}

mlir::Location CIRGenModule::getLoc(SourceLocation SLoc) {
  assert(SLoc.isValid() && "expected valid source location");
  const SourceManager &SM = astCtx.getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(SLoc);
  StringRef Filename = PLoc.getFilename();
  return mlir::FileLineColLoc::get(builder.getStringAttr(Filename),
                                   PLoc.getLine(), PLoc.getColumn());
}

mlir::Location CIRGenModule::getLoc(SourceRange SLoc) {
  assert(SLoc.isValid() && "expected valid source location");
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

void CIRGenModule::buildGlobalDecl(clang::GlobalDecl &D) {
  // We should call GetAddrOfGlobal with IsForDefinition set to true in order
  // to get a Value with exactly the type we need, not something that might
  // have been created for another decl with the same mangled name but
  // different type.
  auto *Op = GetAddrOfGlobal(D, ForDefinition);

  // In case of different address spaces, we may still get a cast, even with
  // IsForDefinition equal to true. Query mangled names table to get
  // GlobalValue.
  if (!Op) {
    Op = getGlobalValue(getMangledName(D));
  }

  // In case of different address spaces, we may still get a cast, even with
  // IsForDefinition equal to true. Query mangled names table to get
  // GlobalValue.
  if (!Op)
    llvm_unreachable("Address spaces NYI");

  // Make sure getGlobalValue returned non-null.
  assert(Op);

  // Check to see if we've already emitted this. This is necessary for a
  // couple of reasons: first, decls can end up in deferred-decls queue
  // multiple times, and second, decls can end up with definitions in unusual
  // ways (e.g. by an extern inline function acquiring a strong function
  // redefinition). Just ignore those cases.
  // TODO: Not sure what to map this to for MLIR
  if (auto Fn = dyn_cast<mlir::cir::FuncOp>(Op))
    if (!Fn.isDeclaration())
      return;

  // TODO(cir): create a global value trait that allow us to uniformly handle
  //       global variables and functions.
  if (auto Gv = dyn_cast<mlir::cir::GetGlobalOp>(Op)) {
    auto *result =
        mlir::SymbolTable::lookupSymbolIn(getModule(), Gv.getNameAttr());
    if (auto globalOp = dyn_cast<mlir::cir::GlobalOp>(result))
      if (!globalOp.isDeclaration())
        return;
  }

  // If this is OpenMP, check if it is legal to emit this global normally.
  if (getLangOpts().OpenMP && openMPRuntime &&
      openMPRuntime->emitTargetGlobal(D))
    return;

  // Otherwise, emit the definition and move on to the next one.
  buildGlobalDefinition(D, Op);
}

void CIRGenModule::buildDeferred(unsigned recursionLimit) {
  // Emit deferred declare target declarations
  if (getLangOpts().OpenMP && !getLangOpts().OpenMPSimd)
    getOpenMPRuntime().emitDeferredTargetDecls();

  // Emit code for any potentially referenced deferred decls. Since a previously
  // unused static decl may become used during the generation of code for a
  // static function, iterate until no changes are made.

  if (!DeferredVTables.empty()) {
    buildDeferredVTables();

    // Emitting a vtable doesn't directly cause more vtables to
    // become deferred, although it can cause functions to be
    // emitted that then need those vtables.
    assert(DeferredVTables.empty());
  }

  // Emit CUDA/HIP static device variables referenced by host code only. Note we
  // should not clear CUDADeviceVarODRUsedByHost since it is still needed for
  // further handling.
  if (getLangOpts().CUDA && getLangOpts().CUDAIsDevice) {
    llvm_unreachable("NYI");
  }

  // Stop if we're out of both deferred vtables and deferred declarations.
  if (DeferredDeclsToEmit.empty())
    return;

  // Grab the list of decls to emit. If buildGlobalDefinition schedules more
  // work, it will not interfere with this.
  std::vector<GlobalDecl> CurDeclsToEmit;
  CurDeclsToEmit.swap(DeferredDeclsToEmit);
  if (recursionLimit == 0)
    return;
  recursionLimit--;

  for (auto &D : CurDeclsToEmit) {
    if (getCodeGenOpts().ClangIRSkipFunctionsFromSystemHeaders) {
      auto *decl = D.getDecl();
      assert(decl && "expected decl");
      if (astCtx.getSourceManager().isInSystemHeader(decl->getLocation()))
        continue;
    }

    buildGlobalDecl(D);

    // If we found out that we need to emit more decls, do that recursively.
    // This has the advantage that the decls are emitted in a DFS and related
    // ones are close together, which is convenient for testing.
    if (!DeferredVTables.empty() || !DeferredDeclsToEmit.empty()) {
      buildDeferred(recursionLimit);
      assert(DeferredVTables.empty() && DeferredDeclsToEmit.empty());
    }
  }
}

void CIRGenModule::buildDefaultMethods() {
  // Differently from DeferredDeclsToEmit, there's no recurrent use of
  // DefaultMethodsToEmit, so use it directly for emission.
  for (auto &D : DefaultMethodsToEmit)
    buildGlobalDecl(D);
}

mlir::IntegerAttr CIRGenModule::getSize(CharUnits size) {
  // Note that mlir::IntegerType is used instead of mlir::cir::IntType here
  // because we don't need sign information for this to be useful, so keep
  // it simple.
  return mlir::IntegerAttr::get(
      mlir::IntegerType::get(builder.getContext(), 64), size.getQuantity());
}

mlir::Operation *
CIRGenModule::GetAddrOfGlobal(GlobalDecl GD, ForDefinition_t IsForDefinition) {
  const Decl *D = GD.getDecl();

  if (isa<CXXConstructorDecl>(D) || isa<CXXDestructorDecl>(D))
    return getAddrOfCXXStructor(GD, /*FnInfo=*/nullptr, /*FnType=*/nullptr,
                                /*DontDefer=*/false, IsForDefinition);

  if (isa<CXXMethodDecl>(D)) {
    auto FInfo =
        &getTypes().arrangeCXXMethodDeclaration(cast<CXXMethodDecl>(D));
    auto Ty = getTypes().GetFunctionType(*FInfo);
    return GetAddrOfFunction(GD, Ty, /*ForVTable=*/false, /*DontDefer=*/false,
                             IsForDefinition);
  }

  if (isa<FunctionDecl>(D)) {
    const CIRGenFunctionInfo &FI = getTypes().arrangeGlobalDeclaration(GD);
    auto Ty = getTypes().GetFunctionType(FI);
    return GetAddrOfFunction(GD, Ty, /*ForVTable=*/false, /*DontDefer=*/false,
                             IsForDefinition);
  }

  return getAddrOfGlobalVar(cast<VarDecl>(D), /*Ty=*/nullptr, IsForDefinition)
      .getDefiningOp();
}

void CIRGenModule::Release() {
  buildDeferred(getCodeGenOpts().ClangIRBuildDeferredThreshold);
  // TODO: buildVTablesOpportunistically();
  // TODO: applyGlobalValReplacements();
  applyReplacements();
  // TODO: checkAliases();
  // TODO: buildMultiVersionFunctions();
  buildCXXGlobalInitFunc();
  // TODO: buildCXXGlobalCleanUpFunc();
  // TODO: registerGlobalDtorsWithAtExit();
  // TODO: buildCXXThreadLocalInitFunc();
  // TODO: ObjCRuntime
  if (astCtx.getLangOpts().CUDA) {
    llvm_unreachable("NYI");
  }
  // TODO: OpenMPRuntime
  // TODO: PGOReader
  // TODO: buildCtorList(GlobalCtors);
  // TODO: builtCtorList(GlobalDtors);
  // TODO: buildGlobalAnnotations();
  // TODO: buildDeferredUnusedCoverageMappings();
  // TODO: CIRGenPGO
  // TODO: CoverageMapping
  if (getCodeGenOpts().SanitizeCfiCrossDso) {
    llvm_unreachable("NYI");
  }
  // TODO: buildAtAvailableLinkGuard();
  if (astCtx.getTargetInfo().getTriple().isWasm() &&
      !astCtx.getTargetInfo().getTriple().isOSEmscripten()) {
    llvm_unreachable("NYI");
  }

  // Emit reference of __amdgpu_device_library_preserve_asan_functions to
  // preserve ASAN functions in bitcode libraries.
  if (getLangOpts().Sanitize.has(SanitizerKind::Address)) {
    llvm_unreachable("NYI");
  }

  // TODO: buildLLVMUsed();
  // TODO: SanStats

  if (getCodeGenOpts().Autolink) {
    // TODO: buildModuleLinkOptions
  }

  // TODO: FINISH THE REST OF THIS
}

bool CIRGenModule::shouldEmitFunction(GlobalDecl GD) {
  // TODO: implement this -- requires defining linkage for CIR
  return true;
}

bool CIRGenModule::supportsCOMDAT() const {
  return getTriple().supportsCOMDAT();
}

void CIRGenModule::maybeSetTrivialComdat(const Decl &D, mlir::Operation *Op) {
  if (!shouldBeInCOMDAT(*this, D))
    return;

  // TODO: Op.setComdat
  assert(!UnimplementedFeature::setComdat() && "NYI");
}

bool CIRGenModule::isInNoSanitizeList(SanitizerMask Kind, mlir::cir::FuncOp Fn,
                                      SourceLocation Loc) const {
  const auto &NoSanitizeL = getASTContext().getNoSanitizeList();
  // NoSanitize by function name.
  if (NoSanitizeL.containsFunction(Kind, Fn.getName()))
    llvm_unreachable("NYI");
  // NoSanitize by location.
  if (Loc.isValid())
    return NoSanitizeL.containsLocation(Kind, Loc);
  // If location is unknown, this may be a compiler-generated function. Assume
  // it's located in the main file.
  auto &SM = getASTContext().getSourceManager();
  FileEntryRef MainFile = *SM.getFileEntryRefForID(SM.getMainFileID());
  if (NoSanitizeL.containsFile(Kind, MainFile.getName()))
    return true;

  // Check "src" prefix.
  if (Loc.isValid())
    return NoSanitizeL.containsLocation(Kind, Loc);
  // If location is unknown, this may be a compiler-generated function. Assume
  // it's located in the main file.
  return NoSanitizeL.containsFile(Kind, MainFile.getName());
}

void CIRGenModule::AddDeferredUnusedCoverageMapping(Decl *D) {
  // Do we need to generate coverage mapping?
  if (!codeGenOpts.CoverageMapping)
    return;

  llvm_unreachable("NYI");
}

void CIRGenModule::UpdateCompletedType(const TagDecl *TD) {
  // Make sure that this type is translated.
  genTypes.UpdateCompletedType(TD);
}

void CIRGenModule::addReplacement(StringRef Name, mlir::Operation *Op) {
  Replacements[Name] = Op;
}

void CIRGenModule::applyReplacements() {
  for (auto &I : Replacements) {
    StringRef MangledName = I.first();
    mlir::Operation *Replacement = I.second;
    auto *Entry = getGlobalValue(MangledName);
    if (!Entry)
      continue;
    assert(isa<mlir::cir::FuncOp>(Entry) && "expected function");
    auto OldF = cast<mlir::cir::FuncOp>(Entry);
    auto NewF = dyn_cast<mlir::cir::FuncOp>(Replacement);
    assert(NewF && "not implemented");

    // Replace old with new, but keep the old order.
    if (OldF.replaceAllSymbolUses(NewF.getSymNameAttr(), theModule).failed())
      llvm_unreachable("internal error, cannot RAUW symbol");
    if (NewF) {
      NewF->moveBefore(OldF);
      OldF->erase();
    }
  }
}

void CIRGenModule::buildExplicitCastExprType(const ExplicitCastExpr *E,
                                             CIRGenFunction *CGF) {
  // Bind VLAs in the cast type.
  if (CGF && E->getType()->isVariablyModifiedType())
    llvm_unreachable("NYI");

  assert(!UnimplementedFeature::generateDebugInfo() && "NYI");
}

void CIRGenModule::HandleCXXStaticMemberVarInstantiation(VarDecl *VD) {
  auto DK = VD->isThisDeclarationADefinition();
  if (DK == VarDecl::Definition && VD->hasAttr<DLLImportAttr>())
    return;

  TemplateSpecializationKind TSK = VD->getTemplateSpecializationKind();
  // If we have a definition, this might be a deferred decl. If the
  // instantiation is explicit, make sure we emit it at the end.
  if (VD->getDefinition() && TSK == TSK_ExplicitInstantiationDefinition) {
    llvm_unreachable("NYI");
  }

  buildTopLevelDecl(VD);
}

mlir::cir::GlobalOp CIRGenModule::createOrReplaceCXXRuntimeVariable(
    mlir::Location loc, StringRef Name, mlir::Type Ty,
    mlir::cir::GlobalLinkageKind Linkage, clang::CharUnits Alignment) {
  mlir::cir::GlobalOp OldGV{};
  auto GV = dyn_cast_or_null<mlir::cir::GlobalOp>(
      mlir::SymbolTable::lookupSymbolIn(getModule(), Name));

  if (GV) {
    // Check if the variable has the right type.
    if (GV.getSymType() == Ty)
      return GV;

    // Because C++ name mangling, the only way we can end up with an already
    // existing global with the same name is if it has been declared extern
    // "C".
    assert(GV.isDeclaration() && "Declaration has wrong type!");
    OldGV = GV;
  }

  // Create a new variable.
  GV = CIRGenModule::createGlobalOp(*this, loc, Name, Ty);

  // Set up extra information and add to the module
  GV.setLinkageAttr(
      mlir::cir::GlobalLinkageKindAttr::get(builder.getContext(), Linkage));
  mlir::SymbolTable::setSymbolVisibility(GV,
                                         CIRGenModule::getMLIRVisibility(GV));

  if (OldGV) {
    // Replace occurrences of the old variable if needed.
    GV.setName(OldGV.getName());
    if (!OldGV->use_empty()) {
      llvm_unreachable("NYI");
    }
    OldGV->erase();
  }

  assert(!UnimplementedFeature::setComdat());
  if (supportsCOMDAT() && mlir::cir::isWeakForLinker(Linkage) &&
      !GV.hasAvailableExternallyLinkage())
    assert(!UnimplementedFeature::setComdat());

  GV.setAlignmentAttr(getSize(Alignment));
  return GV;
}

bool CIRGenModule::shouldOpportunisticallyEmitVTables() {
  if (codeGenOpts.OptimizationLevel != 0)
    llvm_unreachable("NYI");
  return codeGenOpts.OptimizationLevel > 0;
}

void CIRGenModule::buildVTableTypeMetadata(const CXXRecordDecl *RD,
                                           mlir::cir::GlobalOp VTable,
                                           const VTableLayout &VTLayout) {
  if (!getCodeGenOpts().LTOUnit)
    return;
  llvm_unreachable("NYI");
}

mlir::Attribute CIRGenModule::getAddrOfRTTIDescriptor(mlir::Location loc,
                                                      QualType Ty, bool ForEH) {
  // Return a bogus pointer if RTTI is disabled, unless it's for EH.
  // FIXME: should we even be calling this method if RTTI is disabled
  // and it's not for EH?
  if ((!ForEH && !getLangOpts().RTTI) || getLangOpts().CUDAIsDevice ||
      (getLangOpts().OpenMP && getLangOpts().OpenMPIsTargetDevice &&
       getTriple().isNVPTX()))
    llvm_unreachable("NYI");

  if (ForEH && Ty->isObjCObjectPointerType() &&
      getLangOpts().ObjCRuntime.isGNUFamily()) {
    llvm_unreachable("NYI");
  }

  return getCXXABI().getAddrOfRTTIDescriptor(loc, Ty);
}

/// TODO(cir): once we have cir.module, add this as a convenience method there.
///
/// Look up the specified global in the module symbol table.
///   1. If it does not exist, add a declaration of the global and return it.
///   2. Else, the global exists but has the wrong type: return the function
///      with a constantexpr cast to the right type.
///   3. Finally, if the existing global is the correct declaration, return the
///      existing global.
mlir::cir::GlobalOp CIRGenModule::getOrInsertGlobal(
    mlir::Location loc, StringRef Name, mlir::Type Ty,
    llvm::function_ref<mlir::cir::GlobalOp()> CreateGlobalCallback) {
  // See if we have a definition for the specified global already.
  auto GV = dyn_cast_or_null<mlir::cir::GlobalOp>(getGlobalValue(Name));
  if (!GV) {
    GV = CreateGlobalCallback();
  }
  assert(GV && "The CreateGlobalCallback is expected to create a global");

  // If the variable exists but has the wrong type, return a bitcast to the
  // right type.
  auto GVTy = GV.getSymType();
  assert(!UnimplementedFeature::addressSpace());
  auto PTy = builder.getPointerTo(Ty);

  if (GVTy != PTy)
    llvm_unreachable("NYI");

  // Otherwise, we just found the existing function or a prototype.
  return GV;
}

// Overload to construct a global variable using its constructor's defaults.
mlir::cir::GlobalOp CIRGenModule::getOrInsertGlobal(mlir::Location loc,
                                                    StringRef Name,
                                                    mlir::Type Ty) {
  return getOrInsertGlobal(loc, Name, Ty, [&] {
    return CIRGenModule::createGlobalOp(*this, loc, Name,
                                        builder.getPointerTo(Ty));
  });
}

// TODO(cir): this can be shared with LLVM codegen.
CharUnits CIRGenModule::computeNonVirtualBaseClassOffset(
    const CXXRecordDecl *DerivedClass, CastExpr::path_const_iterator Start,
    CastExpr::path_const_iterator End) {
  CharUnits Offset = CharUnits::Zero();

  const ASTContext &Context = getASTContext();
  const CXXRecordDecl *RD = DerivedClass;

  for (CastExpr::path_const_iterator I = Start; I != End; ++I) {
    const CXXBaseSpecifier *Base = *I;
    assert(!Base->isVirtual() && "Should not see virtual bases here!");

    // Get the layout.
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

    const auto *BaseDecl =
        cast<CXXRecordDecl>(Base->getType()->castAs<RecordType>()->getDecl());

    // Add the offset.
    Offset += Layout.getBaseClassOffset(BaseDecl);

    RD = BaseDecl;
  }

  return Offset;
}

void CIRGenModule::Error(SourceLocation loc, StringRef message) {
  unsigned diagID = getDiags().getCustomDiagID(DiagnosticsEngine::Error, "%0");
  getDiags().Report(astCtx.getFullLoc(loc), diagID) << message;
}

/// Print out an error that codegen doesn't support the specified stmt yet.
void CIRGenModule::ErrorUnsupported(const Stmt *S, const char *Type) {
  unsigned DiagID = getDiags().getCustomDiagID(DiagnosticsEngine::Error,
                                               "cannot compile this %0 yet");
  std::string Msg = Type;
  getDiags().Report(astCtx.getFullLoc(S->getBeginLoc()), DiagID)
      << Msg << S->getSourceRange();
}

/// Print out an error that codegen doesn't support the specified decl yet.
void CIRGenModule::ErrorUnsupported(const Decl *D, const char *Type) {
  unsigned DiagID = getDiags().getCustomDiagID(DiagnosticsEngine::Error,
                                               "cannot compile this %0 yet");
  std::string Msg = Type;
  getDiags().Report(astCtx.getFullLoc(D->getLocation()), DiagID) << Msg;
}

mlir::cir::SourceLanguage CIRGenModule::getCIRSourceLanguage() {
  using ClangStd = clang::LangStandard;
  using CIRLang = mlir::cir::SourceLanguage;
  auto opts = getLangOpts();

  if (opts.CPlusPlus || opts.CPlusPlus11 || opts.CPlusPlus14 ||
      opts.CPlusPlus17 || opts.CPlusPlus20 || opts.CPlusPlus23 ||
      opts.CPlusPlus26)
    return CIRLang::CXX;
  if (opts.C99 || opts.C11 || opts.C17 || opts.C23 ||
      opts.LangStd == ClangStd::lang_c89 ||
      opts.LangStd == ClangStd::lang_gnu89)
    return CIRLang::C;

  // TODO(cir): support remaining source languages.
  llvm_unreachable("CIR does not yet support the given source language");
}
