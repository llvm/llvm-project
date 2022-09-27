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
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/NoSanitizeList.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/CIR/CIRGenerator.h"
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
    : builder(&context), astCtx(astctx), langOpts(astctx.getLangOpts()),
      codeGenOpts(CGO),
      theModule{mlir::ModuleOp::create(builder.getUnknownLoc())}, Diags(Diags),
      target(astCtx.getTargetInfo()), ABI(createCXXABI(*this)),
      genTypes{*this} {}

CIRGenModule::~CIRGenModule() {}

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
  assert(!langOpts.OpenMP && "NYI");

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

  assert((FD || VD) &&
         "Only FunctionDecl and VarDecl should hit this path so far.");
  return true;
}

void CIRGenModule::buildGlobal(GlobalDecl GD) {
  const auto *Global = cast<ValueDecl>(GD.getDecl());

  assert(!Global->hasAttr<WeakRefAttr>() && "NYI");
  assert(!Global->hasAttr<AliasAttr>() && "NYI");
  assert(!Global->hasAttr<IFuncAttr>() && "NYI");
  assert(!Global->hasAttr<CPUDispatchAttr>() && "NYI");
  assert(!langOpts.CUDA && "NYI");
  assert(!langOpts.OpenMP && "NYI");

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
      assert(!getLangOpts().OpenMP && "not implemented");
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
  mlir::FunctionType Ty = getTypes().GetFunctionType(FI);

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
  // TODO(cir): setGVProperties
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

static mlir::cir::GlobalOp createGlobalOp(CIRGenModule &CGM, mlir::Location loc,
                                          StringRef name, mlir::Type t,
                                          bool isCst = false) {
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
      builder.setInsertionPoint(curCGF->CurFn.getOperation());

    g = builder.create<mlir::cir::GlobalOp>(loc, name, t, isCst);
    if (!curCGF)
      CGM.getModule().push_back(g);
  }
  return g;
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
        mlir::SymbolTable::setSymbolVisibility(
            Entry, getMLIRVisibilityFromCIRLinkage(LT));
      }
    }

    // Handle dropped DLL attributes.
    if (D && !D->hasAttr<clang::DLLImportAttr>() &&
        !D->hasAttr<clang::DLLExportAttr>())
      assert(!UnimplementedFeature::setDLLStorageClass() && "NYI");

    if (getLangOpts().OpenMP && !getLangOpts().OpenMPSimd && D)
      assert(0 && "not implemented");

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
  auto GV = createGlobalOp(*this, loc, MangledName, Ty,
                           /*isConstant=*/false);

  // If we already created a global with the same mangled name (but different
  // type) before, take its name and remove it from its parent.
  assert(!Entry && "not implemented");

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
  auto &LangOpts = getLangOpts();
  if (D) {
    if (LangOpts.OpenMP && !LangOpts.OpenMPSimd)
      assert(0 && "not implemented");

    // FIXME: This code is overly simple and should be merged with other global
    // handling.

    // TODO(cir):
    //   GV->setConstant(isTypeConstant(D->getType(), false));
    //   GV->setAlignment(getContext().getDeclAlign(D).getAsAlign());
    //   setLinkageForGV(GV, D);

    if (D->getTLSKind()) {
      assert(0 && "not implemented");
    }

    // TODO(cir):
    //   setGVProperties(GV, D);

    // If required by the ABI, treat declarations of static data members with
    // inline initializers as definitions.
    if (astCtx.isMSStaticDataMemberInlineDefinition(D)) {
      assert(0 && "not implemented");
    }

    // Emit section information for extern variables.
    if (D->hasExternalStorage())
      assert(0 && "not implemented");

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

mlir::cir::GlobalOp CIRGenModule::buildGlobal(const VarDecl *D,
                                              std::optional<mlir::Type> Ty,
                                              ForDefinition_t IsForDefinition) {
  assert(D->hasGlobalStorage() && "Not a global variable");
  QualType ASTTy = D->getType();
  if (!Ty)
    Ty = getTypes().convertTypeForMem(ASTTy);

  StringRef MangledName = getMangledName(D);
  return getOrCreateCIRGlobal(MangledName, *Ty, ASTTy.getAddressSpace(), D,
                              IsForDefinition);
}

/// Return the mlir::Value for the address of the given global variable. If Ty
/// is non-null and if the global doesn't exist, then it will be created with
/// the specified type instead of whatever the normal requested type would be.
/// If IsForDefinition is true, it is guaranteed that an actual global with type
/// Ty will be returned, not conversion of a variable with the same mangled name
/// but some other type.
mlir::Value CIRGenModule::getAddrOfGlobalVar(const VarDecl *D,
                                             std::optional<mlir::Type> Ty,
                                             ForDefinition_t IsForDefinition) {
  auto g = buildGlobal(D, Ty, IsForDefinition);
  auto ptrTy =
      mlir::cir::PointerType::get(builder.getContext(), g.getSymType());
  return builder.create<mlir::cir::GetGlobalOp>(getLoc(D->getSourceRange()),
                                                ptrTy, g.getSymName());
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
  assert(!(getLangOpts().OpenCL || getLangOpts().OpenMP) && "not implemented");

  // TODO(cir): LLVM's codegen uses a llvm::TrackingVH here. Is that
  // necessary here for CIR gen?
  mlir::Attribute Init;
  // TODO(cir): bool NeedsGlobalCtor = false;
  bool NeedsGlobalDtor =
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
    assert(0 && "not implemented");
  } else {
    initializedGlobalDecl = GlobalDecl(D);
    emitter.emplace(*this);
    auto Initializer = emitter->tryEmitForInitializer(*InitDecl);
    if (!Initializer) {
      assert(0 && "not implemented");
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

  // TODO(cir):
  // Set the llvm linkage type as appropriate.
  // llvm::GlobalValue::LinkageTypes Linkage =
  //     getLLVMLinkageVarDefinition(D, GV->isConstant());

  // TODO(cir):
  // CUDA B.2.1 "The __device__ qualifier declares a variable that resides on
  // the device. [...]"
  // CUDA B.2.2 "The __constant__ qualifier, optionally used together with
  // __device__, declares a variable that: [...]
  if (GV && getLangOpts().CUDA) {
    assert(0 && "not implemented");
  }

  // Set initializer and finalize emission
  GV.setInitialValueAttr(Init);
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

  // TODO(cir): set linkage, dll stuff and common linkage
  // GV->setLinkage(Linkage);
  // if (D->hasAttr<DLLImportAttr>())
  //   GV->setDLLStorageClass(llvm::GlobalVariable::DLLImportStorageClass);
  // else if (D->hasAttr<DLLExportAttr>())
  //   GV->setDLLStorageClass(llvm::GlobalVariable::DLLExportStorageClass);
  // else
  //   GV->setDLLStorageClass(llvm::GlobalVariable::DefaultStorageClass);
  //
  // if (Linkage == llvm::GlobalVariable::CommonLinkage) {
  //   // common vars aren't constant even if declared const.
  //   GV->setConstant(false);
  //   // Tentative definition of global variables may be initialized with
  //   // non-zero null pointers. In this case they should have weak linkage
  //   // since common linkage must have zero initializer and must not have
  //   // explicit section therefore cannot have non-zero initial value.
  //   if (!GV->getInitializer()->isNullValue())
  //     GV->setLinkage(llvm::GlobalVariable::WeakAnyLinkage);
  // }

  // TODO(cir): setNonAliasAttributes(D, GV);

  // TODO(cir): handle TLSKind if GV is not thread local
  if (D->getTLSKind()) { // && !GV->isThreadLocal())
    assert(0 && "not implemented");
  }

  // TODO(cir): maybeSetTrivialComdat(*D, *GV);

  // TODO(cir):
  // Emit the initializer function if necessary.
  // if (NeedsGlobalCtor || NeedsGlobalDtor)
  //   EmitCXXGlobalVarDeclInitFunc(D, GV, NeedsGlobalCtor);

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
        llvm_unreachable("NYI");

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
    auto TheType =
        mlir::cir::ArrayType::get(builder.getContext(), eltTy, finalSize);
    auto cstArray = mlir::cir::CstArrayAttr::get(
        TheType, mlir::StringAttr::get(Str, TheType));
    return cstArray;
  }

  assert(0 && "not implemented");
  return {};
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
  auto GV = createGlobalOp(CGM, loc, GlobalName, C.getType(),
                           !CGM.getLangOpts().WritableStrings);

  // Set up extra information and add to the module
  GV.setAlignmentAttr(CGM.getSize(Alignment));
  GV.setLinkageAttr(
      mlir::cir::GlobalLinkageKindAttr::get(CGM.getBuilder().getContext(), LT));
  mlir::SymbolTable::setSymbolVisibility(
      GV, CIRGenModule::getMLIRVisibilityFromCIRLinkage(LT));
  GV.setInitialValueAttr(C);

  // TODO(cir)
  assert(!cir::UnimplementedFeature::threadLocal() && "NYI");
  assert(!cir::UnimplementedFeature::unnamedAddr() && "NYI");
  assert(!mlir::cir::isWeakForLinker(LT) && "NYI");
  assert(!cir::UnimplementedFeature::setDSOLocal() && "NYI");
  return GV;
}

// In address space agnostic languages, string literals are in default address
// space in AST. However, certain targets (e.g. amdgcn) request them to be
// emitted in constant address space in LLVM IR. To be consistent with other
// parts of AST, string literal global variables in constant address space
// need to be casted to default address space before being put into address
// map and referenced by other part of CodeGen.
// In OpenCL, string literals are in constant address space in AST, therefore
// they should not be casted to default address space.
static mlir::StringAttr
castStringLiteralToDefaultAddressSpace(CIRGenModule &CGM, mlir::StringAttr GV) {
  if (!CGM.getLangOpts().OpenCL) {
    auto AS = CGM.getGlobalConstantAddressSpace();
    if (AS != LangAS::Default)
      assert(0 && "not implemented");
  }
  return GV;
}

/// Return a pointer to a constant array for the given string literal.
mlir::SymbolRefAttr
CIRGenModule::getAddrOfConstantStringFromLiteral(const StringLiteral *S,
                                                 StringRef Name) {
  CharUnits Alignment =
      astCtx.getAlignOfGlobalVarInChars(S->getType(), /*VD=*/nullptr);

  mlir::Attribute C = getConstantArrayFromStringLiteral(S);
  mlir::cir::GlobalOp Entry;
  if (!getLangOpts().WritableStrings) {
    if (ConstantStringMap.count(C)) {
      auto g = ConstantStringMap[C];
      // The bigger alignment always wins.
      if (!g.getAlignment() ||
          uint64_t(Alignment.getQuantity()) > *g.getAlignment())
        g.setAlignmentAttr(getSize(Alignment));
      return mlir::SymbolRefAttr::get(
          castStringLiteralToDefaultAddressSpace(*this, g.getSymNameAttr()));
    }
  }

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
  auto GV = generateStringLiteral(loc, typedC, LT, *this, GlobalVariableName,
                                  Alignment);
  ConstantStringMap[C] = GV;

  assert(!cir::UnimplementedFeature::reportGlobalToASan() && "NYI");
  return mlir::SymbolRefAttr::get(
      castStringLiteralToDefaultAddressSpace(*this, GV.getSymNameAttr()));
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

  case Decl::CXXMethod:
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
  case Decl::CXXConstructor:
    getCXXABI().buildCXXConstructors(cast<CXXConstructorDecl>(decl));
    break;
  case Decl::Record:
    // There's nothing to do here, we emit everything pertaining to `Record`s
    // lazily.
    // TODO: handle debug info here? See clang's
    // CodeGenModule::EmitTopLevelDecl
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

mlir::SymbolTable::Visibility CIRGenModule::getMLIRVisibilityFromCIRLinkage(
    mlir::cir::GlobalLinkageKind GLK) {
  switch (GLK) {
  case mlir::cir::GlobalLinkageKind::InternalLinkage:
  case mlir::cir::GlobalLinkageKind::PrivateLinkage:
    return mlir::SymbolTable::Visibility::Private;
  case mlir::cir::GlobalLinkageKind::ExternalLinkage:
  case mlir::cir::GlobalLinkageKind::ExternalWeakLinkage:
  case mlir::cir::GlobalLinkageKind::LinkOnceODRLinkage:
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

mlir::cir::GlobalLinkageKind CIRGenModule::getFunctionLinkage(GlobalDecl GD) {
  const auto *D = cast<FunctionDecl>(GD.getDecl());

  GVALinkage Linkage = astCtx.GetGVALinkageForFunction(D);

  if (const auto *Dtor = dyn_cast<CXXDestructorDecl>(D))
    assert(0 && "NYI");

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

mlir::Type CIRGenModule::getCIRType(const QualType &type) {
  return genTypes.ConvertType(type);
}

bool CIRGenModule::verifyModule() {
  // Verify the module after we have finished constructing it, this will
  // check the structural properties of the IR and invoke any specific
  // verifiers we have on the CIR operations.
  return mlir::verify(theModule).succeeded();
}

std::pair<mlir::FunctionType, mlir::cir::FuncOp>
CIRGenModule::getAddrAndTypeOfCXXStructor(GlobalDecl GD,
                                          const CIRGenFunctionInfo *FnInfo,
                                          mlir::FunctionType FnType,
                                          bool Dontdefer,
                                          ForDefinition_t IsForDefinition) {
  auto *MD = cast<CXXMethodDecl>(GD.getDecl());

  assert(!isa<CXXDestructorDecl>(MD) && "Destructors NYI");

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
  assert(!ForVTable && "NYI");

  assert(!cast<FunctionDecl>(GD.getDecl())->isConsteval() &&
         "consteval function should never be emitted");

  if (!Ty) {
    const auto *FD = cast<FunctionDecl>(GD.getDecl());
    Ty = getTypes().ConvertType(FD->getType());
  }

  assert(!dyn_cast<CXXDestructorDecl>(GD.getDecl()) && "NYI");

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

void CIRGenModule::setDSOLocal(mlir::Operation *Op) const {
  // TODO: Op->setDSOLocal
}

bool CIRGenModule::lookupRepresentativeDecl(StringRef MangledName,
                                            GlobalDecl &Result) const {
  auto Res = Manglings.find(MangledName);
  if (Res == Manglings.end())
    return false;
  Result = Res->getValue();
  return true;
}

mlir::cir::FuncOp CIRGenModule::createCIRFunction(mlir::Location loc,
                                                  StringRef name,
                                                  mlir::FunctionType Ty) {
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
      builder.setInsertionPoint(curCGF->CurFn.getOperation());

    f = builder.create<mlir::cir::FuncOp>(loc, name, Ty);
    assert(f.isDeclaration() && "expected empty body");

    // A declaration gets private visibility by default, but external linkage
    // as the default linkage.
    f.setLinkageAttr(mlir::cir::GlobalLinkageKindAttr::get(
        builder.getContext(), mlir::cir::GlobalLinkageKind::ExternalLinkage));
    mlir::SymbolTable::setSymbolVisibility(
        f, mlir::SymbolTable::Visibility::Private);
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
  assert(!ForVTable && "NYI");
  assert(!IsThunk && "NYI");

  const auto *D = GD.getDecl();

  // Any attempts to use a MultiVersion function should result in retrieving the
  // iFunc instead. Name mangling will handle the rest of the changes.
  if (const auto *FD = cast_or_null<FunctionDecl>(D)) {
    if (getLangOpts().OpenMP)
      llvm_unreachable("open MP NYI");
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
    llvm_unreachable("NYI");

    // TODO: clang checks here if this is a llvm::GlobalAlias... how will we
    // support this?
  }

  // This function doesn't have a complete type (for example, the return type is
  // an incomplete struct). Use a fake type instead, and make sure not to try to
  // set attributes.
  bool IsIncompleteFunction = false;

  mlir::FunctionType FTy;
  if (Ty.isa<mlir::FunctionType>()) {
    FTy = Ty.cast<mlir::FunctionType>();
  } else {
    assert(false && "NYI");
    // FTy = mlir::FunctionType::get(VoidTy, false);
    IsIncompleteFunction = true;
  }

  auto *FD = llvm::cast<FunctionDecl>(D);
  assert(FD && "Only FunctionDecl supported so far.");
  auto fnLoc = getLoc(FD->getSourceRange());
  // TODO: CodeGen includeds the linkage (ExternalLinkage) and only passes the
  // mangledname if Entry is nullptr
  auto F = createCIRFunction(fnLoc, MangledName, FTy);

  if (Entry) {
    llvm_unreachable("NYI");
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
    if (D && isa<CXXDestructorDecl>(D))
      llvm_unreachable("NYI");

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

  assert(false && "Incompmlete functions NYI");
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

  // Make sure getGlobalValue returned non-null.
  assert(Op);
  assert(isa<mlir::cir::FuncOp>(Op) &&
         "not implemented, only supports FuncOp for now");

  // Check to see if we've already emitted this. This is necessary for a
  // couple of reasons: first, decls can end up in deferred-decls queue
  // multiple times, and second, decls can end up with definitions in unusual
  // ways (e.g. by an extern inline function acquiring a strong function
  // redefinition). Just ignore those cases.
  // TODO: Not sure what to map this to for MLIR
  if (auto Fn = cast<mlir::cir::FuncOp>(Op))
    if (!Fn.isDeclaration())
      return;

  // If this is OpenMP, check if it is legal to emit this global normally.
  if (getLangOpts().OpenMP) {
    llvm_unreachable("NYI");
  }

  // Otherwise, emit the definition and move on to the next one.
  buildGlobalDefinition(D, Op);
}

void CIRGenModule::buildDeferred() {
  // Emit deferred declare target declarations
  if (getLangOpts().OpenMP && !getLangOpts().OpenMPSimd)
    llvm_unreachable("NYI");

  // Emit code for any potentially referenced deferred decls. Since a previously
  // unused static decl may become used during the generation of code for a
  // static function, iterate until no changes are made.

  if (!DeferredVTables.empty()) {
    llvm_unreachable("NYI");
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

  for (auto &D : CurDeclsToEmit) {
    buildGlobalDecl(D);

    // If we found out that we need to emit more decls, do that recursively.
    // This has the advantage that the decls are emitted in a DFS and related
    // ones are close together, which is convenient for testing.
    if (!DeferredVTables.empty() || !DeferredDeclsToEmit.empty()) {
      buildDeferred();
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

  llvm_unreachable("NYI");
}

void CIRGenModule::Release() {
  buildDeferred();
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
