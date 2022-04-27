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
    llvm_unreachable("NYI");
  }

  llvm::StringRef MangledName = getMangledName(GD);
  if (GetGlobalValue(MangledName) != nullptr) {
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
  // TODO: Figure out what to do here? llvm uses a GlobalValue for the FuncOp in
  // mlir
  Op = GetAddrOfFunction(GD, Ty, /*ForVTable=*/false, /*DontDefer=*/true,
                         ForDefinition);

  auto Fn = cast<mlir::FuncOp>(Op);
  // Already emitted.
  if (!Fn.isDeclaration())
    return;

  // TODO: setFunctionLinkage
  // TODO: setGVProperties
  // TODO: MaubeHandleStaticInExternC
  // TODO: maybeSetTrivialComdat
  // TODO: setLLVMFunctionFEnvAttributes

  CIRGenFunction CGF{*this, builder};
  CurCGF = &CGF;
  CGF.generateCode(GD, Fn, FI);
  CurCGF = nullptr;

  // TODO: setNonAliasAttributes
  // TODO: SetLLVMFunctionAttributesForDeclaration

  assert(!D->getAttr<ConstructorAttr>() && "NYI");
  assert(!D->getAttr<DestructorAttr>() && "NYI");
  assert(!D->getAttr<AnnotateAttr>() && "NYI");
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
    llvm_unreachable("NYI");

  llvm_unreachable("Invalid argument to buildGlobalDefinition()");
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

std::pair<mlir::FunctionType, mlir::FuncOp>
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

mlir::FuncOp CIRGenModule::GetAddrOfFunction(clang::GlobalDecl GD,
                                             mlir::Type Ty, bool ForVTable,
                                             bool DontDefer,
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
    assert(FD && "Only FunctionDecl supported");
    assert(FD->getType()->castAs<FunctionType>()->getCallConv() !=
               CC_X86RegCall &&
           "NYI");
    assert(!FD->hasAttr<CUDAGlobalAttr>() && "NYI");

    Out << II->getName();
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
  if (const auto *FD = cast_or_null<FunctionDecl>(D)) {
    if (getLangOpts().OpenMP)
      llvm_unreachable("open MP NYI");
    if (FD->isMultiVersion())
      llvm_unreachable("NYI");
  }

  // Lookup the entry, lazily creating it if necessary.
  mlir::Operation *Entry = GetGlobalValue(MangledName);
  if (Entry) {
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
    auto Fn = cast<mlir::FuncOp>(Entry);
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
  mlir::FuncOp F = mlir::FuncOp::create(fnLoc, MangledName, FTy);
  theModule.push_back(F);

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

mlir::Value CIRGenModule::GetGlobalValue(const Decl *D) {
  assert(CurCGF);
  return CurCGF->symbolTable.lookup(D);
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
    // We should call GetAddrOfGlobal with IsForDefinition set to true in order
    // to get a Value with exactly the type we need, not something that might
    // have been created for another decl with the same mangled name but
    // different type.
    auto *Op = GetAddrOfGlobal(D, ForDefinition);

    // In case of different address spaces, we may still get a cast, even with
    // IsForDefinition equal to true. Query mangled names table to get
    // GlobalValue.
    if (!Op) {
      Op = GetGlobalValue(getMangledName(D));
    }

    // Make sure GetGlobalValue returned non-null.
    assert(Op);

    // Check to see if we've already emitted this. This is necessary for a
    // couple of reasons: first, decls can end up in deferred-decls queue
    // multiple times, and second, decls can end up with definitions in unusual
    // ways (e.g. by an extern inline function acquiring a strong function
    // redefinition). Just ignore those cases.
    // TODO: Not sure what to map this to for MLIR
    if (auto Fn = cast<mlir::FuncOp>(Op))
      if (!Fn.isDeclaration())
        continue;

    // If this is OpenMP, check if it is legal to emit this global normally.
    if (getLangOpts().OpenMP) {
      llvm_unreachable("NYI");
    }

    // Otherwise, emit the definition and move on to the next one.
    buildGlobalDefinition(D, Op);

    // If we found out that we need to emit more decls, do that recursively.
    // This has the advantage that the decls are emitted in a DFS and related
    // ones are close together, which is convenient for testing.
    if (!DeferredVTables.empty() || !DeferredDeclsToEmit.empty()) {
      buildDeferred();
      assert(DeferredVTables.empty() && DeferredDeclsToEmit.empty());
    }
  }
}

// TODO: this is gross, make a map
mlir::Operation *CIRGenModule::GetGlobalValue(StringRef Name) {
  for (auto const &op :
       theModule.getBodyRegion().front().getOps<mlir::FuncOp>())
    if (auto Fn = llvm::cast<mlir::FuncOp>(op)) {
      if (Name == Fn.getName())
        return Fn;
    }

  return nullptr;
}

mlir::Operation *
CIRGenModule::GetAddrOfGlobal(GlobalDecl GD, ForDefinition_t IsForDefinition) {
  const Decl *D = GD.getDecl();

  if (isa<CXXConstructorDecl>(D) || isa<CXXDestructorDecl>(D))
    return getAddrOfCXXStructor(GD, /*FnInfo=*/nullptr, /*FnType=*/nullptr,
                                /*DontDefer=*/false, IsForDefinition);

  llvm_unreachable("NYI");
}

void CIRGenModule::Release() {
  buildDeferred();
  // TODO: buildVTablesOpportunistically();
  // TODO: applyGlobalValReplacements();
  // TODO: applyReplacements();
  // TODO: checkAliases();
  // TODO: buildMultiVersionFunctions();
  // TODO: buildCXXGlobalInitFunc();
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

void CIRGenModule::maybeSetTrivialComdat(const Decl &D, mlir::Operation *Op) {
  if (!shouldBeInCOMDAT(*this, D))
    return;

  // TODO: Op.setComdat
}

bool CIRGenModule::isInNoSanitizeList(SanitizerMask Kind, mlir::FuncOp Fn,
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
