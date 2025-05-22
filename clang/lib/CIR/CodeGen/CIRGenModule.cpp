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
#include "CIRGenConstantEmitter.h"
#include "CIRGenFunction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclOpenACC.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/SourceManager.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"

#include "CIRGenFunctionInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

using namespace clang;
using namespace clang::CIRGen;

static CIRGenCXXABI *createCXXABI(CIRGenModule &cgm) {
  switch (cgm.getASTContext().getCXXABIKind()) {
  case TargetCXXABI::GenericItanium:
  case TargetCXXABI::GenericAArch64:
  case TargetCXXABI::AppleARM64:
    return CreateCIRGenItaniumCXXABI(cgm);

  case TargetCXXABI::Fuchsia:
  case TargetCXXABI::GenericARM:
  case TargetCXXABI::iOS:
  case TargetCXXABI::WatchOS:
  case TargetCXXABI::GenericMIPS:
  case TargetCXXABI::WebAssembly:
  case TargetCXXABI::XL:
  case TargetCXXABI::Microsoft:
    cgm.errorNYI("C++ ABI kind not yet implemented");
    return nullptr;
  }

  llvm_unreachable("invalid C++ ABI kind");
}

namespace clang::CIRGen {
// TODO(cir): Implement target-specific CIRGenCXXABIs
CIRGenCXXABI *CreateCIRGenItaniumCXXABI(CIRGenModule &cgm) {
  assert(!cir::MissingFeatures::targetSpecificCXXABI());
  return new CIRGenCXXABI(cgm);
}
} // namespace clang::CIRGen

CIRGenModule::CIRGenModule(mlir::MLIRContext &mlirContext,
                           clang::ASTContext &astContext,
                           const clang::CodeGenOptions &cgo,
                           DiagnosticsEngine &diags)
    : builder(mlirContext, *this), astContext(astContext),
      langOpts(astContext.getLangOpts()), codeGenOpts(cgo),
      theModule{mlir::ModuleOp::create(mlir::UnknownLoc::get(&mlirContext))},
      diags(diags), target(astContext.getTargetInfo()),
      abi(createCXXABI(*this)), genTypes(*this) {

  // Initialize cached types
  VoidTy = cir::VoidType::get(&getMLIRContext());
  SInt8Ty = cir::IntType::get(&getMLIRContext(), 8, /*isSigned=*/true);
  SInt16Ty = cir::IntType::get(&getMLIRContext(), 16, /*isSigned=*/true);
  SInt32Ty = cir::IntType::get(&getMLIRContext(), 32, /*isSigned=*/true);
  SInt64Ty = cir::IntType::get(&getMLIRContext(), 64, /*isSigned=*/true);
  SInt128Ty = cir::IntType::get(&getMLIRContext(), 128, /*isSigned=*/true);
  UInt8Ty = cir::IntType::get(&getMLIRContext(), 8, /*isSigned=*/false);
  UInt16Ty = cir::IntType::get(&getMLIRContext(), 16, /*isSigned=*/false);
  UInt32Ty = cir::IntType::get(&getMLIRContext(), 32, /*isSigned=*/false);
  UInt64Ty = cir::IntType::get(&getMLIRContext(), 64, /*isSigned=*/false);
  UInt128Ty = cir::IntType::get(&getMLIRContext(), 128, /*isSigned=*/false);
  FP16Ty = cir::FP16Type::get(&getMLIRContext());
  BFloat16Ty = cir::BF16Type::get(&getMLIRContext());
  FloatTy = cir::SingleType::get(&getMLIRContext());
  DoubleTy = cir::DoubleType::get(&getMLIRContext());
  FP80Ty = cir::FP80Type::get(&getMLIRContext());
  FP128Ty = cir::FP128Type::get(&getMLIRContext());

  PointerAlignInBytes =
      astContext
          .toCharUnitsFromBits(
              astContext.getTargetInfo().getPointerAlign(LangAS::Default))
          .getQuantity();

  // TODO(CIR): Should be updated once TypeSizeInfoAttr is upstreamed
  const unsigned sizeTypeSize =
      astContext.getTypeSize(astContext.getSignedSizeType());
  PtrDiffTy =
      cir::IntType::get(&getMLIRContext(), sizeTypeSize, /*isSigned=*/true);

  theModule->setAttr(cir::CIRDialect::getTripleAttrName(),
                     builder.getStringAttr(getTriple().str()));
}

CIRGenModule::~CIRGenModule() = default;

CharUnits CIRGenModule::getNaturalTypeAlignment(QualType t,
                                                LValueBaseInfo *baseInfo) {
  assert(!cir::MissingFeatures::opTBAA());

  // FIXME: This duplicates logic in ASTContext::getTypeAlignIfKnown, but
  // that doesn't return the information we need to compute baseInfo.

  // Honor alignment typedef attributes even on incomplete types.
  // We also honor them straight for C++ class types, even as pointees;
  // there's an expressivity gap here.
  if (const auto *tt = t->getAs<TypedefType>()) {
    if (unsigned align = tt->getDecl()->getMaxAlignment()) {
      if (baseInfo)
        *baseInfo = LValueBaseInfo(AlignmentSource::AttributedType);
      return astContext.toCharUnitsFromBits(align);
    }
  }

  // Analyze the base element type, so we don't get confused by incomplete
  // array types.
  t = astContext.getBaseElementType(t);

  if (t->isIncompleteType()) {
    // We could try to replicate the logic from
    // ASTContext::getTypeAlignIfKnown, but nothing uses the alignment if the
    // type is incomplete, so it's impossible to test. We could try to reuse
    // getTypeAlignIfKnown, but that doesn't return the information we need
    // to set baseInfo.  So just ignore the possibility that the alignment is
    // greater than one.
    if (baseInfo)
      *baseInfo = LValueBaseInfo(AlignmentSource::Type);
    return CharUnits::One();
  }

  if (baseInfo)
    *baseInfo = LValueBaseInfo(AlignmentSource::Type);

  CharUnits alignment;
  if (t.getQualifiers().hasUnaligned()) {
    alignment = CharUnits::One();
  } else {
    assert(!cir::MissingFeatures::alignCXXRecordDecl());
    alignment = astContext.getTypeAlignInChars(t);
  }

  // Cap to the global maximum type alignment unless the alignment
  // was somehow explicit on the type.
  if (unsigned maxAlign = astContext.getLangOpts().MaxTypeAlign) {
    if (alignment.getQuantity() > maxAlign &&
        !astContext.isAlignmentRequired(t))
      alignment = CharUnits::fromQuantity(maxAlign);
  }
  return alignment;
}

const TargetCIRGenInfo &CIRGenModule::getTargetCIRGenInfo() {
  if (theTargetCIRGenInfo)
    return *theTargetCIRGenInfo;

  const llvm::Triple &triple = getTarget().getTriple();
  switch (triple.getArch()) {
  default:
    assert(!cir::MissingFeatures::targetCIRGenInfoArch());

    // Currently we just fall through to x86_64.
    [[fallthrough]];

  case llvm::Triple::x86_64: {
    switch (triple.getOS()) {
    default:
      assert(!cir::MissingFeatures::targetCIRGenInfoOS());

      // Currently we just fall through to x86_64.
      [[fallthrough]];

    case llvm::Triple::Linux:
      theTargetCIRGenInfo = createX8664TargetCIRGenInfo(genTypes);
      return *theTargetCIRGenInfo;
    }
  }
  }
}

mlir::Location CIRGenModule::getLoc(SourceLocation cLoc) {
  assert(cLoc.isValid() && "expected valid source location");
  const SourceManager &sm = astContext.getSourceManager();
  PresumedLoc pLoc = sm.getPresumedLoc(cLoc);
  StringRef filename = pLoc.getFilename();
  return mlir::FileLineColLoc::get(builder.getStringAttr(filename),
                                   pLoc.getLine(), pLoc.getColumn());
}

mlir::Location CIRGenModule::getLoc(SourceRange cRange) {
  assert(cRange.isValid() && "expected a valid source range");
  mlir::Location begin = getLoc(cRange.getBegin());
  mlir::Location end = getLoc(cRange.getEnd());
  mlir::Attribute metadata;
  return mlir::FusedLoc::get({begin, end}, metadata, builder.getContext());
}

void CIRGenModule::emitGlobal(clang::GlobalDecl gd) {
  if (const auto *cd = dyn_cast<clang::OpenACCConstructDecl>(gd.getDecl())) {
    emitGlobalOpenACCDecl(cd);
    return;
  }

  const auto *global = cast<ValueDecl>(gd.getDecl());

  if (const auto *fd = dyn_cast<FunctionDecl>(global)) {
    // Update deferred annotations with the latest declaration if the function
    // was already used or defined.
    if (fd->hasAttr<AnnotateAttr>())
      errorNYI(fd->getSourceRange(), "deferredAnnotations");
    if (!fd->doesThisDeclarationHaveABody()) {
      if (!fd->doesDeclarationForceExternallyVisibleDefinition())
        return;

      errorNYI(fd->getSourceRange(),
               "function declaration that forces code gen");
      return;
    }
  } else {
    assert(cast<VarDecl>(global)->isFileVarDecl() &&
           "Cannot emit local var decl as global");
  }

  // TODO(CIR): Defer emitting some global definitions until later
  emitGlobalDefinition(gd);
}

void CIRGenModule::emitGlobalFunctionDefinition(clang::GlobalDecl gd,
                                                mlir::Operation *op) {
  auto const *funcDecl = cast<FunctionDecl>(gd.getDecl());
  if (funcDecl->getIdentifier() == nullptr) {
    errorNYI(funcDecl->getSourceRange().getBegin(),
             "function definition with a non-identifier for a name");
    return;
  }

  const CIRGenFunctionInfo &fi = getTypes().arrangeGlobalDeclaration(gd);
  cir::FuncType funcType = getTypes().getFunctionType(fi);
  cir::FuncOp funcOp = dyn_cast_if_present<cir::FuncOp>(op);
  if (!funcOp || funcOp.getFunctionType() != funcType) {
    funcOp = getAddrOfFunction(gd, funcType, /*ForVTable=*/false,
                               /*DontDefer=*/true, ForDefinition);
  }

  CIRGenFunction cgf(*this, builder);
  curCGF = &cgf;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    cgf.generateCode(gd, funcOp, funcType);
  }
  curCGF = nullptr;
}

mlir::Operation *CIRGenModule::getGlobalValue(StringRef name) {
  return mlir::SymbolTable::lookupSymbolIn(theModule, name);
}

cir::GlobalOp CIRGenModule::createGlobalOp(CIRGenModule &cgm,
                                           mlir::Location loc, StringRef name,
                                           mlir::Type t,
                                           mlir::Operation *insertPoint) {
  cir::GlobalOp g;
  CIRGenBuilderTy &builder = cgm.getBuilder();

  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Some global emissions are triggered while emitting a function, e.g.
    // void s() { const char *s = "yolo"; ... }
    //
    // Be sure to insert global before the current function
    CIRGenFunction *curCGF = cgm.curCGF;
    if (curCGF)
      builder.setInsertionPoint(curCGF->curFn);

    g = builder.create<cir::GlobalOp>(loc, name, t);
    if (!curCGF) {
      if (insertPoint)
        cgm.getModule().insert(insertPoint, g);
      else
        cgm.getModule().push_back(g);
    }

    // Default to private until we can judge based on the initializer,
    // since MLIR doesn't allow public declarations.
    mlir::SymbolTable::setSymbolVisibility(
        g, mlir::SymbolTable::Visibility::Private);
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
/// 2. If \p d is non-null, it specifies a decl that correspond to this.  This
/// is used to set the attributes on the global when it is first created.
///
/// 3. If \p isForDefinition is true, it is guaranteed that an actual global
/// with type \p ty will be returned, not conversion of a variable with the same
/// mangled name but some other type.
cir::GlobalOp
CIRGenModule::getOrCreateCIRGlobal(StringRef mangledName, mlir::Type ty,
                                   LangAS langAS, const VarDecl *d,
                                   ForDefinition_t isForDefinition) {
  // Lookup the entry, lazily creating it if necessary.
  cir::GlobalOp entry;
  if (mlir::Operation *v = getGlobalValue(mangledName)) {
    if (!isa<cir::GlobalOp>(v))
      errorNYI(d->getSourceRange(), "global with non-GlobalOp type");
    entry = cast<cir::GlobalOp>(v);
  }

  if (entry) {
    assert(!cir::MissingFeatures::addressSpace());
    assert(!cir::MissingFeatures::opGlobalWeakRef());

    assert(!cir::MissingFeatures::setDLLStorageClass());
    assert(!cir::MissingFeatures::openMP());

    if (entry.getSymType() == ty)
      return entry;

    // If there are two attempts to define the same mangled name, issue an
    // error.
    //
    // TODO(cir): look at mlir::GlobalValue::isDeclaration for all aspects of
    // recognizing the global as a declaration, for now only check if
    // initializer is present.
    if (isForDefinition && !entry.isDeclaration()) {
      errorNYI(d->getSourceRange(), "global with conflicting type");
    }

    // Address space check removed because it is unnecessary because CIR records
    // address space info in types.

    // (If global is requested for a definition, we always need to create a new
    // global, not just return a bitcast.)
    if (!isForDefinition)
      return entry;
  }

  mlir::Location loc = getLoc(d->getSourceRange());

  // mlir::SymbolTable::Visibility::Public is the default, no need to explicitly
  // mark it as such.
  cir::GlobalOp gv =
      CIRGenModule::createGlobalOp(*this, loc, mangledName, ty,
                                   /*insertPoint=*/entry.getOperation());

  return gv;
}

cir::GlobalOp
CIRGenModule::getOrCreateCIRGlobal(const VarDecl *d, mlir::Type ty,
                                   ForDefinition_t isForDefinition) {
  assert(d->hasGlobalStorage() && "Not a global variable");
  QualType astTy = d->getType();
  if (!ty)
    ty = getTypes().convertTypeForMem(astTy);

  StringRef mangledName = getMangledName(d);
  return getOrCreateCIRGlobal(mangledName, ty, astTy.getAddressSpace(), d,
                              isForDefinition);
}

/// Return the mlir::Value for the address of the given global variable. If
/// \p ty is non-null and if the global doesn't exist, then it will be created
/// with the specified type instead of whatever the normal requested type would
/// be. If \p isForDefinition is true, it is guaranteed that an actual global
/// with type \p ty will be returned, not conversion of a variable with the same
/// mangled name but some other type.
mlir::Value CIRGenModule::getAddrOfGlobalVar(const VarDecl *d, mlir::Type ty,
                                             ForDefinition_t isForDefinition) {
  assert(d->hasGlobalStorage() && "Not a global variable");
  QualType astTy = d->getType();
  if (!ty)
    ty = getTypes().convertTypeForMem(astTy);

  assert(!cir::MissingFeatures::opGlobalThreadLocal());

  cir::GlobalOp g = getOrCreateCIRGlobal(d, ty, isForDefinition);
  mlir::Type ptrTy = builder.getPointerTo(g.getSymType());
  return builder.create<cir::GetGlobalOp>(getLoc(d->getSourceRange()), ptrTy,
                                          g.getSymName());
}

void CIRGenModule::emitGlobalVarDefinition(const clang::VarDecl *vd,
                                           bool isTentative) {
  const QualType astTy = vd->getType();

  if (getLangOpts().OpenCL || getLangOpts().OpenMPIsTargetDevice) {
    errorNYI(vd->getSourceRange(), "emit OpenCL/OpenMP global variable");
    return;
  }

  // Whether the definition of the variable is available externally.
  // If yes, we shouldn't emit the GloablCtor and GlobalDtor for the variable
  // since this is the job for its original source.
  bool isDefinitionAvailableExternally =
      astContext.GetGVALinkageForVariable(vd) == GVA_AvailableExternally;
  assert(!cir::MissingFeatures::needsGlobalCtorDtor());

  // It is useless to emit the definition for an available_externally variable
  // which can't be marked as const.
  if (isDefinitionAvailableExternally &&
      (!vd->hasConstantInitialization() ||
       // TODO: Update this when we have interface to check constexpr
       // destructor.
       vd->needsDestruction(astContext) ||
       !vd->getType().isConstantStorage(astContext, true, true)))
    return;

  mlir::Attribute init;
  const VarDecl *initDecl;
  const Expr *initExpr = vd->getAnyInitializer(initDecl);

  std::optional<ConstantEmitter> emitter;

  assert(!cir::MissingFeatures::cudaSupport());

  if (vd->hasAttr<LoaderUninitializedAttr>()) {
    errorNYI(vd->getSourceRange(), "loader uninitialized attribute");
    return;
  } else if (!initExpr) {
    // This is a tentative definition; tentative definitions are
    // implicitly initialized with { 0 }.
    //
    // Note that tentative definitions are only emitted at the end of
    // a translation unit, so they should never have incomplete
    // type. In addition, EmitTentativeDefinition makes sure that we
    // never attempt to emit a tentative definition if a real one
    // exists. A use may still exists, however, so we still may need
    // to do a RAUW.
    assert(!astTy->isIncompleteType() && "Unexpected incomplete type");
    init = builder.getZeroInitAttr(convertType(vd->getType()));
  } else {
    emitter.emplace(*this);
    mlir::Attribute initializer = emitter->tryEmitForInitializer(*initDecl);
    if (!initializer) {
      QualType qt = initExpr->getType();
      if (vd->getType()->isReferenceType())
        qt = vd->getType();

      if (getLangOpts().CPlusPlus) {
        if (initDecl->hasFlexibleArrayInit(astContext))
          errorNYI(vd->getSourceRange(), "flexible array initializer");
        init = builder.getZeroInitAttr(convertType(qt));
        if (astContext.GetGVALinkageForVariable(vd) != GVA_AvailableExternally)
          errorNYI(vd->getSourceRange(), "global constructor");
      } else {
        errorNYI(vd->getSourceRange(), "static initializer");
      }
    } else {
      init = initializer;
      // We don't need an initializer, so remove the entry for the delayed
      // initializer position (just in case this entry was delayed) if we
      // also don't need to register a destructor.
      if (vd->needsDestruction(astContext) == QualType::DK_cxx_destructor)
        errorNYI(vd->getSourceRange(), "delayed destructor");
    }
  }

  mlir::Type initType;
  if (mlir::isa<mlir::SymbolRefAttr>(init)) {
    errorNYI(vd->getSourceRange(), "global initializer is a symbol reference");
    return;
  } else {
    assert(mlir::isa<mlir::TypedAttr>(init) && "This should have a type");
    auto typedInitAttr = mlir::cast<mlir::TypedAttr>(init);
    initType = typedInitAttr.getType();
  }
  assert(!mlir::isa<mlir::NoneType>(initType) && "Should have a type by now");

  cir::GlobalOp gv =
      getOrCreateCIRGlobal(vd, initType, ForDefinition_t(!isTentative));
  // TODO(cir): Strip off pointer casts from Entry if we get them?

  if (!gv || gv.getSymType() != initType) {
    errorNYI(vd->getSourceRange(), "global initializer with type mismatch");
    return;
  }

  assert(!cir::MissingFeatures::maybeHandleStaticInExternC());

  if (vd->hasAttr<AnnotateAttr>()) {
    errorNYI(vd->getSourceRange(), "annotate global variable");
  }

  assert(!cir::MissingFeatures::opGlobalLinkage());

  if (langOpts.CUDA) {
    errorNYI(vd->getSourceRange(), "CUDA global variable");
  }

  // Set initializer and finalize emission
  CIRGenModule::setInitializer(gv, init);
  if (emitter)
    emitter->finalize(gv);

  // Set CIR's linkage type as appropriate.
  cir::GlobalLinkageKind linkage =
      getCIRLinkageVarDefinition(vd, /*IsConstant=*/false);

  // Set CIR linkage and DLL storage class.
  gv.setLinkage(linkage);

  if (linkage == cir::GlobalLinkageKind::CommonLinkage)
    errorNYI(initExpr->getSourceRange(), "common linkage");
}

void CIRGenModule::emitGlobalDefinition(clang::GlobalDecl gd,
                                        mlir::Operation *op) {
  const auto *decl = cast<ValueDecl>(gd.getDecl());
  if (const auto *fd = dyn_cast<FunctionDecl>(decl)) {
    // TODO(CIR): Skip generation of CIR for functions with available_externally
    // linkage at -O0.

    if (const auto *method = dyn_cast<CXXMethodDecl>(decl)) {
      // Make sure to emit the definition(s) before we emit the thunks. This is
      // necessary for the generation of certain thunks.
      if (isa<CXXConstructorDecl>(method) || isa<CXXDestructorDecl>(method))
        errorNYI(method->getSourceRange(), "C++ ctor/dtor");
      else if (fd->isMultiVersion())
        errorNYI(method->getSourceRange(), "multiversion functions");
      else
        emitGlobalFunctionDefinition(gd, op);

      if (method->isVirtual())
        errorNYI(method->getSourceRange(), "virtual member function");

      return;
    }

    if (fd->isMultiVersion())
      errorNYI(fd->getSourceRange(), "multiversion functions");
    emitGlobalFunctionDefinition(gd, op);
    return;
  }

  if (const auto *vd = dyn_cast<VarDecl>(decl))
    return emitGlobalVarDefinition(vd, !vd->hasDefinition());

  llvm_unreachable("Invalid argument to CIRGenModule::emitGlobalDefinition");
}

mlir::Attribute
CIRGenModule::getConstantArrayFromStringLiteral(const StringLiteral *e) {
  assert(!e->getType()->isPointerType() && "Strings are always arrays");

  // Don't emit it as the address of the string, emit the string data itself
  // as an inline array.
  if (e->getCharByteWidth() == 1) {
    SmallString<64> str(e->getString());

    // Resize the string to the right size, which is indicated by its type.
    const ConstantArrayType *cat =
        astContext.getAsConstantArrayType(e->getType());
    uint64_t finalSize = cat->getZExtSize();
    str.resize(finalSize);

    mlir::Type eltTy = convertType(cat->getElementType());
    return builder.getString(str, eltTy, finalSize);
  }

  errorNYI(e->getSourceRange(),
           "getConstantArrayFromStringLiteral: wide characters");
  return mlir::Attribute();
}

static bool shouldBeInCOMDAT(CIRGenModule &cgm, const Decl &d) {
  assert(!cir::MissingFeatures::supportComdat());

  if (d.hasAttr<SelectAnyAttr>())
    return true;

  GVALinkage linkage;
  if (auto *vd = dyn_cast<VarDecl>(&d))
    linkage = cgm.getASTContext().GetGVALinkageForVariable(vd);
  else
    linkage =
        cgm.getASTContext().GetGVALinkageForFunction(cast<FunctionDecl>(&d));

  switch (linkage) {
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

// TODO(CIR): this could be a common method between LLVM codegen.
static bool isVarDeclStrongDefinition(const ASTContext &astContext,
                                      CIRGenModule &cgm, const VarDecl *vd,
                                      bool noCommon) {
  // Don't give variables common linkage if -fno-common was specified unless it
  // was overridden by a NoCommon attribute.
  if ((noCommon || vd->hasAttr<NoCommonAttr>()) && !vd->hasAttr<CommonAttr>())
    return true;

  // C11 6.9.2/2:
  //   A declaration of an identifier for an object that has file scope without
  //   an initializer, and without a storage-class specifier or with the
  //   storage-class specifier static, constitutes a tentative definition.
  if (vd->getInit() || vd->hasExternalStorage())
    return true;

  // A variable cannot be both common and exist in a section.
  if (vd->hasAttr<SectionAttr>())
    return true;

  // A variable cannot be both common and exist in a section.
  // We don't try to determine which is the right section in the front-end.
  // If no specialized section name is applicable, it will resort to default.
  if (vd->hasAttr<PragmaClangBSSSectionAttr>() ||
      vd->hasAttr<PragmaClangDataSectionAttr>() ||
      vd->hasAttr<PragmaClangRelroSectionAttr>() ||
      vd->hasAttr<PragmaClangRodataSectionAttr>())
    return true;

  // Thread local vars aren't considered common linkage.
  if (vd->getTLSKind())
    return true;

  // Tentative definitions marked with WeakImportAttr are true definitions.
  if (vd->hasAttr<WeakImportAttr>())
    return true;

  // A variable cannot be both common and exist in a comdat.
  if (shouldBeInCOMDAT(cgm, *vd))
    return true;

  // Declarations with a required alignment do not have common linkage in MSVC
  // mode.
  if (astContext.getTargetInfo().getCXXABI().isMicrosoft()) {
    if (vd->hasAttr<AlignedAttr>())
      return true;
    QualType varType = vd->getType();
    if (astContext.isAlignmentRequired(varType))
      return true;

    if (const auto *rt = varType->getAs<RecordType>()) {
      const RecordDecl *rd = rt->getDecl();
      for (const FieldDecl *fd : rd->fields()) {
        if (fd->isBitField())
          continue;
        if (fd->hasAttr<AlignedAttr>())
          return true;
        if (astContext.isAlignmentRequired(fd->getType()))
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
  if (astContext.getTargetInfo().getTriple().isKnownWindowsMSVCEnvironment() &&
      astContext.getTypeAlignIfKnown(vd->getType()) >
          astContext.toBits(CharUnits::fromQuantity(32)))
    return true;

  return false;
}

cir::GlobalLinkageKind CIRGenModule::getCIRLinkageForDeclarator(
    const DeclaratorDecl *dd, GVALinkage linkage, bool isConstantVariable) {
  if (linkage == GVA_Internal)
    return cir::GlobalLinkageKind::InternalLinkage;

  if (dd->hasAttr<WeakAttr>()) {
    if (isConstantVariable)
      return cir::GlobalLinkageKind::WeakODRLinkage;
    return cir::GlobalLinkageKind::WeakAnyLinkage;
  }

  if (const auto *fd = dd->getAsFunction())
    if (fd->isMultiVersion() && linkage == GVA_AvailableExternally)
      return cir::GlobalLinkageKind::LinkOnceAnyLinkage;

  // We are guaranteed to have a strong definition somewhere else,
  // so we can use available_externally linkage.
  if (linkage == GVA_AvailableExternally)
    return cir::GlobalLinkageKind::AvailableExternallyLinkage;

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
  if (linkage == GVA_DiscardableODR)
    return !astContext.getLangOpts().AppleKext
               ? cir::GlobalLinkageKind::LinkOnceODRLinkage
               : cir::GlobalLinkageKind::InternalLinkage;

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
  if (linkage == GVA_StrongODR) {
    if (getLangOpts().AppleKext)
      return cir::GlobalLinkageKind::ExternalLinkage;
    if (getLangOpts().CUDA && getLangOpts().CUDAIsDevice &&
        !getLangOpts().GPURelocatableDeviceCode)
      return dd->hasAttr<CUDAGlobalAttr>()
                 ? cir::GlobalLinkageKind::ExternalLinkage
                 : cir::GlobalLinkageKind::InternalLinkage;
    return cir::GlobalLinkageKind::WeakODRLinkage;
  }

  // C++ doesn't have tentative definitions and thus cannot have common
  // linkage.
  if (!getLangOpts().CPlusPlus && isa<VarDecl>(dd) &&
      !isVarDeclStrongDefinition(astContext, *this, cast<VarDecl>(dd),
                                 getCodeGenOpts().NoCommon)) {
    errorNYI(dd->getBeginLoc(), "common linkage", dd->getDeclKindName());
    return cir::GlobalLinkageKind::CommonLinkage;
  }

  // selectany symbols are externally visible, so use weak instead of
  // linkonce.  MSVC optimizes away references to const selectany globals, so
  // all definitions should be the same and ODR linkage should be used.
  // http://msdn.microsoft.com/en-us/library/5tkz6s71.aspx
  if (dd->hasAttr<SelectAnyAttr>())
    return cir::GlobalLinkageKind::WeakODRLinkage;

  // Otherwise, we have strong external linkage.
  assert(linkage == GVA_StrongExternal);
  return cir::GlobalLinkageKind::ExternalLinkage;
}

cir::GlobalLinkageKind
CIRGenModule::getCIRLinkageVarDefinition(const VarDecl *vd, bool isConstant) {
  assert(!isConstant && "constant variables NYI");
  GVALinkage linkage = astContext.GetGVALinkageForVariable(vd);
  return getCIRLinkageForDeclarator(vd, linkage, isConstant);
}

static cir::GlobalOp generateStringLiteral(mlir::Location loc,
                                           mlir::TypedAttr c, CIRGenModule &cgm,
                                           StringRef globalName) {
  assert(!cir::MissingFeatures::addressSpace());

  // Create a global variable for this string
  // FIXME(cir): check for insertion point in module level.
  cir::GlobalOp gv =
      CIRGenModule::createGlobalOp(cgm, loc, globalName, c.getType());

  // Set up extra information and add to the module
  assert(!cir::MissingFeatures::opGlobalAlignment());
  assert(!cir::MissingFeatures::opGlobalLinkage());
  assert(!cir::MissingFeatures::opGlobalThreadLocal());
  assert(!cir::MissingFeatures::opGlobalUnnamedAddr());
  CIRGenModule::setInitializer(gv, c);
  assert(!cir::MissingFeatures::supportComdat());
  assert(!cir::MissingFeatures::opGlobalDSOLocal());
  return gv;
}

// LLVM IR automatically uniques names when new llvm::GlobalVariables are
// created. This is handy, for example, when creating globals for string
// literals. Since we don't do that when creating cir::GlobalOp's, we need
// a mechanism to generate a unique name in advance.
//
// For now, this mechanism is only used in cases where we know that the
// name is compiler-generated, so we don't use the MLIR symbol table for
// the lookup.
std::string CIRGenModule::getUniqueGlobalName(const std::string &baseName) {
  // If this is the first time we've generated a name for this basename, use
  // it as is and start a counter for this base name.
  auto it = cgGlobalNames.find(baseName);
  if (it == cgGlobalNames.end()) {
    cgGlobalNames[baseName] = 1;
    return baseName;
  }

  std::string result =
      baseName + "." + std::to_string(cgGlobalNames[baseName]++);
  // There should not be any symbol with this name in the module.
  assert(!mlir::SymbolTable::lookupSymbolIn(theModule, result));
  return result;
}

/// Return a pointer to a constant array for the given string literal.
cir::GlobalOp CIRGenModule::getGlobalForStringLiteral(const StringLiteral *s,
                                                      StringRef name) {
  mlir::Attribute c = getConstantArrayFromStringLiteral(s);

  if (getLangOpts().WritableStrings) {
    errorNYI(s->getSourceRange(),
             "getGlobalForStringLiteral: Writable strings");
  }

  // Mangle the string literal if that's how the ABI merges duplicate strings.
  // Don't do it if they are writable, since we don't want writes in one TU to
  // affect strings in another.
  if (getCXXABI().getMangleContext().shouldMangleStringLiteral(s) &&
      !getLangOpts().WritableStrings) {
    errorNYI(s->getSourceRange(),
             "getGlobalForStringLiteral: mangle string literals");
  }

  // Unlike LLVM IR, CIR doesn't automatically unique names for globals, so
  // we need to do that explicitly.
  std::string uniqueName = getUniqueGlobalName(name.str());
  mlir::Location loc = getLoc(s->getSourceRange());
  auto typedC = llvm::cast<mlir::TypedAttr>(c);
  assert(!cir::MissingFeatures::opGlobalAlignment());
  cir::GlobalOp gv = generateStringLiteral(loc, typedC, *this, uniqueName);
  assert(!cir::MissingFeatures::opGlobalDSOLocal());

  assert(!cir::MissingFeatures::sanitizers());

  return gv;
}

void CIRGenModule::emitDeclContext(const DeclContext *dc) {
  for (Decl *decl : dc->decls()) {
    // Unlike other DeclContexts, the contents of an ObjCImplDecl at TU scope
    // are themselves considered "top-level", so EmitTopLevelDecl on an
    // ObjCImplDecl does not recursively visit them. We need to do that in
    // case they're nested inside another construct (LinkageSpecDecl /
    // ExportDecl) that does stop them from being considered "top-level".
    if (auto *oid = dyn_cast<ObjCImplDecl>(decl))
      errorNYI(oid->getSourceRange(), "emitDeclConext: ObjCImplDecl");

    emitTopLevelDecl(decl);
  }
}

// Emit code for a single top level declaration.
void CIRGenModule::emitTopLevelDecl(Decl *decl) {

  // Ignore dependent declarations.
  if (decl->isTemplated())
    return;

  switch (decl->getKind()) {
  default:
    errorNYI(decl->getBeginLoc(), "declaration of kind",
             decl->getDeclKindName());
    break;

  case Decl::CXXMethod:
  case Decl::Function: {
    auto *fd = cast<FunctionDecl>(decl);
    // Consteval functions shouldn't be emitted.
    if (!fd->isConsteval())
      emitGlobal(fd);
    break;
  }

  case Decl::Var: {
    auto *vd = cast<VarDecl>(decl);
    emitGlobal(vd);
    break;
  }
  case Decl::OpenACCRoutine:
    emitGlobalOpenACCDecl(cast<OpenACCRoutineDecl>(decl));
    break;
  case Decl::OpenACCDeclare:
    emitGlobalOpenACCDecl(cast<OpenACCDeclareDecl>(decl));
    break;
  case Decl::Enum:
  case Decl::UsingDirective: // using namespace X; [C++]
  case Decl::Typedef:
  case Decl::TypeAlias: // using foo = bar; [C++11]
  case Decl::Record:
  case Decl::CXXRecord:
    assert(!cir::MissingFeatures::generateDebugInfo());
    break;

  // C++ Decls
  case Decl::LinkageSpec:
  case Decl::Namespace:
    emitDeclContext(Decl::castToDeclContext(decl));
    break;
  }
}

void CIRGenModule::setInitializer(cir::GlobalOp &op, mlir::Attribute value) {
  // Recompute visibility when updating initializer.
  op.setInitialValueAttr(value);
  assert(!cir::MissingFeatures::opGlobalSetVisitibility());
}

cir::FuncOp CIRGenModule::getAddrOfFunction(clang::GlobalDecl gd,
                                            mlir::Type funcType, bool forVTable,
                                            bool dontDefer,
                                            ForDefinition_t isForDefinition) {
  assert(!cast<FunctionDecl>(gd.getDecl())->isConsteval() &&
         "consteval function should never be emitted");

  if (!funcType) {
    const auto *fd = cast<FunctionDecl>(gd.getDecl());
    funcType = convertType(fd->getType());
  }

  StringRef mangledName = getMangledName(gd);
  cir::FuncOp func =
      getOrCreateCIRFunction(mangledName, funcType, gd, forVTable, dontDefer,
                             /*isThunk=*/false, isForDefinition);
  return func;
}

static std::string getMangledNameImpl(CIRGenModule &cgm, GlobalDecl gd,
                                      const NamedDecl *nd) {
  SmallString<256> buffer;

  llvm::raw_svector_ostream out(buffer);
  MangleContext &mc = cgm.getCXXABI().getMangleContext();

  assert(!cir::MissingFeatures::moduleNameHash());

  if (mc.shouldMangleDeclName(nd)) {
    mc.mangleName(gd.getWithDecl(nd), out);
  } else {
    IdentifierInfo *ii = nd->getIdentifier();
    assert(ii && "Attempt to mangle unnamed decl.");

    const auto *fd = dyn_cast<FunctionDecl>(nd);
    if (fd &&
        fd->getType()->castAs<FunctionType>()->getCallConv() == CC_X86RegCall) {
      cgm.errorNYI(nd->getSourceRange(), "getMangledName: X86RegCall");
    } else if (fd && fd->hasAttr<CUDAGlobalAttr>() &&
               gd.getKernelReferenceKind() == KernelReferenceKind::Stub) {
      cgm.errorNYI(nd->getSourceRange(), "getMangledName: CUDA device stub");
    }
    out << ii->getName();
  }

  // Check if the module name hash should be appended for internal linkage
  // symbols. This should come before multi-version target suffixes are
  // appendded. This is to keep the name and module hash suffix of the internal
  // linkage function together. The unique suffix should only be added when name
  // mangling is done to make sure that the final name can be properly
  // demangled. For example, for C functions without prototypes, name mangling
  // is not done and the unique suffix should not be appended then.
  assert(!cir::MissingFeatures::moduleNameHash());

  if (const auto *fd = dyn_cast<FunctionDecl>(nd)) {
    if (fd->isMultiVersion()) {
      cgm.errorNYI(nd->getSourceRange(),
                   "getMangledName: multi-version functions");
    }
  }
  if (cgm.getLangOpts().GPURelocatableDeviceCode) {
    cgm.errorNYI(nd->getSourceRange(),
                 "getMangledName: GPU relocatable device code");
  }

  return std::string(out.str());
}

StringRef CIRGenModule::getMangledName(GlobalDecl gd) {
  GlobalDecl canonicalGd = gd.getCanonicalDecl();

  // Some ABIs don't have constructor variants. Make sure that base and complete
  // constructors get mangled the same.
  if (const auto *cd = dyn_cast<CXXConstructorDecl>(canonicalGd.getDecl())) {
    errorNYI(cd->getSourceRange(), "getMangledName: C++ constructor");
    return cast<NamedDecl>(gd.getDecl())->getIdentifier()->getName();
  }

  // Keep the first result in the case of a mangling collision.
  const auto *nd = cast<NamedDecl>(gd.getDecl());
  std::string mangledName = getMangledNameImpl(*this, gd, nd);

  auto result = manglings.insert(std::make_pair(mangledName, gd));
  return mangledDeclNames[canonicalGd] = result.first->first();
}

cir::FuncOp CIRGenModule::getOrCreateCIRFunction(
    StringRef mangledName, mlir::Type funcType, GlobalDecl gd, bool forVTable,
    bool dontDefer, bool isThunk, ForDefinition_t isForDefinition,
    mlir::ArrayAttr extraAttrs) {
  const Decl *d = gd.getDecl();

  if (isThunk)
    errorNYI(d->getSourceRange(), "getOrCreateCIRFunction: thunk");

  // In what follows, we continue past 'errorNYI' as if nothing happened because
  // the rest of the implementation is better than doing nothing.

  if (const auto *fd = cast_or_null<FunctionDecl>(d)) {
    // For the device mark the function as one that should be emitted.
    if (getLangOpts().OpenMPIsTargetDevice && fd->isDefined() && !dontDefer &&
        !isForDefinition)
      errorNYI(fd->getSourceRange(),
               "getOrCreateCIRFunction: OpenMP target function");

    // Any attempts to use a MultiVersion function should result in retrieving
    // the iFunc instead. Name mangling will handle the rest of the changes.
    if (fd->isMultiVersion())
      errorNYI(fd->getSourceRange(), "getOrCreateCIRFunction: multi-version");
  }

  // Lookup the entry, lazily creating it if necessary.
  mlir::Operation *entry = getGlobalValue(mangledName);
  if (entry) {
    if (!isa<cir::FuncOp>(entry))
      errorNYI(d->getSourceRange(), "getOrCreateCIRFunction: non-FuncOp");

    assert(!cir::MissingFeatures::weakRefReference());

    // Handle dropped DLL attributes.
    if (d && !d->hasAttr<DLLImportAttr>() && !d->hasAttr<DLLExportAttr>()) {
      assert(!cir::MissingFeatures::setDLLStorageClass());
      assert(!cir::MissingFeatures::setDSOLocal());
    }

    // If there are two attempts to define the same mangled name, issue an
    // error.
    auto fn = cast<cir::FuncOp>(entry);
    assert((!isForDefinition || !fn || !fn.isDeclaration()) &&
           "Duplicate function definition");
    if (fn && fn.getFunctionType() == funcType) {
      return fn;
    }

    if (!isForDefinition) {
      return fn;
    }

    // TODO(cir): classic codegen checks here if this is a llvm::GlobalAlias.
    // How will we support this?
  }

  auto *funcDecl = llvm::cast_or_null<FunctionDecl>(gd.getDecl());
  bool invalidLoc = !funcDecl ||
                    funcDecl->getSourceRange().getBegin().isInvalid() ||
                    funcDecl->getSourceRange().getEnd().isInvalid();
  cir::FuncOp funcOp = createCIRFunction(
      invalidLoc ? theModule->getLoc() : getLoc(funcDecl->getSourceRange()),
      mangledName, mlir::cast<cir::FuncType>(funcType), funcDecl);
  return funcOp;
}

cir::FuncOp
CIRGenModule::createCIRFunction(mlir::Location loc, StringRef name,
                                cir::FuncType funcType,
                                const clang::FunctionDecl *funcDecl) {
  cir::FuncOp func;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Some global emissions are triggered while emitting a function, e.g.
    // void s() { x.method() }
    //
    // Be sure to insert a new function before a current one.
    CIRGenFunction *cgf = this->curCGF;
    if (cgf)
      builder.setInsertionPoint(cgf->curFn);

    func = builder.create<cir::FuncOp>(loc, name, funcType);

    if (!cgf)
      theModule.push_back(func);
  }
  return func;
}

mlir::Type CIRGenModule::convertType(QualType type) {
  return genTypes.convertType(type);
}

bool CIRGenModule::verifyModule() const {
  // Verify the module after we have finished constructing it, this will
  // check the structural properties of the IR and invoke any specific
  // verifiers we have on the CIR operations.
  return mlir::verify(theModule).succeeded();
}

DiagnosticBuilder CIRGenModule::errorNYI(SourceLocation loc,
                                         llvm::StringRef feature) {
  unsigned diagID = diags.getCustomDiagID(
      DiagnosticsEngine::Error, "ClangIR code gen Not Yet Implemented: %0");
  return diags.Report(loc, diagID) << feature;
}

DiagnosticBuilder CIRGenModule::errorNYI(SourceRange loc,
                                         llvm::StringRef feature) {
  return errorNYI(loc.getBegin(), feature) << loc;
}
