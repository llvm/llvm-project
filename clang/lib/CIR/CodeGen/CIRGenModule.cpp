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
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/SourceManager.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
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

CIRGenModule::CIRGenModule(mlir::MLIRContext &mlirContext,
                           clang::ASTContext &astContext,
                           const clang::CodeGenOptions &cgo,
                           DiagnosticsEngine &diags)
    : builder(mlirContext, *this), astContext(astContext),
      langOpts(astContext.getLangOpts()), codeGenOpts(cgo),
      theModule{mlir::ModuleOp::create(mlir::UnknownLoc::get(&mlirContext))},
      diags(diags), target(astContext.getTargetInfo()),
      abi(createCXXABI(*this)), genTypes(*this), vtables(*this) {

  // Initialize cached types
  VoidTy = cir::VoidType::get(&getMLIRContext());
  VoidPtrTy = cir::PointerType::get(VoidTy);
  SInt8Ty = cir::IntType::get(&getMLIRContext(), 8, /*isSigned=*/true);
  SInt16Ty = cir::IntType::get(&getMLIRContext(), 16, /*isSigned=*/true);
  SInt32Ty = cir::IntType::get(&getMLIRContext(), 32, /*isSigned=*/true);
  SInt64Ty = cir::IntType::get(&getMLIRContext(), 64, /*isSigned=*/true);
  SInt128Ty = cir::IntType::get(&getMLIRContext(), 128, /*isSigned=*/true);
  UInt8Ty = cir::IntType::get(&getMLIRContext(), 8, /*isSigned=*/false);
  UInt8PtrTy = cir::PointerType::get(UInt8Ty);
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

  const unsigned charSize = astContext.getTargetInfo().getCharWidth();
  UCharTy = cir::IntType::get(&getMLIRContext(), charSize, /*isSigned=*/false);

  // TODO(CIR): Should be updated once TypeSizeInfoAttr is upstreamed
  const unsigned sizeTypeSize =
      astContext.getTypeSize(astContext.getSignedSizeType());
  SizeAlignInBytes = astContext.toCharUnitsFromBits(sizeTypeSize).getQuantity();
  // In CIRGenTypeCache, UIntPtrTy and SizeType are fields of the same union
  UIntPtrTy =
      cir::IntType::get(&getMLIRContext(), sizeTypeSize, /*isSigned=*/false);
  PtrDiffTy =
      cir::IntType::get(&getMLIRContext(), sizeTypeSize, /*isSigned=*/true);

  std::optional<cir::SourceLanguage> sourceLanguage = getCIRSourceLanguage();
  if (sourceLanguage)
    theModule->setAttr(
        cir::CIRDialect::getSourceLanguageAttrName(),
        cir::SourceLanguageAttr::get(&mlirContext, *sourceLanguage));
  theModule->setAttr(cir::CIRDialect::getTripleAttrName(),
                     builder.getStringAttr(getTriple().str()));

  if (cgo.OptimizationLevel > 0 || cgo.OptimizeSize > 0)
    theModule->setAttr(cir::CIRDialect::getOptInfoAttrName(),
                       cir::OptInfoAttr::get(&mlirContext,
                                             cgo.OptimizationLevel,
                                             cgo.OptimizeSize));
}

CIRGenModule::~CIRGenModule() = default;

/// FIXME: this could likely be a common helper and not necessarily related
/// with codegen.
/// Return the best known alignment for an unknown pointer to a
/// particular class.
CharUnits CIRGenModule::getClassPointerAlignment(const CXXRecordDecl *rd) {
  if (!rd->hasDefinition())
    return CharUnits::One(); // Hopefully won't be used anywhere.

  auto &layout = astContext.getASTRecordLayout(rd);

  // If the class is final, then we know that the pointer points to an
  // object of that type and can use the full alignment.
  if (rd->isEffectivelyFinal())
    return layout.getAlignment();

  // Otherwise, we have to assume it could be a subclass.
  return layout.getNonVirtualAlignment();
}

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

mlir::Operation *
CIRGenModule::getAddrOfGlobal(GlobalDecl gd, ForDefinition_t isForDefinition) {
  const Decl *d = gd.getDecl();

  if (isa<CXXConstructorDecl>(d) || isa<CXXDestructorDecl>(d))
    return getAddrOfCXXStructor(gd, /*FnInfo=*/nullptr, /*FnType=*/nullptr,
                                /*DontDefer=*/false, isForDefinition);

  if (isa<CXXMethodDecl>(d)) {
    const CIRGenFunctionInfo &fi =
        getTypes().arrangeCXXMethodDeclaration(cast<CXXMethodDecl>(d));
    cir::FuncType ty = getTypes().getFunctionType(fi);
    return getAddrOfFunction(gd, ty, /*ForVTable=*/false, /*DontDefer=*/false,
                             isForDefinition);
  }

  if (isa<FunctionDecl>(d)) {
    const CIRGenFunctionInfo &fi = getTypes().arrangeGlobalDeclaration(gd);
    cir::FuncType ty = getTypes().getFunctionType(fi);
    return getAddrOfFunction(gd, ty, /*ForVTable=*/false, /*DontDefer=*/false,
                             isForDefinition);
  }

  return getAddrOfGlobalVar(cast<VarDecl>(d), /*ty=*/nullptr, isForDefinition)
      .getDefiningOp();
}

void CIRGenModule::emitGlobalDecl(const clang::GlobalDecl &d) {
  // We call getAddrOfGlobal with isForDefinition set to ForDefinition in
  // order to get a Value with exactly the type we need, not something that
  // might have been created for another decl with the same mangled name but
  // different type.
  mlir::Operation *op = getAddrOfGlobal(d, ForDefinition);

  // In case of different address spaces, we may still get a cast, even with
  // IsForDefinition equal to ForDefinition. Query mangled names table to get
  // GlobalValue.
  if (!op)
    op = getGlobalValue(getMangledName(d));

  assert(op && "expected a valid global op");

  // Check to see if we've already emitted this. This is necessary for a
  // couple of reasons: first, decls can end up in deferred-decls queue
  // multiple times, and second, decls can end up with definitions in unusual
  // ways (e.g. by an extern inline function acquiring a strong function
  // redefinition). Just ignore those cases.
  // TODO: Not sure what to map this to for MLIR
  mlir::Operation *globalValueOp = op;
  if (auto gv = dyn_cast<cir::GetGlobalOp>(op))
    globalValueOp =
        mlir::SymbolTable::lookupSymbolIn(getModule(), gv.getNameAttr());

  if (auto cirGlobalValue =
          dyn_cast<cir::CIRGlobalValueInterface>(globalValueOp))
    if (!cirGlobalValue.isDeclaration())
      return;

  // If this is OpenMP, check if it is legal to emit this global normally.
  assert(!cir::MissingFeatures::openMP());

  // Otherwise, emit the definition and move on to the next one.
  emitGlobalDefinition(d, op);
}

void CIRGenModule::emitDeferred() {
  // Emit code for any potentially referenced deferred decls. Since a previously
  // unused static decl may become used during the generation of code for a
  // static function, iterate until no changes are made.

  assert(!cir::MissingFeatures::openMP());
  assert(!cir::MissingFeatures::deferredVtables());
  assert(!cir::MissingFeatures::cudaSupport());

  // Stop if we're out of both deferred vtables and deferred declarations.
  if (deferredDeclsToEmit.empty())
    return;

  // Grab the list of decls to emit. If emitGlobalDefinition schedules more
  // work, it will not interfere with this.
  std::vector<GlobalDecl> curDeclsToEmit;
  curDeclsToEmit.swap(deferredDeclsToEmit);

  for (const GlobalDecl &d : curDeclsToEmit) {
    emitGlobalDecl(d);

    // If we found out that we need to emit more decls, do that recursively.
    // This has the advantage that the decls are emitted in a DFS and related
    // ones are close together, which is convenient for testing.
    if (!deferredDeclsToEmit.empty()) {
      emitDeferred();
      assert(deferredDeclsToEmit.empty());
    }
  }
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
    const auto *vd = cast<VarDecl>(global);
    assert(vd->isFileVarDecl() && "Cannot emit local var decl as global.");
    if (vd->isThisDeclarationADefinition() != VarDecl::Definition &&
        !astContext.isMSStaticDataMemberInlineDefinition(vd)) {
      assert(!cir::MissingFeatures::openMP());
      // If this declaration may have caused an inline variable definition to
      // change linkage, make sure that it's emitted.
      if (astContext.getInlineVariableDefinitionKind(vd) ==
          ASTContext::InlineVariableDefinitionKind::Strong)
        getAddrOfGlobalVar(vd);
      // Otherwise, we can ignore this declaration. The variable will be emitted
      // on its first use.
      return;
    }
  }

  // Defer code generation to first use when possible, e.g. if this is an inline
  // function. If the global must always be emitted, do it eagerly if possible
  // to benefit from cache locality. Deferring code generation is necessary to
  // avoid adding initializers to external declarations.
  if (mustBeEmitted(global) && mayBeEmittedEagerly(global)) {
    // Emit the definition if it can't be deferred.
    emitGlobalDefinition(gd);
    return;
  }

  // If we're deferring emission of a C++ variable with an initializer, remember
  // the order in which it appeared on the file.
  assert(!cir::MissingFeatures::deferredCXXGlobalInit());

  llvm::StringRef mangledName = getMangledName(gd);
  if (getGlobalValue(mangledName) != nullptr) {
    // The value has already been used and should therefore be emitted.
    addDeferredDeclToEmit(gd);
  } else if (mustBeEmitted(global)) {
    // The value must be emitted, but cannot be emitted eagerly.
    assert(!mayBeEmittedEagerly(global));
    addDeferredDeclToEmit(gd);
  } else {
    // Otherwise, remember that we saw a deferred decl with this name. The first
    // use of the mangled name will cause it to move into deferredDeclsToEmit.
    deferredDecls[mangledName] = gd;
  }
}

void CIRGenModule::emitGlobalFunctionDefinition(clang::GlobalDecl gd,
                                                mlir::Operation *op) {
  auto const *funcDecl = cast<FunctionDecl>(gd.getDecl());
  const CIRGenFunctionInfo &fi = getTypes().arrangeGlobalDeclaration(gd);
  cir::FuncType funcType = getTypes().getFunctionType(fi);
  cir::FuncOp funcOp = dyn_cast_if_present<cir::FuncOp>(op);
  if (!funcOp || funcOp.getFunctionType() != funcType) {
    funcOp = getAddrOfFunction(gd, funcType, /*ForVTable=*/false,
                               /*DontDefer=*/true, ForDefinition);
  }

  // Already emitted.
  if (!funcOp.isDeclaration())
    return;

  setFunctionLinkage(gd, funcOp);
  setGVProperties(funcOp, funcDecl);
  assert(!cir::MissingFeatures::opFuncMaybeHandleStaticInExternC());
  maybeSetTrivialComdat(*funcDecl, funcOp);
  assert(!cir::MissingFeatures::setLLVMFunctionFEnvAttributes());

  CIRGenFunction cgf(*this, builder);
  curCGF = &cgf;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    cgf.generateCode(gd, funcOp, funcType);
  }
  curCGF = nullptr;

  setNonAliasAttributes(gd, funcOp);
  assert(!cir::MissingFeatures::opFuncAttributesForDefinition());

  if (funcDecl->getAttr<ConstructorAttr>())
    errorNYI(funcDecl->getSourceRange(), "constructor attribute");
  if (funcDecl->getAttr<DestructorAttr>())
    errorNYI(funcDecl->getSourceRange(), "destructor attribute");

  if (funcDecl->getAttr<AnnotateAttr>())
    errorNYI(funcDecl->getSourceRange(), "deferredAnnotations");
}

void CIRGenModule::handleCXXStaticMemberVarInstantiation(VarDecl *vd) {
  VarDecl::DefinitionKind dk = vd->isThisDeclarationADefinition();
  if (dk == VarDecl::Definition && vd->hasAttr<DLLImportAttr>())
    return;

  TemplateSpecializationKind tsk = vd->getTemplateSpecializationKind();
  // If we have a definition, this might be a deferred decl. If the
  // instantiation is explicit, make sure we emit it at the end.
  if (vd->getDefinition() && tsk == TSK_ExplicitInstantiationDefinition)
    getAddrOfGlobalVar(vd);

  emitTopLevelDecl(vd);
}

mlir::Operation *CIRGenModule::getGlobalValue(StringRef name) {
  return mlir::SymbolTable::lookupSymbolIn(theModule, name);
}

cir::GlobalOp CIRGenModule::createGlobalOp(CIRGenModule &cgm,
                                           mlir::Location loc, StringRef name,
                                           mlir::Type t, bool isConstant,
                                           mlir::Operation *insertPoint) {
  cir::GlobalOp g;
  CIRGenBuilderTy &builder = cgm.getBuilder();

  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // If an insertion point is provided, we're replacing an existing global,
    // otherwise, create the new global immediately after the last gloabl we
    // emitted.
    if (insertPoint) {
      builder.setInsertionPoint(insertPoint);
    } else {
      // Group global operations together at the top of the module.
      if (cgm.lastGlobalOp)
        builder.setInsertionPointAfter(cgm.lastGlobalOp);
      else
        builder.setInsertionPointToStart(cgm.getModule().getBody());
    }

    g = builder.create<cir::GlobalOp>(loc, name, t, isConstant);
    if (!insertPoint)
      cgm.lastGlobalOp = g;

    // Default to private until we can judge based on the initializer,
    // since MLIR doesn't allow public declarations.
    mlir::SymbolTable::setSymbolVisibility(
        g, mlir::SymbolTable::Visibility::Private);
  }
  return g;
}

void CIRGenModule::setCommonAttributes(GlobalDecl gd, mlir::Operation *gv) {
  const Decl *d = gd.getDecl();
  if (isa_and_nonnull<NamedDecl>(d))
    setGVProperties(gv, dyn_cast<NamedDecl>(d));
  assert(!cir::MissingFeatures::defaultVisibility());
  assert(!cir::MissingFeatures::opGlobalUsedOrCompilerUsed());
}

void CIRGenModule::setNonAliasAttributes(GlobalDecl gd, mlir::Operation *op) {
  setCommonAttributes(gd, op);

  assert(!cir::MissingFeatures::opGlobalUsedOrCompilerUsed());
  assert(!cir::MissingFeatures::opGlobalSection());
  assert(!cir::MissingFeatures::opFuncCPUAndFeaturesAttributes());
  assert(!cir::MissingFeatures::opFuncSection());

  assert(!cir::MissingFeatures::setTargetAttributes());
}

std::optional<cir::SourceLanguage> CIRGenModule::getCIRSourceLanguage() const {
  using ClangStd = clang::LangStandard;
  using CIRLang = cir::SourceLanguage;
  auto opts = getLangOpts();

  if (opts.CPlusPlus)
    return CIRLang::CXX;
  if (opts.C99 || opts.C11 || opts.C17 || opts.C23 || opts.C2y ||
      opts.LangStd == ClangStd::lang_c89 ||
      opts.LangStd == ClangStd::lang_gnu89)
    return CIRLang::C;

  // TODO(cir): support remaining source languages.
  assert(!cir::MissingFeatures::sourceLanguageCases());
  errorNYI("CIR does not yet support the given source language");
  return std::nullopt;
}

static void setLinkageForGV(cir::GlobalOp &gv, const NamedDecl *nd) {
  // Set linkage and visibility in case we never see a definition.
  LinkageInfo lv = nd->getLinkageAndVisibility();
  // Don't set internal linkage on declarations.
  // "extern_weak" is overloaded in LLVM; we probably should have
  // separate linkage types for this.
  if (isExternallyVisible(lv.getLinkage()) &&
      (nd->hasAttr<WeakAttr>() || nd->isWeakImported()))
    gv.setLinkage(cir::GlobalLinkageKind::ExternalWeakLinkage);
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
      CIRGenModule::createGlobalOp(*this, loc, mangledName, ty, false,
                                   /*insertPoint=*/entry.getOperation());

  // This is the first use or definition of a mangled name.  If there is a
  // deferred decl with this name, remember that we need to emit it at the end
  // of the file.
  auto ddi = deferredDecls.find(mangledName);
  if (ddi != deferredDecls.end()) {
    // Move the potentially referenced deferred decl to the DeferredDeclsToEmit
    // list, and remove it from DeferredDecls (since we don't need it anymore).
    addDeferredDeclToEmit(ddi->second);
    deferredDecls.erase(ddi);
  }

  // Handle things which are present even on external declarations.
  if (d) {
    if (langOpts.OpenMP && !langOpts.OpenMPSimd)
      errorNYI(d->getSourceRange(), "OpenMP target global variable");

    gv.setAlignmentAttr(getSize(astContext.getDeclAlign(d)));
    assert(!cir::MissingFeatures::opGlobalConstant());

    setLinkageForGV(gv, d);

    if (d->getTLSKind())
      errorNYI(d->getSourceRange(), "thread local global variable");

    setGVProperties(gv, d);

    // If required by the ABI, treat declarations of static data members with
    // inline initializers as definitions.
    if (astContext.isMSStaticDataMemberInlineDefinition(d))
      errorNYI(d->getSourceRange(), "MS static data member inline definition");

    assert(!cir::MissingFeatures::opGlobalSection());
    gv.setGlobalVisibilityAttr(getGlobalVisibilityAttrFromDecl(d));

    // Handle XCore specific ABI requirements.
    if (getTriple().getArch() == llvm::Triple::xcore)
      errorNYI(d->getSourceRange(), "XCore specific ABI requirements");

    // Check if we a have a const declaration with an initializer, we may be
    // able to emit it as available_externally to expose it's value to the
    // optimizer.
    if (getLangOpts().CPlusPlus && gv.isPublic() &&
        d->getType().isConstQualified() && gv.isDeclaration() &&
        !d->hasDefinition() && d->hasInit() && !d->hasAttr<DLLImportAttr>())
      errorNYI(d->getSourceRange(),
               "external const declaration with initializer");
  }

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

cir::GlobalViewAttr CIRGenModule::getAddrOfGlobalVarAttr(const VarDecl *d) {
  assert(d->hasGlobalStorage() && "Not a global variable");
  mlir::Type ty = getTypes().convertTypeForMem(d->getType());

  cir::GlobalOp globalOp = getOrCreateCIRGlobal(d, ty, NotForDefinition);
  assert(!cir::MissingFeatures::addressSpace());
  cir::PointerType ptrTy = builder.getPointerTo(globalOp.getSymType());
  return builder.getGlobalViewAttr(ptrTy, globalOp);
}

void CIRGenModule::emitGlobalVarDefinition(const clang::VarDecl *vd,
                                           bool isTentative) {
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
    assert(!vd->getType()->isIncompleteType() && "Unexpected incomplete type");
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
  // FIXME(cir): setLinkage should likely set MLIR's visibility automatically.
  gv.setVisibility(getMLIRVisibilityFromCIRLinkage(linkage));
  assert(!cir::MissingFeatures::opGlobalDLLImportExport());
  if (linkage == cir::GlobalLinkageKind::CommonLinkage)
    errorNYI(initExpr->getSourceRange(), "common linkage");

  setNonAliasAttributes(vd, gv);

  assert(!cir::MissingFeatures::opGlobalThreadLocal());

  maybeSetTrivialComdat(*vd, gv);
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
        abi->emitCXXStructor(gd);
      else if (fd->isMultiVersion())
        errorNYI(method->getSourceRange(), "multiversion functions");
      else
        emitGlobalFunctionDefinition(gd, op);

      if (method->isVirtual())
        getVTables().emitThunks(gd);

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

bool CIRGenModule::supportsCOMDAT() const {
  return getTriple().supportsCOMDAT();
}

static bool shouldBeInCOMDAT(CIRGenModule &cgm, const Decl &d) {
  if (!cgm.supportsCOMDAT())
    return false;

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

void CIRGenModule::maybeSetTrivialComdat(const Decl &d, mlir::Operation *op) {
  if (!shouldBeInCOMDAT(*this, d))
    return;
  if (auto globalOp = dyn_cast_or_null<cir::GlobalOp>(op)) {
    globalOp.setComdat(true);
  } else {
    auto funcOp = cast<cir::FuncOp>(op);
    funcOp.setComdat(true);
  }
}

void CIRGenModule::updateCompletedType(const TagDecl *td) {
  // Make sure that this type is translated.
  genTypes.updateCompletedType(td);
}

void CIRGenModule::addReplacement(StringRef name, mlir::Operation *op) {
  replacements[name] = op;
}

void CIRGenModule::replacePointerTypeArgs(cir::FuncOp oldF, cir::FuncOp newF) {
  std::optional<mlir::SymbolTable::UseRange> optionalUseRange =
      oldF.getSymbolUses(theModule);
  if (!optionalUseRange)
    return;

  for (const mlir::SymbolTable::SymbolUse &u : *optionalUseRange) {
    // CallTryOp only shows up after FlattenCFG.
    auto call = mlir::dyn_cast<cir::CallOp>(u.getUser());
    if (!call)
      continue;

    for (const auto [argOp, fnArgType] :
         llvm::zip(call.getArgs(), newF.getFunctionType().getInputs())) {
      if (argOp.getType() == fnArgType)
        continue;

      // The purpose of this entire function is to insert bitcasts in the case
      // where these types don't match, but I haven't seen a case where that
      // happens.
      errorNYI(call.getLoc(), "replace call with mismatched types");
    }
  }
}

void CIRGenModule::applyReplacements() {
  for (auto &i : replacements) {
    StringRef mangledName = i.first();
    mlir::Operation *replacement = i.second;
    mlir::Operation *entry = getGlobalValue(mangledName);
    if (!entry)
      continue;
    assert(isa<cir::FuncOp>(entry) && "expected function");
    auto oldF = cast<cir::FuncOp>(entry);
    auto newF = dyn_cast<cir::FuncOp>(replacement);
    if (!newF) {
      // In classic codegen, this can be a global alias, a bitcast, or a GEP.
      errorNYI(replacement->getLoc(), "replacement is not a function");
      continue;
    }

    // LLVM has opaque pointer but CIR not. So we may have to handle these
    // different pointer types when performing replacement.
    replacePointerTypeArgs(oldF, newF);

    // Replace old with new, but keep the old order.
    if (oldF.replaceAllSymbolUses(newF.getSymNameAttr(), theModule).failed())
      llvm_unreachable("internal error, cannot RAUW symbol");
    if (newF) {
      newF->moveBefore(oldF);
      oldF->erase();
    }
  }
}

cir::GlobalOp CIRGenModule::createOrReplaceCXXRuntimeVariable(
    mlir::Location loc, StringRef name, mlir::Type ty,
    cir::GlobalLinkageKind linkage, clang::CharUnits alignment) {
  auto gv = mlir::dyn_cast_or_null<cir::GlobalOp>(
      mlir::SymbolTable::lookupSymbolIn(theModule, name));

  if (gv) {
    // Check if the variable has the right type.
    if (gv.getSymType() == ty)
      return gv;

    // Because of C++ name mangling, the only way we can end up with an already
    // existing global with the same name is if it has been declared extern
    // "C".
    assert(gv.isDeclaration() && "Declaration has wrong type!");

    errorNYI(loc, "createOrReplaceCXXRuntimeVariable: declaration exists with "
                  "wrong type");
    return gv;
  }

  // Create a new variable.
  gv = createGlobalOp(*this, loc, name, ty);

  // Set up extra information and add to the module
  gv.setLinkageAttr(
      cir::GlobalLinkageKindAttr::get(&getMLIRContext(), linkage));
  mlir::SymbolTable::setSymbolVisibility(gv,
                                         CIRGenModule::getMLIRVisibility(gv));

  if (supportsCOMDAT() && cir::isWeakForLinker(linkage) &&
      !gv.hasAvailableExternallyLinkage()) {
    gv.setComdat(true);
  }

  gv.setAlignmentAttr(getSize(alignment));
  setDSOLocal(static_cast<mlir::Operation *>(gv));
  return gv;
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

    if (const auto *rd = varType->getAsRecordDecl()) {
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

/// This function is called when we implement a function with no prototype, e.g.
/// "int foo() {}". If there are existing call uses of the old function in the
/// module, this adjusts them to call the new function directly.
///
/// This is not just a cleanup: the always_inline pass requires direct calls to
/// functions to be able to inline them.  If there is a bitcast in the way, it
/// won't inline them. Instcombine normally deletes these calls, but it isn't
/// run at -O0.
void CIRGenModule::replaceUsesOfNonProtoTypeWithRealFunction(
    mlir::Operation *old, cir::FuncOp newFn) {
  // If we're redefining a global as a function, don't transform it.
  auto oldFn = mlir::dyn_cast<cir::FuncOp>(old);
  if (!oldFn)
    return;

  // TODO(cir): this RAUW ignores the features below.
  assert(!cir::MissingFeatures::opFuncExceptions());
  assert(!cir::MissingFeatures::opFuncParameterAttributes());
  assert(!cir::MissingFeatures::opFuncOperandBundles());
  if (oldFn->getAttrs().size() <= 1)
    errorNYI(old->getLoc(),
             "replaceUsesOfNonProtoTypeWithRealFunction: Attribute forwarding");

  // Mark new function as originated from a no-proto declaration.
  newFn.setNoProto(oldFn.getNoProto());

  // Iterate through all calls of the no-proto function.
  std::optional<mlir::SymbolTable::UseRange> symUses =
      oldFn.getSymbolUses(oldFn->getParentOp());
  for (const mlir::SymbolTable::SymbolUse &use : symUses.value()) {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto noProtoCallOp = mlir::dyn_cast<cir::CallOp>(use.getUser())) {
      builder.setInsertionPoint(noProtoCallOp);

      // Patch call type with the real function type.
      cir::CallOp realCallOp = builder.createCallOp(
          noProtoCallOp.getLoc(), newFn, noProtoCallOp.getOperands());

      // Replace old no proto call with fixed call.
      noProtoCallOp.replaceAllUsesWith(realCallOp);
      noProtoCallOp.erase();
    } else if (auto getGlobalOp =
                   mlir::dyn_cast<cir::GetGlobalOp>(use.getUser())) {
      // Replace type
      getGlobalOp.getAddr().setType(
          cir::PointerType::get(newFn.getFunctionType()));
    } else {
      errorNYI(use.getUser()->getLoc(),
               "replaceUsesOfNonProtoTypeWithRealFunction: unexpected use");
    }
  }
}

cir::GlobalLinkageKind
CIRGenModule::getCIRLinkageVarDefinition(const VarDecl *vd, bool isConstant) {
  assert(!isConstant && "constant variables NYI");
  GVALinkage linkage = astContext.GetGVALinkageForVariable(vd);
  return getCIRLinkageForDeclarator(vd, linkage, isConstant);
}

cir::GlobalLinkageKind CIRGenModule::getFunctionLinkage(GlobalDecl gd) {
  const auto *d = cast<FunctionDecl>(gd.getDecl());

  GVALinkage linkage = astContext.GetGVALinkageForFunction(d);

  if (const auto *dtor = dyn_cast<CXXDestructorDecl>(d))
    return getCXXABI().getCXXDestructorLinkage(linkage, dtor, gd.getDtorType());

  return getCIRLinkageForDeclarator(d, linkage, /*isConstantVariable=*/false);
}

static cir::GlobalOp
generateStringLiteral(mlir::Location loc, mlir::TypedAttr c,
                      cir::GlobalLinkageKind lt, CIRGenModule &cgm,
                      StringRef globalName, CharUnits alignment) {
  assert(!cir::MissingFeatures::addressSpace());

  // Create a global variable for this string
  // FIXME(cir): check for insertion point in module level.
  cir::GlobalOp gv = CIRGenModule::createGlobalOp(
      cgm, loc, globalName, c.getType(), !cgm.getLangOpts().WritableStrings);

  // Set up extra information and add to the module
  gv.setAlignmentAttr(cgm.getSize(alignment));
  gv.setLinkageAttr(
      cir::GlobalLinkageKindAttr::get(cgm.getBuilder().getContext(), lt));
  assert(!cir::MissingFeatures::opGlobalThreadLocal());
  assert(!cir::MissingFeatures::opGlobalUnnamedAddr());
  CIRGenModule::setInitializer(gv, c);
  if (gv.isWeakForLinker()) {
    assert(cgm.supportsCOMDAT() && "Only COFF uses weak string literals");
    gv.setComdat(true);
  }
  cgm.setDSOLocal(static_cast<mlir::Operation *>(gv));
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
  CharUnits alignment =
      astContext.getAlignOfGlobalVarInChars(s->getType(), /*VD=*/nullptr);

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
  cir::GlobalOp gv =
      generateStringLiteral(loc, typedC, cir::GlobalLinkageKind::PrivateLinkage,
                            *this, uniqueName, alignment);
  setDSOLocal(static_cast<mlir::Operation *>(gv));

  assert(!cir::MissingFeatures::sanitizers());

  return gv;
}

/// Return a pointer to a constant array for the given string literal.
cir::GlobalViewAttr
CIRGenModule::getAddrOfConstantStringFromLiteral(const StringLiteral *s,
                                                 StringRef name) {
  cir::GlobalOp gv = getGlobalForStringLiteral(s, name);
  auto arrayTy = mlir::dyn_cast<cir::ArrayType>(gv.getSymType());
  assert(arrayTy && "String literal must be array");
  assert(!cir::MissingFeatures::addressSpace());
  cir::PointerType ptrTy = getBuilder().getPointerTo(arrayTy.getElementType());

  return builder.getGlobalViewAttr(ptrTy, gv);
}

void CIRGenModule::emitExplicitCastExprType(const ExplicitCastExpr *e,
                                            CIRGenFunction *cgf) {
  if (cgf && e->getType()->isVariablyModifiedType())
    cgf->emitVariablyModifiedType(e->getType());

  assert(!cir::MissingFeatures::generateDebugInfo() &&
         "emitExplicitCastExprType");
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

  case Decl::CXXConversion:
  case Decl::CXXMethod:
  case Decl::Function: {
    auto *fd = cast<FunctionDecl>(decl);
    // Consteval functions shouldn't be emitted.
    if (!fd->isConsteval())
      emitGlobal(fd);
    break;
  }

  case Decl::Var:
  case Decl::Decomposition:
  case Decl::VarTemplateSpecialization: {
    auto *vd = cast<VarDecl>(decl);
    if (isa<DecompositionDecl>(decl)) {
      errorNYI(decl->getSourceRange(), "global variable decompositions");
      break;
    }
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
  case Decl::Using:          // using X; [C++]
  case Decl::UsingDirective: // using namespace X; [C++]
  case Decl::UsingEnum:      // using enum X; [C++]
  case Decl::NamespaceAlias:
  case Decl::Typedef:
  case Decl::TypeAlias: // using foo = bar; [C++11]
  case Decl::Record:
    assert(!cir::MissingFeatures::generateDebugInfo());
    break;

  // No code generation needed.
  case Decl::ClassTemplate:
  case Decl::Concept:
  case Decl::CXXDeductionGuide:
  case Decl::Empty:
  case Decl::FunctionTemplate:
  case Decl::StaticAssert:
  case Decl::TypeAliasTemplate:
  case Decl::UsingShadow:
  case Decl::VarTemplate:
  case Decl::VarTemplatePartialSpecialization:
    break;

  case Decl::CXXConstructor:
    getCXXABI().emitCXXConstructors(cast<CXXConstructorDecl>(decl));
    break;
  case Decl::CXXDestructor:
    getCXXABI().emitCXXDestructors(cast<CXXDestructorDecl>(decl));
    break;

  // C++ Decls
  case Decl::LinkageSpec:
  case Decl::Namespace:
    emitDeclContext(Decl::castToDeclContext(decl));
    break;

  case Decl::ClassTemplateSpecialization:
  case Decl::CXXRecord:
    assert(!cir::MissingFeatures::generateDebugInfo());
    assert(!cir::MissingFeatures::cxxRecordStaticMembers());
    break;

  case Decl::FileScopeAsm:
    // File-scope asm is ignored during device-side CUDA compilation.
    if (langOpts.CUDA && langOpts.CUDAIsDevice)
      break;
    // File-scope asm is ignored during device-side OpenMP compilation.
    if (langOpts.OpenMPIsTargetDevice)
      break;
    // File-scope asm is ignored during device-side SYCL compilation.
    if (langOpts.SYCLIsDevice)
      break;
    auto *file_asm = cast<FileScopeAsmDecl>(decl);
    std::string line = file_asm->getAsmString();
    globalScopeAsm.push_back(builder.getStringAttr(line));
    break;
  }
}

void CIRGenModule::setInitializer(cir::GlobalOp &op, mlir::Attribute value) {
  // Recompute visibility when updating initializer.
  op.setInitialValueAttr(value);
  assert(!cir::MissingFeatures::opGlobalVisibility());
}

std::pair<cir::FuncType, cir::FuncOp> CIRGenModule::getAddrAndTypeOfCXXStructor(
    GlobalDecl gd, const CIRGenFunctionInfo *fnInfo, cir::FuncType fnType,
    bool dontDefer, ForDefinition_t isForDefinition) {
  auto *md = cast<CXXMethodDecl>(gd.getDecl());

  if (isa<CXXDestructorDecl>(md)) {
    // Always alias equivalent complete destructors to base destructors in the
    // MS ABI.
    if (getTarget().getCXXABI().isMicrosoft() &&
        gd.getDtorType() == Dtor_Complete &&
        md->getParent()->getNumVBases() == 0)
      errorNYI(md->getSourceRange(),
               "getAddrAndTypeOfCXXStructor: MS ABI complete destructor");
  }

  if (!fnType) {
    if (!fnInfo)
      fnInfo = &getTypes().arrangeCXXStructorDeclaration(gd);
    fnType = getTypes().getFunctionType(*fnInfo);
  }

  auto fn = getOrCreateCIRFunction(getMangledName(gd), fnType, gd,
                                   /*ForVtable=*/false, dontDefer,
                                   /*IsThunk=*/false, isForDefinition);

  return {fnType, fn};
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

  // Devirtualized destructor calls may come through here instead of via
  // getAddrOfCXXStructor. Make sure we use the MS ABI base destructor instead
  // of the complete destructor when necessary.
  if (const auto *dd = dyn_cast<CXXDestructorDecl>(gd.getDecl())) {
    if (getTarget().getCXXABI().isMicrosoft() &&
        gd.getDtorType() == Dtor_Complete &&
        dd->getParent()->getNumVBases() == 0)
      errorNYI(dd->getSourceRange(),
               "getAddrOfFunction: MS ABI complete destructor");
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
    if (!getTarget().getCXXABI().hasConstructorVariants()) {
      errorNYI(cd->getSourceRange(),
               "getMangledName: C++ constructor without variants");
      return cast<NamedDecl>(gd.getDecl())->getIdentifier()->getName();
    }
  }

  // Keep the first result in the case of a mangling collision.
  const auto *nd = cast<NamedDecl>(gd.getDecl());
  std::string mangledName = getMangledNameImpl(*this, gd, nd);

  auto result = manglings.insert(std::make_pair(mangledName, gd));
  return mangledDeclNames[canonicalGd] = result.first->first();
}

void CIRGenModule::emitTentativeDefinition(const VarDecl *d) {
  assert(!d->getInit() && "Cannot emit definite definitions here!");

  StringRef mangledName = getMangledName(d);
  mlir::Operation *gv = getGlobalValue(mangledName);

  // If we already have a definition, not declaration, with the same mangled
  // name, emitting of declaration is not required (and would actually overwrite
  // the emitted definition).
  if (gv && !mlir::cast<cir::GlobalOp>(gv).isDeclaration())
    return;

  // If we have not seen a reference to this variable yet, place it into the
  // deferred declarations table to be emitted if needed later.
  if (!mustBeEmitted(d) && !gv) {
    deferredDecls[mangledName] = d;
    return;
  }

  // The tentative definition is the only definition.
  emitGlobalVarDefinition(d);
}

bool CIRGenModule::mustBeEmitted(const ValueDecl *global) {
  // Never defer when EmitAllDecls is specified.
  if (langOpts.EmitAllDecls)
    return true;

  const auto *vd = dyn_cast<VarDecl>(global);
  if (vd &&
      ((codeGenOpts.KeepPersistentStorageVariables &&
        (vd->getStorageDuration() == SD_Static ||
         vd->getStorageDuration() == SD_Thread)) ||
       (codeGenOpts.KeepStaticConsts && vd->getStorageDuration() == SD_Static &&
        vd->getType().isConstQualified())))
    return true;

  return getASTContext().DeclMustBeEmitted(global);
}

bool CIRGenModule::mayBeEmittedEagerly(const ValueDecl *global) {
  // In OpenMP 5.0 variables and function may be marked as
  // device_type(host/nohost) and we should not emit them eagerly unless we sure
  // that they must be emitted on the host/device. To be sure we need to have
  // seen a declare target with an explicit mentioning of the function, we know
  // we have if the level of the declare target attribute is -1. Note that we
  // check somewhere else if we should emit this at all.
  if (langOpts.OpenMP >= 50 && !langOpts.OpenMPSimd) {
    std::optional<OMPDeclareTargetDeclAttr *> activeAttr =
        OMPDeclareTargetDeclAttr::getActiveAttr(global);
    if (!activeAttr || (*activeAttr)->getLevel() != (unsigned)-1)
      return false;
  }

  const auto *fd = dyn_cast<FunctionDecl>(global);
  if (fd) {
    // Implicit template instantiations may change linkage if they are later
    // explicitly instantiated, so they should not be emitted eagerly.
    if (fd->getTemplateSpecializationKind() == TSK_ImplicitInstantiation)
      return false;
    // Defer until all versions have been semantically checked.
    if (fd->hasAttr<TargetVersionAttr>() && !fd->isMultiVersion())
      return false;
    if (langOpts.SYCLIsDevice) {
      errorNYI(fd->getSourceRange(), "mayBeEmittedEagerly: SYCL");
      return false;
    }
  }
  const auto *vd = dyn_cast<VarDecl>(global);
  if (vd)
    if (astContext.getInlineVariableDefinitionKind(vd) ==
        ASTContext::InlineVariableDefinitionKind::WeakUnknown)
      // A definition of an inline constexpr static data member may change
      // linkage later if it's redeclared outside the class.
      return false;

  // If OpenMP is enabled and threadprivates must be generated like TLS, delay
  // codegen for global variables, because they may be marked as threadprivate.
  if (langOpts.OpenMP && langOpts.OpenMPUseTLS &&
      astContext.getTargetInfo().isTLSSupported() && isa<VarDecl>(global) &&
      !global->getType().isConstantStorage(astContext, false, false) &&
      !OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(global))
    return false;

  assert((fd || vd) &&
         "Only FunctionDecl and VarDecl should hit this path so far.");
  return true;
}

static bool shouldAssumeDSOLocal(const CIRGenModule &cgm,
                                 cir::CIRGlobalValueInterface gv) {
  if (gv.hasLocalLinkage())
    return true;

  if (!gv.hasDefaultVisibility() && !gv.hasExternalWeakLinkage())
    return true;

  // DLLImport explicitly marks the GV as external.
  // so it shouldn't be dso_local
  // But we don't have the info set now
  assert(!cir::MissingFeatures::opGlobalDLLImportExport());

  const llvm::Triple &tt = cgm.getTriple();
  const CodeGenOptions &cgOpts = cgm.getCodeGenOpts();
  if (tt.isOSCygMing()) {
    // In MinGW and Cygwin, variables without DLLImport can still be
    // automatically imported from a DLL by the linker; don't mark variables
    // that potentially could come from another DLL as DSO local.

    // With EmulatedTLS, TLS variables can be autoimported from other DLLs
    // (and this actually happens in the public interface of libstdc++), so
    // such variables can't be marked as DSO local. (Native TLS variables
    // can't be dllimported at all, though.)
    cgm.errorNYI("shouldAssumeDSOLocal: MinGW");
  }

  // On COFF, don't mark 'extern_weak' symbols as DSO local. If these symbols
  // remain unresolved in the link, they can be resolved to zero, which is
  // outside the current DSO.
  if (tt.isOSBinFormatCOFF() && gv.hasExternalWeakLinkage())
    return false;

  // Every other GV is local on COFF.
  // Make an exception for windows OS in the triple: Some firmware builds use
  // *-win32-macho triples. This (accidentally?) produced windows relocations
  // without GOT tables in older clang versions; Keep this behaviour.
  // FIXME: even thread local variables?
  if (tt.isOSBinFormatCOFF() || (tt.isOSWindows() && tt.isOSBinFormatMachO()))
    return true;

  // Only handle COFF and ELF for now.
  if (!tt.isOSBinFormatELF())
    return false;

  llvm::Reloc::Model rm = cgOpts.RelocationModel;
  const LangOptions &lOpts = cgm.getLangOpts();
  if (rm != llvm::Reloc::Static && !lOpts.PIE) {
    // On ELF, if -fno-semantic-interposition is specified and the target
    // supports local aliases, there will be neither CC1
    // -fsemantic-interposition nor -fhalf-no-semantic-interposition. Set
    // dso_local on the function if using a local alias is preferable (can avoid
    // PLT indirection).
    if (!(isa<cir::FuncOp>(gv) && gv.canBenefitFromLocalAlias()))
      return false;
    return !(lOpts.SemanticInterposition || lOpts.HalfNoSemanticInterposition);
  }

  // A definition cannot be preempted from an executable.
  if (!gv.isDeclarationForLinker())
    return true;

  // Most PIC code sequences that assume that a symbol is local cannot produce a
  // 0 if it turns out the symbol is undefined. While this is ABI and relocation
  // depended, it seems worth it to handle it here.
  if (rm == llvm::Reloc::PIC_ && gv.hasExternalWeakLinkage())
    return false;

  // PowerPC64 prefers TOC indirection to avoid copy relocations.
  if (tt.isPPC64())
    return false;

  if (cgOpts.DirectAccessExternalData) {
    // If -fdirect-access-external-data (default for -fno-pic), set dso_local
    // for non-thread-local variables. If the symbol is not defined in the
    // executable, a copy relocation will be needed at link time. dso_local is
    // excluded for thread-local variables because they generally don't support
    // copy relocations.
    if (auto globalOp = dyn_cast<cir::GlobalOp>(gv.getOperation())) {
      // Assume variables are not thread-local until that support is added.
      assert(!cir::MissingFeatures::opGlobalThreadLocal());
      return true;
    }

    // -fno-pic sets dso_local on a function declaration to allow direct
    // accesses when taking its address (similar to a data symbol). If the
    // function is not defined in the executable, a canonical PLT entry will be
    // needed at link time. -fno-direct-access-external-data can avoid the
    // canonical PLT entry. We don't generalize this condition to -fpie/-fpic as
    // it could just cause trouble without providing perceptible benefits.
    if (isa<cir::FuncOp>(gv) && !cgOpts.NoPLT && rm == llvm::Reloc::Static)
      return true;
  }

  // If we can use copy relocations we can assume it is local.

  // Otherwise don't assume it is local.

  return false;
}

void CIRGenModule::setGlobalVisibility(mlir::Operation *gv,
                                       const NamedDecl *d) const {
  assert(!cir::MissingFeatures::opGlobalVisibility());
}

void CIRGenModule::setDSOLocal(cir::CIRGlobalValueInterface gv) const {
  gv.setDSOLocal(shouldAssumeDSOLocal(*this, gv));
}

void CIRGenModule::setDSOLocal(mlir::Operation *op) const {
  if (auto globalValue = dyn_cast<cir::CIRGlobalValueInterface>(op))
    setDSOLocal(globalValue);
}

void CIRGenModule::setGVProperties(mlir::Operation *op,
                                   const NamedDecl *d) const {
  assert(!cir::MissingFeatures::opGlobalDLLImportExport());
  setGVPropertiesAux(op, d);
}

void CIRGenModule::setGVPropertiesAux(mlir::Operation *op,
                                      const NamedDecl *d) const {
  setGlobalVisibility(op, d);
  setDSOLocal(op);
  assert(!cir::MissingFeatures::opGlobalPartition());
}

void CIRGenModule::setFunctionAttributes(GlobalDecl globalDecl,
                                         cir::FuncOp func,
                                         bool isIncompleteFunction,
                                         bool isThunk) {
  // NOTE(cir): Original CodeGen checks if this is an intrinsic. In CIR we
  // represent them in dedicated ops. The correct attributes are ensured during
  // translation to LLVM. Thus, we don't need to check for them here.

  assert(!cir::MissingFeatures::setFunctionAttributes());
  assert(!cir::MissingFeatures::setTargetAttributes());

  // TODO(cir): This needs a lot of work to better match CodeGen. That
  // ultimately ends up in setGlobalVisibility, which already has the linkage of
  // the LLVM GV (corresponding to our FuncOp) computed, so it doesn't have to
  // recompute it here. This is a minimal fix for now.
  if (!isLocalLinkage(getFunctionLinkage(globalDecl))) {
    const Decl *decl = globalDecl.getDecl();
    func.setGlobalVisibilityAttr(getGlobalVisibilityAttrFromDecl(decl));
  }
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
    assert(mlir::isa<cir::FuncOp>(entry));

    assert(!cir::MissingFeatures::weakRefReference());

    // Handle dropped DLL attributes.
    if (d && !d->hasAttr<DLLImportAttr>() && !d->hasAttr<DLLExportAttr>()) {
      assert(!cir::MissingFeatures::setDLLStorageClass());
      setDSOLocal(entry);
    }

    // If there are two attempts to define the same mangled name, issue an
    // error.
    auto fn = cast<cir::FuncOp>(entry);
    if (isForDefinition && fn && !fn.isDeclaration()) {
      errorNYI(d->getSourceRange(), "Duplicate function definition");
    }
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

  // If we already created a function with the same mangled name (but different
  // type) before, take its name and add it to the list of functions to be
  // replaced with F at the end of CodeGen.
  //
  // This happens if there is a prototype for a function (e.g. "int f()") and
  // then a definition of a different type (e.g. "int f(int x)").
  if (entry) {

    // Fetch a generic symbol-defining operation and its uses.
    auto symbolOp = mlir::cast<mlir::SymbolOpInterface>(entry);

    // This might be an implementation of a function without a prototype, in
    // which case, try to do special replacement of calls which match the new
    // prototype. The really key thing here is that we also potentially drop
    // arguments from the call site so as to make a direct call, which makes the
    // inliner happier and suppresses a number of optimizer warnings (!) about
    // dropping arguments.
    if (symbolOp.getSymbolUses(symbolOp->getParentOp()))
      replaceUsesOfNonProtoTypeWithRealFunction(entry, funcOp);

    // Obliterate no-proto declaration.
    entry->erase();
  }

  if (d)
    setFunctionAttributes(gd, funcOp, /*isIncompleteFunction=*/false, isThunk);

  // 'dontDefer' actually means don't move this to the deferredDeclsToEmit list.
  if (dontDefer) {
    // TODO(cir): This assertion will need an additional condition when we
    // support incomplete functions.
    assert(funcOp.getFunctionType() == funcType);
    return funcOp;
  }

  // All MSVC dtors other than the base dtor are linkonce_odr and delegate to
  // each other bottoming out wiht the base dtor. Therefore we emit non-base
  // dtors on usage, even if there is no dtor definition in the TU.
  if (isa_and_nonnull<CXXDestructorDecl>(d) &&
      getCXXABI().useThunkForDtorVariant(cast<CXXDestructorDecl>(d),
                                         gd.getDtorType()))
    errorNYI(d->getSourceRange(), "getOrCreateCIRFunction: dtor");

  // This is the first use or definition of a mangled name. If there is a
  // deferred decl with this name, remember that we need to emit it at the end
  // of the file.
  auto ddi = deferredDecls.find(mangledName);
  if (ddi != deferredDecls.end()) {
    // Move the potentially referenced deferred decl to the
    // DeferredDeclsToEmit list, and remove it from DeferredDecls (since we
    // don't need it anymore).
    addDeferredDeclToEmit(ddi->second);
    deferredDecls.erase(ddi);

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
  } else if (getLangOpts().CPlusPlus && d) {
    // Look for a declaration that's lexically in a record.
    for (const auto *fd = cast<FunctionDecl>(d)->getMostRecentDecl(); fd;
         fd = fd->getPreviousDecl()) {
      if (isa<CXXRecordDecl>(fd->getLexicalDeclContext())) {
        if (fd->doesThisDeclarationHaveABody()) {
          addDeferredDeclToEmit(gd.getWithDecl(fd));
          break;
        }
      }
    }
  }

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

    assert(!cir::MissingFeatures::opFuncAstDeclAttr());

    if (funcDecl && !funcDecl->hasPrototype())
      func.setNoProto(true);

    assert(func.isDeclaration() && "expected empty body");

    // A declaration gets private visibility by default, but external linkage
    // as the default linkage.
    func.setLinkageAttr(cir::GlobalLinkageKindAttr::get(
        &getMLIRContext(), cir::GlobalLinkageKind::ExternalLinkage));
    mlir::SymbolTable::setSymbolVisibility(
        func, mlir::SymbolTable::Visibility::Private);

    assert(!cir::MissingFeatures::opFuncExtraAttrs());

    if (!cgf)
      theModule.push_back(func);
  }
  return func;
}

mlir::SymbolTable::Visibility
CIRGenModule::getMLIRVisibility(cir::GlobalOp op) {
  // MLIR doesn't accept public symbols declarations (only
  // definitions).
  if (op.isDeclaration())
    return mlir::SymbolTable::Visibility::Private;
  return getMLIRVisibilityFromCIRLinkage(op.getLinkage());
}

mlir::SymbolTable::Visibility
CIRGenModule::getMLIRVisibilityFromCIRLinkage(cir::GlobalLinkageKind glk) {
  switch (glk) {
  case cir::GlobalLinkageKind::InternalLinkage:
  case cir::GlobalLinkageKind::PrivateLinkage:
    return mlir::SymbolTable::Visibility::Private;
  case cir::GlobalLinkageKind::ExternalLinkage:
  case cir::GlobalLinkageKind::ExternalWeakLinkage:
  case cir::GlobalLinkageKind::LinkOnceODRLinkage:
  case cir::GlobalLinkageKind::AvailableExternallyLinkage:
  case cir::GlobalLinkageKind::CommonLinkage:
  case cir::GlobalLinkageKind::WeakAnyLinkage:
  case cir::GlobalLinkageKind::WeakODRLinkage:
    return mlir::SymbolTable::Visibility::Public;
  default: {
    llvm::errs() << "visibility not implemented for '"
                 << stringifyGlobalLinkageKind(glk) << "'\n";
    assert(0 && "not implemented");
  }
  }
  llvm_unreachable("linkage should be handled above!");
}

cir::VisibilityKind CIRGenModule::getGlobalVisibilityKindFromClangVisibility(
    clang::VisibilityAttr::VisibilityType visibility) {
  switch (visibility) {
  case clang::VisibilityAttr::VisibilityType::Default:
    return cir::VisibilityKind::Default;
  case clang::VisibilityAttr::VisibilityType::Hidden:
    return cir::VisibilityKind::Hidden;
  case clang::VisibilityAttr::VisibilityType::Protected:
    return cir::VisibilityKind::Protected;
  }
  llvm_unreachable("unexpected visibility value");
}

cir::VisibilityAttr
CIRGenModule::getGlobalVisibilityAttrFromDecl(const Decl *decl) {
  const clang::VisibilityAttr *va = decl->getAttr<clang::VisibilityAttr>();
  cir::VisibilityAttr cirVisibility =
      cir::VisibilityAttr::get(&getMLIRContext());
  if (va) {
    cirVisibility = cir::VisibilityAttr::get(
        &getMLIRContext(),
        getGlobalVisibilityKindFromClangVisibility(va->getVisibility()));
  }
  return cirVisibility;
}

void CIRGenModule::release() {
  emitDeferred();
  applyReplacements();

  theModule->setAttr(cir::CIRDialect::getModuleLevelAsmAttrName(),
                     builder.getArrayAttr(globalScopeAsm));

  // There's a lot of code that is not implemented yet.
  assert(!cir::MissingFeatures::cgmRelease());
}

void CIRGenModule::emitAliasForGlobal(StringRef mangledName,
                                      mlir::Operation *op, GlobalDecl aliasGD,
                                      cir::FuncOp aliasee,
                                      cir::GlobalLinkageKind linkage) {

  auto *aliasFD = dyn_cast<FunctionDecl>(aliasGD.getDecl());
  assert(aliasFD && "expected FunctionDecl");

  // The aliasee function type is different from the alias one, this difference
  // is specific to CIR because in LLVM the ptr types are already erased at this
  // point.
  const CIRGenFunctionInfo &fnInfo =
      getTypes().arrangeCXXStructorDeclaration(aliasGD);
  cir::FuncType fnType = getTypes().getFunctionType(fnInfo);

  cir::FuncOp alias =
      createCIRFunction(getLoc(aliasGD.getDecl()->getSourceRange()),
                        mangledName, fnType, aliasFD);
  alias.setAliasee(aliasee.getName());
  alias.setLinkage(linkage);
  // Declarations cannot have public MLIR visibility, just mark them private
  // but this really should have no meaning since CIR should not be using
  // this information to derive linkage information.
  mlir::SymbolTable::setSymbolVisibility(
      alias, mlir::SymbolTable::Visibility::Private);

  // Alias constructors and destructors are always unnamed_addr.
  assert(!cir::MissingFeatures::opGlobalUnnamedAddr());

  // Switch any previous uses to the alias.
  if (op) {
    errorNYI(aliasFD->getSourceRange(), "emitAliasForGlobal: previous uses");
  } else {
    // Name already set by createCIRFunction
  }

  // Finally, set up the alias with its proper name and attributes.
  setCommonAttributes(aliasGD, alias);
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

mlir::Attribute CIRGenModule::getAddrOfRTTIDescriptor(mlir::Location loc,
                                                      QualType ty, bool forEh) {
  // Return a bogus pointer if RTTI is disabled, unless it's for EH.
  // FIXME: should we even be calling this method if RTTI is disabled
  // and it's not for EH?
  if (!shouldEmitRTTI(forEh))
    return builder.getConstNullPtrAttr(builder.getUInt8PtrTy());

  errorNYI(loc, "getAddrOfRTTIDescriptor");
  return mlir::Attribute();
}

// TODO(cir): this can be shared with LLVM codegen.
CharUnits CIRGenModule::computeNonVirtualBaseClassOffset(
    const CXXRecordDecl *derivedClass,
    llvm::iterator_range<CastExpr::path_const_iterator> path) {
  CharUnits offset = CharUnits::Zero();

  const ASTContext &astContext = getASTContext();
  const CXXRecordDecl *rd = derivedClass;

  for (const CXXBaseSpecifier *base : path) {
    assert(!base->isVirtual() && "Should not see virtual bases here!");

    // Get the layout.
    const ASTRecordLayout &layout = astContext.getASTRecordLayout(rd);

    const auto *baseDecl = base->getType()->castAsCXXRecordDecl();

    // Add the offset.
    offset += layout.getBaseClassOffset(baseDecl);

    rd = baseDecl;
  }

  return offset;
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
