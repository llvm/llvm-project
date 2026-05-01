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
#include "CIRGenCUDARuntime.h"
#include "CIRGenCXXABI.h"
#include "CIRGenConstantEmitter.h"
#include "CIRGenFunction.h"

#include "mlir/Dialect/OpenMP/OpenMPOffloadUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclOpenACC.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/SourceManager.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include "CIRGenFunctionInfo.h"
#include "TargetInfo.h"
#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"

#include <algorithm>

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
    cgm.errorNYI("createCXXABI: C++ ABI kind");
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
  voidTy = cir::VoidType::get(&getMLIRContext());
  voidPtrTy = cir::PointerType::get(voidTy);
  sInt8Ty = cir::IntType::get(&getMLIRContext(), 8, /*isSigned=*/true);
  sInt16Ty = cir::IntType::get(&getMLIRContext(), 16, /*isSigned=*/true);
  sInt32Ty = cir::IntType::get(&getMLIRContext(), 32, /*isSigned=*/true);
  sInt64Ty = cir::IntType::get(&getMLIRContext(), 64, /*isSigned=*/true);
  sInt128Ty = cir::IntType::get(&getMLIRContext(), 128, /*isSigned=*/true);
  uInt8Ty = cir::IntType::get(&getMLIRContext(), 8, /*isSigned=*/false);
  uInt8PtrTy = cir::PointerType::get(uInt8Ty);
  cirAllocaAddressSpace = getTargetCIRGenInfo().getCIRAllocaAddressSpace();
  uInt16Ty = cir::IntType::get(&getMLIRContext(), 16, /*isSigned=*/false);
  uInt32Ty = cir::IntType::get(&getMLIRContext(), 32, /*isSigned=*/false);
  uInt64Ty = cir::IntType::get(&getMLIRContext(), 64, /*isSigned=*/false);
  uInt128Ty = cir::IntType::get(&getMLIRContext(), 128, /*isSigned=*/false);
  fP16Ty = cir::FP16Type::get(&getMLIRContext());
  bFloat16Ty = cir::BF16Type::get(&getMLIRContext());
  floatTy = cir::SingleType::get(&getMLIRContext());
  doubleTy = cir::DoubleType::get(&getMLIRContext());
  fP80Ty = cir::FP80Type::get(&getMLIRContext());
  fP128Ty = cir::FP128Type::get(&getMLIRContext());

  allocaInt8PtrTy = cir::PointerType::get(uInt8Ty, cirAllocaAddressSpace);

  PointerAlignInBytes =
      astContext
          .toCharUnitsFromBits(
              astContext.getTargetInfo().getPointerAlign(LangAS::Default))
          .getQuantity();

  const unsigned charSize = astContext.getTargetInfo().getCharWidth();
  uCharTy = cir::IntType::get(&getMLIRContext(), charSize, /*isSigned=*/false);

  // TODO(CIR): Should be updated once TypeSizeInfoAttr is upstreamed
  const unsigned sizeTypeSize =
      astContext.getTypeSize(astContext.getSignedSizeType());
  SizeSizeInBytes = astContext.toCharUnitsFromBits(sizeTypeSize).getQuantity();
  // In CIRGenTypeCache, UIntPtrTy and SizeType are fields of the same union
  uIntPtrTy =
      cir::IntType::get(&getMLIRContext(), sizeTypeSize, /*isSigned=*/false);
  ptrDiffTy =
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

  if (langOpts.OpenMP) {
    mlir::omp::OffloadModuleOpts ompOpts(
        langOpts.OpenMPTargetDebug, langOpts.OpenMPTeamSubscription,
        langOpts.OpenMPThreadSubscription, langOpts.OpenMPNoThreadState,
        langOpts.OpenMPNoNestedParallelism, langOpts.OpenMPIsTargetDevice,
        getTriple().isGPU(), langOpts.OpenMPForceUSM, langOpts.OpenMP,
        langOpts.OMPHostIRFile, langOpts.OMPTargetTriples, langOpts.NoGPULib);
    mlir::omp::setOffloadModuleInterfaceAttributes(theModule, ompOpts);
  }

  if (langOpts.CUDA)
    createCUDARuntime();
  if (langOpts.OpenMP)
    createOpenMPRuntime();

  // Set the module name to be the name of the main file. TranslationUnitDecl
  // often contains invalid source locations and isn't a reliable source for the
  // module location.
  FileID mainFileId = astContext.getSourceManager().getMainFileID();
  const FileEntry &mainFile =
      *astContext.getSourceManager().getFileEntryForID(mainFileId);
  StringRef path = mainFile.tryGetRealPathName();
  if (!path.empty()) {
    theModule.setSymName(path);
    theModule->setLoc(mlir::FileLineColLoc::get(&mlirContext, path,
                                                /*line=*/0,
                                                /*column=*/0));
  }

  // Set CUDA GPU binary handle.
  if (langOpts.CUDA) {
    llvm::StringRef cudaBinaryName = codeGenOpts.CudaGpuBinaryFileName;
    if (!cudaBinaryName.empty()) {
      theModule->setAttr(cir::CIRDialect::getCUDABinaryHandleAttrName(),
                         cir::CUDABinaryHandleAttr::get(
                             &mlirContext, mlir::StringAttr::get(
                                               &mlirContext, cudaBinaryName)));
    }
  }
}

CIRGenModule::~CIRGenModule() = default;

void CIRGenModule::createCUDARuntime() {
  cudaRuntime.reset(createNVCUDARuntime(*this));
}

void CIRGenModule::createOpenMPRuntime() {
  openMPRuntime = std::make_unique<CIRGenOpenMPRuntime>(*this);
}

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
                                                LValueBaseInfo *baseInfo,
                                                bool forPointeeType) {
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

  bool alignForArray = t->isArrayType();

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
  const CXXRecordDecl *rd = nullptr;
  if (t.getQualifiers().hasUnaligned()) {
    alignment = CharUnits::One();
  } else if (forPointeeType && !alignForArray &&
             (rd = t->getAsCXXRecordDecl())) {
    alignment = getClassPointerAlignment(rd);
  } else {
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

CharUnits
CIRGenModule::getNaturalPointeeTypeAlignment(QualType t,
                                             LValueBaseInfo *baseInfo) {
  return getNaturalTypeAlignment(t->getPointeeType(), baseInfo,
                                 /*forPointeeType=*/true);
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
  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
    theTargetCIRGenInfo = createNVPTXTargetCIRGenInfo(genTypes);
    return *theTargetCIRGenInfo;
  case llvm::Triple::amdgcn: {
    theTargetCIRGenInfo = createAMDGPUTargetCIRGenInfo(genTypes);
    return *theTargetCIRGenInfo;
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
  if (auto gv = dyn_cast<cir::GetGlobalOp>(op)) {
    globalValueOp = getGlobalValue(gv.getName());
    assert(globalValueOp && "expected a valid global op");
  }

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

  emitDeferredVTables();
  // Emitting a vtable doesn't directly cause more vtables to
  // become deferred, although it can cause functions to be
  // emitted that then need those vtables.
  assert(deferredVTables.empty());

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
    if (!deferredVTables.empty() || !deferredDeclsToEmit.empty()) {
      emitDeferred();
      assert(deferredVTables.empty() && deferredDeclsToEmit.empty());
    }
  }
}

template <typename AttrT> static bool hasImplicitAttr(const ValueDecl *decl) {
  if (!decl)
    return false;
  if (auto *attr = decl->getAttr<AttrT>())
    return attr->isImplicit();
  return decl->isImplicit();
}

// TODO(cir): This should be shared with OG Codegen.
bool CIRGenModule::shouldEmitCUDAGlobalVar(const VarDecl *global) const {
  assert(langOpts.CUDA && "Should not be called by non-CUDA languages");
  // We need to emit host-side 'shadows' for all global
  // device-side variables because the CUDA runtime needs their
  // size and host-side address in order to provide access to
  // their device-side incarnations.
  return !langOpts.CUDAIsDevice || global->hasAttr<CUDADeviceAttr>() ||
         global->hasAttr<CUDAConstantAttr>() ||
         global->hasAttr<CUDASharedAttr>() ||
         global->getType()->isCUDADeviceBuiltinSurfaceType() ||
         global->getType()->isCUDADeviceBuiltinTextureType();
}

void CIRGenModule::emitGlobal(clang::GlobalDecl gd) {
  if (const auto *cd = dyn_cast<clang::OpenACCConstructDecl>(gd.getDecl())) {
    emitGlobalOpenACCDecl(cd);
    return;
  }

  const auto *global = cast<ValueDecl>(gd.getDecl());

  if (global->hasAttr<WeakRefAttr>())
    errorNYI(global->getSourceRange(), "emitGlobal: WeakRefAttr");

  if (global->hasAttr<AliasAttr>())
    errorNYI(global->getSourceRange(), "emitGlobal: AliasAttr");

  // If this is CUDA, be selective about which declarations we emit.
  // Non-constexpr non-lambda implicit host device functions are not emitted
  // unless they are used on device side.
  if (langOpts.CUDA) {
    assert((isa<FunctionDecl>(global) || isa<VarDecl>(global)) &&
           "Expected Variable or Function");
    if (const auto *varDecl = dyn_cast<VarDecl>(global)) {
      if (!shouldEmitCUDAGlobalVar(varDecl))
        return;
      // TODO(cir): This should be shared with OG Codegen.
    } else if (langOpts.CUDAIsDevice) {
      const auto *functionDecl = dyn_cast<FunctionDecl>(global);
      if ((!global->hasAttr<CUDADeviceAttr>() ||
           (langOpts.OffloadImplicitHostDeviceTemplates &&
            hasImplicitAttr<CUDAHostAttr>(functionDecl) &&
            hasImplicitAttr<CUDADeviceAttr>(functionDecl) &&
            !functionDecl->isConstexpr() &&
            !isLambdaCallOperator(functionDecl) &&
            !getASTContext().CUDAImplicitHostDeviceFunUsedByDevice.count(
                functionDecl))) &&
          !global->hasAttr<CUDAGlobalAttr>() &&
          !(langOpts.HIPStdPar && isa<FunctionDecl>(global) &&
            !global->hasAttr<CUDAHostAttr>()))
        return;
      // Device-only functions are the only things we skip.
    } else if (!global->hasAttr<CUDAHostAttr>() &&
               global->hasAttr<CUDADeviceAttr>())
      return;
  }

  if (langOpts.OpenMP) {
    // If this is OpenMP, check if it is legal to emit this global normally.
    if (openMPRuntime && openMPRuntime->emitTargetGlobal(gd))
      return;
    if (auto *drd = dyn_cast<OMPDeclareReductionDecl>(global)) {
      if (mustBeEmitted(global))
        emitOMPDeclareReduction(drd);
      return;
    }
    if (auto *dmd = dyn_cast<OMPDeclareMapperDecl>(global)) {
      if (mustBeEmitted(global))
        emitOMPDeclareMapper(dmd);
      return;
    }
  }

  if (const auto *fd = dyn_cast<FunctionDecl>(global)) {
    // Update deferred annotations with the latest declaration if the function
    // was already used or defined.
    if (fd->hasAttr<AnnotateAttr>()) {
      StringRef mangledName = getMangledName(gd);
      if (getGlobalValue(mangledName))
        deferredAnnotations[mangledName] = fd;
    }
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
  setCIRFunctionAttributesForDefinition(funcDecl, funcOp);

  auto getPriority = [this](const auto *attr) -> int {
    Expr *e = attr->getPriority();
    if (e)
      return e->EvaluateKnownConstInt(this->getASTContext()).getExtValue();
    return attr->DefaultPriority;
  };

  if (const ConstructorAttr *ca = funcDecl->getAttr<ConstructorAttr>())
    addGlobalCtor(funcOp, getPriority(ca));
  if (const DestructorAttr *da = funcDecl->getAttr<DestructorAttr>())
    addGlobalDtor(funcOp, getPriority(da));

  if (funcDecl->getAttr<AnnotateAttr>())
    deferredAnnotations[getMangledName(gd)] = funcDecl;

  if (getLangOpts().OpenMP && funcDecl->hasAttr<OMPDeclareTargetDeclAttr>())
    getOpenMPRuntime().emitDeclareTargetFunction(funcDecl, funcOp);
}

/// Track functions to be called before main() runs.
void CIRGenModule::addGlobalCtor(cir::FuncOp ctor,
                                 std::optional<int> priority) {
  assert(!cir::MissingFeatures::globalCtorLexOrder());
  assert(!cir::MissingFeatures::globalCtorAssociatedData());

  // Traditional LLVM codegen directly adds the function to the list of global
  // ctors. In CIR we just add a global_ctor attribute to the function. The
  // global list is created in LoweringPrepare.
  //
  // FIXME(from traditional LLVM): Type coercion of void()* types.
  ctor.setGlobalCtorPriority(priority);
}

/// Add a function to the list that will be called when the module is unloaded.
void CIRGenModule::addGlobalDtor(cir::FuncOp dtor,
                                 std::optional<int> priority) {
  if (codeGenOpts.RegisterGlobalDtorsWithAtExit &&
      (!getASTContext().getTargetInfo().getTriple().isOSAIX()))
    errorNYI(dtor.getLoc(), "registerGlobalDtorsWithAtExit");

  // FIXME(from traditional LLVM): Type coercion of void()* types.
  dtor.setGlobalDtorPriority(priority);
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
  auto it = symbolLookupCache.find(name);
  return it != symbolLookupCache.end() ? it->second : nullptr;
}

cir::GlobalOp
CIRGenModule::createGlobalOp(CIRGenModule &cgm, mlir::Location loc,
                             StringRef name, mlir::Type t, bool isConstant,
                             mlir::ptr::MemorySpaceAttrInterface addrSpace,
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

    g = cir::GlobalOp::create(builder, loc, name, t, isConstant, addrSpace);
    if (!insertPoint)
      cgm.lastGlobalOp = g;

    // Default to private until we can judge based on the initializer,
    // since MLIR doesn't allow public declarations.
    mlir::SymbolTable::setSymbolVisibility(
        g, mlir::SymbolTable::Visibility::Private);
  }
  cgm.symbolLookupCache[g.getSymNameAttr()] = g;
  return g;
}

void CIRGenModule::setCommonAttributes(GlobalDecl gd, mlir::Operation *gv) {
  const Decl *d = gd.getDecl();
  if (isa_and_nonnull<NamedDecl>(d))
    setGVProperties(gv, dyn_cast<NamedDecl>(d));
  assert(!cir::MissingFeatures::defaultVisibility());

  if (auto gvi = mlir::dyn_cast<cir::CIRGlobalValueInterface>(gv)) {
    if (d && d->hasAttr<UsedAttr>())
      addUsedOrCompilerUsedGlobal(gvi);

    if (const auto *vd = dyn_cast_if_present<VarDecl>(d);
        vd && ((codeGenOpts.KeepPersistentStorageVariables &&
                (vd->getStorageDuration() == SD_Static ||
                 vd->getStorageDuration() == SD_Thread)) ||
               (codeGenOpts.KeepStaticConsts &&
                vd->getStorageDuration() == SD_Static &&
                vd->getType().isConstQualified())))
      addUsedOrCompilerUsedGlobal(gvi);
  }
}

/// Get the feature delta from the default feature map for the given target CPU.
static std::vector<std::string>
getFeatureDeltaFromDefault(const CIRGenModule &cgm, llvm::StringRef targetCPU,
                           llvm::StringMap<bool> &featureMap) {
  llvm::StringMap<bool> defaultFeatureMap;
  cgm.getTarget().initFeatureMap(
      defaultFeatureMap, cgm.getASTContext().getDiagnostics(), targetCPU, {});

  std::vector<std::string> delta;
  for (const auto &[k, v] : featureMap) {
    auto defaultIt = defaultFeatureMap.find(k);
    if (defaultIt == defaultFeatureMap.end() || defaultIt->getValue() != v)
      delta.push_back((v ? "+" : "-") + k.str());
  }

  return delta;
}

bool CIRGenModule::getCPUAndFeaturesAttributes(
    GlobalDecl gd, llvm::StringMap<std::string> &attrs,
    bool setTargetFeatures) {
  // Add target-cpu and target-features attributes to functions. If
  // we have a decl for the function and it has a target attribute then
  // parse that and add it to the feature set.
  llvm::StringRef targetCPU = getTarget().getTargetOpts().CPU;
  llvm::StringRef tuneCPU = getTarget().getTargetOpts().TuneCPU;
  std::vector<std::string> features;
  // `fd` may be null when emitting attributes for globals that don't have a
  // FunctionDecl. The AMDGPU branch below handles
  // the null case via initFeatureMap.
  const auto *fd = dyn_cast_or_null<FunctionDecl>(gd.getDecl());
  fd = fd ? fd->getMostRecentDecl() : fd;
  const auto *td = fd ? fd->getAttr<TargetAttr>() : nullptr;
  const auto *tv = fd ? fd->getAttr<TargetVersionAttr>() : nullptr;
  assert((!td || !tv) && "both target_version and target specified");
  const auto *sd = fd ? fd->getAttr<CPUSpecificAttr>() : nullptr;
  const auto *tc = fd ? fd->getAttr<TargetClonesAttr>() : nullptr;
  bool addedAttr = false;
  if (td || tv || sd || tc) {
    assert(!cir::MissingFeatures::opFuncMultiVersioning());
  } else {
    // Just add the existing target cpu and target features to the function.
    if (setTargetFeatures && getTarget().getTriple().isAMDGPU()) {
      llvm::StringMap<bool> featureMap;
      if (fd)
        astContext.getFunctionFeatureMap(featureMap, gd);
      else
        getTarget().initFeatureMap(featureMap, astContext.getDiagnostics(),
                                   targetCPU,
                                   getTarget().getTargetOpts().Features);
      features = getFeatureDeltaFromDefault(*this, targetCPU, featureMap);
    } else {
      features = getTarget().getTargetOpts().Features;
    }
  }

  if (!targetCPU.empty()) {
    attrs["cir.target-cpu"] = targetCPU.str();
    addedAttr = true;
  }
  if (!tuneCPU.empty()) {
    attrs["cir.tune-cpu"] = tuneCPU.str();
    addedAttr = true;
  }
  if (!features.empty() && setTargetFeatures) {
    llvm::erase_if(features, [&](const std::string &f) {
      assert(!f.empty() && (f[0] == '+' || f[0] == '-') &&
             "feature string must start with '+' or '-'");
      return getTarget().isReadOnlyFeature(f.substr(1));
    });
    llvm::sort(features);
    attrs["cir.target-features"] = llvm::join(features, ",");
    addedAttr = true;
  }
  // TODO(cir): add metadata for AArch64 Function Multi Versioning.
  assert(!cir::MissingFeatures::opFuncMultiVersioning());
  return addedAttr;
}

void CIRGenModule::setNonAliasAttributes(GlobalDecl gd, mlir::Operation *op) {
  setCommonAttributes(gd, op);

  const Decl *d = gd.getDecl();
  if (d) {
    if (auto gvi = mlir::dyn_cast<cir::CIRGlobalValueInterface>(op)) {
      if (const auto *sa = d->getAttr<SectionAttr>())
        gvi.setSection(builder.getStringAttr(sa->getName()));
      if (d->hasAttr<RetainAttr>())
        addUsedGlobal(gvi);

      if (auto func = dyn_cast<cir::FuncOp>(op)) {
        llvm::StringMap<std::string> attrs;
        if (getCPUAndFeaturesAttributes(gd, attrs)) {
          // TODO(cir): Classic codegen removes the existing target-cpu,
          // target-features, tune-cpu and fmv-features attributes here
          // before adding the new ones.
          for (const auto &[key, val] : attrs)
            func->setAttr(key, builder.getStringAttr(val));
        }
      }
    }
  }

  assert(!cir::MissingFeatures::opGlobalPragmaClangSection());
  getTargetCIRGenInfo().setTargetAttributes(gd.getDecl(), op, *this);
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

LangAS CIRGenModule::getGlobalVarAddressSpace(const VarDecl *d) {
  if (langOpts.OpenCL) {
    LangAS as = d ? d->getType().getAddressSpace() : LangAS::opencl_global;
    assert(as == LangAS::opencl_global || as == LangAS::opencl_global_device ||
           as == LangAS::opencl_global_host || as == LangAS::opencl_constant ||
           as == LangAS::opencl_local || as >= LangAS::FirstTargetAddressSpace);
    return as;
  }

  if (langOpts.SYCLIsDevice &&
      (!d || d->getType().getAddressSpace() == LangAS::Default))
    errorNYI("SYCL global address space");

  if (langOpts.CUDA && langOpts.CUDAIsDevice) {
    if (d) {
      if (d->hasAttr<CUDAConstantAttr>())
        return LangAS::cuda_constant;
      if (d->hasAttr<CUDASharedAttr>())
        return LangAS::cuda_shared;
      if (d->hasAttr<CUDADeviceAttr>())
        return LangAS::cuda_device;
      if (d->getType().isConstQualified())
        return LangAS::cuda_constant;
    }
    return LangAS::cuda_device;
  }

  if (langOpts.OpenMP)
    errorNYI("OpenMP global address space");

  return getTargetCIRGenInfo().getGlobalVarAddressSpace(*this, d);
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

static llvm::SmallVector<int64_t> indexesOfArrayAttr(mlir::ArrayAttr indexes) {
  llvm::SmallVector<int64_t> inds;
  for (mlir::Attribute i : indexes) {
    auto ind = mlir::cast<mlir::IntegerAttr>(i);
    inds.push_back(ind.getValue().getSExtValue());
  }
  return inds;
}

static bool isViewOnGlobal(cir::GlobalOp glob, cir::GlobalViewAttr view) {
  return view.getSymbol().getValue() == glob.getSymName();
}

static cir::GlobalViewAttr createNewGlobalView(CIRGenModule &cgm,
                                               cir::GlobalOp newGlob,
                                               cir::GlobalViewAttr attr,
                                               mlir::Type oldTy) {
  // If the attribute does not require indexes or it is not a global view on
  // the global we're replacing, keep the original attribute.
  if (!attr.getIndices() || !isViewOnGlobal(newGlob, attr))
    return attr;

  llvm::SmallVector<int64_t> oldInds = indexesOfArrayAttr(attr.getIndices());
  llvm::SmallVector<int64_t> newInds;
  CIRGenBuilderTy &bld = cgm.getBuilder();
  const cir::CIRDataLayout &layout = cgm.getDataLayout();
  mlir::Type newTy = newGlob.getSymType();

  uint64_t offset =
      bld.computeOffsetFromGlobalViewIndices(layout, oldTy, oldInds);
  bld.computeGlobalViewIndicesFromFlatOffset(offset, newTy, layout, newInds);
  cir::PointerType newPtrTy;

  if (isa<cir::RecordType>(oldTy))
    newPtrTy = cir::PointerType::get(newTy);
  else if (isa<cir::ArrayType>(oldTy))
    newPtrTy = cast<cir::PointerType>(attr.getType());

  if (newPtrTy)
    return bld.getGlobalViewAttr(newPtrTy, newGlob, newInds);

  // This may be unreachable in practice, but keep it as errorNYI while CIR
  // is still under development.
  cgm.errorNYI("Unhandled type in createNewGlobalView");
  return {};
}

static mlir::Attribute getNewInitValue(CIRGenModule &cgm, cir::GlobalOp newGlob,
                                       mlir::Type oldTy,
                                       mlir::Attribute oldInit) {
  if (auto oldView = mlir::dyn_cast<cir::GlobalViewAttr>(oldInit))
    return createNewGlobalView(cgm, newGlob, oldView, oldTy);

  auto getNewInitElements =
      [&](mlir::ArrayAttr oldElements) -> mlir::ArrayAttr {
    llvm::SmallVector<mlir::Attribute> newElements;
    for (mlir::Attribute elt : oldElements) {
      if (auto view = mlir::dyn_cast<cir::GlobalViewAttr>(elt))
        newElements.push_back(createNewGlobalView(cgm, newGlob, view, oldTy));
      else if (mlir::isa<cir::ConstArrayAttr, cir::ConstRecordAttr>(elt))
        newElements.push_back(getNewInitValue(cgm, newGlob, oldTy, elt));
      else
        newElements.push_back(elt);
    }
    return mlir::ArrayAttr::get(cgm.getBuilder().getContext(), newElements);
  };

  if (auto oldArray = mlir::dyn_cast<cir::ConstArrayAttr>(oldInit)) {
    mlir::Attribute newElements =
        getNewInitElements(mlir::cast<mlir::ArrayAttr>(oldArray.getElts()));
    return cgm.getBuilder().getConstArray(
        newElements, mlir::cast<cir::ArrayType>(oldArray.getType()));
  }
  if (auto oldRecord = mlir::dyn_cast<cir::ConstRecordAttr>(oldInit)) {
    mlir::ArrayAttr newMembers = getNewInitElements(oldRecord.getMembers());
    auto recordTy = mlir::cast<cir::RecordType>(oldRecord.getType());
    return cgm.getBuilder().getConstRecordOrZeroAttr(
        newMembers, recordTy.getPacked(), recordTy.getPadded(), recordTy);
  }

  // This may be unreachable in practice, but keep it as errorNYI while CIR
  // is still under development.
  cgm.errorNYI("Unhandled type in getNewInitValue");
  return {};
}

// We want to replace a global value, but because of CIR's typed pointers,
// we need to update the existing uses to reflect the new type, not just replace
// them directly.
void CIRGenModule::replaceGlobal(cir::GlobalOp oldGV, cir::GlobalOp newGV) {
  assert(oldGV.getSymName() == newGV.getSymName() && "symbol names must match");

  mlir::Type oldTy = oldGV.getSymType();
  mlir::Type newTy = newGV.getSymType();

  assert(!cir::MissingFeatures::addressSpace());

  // If the type didn't change, why are we here?
  assert(oldTy != newTy && "expected type change in replaceGlobal");

  // Visit all uses and add handling to fix up the types.
  std::optional<mlir::SymbolTable::UseRange> oldSymUses =
      oldGV.getSymbolUses(theModule);
  for (mlir::SymbolTable::SymbolUse use : *oldSymUses) {
    mlir::Operation *userOp = use.getUser();
    assert(
        (mlir::isa<cir::GetGlobalOp, cir::GlobalOp, cir::ConstantOp>(userOp)) &&
        "Unexpected user for global op");

    if (auto getGlobalOp = dyn_cast<cir::GetGlobalOp>(use.getUser())) {
      mlir::Value useOpResultValue = getGlobalOp.getAddr();
      useOpResultValue.setType(cir::PointerType::get(newTy));

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(getGlobalOp);
      mlir::Type ptrTy = builder.getPointerTo(oldTy);
      mlir::Value cast =
          builder.createBitcast(getGlobalOp->getLoc(), useOpResultValue, ptrTy);
      useOpResultValue.replaceAllUsesExcept(cast, cast.getDefiningOp());
    } else if (auto glob = dyn_cast<cir::GlobalOp>(userOp)) {
      if (auto init = glob.getInitialValue()) {
        mlir::Attribute nw = getNewInitValue(*this, newGV, oldTy, init.value());
        glob.setInitialValueAttr(nw);
      }
    } else if (auto c = dyn_cast<cir::ConstantOp>(userOp)) {
      mlir::Attribute init = getNewInitValue(*this, newGV, oldTy, c.getValue());
      auto typedAttr = mlir::cast<mlir::TypedAttr>(init);
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(c);
      auto newUser = cir::ConstantOp::create(builder, c.getLoc(), typedAttr);
      c.replaceAllUsesWith(newUser.getOperation());
      c.erase();
    }
  }

  // If the old global is being tracked as the most-recently-created global,
  // update it so that subsequent globals are not inserted after a (now
  // erased) operation, which would leave them detached from the module.
  if (lastGlobalOp == oldGV)
    lastGlobalOp = newGV;
  eraseGlobalSymbol(oldGV);
  oldGV.erase();
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
      errorNYI(d->getSourceRange(),
               "getOrCreateCIRGlobal: global with non-GlobalOp type");
    entry = cast<cir::GlobalOp>(v);
  }

  if (entry) {
    mlir::ptr::MemorySpaceAttrInterface entryCIRAS = entry.getAddrSpaceAttr();
    assert(!cir::MissingFeatures::opGlobalWeakRef());

    assert(!cir::MissingFeatures::setDLLStorageClass());
    assert(!cir::MissingFeatures::openMP());

    if (entry.getSymType() == ty &&
        cir::isMatchingAddressSpace(entryCIRAS, langAS))
      return entry;

    // If there are two attempts to define the same mangled name, issue an
    // error.
    //
    // TODO(cir): look at mlir::GlobalValue::isDeclaration for all aspects of
    // recognizing the global as a declaration, for now only check if
    // initializer is present.
    if (isForDefinition && !entry.isDeclaration()) {
      errorNYI(d->getSourceRange(),
               "getOrCreateCIRGlobal: global with conflicting type");
    }

    // Address space check removed because it is unnecessary because CIR records
    // address space info in types.

    // (If global is requested for a definition, we always need to create a new
    // global, not just return a bitcast.)
    if (!isForDefinition)
      return entry;
  }

  mlir::Location loc = getLoc(d->getSourceRange());

  // Calculate constant storage flag before creating the global. This was moved
  // from after the global creation to ensure the constant flag is set correctly
  // at creation time, matching the logic used in emitCXXGlobalVarDeclInit.
  bool isConstant = false;
  if (d) {
    bool needsDtor =
        d->needsDestruction(astContext) == QualType::DK_cxx_destructor;
    isConstant = d->getType().isConstantStorage(
        astContext, /*ExcludeCtor=*/true, /*ExcludeDtor=*/!needsDtor);
  }

  mlir::ptr::MemorySpaceAttrInterface declCIRAS =
      cir::toCIRAddressSpaceAttr(getMLIRContext(), getGlobalVarAddressSpace(d));

  // mlir::SymbolTable::Visibility::Public is the default, no need to explicitly
  // mark it as such.
  cir::GlobalOp gv = CIRGenModule::createGlobalOp(
      *this, loc, mangledName, ty, isConstant, declCIRAS,
      /*insertPoint=*/entry.getOperation());

  // If we already created a global with the same mangled name (but different
  // type) before, remove it from its parent.
  if (entry)
    replaceGlobal(entry, gv);

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
      errorNYI(d->getSourceRange(),
               "getOrCreateCIRGlobal: OpenMP target global variable");

    gv.setAlignmentAttr(getSize(astContext.getDeclAlign(d)));

    setLinkageForGV(gv, d);

    if (d->getTLSKind()) {
      if (d->getTLSKind() == VarDecl::TLS_Dynamic)
        errorNYI(d->getSourceRange(), "getOrCreateCIRGlobal: TLS dynamic");
      setTLSMode(gv, *d);
    }

    setGVProperties(gv, d);

    // If required by the ABI, treat declarations of static data members with
    // inline initializers as definitions.
    if (astContext.isMSStaticDataMemberInlineDefinition(d))
      errorNYI(d->getSourceRange(),
               "getOrCreateCIRGlobal: MS static data member inline definition");

    // Emit section information for extern variables.
    if (d->hasExternalStorage()) {
      if (const SectionAttr *sa = d->getAttr<SectionAttr>())
        gv.setSectionAttr(builder.getStringAttr(sa->getName()));
    }
    gv.setGlobalVisibility(getGlobalVisibilityAttrFromDecl(d).getValue());

    // Handle XCore specific ABI requirements.
    if (getTriple().getArch() == llvm::Triple::xcore)
      errorNYI(d->getSourceRange(),
               "getOrCreateCIRGlobal: XCore specific ABI requirements");

    // Check if we a have a const declaration with an initializer, we may be
    // able to emit it as available_externally to expose it's value to the
    // optimizer.
    if (getLangOpts().CPlusPlus && gv.isPublic() &&
        d->getType().isConstQualified() && gv.isDeclaration() &&
        !d->hasDefinition() && d->hasInit() && !d->hasAttr<DLLImportAttr>())
      errorNYI(
          d->getSourceRange(),
          "getOrCreateCIRGlobal: external const declaration with initializer");
  }

  if (d &&
      d->isThisDeclarationADefinition(astContext) == VarDecl::DeclarationOnly) {
    getTargetCIRGenInfo().setTargetAttributes(d, gv, *this);
    // TODO(cir): set target attributes
    // External HIP managed variables needed to be recorded for transformation
    // in both device and host compilations.
    if (getLangOpts().CUDA && d && d->hasAttr<HIPManagedAttr>() &&
        d->hasExternalStorage())
      errorNYI(d->getSourceRange(),
               "getOrCreateCIRGlobal: HIP managed attribute");
  }

  assert(!cir::MissingFeatures::addressSpace());
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
  return getOrCreateCIRGlobal(mangledName, ty, getGlobalVarAddressSpace(d), d,
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

  bool tlsAccess = d->getTLSKind() != VarDecl::TLS_None;
  cir::GlobalOp g = getOrCreateCIRGlobal(d, ty, isForDefinition);
  mlir::Type ptrTy = builder.getPointerTo(g.getSymType(), g.getAddrSpaceAttr());
  return cir::GetGlobalOp::create(
      builder, getLoc(d->getSourceRange()), ptrTy, g.getSymNameAttr(),
      tlsAccess,
      /*static_local=*/g.getStaticLocalGuard().has_value());
}

cir::GlobalViewAttr CIRGenModule::getAddrOfGlobalVarAttr(const VarDecl *d) {
  assert(d->hasGlobalStorage() && "Not a global variable");
  mlir::Type ty = getTypes().convertTypeForMem(d->getType());

  cir::GlobalOp globalOp = getOrCreateCIRGlobal(d, ty, NotForDefinition);
  cir::PointerType ptrTy =
      builder.getPointerTo(globalOp.getSymType(), globalOp.getAddrSpaceAttr());
  return builder.getGlobalViewAttr(ptrTy, globalOp);
}

void CIRGenModule::addUsedGlobal(cir::CIRGlobalValueInterface gv) {
  assert((mlir::isa<cir::FuncOp>(gv.getOperation()) ||
          !gv.isDeclarationForLinker()) &&
         "Only globals with definition can force usage.");
  llvmUsed.emplace_back(gv);
}

void CIRGenModule::addCompilerUsedGlobal(cir::CIRGlobalValueInterface gv) {
  assert(!gv.isDeclarationForLinker() &&
         "Only globals with definition can force usage.");
  llvmCompilerUsed.emplace_back(gv);
}

void CIRGenModule::addUsedOrCompilerUsedGlobal(
    cir::CIRGlobalValueInterface gv) {
  assert((mlir::isa<cir::FuncOp>(gv.getOperation()) ||
          !gv.isDeclarationForLinker()) &&
         "Only globals with definition can force usage.");
  if (getTriple().isOSBinFormatELF())
    llvmCompilerUsed.emplace_back(gv);
  else
    llvmUsed.emplace_back(gv);
}

static void emitUsed(CIRGenModule &cgm, StringRef name,
                     std::vector<cir::CIRGlobalValueInterface> &list) {
  if (list.empty())
    return;

  CIRGenBuilderTy &builder = cgm.getBuilder();
  mlir::Location loc = builder.getUnknownLoc();
  llvm::SmallVector<mlir::Attribute> usedArray;
  usedArray.resize(list.size());
  for (auto [i, op] : llvm::enumerate(list)) {
    usedArray[i] = cir::GlobalViewAttr::get(
        cgm.voidPtrTy, mlir::FlatSymbolRefAttr::get(op.getNameAttr()));
  }

  cir::ArrayType arrayTy = cir::ArrayType::get(cgm.voidPtrTy, usedArray.size());

  cir::ConstArrayAttr initAttr = cir::ConstArrayAttr::get(
      arrayTy, mlir::ArrayAttr::get(&cgm.getMLIRContext(), usedArray));

  cir::GlobalOp gv = CIRGenModule::createGlobalOp(cgm, loc, name, arrayTy,
                                                  /*isConstant=*/false);
  gv.setLinkage(cir::GlobalLinkageKind::AppendingLinkage);
  gv.setInitialValueAttr(initAttr);
  gv.setSectionAttr(builder.getStringAttr("llvm.metadata"));
}

void CIRGenModule::emitLLVMUsed() {
  emitUsed(*this, "llvm.used", llvmUsed);
  emitUsed(*this, "llvm.compiler.used", llvmCompilerUsed);
}

void CIRGenModule::emitGlobalVarDefinition(const clang::VarDecl *vd,
                                           bool isTentative) {
  if (getLangOpts().OpenCL || getLangOpts().OpenMPIsTargetDevice) {
    errorNYI(vd->getSourceRange(),
             "emitGlobalVarDefinition: emit OpenCL/OpenMP global variable");
    return;
  }

  // Whether the definition of the variable is available externally.
  // If yes, we shouldn't emit the GloablCtor and GlobalDtor for the variable
  // since this is the job for its original source.
  bool isDefinitionAvailableExternally =
      astContext.GetGVALinkageForVariable(vd) == GVA_AvailableExternally;

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
  bool needsGlobalCtor = false;
  bool needsGlobalDtor =
      !isDefinitionAvailableExternally &&
      vd->needsDestruction(astContext) == QualType::DK_cxx_destructor;
  const VarDecl *initDecl;
  const Expr *initExpr = vd->getAnyInitializer(initDecl);

  std::optional<ConstantEmitter> emitter;

  // CUDA E.2.4.1 "__shared__ variables cannot have an initialization
  // as part of their declaration."  Sema has already checked for
  // error cases, so we just need to set Init to PoisonValue.
  bool isCUDASharedVar =
      getLangOpts().CUDAIsDevice && vd->hasAttr<CUDASharedAttr>();
  // Shadows of initialized device-side global variables are also left
  // undefined.
  // Managed Variables should be initialized on both host side and device side.
  bool isCUDAShadowVar =
      !getLangOpts().CUDAIsDevice && !vd->hasAttr<HIPManagedAttr>() &&
      (vd->hasAttr<CUDAConstantAttr>() || vd->hasAttr<CUDADeviceAttr>() ||
       vd->hasAttr<CUDASharedAttr>());
  bool isCUDADeviceShadowVar =
      getLangOpts().CUDAIsDevice && !vd->hasAttr<HIPManagedAttr>() &&
      (vd->getType()->isCUDADeviceBuiltinSurfaceType() ||
       vd->getType()->isCUDADeviceBuiltinTextureType());

  if (getLangOpts().CUDA &&
      (isCUDASharedVar || isCUDAShadowVar || isCUDADeviceShadowVar)) {
    init = cir::PoisonAttr::get(convertType(vd->getType()));
  } else if (vd->hasAttr<LoaderUninitializedAttr>()) {
    errorNYI(vd->getSourceRange(),
             "emitGlobalVarDefinition: loader uninitialized attribute");
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
          errorNYI(vd->getSourceRange(),
                   "emitGlobalVarDefinition: flexible array initializer");
        init = builder.getZeroInitAttr(convertType(qt));
        if (!isDefinitionAvailableExternally)
          needsGlobalCtor = true;
      } else {
        errorNYI(vd->getSourceRange(),
                 "emitGlobalVarDefinition: static initializer");
      }
    } else {
      init = initializer;
      // We don't need an initializer, so remove the entry for the delayed
      // initializer position (just in case this entry was delayed) if we
      // also don't need to register a destructor.
      assert(!cir::MissingFeatures::deferredCXXGlobalInit());
    }
  }

  mlir::Type initType;
  if (mlir::isa<mlir::SymbolRefAttr>(init)) {
    errorNYI(
        vd->getSourceRange(),
        "emitGlobalVarDefinition: global initializer is a symbol reference");
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
    errorNYI(vd->getSourceRange(),
             "emitGlobalVarDefinition: global initializer with type mismatch");
    return;
  }

  assert(!cir::MissingFeatures::maybeHandleStaticInExternC());

  if (vd->hasAttr<AnnotateAttr>())
    addGlobalAnnotations(vd, gv);

  // Set CIR's linkage type as appropriate.
  cir::GlobalLinkageKind linkage = getCIRLinkageVarDefinition(vd);

  // CUDA B.2.1 "The __device__ qualifier declares a variable that resides on
  // the device. [...]"
  // CUDA B.2.2 "The __constant__ qualifier, optionally used together with
  // __device__, declares a variable that: [...]
  // Is accessible from all the threads within the grid and from the host
  // through the runtime library (cudaGetSymbolAddress() / cudaGetSymbolSize()
  // / cudaMemcpyToSymbol() / cudaMemcpyFromSymbol())."
  if (langOpts.CUDA) {
    if (langOpts.CUDAIsDevice) {
      // __shared__ variables is not marked as externally initialized,
      // because they must not be initialized.
      if (linkage != cir::GlobalLinkageKind::InternalLinkage &&
          !vd->isConstexpr() && !vd->getType().isConstQualified() &&
          (vd->hasAttr<CUDADeviceAttr>() || vd->hasAttr<CUDAConstantAttr>() ||
           vd->getType()->isCUDADeviceBuiltinSurfaceType() ||
           vd->getType()->isCUDADeviceBuiltinTextureType())) {
        gv->setAttr(cir::CUDAExternallyInitializedAttr::getMnemonic(),
                    cir::CUDAExternallyInitializedAttr::get(&getMLIRContext()));
      }
    } else {
      // TODO(cir):
      // Adjust linkage of shadow variables in host compilation
      // getCUDARuntime().internalizeDeviceSideVar(vd, linkage);
    }
    // TODO(cir):
    // Handle variable registration
    // getCUDARuntime().handleVarRegistration(vd, gv);
  }

  // Set initializer and finalize emission
  CIRGenModule::setInitializer(gv, init);
  if (emitter)
    emitter->finalize(gv);

  // If it is safe to mark the global 'constant', do so now.
  // Use the same logic as classic codegen EmitGlobalVarDefinition.
  gv.setConstant((vd->hasAttr<CUDAConstantAttr>() && langOpts.CUDAIsDevice) ||
                 (!needsGlobalCtor && !needsGlobalDtor &&
                  vd->getType().isConstantStorage(astContext,
                                                  /*ExcludeCtor=*/true,
                                                  /*ExcludeDtor=*/true)));
  // If it is in a read-only section, mark it 'constant'.
  if (const SectionAttr *sa = vd->getAttr<SectionAttr>()) {
    const ASTContext::SectionInfo &si = astContext.SectionInfos[sa->getName()];
    if ((si.SectionFlags & ASTContext::PSF_Write) == 0)
      gv.setConstant(true);
  }

  // Set CIR linkage and DLL storage class.
  gv.setLinkage(linkage);
  // FIXME(cir): setLinkage should likely set MLIR's visibility automatically.
  gv.setVisibility(getMLIRVisibilityFromCIRLinkage(linkage));
  assert(!cir::MissingFeatures::opGlobalDLLImportExport());
  if (linkage == cir::GlobalLinkageKind::CommonLinkage) {
    // common vars aren't constant even if declared const.
    gv.setConstant(false);
    // Tentative definition of global variables may be initialized with
    // non-zero null pointers. In this case they should have weak linkage
    // since common linkage must have zero initializer and must not have
    // explicit section therefore cannot have non-zero initial value.
    std::optional<mlir::Attribute> initializer = gv.getInitialValue();
    if (initializer && !getBuilder().isNullValue(*initializer))
      gv.setLinkage(cir::GlobalLinkageKind::WeakAnyLinkage);
  }

  setNonAliasAttributes(vd, gv);

  assert(!cir::MissingFeatures::opGlobalThreadLocal());

  maybeSetTrivialComdat(*vd, gv);

  // Emit the initializer function if necessary.
  if (needsGlobalCtor || needsGlobalDtor)
    emitCXXGlobalVarDeclInitFunc(vd, gv, needsGlobalCtor);
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
    return builder.getString(str, eltTy, finalSize, /*ensureNullTerm=*/false);
  }

  auto arrayTy = mlir::cast<cir::ArrayType>(convertType(e->getType()));

  auto arrayEltTy = mlir::cast<cir::IntType>(arrayTy.getElementType());

  uint64_t arraySize = arrayTy.getSize();
  unsigned literalSize = e->getLength();
  assert(arraySize == literalSize + 1 &&
         "wide string literal array size must be literal length plus null "
         "terminator");

  // Check if the string is all null bytes before building the vector.
  // In most non-zero cases, this will break out on the first element.
  bool isAllZero = true;
  for (unsigned i = 0; i < literalSize; ++i) {
    if (e->getCodeUnit(i) != 0) {
      isAllZero = false;
      break;
    }
  }

  if (isAllZero)
    return cir::ZeroAttr::get(arrayTy);

  // Otherwise emit a constant array holding the characters.
  SmallVector<mlir::Attribute> elements;
  elements.reserve(arraySize);
  for (unsigned i = 0; i < literalSize; ++i)
    elements.push_back(cir::IntAttr::get(arrayEltTy, e->getCodeUnit(i)));
  // Add null terminator
  elements.push_back(cir::IntAttr::get(arrayEltTy, 0));

  auto elementsAttr = mlir::ArrayAttr::get(&getMLIRContext(), elements);
  return builder.getConstArray(elementsAttr, arrayTy);
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

#ifndef NDEBUG
static bool verifyPointerTypeArgs(mlir::ModuleOp modOp, cir::FuncOp oldF,
                                  cir::FuncOp newF) {
  std::optional<mlir::SymbolTable::UseRange> optionalUseRange =
      oldF.getSymbolUses(modOp);
  if (!optionalUseRange)
    return true;

  for (const mlir::SymbolTable::SymbolUse &u : *optionalUseRange) {
    auto call = mlir::dyn_cast<cir::CallOp>(u.getUser());
    if (!call)
      continue;

    for (auto [argOp, fnArgType] :
         llvm::zip(call.getArgs(), newF.getFunctionType().getInputs())) {
      if (argOp.getType() != fnArgType)
        return false;
    }
  }

  return true;
}
#endif // NDEBUG

void CIRGenModule::applyReplacements() {
  for (auto &i : replacements) {
    StringRef mangledName = i.first;
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

    assert(verifyPointerTypeArgs(theModule, oldF, newF) &&
           "call argument types do not match replacement function");

    // Replace old with new, but keep the old order.
    if (oldF.replaceAllSymbolUses(newF.getSymNameAttr(), theModule).failed())
      llvm_unreachable("internal error, cannot RAUW symbol");
    if (newF) {
      newF->moveBefore(oldF);
      eraseGlobalSymbol(oldF);
      oldF->erase();
    }
  }
}

cir::GlobalOp CIRGenModule::createOrReplaceCXXRuntimeVariable(
    mlir::Location loc, StringRef name, mlir::Type ty,
    cir::GlobalLinkageKind linkage, clang::CharUnits alignment) {
  auto gv = mlir::dyn_cast_or_null<cir::GlobalOp>(getGlobalValue(name));

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

cir::GlobalLinkageKind
CIRGenModule::getCIRLinkageForDeclarator(const DeclaratorDecl *dd,
                                         GVALinkage linkage) {
  if (linkage == GVA_Internal)
    return cir::GlobalLinkageKind::InternalLinkage;

  if (dd->hasAttr<WeakAttr>())
    return cir::GlobalLinkageKind::WeakAnyLinkage;

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
                                 getCodeGenOpts().NoCommon))
    return cir::GlobalLinkageKind::CommonLinkage;

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
      // The GetGlobal was emitted with the no-proto FuncType. Uses of this
      // operation (cir.store, cir.cast) were built for that pointer type. When
      // we re-type the result to the real FuncType, we need to add a bit the
      // old pointer type so those uses are still valid. This can lead to
      // some redundant bitcast chains, but those will be cleaned up by the
      // canonicalizer.
      mlir::Value res = getGlobalOp.getAddr();
      const mlir::Type oldResTy = res.getType();
      const auto newPtrTy = cir::PointerType::get(newFn.getFunctionType());
      if (oldResTy != newPtrTy) {
        res.setType(newPtrTy);
        builder.setInsertionPointAfter(getGlobalOp.getOperation());
        mlir::Value castRes =
            cir::CastOp::create(builder, getGlobalOp.getLoc(), oldResTy,
                                cir::CastKind::bitcast, res);
        res.replaceAllUsesExcept(castRes, castRes.getDefiningOp());
      }
    } else if (mlir::isa<cir::GlobalOp>(use.getUser())) {
      // Function addresses in global initializers use GlobalViewAttrs typed to
      // the initializer context (e.g. struct field type), not the FuncOp type,
      // so no update is required when the no-proto FuncOp is replaced.
    } else {
      llvm_unreachable(
          "replaceUsesOfNonProtoTypeWithRealFunction: unexpected use type");
    }
  }
}

cir::GlobalLinkageKind
CIRGenModule::getCIRLinkageVarDefinition(const VarDecl *vd) {
  GVALinkage linkage = astContext.GetGVALinkageForVariable(vd);
  return getCIRLinkageForDeclarator(vd, linkage);
}

cir::GlobalLinkageKind CIRGenModule::getFunctionLinkage(GlobalDecl gd) {
  const auto *d = cast<FunctionDecl>(gd.getDecl());

  GVALinkage linkage = astContext.GetGVALinkageForFunction(d);

  if (const auto *dtor = dyn_cast<CXXDestructorDecl>(d))
    return getCXXABI().getCXXDestructorLinkage(linkage, dtor, gd.getDtorType());

  return getCIRLinkageForDeclarator(d, linkage);
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
  assert(!getGlobalValue(result));
  return result;
}

/// Return a pointer to a constant array for the given string literal.
cir::GlobalOp CIRGenModule::getGlobalForStringLiteral(const StringLiteral *s,
                                                      StringRef name) {
  CharUnits alignment =
      astContext.getAlignOfGlobalVarInChars(s->getType(), /*VD=*/nullptr);

  mlir::Attribute c = getConstantArrayFromStringLiteral(s);

  cir::GlobalOp gv;
  if (!getLangOpts().WritableStrings && constantStringMap.count(c)) {
    gv = constantStringMap[c];
    // The bigger alignment always wins.
    if (!gv.getAlignment() ||
        uint64_t(alignment.getQuantity()) > *gv.getAlignment())
      gv.setAlignmentAttr(getSize(alignment));
  } else {
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
    // Synthetic string literals (e.g., from SourceLocExpr) may not have valid
    // source locations. Use unknown location in those cases.
    mlir::Location loc = s->getBeginLoc().isValid()
                             ? getLoc(s->getSourceRange())
                             : builder.getUnknownLoc();
    auto typedC = llvm::cast<mlir::TypedAttr>(c);
    gv = generateStringLiteral(loc, typedC,
                               cir::GlobalLinkageKind::PrivateLinkage, *this,
                               uniqueName, alignment);
    setDSOLocal(static_cast<mlir::Operation *>(gv));
    constantStringMap[c] = gv;

    assert(!cir::MissingFeatures::sanitizers());
  }
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

// TODO(cir): this could be a common AST helper for both CIR and LLVM codegen.
LangAS CIRGenModule::getLangTempAllocaAddressSpace() const {
  if (getLangOpts().OpenCL)
    return LangAS::opencl_private;

  // For temporaries inside functions, CUDA treats them as normal variables.
  // LangAS::cuda_device, on the other hand, is reserved for those variables
  // explicitly marked with __device__.
  if (getLangOpts().CUDAIsDevice)
    return LangAS::Default;

  if (getLangOpts().SYCLIsDevice ||
      (getLangOpts().OpenMP && getLangOpts().OpenMPIsTargetDevice))
    errorNYI("SYCL or OpenMP temp address space");
  return LangAS::Default;
}

void CIRGenModule::emitExplicitCastExprType(const ExplicitCastExpr *e,
                                            CIRGenFunction *cgf) {
  if (cgf && e->getType()->isVariablyModifiedType())
    cgf->emitVariablyModifiedType(e->getType());

  assert(!cir::MissingFeatures::generateDebugInfo() &&
         "emitExplicitCastExprType");
}

mlir::TypedAttr CIRGenModule::emitNullMemberAttr(QualType destTy,
                                                 const MemberPointerType *mpt) {
  if (mpt->isMemberFunctionPointerType()) {
    auto ty = mlir::cast<cir::MethodType>(convertType(destTy));
    return builder.getNullMethodAttr(ty);
  }

  auto ty = mlir::cast<cir::DataMemberType>(convertType(destTy));
  return builder.getNullDataMemberAttr(ty);
}

mlir::Value CIRGenModule::emitMemberPointerConstant(const UnaryOperator *e) {
  assert(!cir::MissingFeatures::cxxABI());

  mlir::Location loc = getLoc(e->getSourceRange());

  const auto *decl = cast<DeclRefExpr>(e->getSubExpr())->getDecl();

  // A member function pointer.
  if (const auto *methodDecl = dyn_cast<CXXMethodDecl>(decl)) {
    auto ty = mlir::cast<cir::MethodType>(convertType(e->getType()));
    if (methodDecl->isVirtual())
      return cir::ConstantOp::create(
          builder, loc, getCXXABI().buildVirtualMethodAttr(ty, methodDecl));

    const CIRGenFunctionInfo &fi =
        getTypes().arrangeCXXMethodDeclaration(methodDecl);
    cir::FuncType funcTy = getTypes().getFunctionType(fi);
    cir::FuncOp methodFuncOp = getAddrOfFunction(methodDecl, funcTy);
    return cir::ConstantOp::create(builder, loc,
                                   builder.getMethodAttr(ty, methodFuncOp));
  }

  // Otherwise, a member data pointer.
  auto ty = mlir::cast<cir::DataMemberType>(convertType(e->getType()));
  const auto *fieldDecl = cast<FieldDecl>(decl);
  return cir::ConstantOp::create(
      builder, loc, builder.getDataMemberAttr(ty, fieldDecl->getFieldIndex()));
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
  case Decl::Export:
    emitDeclContext(cast<ExportDecl>(decl));
    break;

  case Decl::Var:
  case Decl::Decomposition:
  case Decl::VarTemplateSpecialization: {
    emitGlobal(cast<VarDecl>(decl));
    if (auto *decomp = dyn_cast<DecompositionDecl>(decl))
      for (auto *binding : decomp->flat_bindings())
        if (auto *holdingVar = binding->getHoldingVar())
          emitGlobal(holdingVar);
    break;
  }
  case Decl::OpenACCRoutine:
    emitGlobalOpenACCRoutineDecl(cast<OpenACCRoutineDecl>(decl));
    break;
  case Decl::OpenACCDeclare:
    emitGlobalOpenACCDeclareDecl(cast<OpenACCDeclareDecl>(decl));
    break;
  case Decl::OMPThreadPrivate:
    emitOMPThreadPrivateDecl(cast<OMPThreadPrivateDecl>(decl));
    break;
  case Decl::OMPGroupPrivate:
    emitOMPGroupPrivateDecl(cast<OMPGroupPrivateDecl>(decl));
    break;
  case Decl::OMPAllocate:
    emitOMPAllocateDecl(cast<OMPAllocateDecl>(decl));
    break;
  case Decl::OMPCapturedExpr:
    emitOMPCapturedExpr(cast<OMPCapturedExprDecl>(decl));
    break;
  case Decl::OMPDeclareReduction:
    emitOMPDeclareReduction(cast<OMPDeclareReductionDecl>(decl));
    break;
  case Decl::OMPDeclareMapper:
    emitOMPDeclareMapper(cast<OMPDeclareMapperDecl>(decl));
    break;
  case Decl::OMPRequires:
    emitOMPRequiresDecl(cast<OMPRequiresDecl>(decl));
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
  case Decl::ExplicitInstantiation:
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
  case Decl::CXXRecord: {
    CXXRecordDecl *crd = cast<CXXRecordDecl>(decl);
    assert(!cir::MissingFeatures::generateDebugInfo());
    for (auto *childDecl : crd->decls())
      if (isa<VarDecl, CXXRecordDecl, EnumDecl, OpenACCDeclareDecl>(childDecl))
        emitTopLevelDecl(childDecl);
    break;
  }

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
  // Returns kernel handle for HIP kernel stub function.
  if (langOpts.CUDA && !langOpts.CUDAIsDevice &&
      cast<FunctionDecl>(gd.getDecl())->hasAttr<CUDAGlobalAttr>()) {
    mlir::Operation *handle = getCUDARuntime().getKernelHandle(func, gd);

    // For HIP the kernel handle is a GlobalOp, which cannot be cast to
    // FuncOp. Return the stub directly in that case.
    bool isHIPHandle = mlir::isa<cir::GlobalOp>(*handle);
    if (isForDefinition || isHIPHandle)
      return func;
    return mlir::dyn_cast<cir::FuncOp>(*handle);
  }
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
      out << "__device_stub__" << ii->getName();
    } else if (fd &&
               DeviceKernelAttr::isOpenCLSpelling(
                   fd->getAttr<DeviceKernelAttr>()) &&
               gd.getKernelReferenceKind() == KernelReferenceKind::Stub) {
      cgm.errorNYI(nd->getSourceRange(), "getMangledName: OpenCL Stub");
    } else {
      out << ii->getName();
    }
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

static FunctionDecl *
createOpenACCBindTempFunction(ASTContext &ctx, const IdentifierInfo *bindName,
                              const FunctionDecl *protoFunc) {
  // If this is a C no-prototype function, we can take the 'easy' way out and
  // just create a function with no arguments/functions, etc.
  if (!protoFunc->hasPrototype())
    return FunctionDecl::Create(
        ctx, /*DC=*/ctx.getTranslationUnitDecl(),
        /*StartLoc=*/SourceLocation{}, /*NLoc=*/SourceLocation{}, bindName,
        protoFunc->getType(), /*TInfo=*/nullptr, StorageClass::SC_None);

  QualType funcTy = protoFunc->getType();
  auto *fpt = cast<FunctionProtoType>(protoFunc->getType());

  // If this is a member function, add an explicit 'this' to the function type.
  if (auto *methodDecl = dyn_cast<CXXMethodDecl>(protoFunc);
      methodDecl && methodDecl->isImplicitObjectMemberFunction()) {
    llvm::SmallVector<QualType> paramTypes{fpt->getParamTypes()};
    paramTypes.insert(paramTypes.begin(), methodDecl->getThisType());

    funcTy = ctx.getFunctionType(fpt->getReturnType(), paramTypes,
                                 fpt->getExtProtoInfo());
    fpt = cast<FunctionProtoType>(funcTy);
  }

  auto *tempFunc =
      FunctionDecl::Create(ctx, /*DC=*/ctx.getTranslationUnitDecl(),
                           /*StartLoc=*/SourceLocation{},
                           /*NLoc=*/SourceLocation{}, bindName, funcTy,
                           /*TInfo=*/nullptr, StorageClass::SC_None);

  SmallVector<ParmVarDecl *> params;
  params.reserve(fpt->getNumParams());

  // Add all of the parameters.
  for (unsigned i = 0, e = fpt->getNumParams(); i != e; ++i) {
    ParmVarDecl *parm = ParmVarDecl::Create(
        ctx, tempFunc, /*StartLoc=*/SourceLocation{},
        /*IdLoc=*/SourceLocation{},
        /*Id=*/nullptr, fpt->getParamType(i), /*TInfo=*/nullptr,
        StorageClass::SC_None, /*DefArg=*/nullptr);
    parm->setScopeInfo(0, i);
    params.push_back(parm);
  }

  tempFunc->setParams(params);

  return tempFunc;
}

std::string
CIRGenModule::getOpenACCBindMangledName(const IdentifierInfo *bindName,
                                        const FunctionDecl *attachedFunction) {
  FunctionDecl *tempFunc = createOpenACCBindTempFunction(
      getASTContext(), bindName, attachedFunction);

  std::string ret = getMangledNameImpl(*this, GlobalDecl(tempFunc), tempFunc);

  // This does nothing (it is a do-nothing function), since this is a
  // slab-allocator, but leave a call in to immediately destroy this in case we
  // ever come up with a way of getting allocations back.
  getASTContext().Deallocate(tempFunc);
  return ret;
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

bool CIRGenModule::lookupRepresentativeDecl(StringRef mangledName,
                                            GlobalDecl &result) const {
  auto res = manglings.find(mangledName);
  if (res == manglings.end())
    return false;
  result = res->getValue();
  return true;
}

cir::TLS_Model CIRGenModule::getDefaultCIRTLSModel() const {
  switch (getCodeGenOpts().getDefaultTLSModel()) {
  case CodeGenOptions::GeneralDynamicTLSModel:
    return cir::TLS_Model::GeneralDynamic;
  case CodeGenOptions::LocalDynamicTLSModel:
    return cir::TLS_Model::LocalDynamic;
  case CodeGenOptions::InitialExecTLSModel:
    return cir::TLS_Model::InitialExec;
  case CodeGenOptions::LocalExecTLSModel:
    return cir::TLS_Model::LocalExec;
  }
  llvm_unreachable("Invalid TLS model!");
}

void CIRGenModule::setTLSMode(mlir::Operation *op, const VarDecl &d) {
  assert(d.getTLSKind() && "setting TLS mode on non-TLS var!");

  cir::TLS_Model tlm = getDefaultCIRTLSModel();

  // Override the TLS model if it is explicitly specified.
  if (d.getAttr<TLSModelAttr>())
    errorNYI(d.getSourceRange(), "TLS model attribute");

  auto global = cast<cir::GlobalOp>(op);
  global.setTlsModel(tlm);
}

void CIRGenModule::setCIRFunctionAttributes(GlobalDecl globalDecl,
                                            const CIRGenFunctionInfo &info,
                                            cir::FuncOp func, bool isThunk) {
  // TODO(cir): More logic of constructAttributeList is needed.
  cir::CallingConv callingConv;
  cir::SideEffect sideEffect;

  // TODO(cir): The current list should be initialized with the extra function
  // attributes, but we don't have those yet.  For now, the PAL is initialized
  // with nothing.
  assert(!cir::MissingFeatures::opFuncExtraAttrs());
  // Initialize PAL with existing attributes to merge attributes.
  mlir::NamedAttrList pal{};
  std::vector<mlir::NamedAttrList> argAttrs(info.arguments().size());
  mlir::NamedAttrList retAttrs{};
  constructAttributeList(func.getName(), info, globalDecl, pal, argAttrs,
                         retAttrs, callingConv, sideEffect,
                         /*attrOnCallSite=*/false, isThunk);

  for (mlir::NamedAttribute attr : pal)
    func->setAttr(attr.getName(), attr.getValue());

  llvm::for_each(llvm::enumerate(argAttrs), [func](auto idx_arg_pair) {
    mlir::function_interface_impl::setArgAttrs(func, idx_arg_pair.index(),
                                               idx_arg_pair.value());
  });
  if (!retAttrs.empty())
    mlir::function_interface_impl::setResultAttrs(func, 0, retAttrs);

  // TODO(cir): Check X86_VectorCall incompatibility wiht WinARM64EC

  // TODO(cir): Set the calling convention computed by constructAttributeList
  // on the function. FuncOp supports calling_conv, but target-specific
  // CodeGen is needed to set it correctly (e.g., AMDGPU kernel functions
  // should be marked with AMDGPUKernel).
  assert(!cir::MissingFeatures::opFuncCallingConv());
}

void CIRGenModule::setFunctionAttributes(GlobalDecl globalDecl,
                                         cir::FuncOp func,
                                         bool isIncompleteFunction,
                                         bool isThunk) {
  // NOTE(cir): Original CodeGen checks if this is an intrinsic. In CIR we
  // represent them in dedicated ops. The correct attributes are ensured during
  // translation to LLVM. Thus, we don't need to check for them here.

  const auto *funcDecl = cast<FunctionDecl>(globalDecl.getDecl());

  if (!isIncompleteFunction)
    setCIRFunctionAttributes(globalDecl,
                             getTypes().arrangeGlobalDeclaration(globalDecl),
                             func, isThunk);

  if (!isIncompleteFunction && func.isDeclaration())
    getTargetCIRGenInfo().setTargetAttributes(funcDecl, func, *this);

  // TODO(cir): This needs a lot of work to better match CodeGen. That
  // ultimately ends up in setGlobalVisibility, which already has the linkage of
  // the LLVM GV (corresponding to our FuncOp) computed, so it doesn't have to
  // recompute it here. This is a minimal fix for now.
  if (!isLocalLinkage(getFunctionLinkage(globalDecl))) {
    const Decl *decl = globalDecl.getDecl();
    func.setGlobalVisibility(getGlobalVisibilityAttrFromDecl(decl).getValue());
  }

  // If we plan on emitting this inline builtin, we can't treat it as a builtin.
  if (funcDecl->isInlineBuiltinDeclaration()) {
    const FunctionDecl *fdBody;
    bool hasBody = funcDecl->hasBody(fdBody);
    (void)hasBody;
    assert(hasBody && "Inline builtin declarations should always have an "
                      "available body!");
    assert(!cir::MissingFeatures::attributeNoBuiltin());
  }

  if (funcDecl->isReplaceableGlobalAllocationFunction()) {
    // A replaceable global allocation function does not act like a builtin by
    // default, only if it is invoked by a new-expression or delete-expression.
    func->setAttr(cir::CIRDialect::getNoBuiltinAttrName(),
                  mlir::UnitAttr::get(&getMLIRContext()));
  }
}

/// Determines whether the language options require us to model
/// unwind exceptions.  We treat -fexceptions as mandating this
/// except under the fragile ObjC ABI with only ObjC exceptions
/// enabled.  This means, for example, that C with -fexceptions
/// enables this.
static bool hasUnwindExceptions(const LangOptions &langOpts) {
  // If exceptions are completely disabled, obviously this is false.
  if (!langOpts.Exceptions)
    return false;
  // If C++ exceptions are enabled, this is true.
  if (langOpts.CXXExceptions)
    return true;
  // If ObjC exceptions are enabled, this depends on the ABI.
  if (langOpts.ObjCExceptions)
    return langOpts.ObjCRuntime.hasUnwindExceptions();
  return true;
}

void CIRGenModule::setCIRFunctionAttributesForDefinition(
    const clang::FunctionDecl *decl, cir::FuncOp f) {
  assert(!cir::MissingFeatures::opFuncUnwindTablesAttr());
  assert(!cir::MissingFeatures::stackProtector());

  if (!hasUnwindExceptions(langOpts))
    f->setAttr(cir::CIRDialect::getNoThrowAttrName(),
               mlir::UnitAttr::get(&getMLIRContext()));

  std::optional<cir::InlineKind> existingInlineKind = f.getInlineKind();
  bool isNoInline =
      existingInlineKind && *existingInlineKind == cir::InlineKind::NoInline;
  bool isAlwaysInline = existingInlineKind &&
                        *existingInlineKind == cir::InlineKind::AlwaysInline;
  if (!decl) {
    assert(!cir::MissingFeatures::hlsl());

    if (!isAlwaysInline &&
        codeGenOpts.getInlining() == CodeGenOptions::OnlyAlwaysInlining) {
      // If inlining is disabled and we don't have a declaration to control
      // inlining, mark the function as 'noinline' unless it is explicitly
      // marked as 'alwaysinline'.
      f.setInlineKind(cir::InlineKind::NoInline);
    }

    return;
  }

  assert(!cir::MissingFeatures::opFuncArmStreamingAttr());
  assert(!cir::MissingFeatures::opFuncArmNewAttr());
  assert(!cir::MissingFeatures::opFuncOptNoneAttr());
  assert(!cir::MissingFeatures::opFuncMinSizeAttr());
  assert(!cir::MissingFeatures::opFuncNakedAttr());
  assert(!cir::MissingFeatures::opFuncNoDuplicateAttr());
  assert(!cir::MissingFeatures::hlsl());

  // Handle inline attributes
  if (decl->hasAttr<NoInlineAttr>() && !isAlwaysInline) {
    // Add noinline if the function isn't always_inline.
    f.setInlineKind(cir::InlineKind::NoInline);
  } else if (decl->hasAttr<AlwaysInlineAttr>() && !isNoInline) {
    // Don't override AlwaysInline with NoInline, or vice versa, since we can't
    // specify both in IR.
    f.setInlineKind(cir::InlineKind::AlwaysInline);
  } else if (codeGenOpts.getInlining() == CodeGenOptions::OnlyAlwaysInlining) {
    // If inlining is disabled, force everything that isn't always_inline
    // to carry an explicit noinline attribute.
    if (!isAlwaysInline)
      f.setInlineKind(cir::InlineKind::NoInline);
  } else {
    // Otherwise, propagate the inline hint attribute and potentially use its
    // absence to mark things as noinline.
    // Search function and template pattern redeclarations for inline.
    if (auto *fd = dyn_cast<FunctionDecl>(decl)) {
      // TODO: Share this checkForInline implementation with classic codegen.
      // This logic is likely to change over time, so sharing would help ensure
      // consistency.
      auto checkForInline = [](const FunctionDecl *decl) {
        auto checkRedeclForInline = [](const FunctionDecl *redecl) {
          return redecl->isInlineSpecified();
        };
        if (any_of(decl->redecls(), checkRedeclForInline))
          return true;
        const FunctionDecl *pattern = decl->getTemplateInstantiationPattern();
        if (!pattern)
          return false;
        return any_of(pattern->redecls(), checkRedeclForInline);
      };
      if (checkForInline(fd)) {
        f.setInlineKind(cir::InlineKind::InlineHint);
      } else if (codeGenOpts.getInlining() ==
                     CodeGenOptions::OnlyHintInlining &&
                 !fd->isInlined() && !isAlwaysInline) {
        f.setInlineKind(cir::InlineKind::NoInline);
      }
    }
  }

  assert(!cir::MissingFeatures::opFuncColdHotAttr());
}

cir::FuncOp CIRGenModule::getOrCreateCIRFunction(
    StringRef mangledName, mlir::Type funcType, GlobalDecl gd, bool forVTable,
    bool dontDefer, bool isThunk, ForDefinition_t isForDefinition,
    mlir::NamedAttrList extraAttrs) {
  const Decl *d = gd.getDecl();

  if (const auto *fd = cast_or_null<FunctionDecl>(d)) {
    // For the device, mark the function as one that should be emitted.
    if (getLangOpts().OpenMPIsTargetDevice && openMPRuntime &&
        !getOpenMPRuntime().markAsGlobalTarget(gd) && fd->isDefined() &&
        !dontDefer && !isForDefinition) {
      if (const FunctionDecl *fdDef = fd->getDefinition()) {
        GlobalDecl gdDef;
        if (const auto *cd = dyn_cast<CXXConstructorDecl>(fdDef))
          gdDef = GlobalDecl(cd, gd.getCtorType());
        else if (const auto *dd = dyn_cast<CXXDestructorDecl>(fdDef))
          gdDef = GlobalDecl(dd, gd.getDtorType());
        else
          gdDef = GlobalDecl(fdDef);
        emitGlobal(gdDef);
      }
    }

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
      GlobalDecl otherGd;
      // Check that GD is not yet in DiagnosedConflictingDefinitions is required
      // to make sure that we issue an error only once.
      if (lookupRepresentativeDecl(mangledName, otherGd) &&
          (gd.getCanonicalDecl().getDecl() !=
           otherGd.getCanonicalDecl().getDecl()) &&
          diagnosedConflictingDefinitions.insert(gd).second) {
        getDiags().Report(d->getLocation(), diag::err_duplicate_mangled_name)
            << mangledName;
        getDiags().Report(otherGd.getDecl()->getLocation(),
                          diag::note_previous_definition);
      }
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

  if (funcDecl && funcDecl->hasAttr<AnnotateAttr>())
    deferredAnnotations[mangledName] = funcDecl;

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
    eraseGlobalSymbol(entry);
    entry->erase();
  }

  if (d)
    setFunctionAttributes(gd, funcOp, /*isIncompleteFunction=*/false, isThunk);
  if (!extraAttrs.empty()) {
    extraAttrs.append(funcOp->getAttrs());
    funcOp->setAttrs(extraAttrs);
  }

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

    func = cir::FuncOp::create(builder, loc, name, funcType);

    symbolLookupCache[func.getSymNameAttr()] = func;

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

    // Mark C++ special member functions (Constructor, Destructor etc.)
    setCXXSpecialMemberAttr(func, funcDecl);

    if (!cgf)
      theModule.push_back(func);

    if (this->getLangOpts().OpenACC) {
      // We only have to handle this attribute, since OpenACCAnnotAttrs are
      // handled via the end-of-TU work.
      for (const auto *attr :
           funcDecl->specific_attrs<OpenACCRoutineDeclAttr>())
        emitOpenACCRoutineDecl(funcDecl, func, attr->getLocation(),
                               attr->Clauses);
    }
  }
  return func;
}

cir::FuncOp
CIRGenModule::createCIRBuiltinFunction(mlir::Location loc, StringRef name,
                                       cir::FuncType ty,
                                       const clang::FunctionDecl *fd) {
  cir::FuncOp fnOp = createCIRFunction(loc, name, ty, fd);
  fnOp.setBuiltin(true);
  return fnOp;
}

static cir::CtorKind getCtorKindFromDecl(const CXXConstructorDecl *ctor) {
  if (ctor->isDefaultConstructor())
    return cir::CtorKind::Default;
  if (ctor->isCopyConstructor())
    return cir::CtorKind::Copy;
  if (ctor->isMoveConstructor())
    return cir::CtorKind::Move;
  return cir::CtorKind::Custom;
}

static cir::AssignKind getAssignKindFromDecl(const CXXMethodDecl *method) {
  if (method->isCopyAssignmentOperator())
    return cir::AssignKind::Copy;
  if (method->isMoveAssignmentOperator())
    return cir::AssignKind::Move;
  llvm_unreachable("not a copy or move assignment operator");
}

void CIRGenModule::setCXXSpecialMemberAttr(
    cir::FuncOp funcOp, const clang::FunctionDecl *funcDecl) {
  if (!funcDecl)
    return;

  if (const auto *dtor = dyn_cast<CXXDestructorDecl>(funcDecl)) {
    auto cxxDtor = cir::CXXDtorAttr::get(
        convertType(getASTContext().getCanonicalTagType(dtor->getParent())),
        dtor->isTrivial());
    funcOp.setCxxSpecialMemberAttr(cxxDtor);
    return;
  }

  if (const auto *ctor = dyn_cast<CXXConstructorDecl>(funcDecl)) {
    cir::CtorKind kind = getCtorKindFromDecl(ctor);
    auto cxxCtor = cir::CXXCtorAttr::get(
        convertType(getASTContext().getCanonicalTagType(ctor->getParent())),
        kind, ctor->isTrivial());
    funcOp.setCxxSpecialMemberAttr(cxxCtor);
    return;
  }

  const auto *method = dyn_cast<CXXMethodDecl>(funcDecl);
  if (method && (method->isCopyAssignmentOperator() ||
                 method->isMoveAssignmentOperator())) {
    cir::AssignKind assignKind = getAssignKindFromDecl(method);
    auto cxxAssign = cir::CXXAssignAttr::get(
        convertType(getASTContext().getCanonicalTagType(method->getParent())),
        assignKind, method->isTrivial());
    funcOp.setCxxSpecialMemberAttr(cxxAssign);
    return;
  }
}

static void setWindowsItaniumDLLImport(CIRGenModule &cgm, bool isLocal,
                                       cir::FuncOp funcOp, StringRef name) {
  // In Windows Itanium environments, try to mark runtime functions
  // dllimport. For Mingw and MSVC, don't. We don't really know if the user
  // will link their standard library statically or dynamically. Marking
  // functions imported when they are not imported can cause linker errors
  // and warnings.
  if (!isLocal && cgm.getTarget().getTriple().isWindowsItaniumEnvironment() &&
      !cgm.getCodeGenOpts().LTOVisibilityPublicStd) {
    assert(!cir::MissingFeatures::getRuntimeFunctionDecl());
    assert(!cir::MissingFeatures::setDLLStorageClass());
    assert(!cir::MissingFeatures::opGlobalDLLImportExport());
  }
}

cir::FuncOp CIRGenModule::createRuntimeFunction(cir::FuncType ty,
                                                StringRef name,
                                                mlir::NamedAttrList extraAttrs,
                                                bool isLocal,
                                                bool assumeConvergent) {
  if (assumeConvergent)
    errorNYI("createRuntimeFunction: assumeConvergent");

  cir::FuncOp entry = getOrCreateCIRFunction(name, ty, GlobalDecl(),
                                             /*forVtable=*/false, extraAttrs);

  if (entry) {
    // TODO(cir): set the attributes of the function.
    assert(!cir::MissingFeatures::setLLVMFunctionFEnvAttributes());
    assert(!cir::MissingFeatures::opFuncCallingConv());
    setWindowsItaniumDLLImport(*this, isLocal, entry, name);
    entry.setDSOLocal(true);
  }

  return entry;
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
  emitVTablesOpportunistically();
  applyReplacements();

  theModule->setAttr(cir::CIRDialect::getModuleLevelAsmAttrName(),
                     builder.getArrayAttr(globalScopeAsm));

  emitGlobalAnnotations();

  if (!recordLayoutEntries.empty())
    theModule->setAttr(
        cir::CIRDialect::getRecordLayoutsAttrName(),
        mlir::DictionaryAttr::get(&getMLIRContext(), recordLayoutEntries));

  if (getTriple().isAMDGPU() ||
      (getTriple().isSPIRV() && getTriple().getVendor() == llvm::Triple::AMD))
    emitAMDGPUMetadata();

  if (getLangOpts().HIP) {
    // Emit a unique ID so that host and device binaries from the same
    // compilation unit can be associated.
    std::string cuidName =
        ("__hip_cuid_" + getASTContext().getCUIDHash()).str();
    auto int8Ty = cir::IntType::get(&getMLIRContext(), 8, /*isSigned=*/false);
    auto loc = builder.getUnknownLoc();
    mlir::ptr::MemorySpaceAttrInterface addrSpace =
        cir::LangAddressSpaceAttr::get(&getMLIRContext(),
                                       getGlobalVarAddressSpace(nullptr));

    auto gv = createGlobalOp(*this, loc, cuidName, int8Ty,
                             /*isConstant=*/false, addrSpace);
    gv.setLinkage(cir::GlobalLinkageKind::ExternalLinkage);
    // Initialize with zero
    auto zeroAttr = cir::IntAttr::get(int8Ty, 0);
    gv.setInitialValueAttr(zeroAttr);
    // External linkage requires public visibility
    mlir::SymbolTable::setSymbolVisibility(
        gv, mlir::SymbolTable::Visibility::Public);

    addCompilerUsedGlobal(gv);
  }

  emitLLVMUsed();

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

  if (op) {
    // Any existing users of the existing function declaration will be
    // referencing the function by flat symbol reference (i.e. the name), so
    // those uses will automatically resolve to the alias now that we've
    // replaced the function declaration. We can safely erase the existing
    // function declaration.
    assert(cast<cir::FuncOp>(op).getFunctionType() == alias.getFunctionType() &&
           "declaration exists with different type");
    eraseGlobalSymbol(op);
    op->erase();
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

  if (forEh && ty->isObjCObjectPointerType() &&
      langOpts.ObjCRuntime.isGNUFamily()) {
    errorNYI(loc, "getAddrOfRTTIDescriptor: Objc PtrType & Objc RT GUN");
    return {};
  }

  return getCXXABI().getAddrOfRTTIDescriptor(loc, ty);
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

void CIRGenModule::error(SourceLocation loc, StringRef error) {
  unsigned diagID = getDiags().getCustomDiagID(DiagnosticsEngine::Error, "%0");
  getDiags().Report(astContext.getFullLoc(loc), diagID) << error;
}

/// Print out an error that codegen doesn't support the specified stmt yet.
void CIRGenModule::errorUnsupported(const Stmt *s, llvm::StringRef type) {
  unsigned diagId = diags.getCustomDiagID(DiagnosticsEngine::Error,
                                          "cannot compile this %0 yet");
  diags.Report(astContext.getFullLoc(s->getBeginLoc()), diagId)
      << type << s->getSourceRange();
}

/// Print out an error that codegen doesn't support the specified decl yet.
void CIRGenModule::errorUnsupported(const Decl *d, llvm::StringRef type) {
  unsigned diagId = diags.getCustomDiagID(DiagnosticsEngine::Error,
                                          "cannot compile this %0 yet");
  diags.Report(astContext.getFullLoc(d->getLocation()), diagId) << type;
}

void CIRGenModule::mapBlockAddress(cir::BlockAddrInfoAttr blockInfo,
                                   cir::LabelOp label) {
  [[maybe_unused]] auto result =
      blockAddressInfoToLabel.try_emplace(blockInfo, label);
  assert(result.second &&
         "attempting to map a blockaddress info that is already mapped");
}

void CIRGenModule::mapUnresolvedBlockAddress(cir::BlockAddressOp op) {
  [[maybe_unused]] auto result = unresolvedBlockAddressToLabel.insert(op);
  assert(result.second &&
         "attempting to map a blockaddress operation that is already mapped");
}

void CIRGenModule::mapResolvedBlockAddress(cir::BlockAddressOp op,
                                           cir::LabelOp label) {
  [[maybe_unused]] auto result = blockAddressToLabel.try_emplace(op, label);
  assert(result.second &&
         "attempting to map a blockaddress operation that is already mapped");
}

void CIRGenModule::updateResolvedBlockAddress(cir::BlockAddressOp op,
                                              cir::LabelOp newLabel) {
  auto *it = blockAddressToLabel.find(op);
  assert(it != blockAddressToLabel.end() &&
         "trying to update a blockaddress not previously mapped");
  assert(!it->second && "blockaddress already has a resolved label");

  it->second = newLabel;
}

cir::LabelOp
CIRGenModule::lookupBlockAddressInfo(cir::BlockAddrInfoAttr blockInfo) {
  return blockAddressInfoToLabel.lookup(blockInfo);
}

mlir::Operation *
CIRGenModule::getAddrOfGlobalTemporary(const MaterializeTemporaryExpr *mte,
                                       const Expr *init) {
  assert((mte->getStorageDuration() == SD_Static ||
          mte->getStorageDuration() == SD_Thread) &&
         "not a global temporary");
  const auto *varDecl = cast<VarDecl>(mte->getExtendingDecl());

  // Use the MaterializeTemporaryExpr's type if it has the same unqualified
  // base type as Init. This preserves cv-qualifiers (e.g. const from a
  // constexpr or const-ref binding) that skipRValueSubobjectAdjustments may
  // have dropped via NoOp casts, while correctly falling back to Init's type
  // when a real subobject adjustment changed the type (e.g. member access or
  // base-class cast in C++98), where E->getType() reflects the reference type,
  // not the actual storage type.
  QualType materializedType = init->getType();
  if (getASTContext().hasSameUnqualifiedType(mte->getType(), materializedType))
    materializedType = mte->getType();

  CharUnits align = getASTContext().getTypeAlignInChars(materializedType);

  auto insertResult = materializedGlobalTemporaryMap.insert({mte, nullptr});
  if (!insertResult.second)
    errorNYI(mte->getSourceRange(), "duplicate materialized temporaries");

  // FIXME: If an externally-visible declaration extends multiple temporaries,
  // we need to give each temporary the same name in every translation unit (and
  // we also need to make the temporaries externally-visible).
  llvm::SmallString<256> name;
  llvm::raw_svector_ostream out(name);
  getCXXABI().getMangleContext().mangleReferenceTemporary(
      varDecl, mte->getManglingNumber(), out);

  APValue *value = nullptr;
  if (mte->getStorageDuration() == SD_Static && varDecl->evaluateValue()) {
    // If the initializer of the extending declaration is a constant
    // initializer, we should have a cached constant initializer for this
    // temporay. Note taht this m ight have a different value from the value
    // computed by evaluating the initializer if the surrounding constant
    // expression modifies the temporary.
    value = mte->getOrCreateValue(/*MayCreate=*/false);
  }

  // Try evaluating it now, it might have a constant initializer
  Expr::EvalResult evalResult;
  if (!value && init->EvaluateAsRValue(evalResult, getASTContext()) &&
      !evalResult.hasSideEffects())
    value = &evalResult.Val;

  assert(!cir::MissingFeatures::addressSpace());

  std::optional<ConstantEmitter> emitter;
  mlir::Attribute initialValue = nullptr;
  bool isConstant = false;
  mlir::Type type;

  if (value) {
    emitter.emplace(*this);
    initialValue = emitter->emitForInitializer(*value, materializedType);

    isConstant = materializedType.isConstantStorage(
        getASTContext(), /*ExcludeCtor=*/value, /*ExcludeDtor=*/false);

    type = mlir::cast<mlir::TypedAttr>(initialValue).getType();
  } else {
    // No initializer, the initialization will be provided when we initialize
    // the declaration which performed lifetime extension.
    type = getTypes().convertTypeForMem(materializedType);
  }

  // Create a global variable for this lifetime-extended temporary.
  cir::GlobalLinkageKind linkage = getCIRLinkageVarDefinition(varDecl);
  if (linkage == cir::GlobalLinkageKind::ExternalLinkage) {
    const VarDecl *initVD;
    if (varDecl->isStaticDataMember() && varDecl->getAnyInitializer(initVD) &&
        isa<CXXRecordDecl>(initVD->getLexicalDeclContext())) {
      // Temporaries defined inside a class get linkonce_odr linkage because the
      // calss can be defined in multiple translation units.
      errorNYI(mte->getSourceRange(), "static data member initialization");
    } else {
      // There is no need for this temporary to have external linkage if the
      // VarDecl has external linkage.
      linkage = cir::GlobalLinkageKind::InternalLinkage;
    }
  }
  mlir::Location loc = getLoc(mte->getSourceRange());
  cir::GlobalOp gv = createGlobalOp(*this, loc, name, type, isConstant);
  gv.setInitialValueAttr(initialValue);

  if (emitter)
    emitter->finalize(gv);
  // Don't assign dllimport or dllexport to local linkage globals
  if (!gv.hasLocalLinkage()) {
    setGVProperties(gv, varDecl);
    assert(!cir::MissingFeatures::setDLLStorageClass());
  }

  gv.setAlignment(align.getAsAlign().value());
  if (supportsCOMDAT() && gv.isWeakForLinker())
    errorNYI(mte->getSourceRange(),
             "Global temporary with comdat/weak linkage");
  if (varDecl->getTLSKind())
    errorNYI(mte->getSourceRange(),
             "Global temporary with thread local storage");
  mlir::Operation *cv = gv;

  assert(!cir::MissingFeatures::addressSpace());

  // Update the map with the new temporary. If we created a placeholder above,
  // replace it with the new global now.
  mlir::Operation *&entry = materializedGlobalTemporaryMap[mte];
  if (entry) {
    entry->replaceAllUsesWith(cv);
    eraseGlobalSymbol(entry);
    entry->erase();
  }
  entry = cv;

  return cv;
}

cir::GlobalOp CIRGenModule::getAddrOfUnnamedGlobalConstantDecl(
    const UnnamedGlobalConstantDecl *gcd) {
  unsigned numEntries = unnamedGlobalConstantDeclMap.size();
  cir::GlobalOp *globalOpEntry = &unnamedGlobalConstantDeclMap[gcd];

  if (*globalOpEntry)
    return *globalOpEntry;

  ConstantEmitter emitter(*this);

  const APValue &value = gcd->getValue();
  assert(!value.isAbsent());
  assert(!cir::MissingFeatures::addressSpace() &&
         "emitForInitializer should take gcd->getType().getAddressSpace()");
  mlir::Attribute init = emitter.emitForInitializer(value, gcd->getType());
  auto typedInit = dyn_cast<mlir::TypedAttr>(init);

  if (!typedInit)
    errorNYI(gcd->getSourceRange(),
             "getAddrOfUnnamedGlobalConstantDecl: non-typed initializer");

  assert(!cir::MissingFeatures::addressSpace());

  // Classic codegen always creates these with .constant, then counts on the
  // auto-addition of '.#'. CIR global doesn't have this, so we'll just auto-add
  // one if this isn't the first.  We could probably choose a better name than
  // .constant to be unique for this type of decl, but this is consistent with
  // classic codegen.
  std::string name = numEntries == 0
                         ? ".constant"
                         : (Twine(".constant.") + Twine(numEntries)).str();
  auto globalOp = createGlobalOp(*this, builder.getUnknownLoc(), name,
                                 typedInit.getType(), /*is_constant=*/true);
  globalOp.setLinkage(cir::GlobalLinkageKind::PrivateLinkage);

  CharUnits alignment = getASTContext().getTypeAlignInChars(gcd->getType());
  globalOp.setAlignment(alignment.getAsAlign().value());
  CIRGenModule::setInitializer(globalOp, init);

  emitter.finalize(globalOp);
  *globalOpEntry = globalOp;
  return globalOp;
}

cir::GlobalOp
CIRGenModule::getAddrOfTemplateParamObject(const TemplateParamObjectDecl *tpo) {
  StringRef name = getMangledName(tpo);
  CharUnits alignment = getNaturalTypeAlignment(tpo->getType());

  if (auto globalOp =
          mlir::dyn_cast_or_null<cir::GlobalOp>(getGlobalValue(name)))
    return globalOp;

  ConstantEmitter emitter(*this);
  assert(!cir::MissingFeatures::addressSpace() &&
         "emitForInitializer should take tpo->getType().getAddressSpace()");
  mlir::Attribute init =
      emitter.emitForInitializer(tpo->getValue(), tpo->getType());

  if (!init) {
    errorUnsupported(tpo, "template parameter object");
    return {};
  }

  mlir::TypedAttr typedInit = cast<mlir::TypedAttr>(init);

  cir::GlobalLinkageKind linkage =
      isExternallyVisible(tpo->getLinkageAndVisibility().getLinkage())
          ? cir::GlobalLinkageKind::LinkOnceODRLinkage
          : cir::GlobalLinkageKind::InternalLinkage;

  assert(!cir::MissingFeatures::addressSpace());
  auto globalOp = createGlobalOp(*this, builder.getUnknownLoc(), name,
                                 typedInit.getType(), /*is_constant=*/true);
  globalOp.setLinkage(linkage);
  globalOp.setAlignment(alignment.getAsAlign().value());
  globalOp.setComdat(supportsCOMDAT() &&
                     linkage == cir::GlobalLinkageKind::LinkOnceODRLinkage);

  CIRGenModule::setInitializer(globalOp, init);
  emitter.finalize(globalOp);

  insertGlobalSymbol(globalOp);

  return globalOp;
}

//===----------------------------------------------------------------------===//
// Annotations
//===----------------------------------------------------------------------===//

mlir::ArrayAttr
CIRGenModule::getOrCreateAnnotationArgs(const AnnotateAttr *attr) {
  ArrayRef<Expr *> exprs = {attr->args_begin(), attr->args_size()};
  // Return a null attr for no-args annotations so OptionalParameter omits
  // the args portion entirely from the printed IR.
  if (exprs.empty())
    return {};

  llvm::FoldingSetNodeID id;
  for (Expr *e : exprs)
    id.Add(cast<clang::ConstantExpr>(e)->getAPValueResult());

  mlir::ArrayAttr &lookup = annotationArgs[id.ComputeHash()];
  if (lookup)
    return lookup;

  llvm::SmallVector<mlir::Attribute> args;
  args.reserve(exprs.size());
  for (Expr *e : exprs) {
    if (auto *strE = dyn_cast<clang::StringLiteral>(e->IgnoreParenCasts())) {
      args.push_back(builder.getStringAttr(strE->getString()));
    } else if (auto *intE =
                   dyn_cast<clang::IntegerLiteral>(e->IgnoreParenCasts())) {
      auto intTy = builder.getIntegerType(intE->getValue().getBitWidth());
      args.push_back(builder.getIntegerAttr(intTy, intE->getValue()));
    } else {
      errorNYI(e->getExprLoc(), "annotation argument expression");
    }
  }

  return lookup = builder.getArrayAttr(args);
}

cir::AnnotationAttr CIRGenModule::emitAnnotateAttr(const AnnotateAttr *aa) {
  mlir::StringAttr annoGV = builder.getStringAttr(aa->getAnnotation());
  mlir::ArrayAttr args = getOrCreateAnnotationArgs(aa);
  return cir::AnnotationAttr::get(&getMLIRContext(), annoGV, args);
}

void CIRGenModule::addGlobalAnnotations(const ValueDecl *d,
                                        mlir::Operation *gv) {
  assert(d->hasAttr<AnnotateAttr>() && "no annotate attribute");
  assert((isa<cir::GlobalOp>(gv) || isa<cir::FuncOp>(gv)) &&
         "annotation only on globals");
  llvm::SmallVector<mlir::Attribute> annotations;
  for (const auto *i : d->specific_attrs<AnnotateAttr>())
    annotations.push_back(emitAnnotateAttr(i));
  if (auto global = dyn_cast<cir::GlobalOp>(gv))
    global.setAnnotationsAttr(builder.getArrayAttr(annotations));
  else if (auto func = dyn_cast<cir::FuncOp>(gv))
    func.setAnnotationsAttr(builder.getArrayAttr(annotations));
}

void CIRGenModule::emitGlobalAnnotations() {
  for (const auto &[mangledName, vd] : deferredAnnotations) {
    mlir::Operation *gv = getGlobalValue(mangledName);
    if (gv)
      addGlobalAnnotations(vd, gv);
  }
  deferredAnnotations.clear();
}
