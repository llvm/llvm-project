//===--- CodeGenModule.cpp - Emit LLVM Code from ASTs for a Module --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This coordinates the per-module state used while generating code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenModule.h"
#include "CGBlocks.h"
#include "CGCUDARuntime.h"
#include "CGCXXABI.h"
#include "CGCall.h"
#include "CGDebugInfo.h"
#include "CGObjCRuntime.h"
#include "CGOpenCLRuntime.h"
#include "CGOpenMPRuntime.h"
#include "CGOpenMPRuntimeNVPTX.h"
#include "CodeGenFunction.h"
#include "CodeGenPGO.h"
#include "CodeGenTBAA.h"
#include "CoverageMappingGen.h"
#include "TargetInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/CodeGen/ConstantInitBuilder.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MD5.h"

using namespace clang;
using namespace CodeGen;

static const char AnnotationSection[] = "llvm.metadata";

static CGCXXABI *createCXXABI(CodeGenModule &CGM) {
  switch (CGM.getTarget().getCXXABI().getKind()) {
  case TargetCXXABI::GenericAArch64:
  case TargetCXXABI::GenericARM:
  case TargetCXXABI::iOS:
  case TargetCXXABI::iOS64:
  case TargetCXXABI::WatchOS:
  case TargetCXXABI::GenericMIPS:
  case TargetCXXABI::GenericItanium:
  case TargetCXXABI::WebAssembly:
    return CreateItaniumCXXABI(CGM);
  case TargetCXXABI::Microsoft:
    return CreateMicrosoftCXXABI(CGM);
  }

  llvm_unreachable("invalid C++ ABI kind");
}

CodeGenModule::CodeGenModule(ASTContext &C, const HeaderSearchOptions &HSO,
                             const PreprocessorOptions &PPO,
                             const CodeGenOptions &CGO, llvm::Module &M,
                             DiagnosticsEngine &diags,
                             CoverageSourceInfo *CoverageInfo)
    : Context(C), LangOpts(C.getLangOpts()), HeaderSearchOpts(HSO),
      PreprocessorOpts(PPO), CodeGenOpts(CGO), TheModule(M), Diags(diags),
      Target(C.getTargetInfo()), ABI(createCXXABI(*this)),
      VMContext(M.getContext()), Types(*this), VTables(*this),
      SanitizerMD(new SanitizerMetadata(*this)) {

  // Initialize the type cache.
  llvm::LLVMContext &LLVMContext = M.getContext();
  VoidTy = llvm::Type::getVoidTy(LLVMContext);
  Int8Ty = llvm::Type::getInt8Ty(LLVMContext);
  Int16Ty = llvm::Type::getInt16Ty(LLVMContext);
  Int32Ty = llvm::Type::getInt32Ty(LLVMContext);
  Int64Ty = llvm::Type::getInt64Ty(LLVMContext);
  HalfTy = llvm::Type::getHalfTy(LLVMContext);
  FloatTy = llvm::Type::getFloatTy(LLVMContext);
  DoubleTy = llvm::Type::getDoubleTy(LLVMContext);
  PointerWidthInBits = C.getTargetInfo().getPointerWidth(0);
  PointerAlignInBytes =
    C.toCharUnitsFromBits(C.getTargetInfo().getPointerAlign(0)).getQuantity();
  SizeSizeInBytes =
    C.toCharUnitsFromBits(C.getTargetInfo().getMaxPointerWidth()).getQuantity();
  IntAlignInBytes =
    C.toCharUnitsFromBits(C.getTargetInfo().getIntAlign()).getQuantity();
  IntTy = llvm::IntegerType::get(LLVMContext, C.getTargetInfo().getIntWidth());
  IntPtrTy = llvm::IntegerType::get(LLVMContext,
    C.getTargetInfo().getMaxPointerWidth());
  Int8PtrTy = Int8Ty->getPointerTo(0);
  Int8PtrPtrTy = Int8PtrTy->getPointerTo(0);
  AllocaInt8PtrTy = Int8Ty->getPointerTo(
      M.getDataLayout().getAllocaAddrSpace());
  ASTAllocaAddressSpace = getTargetCodeGenInfo().getASTAllocaAddressSpace();

  RuntimeCC = getTargetCodeGenInfo().getABIInfo().getRuntimeCC();
  BuiltinCC = getTargetCodeGenInfo().getABIInfo().getBuiltinCC();

  if (LangOpts.ObjC1)
    createObjCRuntime();
  if (LangOpts.OpenCL)
    createOpenCLRuntime();
  if (LangOpts.OpenMP)
    createOpenMPRuntime();
  if (LangOpts.CUDA)
    createCUDARuntime();

  // Enable TBAA unless it's suppressed. ThreadSanitizer needs TBAA even at O0.
  if (LangOpts.Sanitize.has(SanitizerKind::Thread) ||
      (!CodeGenOpts.RelaxedAliasing && CodeGenOpts.OptimizationLevel > 0))
    TBAA.reset(new CodeGenTBAA(Context, VMContext, CodeGenOpts, getLangOpts(),
                               getCXXABI().getMangleContext()));

  // If debug info or coverage generation is enabled, create the CGDebugInfo
  // object.
  if (CodeGenOpts.getDebugInfo() != codegenoptions::NoDebugInfo ||
      CodeGenOpts.EmitGcovArcs || CodeGenOpts.EmitGcovNotes)
    DebugInfo.reset(new CGDebugInfo(*this));

  Block.GlobalUniqueCount = 0;

  if (C.getLangOpts().ObjC1)
    ObjCData.reset(new ObjCEntrypoints());

  if (CodeGenOpts.hasProfileClangUse()) {
    auto ReaderOrErr = llvm::IndexedInstrProfReader::create(
        CodeGenOpts.ProfileInstrumentUsePath);
    if (auto E = ReaderOrErr.takeError()) {
      unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                              "Could not read profile %0: %1");
      llvm::handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EI) {
        getDiags().Report(DiagID) << CodeGenOpts.ProfileInstrumentUsePath
                                  << EI.message();
      });
    } else
      PGOReader = std::move(ReaderOrErr.get());
  }

  // If coverage mapping generation is enabled, create the
  // CoverageMappingModuleGen object.
  if (CodeGenOpts.CoverageMapping)
    CoverageMapping.reset(new CoverageMappingModuleGen(*this, *CoverageInfo));
}

CodeGenModule::~CodeGenModule() {}

void CodeGenModule::createObjCRuntime() {
  // This is just isGNUFamily(), but we want to force implementors of
  // new ABIs to decide how best to do this.
  switch (LangOpts.ObjCRuntime.getKind()) {
  case ObjCRuntime::GNUstep:
  case ObjCRuntime::GCC:
  case ObjCRuntime::ObjFW:
    ObjCRuntime.reset(CreateGNUObjCRuntime(*this));
    return;

  case ObjCRuntime::FragileMacOSX:
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
  case ObjCRuntime::WatchOS:
    ObjCRuntime.reset(CreateMacObjCRuntime(*this));
    return;
  }
  llvm_unreachable("bad runtime kind");
}

void CodeGenModule::createOpenCLRuntime() {
  OpenCLRuntime.reset(new CGOpenCLRuntime(*this));
}

void CodeGenModule::createOpenMPRuntime() {
  // Select a specialized code generation class based on the target, if any.
  // If it does not exist use the default implementation.
  switch (getTriple().getArch()) {
  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
    assert(getLangOpts().OpenMPIsDevice &&
           "OpenMP NVPTX is only prepared to deal with device code.");
    OpenMPRuntime.reset(new CGOpenMPRuntimeNVPTX(*this));
    break;
  default:
    OpenMPRuntime.reset(new CGOpenMPRuntime(*this));
    break;
  }
}

void CodeGenModule::createCUDARuntime() {
  CUDARuntime.reset(CreateNVCUDARuntime(*this));
}

void CodeGenModule::addReplacement(StringRef Name, llvm::Constant *C) {
  Replacements[Name] = C;
}

void CodeGenModule::applyReplacements() {
  for (auto &I : Replacements) {
    StringRef MangledName = I.first();
    llvm::Constant *Replacement = I.second;
    llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
    if (!Entry)
      continue;
    auto *OldF = cast<llvm::Function>(Entry);
    auto *NewF = dyn_cast<llvm::Function>(Replacement);
    if (!NewF) {
      if (auto *Alias = dyn_cast<llvm::GlobalAlias>(Replacement)) {
        NewF = dyn_cast<llvm::Function>(Alias->getAliasee());
      } else {
        auto *CE = cast<llvm::ConstantExpr>(Replacement);
        assert(CE->getOpcode() == llvm::Instruction::BitCast ||
               CE->getOpcode() == llvm::Instruction::GetElementPtr);
        NewF = dyn_cast<llvm::Function>(CE->getOperand(0));
      }
    }

    // Replace old with new, but keep the old order.
    OldF->replaceAllUsesWith(Replacement);
    if (NewF) {
      NewF->removeFromParent();
      OldF->getParent()->getFunctionList().insertAfter(OldF->getIterator(),
                                                       NewF);
    }
    OldF->eraseFromParent();
  }
}

void CodeGenModule::addGlobalValReplacement(llvm::GlobalValue *GV, llvm::Constant *C) {
  GlobalValReplacements.push_back(std::make_pair(GV, C));
}

void CodeGenModule::applyGlobalValReplacements() {
  for (auto &I : GlobalValReplacements) {
    llvm::GlobalValue *GV = I.first;
    llvm::Constant *C = I.second;

    GV->replaceAllUsesWith(C);
    GV->eraseFromParent();
  }
}

// This is only used in aliases that we created and we know they have a
// linear structure.
static const llvm::GlobalObject *getAliasedGlobal(
    const llvm::GlobalIndirectSymbol &GIS) {
  llvm::SmallPtrSet<const llvm::GlobalIndirectSymbol*, 4> Visited;
  const llvm::Constant *C = &GIS;
  for (;;) {
    C = C->stripPointerCasts();
    if (auto *GO = dyn_cast<llvm::GlobalObject>(C))
      return GO;
    // stripPointerCasts will not walk over weak aliases.
    auto *GIS2 = dyn_cast<llvm::GlobalIndirectSymbol>(C);
    if (!GIS2)
      return nullptr;
    if (!Visited.insert(GIS2).second)
      return nullptr;
    C = GIS2->getIndirectSymbol();
  }
}

void CodeGenModule::checkAliases() {
  // Check if the constructed aliases are well formed. It is really unfortunate
  // that we have to do this in CodeGen, but we only construct mangled names
  // and aliases during codegen.
  bool Error = false;
  DiagnosticsEngine &Diags = getDiags();
  for (const GlobalDecl &GD : Aliases) {
    const auto *D = cast<ValueDecl>(GD.getDecl());
    SourceLocation Location;
    bool IsIFunc = D->hasAttr<IFuncAttr>();
    if (const Attr *A = D->getDefiningAttr())
      Location = A->getLocation();
    else
      llvm_unreachable("Not an alias or ifunc?");
    StringRef MangledName = getMangledName(GD);
    llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
    auto *Alias  = cast<llvm::GlobalIndirectSymbol>(Entry);
    const llvm::GlobalValue *GV = getAliasedGlobal(*Alias);
    if (!GV) {
      Error = true;
      Diags.Report(Location, diag::err_cyclic_alias) << IsIFunc;
    } else if (GV->isDeclaration()) {
      Error = true;
      Diags.Report(Location, diag::err_alias_to_undefined)
          << IsIFunc << IsIFunc;
    } else if (IsIFunc) {
      // Check resolver function type.
      llvm::FunctionType *FTy = dyn_cast<llvm::FunctionType>(
          GV->getType()->getPointerElementType());
      assert(FTy);
      if (!FTy->getReturnType()->isPointerTy())
        Diags.Report(Location, diag::err_ifunc_resolver_return);
      if (FTy->getNumParams())
        Diags.Report(Location, diag::err_ifunc_resolver_params);
    }

    llvm::Constant *Aliasee = Alias->getIndirectSymbol();
    llvm::GlobalValue *AliaseeGV;
    if (auto CE = dyn_cast<llvm::ConstantExpr>(Aliasee))
      AliaseeGV = cast<llvm::GlobalValue>(CE->getOperand(0));
    else
      AliaseeGV = cast<llvm::GlobalValue>(Aliasee);

    if (const SectionAttr *SA = D->getAttr<SectionAttr>()) {
      StringRef AliasSection = SA->getName();
      if (AliasSection != AliaseeGV->getSection())
        Diags.Report(SA->getLocation(), diag::warn_alias_with_section)
            << AliasSection << IsIFunc << IsIFunc;
    }

    // We have to handle alias to weak aliases in here. LLVM itself disallows
    // this since the object semantics would not match the IL one. For
    // compatibility with gcc we implement it by just pointing the alias
    // to its aliasee's aliasee. We also warn, since the user is probably
    // expecting the link to be weak.
    if (auto GA = dyn_cast<llvm::GlobalIndirectSymbol>(AliaseeGV)) {
      if (GA->isInterposable()) {
        Diags.Report(Location, diag::warn_alias_to_weak_alias)
            << GV->getName() << GA->getName() << IsIFunc;
        Aliasee = llvm::ConstantExpr::getPointerBitCastOrAddrSpaceCast(
            GA->getIndirectSymbol(), Alias->getType());
        Alias->setIndirectSymbol(Aliasee);
      }
    }
  }
  if (!Error)
    return;

  for (const GlobalDecl &GD : Aliases) {
    StringRef MangledName = getMangledName(GD);
    llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
    auto *Alias = dyn_cast<llvm::GlobalIndirectSymbol>(Entry);
    Alias->replaceAllUsesWith(llvm::UndefValue::get(Alias->getType()));
    Alias->eraseFromParent();
  }
}

void CodeGenModule::clear() {
  DeferredDeclsToEmit.clear();
  if (OpenMPRuntime)
    OpenMPRuntime->clear();
}

void InstrProfStats::reportDiagnostics(DiagnosticsEngine &Diags,
                                       StringRef MainFile) {
  if (!hasDiagnostics())
    return;
  if (VisitedInMainFile > 0 && VisitedInMainFile == MissingInMainFile) {
    if (MainFile.empty())
      MainFile = "<stdin>";
    Diags.Report(diag::warn_profile_data_unprofiled) << MainFile;
  } else {
    if (Mismatched > 0)
      Diags.Report(diag::warn_profile_data_out_of_date) << Visited << Mismatched;

    if (Missing > 0)
      Diags.Report(diag::warn_profile_data_missing) << Visited << Missing;
  }
}

void CodeGenModule::Release() {
  EmitDeferred();
  EmitVTablesOpportunistically();
  applyGlobalValReplacements();
  applyReplacements();
  checkAliases();
  EmitCXXGlobalInitFunc();
  EmitCXXGlobalDtorFunc();
  EmitCXXThreadLocalInitFunc();
  if (ObjCRuntime)
    if (llvm::Function *ObjCInitFunction = ObjCRuntime->ModuleInitFunction())
      AddGlobalCtor(ObjCInitFunction);
  if (Context.getLangOpts().CUDA && !Context.getLangOpts().CUDAIsDevice &&
      CUDARuntime) {
    if (llvm::Function *CudaCtorFunction = CUDARuntime->makeModuleCtorFunction())
      AddGlobalCtor(CudaCtorFunction);
    if (llvm::Function *CudaDtorFunction = CUDARuntime->makeModuleDtorFunction())
      AddGlobalDtor(CudaDtorFunction);
  }
  if (OpenMPRuntime)
    if (llvm::Function *OpenMPRegistrationFunction =
            OpenMPRuntime->emitRegistrationFunction()) {
      auto ComdatKey = OpenMPRegistrationFunction->hasComdat() ?
        OpenMPRegistrationFunction : nullptr;
      AddGlobalCtor(OpenMPRegistrationFunction, 0, ComdatKey);
    }
  if (PGOReader) {
    getModule().setProfileSummary(PGOReader->getSummary().getMD(VMContext));
    if (PGOStats.hasDiagnostics())
      PGOStats.reportDiagnostics(getDiags(), getCodeGenOpts().MainFileName);
  }
  EmitCtorList(GlobalCtors, "llvm.global_ctors");
  EmitCtorList(GlobalDtors, "llvm.global_dtors");
  EmitGlobalAnnotations();
  EmitStaticExternCAliases();
  EmitDeferredUnusedCoverageMappings();
  if (CoverageMapping)
    CoverageMapping->emit();
  if (CodeGenOpts.SanitizeCfiCrossDso) {
    CodeGenFunction(*this).EmitCfiCheckFail();
    CodeGenFunction(*this).EmitCfiCheckStub();
  }
  emitAtAvailableLinkGuard();
  emitLLVMUsed();
  if (SanStats)
    SanStats->finish();

  if (CodeGenOpts.Autolink &&
      (Context.getLangOpts().Modules || !LinkerOptionsMetadata.empty())) {
    EmitModuleLinkOptions();
  }

  // Record mregparm value now so it is visible through rest of codegen.
  if (Context.getTargetInfo().getTriple().getArch() == llvm::Triple::x86)
    getModule().addModuleFlag(llvm::Module::Error, "NumRegisterParameters",
                              CodeGenOpts.NumRegisterParameters);
  
  if (CodeGenOpts.DwarfVersion) {
    // We actually want the latest version when there are conflicts.
    // We can change from Warning to Latest if such mode is supported.
    getModule().addModuleFlag(llvm::Module::Warning, "Dwarf Version",
                              CodeGenOpts.DwarfVersion);
  }
  if (CodeGenOpts.EmitCodeView) {
    // Indicate that we want CodeView in the metadata.
    getModule().addModuleFlag(llvm::Module::Warning, "CodeView", 1);
  }
  if (CodeGenOpts.OptimizationLevel > 0 && CodeGenOpts.StrictVTablePointers) {
    // We don't support LTO with 2 with different StrictVTablePointers
    // FIXME: we could support it by stripping all the information introduced
    // by StrictVTablePointers.

    getModule().addModuleFlag(llvm::Module::Error, "StrictVTablePointers",1);

    llvm::Metadata *Ops[2] = {
              llvm::MDString::get(VMContext, "StrictVTablePointers"),
              llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                  llvm::Type::getInt32Ty(VMContext), 1))};

    getModule().addModuleFlag(llvm::Module::Require,
                              "StrictVTablePointersRequirement",
                              llvm::MDNode::get(VMContext, Ops));
  }
  if (DebugInfo)
    // We support a single version in the linked module. The LLVM
    // parser will drop debug info with a different version number
    // (and warn about it, too).
    getModule().addModuleFlag(llvm::Module::Warning, "Debug Info Version",
                              llvm::DEBUG_METADATA_VERSION);

  // Width of wchar_t in bytes
  uint64_t WCharWidth =
      Context.getTypeSizeInChars(Context.getWideCharType()).getQuantity();
  assert((LangOpts.ShortWChar ||
          llvm::TargetLibraryInfoImpl::getTargetWCharSize(Target.getTriple()) ==
              Target.getWCharWidth() / 8) &&
         "LLVM wchar_t size out of sync");

  // We need to record the widths of enums and wchar_t, so that we can generate
  // the correct build attributes in the ARM backend. wchar_size is also used by
  // TargetLibraryInfo.
  getModule().addModuleFlag(llvm::Module::Error, "wchar_size", WCharWidth);

  llvm::Triple::ArchType Arch = Context.getTargetInfo().getTriple().getArch();
  if (   Arch == llvm::Triple::arm
      || Arch == llvm::Triple::armeb
      || Arch == llvm::Triple::thumb
      || Arch == llvm::Triple::thumbeb) {
    // The minimum width of an enum in bytes
    uint64_t EnumWidth = Context.getLangOpts().ShortEnums ? 1 : 4;
    getModule().addModuleFlag(llvm::Module::Error, "min_enum_size", EnumWidth);
  }

  if (CodeGenOpts.SanitizeCfiCrossDso) {
    // Indicate that we want cross-DSO control flow integrity checks.
    getModule().addModuleFlag(llvm::Module::Override, "Cross-DSO CFI", 1);
  }

  if (LangOpts.CUDAIsDevice && getTriple().isNVPTX()) {
    // Indicate whether __nvvm_reflect should be configured to flush denormal
    // floating point values to 0.  (This corresponds to its "__CUDA_FTZ"
    // property.)
    getModule().addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                              LangOpts.CUDADeviceFlushDenormalsToZero ? 1 : 0);
  }

  // Emit OpenCL specific module metadata: OpenCL/SPIR version.
  if (LangOpts.OpenCL) {
    EmitOpenCLMetadata();
    // Emit SPIR version.
    if (getTriple().getArch() == llvm::Triple::spir ||
        getTriple().getArch() == llvm::Triple::spir64) {
      // SPIR v2.0 s2.12 - The SPIR version used by the module is stored in the
      // opencl.spir.version named metadata.
      llvm::Metadata *SPIRVerElts[] = {
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
              Int32Ty, LangOpts.OpenCLVersion / 100)),
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
              Int32Ty, (LangOpts.OpenCLVersion / 100 > 1) ? 0 : 2))};
      llvm::NamedMDNode *SPIRVerMD =
          TheModule.getOrInsertNamedMetadata("opencl.spir.version");
      llvm::LLVMContext &Ctx = TheModule.getContext();
      SPIRVerMD->addOperand(llvm::MDNode::get(Ctx, SPIRVerElts));
    }
  }

  if (uint32_t PLevel = Context.getLangOpts().PICLevel) {
    assert(PLevel < 3 && "Invalid PIC Level");
    getModule().setPICLevel(static_cast<llvm::PICLevel::Level>(PLevel));
    if (Context.getLangOpts().PIE)
      getModule().setPIELevel(static_cast<llvm::PIELevel::Level>(PLevel));
  }

  SimplifyPersonality();

  if (getCodeGenOpts().EmitDeclMetadata)
    EmitDeclMetadata();

  if (getCodeGenOpts().EmitGcovArcs || getCodeGenOpts().EmitGcovNotes)
    EmitCoverageFile();

  if (DebugInfo)
    DebugInfo->finalize();

  EmitVersionIdentMetadata();

  EmitTargetMetadata();
}

void CodeGenModule::EmitOpenCLMetadata() {
  // SPIR v2.0 s2.13 - The OpenCL version used by the module is stored in the
  // opencl.ocl.version named metadata node.
  llvm::Metadata *OCLVerElts[] = {
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
          Int32Ty, LangOpts.OpenCLVersion / 100)),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
          Int32Ty, (LangOpts.OpenCLVersion % 100) / 10))};
  llvm::NamedMDNode *OCLVerMD =
      TheModule.getOrInsertNamedMetadata("opencl.ocl.version");
  llvm::LLVMContext &Ctx = TheModule.getContext();
  OCLVerMD->addOperand(llvm::MDNode::get(Ctx, OCLVerElts));
}

void CodeGenModule::UpdateCompletedType(const TagDecl *TD) {
  // Make sure that this type is translated.
  Types.UpdateCompletedType(TD);
}

void CodeGenModule::RefreshTypeCacheForClass(const CXXRecordDecl *RD) {
  // Make sure that this type is translated.
  Types.RefreshTypeCacheForClass(RD);
}

llvm::MDNode *CodeGenModule::getTBAAInfo(QualType QTy) {
  if (!TBAA)
    return nullptr;
  return TBAA->getTBAAInfo(QTy);
}

llvm::MDNode *CodeGenModule::getTBAAInfoForVTablePtr() {
  if (!TBAA)
    return nullptr;
  return TBAA->getTBAAInfoForVTablePtr();
}

llvm::MDNode *CodeGenModule::getTBAAStructInfo(QualType QTy) {
  if (!TBAA)
    return nullptr;
  return TBAA->getTBAAStructInfo(QTy);
}

llvm::MDNode *CodeGenModule::getTBAAStructTagInfo(QualType BaseTy,
                                                  llvm::MDNode *AccessN,
                                                  uint64_t O) {
  if (!TBAA)
    return nullptr;
  return TBAA->getTBAAStructTagInfo(BaseTy, AccessN, O);
}

/// Decorate the instruction with a TBAA tag. For both scalar TBAA
/// and struct-path aware TBAA, the tag has the same format:
/// base type, access type and offset.
/// When ConvertTypeToTag is true, we create a tag based on the scalar type.
void CodeGenModule::DecorateInstructionWithTBAA(llvm::Instruction *Inst,
                                                llvm::MDNode *TBAAInfo,
                                                bool ConvertTypeToTag) {
  if (ConvertTypeToTag && TBAA)
    Inst->setMetadata(llvm::LLVMContext::MD_tbaa,
                      TBAA->getTBAAScalarTagInfo(TBAAInfo));
  else
    Inst->setMetadata(llvm::LLVMContext::MD_tbaa, TBAAInfo);
}

void CodeGenModule::DecorateInstructionWithInvariantGroup(
    llvm::Instruction *I, const CXXRecordDecl *RD) {
  I->setMetadata(llvm::LLVMContext::MD_invariant_group,
                 llvm::MDNode::get(getLLVMContext(), {}));
}

void CodeGenModule::Error(SourceLocation loc, StringRef message) {
  unsigned diagID = getDiags().getCustomDiagID(DiagnosticsEngine::Error, "%0");
  getDiags().Report(Context.getFullLoc(loc), diagID) << message;
}

/// ErrorUnsupported - Print out an error that codegen doesn't support the
/// specified stmt yet.
void CodeGenModule::ErrorUnsupported(const Stmt *S, const char *Type) {
  unsigned DiagID = getDiags().getCustomDiagID(DiagnosticsEngine::Error,
                                               "cannot compile this %0 yet");
  std::string Msg = Type;
  getDiags().Report(Context.getFullLoc(S->getLocStart()), DiagID)
    << Msg << S->getSourceRange();
}

/// ErrorUnsupported - Print out an error that codegen doesn't support the
/// specified decl yet.
void CodeGenModule::ErrorUnsupported(const Decl *D, const char *Type) {
  unsigned DiagID = getDiags().getCustomDiagID(DiagnosticsEngine::Error,
                                               "cannot compile this %0 yet");
  std::string Msg = Type;
  getDiags().Report(Context.getFullLoc(D->getLocation()), DiagID) << Msg;
}

llvm::ConstantInt *CodeGenModule::getSize(CharUnits size) {
  return llvm::ConstantInt::get(SizeTy, size.getQuantity());
}

void CodeGenModule::setGlobalVisibility(llvm::GlobalValue *GV,
                                        const NamedDecl *D) const {
  // Internal definitions always have default visibility.
  if (GV->hasLocalLinkage()) {
    GV->setVisibility(llvm::GlobalValue::DefaultVisibility);
    return;
  }

  // Set visibility for definitions.
  LinkageInfo LV = D->getLinkageAndVisibility();
  if (LV.isVisibilityExplicit() || !GV->hasAvailableExternallyLinkage())
    GV->setVisibility(GetLLVMVisibility(LV.getVisibility()));
}

static llvm::GlobalVariable::ThreadLocalMode GetLLVMTLSModel(StringRef S) {
  return llvm::StringSwitch<llvm::GlobalVariable::ThreadLocalMode>(S)
      .Case("global-dynamic", llvm::GlobalVariable::GeneralDynamicTLSModel)
      .Case("local-dynamic", llvm::GlobalVariable::LocalDynamicTLSModel)
      .Case("initial-exec", llvm::GlobalVariable::InitialExecTLSModel)
      .Case("local-exec", llvm::GlobalVariable::LocalExecTLSModel);
}

static llvm::GlobalVariable::ThreadLocalMode GetLLVMTLSModel(
    CodeGenOptions::TLSModel M) {
  switch (M) {
  case CodeGenOptions::GeneralDynamicTLSModel:
    return llvm::GlobalVariable::GeneralDynamicTLSModel;
  case CodeGenOptions::LocalDynamicTLSModel:
    return llvm::GlobalVariable::LocalDynamicTLSModel;
  case CodeGenOptions::InitialExecTLSModel:
    return llvm::GlobalVariable::InitialExecTLSModel;
  case CodeGenOptions::LocalExecTLSModel:
    return llvm::GlobalVariable::LocalExecTLSModel;
  }
  llvm_unreachable("Invalid TLS model!");
}

void CodeGenModule::setTLSMode(llvm::GlobalValue *GV, const VarDecl &D) const {
  assert(D.getTLSKind() && "setting TLS mode on non-TLS var!");

  llvm::GlobalValue::ThreadLocalMode TLM;
  TLM = GetLLVMTLSModel(CodeGenOpts.getDefaultTLSModel());

  // Override the TLS model if it is explicitly specified.
  if (const TLSModelAttr *Attr = D.getAttr<TLSModelAttr>()) {
    TLM = GetLLVMTLSModel(Attr->getModel());
  }

  GV->setThreadLocalMode(TLM);
}

StringRef CodeGenModule::getMangledName(GlobalDecl GD) {
  GlobalDecl CanonicalGD = GD.getCanonicalDecl();

  // Some ABIs don't have constructor variants.  Make sure that base and
  // complete constructors get mangled the same.
  if (const auto *CD = dyn_cast<CXXConstructorDecl>(CanonicalGD.getDecl())) {
    if (!getTarget().getCXXABI().hasConstructorVariants()) {
      CXXCtorType OrigCtorType = GD.getCtorType();
      assert(OrigCtorType == Ctor_Base || OrigCtorType == Ctor_Complete);
      if (OrigCtorType == Ctor_Base)
        CanonicalGD = GlobalDecl(CD, Ctor_Complete);
    }
  }

  StringRef &FoundStr = MangledDeclNames[CanonicalGD];
  if (!FoundStr.empty())
    return FoundStr;

  const auto *ND = cast<NamedDecl>(GD.getDecl());
  SmallString<256> Buffer;
  StringRef Str;
  if (getCXXABI().getMangleContext().shouldMangleDeclName(ND)) {
    llvm::raw_svector_ostream Out(Buffer);
    if (const auto *D = dyn_cast<CXXConstructorDecl>(ND))
      getCXXABI().getMangleContext().mangleCXXCtor(D, GD.getCtorType(), Out);
    else if (const auto *D = dyn_cast<CXXDestructorDecl>(ND))
      getCXXABI().getMangleContext().mangleCXXDtor(D, GD.getDtorType(), Out);
    else
      getCXXABI().getMangleContext().mangleName(ND, Out);
    Str = Out.str();
  } else {
    IdentifierInfo *II = ND->getIdentifier();
    assert(II && "Attempt to mangle unnamed decl.");
    const auto *FD = dyn_cast<FunctionDecl>(ND);

    if (FD &&
        FD->getType()->castAs<FunctionType>()->getCallConv() == CC_X86RegCall) {
      llvm::raw_svector_ostream Out(Buffer);
      Out << "__regcall3__" << II->getName();
      Str = Out.str();
    } else {
      Str = II->getName();
    }
  }

  // Keep the first result in the case of a mangling collision.
  auto Result = Manglings.insert(std::make_pair(Str, GD));
  return FoundStr = Result.first->first();
}

StringRef CodeGenModule::getBlockMangledName(GlobalDecl GD,
                                             const BlockDecl *BD) {
  MangleContext &MangleCtx = getCXXABI().getMangleContext();
  const Decl *D = GD.getDecl();

  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  if (!D)
    MangleCtx.mangleGlobalBlock(BD, 
      dyn_cast_or_null<VarDecl>(initializedGlobalDecl.getDecl()), Out);
  else if (const auto *CD = dyn_cast<CXXConstructorDecl>(D))
    MangleCtx.mangleCtorBlock(CD, GD.getCtorType(), BD, Out);
  else if (const auto *DD = dyn_cast<CXXDestructorDecl>(D))
    MangleCtx.mangleDtorBlock(DD, GD.getDtorType(), BD, Out);
  else
    MangleCtx.mangleBlock(cast<DeclContext>(D), BD, Out);

  auto Result = Manglings.insert(std::make_pair(Out.str(), BD));
  return Result.first->first();
}

llvm::GlobalValue *CodeGenModule::GetGlobalValue(StringRef Name) {
  return getModule().getNamedValue(Name);
}

/// AddGlobalCtor - Add a function to the list that will be called before
/// main() runs.
void CodeGenModule::AddGlobalCtor(llvm::Function *Ctor, int Priority,
                                  llvm::Constant *AssociatedData) {
  // FIXME: Type coercion of void()* types.
  GlobalCtors.push_back(Structor(Priority, Ctor, AssociatedData));
}

/// AddGlobalDtor - Add a function to the list that will be called
/// when the module is unloaded.
void CodeGenModule::AddGlobalDtor(llvm::Function *Dtor, int Priority) {
  // FIXME: Type coercion of void()* types.
  GlobalDtors.push_back(Structor(Priority, Dtor, nullptr));
}

void CodeGenModule::EmitCtorList(CtorList &Fns, const char *GlobalName) {
  if (Fns.empty()) return;

  // Ctor function type is void()*.
  llvm::FunctionType* CtorFTy = llvm::FunctionType::get(VoidTy, false);
  llvm::Type *CtorPFTy = llvm::PointerType::getUnqual(CtorFTy);

  // Get the type of a ctor entry, { i32, void ()*, i8* }.
  llvm::StructType *CtorStructTy = llvm::StructType::get(
      Int32Ty, llvm::PointerType::getUnqual(CtorFTy), VoidPtrTy);

  // Construct the constructor and destructor arrays.
  ConstantInitBuilder builder(*this);
  auto ctors = builder.beginArray(CtorStructTy);
  for (const auto &I : Fns) {
    auto ctor = ctors.beginStruct(CtorStructTy);
    ctor.addInt(Int32Ty, I.Priority);
    ctor.add(llvm::ConstantExpr::getBitCast(I.Initializer, CtorPFTy));
    if (I.AssociatedData)
      ctor.add(llvm::ConstantExpr::getBitCast(I.AssociatedData, VoidPtrTy));
    else
      ctor.addNullPointer(VoidPtrTy);
    ctor.finishAndAddTo(ctors);
  }

  auto list =
    ctors.finishAndCreateGlobal(GlobalName, getPointerAlign(),
                                /*constant*/ false,
                                llvm::GlobalValue::AppendingLinkage);

  // The LTO linker doesn't seem to like it when we set an alignment
  // on appending variables.  Take it off as a workaround.
  list->setAlignment(0);

  Fns.clear();
}

llvm::GlobalValue::LinkageTypes
CodeGenModule::getFunctionLinkage(GlobalDecl GD) {
  const auto *D = cast<FunctionDecl>(GD.getDecl());

  GVALinkage Linkage = getContext().GetGVALinkageForFunction(D);

  if (isa<CXXDestructorDecl>(D) &&
      getCXXABI().useThunkForDtorVariant(cast<CXXDestructorDecl>(D),
                                         GD.getDtorType())) {
    // Destructor variants in the Microsoft C++ ABI are always internal or
    // linkonce_odr thunks emitted on an as-needed basis.
    return Linkage == GVA_Internal ? llvm::GlobalValue::InternalLinkage
                                   : llvm::GlobalValue::LinkOnceODRLinkage;
  }

  if (isa<CXXConstructorDecl>(D) &&
      cast<CXXConstructorDecl>(D)->isInheritingConstructor() &&
      Context.getTargetInfo().getCXXABI().isMicrosoft()) {
    // Our approach to inheriting constructors is fundamentally different from
    // that used by the MS ABI, so keep our inheriting constructor thunks
    // internal rather than trying to pick an unambiguous mangling for them.
    return llvm::GlobalValue::InternalLinkage;
  }

  return getLLVMLinkageForDeclarator(D, Linkage, /*isConstantVariable=*/false);
}

void CodeGenModule::setFunctionDLLStorageClass(GlobalDecl GD, llvm::Function *F) {
  const auto *FD = cast<FunctionDecl>(GD.getDecl());

  if (const auto *Dtor = dyn_cast_or_null<CXXDestructorDecl>(FD)) {
    if (getCXXABI().useThunkForDtorVariant(Dtor, GD.getDtorType())) {
      // Don't dllexport/import destructor thunks.
      F->setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);
      return;
    }
  }

  if (FD->hasAttr<DLLImportAttr>())
    F->setDLLStorageClass(llvm::GlobalVariable::DLLImportStorageClass);
  else if (FD->hasAttr<DLLExportAttr>())
    F->setDLLStorageClass(llvm::GlobalVariable::DLLExportStorageClass);
  else
    F->setDLLStorageClass(llvm::GlobalVariable::DefaultStorageClass);
}

llvm::ConstantInt *CodeGenModule::CreateCrossDsoCfiTypeId(llvm::Metadata *MD) {
  llvm::MDString *MDS = dyn_cast<llvm::MDString>(MD);
  if (!MDS) return nullptr;

  return llvm::ConstantInt::get(Int64Ty, llvm::MD5Hash(MDS->getString()));
}

void CodeGenModule::setFunctionDefinitionAttributes(const FunctionDecl *D,
                                                    llvm::Function *F) {
  setNonAliasAttributes(D, F);
}

void CodeGenModule::SetLLVMFunctionAttributes(const Decl *D,
                                              const CGFunctionInfo &Info,
                                              llvm::Function *F) {
  unsigned CallingConv;
  llvm::AttributeList PAL;
  ConstructAttributeList(F->getName(), Info, D, PAL, CallingConv, false);
  F->setAttributes(PAL);
  F->setCallingConv(static_cast<llvm::CallingConv::ID>(CallingConv));
}

/// Determines whether the language options require us to model
/// unwind exceptions.  We treat -fexceptions as mandating this
/// except under the fragile ObjC ABI with only ObjC exceptions
/// enabled.  This means, for example, that C with -fexceptions
/// enables this.
static bool hasUnwindExceptions(const LangOptions &LangOpts) {
  // If exceptions are completely disabled, obviously this is false.
  if (!LangOpts.Exceptions) return false;

  // If C++ exceptions are enabled, this is true.
  if (LangOpts.CXXExceptions) return true;

  // If ObjC exceptions are enabled, this depends on the ABI.
  if (LangOpts.ObjCExceptions) {
    return LangOpts.ObjCRuntime.hasUnwindExceptions();
  }

  return true;
}

void CodeGenModule::SetLLVMFunctionAttributesForDefinition(const Decl *D,
                                                           llvm::Function *F) {
  llvm::AttrBuilder B;

  if (CodeGenOpts.UnwindTables)
    B.addAttribute(llvm::Attribute::UWTable);

  if (!hasUnwindExceptions(LangOpts))
    B.addAttribute(llvm::Attribute::NoUnwind);

  if (LangOpts.getStackProtector() == LangOptions::SSPOn)
    B.addAttribute(llvm::Attribute::StackProtect);
  else if (LangOpts.getStackProtector() == LangOptions::SSPStrong)
    B.addAttribute(llvm::Attribute::StackProtectStrong);
  else if (LangOpts.getStackProtector() == LangOptions::SSPReq)
    B.addAttribute(llvm::Attribute::StackProtectReq);

  if (!D) {
    // If we don't have a declaration to control inlining, the function isn't
    // explicitly marked as alwaysinline for semantic reasons, and inlining is
    // disabled, mark the function as noinline.
    if (!F->hasFnAttribute(llvm::Attribute::AlwaysInline) &&
        CodeGenOpts.getInlining() == CodeGenOptions::OnlyAlwaysInlining)
      B.addAttribute(llvm::Attribute::NoInline);

    F->addAttributes(llvm::AttributeList::FunctionIndex, B);
    return;
  }

  // Track whether we need to add the optnone LLVM attribute,
  // starting with the default for this optimization level.
  bool ShouldAddOptNone =
      !CodeGenOpts.DisableO0ImplyOptNone && CodeGenOpts.OptimizationLevel == 0;
  // We can't add optnone in the following cases, it won't pass the verifier.
  ShouldAddOptNone &= !D->hasAttr<MinSizeAttr>();
  ShouldAddOptNone &= !F->hasFnAttribute(llvm::Attribute::AlwaysInline);
  ShouldAddOptNone &= !D->hasAttr<AlwaysInlineAttr>();

  if (ShouldAddOptNone || D->hasAttr<OptimizeNoneAttr>()) {
    B.addAttribute(llvm::Attribute::OptimizeNone);

    // OptimizeNone implies noinline; we should not be inlining such functions.
    B.addAttribute(llvm::Attribute::NoInline);
    assert(!F->hasFnAttribute(llvm::Attribute::AlwaysInline) &&
           "OptimizeNone and AlwaysInline on same function!");

    // We still need to handle naked functions even though optnone subsumes
    // much of their semantics.
    if (D->hasAttr<NakedAttr>())
      B.addAttribute(llvm::Attribute::Naked);

    // OptimizeNone wins over OptimizeForSize and MinSize.
    F->removeFnAttr(llvm::Attribute::OptimizeForSize);
    F->removeFnAttr(llvm::Attribute::MinSize);
  } else if (D->hasAttr<NakedAttr>()) {
    // Naked implies noinline: we should not be inlining such functions.
    B.addAttribute(llvm::Attribute::Naked);
    B.addAttribute(llvm::Attribute::NoInline);
  } else if (D->hasAttr<NoDuplicateAttr>()) {
    B.addAttribute(llvm::Attribute::NoDuplicate);
  } else if (D->hasAttr<NoInlineAttr>()) {
    B.addAttribute(llvm::Attribute::NoInline);
  } else if (D->hasAttr<AlwaysInlineAttr>() &&
             !F->hasFnAttribute(llvm::Attribute::NoInline)) {
    // (noinline wins over always_inline, and we can't specify both in IR)
    B.addAttribute(llvm::Attribute::AlwaysInline);
  } else if (CodeGenOpts.getInlining() == CodeGenOptions::OnlyAlwaysInlining) {
    // If we're not inlining, then force everything that isn't always_inline to
    // carry an explicit noinline attribute.
    if (!F->hasFnAttribute(llvm::Attribute::AlwaysInline))
      B.addAttribute(llvm::Attribute::NoInline);
  } else {
    // Otherwise, propagate the inline hint attribute and potentially use its
    // absence to mark things as noinline.
    if (auto *FD = dyn_cast<FunctionDecl>(D)) {
      if (any_of(FD->redecls(), [&](const FunctionDecl *Redecl) {
            return Redecl->isInlineSpecified();
          })) {
        B.addAttribute(llvm::Attribute::InlineHint);
      } else if (CodeGenOpts.getInlining() ==
                     CodeGenOptions::OnlyHintInlining &&
                 !FD->isInlined() &&
                 !F->hasFnAttribute(llvm::Attribute::AlwaysInline)) {
        B.addAttribute(llvm::Attribute::NoInline);
      }
    }
  }

  // Add other optimization related attributes if we are optimizing this
  // function.
  if (!D->hasAttr<OptimizeNoneAttr>()) {
    if (D->hasAttr<ColdAttr>()) {
      if (!ShouldAddOptNone)
        B.addAttribute(llvm::Attribute::OptimizeForSize);
      B.addAttribute(llvm::Attribute::Cold);
    }

    if (D->hasAttr<MinSizeAttr>())
      B.addAttribute(llvm::Attribute::MinSize);
  }

  F->addAttributes(llvm::AttributeList::FunctionIndex, B);

  unsigned alignment = D->getMaxAlignment() / Context.getCharWidth();
  if (alignment)
    F->setAlignment(alignment);

  // Some C++ ABIs require 2-byte alignment for member functions, in order to
  // reserve a bit for differentiating between virtual and non-virtual member
  // functions. If the current target's C++ ABI requires this and this is a
  // member function, set its alignment accordingly.
  if (getTarget().getCXXABI().areMemberFunctionsAligned()) {
    if (F->getAlignment() < 2 && isa<CXXMethodDecl>(D))
      F->setAlignment(2);
  }

  // In the cross-dso CFI mode, we want !type attributes on definitions only.
  if (CodeGenOpts.SanitizeCfiCrossDso)
    if (auto *FD = dyn_cast<FunctionDecl>(D))
      CreateFunctionTypeMetadata(FD, F);
}

void CodeGenModule::SetCommonAttributes(const Decl *D,
                                        llvm::GlobalValue *GV) {
  if (const auto *ND = dyn_cast_or_null<NamedDecl>(D))
    setGlobalVisibility(GV, ND);
  else
    GV->setVisibility(llvm::GlobalValue::DefaultVisibility);

  if (D && D->hasAttr<UsedAttr>())
    addUsedGlobal(GV);
}

void CodeGenModule::setAliasAttributes(const Decl *D,
                                       llvm::GlobalValue *GV) {
  SetCommonAttributes(D, GV);

  // Process the dllexport attribute based on whether the original definition
  // (not necessarily the aliasee) was exported.
  if (D->hasAttr<DLLExportAttr>())
    GV->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
}

void CodeGenModule::setNonAliasAttributes(const Decl *D,
                                          llvm::GlobalObject *GO) {
  SetCommonAttributes(D, GO);

  if (D) {
    if (auto *GV = dyn_cast<llvm::GlobalVariable>(GO)) {
      if (auto *SA = D->getAttr<PragmaClangBSSSectionAttr>())
        GV->addAttribute("bss-section", SA->getName());
      if (auto *SA = D->getAttr<PragmaClangDataSectionAttr>())
        GV->addAttribute("data-section", SA->getName());
      if (auto *SA = D->getAttr<PragmaClangRodataSectionAttr>())
        GV->addAttribute("rodata-section", SA->getName());
    }

    if (auto *F = dyn_cast<llvm::Function>(GO)) {
      if (auto *SA = D->getAttr<PragmaClangTextSectionAttr>())
       if (!D->getAttr<SectionAttr>())
         F->addFnAttr("implicit-section-name", SA->getName());
    }

    if (const SectionAttr *SA = D->getAttr<SectionAttr>())
      GO->setSection(SA->getName());
  }

  getTargetCodeGenInfo().setTargetAttributes(D, GO, *this);
}

void CodeGenModule::SetInternalFunctionAttributes(const Decl *D,
                                                  llvm::Function *F,
                                                  const CGFunctionInfo &FI) {
  SetLLVMFunctionAttributes(D, FI, F);
  SetLLVMFunctionAttributesForDefinition(D, F);

  F->setLinkage(llvm::Function::InternalLinkage);

  setNonAliasAttributes(D, F);
}

static void setLinkageAndVisibilityForGV(llvm::GlobalValue *GV,
                                         const NamedDecl *ND) {
  // Set linkage and visibility in case we never see a definition.
  LinkageInfo LV = ND->getLinkageAndVisibility();
  if (LV.getLinkage() != ExternalLinkage) {
    // Don't set internal linkage on declarations.
  } else {
    if (ND->hasAttr<DLLImportAttr>()) {
      GV->setLinkage(llvm::GlobalValue::ExternalLinkage);
      GV->setDLLStorageClass(llvm::GlobalValue::DLLImportStorageClass);
    } else if (ND->hasAttr<DLLExportAttr>()) {
      GV->setLinkage(llvm::GlobalValue::ExternalLinkage);
    } else if (ND->hasAttr<WeakAttr>() || ND->isWeakImported()) {
      // "extern_weak" is overloaded in LLVM; we probably should have
      // separate linkage types for this.
      GV->setLinkage(llvm::GlobalValue::ExternalWeakLinkage);
    }

    // Set visibility on a declaration only if it's explicit.
    if (LV.isVisibilityExplicit())
      GV->setVisibility(CodeGenModule::GetLLVMVisibility(LV.getVisibility()));
  }
}

void CodeGenModule::CreateFunctionTypeMetadata(const FunctionDecl *FD,
                                               llvm::Function *F) {
  // Only if we are checking indirect calls.
  if (!LangOpts.Sanitize.has(SanitizerKind::CFIICall))
    return;

  // Non-static class methods are handled via vtable pointer checks elsewhere.
  if (isa<CXXMethodDecl>(FD) && !cast<CXXMethodDecl>(FD)->isStatic())
    return;

  // Additionally, if building with cross-DSO support...
  if (CodeGenOpts.SanitizeCfiCrossDso) {
    // Skip available_externally functions. They won't be codegen'ed in the
    // current module anyway.
    if (getContext().GetGVALinkageForFunction(FD) == GVA_AvailableExternally)
      return;
  }

  llvm::Metadata *MD = CreateMetadataIdentifierForType(FD->getType());
  F->addTypeMetadata(0, MD);

  // Emit a hash-based bit set entry for cross-DSO calls.
  if (CodeGenOpts.SanitizeCfiCrossDso)
    if (auto CrossDsoTypeId = CreateCrossDsoCfiTypeId(MD))
      F->addTypeMetadata(0, llvm::ConstantAsMetadata::get(CrossDsoTypeId));
}

void CodeGenModule::SetFunctionAttributes(GlobalDecl GD, llvm::Function *F,
                                          bool IsIncompleteFunction,
                                          bool IsThunk) {
  if (llvm::Intrinsic::ID IID = F->getIntrinsicID()) {
    // If this is an intrinsic function, set the function's attributes
    // to the intrinsic's attributes.
    F->setAttributes(llvm::Intrinsic::getAttributes(getLLVMContext(), IID));
    return;
  }

  const auto *FD = cast<FunctionDecl>(GD.getDecl());

  if (!IsIncompleteFunction)
    SetLLVMFunctionAttributes(FD, getTypes().arrangeGlobalDeclaration(GD), F);

  // Add the Returned attribute for "this", except for iOS 5 and earlier
  // where substantial code, including the libstdc++ dylib, was compiled with
  // GCC and does not actually return "this".
  if (!IsThunk && getCXXABI().HasThisReturn(GD) &&
      !(getTriple().isiOS() && getTriple().isOSVersionLT(6))) {
    assert(!F->arg_empty() &&
           F->arg_begin()->getType()
             ->canLosslesslyBitCastTo(F->getReturnType()) &&
           "unexpected this return");
    F->addAttribute(1, llvm::Attribute::Returned);
  }

  // Only a few attributes are set on declarations; these may later be
  // overridden by a definition.

  setLinkageAndVisibilityForGV(F, FD);

  if (FD->getAttr<PragmaClangTextSectionAttr>()) {
    F->addFnAttr("implicit-section-name");
  }

  if (const SectionAttr *SA = FD->getAttr<SectionAttr>())
    F->setSection(SA->getName());

  if (FD->isReplaceableGlobalAllocationFunction()) {
    // A replaceable global allocation function does not act like a builtin by
    // default, only if it is invoked by a new-expression or delete-expression.
    F->addAttribute(llvm::AttributeList::FunctionIndex,
                    llvm::Attribute::NoBuiltin);

    // A sane operator new returns a non-aliasing pointer.
    // FIXME: Also add NonNull attribute to the return value
    // for the non-nothrow forms?
    auto Kind = FD->getDeclName().getCXXOverloadedOperator();
    if (getCodeGenOpts().AssumeSaneOperatorNew &&
        (Kind == OO_New || Kind == OO_Array_New))
      F->addAttribute(llvm::AttributeList::ReturnIndex,
                      llvm::Attribute::NoAlias);
  }

  if (isa<CXXConstructorDecl>(FD) || isa<CXXDestructorDecl>(FD))
    F->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  else if (const auto *MD = dyn_cast<CXXMethodDecl>(FD))
    if (MD->isVirtual())
      F->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

  // Don't emit entries for function declarations in the cross-DSO mode. This
  // is handled with better precision by the receiving DSO.
  if (!CodeGenOpts.SanitizeCfiCrossDso)
    CreateFunctionTypeMetadata(FD, F);
}

void CodeGenModule::addUsedGlobal(llvm::GlobalValue *GV) {
  assert(!GV->isDeclaration() &&
         "Only globals with definition can force usage.");
  LLVMUsed.emplace_back(GV);
}

void CodeGenModule::addCompilerUsedGlobal(llvm::GlobalValue *GV) {
  assert(!GV->isDeclaration() &&
         "Only globals with definition can force usage.");
  LLVMCompilerUsed.emplace_back(GV);
}

static void emitUsed(CodeGenModule &CGM, StringRef Name,
                     std::vector<llvm::WeakTrackingVH> &List) {
  // Don't create llvm.used if there is no need.
  if (List.empty())
    return;

  // Convert List to what ConstantArray needs.
  SmallVector<llvm::Constant*, 8> UsedArray;
  UsedArray.resize(List.size());
  for (unsigned i = 0, e = List.size(); i != e; ++i) {
    UsedArray[i] =
        llvm::ConstantExpr::getPointerBitCastOrAddrSpaceCast(
            cast<llvm::Constant>(&*List[i]), CGM.Int8PtrTy);
  }

  if (UsedArray.empty())
    return;
  llvm::ArrayType *ATy = llvm::ArrayType::get(CGM.Int8PtrTy, UsedArray.size());

  auto *GV = new llvm::GlobalVariable(
      CGM.getModule(), ATy, false, llvm::GlobalValue::AppendingLinkage,
      llvm::ConstantArray::get(ATy, UsedArray), Name);

  GV->setSection("llvm.metadata");
}

void CodeGenModule::emitLLVMUsed() {
  emitUsed(*this, "llvm.used", LLVMUsed);
  emitUsed(*this, "llvm.compiler.used", LLVMCompilerUsed);
}

void CodeGenModule::AppendLinkerOptions(StringRef Opts) {
  auto *MDOpts = llvm::MDString::get(getLLVMContext(), Opts);
  LinkerOptionsMetadata.push_back(llvm::MDNode::get(getLLVMContext(), MDOpts));
}

void CodeGenModule::AddDetectMismatch(StringRef Name, StringRef Value) {
  llvm::SmallString<32> Opt;
  getTargetCodeGenInfo().getDetectMismatchOption(Name, Value, Opt);
  auto *MDOpts = llvm::MDString::get(getLLVMContext(), Opt);
  LinkerOptionsMetadata.push_back(llvm::MDNode::get(getLLVMContext(), MDOpts));
}

void CodeGenModule::AddDependentLib(StringRef Lib) {
  llvm::SmallString<24> Opt;
  getTargetCodeGenInfo().getDependentLibraryOption(Lib, Opt);
  auto *MDOpts = llvm::MDString::get(getLLVMContext(), Opt);
  LinkerOptionsMetadata.push_back(llvm::MDNode::get(getLLVMContext(), MDOpts));
}

/// \brief Add link options implied by the given module, including modules
/// it depends on, using a postorder walk.
static void addLinkOptionsPostorder(CodeGenModule &CGM, Module *Mod,
                                    SmallVectorImpl<llvm::MDNode *> &Metadata,
                                    llvm::SmallPtrSet<Module *, 16> &Visited) {
  // Import this module's parent.
  if (Mod->Parent && Visited.insert(Mod->Parent).second) {
    addLinkOptionsPostorder(CGM, Mod->Parent, Metadata, Visited);
  }

  // Import this module's dependencies.
  for (unsigned I = Mod->Imports.size(); I > 0; --I) {
    if (Visited.insert(Mod->Imports[I - 1]).second)
      addLinkOptionsPostorder(CGM, Mod->Imports[I-1], Metadata, Visited);
  }

  // Add linker options to link against the libraries/frameworks
  // described by this module.
  llvm::LLVMContext &Context = CGM.getLLVMContext();
  for (unsigned I = Mod->LinkLibraries.size(); I > 0; --I) {
    // Link against a framework.  Frameworks are currently Darwin only, so we
    // don't to ask TargetCodeGenInfo for the spelling of the linker option.
    if (Mod->LinkLibraries[I-1].IsFramework) {
      llvm::Metadata *Args[2] = {
          llvm::MDString::get(Context, "-framework"),
          llvm::MDString::get(Context, Mod->LinkLibraries[I - 1].Library)};

      Metadata.push_back(llvm::MDNode::get(Context, Args));
      continue;
    }

    // Link against a library.
    llvm::SmallString<24> Opt;
    CGM.getTargetCodeGenInfo().getDependentLibraryOption(
      Mod->LinkLibraries[I-1].Library, Opt);
    auto *OptString = llvm::MDString::get(Context, Opt);
    Metadata.push_back(llvm::MDNode::get(Context, OptString));
  }
}

void CodeGenModule::EmitModuleLinkOptions() {
  // Collect the set of all of the modules we want to visit to emit link
  // options, which is essentially the imported modules and all of their
  // non-explicit child modules.
  llvm::SetVector<clang::Module *> LinkModules;
  llvm::SmallPtrSet<clang::Module *, 16> Visited;
  SmallVector<clang::Module *, 16> Stack;

  // Seed the stack with imported modules.
  for (Module *M : ImportedModules) {
    // Do not add any link flags when an implementation TU of a module imports
    // a header of that same module.
    if (M->getTopLevelModuleName() == getLangOpts().CurrentModule &&
        !getLangOpts().isCompilingModule())
      continue;
    if (Visited.insert(M).second)
      Stack.push_back(M);
  }

  // Find all of the modules to import, making a little effort to prune
  // non-leaf modules.
  while (!Stack.empty()) {
    clang::Module *Mod = Stack.pop_back_val();

    bool AnyChildren = false;

    // Visit the submodules of this module.
    for (clang::Module::submodule_iterator Sub = Mod->submodule_begin(),
                                        SubEnd = Mod->submodule_end();
         Sub != SubEnd; ++Sub) {
      // Skip explicit children; they need to be explicitly imported to be
      // linked against.
      if ((*Sub)->IsExplicit)
        continue;

      if (Visited.insert(*Sub).second) {
        Stack.push_back(*Sub);
        AnyChildren = true;
      }
    }

    // We didn't find any children, so add this module to the list of
    // modules to link against.
    if (!AnyChildren) {
      LinkModules.insert(Mod);
    }
  }

  // Add link options for all of the imported modules in reverse topological
  // order.  We don't do anything to try to order import link flags with respect
  // to linker options inserted by things like #pragma comment().
  SmallVector<llvm::MDNode *, 16> MetadataArgs;
  Visited.clear();
  for (Module *M : LinkModules)
    if (Visited.insert(M).second)
      addLinkOptionsPostorder(*this, M, MetadataArgs, Visited);
  std::reverse(MetadataArgs.begin(), MetadataArgs.end());
  LinkerOptionsMetadata.append(MetadataArgs.begin(), MetadataArgs.end());

  // Add the linker options metadata flag.
  auto *NMD = getModule().getOrInsertNamedMetadata("llvm.linker.options");
  for (auto *MD : LinkerOptionsMetadata)
    NMD->addOperand(MD);
}

void CodeGenModule::EmitDeferred() {
  // Emit code for any potentially referenced deferred decls.  Since a
  // previously unused static decl may become used during the generation of code
  // for a static function, iterate until no changes are made.

  if (!DeferredVTables.empty()) {
    EmitDeferredVTables();

    // Emitting a vtable doesn't directly cause more vtables to
    // become deferred, although it can cause functions to be
    // emitted that then need those vtables.
    assert(DeferredVTables.empty());
  }

  // Stop if we're out of both deferred vtables and deferred declarations.
  if (DeferredDeclsToEmit.empty())
    return;

  // Grab the list of decls to emit. If EmitGlobalDefinition schedules more
  // work, it will not interfere with this.
  std::vector<GlobalDecl> CurDeclsToEmit;
  CurDeclsToEmit.swap(DeferredDeclsToEmit);

  for (GlobalDecl &D : CurDeclsToEmit) {
    // We should call GetAddrOfGlobal with IsForDefinition set to true in order
    // to get GlobalValue with exactly the type we need, not something that
    // might had been created for another decl with the same mangled name but
    // different type.
    llvm::GlobalValue *GV = dyn_cast<llvm::GlobalValue>(
        GetAddrOfGlobal(D, ForDefinition));

    // In case of different address spaces, we may still get a cast, even with
    // IsForDefinition equal to true. Query mangled names table to get
    // GlobalValue.
    if (!GV)
      GV = GetGlobalValue(getMangledName(D));

    // Make sure GetGlobalValue returned non-null.
    assert(GV);

    // Check to see if we've already emitted this.  This is necessary
    // for a couple of reasons: first, decls can end up in the
    // deferred-decls queue multiple times, and second, decls can end
    // up with definitions in unusual ways (e.g. by an extern inline
    // function acquiring a strong function redefinition).  Just
    // ignore these cases.
    if (!GV->isDeclaration())
      continue;

    // Otherwise, emit the definition and move on to the next one.
    EmitGlobalDefinition(D, GV);

    // If we found out that we need to emit more decls, do that recursively.
    // This has the advantage that the decls are emitted in a DFS and related
    // ones are close together, which is convenient for testing.
    if (!DeferredVTables.empty() || !DeferredDeclsToEmit.empty()) {
      EmitDeferred();
      assert(DeferredVTables.empty() && DeferredDeclsToEmit.empty());
    }
  }
}

void CodeGenModule::EmitVTablesOpportunistically() {
  // Try to emit external vtables as available_externally if they have emitted
  // all inlined virtual functions.  It runs after EmitDeferred() and therefore
  // is not allowed to create new references to things that need to be emitted
  // lazily. Note that it also uses fact that we eagerly emitting RTTI.

  assert((OpportunisticVTables.empty() || shouldOpportunisticallyEmitVTables())
         && "Only emit opportunistic vtables with optimizations");

  for (const CXXRecordDecl *RD : OpportunisticVTables) {
    assert(getVTables().isVTableExternal(RD) &&
           "This queue should only contain external vtables");
    if (getCXXABI().canSpeculativelyEmitVTable(RD))
      VTables.GenerateClassData(RD);
  }
  OpportunisticVTables.clear();
}

void CodeGenModule::EmitGlobalAnnotations() {
  if (Annotations.empty())
    return;

  // Create a new global variable for the ConstantStruct in the Module.
  llvm::Constant *Array = llvm::ConstantArray::get(llvm::ArrayType::get(
    Annotations[0]->getType(), Annotations.size()), Annotations);
  auto *gv = new llvm::GlobalVariable(getModule(), Array->getType(), false,
                                      llvm::GlobalValue::AppendingLinkage,
                                      Array, "llvm.global.annotations");
  gv->setSection(AnnotationSection);
}

llvm::Constant *CodeGenModule::EmitAnnotationString(StringRef Str) {
  llvm::Constant *&AStr = AnnotationStrings[Str];
  if (AStr)
    return AStr;

  // Not found yet, create a new global.
  llvm::Constant *s = llvm::ConstantDataArray::getString(getLLVMContext(), Str);
  auto *gv =
      new llvm::GlobalVariable(getModule(), s->getType(), true,
                               llvm::GlobalValue::PrivateLinkage, s, ".str");
  gv->setSection(AnnotationSection);
  gv->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  AStr = gv;
  return gv;
}

llvm::Constant *CodeGenModule::EmitAnnotationUnit(SourceLocation Loc) {
  SourceManager &SM = getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(Loc);
  if (PLoc.isValid())
    return EmitAnnotationString(PLoc.getFilename());
  return EmitAnnotationString(SM.getBufferName(Loc));
}

llvm::Constant *CodeGenModule::EmitAnnotationLineNo(SourceLocation L) {
  SourceManager &SM = getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(L);
  unsigned LineNo = PLoc.isValid() ? PLoc.getLine() :
    SM.getExpansionLineNumber(L);
  return llvm::ConstantInt::get(Int32Ty, LineNo);
}

llvm::Constant *CodeGenModule::EmitAnnotateAttr(llvm::GlobalValue *GV,
                                                const AnnotateAttr *AA,
                                                SourceLocation L) {
  // Get the globals for file name, annotation, and the line number.
  llvm::Constant *AnnoGV = EmitAnnotationString(AA->getAnnotation()),
                 *UnitGV = EmitAnnotationUnit(L),
                 *LineNoCst = EmitAnnotationLineNo(L);

  // Create the ConstantStruct for the global annotation.
  llvm::Constant *Fields[4] = {
    llvm::ConstantExpr::getBitCast(GV, Int8PtrTy),
    llvm::ConstantExpr::getBitCast(AnnoGV, Int8PtrTy),
    llvm::ConstantExpr::getBitCast(UnitGV, Int8PtrTy),
    LineNoCst
  };
  return llvm::ConstantStruct::getAnon(Fields);
}

void CodeGenModule::AddGlobalAnnotations(const ValueDecl *D,
                                         llvm::GlobalValue *GV) {
  assert(D->hasAttr<AnnotateAttr>() && "no annotate attribute");
  // Get the struct elements for these annotations.
  for (const auto *I : D->specific_attrs<AnnotateAttr>())
    Annotations.push_back(EmitAnnotateAttr(GV, I, D->getLocation()));
}

bool CodeGenModule::isInSanitizerBlacklist(llvm::Function *Fn,
                                           SourceLocation Loc) const {
  const auto &SanitizerBL = getContext().getSanitizerBlacklist();
  // Blacklist by function name.
  if (SanitizerBL.isBlacklistedFunction(Fn->getName()))
    return true;
  // Blacklist by location.
  if (Loc.isValid())
    return SanitizerBL.isBlacklistedLocation(Loc);
  // If location is unknown, this may be a compiler-generated function. Assume
  // it's located in the main file.
  auto &SM = Context.getSourceManager();
  if (const auto *MainFile = SM.getFileEntryForID(SM.getMainFileID())) {
    return SanitizerBL.isBlacklistedFile(MainFile->getName());
  }
  return false;
}

bool CodeGenModule::isInSanitizerBlacklist(llvm::GlobalVariable *GV,
                                           SourceLocation Loc, QualType Ty,
                                           StringRef Category) const {
  // For now globals can be blacklisted only in ASan and KASan.
  if (!LangOpts.Sanitize.hasOneOf(
          SanitizerKind::Address | SanitizerKind::KernelAddress))
    return false;
  const auto &SanitizerBL = getContext().getSanitizerBlacklist();
  if (SanitizerBL.isBlacklistedGlobal(GV->getName(), Category))
    return true;
  if (SanitizerBL.isBlacklistedLocation(Loc, Category))
    return true;
  // Check global type.
  if (!Ty.isNull()) {
    // Drill down the array types: if global variable of a fixed type is
    // blacklisted, we also don't instrument arrays of them.
    while (auto AT = dyn_cast<ArrayType>(Ty.getTypePtr()))
      Ty = AT->getElementType();
    Ty = Ty.getCanonicalType().getUnqualifiedType();
    // We allow to blacklist only record types (classes, structs etc.)
    if (Ty->isRecordType()) {
      std::string TypeStr = Ty.getAsString(getContext().getPrintingPolicy());
      if (SanitizerBL.isBlacklistedType(TypeStr, Category))
        return true;
    }
  }
  return false;
}

bool CodeGenModule::imbueXRayAttrs(llvm::Function *Fn, SourceLocation Loc,
                                   StringRef Category) const {
  if (!LangOpts.XRayInstrument)
    return false;
  const auto &XRayFilter = getContext().getXRayFilter();
  using ImbueAttr = XRayFunctionFilter::ImbueAttribute;
  auto Attr = XRayFunctionFilter::ImbueAttribute::NONE;
  if (Loc.isValid())
    Attr = XRayFilter.shouldImbueLocation(Loc, Category);
  if (Attr == ImbueAttr::NONE)
    Attr = XRayFilter.shouldImbueFunction(Fn->getName());
  switch (Attr) {
  case ImbueAttr::NONE:
    return false;
  case ImbueAttr::ALWAYS:
    Fn->addFnAttr("function-instrument", "xray-always");
    break;
  case ImbueAttr::ALWAYS_ARG1:
    Fn->addFnAttr("function-instrument", "xray-always");
    Fn->addFnAttr("xray-log-args", "1");
    break;
  case ImbueAttr::NEVER:
    Fn->addFnAttr("function-instrument", "xray-never");
    break;
  }
  return true;
}

bool CodeGenModule::MustBeEmitted(const ValueDecl *Global) {
  // Never defer when EmitAllDecls is specified.
  if (LangOpts.EmitAllDecls)
    return true;

  return getContext().DeclMustBeEmitted(Global);
}

bool CodeGenModule::MayBeEmittedEagerly(const ValueDecl *Global) {
  if (const auto *FD = dyn_cast<FunctionDecl>(Global))
    if (FD->getTemplateSpecializationKind() == TSK_ImplicitInstantiation)
      // Implicit template instantiations may change linkage if they are later
      // explicitly instantiated, so they should not be emitted eagerly.
      return false;
  if (const auto *VD = dyn_cast<VarDecl>(Global))
    if (Context.getInlineVariableDefinitionKind(VD) ==
        ASTContext::InlineVariableDefinitionKind::WeakUnknown)
      // A definition of an inline constexpr static data member may change
      // linkage later if it's redeclared outside the class.
      return false;
  // If OpenMP is enabled and threadprivates must be generated like TLS, delay
  // codegen for global variables, because they may be marked as threadprivate.
  if (LangOpts.OpenMP && LangOpts.OpenMPUseTLS &&
      getContext().getTargetInfo().isTLSSupported() && isa<VarDecl>(Global))
    return false;

  return true;
}

ConstantAddress CodeGenModule::GetAddrOfUuidDescriptor(
    const CXXUuidofExpr* E) {
  // Sema has verified that IIDSource has a __declspec(uuid()), and that its
  // well-formed.
  StringRef Uuid = E->getUuidStr();
  std::string Name = "_GUID_" + Uuid.lower();
  std::replace(Name.begin(), Name.end(), '-', '_');

  // The UUID descriptor should be pointer aligned.
  CharUnits Alignment = CharUnits::fromQuantity(PointerAlignInBytes);

  // Look for an existing global.
  if (llvm::GlobalVariable *GV = getModule().getNamedGlobal(Name))
    return ConstantAddress(GV, Alignment);

  llvm::Constant *Init = EmitUuidofInitializer(Uuid);
  assert(Init && "failed to initialize as constant");

  auto *GV = new llvm::GlobalVariable(
      getModule(), Init->getType(),
      /*isConstant=*/true, llvm::GlobalValue::LinkOnceODRLinkage, Init, Name);
  if (supportsCOMDAT())
    GV->setComdat(TheModule.getOrInsertComdat(GV->getName()));
  return ConstantAddress(GV, Alignment);
}

ConstantAddress CodeGenModule::GetWeakRefReference(const ValueDecl *VD) {
  const AliasAttr *AA = VD->getAttr<AliasAttr>();
  assert(AA && "No alias?");

  CharUnits Alignment = getContext().getDeclAlign(VD);
  llvm::Type *DeclTy = getTypes().ConvertTypeForMem(VD->getType());

  // See if there is already something with the target's name in the module.
  llvm::GlobalValue *Entry = GetGlobalValue(AA->getAliasee());
  if (Entry) {
    unsigned AS = getContext().getTargetAddressSpace(VD->getType());
    auto Ptr = llvm::ConstantExpr::getBitCast(Entry, DeclTy->getPointerTo(AS));
    return ConstantAddress(Ptr, Alignment);
  }

  llvm::Constant *Aliasee;
  if (isa<llvm::FunctionType>(DeclTy))
    Aliasee = GetOrCreateLLVMFunction(AA->getAliasee(), DeclTy,
                                      GlobalDecl(cast<FunctionDecl>(VD)),
                                      /*ForVTable=*/false);
  else
    Aliasee = GetOrCreateLLVMGlobal(AA->getAliasee(),
                                    llvm::PointerType::getUnqual(DeclTy),
                                    nullptr);

  auto *F = cast<llvm::GlobalValue>(Aliasee);
  F->setLinkage(llvm::Function::ExternalWeakLinkage);
  WeakRefReferences.insert(F);

  return ConstantAddress(Aliasee, Alignment);
}

void CodeGenModule::EmitGlobal(GlobalDecl GD) {
  const auto *Global = cast<ValueDecl>(GD.getDecl());

  // Weak references don't produce any output by themselves.
  if (Global->hasAttr<WeakRefAttr>())
    return;

  // If this is an alias definition (which otherwise looks like a declaration)
  // emit it now.
  if (Global->hasAttr<AliasAttr>())
    return EmitAliasDefinition(GD);

  // IFunc like an alias whose value is resolved at runtime by calling resolver.
  if (Global->hasAttr<IFuncAttr>())
    return emitIFuncDefinition(GD);

  // If this is CUDA, be selective about which declarations we emit.
  if (LangOpts.CUDA) {
    if (LangOpts.CUDAIsDevice) {
      if (!Global->hasAttr<CUDADeviceAttr>() &&
          !Global->hasAttr<CUDAGlobalAttr>() &&
          !Global->hasAttr<CUDAConstantAttr>() &&
          !Global->hasAttr<CUDASharedAttr>())
        return;
    } else {
      // We need to emit host-side 'shadows' for all global
      // device-side variables because the CUDA runtime needs their
      // size and host-side address in order to provide access to
      // their device-side incarnations.

      // So device-only functions are the only things we skip.
      if (isa<FunctionDecl>(Global) && !Global->hasAttr<CUDAHostAttr>() &&
          Global->hasAttr<CUDADeviceAttr>())
        return;

      assert((isa<FunctionDecl>(Global) || isa<VarDecl>(Global)) &&
             "Expected Variable or Function");
    }
  }

  if (LangOpts.OpenMP) {
    // If this is OpenMP device, check if it is legal to emit this global
    // normally.
    if (OpenMPRuntime && OpenMPRuntime->emitTargetGlobal(GD))
      return;
    if (auto *DRD = dyn_cast<OMPDeclareReductionDecl>(Global)) {
      if (MustBeEmitted(Global))
        EmitOMPDeclareReduction(DRD);
      return;
    }
  }

  // Ignore declarations, they will be emitted on their first use.
  if (const auto *FD = dyn_cast<FunctionDecl>(Global)) {
    // Forward declarations are emitted lazily on first use.
    if (!FD->doesThisDeclarationHaveABody()) {
      if (!FD->doesDeclarationForceExternallyVisibleDefinition())
        return;

      StringRef MangledName = getMangledName(GD);

      // Compute the function info and LLVM type.
      const CGFunctionInfo &FI = getTypes().arrangeGlobalDeclaration(GD);
      llvm::Type *Ty = getTypes().GetFunctionType(FI);

      GetOrCreateLLVMFunction(MangledName, Ty, GD, /*ForVTable=*/false,
                              /*DontDefer=*/false);
      return;
    }
  } else {
    const auto *VD = cast<VarDecl>(Global);
    assert(VD->isFileVarDecl() && "Cannot emit local var decl as global.");
    // We need to emit device-side global CUDA variables even if a
    // variable does not have a definition -- we still need to define
    // host-side shadow for it.
    bool MustEmitForCuda = LangOpts.CUDA && !LangOpts.CUDAIsDevice &&
                           !VD->hasDefinition() &&
                           (VD->hasAttr<CUDAConstantAttr>() ||
                            VD->hasAttr<CUDADeviceAttr>());
    if (!MustEmitForCuda &&
        VD->isThisDeclarationADefinition() != VarDecl::Definition &&
        !Context.isMSStaticDataMemberInlineDefinition(VD)) {
      // If this declaration may have caused an inline variable definition to
      // change linkage, make sure that it's emitted.
      if (Context.getInlineVariableDefinitionKind(VD) ==
          ASTContext::InlineVariableDefinitionKind::Strong)
        GetAddrOfGlobalVar(VD);
      return;
    }
  }

  // Defer code generation to first use when possible, e.g. if this is an inline
  // function. If the global must always be emitted, do it eagerly if possible
  // to benefit from cache locality.
  if (MustBeEmitted(Global) && MayBeEmittedEagerly(Global)) {
    // Emit the definition if it can't be deferred.
    EmitGlobalDefinition(GD);
    return;
  }

  // If we're deferring emission of a C++ variable with an
  // initializer, remember the order in which it appeared in the file.
  if (getLangOpts().CPlusPlus && isa<VarDecl>(Global) &&
      cast<VarDecl>(Global)->hasInit()) {
    DelayedCXXInitPosition[Global] = CXXGlobalInits.size();
    CXXGlobalInits.push_back(nullptr);
  }

  StringRef MangledName = getMangledName(GD);
  if (GetGlobalValue(MangledName) != nullptr) {
    // The value has already been used and should therefore be emitted.
    addDeferredDeclToEmit(GD);
  } else if (MustBeEmitted(Global)) {
    // The value must be emitted, but cannot be emitted eagerly.
    assert(!MayBeEmittedEagerly(Global));
    addDeferredDeclToEmit(GD);
  } else {
    // Otherwise, remember that we saw a deferred decl with this name.  The
    // first use of the mangled name will cause it to move into
    // DeferredDeclsToEmit.
    DeferredDecls[MangledName] = GD;
  }
}

// Check if T is a class type with a destructor that's not dllimport.
static bool HasNonDllImportDtor(QualType T) {
  if (const auto *RT = T->getBaseElementTypeUnsafe()->getAs<RecordType>())
    if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl()))
      if (RD->getDestructor() && !RD->getDestructor()->hasAttr<DLLImportAttr>())
        return true;

  return false;
}

namespace {
  struct FunctionIsDirectlyRecursive :
    public RecursiveASTVisitor<FunctionIsDirectlyRecursive> {
    const StringRef Name;
    const Builtin::Context &BI;
    bool Result;
    FunctionIsDirectlyRecursive(StringRef N, const Builtin::Context &C) :
      Name(N), BI(C), Result(false) {
    }
    typedef RecursiveASTVisitor<FunctionIsDirectlyRecursive> Base;

    bool TraverseCallExpr(CallExpr *E) {
      const FunctionDecl *FD = E->getDirectCallee();
      if (!FD)
        return true;
      AsmLabelAttr *Attr = FD->getAttr<AsmLabelAttr>();
      if (Attr && Name == Attr->getLabel()) {
        Result = true;
        return false;
      }
      unsigned BuiltinID = FD->getBuiltinID();
      if (!BuiltinID || !BI.isLibFunction(BuiltinID))
        return true;
      StringRef BuiltinName = BI.getName(BuiltinID);
      if (BuiltinName.startswith("__builtin_") &&
          Name == BuiltinName.slice(strlen("__builtin_"), StringRef::npos)) {
        Result = true;
        return false;
      }
      return true;
    }
  };

  // Make sure we're not referencing non-imported vars or functions.
  struct DLLImportFunctionVisitor
      : public RecursiveASTVisitor<DLLImportFunctionVisitor> {
    bool SafeToInline = true;

    bool shouldVisitImplicitCode() const { return true; }

    bool VisitVarDecl(VarDecl *VD) {
      if (VD->getTLSKind()) {
        // A thread-local variable cannot be imported.
        SafeToInline = false;
        return SafeToInline;
      }

      // A variable definition might imply a destructor call.
      if (VD->isThisDeclarationADefinition())
        SafeToInline = !HasNonDllImportDtor(VD->getType());

      return SafeToInline;
    }

    bool VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
      if (const auto *D = E->getTemporary()->getDestructor())
        SafeToInline = D->hasAttr<DLLImportAttr>();
      return SafeToInline;
    }

    bool VisitDeclRefExpr(DeclRefExpr *E) {
      ValueDecl *VD = E->getDecl();
      if (isa<FunctionDecl>(VD))
        SafeToInline = VD->hasAttr<DLLImportAttr>();
      else if (VarDecl *V = dyn_cast<VarDecl>(VD))
        SafeToInline = !V->hasGlobalStorage() || V->hasAttr<DLLImportAttr>();
      return SafeToInline;
    }

    bool VisitCXXConstructExpr(CXXConstructExpr *E) {
      SafeToInline = E->getConstructor()->hasAttr<DLLImportAttr>();
      return SafeToInline;
    }

    bool VisitCXXMemberCallExpr(CXXMemberCallExpr *E) {
      CXXMethodDecl *M = E->getMethodDecl();
      if (!M) {
        // Call through a pointer to member function. This is safe to inline.
        SafeToInline = true;
      } else {
        SafeToInline = M->hasAttr<DLLImportAttr>();
      }
      return SafeToInline;
    }

    bool VisitCXXDeleteExpr(CXXDeleteExpr *E) {
      SafeToInline = E->getOperatorDelete()->hasAttr<DLLImportAttr>();
      return SafeToInline;
    }

    bool VisitCXXNewExpr(CXXNewExpr *E) {
      SafeToInline = E->getOperatorNew()->hasAttr<DLLImportAttr>();
      return SafeToInline;
    }
  };
}

// isTriviallyRecursive - Check if this function calls another
// decl that, because of the asm attribute or the other decl being a builtin,
// ends up pointing to itself.
bool
CodeGenModule::isTriviallyRecursive(const FunctionDecl *FD) {
  StringRef Name;
  if (getCXXABI().getMangleContext().shouldMangleDeclName(FD)) {
    // asm labels are a special kind of mangling we have to support.
    AsmLabelAttr *Attr = FD->getAttr<AsmLabelAttr>();
    if (!Attr)
      return false;
    Name = Attr->getLabel();
  } else {
    Name = FD->getName();
  }

  FunctionIsDirectlyRecursive Walker(Name, Context.BuiltinInfo);
  Walker.TraverseFunctionDecl(const_cast<FunctionDecl*>(FD));
  return Walker.Result;
}

bool CodeGenModule::shouldEmitFunction(GlobalDecl GD) {
  if (getFunctionLinkage(GD) != llvm::Function::AvailableExternallyLinkage)
    return true;
  const auto *F = cast<FunctionDecl>(GD.getDecl());
  if (CodeGenOpts.OptimizationLevel == 0 && !F->hasAttr<AlwaysInlineAttr>())
    return false;

  if (F->hasAttr<DLLImportAttr>()) {
    // Check whether it would be safe to inline this dllimport function.
    DLLImportFunctionVisitor Visitor;
    Visitor.TraverseFunctionDecl(const_cast<FunctionDecl*>(F));
    if (!Visitor.SafeToInline)
      return false;

    if (const CXXDestructorDecl *Dtor = dyn_cast<CXXDestructorDecl>(F)) {
      // Implicit destructor invocations aren't captured in the AST, so the
      // check above can't see them. Check for them manually here.
      for (const Decl *Member : Dtor->getParent()->decls())
        if (isa<FieldDecl>(Member))
          if (HasNonDllImportDtor(cast<FieldDecl>(Member)->getType()))
            return false;
      for (const CXXBaseSpecifier &B : Dtor->getParent()->bases())
        if (HasNonDllImportDtor(B.getType()))
          return false;
    }
  }

  // PR9614. Avoid cases where the source code is lying to us. An available
  // externally function should have an equivalent function somewhere else,
  // but a function that calls itself is clearly not equivalent to the real
  // implementation.
  // This happens in glibc's btowc and in some configure checks.
  return !isTriviallyRecursive(F);
}

bool CodeGenModule::shouldOpportunisticallyEmitVTables() {
  return CodeGenOpts.OptimizationLevel > 0;
}

void CodeGenModule::EmitGlobalDefinition(GlobalDecl GD, llvm::GlobalValue *GV) {
  const auto *D = cast<ValueDecl>(GD.getDecl());

  PrettyStackTraceDecl CrashInfo(const_cast<ValueDecl *>(D), D->getLocation(), 
                                 Context.getSourceManager(),
                                 "Generating code for declaration");
  
  if (isa<FunctionDecl>(D)) {
    // At -O0, don't generate IR for functions with available_externally 
    // linkage.
    if (!shouldEmitFunction(GD))
      return;

    if (const auto *Method = dyn_cast<CXXMethodDecl>(D)) {
      // Make sure to emit the definition(s) before we emit the thunks.
      // This is necessary for the generation of certain thunks.
      if (const auto *CD = dyn_cast<CXXConstructorDecl>(Method))
        ABI->emitCXXStructor(CD, getFromCtorType(GD.getCtorType()));
      else if (const auto *DD = dyn_cast<CXXDestructorDecl>(Method))
        ABI->emitCXXStructor(DD, getFromDtorType(GD.getDtorType()));
      else
        EmitGlobalFunctionDefinition(GD, GV);

      if (Method->isVirtual())
        getVTables().EmitThunks(GD);

      return;
    }

    return EmitGlobalFunctionDefinition(GD, GV);
  }

  if (const auto *VD = dyn_cast<VarDecl>(D))
    return EmitGlobalVarDefinition(VD, !VD->hasDefinition());
  
  llvm_unreachable("Invalid argument to EmitGlobalDefinition()");
}

static void ReplaceUsesOfNonProtoTypeWithRealFunction(llvm::GlobalValue *Old,
                                                      llvm::Function *NewFn);

/// GetOrCreateLLVMFunction - If the specified mangled name is not in the
/// module, create and return an llvm Function with the specified type. If there
/// is something in the module with the specified name, return it potentially
/// bitcasted to the right type.
///
/// If D is non-null, it specifies a decl that correspond to this.  This is used
/// to set the attributes on the function when it is first created.
llvm::Constant *CodeGenModule::GetOrCreateLLVMFunction(
    StringRef MangledName, llvm::Type *Ty, GlobalDecl GD, bool ForVTable,
    bool DontDefer, bool IsThunk, llvm::AttributeList ExtraAttrs,
    ForDefinition_t IsForDefinition) {
  const Decl *D = GD.getDecl();

  // Lookup the entry, lazily creating it if necessary.
  llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
  if (Entry) {
    if (WeakRefReferences.erase(Entry)) {
      const FunctionDecl *FD = cast_or_null<FunctionDecl>(D);
      if (FD && !FD->hasAttr<WeakAttr>())
        Entry->setLinkage(llvm::Function::ExternalLinkage);
    }

    // Handle dropped DLL attributes.
    if (D && !D->hasAttr<DLLImportAttr>() && !D->hasAttr<DLLExportAttr>())
      Entry->setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);

    // If there are two attempts to define the same mangled name, issue an
    // error.
    if (IsForDefinition && !Entry->isDeclaration()) {
      GlobalDecl OtherGD;
      // Check that GD is not yet in DiagnosedConflictingDefinitions is required
      // to make sure that we issue an error only once.
      if (lookupRepresentativeDecl(MangledName, OtherGD) &&
          (GD.getCanonicalDecl().getDecl() !=
           OtherGD.getCanonicalDecl().getDecl()) &&
          DiagnosedConflictingDefinitions.insert(GD).second) {
        getDiags().Report(D->getLocation(),
                          diag::err_duplicate_mangled_name);
        getDiags().Report(OtherGD.getDecl()->getLocation(),
                          diag::note_previous_definition);
      }
    }

    if ((isa<llvm::Function>(Entry) || isa<llvm::GlobalAlias>(Entry)) &&
        (Entry->getType()->getElementType() == Ty)) {
      return Entry;
    }

    // Make sure the result is of the correct type.
    // (If function is requested for a definition, we always need to create a new
    // function, not just return a bitcast.)
    if (!IsForDefinition)
      return llvm::ConstantExpr::getBitCast(Entry, Ty->getPointerTo());
  }

  // This function doesn't have a complete type (for example, the return
  // type is an incomplete struct). Use a fake type instead, and make
  // sure not to try to set attributes.
  bool IsIncompleteFunction = false;

  llvm::FunctionType *FTy;
  if (isa<llvm::FunctionType>(Ty)) {
    FTy = cast<llvm::FunctionType>(Ty);
  } else {
    FTy = llvm::FunctionType::get(VoidTy, false);
    IsIncompleteFunction = true;
  }

  llvm::Function *F =
      llvm::Function::Create(FTy, llvm::Function::ExternalLinkage,
                             Entry ? StringRef() : MangledName, &getModule());

  // If we already created a function with the same mangled name (but different
  // type) before, take its name and add it to the list of functions to be
  // replaced with F at the end of CodeGen.
  //
  // This happens if there is a prototype for a function (e.g. "int f()") and
  // then a definition of a different type (e.g. "int f(int x)").
  if (Entry) {
    F->takeName(Entry);

    // This might be an implementation of a function without a prototype, in
    // which case, try to do special replacement of calls which match the new
    // prototype.  The really key thing here is that we also potentially drop
    // arguments from the call site so as to make a direct call, which makes the
    // inliner happier and suppresses a number of optimizer warnings (!) about
    // dropping arguments.
    if (!Entry->use_empty()) {
      ReplaceUsesOfNonProtoTypeWithRealFunction(Entry, F);
      Entry->removeDeadConstantUsers();
    }

    llvm::Constant *BC = llvm::ConstantExpr::getBitCast(
        F, Entry->getType()->getElementType()->getPointerTo());
    addGlobalValReplacement(Entry, BC);
  }

  assert(F->getName() == MangledName && "name was uniqued!");
  if (D)
    SetFunctionAttributes(GD, F, IsIncompleteFunction, IsThunk);
  if (ExtraAttrs.hasAttributes(llvm::AttributeList::FunctionIndex)) {
    llvm::AttrBuilder B(ExtraAttrs, llvm::AttributeList::FunctionIndex);
    F->addAttributes(llvm::AttributeList::FunctionIndex, B);
  }

  if (!DontDefer) {
    // All MSVC dtors other than the base dtor are linkonce_odr and delegate to
    // each other bottoming out with the base dtor.  Therefore we emit non-base
    // dtors on usage, even if there is no dtor definition in the TU.
    if (D && isa<CXXDestructorDecl>(D) &&
        getCXXABI().useThunkForDtorVariant(cast<CXXDestructorDecl>(D),
                                           GD.getDtorType()))
      addDeferredDeclToEmit(GD);

    // This is the first use or definition of a mangled name.  If there is a
    // deferred decl with this name, remember that we need to emit it at the end
    // of the file.
    auto DDI = DeferredDecls.find(MangledName);
    if (DDI != DeferredDecls.end()) {
      // Move the potentially referenced deferred decl to the
      // DeferredDeclsToEmit list, and remove it from DeferredDecls (since we
      // don't need it anymore).
      addDeferredDeclToEmit(DDI->second);
      DeferredDecls.erase(DDI);

      // Otherwise, there are cases we have to worry about where we're
      // using a declaration for which we must emit a definition but where
      // we might not find a top-level definition:
      //   - member functions defined inline in their classes
      //   - friend functions defined inline in some class
      //   - special member functions with implicit definitions
      // If we ever change our AST traversal to walk into class methods,
      // this will be unnecessary.
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

  // Make sure the result is of the requested type.
  if (!IsIncompleteFunction) {
    assert(F->getType()->getElementType() == Ty);
    return F;
  }

  llvm::Type *PTy = llvm::PointerType::getUnqual(Ty);
  return llvm::ConstantExpr::getBitCast(F, PTy);
}

/// GetAddrOfFunction - Return the address of the given function.  If Ty is
/// non-null, then this function will use the specified type if it has to
/// create it (this occurs when we see a definition of the function).
llvm::Constant *CodeGenModule::GetAddrOfFunction(GlobalDecl GD,
                                                 llvm::Type *Ty,
                                                 bool ForVTable,
                                                 bool DontDefer,
                                              ForDefinition_t IsForDefinition) {
  // If there was no specific requested type, just convert it now.
  if (!Ty) {
    const auto *FD = cast<FunctionDecl>(GD.getDecl());
    auto CanonTy = Context.getCanonicalType(FD->getType());
    Ty = getTypes().ConvertFunctionType(CanonTy, FD);
  }

  StringRef MangledName = getMangledName(GD);
  return GetOrCreateLLVMFunction(MangledName, Ty, GD, ForVTable, DontDefer,
                                 /*IsThunk=*/false, llvm::AttributeList(),
                                 IsForDefinition);
}

static const FunctionDecl *
GetRuntimeFunctionDecl(ASTContext &C, StringRef Name) {
  TranslationUnitDecl *TUDecl = C.getTranslationUnitDecl();
  DeclContext *DC = TranslationUnitDecl::castToDeclContext(TUDecl);

  IdentifierInfo &CII = C.Idents.get(Name);
  for (const auto &Result : DC->lookup(&CII))
    if (const auto FD = dyn_cast<FunctionDecl>(Result))
      return FD;

  if (!C.getLangOpts().CPlusPlus)
    return nullptr;

  // Demangle the premangled name from getTerminateFn()
  IdentifierInfo &CXXII =
      (Name == "_ZSt9terminatev" || Name == "\01?terminate@@YAXXZ")
          ? C.Idents.get("terminate")
          : C.Idents.get(Name);

  for (const auto &N : {"__cxxabiv1", "std"}) {
    IdentifierInfo &NS = C.Idents.get(N);
    for (const auto &Result : DC->lookup(&NS)) {
      NamespaceDecl *ND = dyn_cast<NamespaceDecl>(Result);
      if (auto LSD = dyn_cast<LinkageSpecDecl>(Result))
        for (const auto &Result : LSD->lookup(&NS))
          if ((ND = dyn_cast<NamespaceDecl>(Result)))
            break;

      if (ND)
        for (const auto &Result : ND->lookup(&CXXII))
          if (const auto *FD = dyn_cast<FunctionDecl>(Result))
            return FD;
    }
  }

  return nullptr;
}

/// CreateRuntimeFunction - Create a new runtime function with the specified
/// type and name.
llvm::Constant *
CodeGenModule::CreateRuntimeFunction(llvm::FunctionType *FTy, StringRef Name,
                                     llvm::AttributeList ExtraAttrs,
                                     bool Local) {
  llvm::Constant *C =
      GetOrCreateLLVMFunction(Name, FTy, GlobalDecl(), /*ForVTable=*/false,
                              /*DontDefer=*/false, /*IsThunk=*/false,
                              ExtraAttrs);

  if (auto *F = dyn_cast<llvm::Function>(C)) {
    if (F->empty()) {
      F->setCallingConv(getRuntimeCC());

      if (!Local && getTriple().isOSBinFormatCOFF() &&
          !getCodeGenOpts().LTOVisibilityPublicStd) {
        const FunctionDecl *FD = GetRuntimeFunctionDecl(Context, Name);
        if (!FD || FD->hasAttr<DLLImportAttr>()) {
          F->setDLLStorageClass(llvm::GlobalValue::DLLImportStorageClass);
          F->setLinkage(llvm::GlobalValue::ExternalLinkage);
        }
      }
    }
  }

  return C;
}

/// CreateBuiltinFunction - Create a new builtin function with the specified
/// type and name.
llvm::Constant *
CodeGenModule::CreateBuiltinFunction(llvm::FunctionType *FTy, StringRef Name,
                                     llvm::AttributeList ExtraAttrs) {
  llvm::Constant *C =
      GetOrCreateLLVMFunction(Name, FTy, GlobalDecl(), /*ForVTable=*/false,
                              /*DontDefer=*/false, /*IsThunk=*/false, ExtraAttrs);
  if (auto *F = dyn_cast<llvm::Function>(C))
    if (F->empty())
      F->setCallingConv(getBuiltinCC());
  return C;
}

/// isTypeConstant - Determine whether an object of this type can be emitted
/// as a constant.
///
/// If ExcludeCtor is true, the duration when the object's constructor runs
/// will not be considered. The caller will need to verify that the object is
/// not written to during its construction.
bool CodeGenModule::isTypeConstant(QualType Ty, bool ExcludeCtor) {
  if (!Ty.isConstant(Context) && !Ty->isReferenceType())
    return false;

  if (Context.getLangOpts().CPlusPlus) {
    if (const CXXRecordDecl *Record
          = Context.getBaseElementType(Ty)->getAsCXXRecordDecl())
      return ExcludeCtor && !Record->hasMutableFields() &&
             Record->hasTrivialDestructor();
  }

  return true;
}

/// GetOrCreateLLVMGlobal - If the specified mangled name is not in the module,
/// create and return an llvm GlobalVariable with the specified type.  If there
/// is something in the module with the specified name, return it potentially
/// bitcasted to the right type.
///
/// If D is non-null, it specifies a decl that correspond to this.  This is used
/// to set the attributes on the global when it is first created.
///
/// If IsForDefinition is true, it is guranteed that an actual global with
/// type Ty will be returned, not conversion of a variable with the same
/// mangled name but some other type.
llvm::Constant *
CodeGenModule::GetOrCreateLLVMGlobal(StringRef MangledName,
                                     llvm::PointerType *Ty,
                                     const VarDecl *D,
                                     ForDefinition_t IsForDefinition) {
  // Lookup the entry, lazily creating it if necessary.
  llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
  if (Entry) {
    if (WeakRefReferences.erase(Entry)) {
      if (D && !D->hasAttr<WeakAttr>())
        Entry->setLinkage(llvm::Function::ExternalLinkage);
    }

    // Handle dropped DLL attributes.
    if (D && !D->hasAttr<DLLImportAttr>() && !D->hasAttr<DLLExportAttr>())
      Entry->setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);

    if (Entry->getType() == Ty)
      return Entry;

    // If there are two attempts to define the same mangled name, issue an
    // error.
    if (IsForDefinition && !Entry->isDeclaration()) {
      GlobalDecl OtherGD;
      const VarDecl *OtherD;

      // Check that D is not yet in DiagnosedConflictingDefinitions is required
      // to make sure that we issue an error only once.
      if (D && lookupRepresentativeDecl(MangledName, OtherGD) &&
          (D->getCanonicalDecl() != OtherGD.getCanonicalDecl().getDecl()) &&
          (OtherD = dyn_cast<VarDecl>(OtherGD.getDecl())) &&
          OtherD->hasInit() &&
          DiagnosedConflictingDefinitions.insert(D).second) {
        getDiags().Report(D->getLocation(),
                          diag::err_duplicate_mangled_name);
        getDiags().Report(OtherGD.getDecl()->getLocation(),
                          diag::note_previous_definition);
      }
    }

    // Make sure the result is of the correct type.
    if (Entry->getType()->getAddressSpace() != Ty->getAddressSpace())
      return llvm::ConstantExpr::getAddrSpaceCast(Entry, Ty);

    // (If global is requested for a definition, we always need to create a new
    // global, not just return a bitcast.)
    if (!IsForDefinition)
      return llvm::ConstantExpr::getBitCast(Entry, Ty);
  }

  unsigned AddrSpace = GetGlobalVarAddressSpace(D, Ty->getAddressSpace());
  auto *GV = new llvm::GlobalVariable(
      getModule(), Ty->getElementType(), false,
      llvm::GlobalValue::ExternalLinkage, nullptr, MangledName, nullptr,
      llvm::GlobalVariable::NotThreadLocal, AddrSpace);

  // If we already created a global with the same mangled name (but different
  // type) before, take its name and remove it from its parent.
  if (Entry) {
    GV->takeName(Entry);

    if (!Entry->use_empty()) {
      llvm::Constant *NewPtrForOldDecl =
          llvm::ConstantExpr::getBitCast(GV, Entry->getType());
      Entry->replaceAllUsesWith(NewPtrForOldDecl);
    }

    Entry->eraseFromParent();
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
    // FIXME: This code is overly simple and should be merged with other global
    // handling.
    GV->setConstant(isTypeConstant(D->getType(), false));

    GV->setAlignment(getContext().getDeclAlign(D).getQuantity());

    setLinkageAndVisibilityForGV(GV, D);

    if (D->getTLSKind()) {
      if (D->getTLSKind() == VarDecl::TLS_Dynamic)
        CXXThreadLocals.push_back(D);
      setTLSMode(GV, *D);
    }

    // If required by the ABI, treat declarations of static data members with
    // inline initializers as definitions.
    if (getContext().isMSStaticDataMemberInlineDefinition(D)) {
      EmitGlobalVarDefinition(D);
    }

    // Handle XCore specific ABI requirements.
    if (getTriple().getArch() == llvm::Triple::xcore &&
        D->getLanguageLinkage() == CLanguageLinkage &&
        D->getType().isConstant(Context) &&
        isExternallyVisible(D->getLinkageAndVisibility().getLinkage()))
      GV->setSection(".cp.rodata");
  }

  if (AddrSpace != Ty->getAddressSpace())
    return llvm::ConstantExpr::getAddrSpaceCast(GV, Ty);

  return GV;
}

llvm::Constant *
CodeGenModule::GetAddrOfGlobal(GlobalDecl GD,
                               ForDefinition_t IsForDefinition) {
  const Decl *D = GD.getDecl();
  if (isa<CXXConstructorDecl>(D))
    return getAddrOfCXXStructor(cast<CXXConstructorDecl>(D),
                                getFromCtorType(GD.getCtorType()),
                                /*FnInfo=*/nullptr, /*FnType=*/nullptr,
                                /*DontDefer=*/false, IsForDefinition);
  else if (isa<CXXDestructorDecl>(D))
    return getAddrOfCXXStructor(cast<CXXDestructorDecl>(D),
                                getFromDtorType(GD.getDtorType()),
                                /*FnInfo=*/nullptr, /*FnType=*/nullptr,
                                /*DontDefer=*/false, IsForDefinition);
  else if (isa<CXXMethodDecl>(D)) {
    auto FInfo = &getTypes().arrangeCXXMethodDeclaration(
        cast<CXXMethodDecl>(D));
    auto Ty = getTypes().GetFunctionType(*FInfo);
    return GetAddrOfFunction(GD, Ty, /*ForVTable=*/false, /*DontDefer=*/false,
                             IsForDefinition);
  } else if (isa<FunctionDecl>(D)) {
    const CGFunctionInfo &FI = getTypes().arrangeGlobalDeclaration(GD);
    llvm::FunctionType *Ty = getTypes().GetFunctionType(FI);
    return GetAddrOfFunction(GD, Ty, /*ForVTable=*/false, /*DontDefer=*/false,
                             IsForDefinition);
  } else
    return GetAddrOfGlobalVar(cast<VarDecl>(D), /*Ty=*/nullptr,
                              IsForDefinition);
}

llvm::GlobalVariable *
CodeGenModule::CreateOrReplaceCXXRuntimeVariable(StringRef Name, 
                                      llvm::Type *Ty,
                                      llvm::GlobalValue::LinkageTypes Linkage) {
  llvm::GlobalVariable *GV = getModule().getNamedGlobal(Name);
  llvm::GlobalVariable *OldGV = nullptr;

  if (GV) {
    // Check if the variable has the right type.
    if (GV->getType()->getElementType() == Ty)
      return GV;

    // Because C++ name mangling, the only way we can end up with an already
    // existing global with the same name is if it has been declared extern "C".
    assert(GV->isDeclaration() && "Declaration has wrong type!");
    OldGV = GV;
  }
  
  // Create a new variable.
  GV = new llvm::GlobalVariable(getModule(), Ty, /*isConstant=*/true,
                                Linkage, nullptr, Name);

  if (OldGV) {
    // Replace occurrences of the old variable if needed.
    GV->takeName(OldGV);
    
    if (!OldGV->use_empty()) {
      llvm::Constant *NewPtrForOldDecl =
      llvm::ConstantExpr::getBitCast(GV, OldGV->getType());
      OldGV->replaceAllUsesWith(NewPtrForOldDecl);
    }
    
    OldGV->eraseFromParent();
  }

  if (supportsCOMDAT() && GV->isWeakForLinker() &&
      !GV->hasAvailableExternallyLinkage())
    GV->setComdat(TheModule.getOrInsertComdat(GV->getName()));

  return GV;
}

/// GetAddrOfGlobalVar - Return the llvm::Constant for the address of the
/// given global variable.  If Ty is non-null and if the global doesn't exist,
/// then it will be created with the specified type instead of whatever the
/// normal requested type would be. If IsForDefinition is true, it is guranteed
/// that an actual global with type Ty will be returned, not conversion of a
/// variable with the same mangled name but some other type.
llvm::Constant *CodeGenModule::GetAddrOfGlobalVar(const VarDecl *D,
                                                  llvm::Type *Ty,
                                           ForDefinition_t IsForDefinition) {
  assert(D->hasGlobalStorage() && "Not a global variable");
  QualType ASTTy = D->getType();
  if (!Ty)
    Ty = getTypes().ConvertTypeForMem(ASTTy);

  llvm::PointerType *PTy =
    llvm::PointerType::get(Ty, getContext().getTargetAddressSpace(ASTTy));

  StringRef MangledName = getMangledName(D);
  return GetOrCreateLLVMGlobal(MangledName, PTy, D, IsForDefinition);
}

/// CreateRuntimeVariable - Create a new runtime global variable with the
/// specified type and name.
llvm::Constant *
CodeGenModule::CreateRuntimeVariable(llvm::Type *Ty,
                                     StringRef Name) {
  return GetOrCreateLLVMGlobal(Name, llvm::PointerType::getUnqual(Ty), nullptr);
}

void CodeGenModule::EmitTentativeDefinition(const VarDecl *D) {
  assert(!D->getInit() && "Cannot emit definite definitions here!");

  StringRef MangledName = getMangledName(D);
  llvm::GlobalValue *GV = GetGlobalValue(MangledName);

  // We already have a definition, not declaration, with the same mangled name.
  // Emitting of declaration is not required (and actually overwrites emitted
  // definition).
  if (GV && !GV->isDeclaration())
    return;

  // If we have not seen a reference to this variable yet, place it into the
  // deferred declarations table to be emitted if needed later.
  if (!MustBeEmitted(D) && !GV) {
      DeferredDecls[MangledName] = D;
      return;
  }

  // The tentative definition is the only definition.
  EmitGlobalVarDefinition(D);
}

CharUnits CodeGenModule::GetTargetTypeStoreSize(llvm::Type *Ty) const {
  return Context.toCharUnitsFromBits(
      getDataLayout().getTypeStoreSizeInBits(Ty));
}

unsigned CodeGenModule::GetGlobalVarAddressSpace(const VarDecl *D,
                                                 unsigned AddrSpace) {
  if (D && LangOpts.CUDA && LangOpts.CUDAIsDevice) {
    if (D->hasAttr<CUDAConstantAttr>())
      AddrSpace = getContext().getTargetAddressSpace(LangAS::cuda_constant);
    else if (D->hasAttr<CUDASharedAttr>())
      AddrSpace = getContext().getTargetAddressSpace(LangAS::cuda_shared);
    else
      AddrSpace = getContext().getTargetAddressSpace(LangAS::cuda_device);
  }

  return AddrSpace;
}

template<typename SomeDecl>
void CodeGenModule::MaybeHandleStaticInExternC(const SomeDecl *D,
                                               llvm::GlobalValue *GV) {
  if (!getLangOpts().CPlusPlus)
    return;

  // Must have 'used' attribute, or else inline assembly can't rely on
  // the name existing.
  if (!D->template hasAttr<UsedAttr>())
    return;

  // Must have internal linkage and an ordinary name.
  if (!D->getIdentifier() || D->getFormalLinkage() != InternalLinkage)
    return;

  // Must be in an extern "C" context. Entities declared directly within
  // a record are not extern "C" even if the record is in such a context.
  const SomeDecl *First = D->getFirstDecl();
  if (First->getDeclContext()->isRecord() || !First->isInExternCContext())
    return;

  // OK, this is an internal linkage entity inside an extern "C" linkage
  // specification. Make a note of that so we can give it the "expected"
  // mangled name if nothing else is using that name.
  std::pair<StaticExternCMap::iterator, bool> R =
      StaticExternCValues.insert(std::make_pair(D->getIdentifier(), GV));

  // If we have multiple internal linkage entities with the same name
  // in extern "C" regions, none of them gets that name.
  if (!R.second)
    R.first->second = nullptr;
}

static bool shouldBeInCOMDAT(CodeGenModule &CGM, const Decl &D) {
  if (!CGM.supportsCOMDAT())
    return false;

  if (D.hasAttr<SelectAnyAttr>())
    return true;

  GVALinkage Linkage;
  if (auto *VD = dyn_cast<VarDecl>(&D))
    Linkage = CGM.getContext().GetGVALinkageForVariable(VD);
  else
    Linkage = CGM.getContext().GetGVALinkageForFunction(cast<FunctionDecl>(&D));

  switch (Linkage) {
  case GVA_Internal:
  case GVA_AvailableExternally:
  case GVA_StrongExternal:
    return false;
  case GVA_DiscardableODR:
  case GVA_StrongODR:
    return true;
  }
  llvm_unreachable("No such linkage");
}

void CodeGenModule::maybeSetTrivialComdat(const Decl &D,
                                          llvm::GlobalObject &GO) {
  if (!shouldBeInCOMDAT(*this, D))
    return;
  GO.setComdat(TheModule.getOrInsertComdat(GO.getName()));
}

/// Pass IsTentative as true if you want to create a tentative definition.
void CodeGenModule::EmitGlobalVarDefinition(const VarDecl *D,
                                            bool IsTentative) {
  // OpenCL global variables of sampler type are translated to function calls,
  // therefore no need to be translated.
  QualType ASTTy = D->getType();
  if (getLangOpts().OpenCL && ASTTy->isSamplerT())
    return;

  llvm::Constant *Init = nullptr;
  CXXRecordDecl *RD = ASTTy->getBaseElementTypeUnsafe()->getAsCXXRecordDecl();
  bool NeedsGlobalCtor = false;
  bool NeedsGlobalDtor = RD && !RD->hasTrivialDestructor();

  const VarDecl *InitDecl;
  const Expr *InitExpr = D->getAnyInitializer(InitDecl);

  // CUDA E.2.4.1 "__shared__ variables cannot have an initialization
  // as part of their declaration."  Sema has already checked for
  // error cases, so we just need to set Init to UndefValue.
  if (getLangOpts().CUDA && getLangOpts().CUDAIsDevice &&
      D->hasAttr<CUDASharedAttr>())
    Init = llvm::UndefValue::get(getTypes().ConvertType(ASTTy));
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
    Init = EmitNullConstant(D->getType());
  } else {
    initializedGlobalDecl = GlobalDecl(D);
    Init = EmitConstantInit(*InitDecl);

    if (!Init) {
      QualType T = InitExpr->getType();
      if (D->getType()->isReferenceType())
        T = D->getType();

      if (getLangOpts().CPlusPlus) {
        Init = EmitNullConstant(T);
        NeedsGlobalCtor = true;
      } else {
        ErrorUnsupported(D, "static initializer");
        Init = llvm::UndefValue::get(getTypes().ConvertType(T));
      }
    } else {
      // We don't need an initializer, so remove the entry for the delayed
      // initializer position (just in case this entry was delayed) if we
      // also don't need to register a destructor.
      if (getLangOpts().CPlusPlus && !NeedsGlobalDtor)
        DelayedCXXInitPosition.erase(D);
    }
  }

  llvm::Type* InitType = Init->getType();
  llvm::Constant *Entry =
      GetAddrOfGlobalVar(D, InitType, ForDefinition_t(!IsTentative));

  // Strip off a bitcast if we got one back.
  if (auto *CE = dyn_cast<llvm::ConstantExpr>(Entry)) {
    assert(CE->getOpcode() == llvm::Instruction::BitCast ||
           CE->getOpcode() == llvm::Instruction::AddrSpaceCast ||
           // All zero index gep.
           CE->getOpcode() == llvm::Instruction::GetElementPtr);
    Entry = CE->getOperand(0);
  }

  // Entry is now either a Function or GlobalVariable.
  auto *GV = dyn_cast<llvm::GlobalVariable>(Entry);

  // We have a definition after a declaration with the wrong type.
  // We must make a new GlobalVariable* and update everything that used OldGV
  // (a declaration or tentative definition) with the new GlobalVariable*
  // (which will be a definition).
  //
  // This happens if there is a prototype for a global (e.g.
  // "extern int x[];") and then a definition of a different type (e.g.
  // "int x[10];"). This also happens when an initializer has a different type
  // from the type of the global (this happens with unions).
  if (!GV ||
      GV->getType()->getElementType() != InitType ||
      GV->getType()->getAddressSpace() !=
       GetGlobalVarAddressSpace(D, getContext().getTargetAddressSpace(ASTTy))) {

    // Move the old entry aside so that we'll create a new one.
    Entry->setName(StringRef());

    // Make a new global with the correct type, this is now guaranteed to work.
    GV = cast<llvm::GlobalVariable>(
        GetAddrOfGlobalVar(D, InitType, ForDefinition_t(!IsTentative)));

    // Replace all uses of the old global with the new global
    llvm::Constant *NewPtrForOldDecl =
        llvm::ConstantExpr::getBitCast(GV, Entry->getType());
    Entry->replaceAllUsesWith(NewPtrForOldDecl);

    // Erase the old global, since it is no longer used.
    cast<llvm::GlobalValue>(Entry)->eraseFromParent();
  }

  MaybeHandleStaticInExternC(D, GV);

  if (D->hasAttr<AnnotateAttr>())
    AddGlobalAnnotations(D, GV);

  // Set the llvm linkage type as appropriate.
  llvm::GlobalValue::LinkageTypes Linkage =
      getLLVMLinkageVarDefinition(D, GV->isConstant());

  // CUDA B.2.1 "The __device__ qualifier declares a variable that resides on
  // the device. [...]"
  // CUDA B.2.2 "The __constant__ qualifier, optionally used together with
  // __device__, declares a variable that: [...]
  // Is accessible from all the threads within the grid and from the host
  // through the runtime library (cudaGetSymbolAddress() / cudaGetSymbolSize()
  // / cudaMemcpyToSymbol() / cudaMemcpyFromSymbol())."
  if (GV && LangOpts.CUDA) {
    if (LangOpts.CUDAIsDevice) {
      if (D->hasAttr<CUDADeviceAttr>() || D->hasAttr<CUDAConstantAttr>())
        GV->setExternallyInitialized(true);
    } else {
      // Host-side shadows of external declarations of device-side
      // global variables become internal definitions. These have to
      // be internal in order to prevent name conflicts with global
      // host variables with the same name in a different TUs.
      if (D->hasAttr<CUDADeviceAttr>() || D->hasAttr<CUDAConstantAttr>()) {
        Linkage = llvm::GlobalValue::InternalLinkage;

        // Shadow variables and their properties must be registered
        // with CUDA runtime.
        unsigned Flags = 0;
        if (!D->hasDefinition())
          Flags |= CGCUDARuntime::ExternDeviceVar;
        if (D->hasAttr<CUDAConstantAttr>())
          Flags |= CGCUDARuntime::ConstantDeviceVar;
        getCUDARuntime().registerDeviceVar(*GV, Flags);
      } else if (D->hasAttr<CUDASharedAttr>())
        // __shared__ variables are odd. Shadows do get created, but
        // they are not registered with the CUDA runtime, so they
        // can't really be used to access their device-side
        // counterparts. It's not clear yet whether it's nvcc's bug or
        // a feature, but we've got to do the same for compatibility.
        Linkage = llvm::GlobalValue::InternalLinkage;
    }
  }
  GV->setInitializer(Init);

  // If it is safe to mark the global 'constant', do so now.
  GV->setConstant(!NeedsGlobalCtor && !NeedsGlobalDtor &&
                  isTypeConstant(D->getType(), true));

  // If it is in a read-only section, mark it 'constant'.
  if (const SectionAttr *SA = D->getAttr<SectionAttr>()) {
    const ASTContext::SectionInfo &SI = Context.SectionInfos[SA->getName()];
    if ((SI.SectionFlags & ASTContext::PSF_Write) == 0)
      GV->setConstant(true);
  }

  GV->setAlignment(getContext().getDeclAlign(D).getQuantity());


  // On Darwin, if the normal linkage of a C++ thread_local variable is
  // LinkOnce or Weak, we keep the normal linkage to prevent multiple
  // copies within a linkage unit; otherwise, the backing variable has
  // internal linkage and all accesses should just be calls to the
  // Itanium-specified entry point, which has the normal linkage of the
  // variable. This is to preserve the ability to change the implementation
  // behind the scenes.
  if (!D->isStaticLocal() && D->getTLSKind() == VarDecl::TLS_Dynamic &&
      Context.getTargetInfo().getTriple().isOSDarwin() &&
      !llvm::GlobalVariable::isLinkOnceLinkage(Linkage) &&
      !llvm::GlobalVariable::isWeakLinkage(Linkage))
    Linkage = llvm::GlobalValue::InternalLinkage;

  GV->setLinkage(Linkage);
  if (D->hasAttr<DLLImportAttr>())
    GV->setDLLStorageClass(llvm::GlobalVariable::DLLImportStorageClass);
  else if (D->hasAttr<DLLExportAttr>())
    GV->setDLLStorageClass(llvm::GlobalVariable::DLLExportStorageClass);
  else
    GV->setDLLStorageClass(llvm::GlobalVariable::DefaultStorageClass);

  if (Linkage == llvm::GlobalVariable::CommonLinkage) {
    // common vars aren't constant even if declared const.
    GV->setConstant(false);
    // Tentative definition of global variables may be initialized with
    // non-zero null pointers. In this case they should have weak linkage
    // since common linkage must have zero initializer and must not have
    // explicit section therefore cannot have non-zero initial value.
    if (!GV->getInitializer()->isNullValue())
      GV->setLinkage(llvm::GlobalVariable::WeakAnyLinkage);
  }

  setNonAliasAttributes(D, GV);

  if (D->getTLSKind() && !GV->isThreadLocal()) {
    if (D->getTLSKind() == VarDecl::TLS_Dynamic)
      CXXThreadLocals.push_back(D);
    setTLSMode(GV, *D);
  }

  maybeSetTrivialComdat(*D, *GV);

  // Emit the initializer function if necessary.
  if (NeedsGlobalCtor || NeedsGlobalDtor)
    EmitCXXGlobalVarDeclInitFunc(D, GV, NeedsGlobalCtor);

  SanitizerMD->reportGlobalToASan(GV, *D, NeedsGlobalCtor);

  // Emit global variable debug information.
  if (CGDebugInfo *DI = getModuleDebugInfo())
    if (getCodeGenOpts().getDebugInfo() >= codegenoptions::LimitedDebugInfo)
      DI->EmitGlobalVariable(GV, D);
}

static bool isVarDeclStrongDefinition(const ASTContext &Context,
                                      CodeGenModule &CGM, const VarDecl *D,
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
  // We dont try to determine which is the right section in the front-end.
  // If no specialized section name is applicable, it will resort to default.
  if (D->hasAttr<PragmaClangBSSSectionAttr>() ||
      D->hasAttr<PragmaClangDataSectionAttr>() ||
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

  return false;
}

llvm::GlobalValue::LinkageTypes CodeGenModule::getLLVMLinkageForDeclarator(
    const DeclaratorDecl *D, GVALinkage Linkage, bool IsConstantVariable) {
  if (Linkage == GVA_Internal)
    return llvm::Function::InternalLinkage;

  if (D->hasAttr<WeakAttr>()) {
    if (IsConstantVariable)
      return llvm::GlobalVariable::WeakODRLinkage;
    else
      return llvm::GlobalVariable::WeakAnyLinkage;
  }

  // We are guaranteed to have a strong definition somewhere else,
  // so we can use available_externally linkage.
  if (Linkage == GVA_AvailableExternally)
    return llvm::GlobalValue::AvailableExternallyLinkage;

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
    return !Context.getLangOpts().AppleKext ? llvm::Function::LinkOnceODRLinkage
                                            : llvm::Function::InternalLinkage;

  // An explicit instantiation of a template has weak linkage, since
  // explicit instantiations can occur in multiple translation units
  // and must all be equivalent. However, we are not allowed to
  // throw away these explicit instantiations.
  //
  // We don't currently support CUDA device code spread out across multiple TUs,
  // so say that CUDA templates are either external (for kernels) or internal.
  // This lets llvm perform aggressive inter-procedural optimizations.
  if (Linkage == GVA_StrongODR) {
    if (Context.getLangOpts().AppleKext)
      return llvm::Function::ExternalLinkage;
    if (Context.getLangOpts().CUDA && Context.getLangOpts().CUDAIsDevice)
      return D->hasAttr<CUDAGlobalAttr>() ? llvm::Function::ExternalLinkage
                                          : llvm::Function::InternalLinkage;
    return llvm::Function::WeakODRLinkage;
  }

  // C++ doesn't have tentative definitions and thus cannot have common
  // linkage.
  if (!getLangOpts().CPlusPlus && isa<VarDecl>(D) &&
      !isVarDeclStrongDefinition(Context, *this, cast<VarDecl>(D),
                                 CodeGenOpts.NoCommon))
    return llvm::GlobalVariable::CommonLinkage;

  // selectany symbols are externally visible, so use weak instead of
  // linkonce.  MSVC optimizes away references to const selectany globals, so
  // all definitions should be the same and ODR linkage should be used.
  // http://msdn.microsoft.com/en-us/library/5tkz6s71.aspx
  if (D->hasAttr<SelectAnyAttr>())
    return llvm::GlobalVariable::WeakODRLinkage;

  // Otherwise, we have strong external linkage.
  assert(Linkage == GVA_StrongExternal);
  return llvm::GlobalVariable::ExternalLinkage;
}

llvm::GlobalValue::LinkageTypes CodeGenModule::getLLVMLinkageVarDefinition(
    const VarDecl *VD, bool IsConstant) {
  GVALinkage Linkage = getContext().GetGVALinkageForVariable(VD);
  return getLLVMLinkageForDeclarator(VD, Linkage, IsConstant);
}

/// Replace the uses of a function that was declared with a non-proto type.
/// We want to silently drop extra arguments from call sites
static void replaceUsesOfNonProtoConstant(llvm::Constant *old,
                                          llvm::Function *newFn) {
  // Fast path.
  if (old->use_empty()) return;

  llvm::Type *newRetTy = newFn->getReturnType();
  SmallVector<llvm::Value*, 4> newArgs;
  SmallVector<llvm::OperandBundleDef, 1> newBundles;

  for (llvm::Value::use_iterator ui = old->use_begin(), ue = old->use_end();
         ui != ue; ) {
    llvm::Value::use_iterator use = ui++; // Increment before the use is erased.
    llvm::User *user = use->getUser();

    // Recognize and replace uses of bitcasts.  Most calls to
    // unprototyped functions will use bitcasts.
    if (auto *bitcast = dyn_cast<llvm::ConstantExpr>(user)) {
      if (bitcast->getOpcode() == llvm::Instruction::BitCast)
        replaceUsesOfNonProtoConstant(bitcast, newFn);
      continue;
    }

    // Recognize calls to the function.
    llvm::CallSite callSite(user);
    if (!callSite) continue;
    if (!callSite.isCallee(&*use)) continue;

    // If the return types don't match exactly, then we can't
    // transform this call unless it's dead.
    if (callSite->getType() != newRetTy && !callSite->use_empty())
      continue;

    // Get the call site's attribute list.
    SmallVector<llvm::AttributeSet, 8> newArgAttrs;
    llvm::AttributeList oldAttrs = callSite.getAttributes();

    // If the function was passed too few arguments, don't transform.
    unsigned newNumArgs = newFn->arg_size();
    if (callSite.arg_size() < newNumArgs) continue;

    // If extra arguments were passed, we silently drop them.
    // If any of the types mismatch, we don't transform.
    unsigned argNo = 0;
    bool dontTransform = false;
    for (llvm::Argument &A : newFn->args()) {
      if (callSite.getArgument(argNo)->getType() != A.getType()) {
        dontTransform = true;
        break;
      }

      // Add any parameter attributes.
      newArgAttrs.push_back(oldAttrs.getParamAttributes(argNo));
      argNo++;
    }
    if (dontTransform)
      continue;

    // Okay, we can transform this.  Create the new call instruction and copy
    // over the required information.
    newArgs.append(callSite.arg_begin(), callSite.arg_begin() + argNo);

    // Copy over any operand bundles.
    callSite.getOperandBundlesAsDefs(newBundles);

    llvm::CallSite newCall;
    if (callSite.isCall()) {
      newCall = llvm::CallInst::Create(newFn, newArgs, newBundles, "",
                                       callSite.getInstruction());
    } else {
      auto *oldInvoke = cast<llvm::InvokeInst>(callSite.getInstruction());
      newCall = llvm::InvokeInst::Create(newFn,
                                         oldInvoke->getNormalDest(),
                                         oldInvoke->getUnwindDest(),
                                         newArgs, newBundles, "",
                                         callSite.getInstruction());
    }
    newArgs.clear(); // for the next iteration

    if (!newCall->getType()->isVoidTy())
      newCall->takeName(callSite.getInstruction());
    newCall.setAttributes(llvm::AttributeList::get(
        newFn->getContext(), oldAttrs.getFnAttributes(),
        oldAttrs.getRetAttributes(), newArgAttrs));
    newCall.setCallingConv(callSite.getCallingConv());

    // Finally, remove the old call, replacing any uses with the new one.
    if (!callSite->use_empty())
      callSite->replaceAllUsesWith(newCall.getInstruction());

    // Copy debug location attached to CI.
    if (callSite->getDebugLoc())
      newCall->setDebugLoc(callSite->getDebugLoc());

    callSite->eraseFromParent();
  }
}

/// ReplaceUsesOfNonProtoTypeWithRealFunction - This function is called when we
/// implement a function with no prototype, e.g. "int foo() {}".  If there are
/// existing call uses of the old function in the module, this adjusts them to
/// call the new function directly.
///
/// This is not just a cleanup: the always_inline pass requires direct calls to
/// functions to be able to inline them.  If there is a bitcast in the way, it
/// won't inline them.  Instcombine normally deletes these calls, but it isn't
/// run at -O0.
static void ReplaceUsesOfNonProtoTypeWithRealFunction(llvm::GlobalValue *Old,
                                                      llvm::Function *NewFn) {
  // If we're redefining a global as a function, don't transform it.
  if (!isa<llvm::Function>(Old)) return;

  replaceUsesOfNonProtoConstant(Old, NewFn);
}

void CodeGenModule::HandleCXXStaticMemberVarInstantiation(VarDecl *VD) {
  auto DK = VD->isThisDeclarationADefinition();
  if (DK == VarDecl::Definition && VD->hasAttr<DLLImportAttr>())
    return;

  TemplateSpecializationKind TSK = VD->getTemplateSpecializationKind();
  // If we have a definition, this might be a deferred decl. If the
  // instantiation is explicit, make sure we emit it at the end.
  if (VD->getDefinition() && TSK == TSK_ExplicitInstantiationDefinition)
    GetAddrOfGlobalVar(VD);

  EmitTopLevelDecl(VD);
}

void CodeGenModule::EmitGlobalFunctionDefinition(GlobalDecl GD,
                                                 llvm::GlobalValue *GV) {
  const auto *D = cast<FunctionDecl>(GD.getDecl());

  // Compute the function info and LLVM type.
  const CGFunctionInfo &FI = getTypes().arrangeGlobalDeclaration(GD);
  llvm::FunctionType *Ty = getTypes().GetFunctionType(FI);

  // Get or create the prototype for the function.
  if (!GV || (GV->getType()->getElementType() != Ty))
    GV = cast<llvm::GlobalValue>(GetAddrOfFunction(GD, Ty, /*ForVTable=*/false,
                                                   /*DontDefer=*/true,
                                                   ForDefinition));

  // Already emitted.
  if (!GV->isDeclaration())
    return;

  // We need to set linkage and visibility on the function before
  // generating code for it because various parts of IR generation
  // want to propagate this information down (e.g. to local static
  // declarations).
  auto *Fn = cast<llvm::Function>(GV);
  setFunctionLinkage(GD, Fn);
  setFunctionDLLStorageClass(GD, Fn);

  // FIXME: this is redundant with part of setFunctionDefinitionAttributes
  setGlobalVisibility(Fn, D);

  MaybeHandleStaticInExternC(D, Fn);

  maybeSetTrivialComdat(*D, *Fn);

  CodeGenFunction(*this).GenerateCode(D, Fn, FI);

  setFunctionDefinitionAttributes(D, Fn);
  SetLLVMFunctionAttributesForDefinition(D, Fn);

  if (const ConstructorAttr *CA = D->getAttr<ConstructorAttr>())
    AddGlobalCtor(Fn, CA->getPriority());
  if (const DestructorAttr *DA = D->getAttr<DestructorAttr>())
    AddGlobalDtor(Fn, DA->getPriority());
  if (D->hasAttr<AnnotateAttr>())
    AddGlobalAnnotations(D, Fn);
}

void CodeGenModule::EmitAliasDefinition(GlobalDecl GD) {
  const auto *D = cast<ValueDecl>(GD.getDecl());
  const AliasAttr *AA = D->getAttr<AliasAttr>();
  assert(AA && "Not an alias?");

  StringRef MangledName = getMangledName(GD);

  if (AA->getAliasee() == MangledName) {
    Diags.Report(AA->getLocation(), diag::err_cyclic_alias) << 0;
    return;
  }

  // If there is a definition in the module, then it wins over the alias.
  // This is dubious, but allow it to be safe.  Just ignore the alias.
  llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
  if (Entry && !Entry->isDeclaration())
    return;

  Aliases.push_back(GD);

  llvm::Type *DeclTy = getTypes().ConvertTypeForMem(D->getType());

  // Create a reference to the named value.  This ensures that it is emitted
  // if a deferred decl.
  llvm::Constant *Aliasee;
  if (isa<llvm::FunctionType>(DeclTy))
    Aliasee = GetOrCreateLLVMFunction(AA->getAliasee(), DeclTy, GD,
                                      /*ForVTable=*/false);
  else
    Aliasee = GetOrCreateLLVMGlobal(AA->getAliasee(),
                                    llvm::PointerType::getUnqual(DeclTy),
                                    /*D=*/nullptr);

  // Create the new alias itself, but don't set a name yet.
  auto *GA = llvm::GlobalAlias::create(
      DeclTy, 0, llvm::Function::ExternalLinkage, "", Aliasee, &getModule());

  if (Entry) {
    if (GA->getAliasee() == Entry) {
      Diags.Report(AA->getLocation(), diag::err_cyclic_alias) << 0;
      return;
    }

    assert(Entry->isDeclaration());

    // If there is a declaration in the module, then we had an extern followed
    // by the alias, as in:
    //   extern int test6();
    //   ...
    //   int test6() __attribute__((alias("test7")));
    //
    // Remove it and replace uses of it with the alias.
    GA->takeName(Entry);

    Entry->replaceAllUsesWith(llvm::ConstantExpr::getBitCast(GA,
                                                          Entry->getType()));
    Entry->eraseFromParent();
  } else {
    GA->setName(MangledName);
  }

  // Set attributes which are particular to an alias; this is a
  // specialization of the attributes which may be set on a global
  // variable/function.
  if (D->hasAttr<WeakAttr>() || D->hasAttr<WeakRefAttr>() ||
      D->isWeakImported()) {
    GA->setLinkage(llvm::Function::WeakAnyLinkage);
  }

  if (const auto *VD = dyn_cast<VarDecl>(D))
    if (VD->getTLSKind())
      setTLSMode(GA, *VD);

  setAliasAttributes(D, GA);
}

void CodeGenModule::emitIFuncDefinition(GlobalDecl GD) {
  const auto *D = cast<ValueDecl>(GD.getDecl());
  const IFuncAttr *IFA = D->getAttr<IFuncAttr>();
  assert(IFA && "Not an ifunc?");

  StringRef MangledName = getMangledName(GD);

  if (IFA->getResolver() == MangledName) {
    Diags.Report(IFA->getLocation(), diag::err_cyclic_alias) << 1;
    return;
  }

  // Report an error if some definition overrides ifunc.
  llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
  if (Entry && !Entry->isDeclaration()) {
    GlobalDecl OtherGD;
    if (lookupRepresentativeDecl(MangledName, OtherGD) &&
        DiagnosedConflictingDefinitions.insert(GD).second) {
      Diags.Report(D->getLocation(), diag::err_duplicate_mangled_name);
      Diags.Report(OtherGD.getDecl()->getLocation(),
                   diag::note_previous_definition);
    }
    return;
  }

  Aliases.push_back(GD);

  llvm::Type *DeclTy = getTypes().ConvertTypeForMem(D->getType());
  llvm::Constant *Resolver =
      GetOrCreateLLVMFunction(IFA->getResolver(), DeclTy, GD,
                              /*ForVTable=*/false);
  llvm::GlobalIFunc *GIF =
      llvm::GlobalIFunc::create(DeclTy, 0, llvm::Function::ExternalLinkage,
                                "", Resolver, &getModule());
  if (Entry) {
    if (GIF->getResolver() == Entry) {
      Diags.Report(IFA->getLocation(), diag::err_cyclic_alias) << 1;
      return;
    }
    assert(Entry->isDeclaration());

    // If there is a declaration in the module, then we had an extern followed
    // by the ifunc, as in:
    //   extern int test();
    //   ...
    //   int test() __attribute__((ifunc("resolver")));
    //
    // Remove it and replace uses of it with the ifunc.
    GIF->takeName(Entry);

    Entry->replaceAllUsesWith(llvm::ConstantExpr::getBitCast(GIF,
                                                          Entry->getType()));
    Entry->eraseFromParent();
  } else
    GIF->setName(MangledName);

  SetCommonAttributes(D, GIF);
}

llvm::Function *CodeGenModule::getIntrinsic(unsigned IID,
                                            ArrayRef<llvm::Type*> Tys) {
  return llvm::Intrinsic::getDeclaration(&getModule(), (llvm::Intrinsic::ID)IID,
                                         Tys);
}

static llvm::StringMapEntry<llvm::GlobalVariable *> &
GetConstantCFStringEntry(llvm::StringMap<llvm::GlobalVariable *> &Map,
                         const StringLiteral *Literal, bool TargetIsLSB,
                         bool &IsUTF16, unsigned &StringLength) {
  StringRef String = Literal->getString();
  unsigned NumBytes = String.size();

  // Check for simple case.
  if (!Literal->containsNonAsciiOrNull()) {
    StringLength = NumBytes;
    return *Map.insert(std::make_pair(String, nullptr)).first;
  }

  // Otherwise, convert the UTF8 literals into a string of shorts.
  IsUTF16 = true;

  SmallVector<llvm::UTF16, 128> ToBuf(NumBytes + 1); // +1 for ending nulls.
  const llvm::UTF8 *FromPtr = (const llvm::UTF8 *)String.data();
  llvm::UTF16 *ToPtr = &ToBuf[0];

  (void)llvm::ConvertUTF8toUTF16(&FromPtr, FromPtr + NumBytes, &ToPtr,
                                 ToPtr + NumBytes, llvm::strictConversion);

  // ConvertUTF8toUTF16 returns the length in ToPtr.
  StringLength = ToPtr - &ToBuf[0];

  // Add an explicit null.
  *ToPtr = 0;
  return *Map.insert(std::make_pair(
                         StringRef(reinterpret_cast<const char *>(ToBuf.data()),
                                   (StringLength + 1) * 2),
                         nullptr)).first;
}

ConstantAddress
CodeGenModule::GetAddrOfConstantCFString(const StringLiteral *Literal) {
  unsigned StringLength = 0;
  bool isUTF16 = false;
  llvm::StringMapEntry<llvm::GlobalVariable *> &Entry =
      GetConstantCFStringEntry(CFConstantStringMap, Literal,
                               getDataLayout().isLittleEndian(), isUTF16,
                               StringLength);

  if (auto *C = Entry.second)
    return ConstantAddress(C, CharUnits::fromQuantity(C->getAlignment()));

  llvm::Constant *Zero = llvm::Constant::getNullValue(Int32Ty);
  llvm::Constant *Zeros[] = { Zero, Zero };

  // If we don't already have it, get __CFConstantStringClassReference.
  if (!CFConstantStringClassRef) {
    llvm::Type *Ty = getTypes().ConvertType(getContext().IntTy);
    Ty = llvm::ArrayType::get(Ty, 0);
    llvm::Constant *GV =
        CreateRuntimeVariable(Ty, "__CFConstantStringClassReference");

    if (getTriple().isOSBinFormatCOFF()) {
      IdentifierInfo &II = getContext().Idents.get(GV->getName());
      TranslationUnitDecl *TUDecl = getContext().getTranslationUnitDecl();
      DeclContext *DC = TranslationUnitDecl::castToDeclContext(TUDecl);
      llvm::GlobalValue *CGV = cast<llvm::GlobalValue>(GV);

      const VarDecl *VD = nullptr;
      for (const auto &Result : DC->lookup(&II))
        if ((VD = dyn_cast<VarDecl>(Result)))
          break;

      if (!VD || !VD->hasAttr<DLLExportAttr>()) {
        CGV->setDLLStorageClass(llvm::GlobalValue::DLLImportStorageClass);
        CGV->setLinkage(llvm::GlobalValue::ExternalLinkage);
      } else {
        CGV->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
        CGV->setLinkage(llvm::GlobalValue::ExternalLinkage);
      }
    }

    // Decay array -> ptr
    CFConstantStringClassRef =
        llvm::ConstantExpr::getGetElementPtr(Ty, GV, Zeros);
  }

  QualType CFTy = getContext().getCFConstantStringType();

  auto *STy = cast<llvm::StructType>(getTypes().ConvertType(CFTy));

  ConstantInitBuilder Builder(*this);
  auto Fields = Builder.beginStruct(STy);

  // Class pointer.
  Fields.add(cast<llvm::ConstantExpr>(CFConstantStringClassRef));

  // Flags.
  Fields.addInt(IntTy, isUTF16 ? 0x07d0 : 0x07C8);

  // String pointer.
  llvm::Constant *C = nullptr;
  if (isUTF16) {
    auto Arr = llvm::makeArrayRef(
        reinterpret_cast<uint16_t *>(const_cast<char *>(Entry.first().data())),
        Entry.first().size() / 2);
    C = llvm::ConstantDataArray::get(VMContext, Arr);
  } else {
    C = llvm::ConstantDataArray::getString(VMContext, Entry.first());
  }

  // Note: -fwritable-strings doesn't make the backing store strings of
  // CFStrings writable. (See <rdar://problem/10657500>)
  auto *GV =
      new llvm::GlobalVariable(getModule(), C->getType(), /*isConstant=*/true,
                               llvm::GlobalValue::PrivateLinkage, C, ".str");
  GV->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  // Don't enforce the target's minimum global alignment, since the only use
  // of the string is via this class initializer.
  CharUnits Align = isUTF16
                        ? getContext().getTypeAlignInChars(getContext().ShortTy)
                        : getContext().getTypeAlignInChars(getContext().CharTy);
  GV->setAlignment(Align.getQuantity());

  // FIXME: We set the section explicitly to avoid a bug in ld64 224.1.
  // Without it LLVM can merge the string with a non unnamed_addr one during
  // LTO.  Doing that changes the section it ends in, which surprises ld64.
  if (getTriple().isOSBinFormatMachO())
    GV->setSection(isUTF16 ? "__TEXT,__ustring"
                           : "__TEXT,__cstring,cstring_literals");

  // String.
  llvm::Constant *Str =
      llvm::ConstantExpr::getGetElementPtr(GV->getValueType(), GV, Zeros);

  if (isUTF16)
    // Cast the UTF16 string to the correct type.
    Str = llvm::ConstantExpr::getBitCast(Str, Int8PtrTy);
  Fields.add(Str);

  // String length.
  auto Ty = getTypes().ConvertType(getContext().LongTy);
  Fields.addInt(cast<llvm::IntegerType>(Ty), StringLength);

  CharUnits Alignment = getPointerAlign();

  // The struct.
  GV = Fields.finishAndCreateGlobal("_unnamed_cfstring_", Alignment,
                                    /*isConstant=*/false,
                                    llvm::GlobalVariable::PrivateLinkage);
  switch (getTriple().getObjectFormat()) {
  case llvm::Triple::UnknownObjectFormat:
    llvm_unreachable("unknown file format");
  case llvm::Triple::COFF:
  case llvm::Triple::ELF:
  case llvm::Triple::Wasm:
    GV->setSection("cfstring");
    break;
  case llvm::Triple::MachO:
    GV->setSection("__DATA,__cfstring");
    break;
  }
  Entry.second = GV;

  return ConstantAddress(GV, Alignment);
}

QualType CodeGenModule::getObjCFastEnumerationStateType() {
  if (ObjCFastEnumerationStateType.isNull()) {
    RecordDecl *D = Context.buildImplicitRecord("__objcFastEnumerationState");
    D->startDefinition();
    
    QualType FieldTypes[] = {
      Context.UnsignedLongTy,
      Context.getPointerType(Context.getObjCIdType()),
      Context.getPointerType(Context.UnsignedLongTy),
      Context.getConstantArrayType(Context.UnsignedLongTy,
                           llvm::APInt(32, 5), ArrayType::Normal, 0)
    };
    
    for (size_t i = 0; i < 4; ++i) {
      FieldDecl *Field = FieldDecl::Create(Context,
                                           D,
                                           SourceLocation(),
                                           SourceLocation(), nullptr,
                                           FieldTypes[i], /*TInfo=*/nullptr,
                                           /*BitWidth=*/nullptr,
                                           /*Mutable=*/false,
                                           ICIS_NoInit);
      Field->setAccess(AS_public);
      D->addDecl(Field);
    }
    
    D->completeDefinition();
    ObjCFastEnumerationStateType = Context.getTagDeclType(D);
  }
  
  return ObjCFastEnumerationStateType;
}

llvm::Constant *
CodeGenModule::GetConstantArrayFromStringLiteral(const StringLiteral *E) {
  assert(!E->getType()->isPointerType() && "Strings are always arrays");
  
  // Don't emit it as the address of the string, emit the string data itself
  // as an inline array.
  if (E->getCharByteWidth() == 1) {
    SmallString<64> Str(E->getString());

    // Resize the string to the right size, which is indicated by its type.
    const ConstantArrayType *CAT = Context.getAsConstantArrayType(E->getType());
    Str.resize(CAT->getSize().getZExtValue());
    return llvm::ConstantDataArray::getString(VMContext, Str, false);
  }

  auto *AType = cast<llvm::ArrayType>(getTypes().ConvertType(E->getType()));
  llvm::Type *ElemTy = AType->getElementType();
  unsigned NumElements = AType->getNumElements();

  // Wide strings have either 2-byte or 4-byte elements.
  if (ElemTy->getPrimitiveSizeInBits() == 16) {
    SmallVector<uint16_t, 32> Elements;
    Elements.reserve(NumElements);

    for(unsigned i = 0, e = E->getLength(); i != e; ++i)
      Elements.push_back(E->getCodeUnit(i));
    Elements.resize(NumElements);
    return llvm::ConstantDataArray::get(VMContext, Elements);
  }
  
  assert(ElemTy->getPrimitiveSizeInBits() == 32);
  SmallVector<uint32_t, 32> Elements;
  Elements.reserve(NumElements);
  
  for(unsigned i = 0, e = E->getLength(); i != e; ++i)
    Elements.push_back(E->getCodeUnit(i));
  Elements.resize(NumElements);
  return llvm::ConstantDataArray::get(VMContext, Elements);
}

static llvm::GlobalVariable *
GenerateStringLiteral(llvm::Constant *C, llvm::GlobalValue::LinkageTypes LT,
                      CodeGenModule &CGM, StringRef GlobalName,
                      CharUnits Alignment) {
  // OpenCL v1.2 s6.5.3: a string literal is in the constant address space.
  unsigned AddrSpace = 0;
  if (CGM.getLangOpts().OpenCL)
    AddrSpace = CGM.getContext().getTargetAddressSpace(LangAS::opencl_constant);

  llvm::Module &M = CGM.getModule();
  // Create a global variable for this string
  auto *GV = new llvm::GlobalVariable(
      M, C->getType(), !CGM.getLangOpts().WritableStrings, LT, C, GlobalName,
      nullptr, llvm::GlobalVariable::NotThreadLocal, AddrSpace);
  GV->setAlignment(Alignment.getQuantity());
  GV->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  if (GV->isWeakForLinker()) {
    assert(CGM.supportsCOMDAT() && "Only COFF uses weak string literals");
    GV->setComdat(M.getOrInsertComdat(GV->getName()));
  }

  return GV;
}

/// GetAddrOfConstantStringFromLiteral - Return a pointer to a
/// constant array for the given string literal.
ConstantAddress
CodeGenModule::GetAddrOfConstantStringFromLiteral(const StringLiteral *S,
                                                  StringRef Name) {
  CharUnits Alignment = getContext().getAlignOfGlobalVarInChars(S->getType());

  llvm::Constant *C = GetConstantArrayFromStringLiteral(S);
  llvm::GlobalVariable **Entry = nullptr;
  if (!LangOpts.WritableStrings) {
    Entry = &ConstantStringMap[C];
    if (auto GV = *Entry) {
      if (Alignment.getQuantity() > GV->getAlignment())
        GV->setAlignment(Alignment.getQuantity());
      return ConstantAddress(GV, Alignment);
    }
  }

  SmallString<256> MangledNameBuffer;
  StringRef GlobalVariableName;
  llvm::GlobalValue::LinkageTypes LT;

  // Mangle the string literal if the ABI allows for it.  However, we cannot
  // do this if  we are compiling with ASan or -fwritable-strings because they
  // rely on strings having normal linkage.
  if (!LangOpts.WritableStrings &&
      !LangOpts.Sanitize.has(SanitizerKind::Address) &&
      getCXXABI().getMangleContext().shouldMangleStringLiteral(S)) {
    llvm::raw_svector_ostream Out(MangledNameBuffer);
    getCXXABI().getMangleContext().mangleStringLiteral(S, Out);

    LT = llvm::GlobalValue::LinkOnceODRLinkage;
    GlobalVariableName = MangledNameBuffer;
  } else {
    LT = llvm::GlobalValue::PrivateLinkage;
    GlobalVariableName = Name;
  }

  auto GV = GenerateStringLiteral(C, LT, *this, GlobalVariableName, Alignment);
  if (Entry)
    *Entry = GV;

  SanitizerMD->reportGlobalToASan(GV, S->getStrTokenLoc(0), "<string literal>",
                                  QualType());
  return ConstantAddress(GV, Alignment);
}

/// GetAddrOfConstantStringFromObjCEncode - Return a pointer to a constant
/// array for the given ObjCEncodeExpr node.
ConstantAddress
CodeGenModule::GetAddrOfConstantStringFromObjCEncode(const ObjCEncodeExpr *E) {
  std::string Str;
  getContext().getObjCEncodingForType(E->getEncodedType(), Str);

  return GetAddrOfConstantCString(Str);
}

/// GetAddrOfConstantCString - Returns a pointer to a character array containing
/// the literal and a terminating '\0' character.
/// The result has pointer to array type.
ConstantAddress CodeGenModule::GetAddrOfConstantCString(
    const std::string &Str, const char *GlobalName) {
  StringRef StrWithNull(Str.c_str(), Str.size() + 1);
  CharUnits Alignment =
    getContext().getAlignOfGlobalVarInChars(getContext().CharTy);

  llvm::Constant *C =
      llvm::ConstantDataArray::getString(getLLVMContext(), StrWithNull, false);

  // Don't share any string literals if strings aren't constant.
  llvm::GlobalVariable **Entry = nullptr;
  if (!LangOpts.WritableStrings) {
    Entry = &ConstantStringMap[C];
    if (auto GV = *Entry) {
      if (Alignment.getQuantity() > GV->getAlignment())
        GV->setAlignment(Alignment.getQuantity());
      return ConstantAddress(GV, Alignment);
    }
  }

  // Get the default prefix if a name wasn't specified.
  if (!GlobalName)
    GlobalName = ".str";
  // Create a global variable for this.
  auto GV = GenerateStringLiteral(C, llvm::GlobalValue::PrivateLinkage, *this,
                                  GlobalName, Alignment);
  if (Entry)
    *Entry = GV;
  return ConstantAddress(GV, Alignment);
}

ConstantAddress CodeGenModule::GetAddrOfGlobalTemporary(
    const MaterializeTemporaryExpr *E, const Expr *Init) {
  assert((E->getStorageDuration() == SD_Static ||
          E->getStorageDuration() == SD_Thread) && "not a global temporary");
  const auto *VD = cast<VarDecl>(E->getExtendingDecl());

  // If we're not materializing a subobject of the temporary, keep the
  // cv-qualifiers from the type of the MaterializeTemporaryExpr.
  QualType MaterializedType = Init->getType();
  if (Init == E->GetTemporaryExpr())
    MaterializedType = E->getType();

  CharUnits Align = getContext().getTypeAlignInChars(MaterializedType);

  if (llvm::Constant *Slot = MaterializedGlobalTemporaryMap[E])
    return ConstantAddress(Slot, Align);

  // FIXME: If an externally-visible declaration extends multiple temporaries,
  // we need to give each temporary the same name in every translation unit (and
  // we also need to make the temporaries externally-visible).
  SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  getCXXABI().getMangleContext().mangleReferenceTemporary(
      VD, E->getManglingNumber(), Out);

  APValue *Value = nullptr;
  if (E->getStorageDuration() == SD_Static) {
    // We might have a cached constant initializer for this temporary. Note
    // that this might have a different value from the value computed by
    // evaluating the initializer if the surrounding constant expression
    // modifies the temporary.
    Value = getContext().getMaterializedTemporaryValue(E, false);
    if (Value && Value->isUninit())
      Value = nullptr;
  }

  // Try evaluating it now, it might have a constant initializer.
  Expr::EvalResult EvalResult;
  if (!Value && Init->EvaluateAsRValue(EvalResult, getContext()) &&
      !EvalResult.hasSideEffects())
    Value = &EvalResult.Val;

  llvm::Constant *InitialValue = nullptr;
  bool Constant = false;
  llvm::Type *Type;
  if (Value) {
    // The temporary has a constant initializer, use it.
    InitialValue = EmitConstantValue(*Value, MaterializedType, nullptr);
    Constant = isTypeConstant(MaterializedType, /*ExcludeCtor*/Value);
    Type = InitialValue->getType();
  } else {
    // No initializer, the initialization will be provided when we
    // initialize the declaration which performed lifetime extension.
    Type = getTypes().ConvertTypeForMem(MaterializedType);
  }

  // Create a global variable for this lifetime-extended temporary.
  llvm::GlobalValue::LinkageTypes Linkage =
      getLLVMLinkageVarDefinition(VD, Constant);
  if (Linkage == llvm::GlobalVariable::ExternalLinkage) {
    const VarDecl *InitVD;
    if (VD->isStaticDataMember() && VD->getAnyInitializer(InitVD) &&
        isa<CXXRecordDecl>(InitVD->getLexicalDeclContext())) {
      // Temporaries defined inside a class get linkonce_odr linkage because the
      // class can be defined in multipe translation units.
      Linkage = llvm::GlobalVariable::LinkOnceODRLinkage;
    } else {
      // There is no need for this temporary to have external linkage if the
      // VarDecl has external linkage.
      Linkage = llvm::GlobalVariable::InternalLinkage;
    }
  }
  unsigned AddrSpace = GetGlobalVarAddressSpace(
      VD, getContext().getTargetAddressSpace(MaterializedType));
  auto *GV = new llvm::GlobalVariable(
      getModule(), Type, Constant, Linkage, InitialValue, Name.c_str(),
      /*InsertBefore=*/nullptr, llvm::GlobalVariable::NotThreadLocal,
      AddrSpace);
  setGlobalVisibility(GV, VD);
  GV->setAlignment(Align.getQuantity());
  if (supportsCOMDAT() && GV->isWeakForLinker())
    GV->setComdat(TheModule.getOrInsertComdat(GV->getName()));
  if (VD->getTLSKind())
    setTLSMode(GV, *VD);
  MaterializedGlobalTemporaryMap[E] = GV;
  return ConstantAddress(GV, Align);
}

/// EmitObjCPropertyImplementations - Emit information for synthesized
/// properties for an implementation.
void CodeGenModule::EmitObjCPropertyImplementations(const
                                                    ObjCImplementationDecl *D) {
  for (const auto *PID : D->property_impls()) {
    // Dynamic is just for type-checking.
    if (PID->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize) {
      ObjCPropertyDecl *PD = PID->getPropertyDecl();

      // Determine which methods need to be implemented, some may have
      // been overridden. Note that ::isPropertyAccessor is not the method
      // we want, that just indicates if the decl came from a
      // property. What we want to know is if the method is defined in
      // this implementation.
      if (!D->getInstanceMethod(PD->getGetterName()))
        CodeGenFunction(*this).GenerateObjCGetter(
                                 const_cast<ObjCImplementationDecl *>(D), PID);
      if (!PD->isReadOnly() &&
          !D->getInstanceMethod(PD->getSetterName()))
        CodeGenFunction(*this).GenerateObjCSetter(
                                 const_cast<ObjCImplementationDecl *>(D), PID);
    }
  }
}

static bool needsDestructMethod(ObjCImplementationDecl *impl) {
  const ObjCInterfaceDecl *iface = impl->getClassInterface();
  for (const ObjCIvarDecl *ivar = iface->all_declared_ivar_begin();
       ivar; ivar = ivar->getNextIvar())
    if (ivar->getType().isDestructedType())
      return true;

  return false;
}

static bool AllTrivialInitializers(CodeGenModule &CGM,
                                   ObjCImplementationDecl *D) {
  CodeGenFunction CGF(CGM);
  for (ObjCImplementationDecl::init_iterator B = D->init_begin(),
       E = D->init_end(); B != E; ++B) {
    CXXCtorInitializer *CtorInitExp = *B;
    Expr *Init = CtorInitExp->getInit();
    if (!CGF.isTrivialInitializer(Init))
      return false;
  }
  return true;
}

/// EmitObjCIvarInitializations - Emit information for ivar initialization
/// for an implementation.
void CodeGenModule::EmitObjCIvarInitializations(ObjCImplementationDecl *D) {
  // We might need a .cxx_destruct even if we don't have any ivar initializers.
  if (needsDestructMethod(D)) {
    IdentifierInfo *II = &getContext().Idents.get(".cxx_destruct");
    Selector cxxSelector = getContext().Selectors.getSelector(0, &II);
    ObjCMethodDecl *DTORMethod =
      ObjCMethodDecl::Create(getContext(), D->getLocation(), D->getLocation(),
                             cxxSelector, getContext().VoidTy, nullptr, D,
                             /*isInstance=*/true, /*isVariadic=*/false,
                          /*isPropertyAccessor=*/true, /*isImplicitlyDeclared=*/true,
                             /*isDefined=*/false, ObjCMethodDecl::Required);
    D->addInstanceMethod(DTORMethod);
    CodeGenFunction(*this).GenerateObjCCtorDtorMethod(D, DTORMethod, false);
    D->setHasDestructors(true);
  }

  // If the implementation doesn't have any ivar initializers, we don't need
  // a .cxx_construct.
  if (D->getNumIvarInitializers() == 0 ||
      AllTrivialInitializers(*this, D))
    return;
  
  IdentifierInfo *II = &getContext().Idents.get(".cxx_construct");
  Selector cxxSelector = getContext().Selectors.getSelector(0, &II);
  // The constructor returns 'self'.
  ObjCMethodDecl *CTORMethod = ObjCMethodDecl::Create(getContext(), 
                                                D->getLocation(),
                                                D->getLocation(),
                                                cxxSelector,
                                                getContext().getObjCIdType(),
                                                nullptr, D, /*isInstance=*/true,
                                                /*isVariadic=*/false,
                                                /*isPropertyAccessor=*/true,
                                                /*isImplicitlyDeclared=*/true,
                                                /*isDefined=*/false,
                                                ObjCMethodDecl::Required);
  D->addInstanceMethod(CTORMethod);
  CodeGenFunction(*this).GenerateObjCCtorDtorMethod(D, CTORMethod, true);
  D->setHasNonZeroConstructors(true);
}

// EmitLinkageSpec - Emit all declarations in a linkage spec.
void CodeGenModule::EmitLinkageSpec(const LinkageSpecDecl *LSD) {
  if (LSD->getLanguage() != LinkageSpecDecl::lang_c &&
      LSD->getLanguage() != LinkageSpecDecl::lang_cxx) {
    ErrorUnsupported(LSD, "linkage spec");
    return;
  }

  EmitDeclContext(LSD);
}

void CodeGenModule::EmitDeclContext(const DeclContext *DC) {
  for (auto *I : DC->decls()) {
    // Unlike other DeclContexts, the contents of an ObjCImplDecl at TU scope
    // are themselves considered "top-level", so EmitTopLevelDecl on an
    // ObjCImplDecl does not recursively visit them. We need to do that in
    // case they're nested inside another construct (LinkageSpecDecl /
    // ExportDecl) that does stop them from being considered "top-level".
    if (auto *OID = dyn_cast<ObjCImplDecl>(I)) {
      for (auto *M : OID->methods())
        EmitTopLevelDecl(M);
    }

    EmitTopLevelDecl(I);
  }
}

/// EmitTopLevelDecl - Emit code for a single top level declaration.
void CodeGenModule::EmitTopLevelDecl(Decl *D) {
  // Ignore dependent declarations.
  if (D->getDeclContext() && D->getDeclContext()->isDependentContext())
    return;

  switch (D->getKind()) {
  case Decl::CXXConversion:
  case Decl::CXXMethod:
  case Decl::Function:
    // Skip function templates
    if (cast<FunctionDecl>(D)->getDescribedFunctionTemplate() ||
        cast<FunctionDecl>(D)->isLateTemplateParsed())
      return;

    EmitGlobal(cast<FunctionDecl>(D));
    // Always provide some coverage mapping
    // even for the functions that aren't emitted.
    AddDeferredUnusedCoverageMapping(D);
    break;

  case Decl::CXXDeductionGuide:
    // Function-like, but does not result in code emission.
    break;

  case Decl::Var:
  case Decl::Decomposition:
    // Skip variable templates
    if (cast<VarDecl>(D)->getDescribedVarTemplate())
      return;
    LLVM_FALLTHROUGH;
  case Decl::VarTemplateSpecialization:
    EmitGlobal(cast<VarDecl>(D));
    if (auto *DD = dyn_cast<DecompositionDecl>(D))
      for (auto *B : DD->bindings())
        if (auto *HD = B->getHoldingVar())
          EmitGlobal(HD);
    break;

  // Indirect fields from global anonymous structs and unions can be
  // ignored; only the actual variable requires IR gen support.
  case Decl::IndirectField:
    break;

  // C++ Decls
  case Decl::Namespace:
    EmitDeclContext(cast<NamespaceDecl>(D));
    break;
  case Decl::CXXRecord:
    if (DebugInfo) {
      if (auto *ES = D->getASTContext().getExternalSource())
        if (ES->hasExternalDefinitions(D) == ExternalASTSource::EK_Never)
          DebugInfo->completeUnusedClass(cast<CXXRecordDecl>(*D));
    }
    // Emit any static data members, they may be definitions.
    for (auto *I : cast<CXXRecordDecl>(D)->decls())
      if (isa<VarDecl>(I) || isa<CXXRecordDecl>(I))
        EmitTopLevelDecl(I);
    break;
    // No code generation needed.
  case Decl::UsingShadow:
  case Decl::ClassTemplate:
  case Decl::VarTemplate:
  case Decl::VarTemplatePartialSpecialization:
  case Decl::FunctionTemplate:
  case Decl::TypeAliasTemplate:
  case Decl::Block:
  case Decl::Empty:
    break;
  case Decl::Using:          // using X; [C++]
    if (CGDebugInfo *DI = getModuleDebugInfo())
        DI->EmitUsingDecl(cast<UsingDecl>(*D));
    return;
  case Decl::NamespaceAlias:
    if (CGDebugInfo *DI = getModuleDebugInfo())
        DI->EmitNamespaceAlias(cast<NamespaceAliasDecl>(*D));
    return;
  case Decl::UsingDirective: // using namespace X; [C++]
    if (CGDebugInfo *DI = getModuleDebugInfo())
      DI->EmitUsingDirective(cast<UsingDirectiveDecl>(*D));
    return;
  case Decl::CXXConstructor:
    // Skip function templates
    if (cast<FunctionDecl>(D)->getDescribedFunctionTemplate() ||
        cast<FunctionDecl>(D)->isLateTemplateParsed())
      return;
      
    getCXXABI().EmitCXXConstructors(cast<CXXConstructorDecl>(D));
    break;
  case Decl::CXXDestructor:
    if (cast<FunctionDecl>(D)->isLateTemplateParsed())
      return;
    getCXXABI().EmitCXXDestructors(cast<CXXDestructorDecl>(D));
    break;

  case Decl::StaticAssert:
    // Nothing to do.
    break;

  // Objective-C Decls

  // Forward declarations, no (immediate) code generation.
  case Decl::ObjCInterface:
  case Decl::ObjCCategory:
    break;

  case Decl::ObjCProtocol: {
    auto *Proto = cast<ObjCProtocolDecl>(D);
    if (Proto->isThisDeclarationADefinition())
      ObjCRuntime->GenerateProtocol(Proto);
    break;
  }
      
  case Decl::ObjCCategoryImpl:
    // Categories have properties but don't support synthesize so we
    // can ignore them here.
    ObjCRuntime->GenerateCategory(cast<ObjCCategoryImplDecl>(D));
    break;

  case Decl::ObjCImplementation: {
    auto *OMD = cast<ObjCImplementationDecl>(D);
    EmitObjCPropertyImplementations(OMD);
    EmitObjCIvarInitializations(OMD);
    ObjCRuntime->GenerateClass(OMD);
    // Emit global variable debug information.
    if (CGDebugInfo *DI = getModuleDebugInfo())
      if (getCodeGenOpts().getDebugInfo() >= codegenoptions::LimitedDebugInfo)
        DI->getOrCreateInterfaceType(getContext().getObjCInterfaceType(
            OMD->getClassInterface()), OMD->getLocation());
    break;
  }
  case Decl::ObjCMethod: {
    auto *OMD = cast<ObjCMethodDecl>(D);
    // If this is not a prototype, emit the body.
    if (OMD->getBody())
      CodeGenFunction(*this).GenerateObjCMethod(OMD);
    break;
  }
  case Decl::ObjCCompatibleAlias:
    ObjCRuntime->RegisterAlias(cast<ObjCCompatibleAliasDecl>(D));
    break;

  case Decl::PragmaComment: {
    const auto *PCD = cast<PragmaCommentDecl>(D);
    switch (PCD->getCommentKind()) {
    case PCK_Unknown:
      llvm_unreachable("unexpected pragma comment kind");
    case PCK_Linker:
      AppendLinkerOptions(PCD->getArg());
      break;
    case PCK_Lib:
      AddDependentLib(PCD->getArg());
      break;
    case PCK_Compiler:
    case PCK_ExeStr:
    case PCK_User:
      break; // We ignore all of these.
    }
    break;
  }

  case Decl::PragmaDetectMismatch: {
    const auto *PDMD = cast<PragmaDetectMismatchDecl>(D);
    AddDetectMismatch(PDMD->getName(), PDMD->getValue());
    break;
  }

  case Decl::LinkageSpec:
    EmitLinkageSpec(cast<LinkageSpecDecl>(D));
    break;

  case Decl::FileScopeAsm: {
    // File-scope asm is ignored during device-side CUDA compilation.
    if (LangOpts.CUDA && LangOpts.CUDAIsDevice)
      break;
    // File-scope asm is ignored during device-side OpenMP compilation.
    if (LangOpts.OpenMPIsDevice)
      break;
    auto *AD = cast<FileScopeAsmDecl>(D);
    getModule().appendModuleInlineAsm(AD->getAsmString()->getString());
    break;
  }

  case Decl::Import: {
    auto *Import = cast<ImportDecl>(D);

    // If we've already imported this module, we're done.
    if (!ImportedModules.insert(Import->getImportedModule()))
      break;

    // Emit debug information for direct imports.
    if (!Import->getImportedOwningModule()) {
      if (CGDebugInfo *DI = getModuleDebugInfo())
        DI->EmitImportDecl(*Import);
    }

    // Find all of the submodules and emit the module initializers.
    llvm::SmallPtrSet<clang::Module *, 16> Visited;
    SmallVector<clang::Module *, 16> Stack;
    Visited.insert(Import->getImportedModule());
    Stack.push_back(Import->getImportedModule());

    while (!Stack.empty()) {
      clang::Module *Mod = Stack.pop_back_val();
      if (!EmittedModuleInitializers.insert(Mod).second)
        continue;

      for (auto *D : Context.getModuleInitializers(Mod))
        EmitTopLevelDecl(D);

      // Visit the submodules of this module.
      for (clang::Module::submodule_iterator Sub = Mod->submodule_begin(),
                                             SubEnd = Mod->submodule_end();
           Sub != SubEnd; ++Sub) {
        // Skip explicit children; they need to be explicitly imported to emit
        // the initializers.
        if ((*Sub)->IsExplicit)
          continue;

        if (Visited.insert(*Sub).second)
          Stack.push_back(*Sub);
      }
    }
    break;
  }

  case Decl::Export:
    EmitDeclContext(cast<ExportDecl>(D));
    break;

  case Decl::OMPThreadPrivate:
    EmitOMPThreadPrivateDecl(cast<OMPThreadPrivateDecl>(D));
    break;

  case Decl::ClassTemplateSpecialization: {
    const auto *Spec = cast<ClassTemplateSpecializationDecl>(D);
    if (DebugInfo &&
        Spec->getSpecializationKind() == TSK_ExplicitInstantiationDefinition &&
        Spec->hasDefinition())
      DebugInfo->completeTemplateDefinition(*Spec);
    break;
  }

  case Decl::OMPDeclareReduction:
    EmitOMPDeclareReduction(cast<OMPDeclareReductionDecl>(D));
    break;

  default:
    // Make sure we handled everything we should, every other kind is a
    // non-top-level decl.  FIXME: Would be nice to have an isTopLevelDeclKind
    // function. Need to recode Decl::Kind to do that easily.
    assert(isa<TypeDecl>(D) && "Unsupported decl kind");
    break;
  }
}

void CodeGenModule::AddDeferredUnusedCoverageMapping(Decl *D) {
  // Do we need to generate coverage mapping?
  if (!CodeGenOpts.CoverageMapping)
    return;
  switch (D->getKind()) {
  case Decl::CXXConversion:
  case Decl::CXXMethod:
  case Decl::Function:
  case Decl::ObjCMethod:
  case Decl::CXXConstructor:
  case Decl::CXXDestructor: {
    if (!cast<FunctionDecl>(D)->doesThisDeclarationHaveABody())
      return;
    auto I = DeferredEmptyCoverageMappingDecls.find(D);
    if (I == DeferredEmptyCoverageMappingDecls.end())
      DeferredEmptyCoverageMappingDecls[D] = true;
    break;
  }
  default:
    break;
  };
}

void CodeGenModule::ClearUnusedCoverageMapping(const Decl *D) {
  // Do we need to generate coverage mapping?
  if (!CodeGenOpts.CoverageMapping)
    return;
  if (const auto *Fn = dyn_cast<FunctionDecl>(D)) {
    if (Fn->isTemplateInstantiation())
      ClearUnusedCoverageMapping(Fn->getTemplateInstantiationPattern());
  }
  auto I = DeferredEmptyCoverageMappingDecls.find(D);
  if (I == DeferredEmptyCoverageMappingDecls.end())
    DeferredEmptyCoverageMappingDecls[D] = false;
  else
    I->second = false;
}

void CodeGenModule::EmitDeferredUnusedCoverageMappings() {
  std::vector<const Decl *> DeferredDecls;
  for (const auto &I : DeferredEmptyCoverageMappingDecls) {
    if (!I.second)
      continue;
    DeferredDecls.push_back(I.first);
  }
  // Sort the declarations by their location to make sure that the tests get a
  // predictable order for the coverage mapping for the unused declarations.
  if (CodeGenOpts.DumpCoverageMapping)
    std::sort(DeferredDecls.begin(), DeferredDecls.end(),
              [] (const Decl *LHS, const Decl *RHS) {
      return LHS->getLocStart() < RHS->getLocStart();
    });
  for (const auto *D : DeferredDecls) {
    switch (D->getKind()) {
    case Decl::CXXConversion:
    case Decl::CXXMethod:
    case Decl::Function:
    case Decl::ObjCMethod: {
      CodeGenPGO PGO(*this);
      GlobalDecl GD(cast<FunctionDecl>(D));
      PGO.emitEmptyCounterMapping(D, getMangledName(GD),
                                  getFunctionLinkage(GD));
      break;
    }
    case Decl::CXXConstructor: {
      CodeGenPGO PGO(*this);
      GlobalDecl GD(cast<CXXConstructorDecl>(D), Ctor_Base);
      PGO.emitEmptyCounterMapping(D, getMangledName(GD),
                                  getFunctionLinkage(GD));
      break;
    }
    case Decl::CXXDestructor: {
      CodeGenPGO PGO(*this);
      GlobalDecl GD(cast<CXXDestructorDecl>(D), Dtor_Base);
      PGO.emitEmptyCounterMapping(D, getMangledName(GD),
                                  getFunctionLinkage(GD));
      break;
    }
    default:
      break;
    };
  }
}

/// Turns the given pointer into a constant.
static llvm::Constant *GetPointerConstant(llvm::LLVMContext &Context,
                                          const void *Ptr) {
  uintptr_t PtrInt = reinterpret_cast<uintptr_t>(Ptr);
  llvm::Type *i64 = llvm::Type::getInt64Ty(Context);
  return llvm::ConstantInt::get(i64, PtrInt);
}

static void EmitGlobalDeclMetadata(CodeGenModule &CGM,
                                   llvm::NamedMDNode *&GlobalMetadata,
                                   GlobalDecl D,
                                   llvm::GlobalValue *Addr) {
  if (!GlobalMetadata)
    GlobalMetadata =
      CGM.getModule().getOrInsertNamedMetadata("clang.global.decl.ptrs");

  // TODO: should we report variant information for ctors/dtors?
  llvm::Metadata *Ops[] = {llvm::ConstantAsMetadata::get(Addr),
                           llvm::ConstantAsMetadata::get(GetPointerConstant(
                               CGM.getLLVMContext(), D.getDecl()))};
  GlobalMetadata->addOperand(llvm::MDNode::get(CGM.getLLVMContext(), Ops));
}

/// For each function which is declared within an extern "C" region and marked
/// as 'used', but has internal linkage, create an alias from the unmangled
/// name to the mangled name if possible. People expect to be able to refer
/// to such functions with an unmangled name from inline assembly within the
/// same translation unit.
void CodeGenModule::EmitStaticExternCAliases() {
  // Don't do anything if we're generating CUDA device code -- the NVPTX
  // assembly target doesn't support aliases.
  if (Context.getTargetInfo().getTriple().isNVPTX())
    return;
  for (auto &I : StaticExternCValues) {
    IdentifierInfo *Name = I.first;
    llvm::GlobalValue *Val = I.second;
    if (Val && !getModule().getNamedValue(Name->getName()))
      addUsedGlobal(llvm::GlobalAlias::create(Name->getName(), Val));
  }
}

bool CodeGenModule::lookupRepresentativeDecl(StringRef MangledName,
                                             GlobalDecl &Result) const {
  auto Res = Manglings.find(MangledName);
  if (Res == Manglings.end())
    return false;
  Result = Res->getValue();
  return true;
}

/// Emits metadata nodes associating all the global values in the
/// current module with the Decls they came from.  This is useful for
/// projects using IR gen as a subroutine.
///
/// Since there's currently no way to associate an MDNode directly
/// with an llvm::GlobalValue, we create a global named metadata
/// with the name 'clang.global.decl.ptrs'.
void CodeGenModule::EmitDeclMetadata() {
  llvm::NamedMDNode *GlobalMetadata = nullptr;

  for (auto &I : MangledDeclNames) {
    llvm::GlobalValue *Addr = getModule().getNamedValue(I.second);
    // Some mangled names don't necessarily have an associated GlobalValue
    // in this module, e.g. if we mangled it for DebugInfo.
    if (Addr)
      EmitGlobalDeclMetadata(*this, GlobalMetadata, I.first, Addr);
  }
}

/// Emits metadata nodes for all the local variables in the current
/// function.
void CodeGenFunction::EmitDeclMetadata() {
  if (LocalDeclMap.empty()) return;

  llvm::LLVMContext &Context = getLLVMContext();

  // Find the unique metadata ID for this name.
  unsigned DeclPtrKind = Context.getMDKindID("clang.decl.ptr");

  llvm::NamedMDNode *GlobalMetadata = nullptr;

  for (auto &I : LocalDeclMap) {
    const Decl *D = I.first;
    llvm::Value *Addr = I.second.getPointer();
    if (auto *Alloca = dyn_cast<llvm::AllocaInst>(Addr)) {
      llvm::Value *DAddr = GetPointerConstant(getLLVMContext(), D);
      Alloca->setMetadata(
          DeclPtrKind, llvm::MDNode::get(
                           Context, llvm::ValueAsMetadata::getConstant(DAddr)));
    } else if (auto *GV = dyn_cast<llvm::GlobalValue>(Addr)) {
      GlobalDecl GD = GlobalDecl(cast<VarDecl>(D));
      EmitGlobalDeclMetadata(CGM, GlobalMetadata, GD, GV);
    }
  }
}

void CodeGenModule::EmitVersionIdentMetadata() {
  llvm::NamedMDNode *IdentMetadata =
    TheModule.getOrInsertNamedMetadata("llvm.ident");
  std::string Version = getClangFullVersion();
  llvm::LLVMContext &Ctx = TheModule.getContext();

  llvm::Metadata *IdentNode[] = {llvm::MDString::get(Ctx, Version)};
  IdentMetadata->addOperand(llvm::MDNode::get(Ctx, IdentNode));
}

void CodeGenModule::EmitTargetMetadata() {
  // Warning, new MangledDeclNames may be appended within this loop.
  // We rely on MapVector insertions adding new elements to the end
  // of the container.
  // FIXME: Move this loop into the one target that needs it, and only
  // loop over those declarations for which we couldn't emit the target
  // metadata when we emitted the declaration.
  for (unsigned I = 0; I != MangledDeclNames.size(); ++I) {
    auto Val = *(MangledDeclNames.begin() + I);
    const Decl *D = Val.first.getDecl()->getMostRecentDecl();
    llvm::GlobalValue *GV = GetGlobalValue(Val.second);
    getTargetCodeGenInfo().emitTargetMD(D, GV, *this);
  }
}

void CodeGenModule::EmitCoverageFile() {
  if (getCodeGenOpts().CoverageDataFile.empty() &&
      getCodeGenOpts().CoverageNotesFile.empty())
    return;

  llvm::NamedMDNode *CUNode = TheModule.getNamedMetadata("llvm.dbg.cu");
  if (!CUNode)
    return;

  llvm::NamedMDNode *GCov = TheModule.getOrInsertNamedMetadata("llvm.gcov");
  llvm::LLVMContext &Ctx = TheModule.getContext();
  auto *CoverageDataFile =
      llvm::MDString::get(Ctx, getCodeGenOpts().CoverageDataFile);
  auto *CoverageNotesFile =
      llvm::MDString::get(Ctx, getCodeGenOpts().CoverageNotesFile);
  for (int i = 0, e = CUNode->getNumOperands(); i != e; ++i) {
    llvm::MDNode *CU = CUNode->getOperand(i);
    llvm::Metadata *Elts[] = {CoverageNotesFile, CoverageDataFile, CU};
    GCov->addOperand(llvm::MDNode::get(Ctx, Elts));
  }
}

llvm::Constant *CodeGenModule::EmitUuidofInitializer(StringRef Uuid) {
  // Sema has checked that all uuid strings are of the form
  // "12345678-1234-1234-1234-1234567890ab".
  assert(Uuid.size() == 36);
  for (unsigned i = 0; i < 36; ++i) {
    if (i == 8 || i == 13 || i == 18 || i == 23) assert(Uuid[i] == '-');
    else                                         assert(isHexDigit(Uuid[i]));
  }

  // The starts of all bytes of Field3 in Uuid. Field 3 is "1234-1234567890ab".
  const unsigned Field3ValueOffsets[8] = { 19, 21, 24, 26, 28, 30, 32, 34 };

  llvm::Constant *Field3[8];
  for (unsigned Idx = 0; Idx < 8; ++Idx)
    Field3[Idx] = llvm::ConstantInt::get(
        Int8Ty, Uuid.substr(Field3ValueOffsets[Idx], 2), 16);

  llvm::Constant *Fields[4] = {
    llvm::ConstantInt::get(Int32Ty, Uuid.substr(0,  8), 16),
    llvm::ConstantInt::get(Int16Ty, Uuid.substr(9,  4), 16),
    llvm::ConstantInt::get(Int16Ty, Uuid.substr(14, 4), 16),
    llvm::ConstantArray::get(llvm::ArrayType::get(Int8Ty, 8), Field3)
  };

  return llvm::ConstantStruct::getAnon(Fields);
}

llvm::Constant *CodeGenModule::GetAddrOfRTTIDescriptor(QualType Ty,
                                                       bool ForEH) {
  // Return a bogus pointer if RTTI is disabled, unless it's for EH.
  // FIXME: should we even be calling this method if RTTI is disabled
  // and it's not for EH?
  if (!ForEH && !getLangOpts().RTTI)
    return llvm::Constant::getNullValue(Int8PtrTy);
  
  if (ForEH && Ty->isObjCObjectPointerType() &&
      LangOpts.ObjCRuntime.isGNUFamily())
    return ObjCRuntime->GetEHType(Ty);

  return getCXXABI().getAddrOfRTTIDescriptor(Ty);
}

void CodeGenModule::EmitOMPThreadPrivateDecl(const OMPThreadPrivateDecl *D) {
  for (auto RefExpr : D->varlists()) {
    auto *VD = cast<VarDecl>(cast<DeclRefExpr>(RefExpr)->getDecl());
    bool PerformInit =
        VD->getAnyInitializer() &&
        !VD->getAnyInitializer()->isConstantInitializer(getContext(),
                                                        /*ForRef=*/false);

    Address Addr(GetAddrOfGlobalVar(VD), getContext().getDeclAlign(VD));
    if (auto InitFunction = getOpenMPRuntime().emitThreadPrivateVarDefinition(
            VD, Addr, RefExpr->getLocStart(), PerformInit))
      CXXGlobalInits.push_back(InitFunction);
  }
}

llvm::Metadata *CodeGenModule::CreateMetadataIdentifierForType(QualType T) {
  llvm::Metadata *&InternalId = MetadataIdMap[T.getCanonicalType()];
  if (InternalId)
    return InternalId;

  if (isExternallyVisible(T->getLinkage())) {
    std::string OutName;
    llvm::raw_string_ostream Out(OutName);
    getCXXABI().getMangleContext().mangleTypeName(T, Out);

    InternalId = llvm::MDString::get(getLLVMContext(), Out.str());
  } else {
    InternalId = llvm::MDNode::getDistinct(getLLVMContext(),
                                           llvm::ArrayRef<llvm::Metadata *>());
  }

  return InternalId;
}

/// Returns whether this module needs the "all-vtables" type identifier.
bool CodeGenModule::NeedAllVtablesTypeId() const {
  // Returns true if at least one of vtable-based CFI checkers is enabled and
  // is not in the trapping mode.
  return ((LangOpts.Sanitize.has(SanitizerKind::CFIVCall) &&
           !CodeGenOpts.SanitizeTrap.has(SanitizerKind::CFIVCall)) ||
          (LangOpts.Sanitize.has(SanitizerKind::CFINVCall) &&
           !CodeGenOpts.SanitizeTrap.has(SanitizerKind::CFINVCall)) ||
          (LangOpts.Sanitize.has(SanitizerKind::CFIDerivedCast) &&
           !CodeGenOpts.SanitizeTrap.has(SanitizerKind::CFIDerivedCast)) ||
          (LangOpts.Sanitize.has(SanitizerKind::CFIUnrelatedCast) &&
           !CodeGenOpts.SanitizeTrap.has(SanitizerKind::CFIUnrelatedCast)));
}

void CodeGenModule::AddVTableTypeMetadata(llvm::GlobalVariable *VTable,
                                          CharUnits Offset,
                                          const CXXRecordDecl *RD) {
  llvm::Metadata *MD =
      CreateMetadataIdentifierForType(QualType(RD->getTypeForDecl(), 0));
  VTable->addTypeMetadata(Offset.getQuantity(), MD);

  if (CodeGenOpts.SanitizeCfiCrossDso)
    if (auto CrossDsoTypeId = CreateCrossDsoCfiTypeId(MD))
      VTable->addTypeMetadata(Offset.getQuantity(),
                              llvm::ConstantAsMetadata::get(CrossDsoTypeId));

  if (NeedAllVtablesTypeId()) {
    llvm::Metadata *MD = llvm::MDString::get(getLLVMContext(), "all-vtables");
    VTable->addTypeMetadata(Offset.getQuantity(), MD);
  }
}

// Fills in the supplied string map with the set of target features for the
// passed in function.
void CodeGenModule::getFunctionFeatureMap(llvm::StringMap<bool> &FeatureMap,
                                          const FunctionDecl *FD) {
  StringRef TargetCPU = Target.getTargetOpts().CPU;
  if (const auto *TD = FD->getAttr<TargetAttr>()) {
    // If we have a TargetAttr build up the feature map based on that.
    TargetAttr::ParsedTargetAttr ParsedAttr = TD->parse();

    // Make a copy of the features as passed on the command line into the
    // beginning of the additional features from the function to override.
    ParsedAttr.first.insert(ParsedAttr.first.begin(),
                            Target.getTargetOpts().FeaturesAsWritten.begin(),
                            Target.getTargetOpts().FeaturesAsWritten.end());

    if (ParsedAttr.second != "")
      TargetCPU = ParsedAttr.second;

    // Now populate the feature map, first with the TargetCPU which is either
    // the default or a new one from the target attribute string. Then we'll use
    // the passed in features (FeaturesAsWritten) along with the new ones from
    // the attribute.
    Target.initFeatureMap(FeatureMap, getDiags(), TargetCPU, ParsedAttr.first);
  } else {
    Target.initFeatureMap(FeatureMap, getDiags(), TargetCPU,
                          Target.getTargetOpts().Features);
  }
}

llvm::SanitizerStatReport &CodeGenModule::getSanStats() {
  if (!SanStats)
    SanStats = llvm::make_unique<llvm::SanitizerStatReport>(&getModule());

  return *SanStats;
}
llvm::Value *
CodeGenModule::createOpenCLIntToSamplerConversion(const Expr *E,
                                                  CodeGenFunction &CGF) {
  llvm::Constant *C = EmitConstantExpr(E, E->getType(), &CGF);
  auto SamplerT = getOpenCLRuntime().getSamplerType();
  auto FTy = llvm::FunctionType::get(SamplerT, {C->getType()}, false);
  return CGF.Builder.CreateCall(CreateRuntimeFunction(FTy,
                                "__translate_sampler_initializer"),
                                {C});
}
