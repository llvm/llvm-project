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
#include "CGCUDARuntime.h"
#include "CGCXXABI.h"
#include "CGCall.h"
#include "CGDebugInfo.h"
#include "CGObjCRuntime.h"
#include "CGOpenCLRuntime.h"
#include "CGOpenMPRuntime.h"
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
#include "clang/Frontend/CodeGenOptions.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace CodeGen;

static const char AnnotationSection[] = "llvm.metadata";

static CGCXXABI *createCXXABI(CodeGenModule &CGM) {
  switch (CGM.getTarget().getCXXABI().getKind()) {
  case TargetCXXABI::GenericAArch64:
  case TargetCXXABI::GenericARM:
  case TargetCXXABI::iOS:
  case TargetCXXABI::iOS64:
  case TargetCXXABI::GenericItanium:
    return CreateItaniumCXXABI(CGM);
  case TargetCXXABI::Microsoft:
    return CreateMicrosoftCXXABI(CGM);
  }

  llvm_unreachable("invalid C++ ABI kind");
}

CodeGenModule::CodeGenModule(ASTContext &C, const CodeGenOptions &CGO,
                             llvm::Module &M, const llvm::DataLayout &TD,
                             DiagnosticsEngine &diags,
                             CoverageSourceInfo *CoverageInfo)
    : Context(C), LangOpts(C.getLangOpts()), CodeGenOpts(CGO), TheModule(M),
      Diags(diags), TheDataLayout(TD), Target(C.getTargetInfo()),
      ABI(createCXXABI(*this)), VMContext(M.getContext()), TBAA(nullptr),
      TheTargetCodeGenInfo(nullptr), Types(*this), VTables(*this),
      ObjCRuntime(nullptr), OpenCLRuntime(nullptr), OpenMPRuntime(nullptr),
      CUDARuntime(nullptr), DebugInfo(nullptr), ARCData(nullptr),
      NoObjCARCExceptionsMetadata(nullptr), RRData(nullptr), PGOReader(nullptr),
      CFConstantStringClassRef(nullptr), ConstantStringClassRef(nullptr),
      NSConstantStringType(nullptr), NSConcreteGlobalBlock(nullptr),
      NSConcreteStackBlock(nullptr), BlockObjectAssign(nullptr),
      BlockObjectDispose(nullptr), BlockDescriptorType(nullptr),
      GenericBlockLiteralType(nullptr), LifetimeStartFn(nullptr),
      LifetimeEndFn(nullptr), SanitizerMD(new SanitizerMetadata(*this)) {

  // Initialize the type cache.
  llvm::LLVMContext &LLVMContext = M.getContext();
  VoidTy = llvm::Type::getVoidTy(LLVMContext);
  Int8Ty = llvm::Type::getInt8Ty(LLVMContext);
  Int16Ty = llvm::Type::getInt16Ty(LLVMContext);
  Int32Ty = llvm::Type::getInt32Ty(LLVMContext);
  Int64Ty = llvm::Type::getInt64Ty(LLVMContext);
  FloatTy = llvm::Type::getFloatTy(LLVMContext);
  DoubleTy = llvm::Type::getDoubleTy(LLVMContext);
  PointerWidthInBits = C.getTargetInfo().getPointerWidth(0);
  PointerAlignInBytes =
  C.toCharUnitsFromBits(C.getTargetInfo().getPointerAlign(0)).getQuantity();
  IntTy = llvm::IntegerType::get(LLVMContext, C.getTargetInfo().getIntWidth());
  IntPtrTy = llvm::IntegerType::get(LLVMContext, PointerWidthInBits);
  Int8PtrTy = Int8Ty->getPointerTo(0);
  Int8PtrPtrTy = Int8PtrTy->getPointerTo(0);

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
    TBAA = new CodeGenTBAA(Context, VMContext, CodeGenOpts, getLangOpts(),
                           getCXXABI().getMangleContext());

  // If debug info or coverage generation is enabled, create the CGDebugInfo
  // object.
  if (CodeGenOpts.getDebugInfo() != CodeGenOptions::NoDebugInfo ||
      CodeGenOpts.EmitGcovArcs ||
      CodeGenOpts.EmitGcovNotes)
    DebugInfo = new CGDebugInfo(*this);

  Block.GlobalUniqueCount = 0;

  if (C.getLangOpts().ObjCAutoRefCount)
    ARCData = new ARCEntrypoints();
  RRData = new RREntrypoints();

  if (!CodeGenOpts.InstrProfileInput.empty()) {
    if (std::error_code EC = llvm::IndexedInstrProfReader::create(
            CodeGenOpts.InstrProfileInput, PGOReader)) {
      unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                              "Could not read profile: %0");
      getDiags().Report(DiagID) << EC.message();
    }
  }

  // If coverage mapping generation is enabled, create the
  // CoverageMappingModuleGen object.
  if (CodeGenOpts.CoverageMapping)
    CoverageMapping.reset(new CoverageMappingModuleGen(*this, *CoverageInfo));
}

CodeGenModule::~CodeGenModule() {
  delete ObjCRuntime;
  delete OpenCLRuntime;
  delete OpenMPRuntime;
  delete CUDARuntime;
  delete TheTargetCodeGenInfo;
  delete TBAA;
  delete DebugInfo;
  delete ARCData;
  delete RRData;
}

void CodeGenModule::createObjCRuntime() {
  // This is just isGNUFamily(), but we want to force implementors of
  // new ABIs to decide how best to do this.
  switch (LangOpts.ObjCRuntime.getKind()) {
  case ObjCRuntime::GNUstep:
  case ObjCRuntime::GCC:
  case ObjCRuntime::ObjFW:
    ObjCRuntime = CreateGNUObjCRuntime(*this);
    return;

  case ObjCRuntime::FragileMacOSX:
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
    ObjCRuntime = CreateMacObjCRuntime(*this);
    return;
  }
  llvm_unreachable("bad runtime kind");
}

void CodeGenModule::createOpenCLRuntime() {
  OpenCLRuntime = new CGOpenCLRuntime(*this);
}

void CodeGenModule::createOpenMPRuntime() {
  OpenMPRuntime = new CGOpenMPRuntime(*this);
}

void CodeGenModule::createCUDARuntime() {
  CUDARuntime = CreateNVCUDARuntime(*this);
}

void CodeGenModule::addReplacement(StringRef Name, llvm::Constant *C) {
  Replacements[Name] = C;
}

void CodeGenModule::applyReplacements() {
  for (ReplacementsTy::iterator I = Replacements.begin(),
                                E = Replacements.end();
       I != E; ++I) {
    StringRef MangledName = I->first();
    llvm::Constant *Replacement = I->second;
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
      OldF->getParent()->getFunctionList().insertAfter(OldF, NewF);
    }
    OldF->eraseFromParent();
  }
}

// This is only used in aliases that we created and we know they have a
// linear structure.
static const llvm::GlobalObject *getAliasedGlobal(const llvm::GlobalAlias &GA) {
  llvm::SmallPtrSet<const llvm::GlobalAlias*, 4> Visited;
  const llvm::Constant *C = &GA;
  for (;;) {
    C = C->stripPointerCasts();
    if (auto *GO = dyn_cast<llvm::GlobalObject>(C))
      return GO;
    // stripPointerCasts will not walk over weak aliases.
    auto *GA2 = dyn_cast<llvm::GlobalAlias>(C);
    if (!GA2)
      return nullptr;
    if (!Visited.insert(GA2).second)
      return nullptr;
    C = GA2->getAliasee();
  }
}

void CodeGenModule::checkAliases() {
  // Check if the constructed aliases are well formed. It is really unfortunate
  // that we have to do this in CodeGen, but we only construct mangled names
  // and aliases during codegen.
  bool Error = false;
  DiagnosticsEngine &Diags = getDiags();
  for (std::vector<GlobalDecl>::iterator I = Aliases.begin(),
         E = Aliases.end(); I != E; ++I) {
    const GlobalDecl &GD = *I;
    const auto *D = cast<ValueDecl>(GD.getDecl());
    const AliasAttr *AA = D->getAttr<AliasAttr>();
    StringRef MangledName = getMangledName(GD);
    llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
    auto *Alias = cast<llvm::GlobalAlias>(Entry);
    const llvm::GlobalValue *GV = getAliasedGlobal(*Alias);
    if (!GV) {
      Error = true;
      Diags.Report(AA->getLocation(), diag::err_cyclic_alias);
    } else if (GV->isDeclaration()) {
      Error = true;
      Diags.Report(AA->getLocation(), diag::err_alias_to_undefined);
    }

    llvm::Constant *Aliasee = Alias->getAliasee();
    llvm::GlobalValue *AliaseeGV;
    if (auto CE = dyn_cast<llvm::ConstantExpr>(Aliasee))
      AliaseeGV = cast<llvm::GlobalValue>(CE->getOperand(0));
    else
      AliaseeGV = cast<llvm::GlobalValue>(Aliasee);

    if (const SectionAttr *SA = D->getAttr<SectionAttr>()) {
      StringRef AliasSection = SA->getName();
      if (AliasSection != AliaseeGV->getSection())
        Diags.Report(SA->getLocation(), diag::warn_alias_with_section)
            << AliasSection;
    }

    // We have to handle alias to weak aliases in here. LLVM itself disallows
    // this since the object semantics would not match the IL one. For
    // compatibility with gcc we implement it by just pointing the alias
    // to its aliasee's aliasee. We also warn, since the user is probably
    // expecting the link to be weak.
    if (auto GA = dyn_cast<llvm::GlobalAlias>(AliaseeGV)) {
      if (GA->mayBeOverridden()) {
        Diags.Report(AA->getLocation(), diag::warn_alias_to_weak_alias)
            << GV->getName() << GA->getName();
        Aliasee = llvm::ConstantExpr::getPointerBitCastOrAddrSpaceCast(
            GA->getAliasee(), Alias->getType());
        Alias->setAliasee(Aliasee);
      }
    }
  }
  if (!Error)
    return;

  for (std::vector<GlobalDecl>::iterator I = Aliases.begin(),
         E = Aliases.end(); I != E; ++I) {
    const GlobalDecl &GD = *I;
    StringRef MangledName = getMangledName(GD);
    llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
    auto *Alias = cast<llvm::GlobalAlias>(Entry);
    Alias->replaceAllUsesWith(llvm::UndefValue::get(Alias->getType()));
    Alias->eraseFromParent();
  }
}

void CodeGenModule::clear() {
  DeferredDeclsToEmit.clear();
}

void InstrProfStats::reportDiagnostics(DiagnosticsEngine &Diags,
                                       StringRef MainFile) {
  if (!hasDiagnostics())
    return;
  if (VisitedInMainFile > 0 && VisitedInMainFile == MissingInMainFile) {
    if (MainFile.empty())
      MainFile = "<stdin>";
    Diags.Report(diag::warn_profile_data_unprofiled) << MainFile;
  } else
    Diags.Report(diag::warn_profile_data_out_of_date) << Visited << Missing
                                                      << Mismatched;
}

void CodeGenModule::Release() {
  EmitDeferred();
  applyReplacements();
  checkAliases();
  EmitCXXGlobalInitFunc();
  EmitCXXGlobalDtorFunc();
  EmitCXXThreadLocalInitFunc();
  if (ObjCRuntime)
    if (llvm::Function *ObjCInitFunction = ObjCRuntime->ModuleInitFunction())
      AddGlobalCtor(ObjCInitFunction);
  if (PGOReader && PGOStats.hasDiagnostics())
    PGOStats.reportDiagnostics(getDiags(), getCodeGenOpts().MainFileName);
  EmitCtorList(GlobalCtors, "llvm.global_ctors");
  EmitCtorList(GlobalDtors, "llvm.global_dtors");
  EmitGlobalAnnotations();
  EmitStaticExternCAliases();
  EmitDeferredUnusedCoverageMappings();
  if (CoverageMapping)
    CoverageMapping->emit();
  emitLLVMUsed();

  if (CodeGenOpts.Autolink &&
      (Context.getLangOpts().Modules || !LinkerOptionsMetadata.empty())) {
    EmitModuleLinkOptions();
  }
  if (CodeGenOpts.DwarfVersion)
    // We actually want the latest version when there are conflicts.
    // We can change from Warning to Latest if such mode is supported.
    getModule().addModuleFlag(llvm::Module::Warning, "Dwarf Version",
                              CodeGenOpts.DwarfVersion);
  if (DebugInfo)
    // We support a single version in the linked module. The LLVM
    // parser will drop debug info with a different version number
    // (and warn about it, too).
    getModule().addModuleFlag(llvm::Module::Warning, "Debug Info Version",
                              llvm::DEBUG_METADATA_VERSION);

  // We need to record the widths of enums and wchar_t, so that we can generate
  // the correct build attributes in the ARM backend.
  llvm::Triple::ArchType Arch = Context.getTargetInfo().getTriple().getArch();
  if (   Arch == llvm::Triple::arm
      || Arch == llvm::Triple::armeb
      || Arch == llvm::Triple::thumb
      || Arch == llvm::Triple::thumbeb) {
    // Width of wchar_t in bytes
    uint64_t WCharWidth =
        Context.getTypeSizeInChars(Context.getWideCharType()).getQuantity();
    getModule().addModuleFlag(llvm::Module::Error, "wchar_size", WCharWidth);

    // The minimum width of an enum in bytes
    uint64_t EnumWidth = Context.getLangOpts().ShortEnums ? 1 : 4;
    getModule().addModuleFlag(llvm::Module::Error, "min_enum_size", EnumWidth);
  }

  if (uint32_t PLevel = Context.getLangOpts().PICLevel) {
    llvm::PICLevel::Level PL = llvm::PICLevel::Default;
    switch (PLevel) {
    case 0: break;
    case 1: PL = llvm::PICLevel::Small; break;
    case 2: PL = llvm::PICLevel::Large; break;
    default: llvm_unreachable("Invalid PIC Level");
    }

    getModule().setPICLevel(PL);
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

void CodeGenModule::UpdateCompletedType(const TagDecl *TD) {
  // Make sure that this type is translated.
  Types.UpdateCompletedType(TD);
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

llvm::MDNode *CodeGenModule::getTBAAStructTypeInfo(QualType QTy) {
  if (!TBAA)
    return nullptr;
  return TBAA->getTBAAStructTypeInfo(QTy);
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
void CodeGenModule::DecorateInstruction(llvm::Instruction *Inst,
                                        llvm::MDNode *TBAAInfo,
                                        bool ConvertTypeToTag) {
  if (ConvertTypeToTag && TBAA)
    Inst->setMetadata(llvm::LLVMContext::MD_tbaa,
                      TBAA->getTBAAScalarTagInfo(TBAAInfo));
  else
    Inst->setMetadata(llvm::LLVMContext::MD_tbaa, TBAAInfo);
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
  StringRef &FoundStr = MangledDeclNames[GD.getCanonicalDecl()];
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
    Str = II->getName();
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

void CodeGenModule::EmitCtorList(const CtorList &Fns, const char *GlobalName) {
  // Ctor function type is void()*.
  llvm::FunctionType* CtorFTy = llvm::FunctionType::get(VoidTy, false);
  llvm::Type *CtorPFTy = llvm::PointerType::getUnqual(CtorFTy);

  // Get the type of a ctor entry, { i32, void ()*, i8* }.
  llvm::StructType *CtorStructTy = llvm::StructType::get(
      Int32Ty, llvm::PointerType::getUnqual(CtorFTy), VoidPtrTy, nullptr);

  // Construct the constructor and destructor arrays.
  SmallVector<llvm::Constant*, 8> Ctors;
  for (CtorList::const_iterator I = Fns.begin(), E = Fns.end(); I != E; ++I) {
    llvm::Constant *S[] = {
      llvm::ConstantInt::get(Int32Ty, I->Priority, false),
      llvm::ConstantExpr::getBitCast(I->Initializer, CtorPFTy),
      (I->AssociatedData
           ? llvm::ConstantExpr::getBitCast(I->AssociatedData, VoidPtrTy)
           : llvm::Constant::getNullValue(VoidPtrTy))
    };
    Ctors.push_back(llvm::ConstantStruct::get(CtorStructTy, S));
  }

  if (!Ctors.empty()) {
    llvm::ArrayType *AT = llvm::ArrayType::get(CtorStructTy, Ctors.size());
    new llvm::GlobalVariable(TheModule, AT, false,
                             llvm::GlobalValue::AppendingLinkage,
                             llvm::ConstantArray::get(AT, Ctors),
                             GlobalName);
  }
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

  return getLLVMLinkageForDeclarator(D, Linkage, /*isConstantVariable=*/false);
}

void CodeGenModule::setFunctionDefinitionAttributes(const FunctionDecl *D,
                                                    llvm::Function *F) {
  setNonAliasAttributes(D, F);
}

void CodeGenModule::SetLLVMFunctionAttributes(const Decl *D,
                                              const CGFunctionInfo &Info,
                                              llvm::Function *F) {
  unsigned CallingConv;
  AttributeListType AttributeList;
  ConstructAttributeList(Info, D, AttributeList, CallingConv, false);
  F->setAttributes(llvm::AttributeSet::get(getLLVMContext(), AttributeList));
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

  if (D->hasAttr<NakedAttr>()) {
    // Naked implies noinline: we should not be inlining such functions.
    B.addAttribute(llvm::Attribute::Naked);
    B.addAttribute(llvm::Attribute::NoInline);
  } else if (D->hasAttr<NoDuplicateAttr>()) {
    B.addAttribute(llvm::Attribute::NoDuplicate);
  } else if (D->hasAttr<NoInlineAttr>()) {
    B.addAttribute(llvm::Attribute::NoInline);
  } else if (D->hasAttr<AlwaysInlineAttr>() &&
             !F->getAttributes().hasAttribute(llvm::AttributeSet::FunctionIndex,
                                              llvm::Attribute::NoInline)) {
    // (noinline wins over always_inline, and we can't specify both in IR)
    B.addAttribute(llvm::Attribute::AlwaysInline);
  }

  if (D->hasAttr<ColdAttr>()) {
    if (!D->hasAttr<OptimizeNoneAttr>())
      B.addAttribute(llvm::Attribute::OptimizeForSize);
    B.addAttribute(llvm::Attribute::Cold);
  }

  if (D->hasAttr<MinSizeAttr>())
    B.addAttribute(llvm::Attribute::MinSize);

  if (LangOpts.getStackProtector() == LangOptions::SSPOn)
    B.addAttribute(llvm::Attribute::StackProtect);
  else if (LangOpts.getStackProtector() == LangOptions::SSPStrong)
    B.addAttribute(llvm::Attribute::StackProtectStrong);
  else if (LangOpts.getStackProtector() == LangOptions::SSPReq)
    B.addAttribute(llvm::Attribute::StackProtectReq);

  // Add sanitizer attributes if function is not blacklisted.
  if (!isInSanitizerBlacklist(F, D->getLocation())) {
    // When AddressSanitizer is enabled, set SanitizeAddress attribute
    // unless __attribute__((no_sanitize_address)) is used.
    if (LangOpts.Sanitize.has(SanitizerKind::Address) &&
        !D->hasAttr<NoSanitizeAddressAttr>())
      B.addAttribute(llvm::Attribute::SanitizeAddress);
    // Same for ThreadSanitizer and __attribute__((no_sanitize_thread))
    if (LangOpts.Sanitize.has(SanitizerKind::Thread) &&
        !D->hasAttr<NoSanitizeThreadAttr>())
      B.addAttribute(llvm::Attribute::SanitizeThread);
    // Same for MemorySanitizer and __attribute__((no_sanitize_memory))
    if (LangOpts.Sanitize.has(SanitizerKind::Memory) &&
        !D->hasAttr<NoSanitizeMemoryAttr>())
      B.addAttribute(llvm::Attribute::SanitizeMemory);
  }

  F->addAttributes(llvm::AttributeSet::FunctionIndex,
                   llvm::AttributeSet::get(
                       F->getContext(), llvm::AttributeSet::FunctionIndex, B));

  if (D->hasAttr<OptimizeNoneAttr>()) {
    // OptimizeNone implies noinline; we should not be inlining such functions.
    F->addFnAttr(llvm::Attribute::OptimizeNone);
    F->addFnAttr(llvm::Attribute::NoInline);

    // OptimizeNone wins over OptimizeForSize, MinSize, AlwaysInline.
    assert(!F->hasFnAttribute(llvm::Attribute::OptimizeForSize) &&
           "OptimizeNone and OptimizeForSize on same function!");
    assert(!F->hasFnAttribute(llvm::Attribute::MinSize) &&
           "OptimizeNone and MinSize on same function!");
    assert(!F->hasFnAttribute(llvm::Attribute::AlwaysInline) &&
           "OptimizeNone and AlwaysInline on same function!");

    // Attribute 'inlinehint' has no effect on 'optnone' functions.
    // Explicitly remove it from the set of function attributes.
    F->removeFnAttr(llvm::Attribute::InlineHint);
  }

  if (isa<CXXConstructorDecl>(D) || isa<CXXDestructorDecl>(D))
    F->setUnnamedAddr(true);
  else if (const auto *MD = dyn_cast<CXXMethodDecl>(D))
    if (MD->isVirtual())
      F->setUnnamedAddr(true);

  unsigned alignment = D->getMaxAlignment() / Context.getCharWidth();
  if (alignment)
    F->setAlignment(alignment);

  // C++ ABI requires 2-byte alignment for member functions.
  if (F->getAlignment() < 2 && isa<CXXMethodDecl>(D))
    F->setAlignment(2);
}

void CodeGenModule::SetCommonAttributes(const Decl *D,
                                        llvm::GlobalValue *GV) {
  if (const auto *ND = dyn_cast<NamedDecl>(D))
    setGlobalVisibility(GV, ND);
  else
    GV->setVisibility(llvm::GlobalValue::DefaultVisibility);

  if (D->hasAttr<UsedAttr>())
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

  if (const SectionAttr *SA = D->getAttr<SectionAttr>())
    GO->setSection(SA->getName());

  getTargetCodeGenInfo().SetTargetAttributes(D, GO, *this);
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
      GV->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
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

void CodeGenModule::SetFunctionAttributes(GlobalDecl GD, llvm::Function *F,
                                          bool IsIncompleteFunction,
                                          bool IsThunk) {
  if (unsigned IID = F->getIntrinsicID()) {
    // If this is an intrinsic function, set the function's attributes
    // to the intrinsic's attributes.
    F->setAttributes(llvm::Intrinsic::getAttributes(getLLVMContext(),
                                                    (llvm::Intrinsic::ID)IID));
    return;
  }

  const auto *FD = cast<FunctionDecl>(GD.getDecl());

  if (!IsIncompleteFunction)
    SetLLVMFunctionAttributes(FD, getTypes().arrangeGlobalDeclaration(GD), F);

  // Add the Returned attribute for "this", except for iOS 5 and earlier
  // where substantial code, including the libstdc++ dylib, was compiled with
  // GCC and does not actually return "this".
  if (!IsThunk && getCXXABI().HasThisReturn(GD) &&
      !(getTarget().getTriple().isiOS() &&
        getTarget().getTriple().isOSVersionLT(6))) {
    assert(!F->arg_empty() &&
           F->arg_begin()->getType()
             ->canLosslesslyBitCastTo(F->getReturnType()) &&
           "unexpected this return");
    F->addAttribute(1, llvm::Attribute::Returned);
  }

  // Only a few attributes are set on declarations; these may later be
  // overridden by a definition.

  setLinkageAndVisibilityForGV(F, FD);

  if (const auto *Dtor = dyn_cast_or_null<CXXDestructorDecl>(FD)) {
    if (getCXXABI().useThunkForDtorVariant(Dtor, GD.getDtorType())) {
      // Don't dllexport/import destructor thunks.
      F->setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);
    }
  }

  if (const SectionAttr *SA = FD->getAttr<SectionAttr>())
    F->setSection(SA->getName());

  // A replaceable global allocation function does not act like a builtin by
  // default, only if it is invoked by a new-expression or delete-expression.
  if (FD->isReplaceableGlobalAllocationFunction())
    F->addAttribute(llvm::AttributeSet::FunctionIndex,
                    llvm::Attribute::NoBuiltin);
}

void CodeGenModule::addUsedGlobal(llvm::GlobalValue *GV) {
  assert(!GV->isDeclaration() &&
         "Only globals with definition can force usage.");
  LLVMUsed.push_back(GV);
}

void CodeGenModule::addCompilerUsedGlobal(llvm::GlobalValue *GV) {
  assert(!GV->isDeclaration() &&
         "Only globals with definition can force usage.");
  LLVMCompilerUsed.push_back(GV);
}

static void emitUsed(CodeGenModule &CGM, StringRef Name,
                     std::vector<llvm::WeakVH> &List) {
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
                                    SmallVectorImpl<llvm::Metadata *> &Metadata,
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
  for (llvm::SetVector<clang::Module *>::iterator M = ImportedModules.begin(),
                                               MEnd = ImportedModules.end();
       M != MEnd; ++M) {
    if (Visited.insert(*M).second)
      Stack.push_back(*M);
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
  SmallVector<llvm::Metadata *, 16> MetadataArgs;
  Visited.clear();
  for (llvm::SetVector<clang::Module *>::iterator M = LinkModules.begin(),
                                               MEnd = LinkModules.end();
       M != MEnd; ++M) {
    if (Visited.insert(*M).second)
      addLinkOptionsPostorder(*this, *M, MetadataArgs, Visited);
  }
  std::reverse(MetadataArgs.begin(), MetadataArgs.end());
  LinkerOptionsMetadata.append(MetadataArgs.begin(), MetadataArgs.end());

  // Add the linker options metadata flag.
  getModule().addModuleFlag(llvm::Module::AppendUnique, "Linker Options",
                            llvm::MDNode::get(getLLVMContext(),
                                              LinkerOptionsMetadata));
}

void CodeGenModule::EmitDeferred() {
  // Emit code for any potentially referenced deferred decls.  Since a
  // previously unused static decl may become used during the generation of code
  // for a static function, iterate until no changes are made.

  while (true) {
    if (!DeferredVTables.empty()) {
      EmitDeferredVTables();

      // Emitting a v-table doesn't directly cause more v-tables to
      // become deferred, although it can cause functions to be
      // emitted that then need those v-tables.
      assert(DeferredVTables.empty());
    }

    // Stop if we're out of both deferred v-tables and deferred declarations.
    if (DeferredDeclsToEmit.empty()) break;

    DeferredGlobal &G = DeferredDeclsToEmit.back();
    GlobalDecl D = G.GD;
    llvm::GlobalValue *GV = G.GV;
    DeferredDeclsToEmit.pop_back();

    assert(!GV || GV == GetGlobalValue(getMangledName(D)));
    if (!GV)
      GV = GetGlobalValue(getMangledName(D));


    // Check to see if we've already emitted this.  This is necessary
    // for a couple of reasons: first, decls can end up in the
    // deferred-decls queue multiple times, and second, decls can end
    // up with definitions in unusual ways (e.g. by an extern inline
    // function acquiring a strong function redefinition).  Just
    // ignore these cases.
    if (GV && !GV->isDeclaration())
      continue;

    // Otherwise, emit the definition and move on to the next one.
    EmitGlobalDefinition(D, GV);
  }
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
  gv->setUnnamedAddr(true);
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
  if (!Loc.isInvalid())
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
  // For now globals can be blacklisted only in ASan.
  if (!LangOpts.Sanitize.has(SanitizerKind::Address))
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

  return true;
}

llvm::Constant *CodeGenModule::GetAddrOfUuidDescriptor(
    const CXXUuidofExpr* E) {
  // Sema has verified that IIDSource has a __declspec(uuid()), and that its
  // well-formed.
  StringRef Uuid = E->getUuidAsStringRef(Context);
  std::string Name = "_GUID_" + Uuid.lower();
  std::replace(Name.begin(), Name.end(), '-', '_');

  // Look for an existing global.
  if (llvm::GlobalVariable *GV = getModule().getNamedGlobal(Name))
    return GV;

  llvm::Constant *Init = EmitUuidofInitializer(Uuid);
  assert(Init && "failed to initialize as constant");

  auto *GV = new llvm::GlobalVariable(
      getModule(), Init->getType(),
      /*isConstant=*/true, llvm::GlobalValue::LinkOnceODRLinkage, Init, Name);
  return GV;
}

llvm::Constant *CodeGenModule::GetWeakRefReference(const ValueDecl *VD) {
  const AliasAttr *AA = VD->getAttr<AliasAttr>();
  assert(AA && "No alias?");

  llvm::Type *DeclTy = getTypes().ConvertTypeForMem(VD->getType());

  // See if there is already something with the target's name in the module.
  llvm::GlobalValue *Entry = GetGlobalValue(AA->getAliasee());
  if (Entry) {
    unsigned AS = getContext().getTargetAddressSpace(VD->getType());
    return llvm::ConstantExpr::getBitCast(Entry, DeclTy->getPointerTo(AS));
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

  return Aliasee;
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

  // If this is CUDA, be selective about which declarations we emit.
  if (LangOpts.CUDA) {
    if (CodeGenOpts.CUDAIsDevice) {
      if (!Global->hasAttr<CUDADeviceAttr>() &&
          !Global->hasAttr<CUDAGlobalAttr>() &&
          !Global->hasAttr<CUDAConstantAttr>() &&
          !Global->hasAttr<CUDASharedAttr>())
        return;
    } else {
      if (!Global->hasAttr<CUDAHostAttr>() && (
            Global->hasAttr<CUDADeviceAttr>() ||
            Global->hasAttr<CUDAConstantAttr>() ||
            Global->hasAttr<CUDASharedAttr>()))
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

    if (VD->isThisDeclarationADefinition() != VarDecl::Definition &&
        !Context.isMSStaticDataMemberInlineDefinition(VD))
      return;
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
  if (llvm::GlobalValue *GV = GetGlobalValue(MangledName)) {
    // The value has already been used and should therefore be emitted.
    addDeferredDeclToEmit(GV, GD);
  } else if (MustBeEmitted(Global)) {
    // The value must be emitted, but cannot be emitted eagerly.
    assert(!MayBeEmittedEagerly(Global));
    addDeferredDeclToEmit(/*GV=*/nullptr, GD);
  } else {
    // Otherwise, remember that we saw a deferred decl with this name.  The
    // first use of the mangled name will cause it to move into
    // DeferredDeclsToEmit.
    DeferredDecls[MangledName] = GD;
  }
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
      if (!BuiltinID)
        return true;
      StringRef BuiltinName = BI.GetName(BuiltinID);
      if (BuiltinName.startswith("__builtin_") &&
          Name == BuiltinName.slice(strlen("__builtin_"), StringRef::npos)) {
        Result = true;
        return false;
      }
      return true;
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

bool
CodeGenModule::shouldEmitFunction(GlobalDecl GD) {
  if (getFunctionLinkage(GD) != llvm::Function::AvailableExternallyLinkage)
    return true;
  const auto *F = cast<FunctionDecl>(GD.getDecl());
  if (CodeGenOpts.OptimizationLevel == 0 && !F->hasAttr<AlwaysInlineAttr>())
    return false;
  // PR9614. Avoid cases where the source code is lying to us. An available
  // externally function should have an equivalent function somewhere else,
  // but a function that calls itself is clearly not equivalent to the real
  // implementation.
  // This happens in glibc's btowc and in some configure checks.
  return !isTriviallyRecursive(F);
}

/// If the type for the method's class was generated by
/// CGDebugInfo::createContextChain(), the cache contains only a
/// limited DIType without any declarations. Since EmitFunctionStart()
/// needs to find the canonical declaration for each method, we need
/// to construct the complete type prior to emitting the method.
void CodeGenModule::CompleteDIClassType(const CXXMethodDecl* D) {
  if (!D->isInstance())
    return;

  if (CGDebugInfo *DI = getModuleDebugInfo())
    if (getCodeGenOpts().getDebugInfo() >= CodeGenOptions::LimitedDebugInfo) {
      const auto *ThisPtr = cast<PointerType>(D->getThisType(getContext()));
      DI->getOrCreateRecordType(ThisPtr->getPointeeType(), D->getLocation());
    }
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
      CompleteDIClassType(Method);
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
    return EmitGlobalVarDefinition(VD);
  
  llvm_unreachable("Invalid argument to EmitGlobalDefinition()");
}

/// GetOrCreateLLVMFunction - If the specified mangled name is not in the
/// module, create and return an llvm Function with the specified type. If there
/// is something in the module with the specified name, return it potentially
/// bitcasted to the right type.
///
/// If D is non-null, it specifies a decl that correspond to this.  This is used
/// to set the attributes on the function when it is first created.
llvm::Constant *
CodeGenModule::GetOrCreateLLVMFunction(StringRef MangledName,
                                       llvm::Type *Ty,
                                       GlobalDecl GD, bool ForVTable,
                                       bool DontDefer, bool IsThunk,
                                       llvm::AttributeSet ExtraAttrs) {
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

    if (Entry->getType()->getElementType() == Ty)
      return Entry;

    // Make sure the result is of the correct type.
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
  
  llvm::Function *F = llvm::Function::Create(FTy,
                                             llvm::Function::ExternalLinkage,
                                             MangledName, &getModule());
  assert(F->getName() == MangledName && "name was uniqued!");
  if (D)
    SetFunctionAttributes(GD, F, IsIncompleteFunction, IsThunk);
  if (ExtraAttrs.hasAttributes(llvm::AttributeSet::FunctionIndex)) {
    llvm::AttrBuilder B(ExtraAttrs, llvm::AttributeSet::FunctionIndex);
    F->addAttributes(llvm::AttributeSet::FunctionIndex,
                     llvm::AttributeSet::get(VMContext,
                                             llvm::AttributeSet::FunctionIndex,
                                             B));
  }

  if (!DontDefer) {
    // All MSVC dtors other than the base dtor are linkonce_odr and delegate to
    // each other bottoming out with the base dtor.  Therefore we emit non-base
    // dtors on usage, even if there is no dtor definition in the TU.
    if (D && isa<CXXDestructorDecl>(D) &&
        getCXXABI().useThunkForDtorVariant(cast<CXXDestructorDecl>(D),
                                           GD.getDtorType()))
      addDeferredDeclToEmit(F, GD);

    // This is the first use or definition of a mangled name.  If there is a
    // deferred decl with this name, remember that we need to emit it at the end
    // of the file.
    auto DDI = DeferredDecls.find(MangledName);
    if (DDI != DeferredDecls.end()) {
      // Move the potentially referenced deferred decl to the
      // DeferredDeclsToEmit list, and remove it from DeferredDecls (since we
      // don't need it anymore).
      addDeferredDeclToEmit(F, DDI->second);
      DeferredDecls.erase(DDI);

      // Otherwise, if this is a sized deallocation function, emit a weak
      // definition
      // for it at the end of the translation unit.
    } else if (D && cast<FunctionDecl>(D)
                        ->getCorrespondingUnsizedGlobalDeallocationFunction()) {
      addDeferredDeclToEmit(F, GD);

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
            addDeferredDeclToEmit(F, GD.getWithDecl(FD));
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
                                                 bool DontDefer) {
  // If there was no specific requested type, just convert it now.
  if (!Ty)
    Ty = getTypes().ConvertType(cast<ValueDecl>(GD.getDecl())->getType());
  
  StringRef MangledName = getMangledName(GD);
  return GetOrCreateLLVMFunction(MangledName, Ty, GD, ForVTable, DontDefer);
}

/// CreateRuntimeFunction - Create a new runtime function with the specified
/// type and name.
llvm::Constant *
CodeGenModule::CreateRuntimeFunction(llvm::FunctionType *FTy,
                                     StringRef Name,
                                     llvm::AttributeSet ExtraAttrs) {
  llvm::Constant *C =
      GetOrCreateLLVMFunction(Name, FTy, GlobalDecl(), /*ForVTable=*/false,
                              /*DontDefer=*/false, /*IsThunk=*/false, ExtraAttrs);
  if (auto *F = dyn_cast<llvm::Function>(C))
    if (F->empty())
      F->setCallingConv(getRuntimeCC());
  return C;
}

/// CreateBuiltinFunction - Create a new builtin function with the specified
/// type and name.
llvm::Constant *
CodeGenModule::CreateBuiltinFunction(llvm::FunctionType *FTy,
                                     StringRef Name,
                                     llvm::AttributeSet ExtraAttrs) {
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
llvm::Constant *
CodeGenModule::GetOrCreateLLVMGlobal(StringRef MangledName,
                                     llvm::PointerType *Ty,
                                     const VarDecl *D) {
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

    // Make sure the result is of the correct type.
    if (Entry->getType()->getAddressSpace() != Ty->getAddressSpace())
      return llvm::ConstantExpr::getAddrSpaceCast(Entry, Ty);

    return llvm::ConstantExpr::getBitCast(Entry, Ty);
  }

  unsigned AddrSpace = GetGlobalVarAddressSpace(D, Ty->getAddressSpace());
  auto *GV = new llvm::GlobalVariable(
      getModule(), Ty->getElementType(), false,
      llvm::GlobalValue::ExternalLinkage, nullptr, MangledName, nullptr,
      llvm::GlobalVariable::NotThreadLocal, AddrSpace);

  // This is the first use or definition of a mangled name.  If there is a
  // deferred decl with this name, remember that we need to emit it at the end
  // of the file.
  auto DDI = DeferredDecls.find(MangledName);
  if (DDI != DeferredDecls.end()) {
    // Move the potentially referenced deferred decl to the DeferredDeclsToEmit
    // list, and remove it from DeferredDecls (since we don't need it anymore).
    addDeferredDeclToEmit(GV, DDI->second);
    DeferredDecls.erase(DDI);
  }

  // Handle things which are present even on external declarations.
  if (D) {
    // FIXME: This code is overly simple and should be merged with other global
    // handling.
    GV->setConstant(isTypeConstant(D->getType(), false));

    setLinkageAndVisibilityForGV(GV, D);

    if (D->getTLSKind()) {
      if (D->getTLSKind() == VarDecl::TLS_Dynamic)
        CXXThreadLocals.push_back(std::make_pair(D, GV));
      setTLSMode(GV, *D);
    }

    // If required by the ABI, treat declarations of static data members with
    // inline initializers as definitions.
    if (getContext().isMSStaticDataMemberInlineDefinition(D)) {
      EmitGlobalVarDefinition(D);
    }

    // Handle XCore specific ABI requirements.
    if (getTarget().getTriple().getArch() == llvm::Triple::xcore &&
        D->getLanguageLinkage() == CLanguageLinkage &&
        D->getType().isConstant(Context) &&
        isExternallyVisible(D->getLinkageAndVisibility().getLinkage()))
      GV->setSection(".cp.rodata");
  }

  if (AddrSpace != Ty->getAddressSpace())
    return llvm::ConstantExpr::getAddrSpaceCast(GV, Ty);

  return GV;
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
  
  return GV;
}

/// GetAddrOfGlobalVar - Return the llvm::Constant for the address of the
/// given global variable.  If Ty is non-null and if the global doesn't exist,
/// then it will be created with the specified type instead of whatever the
/// normal requested type would be.
llvm::Constant *CodeGenModule::GetAddrOfGlobalVar(const VarDecl *D,
                                                  llvm::Type *Ty) {
  assert(D->hasGlobalStorage() && "Not a global variable");
  QualType ASTTy = D->getType();
  if (!Ty)
    Ty = getTypes().ConvertTypeForMem(ASTTy);

  llvm::PointerType *PTy =
    llvm::PointerType::get(Ty, getContext().getTargetAddressSpace(ASTTy));

  StringRef MangledName = getMangledName(D);
  return GetOrCreateLLVMGlobal(MangledName, PTy, D);
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

  if (!MustBeEmitted(D)) {
    // If we have not seen a reference to this variable yet, place it
    // into the deferred declarations table to be emitted if needed
    // later.
    StringRef MangledName = getMangledName(D);
    if (!GetGlobalValue(MangledName)) {
      DeferredDecls[MangledName] = D;
      return;
    }
  }

  // The tentative definition is the only definition.
  EmitGlobalVarDefinition(D);
}

CharUnits CodeGenModule::GetTargetTypeStoreSize(llvm::Type *Ty) const {
    return Context.toCharUnitsFromBits(
      TheDataLayout.getTypeStoreSizeInBits(Ty));
}

unsigned CodeGenModule::GetGlobalVarAddressSpace(const VarDecl *D,
                                                 unsigned AddrSpace) {
  if (LangOpts.CUDA && CodeGenOpts.CUDAIsDevice) {
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

void CodeGenModule::EmitGlobalVarDefinition(const VarDecl *D) {
  llvm::Constant *Init = nullptr;
  QualType ASTTy = D->getType();
  CXXRecordDecl *RD = ASTTy->getBaseElementTypeUnsafe()->getAsCXXRecordDecl();
  bool NeedsGlobalCtor = false;
  bool NeedsGlobalDtor = RD && !RD->hasTrivialDestructor();

  const VarDecl *InitDecl;
  const Expr *InitExpr = D->getAnyInitializer(InitDecl);

  if (!InitExpr) {
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
  llvm::Constant *Entry = GetAddrOfGlobalVar(D, InitType);

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
    GV = cast<llvm::GlobalVariable>(GetAddrOfGlobalVar(D, InitType));

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

  // Set the llvm linkage type as appropriate.
  llvm::GlobalValue::LinkageTypes Linkage =
      getLLVMLinkageVarDefinition(D, GV->isConstant());

  // On Darwin, the backing variable for a C++11 thread_local variable always
  // has internal linkage; all accesses should just be calls to the
  // Itanium-specified entry point, which has the normal linkage of the
  // variable.
  if (!D->isStaticLocal() && D->getTLSKind() == VarDecl::TLS_Dynamic &&
      Context.getTargetInfo().getTriple().isMacOSX())
    Linkage = llvm::GlobalValue::InternalLinkage;

  GV->setLinkage(Linkage);
  if (D->hasAttr<DLLImportAttr>())
    GV->setDLLStorageClass(llvm::GlobalVariable::DLLImportStorageClass);
  else if (D->hasAttr<DLLExportAttr>())
    GV->setDLLStorageClass(llvm::GlobalVariable::DLLExportStorageClass);
  else
    GV->setDLLStorageClass(llvm::GlobalVariable::DefaultStorageClass);

  if (Linkage == llvm::GlobalVariable::CommonLinkage)
    // common vars aren't constant even if declared const.
    GV->setConstant(false);

  setNonAliasAttributes(D, GV);

  if (D->getTLSKind() && !GV->isThreadLocal()) {
    if (D->getTLSKind() == VarDecl::TLS_Dynamic)
      CXXThreadLocals.push_back(std::make_pair(D, GV));
    setTLSMode(GV, *D);
  }

  // Emit the initializer function if necessary.
  if (NeedsGlobalCtor || NeedsGlobalDtor)
    EmitCXXGlobalVarDeclInitFunc(D, GV, NeedsGlobalCtor);

  SanitizerMD->reportGlobalToASan(GV, *D, NeedsGlobalCtor);

  // Emit global variable debug information.
  if (CGDebugInfo *DI = getModuleDebugInfo())
    if (getCodeGenOpts().getDebugInfo() >= CodeGenOptions::LimitedDebugInfo)
      DI->EmitGlobalVariable(GV, D);
}

static bool isVarDeclStrongDefinition(const ASTContext &Context,
                                      const VarDecl *D, bool NoCommon) {
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

  // Thread local vars aren't considered common linkage.
  if (D->getTLSKind())
    return true;

  // Tentative definitions marked with WeakImportAttr are true definitions.
  if (D->hasAttr<WeakImportAttr>())
    return true;

  // Declarations with a required alignment do not have common linakge in MSVC
  // mode.
  if (Context.getLangOpts().MSVCCompat &&
      (Context.isAlignmentRequired(D->getType()) || D->hasAttr<AlignedAttr>()))
    return true;

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
    return llvm::Function::AvailableExternallyLinkage;

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
  if (Linkage == GVA_StrongODR)
    return !Context.getLangOpts().AppleKext ? llvm::Function::WeakODRLinkage
                                            : llvm::Function::ExternalLinkage;

  // C++ doesn't have tentative definitions and thus cannot have common
  // linkage.
  if (!getLangOpts().CPlusPlus && isa<VarDecl>(D) &&
      !isVarDeclStrongDefinition(Context, cast<VarDecl>(D),
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
    SmallVector<llvm::AttributeSet, 8> newAttrs;
    llvm::AttributeSet oldAttrs = callSite.getAttributes();

    // Collect any return attributes from the call.
    if (oldAttrs.hasAttributes(llvm::AttributeSet::ReturnIndex))
      newAttrs.push_back(
        llvm::AttributeSet::get(newFn->getContext(),
                                oldAttrs.getRetAttributes()));

    // If the function was passed too few arguments, don't transform.
    unsigned newNumArgs = newFn->arg_size();
    if (callSite.arg_size() < newNumArgs) continue;

    // If extra arguments were passed, we silently drop them.
    // If any of the types mismatch, we don't transform.
    unsigned argNo = 0;
    bool dontTransform = false;
    for (llvm::Function::arg_iterator ai = newFn->arg_begin(),
           ae = newFn->arg_end(); ai != ae; ++ai, ++argNo) {
      if (callSite.getArgument(argNo)->getType() != ai->getType()) {
        dontTransform = true;
        break;
      }

      // Add any parameter attributes.
      if (oldAttrs.hasAttributes(argNo + 1))
        newAttrs.
          push_back(llvm::
                    AttributeSet::get(newFn->getContext(),
                                      oldAttrs.getParamAttributes(argNo + 1)));
    }
    if (dontTransform)
      continue;

    if (oldAttrs.hasAttributes(llvm::AttributeSet::FunctionIndex))
      newAttrs.push_back(llvm::AttributeSet::get(newFn->getContext(),
                                                 oldAttrs.getFnAttributes()));

    // Okay, we can transform this.  Create the new call instruction and copy
    // over the required information.
    newArgs.append(callSite.arg_begin(), callSite.arg_begin() + argNo);

    llvm::CallSite newCall;
    if (callSite.isCall()) {
      newCall = llvm::CallInst::Create(newFn, newArgs, "",
                                       callSite.getInstruction());
    } else {
      auto *oldInvoke = cast<llvm::InvokeInst>(callSite.getInstruction());
      newCall = llvm::InvokeInst::Create(newFn,
                                         oldInvoke->getNormalDest(),
                                         oldInvoke->getUnwindDest(),
                                         newArgs, "",
                                         callSite.getInstruction());
    }
    newArgs.clear(); // for the next iteration

    if (!newCall->getType()->isVoidTy())
      newCall->takeName(callSite.getInstruction());
    newCall.setAttributes(
                     llvm::AttributeSet::get(newFn->getContext(), newAttrs));
    newCall.setCallingConv(callSite.getCallingConv());

    // Finally, remove the old call, replacing any uses with the new one.
    if (!callSite->use_empty())
      callSite->replaceAllUsesWith(newCall.getInstruction());

    // Copy debug location attached to CI.
    if (!callSite->getDebugLoc().isUnknown())
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
  if (!GV) {
    llvm::Constant *C =
        GetAddrOfFunction(GD, Ty, /*ForVTable=*/false, /*DontDefer*/ true);

    // Strip off a bitcast if we got one back.
    if (auto *CE = dyn_cast<llvm::ConstantExpr>(C)) {
      assert(CE->getOpcode() == llvm::Instruction::BitCast);
      GV = cast<llvm::GlobalValue>(CE->getOperand(0));
    } else {
      GV = cast<llvm::GlobalValue>(C);
    }
  }

  if (!GV->isDeclaration()) {
    getDiags().Report(D->getLocation(), diag::err_duplicate_mangled_name);
    GlobalDecl OldGD = Manglings.lookup(GV->getName());
    if (auto *Prev = OldGD.getDecl())
      getDiags().Report(Prev->getLocation(), diag::note_previous_definition);
    return;
  }

  if (GV->getType()->getElementType() != Ty) {
    // If the types mismatch then we have to rewrite the definition.
    assert(GV->isDeclaration() && "Shouldn't replace non-declaration");

    // F is the Function* for the one with the wrong type, we must make a new
    // Function* and update everything that used F (a declaration) with the new
    // Function* (which will be a definition).
    //
    // This happens if there is a prototype for a function
    // (e.g. "int f()") and then a definition of a different type
    // (e.g. "int f(int x)").  Move the old function aside so that it
    // doesn't interfere with GetAddrOfFunction.
    GV->setName(StringRef());
    auto *NewFn = cast<llvm::Function>(GetAddrOfFunction(GD, Ty));

    // This might be an implementation of a function without a
    // prototype, in which case, try to do special replacement of
    // calls which match the new prototype.  The really key thing here
    // is that we also potentially drop arguments from the call site
    // so as to make a direct call, which makes the inliner happier
    // and suppresses a number of optimizer warnings (!) about
    // dropping arguments.
    if (!GV->use_empty()) {
      ReplaceUsesOfNonProtoTypeWithRealFunction(GV, NewFn);
      GV->removeDeadConstantUsers();
    }

    // Replace uses of F with the Function we will endow with a body.
    if (!GV->use_empty()) {
      llvm::Constant *NewPtrForOldDecl =
          llvm::ConstantExpr::getBitCast(NewFn, GV->getType());
      GV->replaceAllUsesWith(NewPtrForOldDecl);
    }

    // Ok, delete the old function now, which is dead.
    GV->eraseFromParent();

    GV = NewFn;
  }

  // We need to set linkage and visibility on the function before
  // generating code for it because various parts of IR generation
  // want to propagate this information down (e.g. to local static
  // declarations).
  auto *Fn = cast<llvm::Function>(GV);
  setFunctionLinkage(GD, Fn);
  if (D->hasAttr<DLLImportAttr>())
    GV->setDLLStorageClass(llvm::GlobalVariable::DLLImportStorageClass);
  else if (D->hasAttr<DLLExportAttr>())
    GV->setDLLStorageClass(llvm::GlobalVariable::DLLExportStorageClass);
  else
    GV->setDLLStorageClass(llvm::GlobalVariable::DefaultStorageClass);

  // FIXME: this is redundant with part of setFunctionDefinitionAttributes
  setGlobalVisibility(Fn, D);

  MaybeHandleStaticInExternC(D, Fn);

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
      cast<llvm::PointerType>(Aliasee->getType())->getElementType(), 0,
      llvm::Function::ExternalLinkage, "", Aliasee, &getModule());

  if (Entry) {
    if (GA->getAliasee() == Entry) {
      Diags.Report(AA->getLocation(), diag::err_cyclic_alias);
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

llvm::Function *CodeGenModule::getIntrinsic(unsigned IID,
                                            ArrayRef<llvm::Type*> Tys) {
  return llvm::Intrinsic::getDeclaration(&getModule(), (llvm::Intrinsic::ID)IID,
                                         Tys);
}

static llvm::StringMapEntry<llvm::Constant*> &
GetConstantCFStringEntry(llvm::StringMap<llvm::Constant*> &Map,
                         const StringLiteral *Literal,
                         bool TargetIsLSB,
                         bool &IsUTF16,
                         unsigned &StringLength) {
  StringRef String = Literal->getString();
  unsigned NumBytes = String.size();

  // Check for simple case.
  if (!Literal->containsNonAsciiOrNull()) {
    StringLength = NumBytes;
    return *Map.insert(std::make_pair(String, nullptr)).first;
  }

  // Otherwise, convert the UTF8 literals into a string of shorts.
  IsUTF16 = true;

  SmallVector<UTF16, 128> ToBuf(NumBytes + 1); // +1 for ending nulls.
  const UTF8 *FromPtr = (const UTF8 *)String.data();
  UTF16 *ToPtr = &ToBuf[0];

  (void)ConvertUTF8toUTF16(&FromPtr, FromPtr + NumBytes,
                           &ToPtr, ToPtr + NumBytes,
                           strictConversion);

  // ConvertUTF8toUTF16 returns the length in ToPtr.
  StringLength = ToPtr - &ToBuf[0];

  // Add an explicit null.
  *ToPtr = 0;
  return *Map.insert(std::make_pair(
                         StringRef(reinterpret_cast<const char *>(ToBuf.data()),
                                   (StringLength + 1) * 2),
                         nullptr)).first;
}

static llvm::StringMapEntry<llvm::Constant*> &
GetConstantStringEntry(llvm::StringMap<llvm::Constant*> &Map,
                       const StringLiteral *Literal,
                       unsigned &StringLength) {
  StringRef String = Literal->getString();
  StringLength = String.size();
  return *Map.insert(std::make_pair(String, nullptr)).first;
}

llvm::Constant *
CodeGenModule::GetAddrOfConstantCFString(const StringLiteral *Literal) {
  unsigned StringLength = 0;
  bool isUTF16 = false;
  llvm::StringMapEntry<llvm::Constant*> &Entry =
    GetConstantCFStringEntry(CFConstantStringMap, Literal,
                             getDataLayout().isLittleEndian(),
                             isUTF16, StringLength);

  if (auto *C = Entry.second)
    return C;

  llvm::Constant *Zero = llvm::Constant::getNullValue(Int32Ty);
  llvm::Constant *Zeros[] = { Zero, Zero };
  llvm::Value *V;
  
  // If we don't already have it, get __CFConstantStringClassReference.
  if (!CFConstantStringClassRef) {
    llvm::Type *Ty = getTypes().ConvertType(getContext().IntTy);
    Ty = llvm::ArrayType::get(Ty, 0);
    llvm::Constant *GV = CreateRuntimeVariable(Ty,
                                           "__CFConstantStringClassReference");
    // Decay array -> ptr
    V = llvm::ConstantExpr::getGetElementPtr(GV, Zeros);
    CFConstantStringClassRef = V;
  }
  else
    V = CFConstantStringClassRef;

  QualType CFTy = getContext().getCFConstantStringType();

  auto *STy = cast<llvm::StructType>(getTypes().ConvertType(CFTy));

  llvm::Constant *Fields[4];

  // Class pointer.
  Fields[0] = cast<llvm::ConstantExpr>(V);

  // Flags.
  llvm::Type *Ty = getTypes().ConvertType(getContext().UnsignedIntTy);
  Fields[1] = isUTF16 ? llvm::ConstantInt::get(Ty, 0x07d0) :
    llvm::ConstantInt::get(Ty, 0x07C8);

  // String pointer.
  llvm::Constant *C = nullptr;
  if (isUTF16) {
    ArrayRef<uint16_t> Arr = llvm::makeArrayRef<uint16_t>(
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
  GV->setUnnamedAddr(true);
  // Don't enforce the target's minimum global alignment, since the only use
  // of the string is via this class initializer.
  // FIXME: We set the section explicitly to avoid a bug in ld64 224.1. Without
  // it LLVM can merge the string with a non unnamed_addr one during LTO. Doing
  // that changes the section it ends in, which surprises ld64.
  if (isUTF16) {
    CharUnits Align = getContext().getTypeAlignInChars(getContext().ShortTy);
    GV->setAlignment(Align.getQuantity());
    GV->setSection("__TEXT,__ustring");
  } else {
    CharUnits Align = getContext().getTypeAlignInChars(getContext().CharTy);
    GV->setAlignment(Align.getQuantity());
    GV->setSection("__TEXT,__cstring,cstring_literals");
  }

  // String.
  Fields[2] = llvm::ConstantExpr::getGetElementPtr(GV, Zeros);

  if (isUTF16)
    // Cast the UTF16 string to the correct type.
    Fields[2] = llvm::ConstantExpr::getBitCast(Fields[2], Int8PtrTy);

  // String length.
  Ty = getTypes().ConvertType(getContext().LongTy);
  Fields[3] = llvm::ConstantInt::get(Ty, StringLength);

  // The struct.
  C = llvm::ConstantStruct::get(STy, Fields);
  GV = new llvm::GlobalVariable(getModule(), C->getType(), true,
                                llvm::GlobalVariable::PrivateLinkage, C,
                                "_unnamed_cfstring_");
  GV->setSection("__DATA,__cfstring");
  Entry.second = GV;

  return GV;
}

llvm::Constant *
CodeGenModule::GetAddrOfConstantString(const StringLiteral *Literal) {
  unsigned StringLength = 0;
  llvm::StringMapEntry<llvm::Constant*> &Entry =
    GetConstantStringEntry(CFConstantStringMap, Literal, StringLength);

  if (auto *C = Entry.second)
    return C;
  
  llvm::Constant *Zero = llvm::Constant::getNullValue(Int32Ty);
  llvm::Constant *Zeros[] = { Zero, Zero };
  llvm::Value *V;
  // If we don't already have it, get _NSConstantStringClassReference.
  if (!ConstantStringClassRef) {
    std::string StringClass(getLangOpts().ObjCConstantStringClass);
    llvm::Type *Ty = getTypes().ConvertType(getContext().IntTy);
    llvm::Constant *GV;
    if (LangOpts.ObjCRuntime.isNonFragile()) {
      std::string str = 
        StringClass.empty() ? "OBJC_CLASS_$_NSConstantString" 
                            : "OBJC_CLASS_$_" + StringClass;
      GV = getObjCRuntime().GetClassGlobal(str);
      // Make sure the result is of the correct type.
      llvm::Type *PTy = llvm::PointerType::getUnqual(Ty);
      V = llvm::ConstantExpr::getBitCast(GV, PTy);
      ConstantStringClassRef = V;
    } else {
      std::string str =
        StringClass.empty() ? "_NSConstantStringClassReference"
                            : "_" + StringClass + "ClassReference";
      llvm::Type *PTy = llvm::ArrayType::get(Ty, 0);
      GV = CreateRuntimeVariable(PTy, str);
      // Decay array -> ptr
      V = llvm::ConstantExpr::getGetElementPtr(GV, Zeros);
      ConstantStringClassRef = V;
    }
  }
  else
    V = ConstantStringClassRef;

  if (!NSConstantStringType) {
    // Construct the type for a constant NSString.
    RecordDecl *D = Context.buildImplicitRecord("__builtin_NSString");
    D->startDefinition();
      
    QualType FieldTypes[3];
    
    // const int *isa;
    FieldTypes[0] = Context.getPointerType(Context.IntTy.withConst());
    // const char *str;
    FieldTypes[1] = Context.getPointerType(Context.CharTy.withConst());
    // unsigned int length;
    FieldTypes[2] = Context.UnsignedIntTy;
    
    // Create fields
    for (unsigned i = 0; i < 3; ++i) {
      FieldDecl *Field = FieldDecl::Create(Context, D,
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
    QualType NSTy = Context.getTagDeclType(D);
    NSConstantStringType = cast<llvm::StructType>(getTypes().ConvertType(NSTy));
  }
  
  llvm::Constant *Fields[3];
  
  // Class pointer.
  Fields[0] = cast<llvm::ConstantExpr>(V);
  
  // String pointer.
  llvm::Constant *C =
      llvm::ConstantDataArray::getString(VMContext, Entry.first());

  llvm::GlobalValue::LinkageTypes Linkage;
  bool isConstant;
  Linkage = llvm::GlobalValue::PrivateLinkage;
  isConstant = !LangOpts.WritableStrings;

  auto *GV = new llvm::GlobalVariable(getModule(), C->getType(), isConstant,
                                      Linkage, C, ".str");
  GV->setUnnamedAddr(true);
  // Don't enforce the target's minimum global alignment, since the only use
  // of the string is via this class initializer.
  CharUnits Align = getContext().getTypeAlignInChars(getContext().CharTy);
  GV->setAlignment(Align.getQuantity());
  Fields[1] = llvm::ConstantExpr::getGetElementPtr(GV, Zeros);
  
  // String length.
  llvm::Type *Ty = getTypes().ConvertType(getContext().UnsignedIntTy);
  Fields[2] = llvm::ConstantInt::get(Ty, StringLength);
  
  // The struct.
  C = llvm::ConstantStruct::get(NSConstantStringType, Fields);
  GV = new llvm::GlobalVariable(getModule(), C->getType(), true,
                                llvm::GlobalVariable::PrivateLinkage, C,
                                "_unnamed_nsstring_");
  const char *NSStringSection = "__OBJC,__cstring_object,regular,no_dead_strip";
  const char *NSStringNonFragileABISection =
      "__DATA,__objc_stringobj,regular,no_dead_strip";
  // FIXME. Fix section.
  GV->setSection(LangOpts.ObjCRuntime.isNonFragile()
                     ? NSStringNonFragileABISection
                     : NSStringSection);
  Entry.second = GV;

  return GV;
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
                      unsigned Alignment) {
  // OpenCL v1.2 s6.5.3: a string literal is in the constant address space.
  unsigned AddrSpace = 0;
  if (CGM.getLangOpts().OpenCL)
    AddrSpace = CGM.getContext().getTargetAddressSpace(LangAS::opencl_constant);

  // Create a global variable for this string
  auto *GV = new llvm::GlobalVariable(
      CGM.getModule(), C->getType(), !CGM.getLangOpts().WritableStrings, LT, C,
      GlobalName, nullptr, llvm::GlobalVariable::NotThreadLocal, AddrSpace);
  GV->setAlignment(Alignment);
  GV->setUnnamedAddr(true);
  return GV;
}

/// GetAddrOfConstantStringFromLiteral - Return a pointer to a
/// constant array for the given string literal.
llvm::GlobalVariable *
CodeGenModule::GetAddrOfConstantStringFromLiteral(const StringLiteral *S,
                                                  StringRef Name) {
  auto Alignment =
      getContext().getAlignOfGlobalVarInChars(S->getType()).getQuantity();

  llvm::Constant *C = GetConstantArrayFromStringLiteral(S);
  llvm::GlobalVariable **Entry = nullptr;
  if (!LangOpts.WritableStrings) {
    Entry = &ConstantStringMap[C];
    if (auto GV = *Entry) {
      if (Alignment > GV->getAlignment())
        GV->setAlignment(Alignment);
      return GV;
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
    Out.flush();

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
  return GV;
}

/// GetAddrOfConstantStringFromObjCEncode - Return a pointer to a constant
/// array for the given ObjCEncodeExpr node.
llvm::GlobalVariable *
CodeGenModule::GetAddrOfConstantStringFromObjCEncode(const ObjCEncodeExpr *E) {
  std::string Str;
  getContext().getObjCEncodingForType(E->getEncodedType(), Str);

  return GetAddrOfConstantCString(Str);
}

/// GetAddrOfConstantCString - Returns a pointer to a character array containing
/// the literal and a terminating '\0' character.
/// The result has pointer to array type.
llvm::GlobalVariable *CodeGenModule::GetAddrOfConstantCString(
    const std::string &Str, const char *GlobalName, unsigned Alignment) {
  StringRef StrWithNull(Str.c_str(), Str.size() + 1);
  if (Alignment == 0) {
    Alignment = getContext()
                    .getAlignOfGlobalVarInChars(getContext().CharTy)
                    .getQuantity();
  }

  llvm::Constant *C =
      llvm::ConstantDataArray::getString(getLLVMContext(), StrWithNull, false);

  // Don't share any string literals if strings aren't constant.
  llvm::GlobalVariable **Entry = nullptr;
  if (!LangOpts.WritableStrings) {
    Entry = &ConstantStringMap[C];
    if (auto GV = *Entry) {
      if (Alignment > GV->getAlignment())
        GV->setAlignment(Alignment);
      return GV;
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
  return GV;
}

llvm::Constant *CodeGenModule::GetAddrOfGlobalTemporary(
    const MaterializeTemporaryExpr *E, const Expr *Init) {
  assert((E->getStorageDuration() == SD_Static ||
          E->getStorageDuration() == SD_Thread) && "not a global temporary");
  const auto *VD = cast<VarDecl>(E->getExtendingDecl());

  // If we're not materializing a subobject of the temporary, keep the
  // cv-qualifiers from the type of the MaterializeTemporaryExpr.
  QualType MaterializedType = Init->getType();
  if (Init == E->GetTemporaryExpr())
    MaterializedType = E->getType();

  llvm::Constant *&Slot = MaterializedGlobalTemporaryMap[E];
  if (Slot)
    return Slot;

  // FIXME: If an externally-visible declaration extends multiple temporaries,
  // we need to give each temporary the same name in every translation unit (and
  // we also need to make the temporaries externally-visible).
  SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  getCXXABI().getMangleContext().mangleReferenceTemporary(
      VD, E->getManglingNumber(), Out);
  Out.flush();

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
  // There is no need for this temporary to have global linkage if the global
  // variable has external linkage.
  if (Linkage == llvm::GlobalVariable::ExternalLinkage)
    Linkage = llvm::GlobalVariable::PrivateLinkage;
  unsigned AddrSpace = GetGlobalVarAddressSpace(
      VD, getContext().getTargetAddressSpace(MaterializedType));
  auto *GV = new llvm::GlobalVariable(
      getModule(), Type, Constant, Linkage, InitialValue, Name.c_str(),
      /*InsertBefore=*/nullptr, llvm::GlobalVariable::NotThreadLocal,
      AddrSpace);
  setGlobalVisibility(GV, VD);
  GV->setAlignment(
      getContext().getTypeAlignInChars(MaterializedType).getQuantity());
  if (VD->getTLSKind())
    setTLSMode(GV, *VD);
  Slot = GV;
  return GV;
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

/// EmitNamespace - Emit all declarations in a namespace.
void CodeGenModule::EmitNamespace(const NamespaceDecl *ND) {
  for (auto *I : ND->decls()) {
    if (const auto *VD = dyn_cast<VarDecl>(I))
      if (VD->getTemplateSpecializationKind() != TSK_ExplicitSpecialization &&
          VD->getTemplateSpecializationKind() != TSK_Undeclared)
        continue;
    EmitTopLevelDecl(I);
  }
}

// EmitLinkageSpec - Emit all declarations in a linkage spec.
void CodeGenModule::EmitLinkageSpec(const LinkageSpecDecl *LSD) {
  if (LSD->getLanguage() != LinkageSpecDecl::lang_c &&
      LSD->getLanguage() != LinkageSpecDecl::lang_cxx) {
    ErrorUnsupported(LSD, "linkage spec");
    return;
  }

  for (auto *I : LSD->decls()) {
    // Meta-data for ObjC class includes references to implemented methods.
    // Generate class's method definitions first.
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

  case Decl::Var:
    // Skip variable templates
    if (cast<VarDecl>(D)->getDescribedVarTemplate())
      return;
  case Decl::VarTemplateSpecialization:
    EmitGlobal(cast<VarDecl>(D));
    break;

  // Indirect fields from global anonymous structs and unions can be
  // ignored; only the actual variable requires IR gen support.
  case Decl::IndirectField:
    break;

  // C++ Decls
  case Decl::Namespace:
    EmitNamespace(cast<NamespaceDecl>(D));
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
      if (getCodeGenOpts().getDebugInfo() >= CodeGenOptions::LimitedDebugInfo)
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

  case Decl::LinkageSpec:
    EmitLinkageSpec(cast<LinkageSpecDecl>(D));
    break;

  case Decl::FileScopeAsm: {
    auto *AD = cast<FileScopeAsmDecl>(D);
    StringRef AsmString = AD->getAsmString()->getString();

    const std::string &S = getModule().getModuleInlineAsm();
    if (S.empty())
      getModule().setModuleInlineAsm(AsmString);
    else if (S.end()[-1] == '\n')
      getModule().setModuleInlineAsm(S + AsmString.str());
    else
      getModule().setModuleInlineAsm(S + '\n' + AsmString.str());
    break;
  }

  case Decl::Import: {
    auto *Import = cast<ImportDecl>(D);

    // Ignore import declarations that come from imported modules.
    if (clang::Module *Owner = Import->getOwningModule()) {
      if (getLangOpts().CurrentModule.empty() ||
          Owner->getTopLevelModule()->Name == getLangOpts().CurrentModule)
        break;
    }

    ImportedModules.insert(Import->getImportedModule());
    break;
  }

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
    if (!cast<FunctionDecl>(D)->hasBody())
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
  for (const auto I : DeferredEmptyCoverageMappingDecls) {
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
  for (StaticExternCMap::iterator I = StaticExternCValues.begin(),
                                  E = StaticExternCValues.end();
       I != E; ++I) {
    IdentifierInfo *Name = I->first;
    llvm::GlobalValue *Val = I->second;
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

  // StaticLocalDeclMap
  for (auto &I : MangledDeclNames) {
    llvm::GlobalValue *Addr = getModule().getNamedValue(I.second);
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
    llvm::Value *Addr = I.second;
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
  if (!getCodeGenOpts().CoverageFile.empty()) {
    if (llvm::NamedMDNode *CUNode = TheModule.getNamedMetadata("llvm.dbg.cu")) {
      llvm::NamedMDNode *GCov = TheModule.getOrInsertNamedMetadata("llvm.gcov");
      llvm::LLVMContext &Ctx = TheModule.getContext();
      llvm::MDString *CoverageFile =
          llvm::MDString::get(Ctx, getCodeGenOpts().CoverageFile);
      for (int i = 0, e = CUNode->getNumOperands(); i != e; ++i) {
        llvm::MDNode *CU = CUNode->getOperand(i);
        llvm::Metadata *Elts[] = {CoverageFile, CU};
        GCov->addOperand(llvm::MDNode::get(Ctx, Elts));
      }
    }
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
    if (auto InitFunction =
            getOpenMPRuntime().EmitOMPThreadPrivateVarDefinition(
                VD, GetAddrOfGlobalVar(VD), RefExpr->getLocStart(),
                PerformInit))
      CXXGlobalInits.push_back(InitFunction);
  }
}

