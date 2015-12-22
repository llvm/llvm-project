//===--- BackendUtil.cpp - LLVM Backend Utilities -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/BackendUtil.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/Utils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/FunctionInfo.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/SymbolRewriter.h"
#include <memory>
using namespace clang;
using namespace llvm;

namespace {

class EmitAssemblyHelper {
  DiagnosticsEngine &Diags;
  const CodeGenOptions &CodeGenOpts;
  const clang::TargetOptions &TargetOpts;
  const LangOptions &LangOpts;
  Module *TheModule;
  std::unique_ptr<FunctionInfoIndex> FunctionIndex;

  Timer CodeGenerationTime;

  mutable legacy::PassManager *CodeGenPasses;
  mutable legacy::PassManager *PerModulePasses;
  mutable legacy::FunctionPassManager *PerFunctionPasses;

private:
  TargetIRAnalysis getTargetIRAnalysis() const {
    if (TM)
      return TM->getTargetIRAnalysis();

    return TargetIRAnalysis();
  }

  legacy::PassManager *getCodeGenPasses() const {
    if (!CodeGenPasses) {
      CodeGenPasses = new legacy::PassManager();
      CodeGenPasses->add(
          createTargetTransformInfoWrapperPass(getTargetIRAnalysis()));
    }
    return CodeGenPasses;
  }

  legacy::PassManager *getPerModulePasses() const {
    if (!PerModulePasses) {
      PerModulePasses = new legacy::PassManager();
      PerModulePasses->add(
          createTargetTransformInfoWrapperPass(getTargetIRAnalysis()));
    }
    return PerModulePasses;
  }

  legacy::FunctionPassManager *getPerFunctionPasses() const {
    if (!PerFunctionPasses) {
      PerFunctionPasses = new legacy::FunctionPassManager(TheModule);
      PerFunctionPasses->add(
          createTargetTransformInfoWrapperPass(getTargetIRAnalysis()));
    }
    return PerFunctionPasses;
  }

  void CreatePasses();

  /// Generates the TargetMachine.
  /// Returns Null if it is unable to create the target machine.
  /// Some of our clang tests specify triples which are not built
  /// into clang. This is okay because these tests check the generated
  /// IR, and they require DataLayout which depends on the triple.
  /// In this case, we allow this method to fail and not report an error.
  /// When MustCreateTM is used, we print an error if we are unable to load
  /// the requested target.
  TargetMachine *CreateTargetMachine(bool MustCreateTM);

  /// Add passes necessary to emit assembly or LLVM IR.
  ///
  /// \return True on success.
  bool AddEmitPasses(BackendAction Action, raw_pwrite_stream &OS);

public:
  EmitAssemblyHelper(DiagnosticsEngine &_Diags, const CodeGenOptions &CGOpts,
                     const clang::TargetOptions &TOpts,
                     const LangOptions &LOpts, Module *M,
                     std::unique_ptr<FunctionInfoIndex> Index)
      : Diags(_Diags), CodeGenOpts(CGOpts), TargetOpts(TOpts), LangOpts(LOpts),
        TheModule(M), FunctionIndex(std::move(Index)),
        CodeGenerationTime("Code Generation Time"), CodeGenPasses(nullptr),
        PerModulePasses(nullptr), PerFunctionPasses(nullptr) {}

  ~EmitAssemblyHelper() {
    delete CodeGenPasses;
    delete PerModulePasses;
    delete PerFunctionPasses;
    if (CodeGenOpts.DisableFree)
      BuryPointer(std::move(TM));
  }

  std::unique_ptr<TargetMachine> TM;

  void EmitAssembly(BackendAction Action, raw_pwrite_stream *OS);
};

// We need this wrapper to access LangOpts and CGOpts from extension functions
// that we add to the PassManagerBuilder.
class PassManagerBuilderWrapper : public PassManagerBuilder {
public:
  PassManagerBuilderWrapper(const CodeGenOptions &CGOpts,
                            const LangOptions &LangOpts)
      : PassManagerBuilder(), CGOpts(CGOpts), LangOpts(LangOpts) {}
  const CodeGenOptions &getCGOpts() const { return CGOpts; }
  const LangOptions &getLangOpts() const { return LangOpts; }
private:
  const CodeGenOptions &CGOpts;
  const LangOptions &LangOpts;
};

}

static void addObjCARCAPElimPass(const PassManagerBuilder &Builder, PassManagerBase &PM) {
  if (Builder.OptLevel > 0)
    PM.add(createObjCARCAPElimPass());
}

static void addObjCARCExpandPass(const PassManagerBuilder &Builder, PassManagerBase &PM) {
  if (Builder.OptLevel > 0)
    PM.add(createObjCARCExpandPass());
}

static void addObjCARCOptPass(const PassManagerBuilder &Builder, PassManagerBase &PM) {
  if (Builder.OptLevel > 0)
    PM.add(createObjCARCOptPass());
}

static void addAddDiscriminatorsPass(const PassManagerBuilder &Builder,
                                     legacy::PassManagerBase &PM) {
  PM.add(createAddDiscriminatorsPass());
}

static void addBoundsCheckingPass(const PassManagerBuilder &Builder,
                                    legacy::PassManagerBase &PM) {
  PM.add(createBoundsCheckingPass());
}

static void addSanitizerCoveragePass(const PassManagerBuilder &Builder,
                                     legacy::PassManagerBase &PM) {
  const PassManagerBuilderWrapper &BuilderWrapper =
      static_cast<const PassManagerBuilderWrapper&>(Builder);
  const CodeGenOptions &CGOpts = BuilderWrapper.getCGOpts();
  SanitizerCoverageOptions Opts;
  Opts.CoverageType =
      static_cast<SanitizerCoverageOptions::Type>(CGOpts.SanitizeCoverageType);
  Opts.IndirectCalls = CGOpts.SanitizeCoverageIndirectCalls;
  Opts.TraceBB = CGOpts.SanitizeCoverageTraceBB;
  Opts.TraceCmp = CGOpts.SanitizeCoverageTraceCmp;
  Opts.Use8bitCounters = CGOpts.SanitizeCoverage8bitCounters;
  PM.add(createSanitizerCoverageModulePass(Opts));
}

static void addAddressSanitizerPasses(const PassManagerBuilder &Builder,
                                      legacy::PassManagerBase &PM) {
  const PassManagerBuilderWrapper &BuilderWrapper =
      static_cast<const PassManagerBuilderWrapper&>(Builder);
  const CodeGenOptions &CGOpts = BuilderWrapper.getCGOpts();
  bool Recover = CGOpts.SanitizeRecover.has(SanitizerKind::Address);
  PM.add(createAddressSanitizerFunctionPass(/*CompileKernel*/false, Recover));
  PM.add(createAddressSanitizerModulePass(/*CompileKernel*/false, Recover));
}

static void addKernelAddressSanitizerPasses(const PassManagerBuilder &Builder,
                                            legacy::PassManagerBase &PM) {
  PM.add(createAddressSanitizerFunctionPass(/*CompileKernel*/true,
                                            /*Recover*/true));
  PM.add(createAddressSanitizerModulePass(/*CompileKernel*/true,
                                          /*Recover*/true));
}

static void addMemorySanitizerPass(const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
  const PassManagerBuilderWrapper &BuilderWrapper =
      static_cast<const PassManagerBuilderWrapper&>(Builder);
  const CodeGenOptions &CGOpts = BuilderWrapper.getCGOpts();
  PM.add(createMemorySanitizerPass(CGOpts.SanitizeMemoryTrackOrigins));

  // MemorySanitizer inserts complex instrumentation that mostly follows
  // the logic of the original code, but operates on "shadow" values.
  // It can benefit from re-running some general purpose optimization passes.
  if (Builder.OptLevel > 0) {
    PM.add(createEarlyCSEPass());
    PM.add(createReassociatePass());
    PM.add(createLICMPass());
    PM.add(createGVNPass());
    PM.add(createInstructionCombiningPass());
    PM.add(createDeadStoreEliminationPass());
  }
}

static void addThreadSanitizerPass(const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
  PM.add(createThreadSanitizerPass());
}

static void addDataFlowSanitizerPass(const PassManagerBuilder &Builder,
                                     legacy::PassManagerBase &PM) {
  const PassManagerBuilderWrapper &BuilderWrapper =
      static_cast<const PassManagerBuilderWrapper&>(Builder);
  const LangOptions &LangOpts = BuilderWrapper.getLangOpts();
  PM.add(createDataFlowSanitizerPass(LangOpts.SanitizerBlacklistFiles));
}

static TargetLibraryInfoImpl *createTLII(llvm::Triple &TargetTriple,
                                         const CodeGenOptions &CodeGenOpts) {
  TargetLibraryInfoImpl *TLII = new TargetLibraryInfoImpl(TargetTriple);
  if (!CodeGenOpts.SimplifyLibCalls)
    TLII->disableAllFunctions();

  switch (CodeGenOpts.getVecLib()) {
  case CodeGenOptions::Accelerate:
    TLII->addVectorizableFunctionsFromVecLib(TargetLibraryInfoImpl::Accelerate);
    break;
  default:
    break;
  }
  return TLII;
}

static void addSymbolRewriterPass(const CodeGenOptions &Opts,
                                  legacy::PassManager *MPM) {
  llvm::SymbolRewriter::RewriteDescriptorList DL;

  llvm::SymbolRewriter::RewriteMapParser MapParser;
  for (const auto &MapFile : Opts.RewriteMapFiles)
    MapParser.parse(MapFile, &DL);

  MPM->add(createRewriteSymbolsPass(DL));
}

void EmitAssemblyHelper::CreatePasses() {
  if (CodeGenOpts.DisableLLVMPasses)
    return;

  unsigned OptLevel = CodeGenOpts.OptimizationLevel;
  CodeGenOptions::InliningMethod Inlining = CodeGenOpts.getInlining();

  // Handle disabling of LLVM optimization, where we want to preserve the
  // internal module before any optimization.
  if (CodeGenOpts.DisableLLVMOpts) {
    OptLevel = 0;
    Inlining = CodeGenOpts.NoInlining;
  }

  PassManagerBuilderWrapper PMBuilder(CodeGenOpts, LangOpts);

  // Figure out TargetLibraryInfo.
  Triple TargetTriple(TheModule->getTargetTriple());
  PMBuilder.LibraryInfo = createTLII(TargetTriple, CodeGenOpts);

  switch (Inlining) {
  case CodeGenOptions::NoInlining:
    break;
  case CodeGenOptions::NormalInlining: {
    PMBuilder.Inliner =
        createFunctionInliningPass(OptLevel, CodeGenOpts.OptimizeSize);
    break;
  }
  case CodeGenOptions::OnlyAlwaysInlining:
    // Respect always_inline.
    if (OptLevel == 0)
      // Do not insert lifetime intrinsics at -O0.
      PMBuilder.Inliner = createAlwaysInlinerPass(false);
    else
      PMBuilder.Inliner = createAlwaysInlinerPass();
    break;
  }

  PMBuilder.OptLevel = OptLevel;
  PMBuilder.SizeLevel = CodeGenOpts.OptimizeSize;
  PMBuilder.BBVectorize = CodeGenOpts.VectorizeBB;
  PMBuilder.SLPVectorize = CodeGenOpts.VectorizeSLP;
  PMBuilder.LoopVectorize = CodeGenOpts.VectorizeLoop;

  PMBuilder.DisableUnitAtATime = !CodeGenOpts.UnitAtATime;
  PMBuilder.DisableUnrollLoops = !CodeGenOpts.UnrollLoops;
  PMBuilder.MergeFunctions = CodeGenOpts.MergeFunctions;
  PMBuilder.PrepareForLTO = CodeGenOpts.PrepareForLTO;
  PMBuilder.RerollLoops = CodeGenOpts.RerollLoops;

  legacy::PassManager *MPM = getPerModulePasses();

  // If we are performing a ThinLTO importing compile, invoke the LTO
  // pipeline and pass down the in-memory function index.
  if (!CodeGenOpts.ThinLTOIndexFile.empty()) {
    assert(FunctionIndex && "Expected non-empty function index");
    PMBuilder.FunctionIndex = FunctionIndex.get();
    PMBuilder.populateLTOPassManager(*MPM);
    return;
  }

  PMBuilder.addExtension(PassManagerBuilder::EP_EarlyAsPossible,
                         addAddDiscriminatorsPass);

  // In ObjC ARC mode, add the main ARC optimization passes.
  if (LangOpts.ObjCAutoRefCount) {
    PMBuilder.addExtension(PassManagerBuilder::EP_EarlyAsPossible,
                           addObjCARCExpandPass);
    PMBuilder.addExtension(PassManagerBuilder::EP_ModuleOptimizerEarly,
                           addObjCARCAPElimPass);
    PMBuilder.addExtension(PassManagerBuilder::EP_ScalarOptimizerLate,
                           addObjCARCOptPass);
  }

  if (LangOpts.Sanitize.has(SanitizerKind::LocalBounds)) {
    PMBuilder.addExtension(PassManagerBuilder::EP_ScalarOptimizerLate,
                           addBoundsCheckingPass);
    PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                           addBoundsCheckingPass);
  }

  if (CodeGenOpts.SanitizeCoverageType ||
      CodeGenOpts.SanitizeCoverageIndirectCalls ||
      CodeGenOpts.SanitizeCoverageTraceCmp) {
    PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                           addSanitizerCoveragePass);
    PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                           addSanitizerCoveragePass);
  }

  if (LangOpts.Sanitize.has(SanitizerKind::Address)) {
    PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                           addAddressSanitizerPasses);
    PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                           addAddressSanitizerPasses);
  }

  if (LangOpts.Sanitize.has(SanitizerKind::KernelAddress)) {
    PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                           addKernelAddressSanitizerPasses);
    PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                           addKernelAddressSanitizerPasses);
  }

  if (LangOpts.Sanitize.has(SanitizerKind::Memory)) {
    PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                           addMemorySanitizerPass);
    PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                           addMemorySanitizerPass);
  }

  if (LangOpts.Sanitize.has(SanitizerKind::Thread)) {
    PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                           addThreadSanitizerPass);
    PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                           addThreadSanitizerPass);
  }

  if (LangOpts.Sanitize.has(SanitizerKind::DataFlow)) {
    PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                           addDataFlowSanitizerPass);
    PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                           addDataFlowSanitizerPass);
  }

  // Set up the per-function pass manager.
  legacy::FunctionPassManager *FPM = getPerFunctionPasses();
  if (CodeGenOpts.VerifyModule)
    FPM->add(createVerifierPass());
  PMBuilder.populateFunctionPassManager(*FPM);

  // Set up the per-module pass manager.
  if (!CodeGenOpts.RewriteMapFiles.empty())
    addSymbolRewriterPass(CodeGenOpts, MPM);

  if (!CodeGenOpts.DisableGCov &&
      (CodeGenOpts.EmitGcovArcs || CodeGenOpts.EmitGcovNotes)) {
    // Not using 'GCOVOptions::getDefault' allows us to avoid exiting if
    // LLVM's -default-gcov-version flag is set to something invalid.
    GCOVOptions Options;
    Options.EmitNotes = CodeGenOpts.EmitGcovNotes;
    Options.EmitData = CodeGenOpts.EmitGcovArcs;
    memcpy(Options.Version, CodeGenOpts.CoverageVersion, 4);
    Options.UseCfgChecksum = CodeGenOpts.CoverageExtraChecksum;
    Options.NoRedZone = CodeGenOpts.DisableRedZone;
    Options.FunctionNamesInData =
        !CodeGenOpts.CoverageNoFunctionNamesInData;
    Options.ExitBlockBeforeBody = CodeGenOpts.CoverageExitBlockBeforeBody;
    MPM->add(createGCOVProfilerPass(Options));
    if (CodeGenOpts.getDebugInfo() == CodeGenOptions::NoDebugInfo)
      MPM->add(createStripSymbolsPass(true));
  }

  if (CodeGenOpts.ProfileInstrGenerate) {
    InstrProfOptions Options;
    Options.NoRedZone = CodeGenOpts.DisableRedZone;
    Options.InstrProfileOutput = CodeGenOpts.InstrProfileOutput;
    MPM->add(createInstrProfilingPass(Options));
  }

  if (!CodeGenOpts.SampleProfileFile.empty())
    MPM->add(createSampleProfileLoaderPass(CodeGenOpts.SampleProfileFile));

  PMBuilder.populateModulePassManager(*MPM);
}

TargetMachine *EmitAssemblyHelper::CreateTargetMachine(bool MustCreateTM) {
  // Create the TargetMachine for generating code.
  std::string Error;
  std::string Triple = TheModule->getTargetTriple();
  const llvm::Target *TheTarget = TargetRegistry::lookupTarget(Triple, Error);
  if (!TheTarget) {
    if (MustCreateTM)
      Diags.Report(diag::err_fe_unable_to_create_target) << Error;
    return nullptr;
  }

  unsigned CodeModel =
    llvm::StringSwitch<unsigned>(CodeGenOpts.CodeModel)
      .Case("small", llvm::CodeModel::Small)
      .Case("kernel", llvm::CodeModel::Kernel)
      .Case("medium", llvm::CodeModel::Medium)
      .Case("large", llvm::CodeModel::Large)
      .Case("default", llvm::CodeModel::Default)
      .Default(~0u);
  assert(CodeModel != ~0u && "invalid code model!");
  llvm::CodeModel::Model CM = static_cast<llvm::CodeModel::Model>(CodeModel);

  SmallVector<const char *, 16> BackendArgs;
  BackendArgs.push_back("clang"); // Fake program name.
  if (!CodeGenOpts.DebugPass.empty()) {
    BackendArgs.push_back("-debug-pass");
    BackendArgs.push_back(CodeGenOpts.DebugPass.c_str());
  }
  if (!CodeGenOpts.LimitFloatPrecision.empty()) {
    BackendArgs.push_back("-limit-float-precision");
    BackendArgs.push_back(CodeGenOpts.LimitFloatPrecision.c_str());
  }
  for (const std::string &BackendOption : CodeGenOpts.BackendOptions)
    BackendArgs.push_back(BackendOption.c_str());
  BackendArgs.push_back(nullptr);
  llvm::cl::ParseCommandLineOptions(BackendArgs.size() - 1,
                                    BackendArgs.data());

  std::string FeaturesStr =
      llvm::join(TargetOpts.Features.begin(), TargetOpts.Features.end(), ",");

  // Keep this synced with the equivalent code in tools/driver/cc1as_main.cpp.
  llvm::Reloc::Model RM = llvm::Reloc::Default;
  if (CodeGenOpts.RelocationModel == "static") {
    RM = llvm::Reloc::Static;
  } else if (CodeGenOpts.RelocationModel == "pic") {
    RM = llvm::Reloc::PIC_;
  } else {
    assert(CodeGenOpts.RelocationModel == "dynamic-no-pic" &&
           "Invalid PIC model!");
    RM = llvm::Reloc::DynamicNoPIC;
  }

  CodeGenOpt::Level OptLevel = CodeGenOpt::Default;
  switch (CodeGenOpts.OptimizationLevel) {
  default: break;
  case 0: OptLevel = CodeGenOpt::None; break;
  case 3: OptLevel = CodeGenOpt::Aggressive; break;
  }

  llvm::TargetOptions Options;

  if (!TargetOpts.Reciprocals.empty())
    Options.Reciprocals = TargetRecip(TargetOpts.Reciprocals);

  Options.ThreadModel =
    llvm::StringSwitch<llvm::ThreadModel::Model>(CodeGenOpts.ThreadModel)
      .Case("posix", llvm::ThreadModel::POSIX)
      .Case("single", llvm::ThreadModel::Single);

  // Set float ABI type.
  assert((CodeGenOpts.FloatABI == "soft" || CodeGenOpts.FloatABI == "softfp" ||
          CodeGenOpts.FloatABI == "hard" || CodeGenOpts.FloatABI.empty()) &&
         "Invalid Floating Point ABI!");
  Options.FloatABIType =
      llvm::StringSwitch<llvm::FloatABI::ABIType>(CodeGenOpts.FloatABI)
          .Case("soft", llvm::FloatABI::Soft)
          .Case("softfp", llvm::FloatABI::Soft)
          .Case("hard", llvm::FloatABI::Hard)
          .Default(llvm::FloatABI::Default);

  // Set FP fusion mode.
  switch (CodeGenOpts.getFPContractMode()) {
  case CodeGenOptions::FPC_Off:
    Options.AllowFPOpFusion = llvm::FPOpFusion::Strict;
    break;
  case CodeGenOptions::FPC_On:
    Options.AllowFPOpFusion = llvm::FPOpFusion::Standard;
    break;
  case CodeGenOptions::FPC_Fast:
    Options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    break;
  }

  Options.UseInitArray = CodeGenOpts.UseInitArray;
  Options.DisableIntegratedAS = CodeGenOpts.DisableIntegratedAS;
  Options.CompressDebugSections = CodeGenOpts.CompressDebugSections;

  // Set EABI version.
  Options.EABIVersion = llvm::StringSwitch<llvm::EABI>(CodeGenOpts.EABIVersion)
                            .Case("4", llvm::EABI::EABI4)
                            .Case("5", llvm::EABI::EABI5)
                            .Case("gnu", llvm::EABI::GNU)
                            .Default(llvm::EABI::Default);

  Options.LessPreciseFPMADOption = CodeGenOpts.LessPreciseFPMAD;
  Options.NoInfsFPMath = CodeGenOpts.NoInfsFPMath;
  Options.NoNaNsFPMath = CodeGenOpts.NoNaNsFPMath;
  Options.NoZerosInBSS = CodeGenOpts.NoZeroInitializedInBSS;
  Options.UnsafeFPMath = CodeGenOpts.UnsafeFPMath;
  Options.StackAlignmentOverride = CodeGenOpts.StackAlignment;
  Options.PositionIndependentExecutable = LangOpts.PIELevel != 0;
  Options.FunctionSections = CodeGenOpts.FunctionSections;
  Options.DataSections = CodeGenOpts.DataSections;
  Options.UniqueSectionNames = CodeGenOpts.UniqueSectionNames;
  Options.EmulatedTLS = CodeGenOpts.EmulatedTLS;
  switch (CodeGenOpts.getDebuggerTuning()) {
  case CodeGenOptions::DebuggerKindGDB:
    Options.DebuggerTuning = llvm::DebuggerKind::GDB;
    break;
  case CodeGenOptions::DebuggerKindLLDB:
    Options.DebuggerTuning = llvm::DebuggerKind::LLDB;
    break;
  case CodeGenOptions::DebuggerKindSCE:
    Options.DebuggerTuning = llvm::DebuggerKind::SCE;
    break;
  default:
    break;
  }

  Options.MCOptions.MCRelaxAll = CodeGenOpts.RelaxAll;
  Options.MCOptions.MCSaveTempLabels = CodeGenOpts.SaveTempLabels;
  Options.MCOptions.MCUseDwarfDirectory = !CodeGenOpts.NoDwarfDirectoryAsm;
  Options.MCOptions.MCNoExecStack = CodeGenOpts.NoExecStack;
  Options.MCOptions.MCIncrementalLinkerCompatible =
      CodeGenOpts.IncrementalLinkerCompatible;
  Options.MCOptions.MCFatalWarnings = CodeGenOpts.FatalWarnings;
  Options.MCOptions.AsmVerbose = CodeGenOpts.AsmVerbose;
  Options.MCOptions.ABIName = TargetOpts.ABI;

  TargetMachine *TM = TheTarget->createTargetMachine(Triple, TargetOpts.CPU,
                                                     FeaturesStr, Options,
                                                     RM, CM, OptLevel);

  return TM;
}

bool EmitAssemblyHelper::AddEmitPasses(BackendAction Action,
                                       raw_pwrite_stream &OS) {

  // Create the code generator passes.
  legacy::PassManager *PM = getCodeGenPasses();

  // Add LibraryInfo.
  llvm::Triple TargetTriple(TheModule->getTargetTriple());
  std::unique_ptr<TargetLibraryInfoImpl> TLII(
      createTLII(TargetTriple, CodeGenOpts));
  PM->add(new TargetLibraryInfoWrapperPass(*TLII));

  // Normal mode, emit a .s or .o file by running the code generator. Note,
  // this also adds codegenerator level optimization passes.
  TargetMachine::CodeGenFileType CGFT = TargetMachine::CGFT_AssemblyFile;
  if (Action == Backend_EmitObj)
    CGFT = TargetMachine::CGFT_ObjectFile;
  else if (Action == Backend_EmitMCNull)
    CGFT = TargetMachine::CGFT_Null;
  else
    assert(Action == Backend_EmitAssembly && "Invalid action!");

  // Add ObjC ARC final-cleanup optimizations. This is done as part of the
  // "codegen" passes so that it isn't run multiple times when there is
  // inlining happening.
  if (CodeGenOpts.OptimizationLevel > 0)
    PM->add(createObjCARCContractPass());

  if (TM->addPassesToEmitFile(*PM, OS, CGFT,
                              /*DisableVerify=*/!CodeGenOpts.VerifyModule)) {
    Diags.Report(diag::err_fe_unable_to_interface_with_target);
    return false;
  }

  return true;
}

void EmitAssemblyHelper::EmitAssembly(BackendAction Action,
                                      raw_pwrite_stream *OS) {
  TimeRegion Region(llvm::TimePassesIsEnabled ? &CodeGenerationTime : nullptr);

  bool UsesCodeGen = (Action != Backend_EmitNothing &&
                      Action != Backend_EmitBC &&
                      Action != Backend_EmitLL);
  if (!TM)
    TM.reset(CreateTargetMachine(UsesCodeGen));

  if (UsesCodeGen && !TM)
    return;
  if (TM)
    TheModule->setDataLayout(TM->createDataLayout());
  CreatePasses();

  switch (Action) {
  case Backend_EmitNothing:
    break;

  case Backend_EmitBC:
    getPerModulePasses()->add(createBitcodeWriterPass(
        *OS, CodeGenOpts.EmitLLVMUseLists, CodeGenOpts.EmitFunctionSummary));
    break;

  case Backend_EmitLL:
    getPerModulePasses()->add(
        createPrintModulePass(*OS, "", CodeGenOpts.EmitLLVMUseLists));
    break;

  default:
    if (!AddEmitPasses(Action, *OS))
      return;
  }

  // Before executing passes, print the final values of the LLVM options.
  cl::PrintOptionValues();

  // Run passes. For now we do all passes at once, but eventually we
  // would like to have the option of streaming code generation.

  if (PerFunctionPasses) {
    PrettyStackTraceString CrashInfo("Per-function optimization");

    PerFunctionPasses->doInitialization();
    for (Function &F : *TheModule)
      if (!F.isDeclaration())
        PerFunctionPasses->run(F);
    PerFunctionPasses->doFinalization();
  }

  if (PerModulePasses) {
    PrettyStackTraceString CrashInfo("Per-module optimization passes");
    PerModulePasses->run(*TheModule);
  }

  if (CodeGenPasses) {
    PrettyStackTraceString CrashInfo("Code generation");
    CodeGenPasses->run(*TheModule);
  }
}

void clang::EmitBackendOutput(DiagnosticsEngine &Diags,
                              const CodeGenOptions &CGOpts,
                              const clang::TargetOptions &TOpts,
                              const LangOptions &LOpts, StringRef TDesc,
                              Module *M, BackendAction Action,
                              raw_pwrite_stream *OS,
                              std::unique_ptr<FunctionInfoIndex> Index) {
  EmitAssemblyHelper AsmHelper(Diags, CGOpts, TOpts, LOpts, M,
                               std::move(Index));

  AsmHelper.EmitAssembly(Action, OS);

  // If an optional clang TargetInfo description string was passed in, use it to
  // verify the LLVM TargetMachine's DataLayout.
  if (AsmHelper.TM && !TDesc.empty()) {
    std::string DLDesc = M->getDataLayout().getStringRepresentation();
    if (DLDesc != TDesc) {
      unsigned DiagID = Diags.getCustomDiagID(
          DiagnosticsEngine::Error, "backend data layout '%0' does not match "
                                    "expected target description '%1'");
      Diags.Report(DiagID) << DLDesc << TDesc;
    }
  }
}
