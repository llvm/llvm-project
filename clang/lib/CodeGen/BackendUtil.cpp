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
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/LTO/LTOBackend.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Object/ModuleSummaryIndexObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
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
#include "llvm/Transforms/Scalar/GVN.h"
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

  Timer CodeGenerationTime;

  std::unique_ptr<raw_pwrite_stream> OS;

private:
  TargetIRAnalysis getTargetIRAnalysis() const {
    if (TM)
      return TM->getTargetIRAnalysis();

    return TargetIRAnalysis();
  }

  /// Set LLVM command line options passed through -backend-option.
  void setCommandLineOpts();

  void CreatePasses(legacy::PassManager &MPM, legacy::FunctionPassManager &FPM);

  /// Generates the TargetMachine.
  /// Leaves TM unchanged if it is unable to create the target machine.
  /// Some of our clang tests specify triples which are not built
  /// into clang. This is okay because these tests check the generated
  /// IR, and they require DataLayout which depends on the triple.
  /// In this case, we allow this method to fail and not report an error.
  /// When MustCreateTM is used, we print an error if we are unable to load
  /// the requested target.
  void CreateTargetMachine(bool MustCreateTM);

  /// Add passes necessary to emit assembly or LLVM IR.
  ///
  /// \return True on success.
  bool AddEmitPasses(legacy::PassManager &CodeGenPasses, BackendAction Action,
                     raw_pwrite_stream &OS);

public:
  EmitAssemblyHelper(DiagnosticsEngine &_Diags, const CodeGenOptions &CGOpts,
                     const clang::TargetOptions &TOpts,
                     const LangOptions &LOpts, Module *M)
      : Diags(_Diags), CodeGenOpts(CGOpts), TargetOpts(TOpts), LangOpts(LOpts),
        TheModule(M), CodeGenerationTime("Code Generation Time") {}

  ~EmitAssemblyHelper() {
    if (CodeGenOpts.DisableFree)
      BuryPointer(std::move(TM));
  }

  std::unique_ptr<TargetMachine> TM;

  void EmitAssembly(BackendAction Action,
                    std::unique_ptr<raw_pwrite_stream> OS);
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

static void addCleanupPassesForSampleProfiler(
    const PassManagerBuilder &Builder, legacy::PassManagerBase &PM) {
  // instcombine is needed before sample profile annotation because it converts
  // certain function calls to be inlinable. simplifycfg and sroa are needed
  // before instcombine for necessary preparation. E.g. load store is eliminated
  // properly so that instcombine will not introduce unecessary liverange.
  PM.add(createCFGSimplificationPass());
  PM.add(createSROAPass());
  PM.add(createInstructionCombiningPass());
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
  Opts.TracePC = CGOpts.SanitizeCoverageTracePC;
  PM.add(createSanitizerCoverageModulePass(Opts));
}

static void addAddressSanitizerPasses(const PassManagerBuilder &Builder,
                                      legacy::PassManagerBase &PM) {
  const PassManagerBuilderWrapper &BuilderWrapper =
      static_cast<const PassManagerBuilderWrapper&>(Builder);
  const CodeGenOptions &CGOpts = BuilderWrapper.getCGOpts();
  bool Recover = CGOpts.SanitizeRecover.has(SanitizerKind::Address);
  bool UseAfterScope = CGOpts.SanitizeAddressUseAfterScope;
  PM.add(createAddressSanitizerFunctionPass(/*CompileKernel*/ false, Recover,
                                            UseAfterScope));
  PM.add(createAddressSanitizerModulePass(/*CompileKernel*/false, Recover));
}

static void addKernelAddressSanitizerPasses(const PassManagerBuilder &Builder,
                                            legacy::PassManagerBase &PM) {
  PM.add(createAddressSanitizerFunctionPass(
      /*CompileKernel*/ true,
      /*Recover*/ true, /*UseAfterScope*/ false));
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

static void addEfficiencySanitizerPass(const PassManagerBuilder &Builder,
                                       legacy::PassManagerBase &PM) {
  const PassManagerBuilderWrapper &BuilderWrapper =
      static_cast<const PassManagerBuilderWrapper&>(Builder);
  const LangOptions &LangOpts = BuilderWrapper.getLangOpts();
  EfficiencySanitizerOptions Opts;
  if (LangOpts.Sanitize.has(SanitizerKind::EfficiencyCacheFrag))
    Opts.ToolType = EfficiencySanitizerOptions::ESAN_CacheFrag;
  else if (LangOpts.Sanitize.has(SanitizerKind::EfficiencyWorkingSet))
    Opts.ToolType = EfficiencySanitizerOptions::ESAN_WorkingSet;
  PM.add(createEfficiencySanitizerPass(Opts));
}

static TargetLibraryInfoImpl *createTLII(llvm::Triple &TargetTriple,
                                         const CodeGenOptions &CodeGenOpts) {
  TargetLibraryInfoImpl *TLII = new TargetLibraryInfoImpl(TargetTriple);
  if (!CodeGenOpts.SimplifyLibCalls)
    TLII->disableAllFunctions();
  else {
    // Disable individual libc/libm calls in TargetLibraryInfo.
    LibFunc::Func F;
    for (auto &FuncName : CodeGenOpts.getNoBuiltinFuncs())
      if (TLII->getLibFunc(FuncName, F))
        TLII->setUnavailable(F);
  }

  switch (CodeGenOpts.getVecLib()) {
  case CodeGenOptions::Accelerate:
    TLII->addVectorizableFunctionsFromVecLib(TargetLibraryInfoImpl::Accelerate);
    break;
  case CodeGenOptions::SVML:
    TLII->addVectorizableFunctionsFromVecLib(TargetLibraryInfoImpl::SVML);
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

void EmitAssemblyHelper::CreatePasses(legacy::PassManager &MPM,
                                      legacy::FunctionPassManager &FPM) {
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
  case CodeGenOptions::NormalInlining:
  case CodeGenOptions::OnlyHintInlining: {
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

  PMBuilder.DisableUnrollLoops = !CodeGenOpts.UnrollLoops;
  PMBuilder.MergeFunctions = CodeGenOpts.MergeFunctions;
  PMBuilder.PrepareForThinLTO = CodeGenOpts.EmitSummaryIndex;
  PMBuilder.PrepareForLTO = CodeGenOpts.PrepareForLTO;
  PMBuilder.RerollLoops = CodeGenOpts.RerollLoops;

  // Add target-specific passes that need to run as early as possible.
  if (TM)
    PMBuilder.addExtension(
        PassManagerBuilder::EP_EarlyAsPossible,
        [&](const PassManagerBuilder &, legacy::PassManagerBase &PM) {
          TM->addEarlyAsPossiblePasses(PM);
        });

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

  if (LangOpts.Sanitize.hasOneOf(SanitizerKind::Efficiency)) {
    PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                           addEfficiencySanitizerPass);
    PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                           addEfficiencySanitizerPass);
  }

  // Set up the per-function pass manager.
  if (CodeGenOpts.VerifyModule)
    FPM.add(createVerifierPass());

  // Set up the per-module pass manager.
  if (!CodeGenOpts.RewriteMapFiles.empty())
    addSymbolRewriterPass(CodeGenOpts, &MPM);

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
    MPM.add(createGCOVProfilerPass(Options));
    if (CodeGenOpts.getDebugInfo() == codegenoptions::NoDebugInfo)
      MPM.add(createStripSymbolsPass(true));
  }

  if (CodeGenOpts.hasProfileClangInstr()) {
    InstrProfOptions Options;
    Options.NoRedZone = CodeGenOpts.DisableRedZone;
    Options.InstrProfileOutput = CodeGenOpts.InstrProfileOutput;
    MPM.add(createInstrProfilingLegacyPass(Options));
  }
  if (CodeGenOpts.hasProfileIRInstr()) {
    PMBuilder.EnablePGOInstrGen = true;
    if (!CodeGenOpts.InstrProfileOutput.empty())
      PMBuilder.PGOInstrGen = CodeGenOpts.InstrProfileOutput;
    else
      PMBuilder.PGOInstrGen = "default_%m.profraw";
  }
  if (CodeGenOpts.hasProfileIRUse())
    PMBuilder.PGOInstrUse = CodeGenOpts.ProfileInstrumentUsePath;

  if (!CodeGenOpts.SampleProfileFile.empty()) {
    MPM.add(createPruneEHPass());
    MPM.add(createSampleProfileLoaderPass(CodeGenOpts.SampleProfileFile));
    PMBuilder.addExtension(PassManagerBuilder::EP_EarlyAsPossible,
                           addCleanupPassesForSampleProfiler);
  }

  PMBuilder.populateFunctionPassManager(FPM);
  PMBuilder.populateModulePassManager(MPM);
}

void EmitAssemblyHelper::setCommandLineOpts() {
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
}

void EmitAssemblyHelper::CreateTargetMachine(bool MustCreateTM) {
  // Create the TargetMachine for generating code.
  std::string Error;
  std::string Triple = TheModule->getTargetTriple();
  const llvm::Target *TheTarget = TargetRegistry::lookupTarget(Triple, Error);
  if (!TheTarget) {
    if (MustCreateTM)
      Diags.Report(diag::err_fe_unable_to_create_target) << Error;
    return;
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

  std::string FeaturesStr =
      llvm::join(TargetOpts.Features.begin(), TargetOpts.Features.end(), ",");

  // Keep this synced with the equivalent code in tools/driver/cc1as_main.cpp.
  llvm::Optional<llvm::Reloc::Model> RM;
  if (CodeGenOpts.RelocationModel == "static") {
    RM = llvm::Reloc::Static;
  } else if (CodeGenOpts.RelocationModel == "pic") {
    RM = llvm::Reloc::PIC_;
  } else if (CodeGenOpts.RelocationModel == "ropi") {
    RM = llvm::Reloc::ROPI;
  } else if (CodeGenOpts.RelocationModel == "rwpi") {
    RM = llvm::Reloc::RWPI;
  } else if (CodeGenOpts.RelocationModel == "ropi-rwpi") {
    RM = llvm::Reloc::ROPI_RWPI;
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
  Options.RelaxELFRelocations = CodeGenOpts.RelaxELFRelocations;

  // Set EABI version.
  Options.EABIVersion = llvm::StringSwitch<llvm::EABI>(TargetOpts.EABIVersion)
                            .Case("4", llvm::EABI::EABI4)
                            .Case("5", llvm::EABI::EABI5)
                            .Case("gnu", llvm::EABI::GNU)
                            .Default(llvm::EABI::Default);

  if (LangOpts.SjLjExceptions)
    Options.ExceptionModel = llvm::ExceptionHandling::SjLj;

  Options.LessPreciseFPMADOption = CodeGenOpts.LessPreciseFPMAD;
  Options.NoInfsFPMath = CodeGenOpts.NoInfsFPMath;
  Options.NoNaNsFPMath = CodeGenOpts.NoNaNsFPMath;
  Options.NoZerosInBSS = CodeGenOpts.NoZeroInitializedInBSS;
  Options.UnsafeFPMath = CodeGenOpts.UnsafeFPMath;
  Options.StackAlignmentOverride = CodeGenOpts.StackAlignment;
  Options.FunctionSections = CodeGenOpts.FunctionSections;
  Options.DataSections = CodeGenOpts.DataSections;
  Options.UniqueSectionNames = CodeGenOpts.UniqueSectionNames;
  Options.EmulatedTLS = CodeGenOpts.EmulatedTLS;
  Options.DebuggerTuning = CodeGenOpts.getDebuggerTuning();

  Options.MCOptions.MCRelaxAll = CodeGenOpts.RelaxAll;
  Options.MCOptions.MCSaveTempLabels = CodeGenOpts.SaveTempLabels;
  Options.MCOptions.MCUseDwarfDirectory = !CodeGenOpts.NoDwarfDirectoryAsm;
  Options.MCOptions.MCNoExecStack = CodeGenOpts.NoExecStack;
  Options.MCOptions.MCIncrementalLinkerCompatible =
      CodeGenOpts.IncrementalLinkerCompatible;
  Options.MCOptions.MCFatalWarnings = CodeGenOpts.FatalWarnings;
  Options.MCOptions.AsmVerbose = CodeGenOpts.AsmVerbose;
  Options.MCOptions.PreserveAsmComments = CodeGenOpts.PreserveAsmComments;
  Options.MCOptions.ABIName = TargetOpts.ABI;

  TM.reset(TheTarget->createTargetMachine(Triple, TargetOpts.CPU, FeaturesStr,
                                          Options, RM, CM, OptLevel));
}

bool EmitAssemblyHelper::AddEmitPasses(legacy::PassManager &CodeGenPasses,
                                       BackendAction Action,
                                       raw_pwrite_stream &OS) {
  // Add LibraryInfo.
  llvm::Triple TargetTriple(TheModule->getTargetTriple());
  std::unique_ptr<TargetLibraryInfoImpl> TLII(
      createTLII(TargetTriple, CodeGenOpts));
  CodeGenPasses.add(new TargetLibraryInfoWrapperPass(*TLII));

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
    CodeGenPasses.add(createObjCARCContractPass());

  if (TM->addPassesToEmitFile(CodeGenPasses, OS, CGFT,
                              /*DisableVerify=*/!CodeGenOpts.VerifyModule)) {
    Diags.Report(diag::err_fe_unable_to_interface_with_target);
    return false;
  }

  return true;
}

void EmitAssemblyHelper::EmitAssembly(BackendAction Action,
                                      std::unique_ptr<raw_pwrite_stream> OS) {
  TimeRegion Region(llvm::TimePassesIsEnabled ? &CodeGenerationTime : nullptr);

  setCommandLineOpts();

  bool UsesCodeGen = (Action != Backend_EmitNothing &&
                      Action != Backend_EmitBC &&
                      Action != Backend_EmitLL);
  CreateTargetMachine(UsesCodeGen);

  if (UsesCodeGen && !TM)
    return;
  if (TM)
    TheModule->setDataLayout(TM->createDataLayout());

  legacy::PassManager PerModulePasses;
  PerModulePasses.add(
      createTargetTransformInfoWrapperPass(getTargetIRAnalysis()));

  legacy::FunctionPassManager PerFunctionPasses(TheModule);
  PerFunctionPasses.add(
      createTargetTransformInfoWrapperPass(getTargetIRAnalysis()));

  CreatePasses(PerModulePasses, PerFunctionPasses);

  legacy::PassManager CodeGenPasses;
  CodeGenPasses.add(
      createTargetTransformInfoWrapperPass(getTargetIRAnalysis()));

  switch (Action) {
  case Backend_EmitNothing:
    break;

  case Backend_EmitBC:
    PerModulePasses.add(createBitcodeWriterPass(
        *OS, CodeGenOpts.EmitLLVMUseLists, CodeGenOpts.EmitSummaryIndex,
        CodeGenOpts.EmitSummaryIndex));
    break;

  case Backend_EmitLL:
    PerModulePasses.add(
        createPrintModulePass(*OS, "", CodeGenOpts.EmitLLVMUseLists));
    break;

  default:
    if (!AddEmitPasses(CodeGenPasses, Action, *OS))
      return;
  }

  // Before executing passes, print the final values of the LLVM options.
  cl::PrintOptionValues();

  // Run passes. For now we do all passes at once, but eventually we
  // would like to have the option of streaming code generation.

  {
    PrettyStackTraceString CrashInfo("Per-function optimization");

    PerFunctionPasses.doInitialization();
    for (Function &F : *TheModule)
      if (!F.isDeclaration())
        PerFunctionPasses.run(F);
    PerFunctionPasses.doFinalization();
  }

  {
    PrettyStackTraceString CrashInfo("Per-module optimization passes");
    PerModulePasses.run(*TheModule);
  }

  {
    PrettyStackTraceString CrashInfo("Code generation");
    CodeGenPasses.run(*TheModule);
  }
}

static void runThinLTOBackend(const CodeGenOptions &CGOpts, Module *M,
                              std::unique_ptr<raw_pwrite_stream> OS) {
  // If we are performing a ThinLTO importing compile, load the function index
  // into memory and pass it into thinBackend, which will run the function
  // importer and invoke LTO passes.
  ErrorOr<std::unique_ptr<ModuleSummaryIndex>> IndexOrErr =
      llvm::getModuleSummaryIndexForFile(
          CGOpts.ThinLTOIndexFile,
          [&](const DiagnosticInfo &DI) { M->getContext().diagnose(DI); });
  if (std::error_code EC = IndexOrErr.getError()) {
    std::string Error = EC.message();
    errs() << "Error loading index file '" << CGOpts.ThinLTOIndexFile
           << "': " << Error << "\n";
    return;
  }
  std::unique_ptr<ModuleSummaryIndex> CombinedIndex = std::move(*IndexOrErr);

  auto AddStream = [&](size_t Task) { return std::move(OS); };

  StringMap<std::map<GlobalValue::GUID, GlobalValueSummary *>>
      ModuleToDefinedGVSummaries;
  CombinedIndex->collectDefinedGVSummariesPerModule(ModuleToDefinedGVSummaries);

  // FIXME: We could simply import the modules mentioned in the combined index
  // here.
  FunctionImporter::ImportMapTy ImportList;
  ComputeCrossModuleImportForModule(M->getModuleIdentifier(), *CombinedIndex,
                                    ImportList);

  std::vector<std::unique_ptr<llvm::MemoryBuffer>> OwnedImports;
  MapVector<llvm::StringRef, llvm::MemoryBufferRef> ModuleMap;

  for (auto &I : ImportList) {
    ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> MBOrErr =
        llvm::MemoryBuffer::getFile(I.first());
    if (!MBOrErr) {
      errs() << "Error loading imported file '" << I.first()
             << "': " << MBOrErr.getError().message() << "\n";
      return;
    }
    ModuleMap[I.first()] = (*MBOrErr)->getMemBufferRef();
    OwnedImports.push_back(std::move(*MBOrErr));
  }

  lto::Config Conf;
  if (Error E = thinBackend(
          Conf, 0, AddStream, *M, *CombinedIndex, ImportList,
          ModuleToDefinedGVSummaries[M->getModuleIdentifier()], ModuleMap)) {
    handleAllErrors(std::move(E), [&](ErrorInfoBase &EIB) {
      errs() << "Error running ThinLTO backend: " << EIB.message() << '\n';
    });
  }
}

void clang::EmitBackendOutput(DiagnosticsEngine &Diags,
                              const CodeGenOptions &CGOpts,
                              const clang::TargetOptions &TOpts,
                              const LangOptions &LOpts, const llvm::DataLayout &TDesc,
                              Module *M, BackendAction Action,
                              std::unique_ptr<raw_pwrite_stream> OS) {
  if (!CGOpts.ThinLTOIndexFile.empty()) {
    runThinLTOBackend(CGOpts, M, std::move(OS));
    return;
  }

  EmitAssemblyHelper AsmHelper(Diags, CGOpts, TOpts, LOpts, M);

  AsmHelper.EmitAssembly(Action, std::move(OS));

  // Verify clang's TargetInfo DataLayout against the LLVM TargetMachine's
  // DataLayout.
  if (AsmHelper.TM) {
    std::string DLDesc = M->getDataLayout().getStringRepresentation();
    if (DLDesc != TDesc.getStringRepresentation()) {
      unsigned DiagID = Diags.getCustomDiagID(
          DiagnosticsEngine::Error, "backend data layout '%0' does not match "
                                    "expected target description '%1'");
      Diags.Report(DiagID) << DLDesc << TDesc.getStringRepresentation();
    }
  }
}

static const char* getSectionNameForBitcode(const Triple &T) {
  switch (T.getObjectFormat()) {
  case Triple::MachO:
    return "__LLVM,__bitcode";
  case Triple::COFF:
  case Triple::ELF:
  case Triple::UnknownObjectFormat:
    return ".llvmbc";
  }
  llvm_unreachable("Unimplemented ObjectFormatType");
}

static const char* getSectionNameForCommandline(const Triple &T) {
  switch (T.getObjectFormat()) {
  case Triple::MachO:
    return "__LLVM,__cmdline";
  case Triple::COFF:
  case Triple::ELF:
  case Triple::UnknownObjectFormat:
    return ".llvmcmd";
  }
  llvm_unreachable("Unimplemented ObjectFormatType");
}

// With -fembed-bitcode, save a copy of the llvm IR as data in the
// __LLVM,__bitcode section.
void clang::EmbedBitcode(llvm::Module *M, const CodeGenOptions &CGOpts,
                         llvm::MemoryBufferRef Buf) {
  if (CGOpts.getEmbedBitcode() == CodeGenOptions::Embed_Off)
    return;

  // Save llvm.compiler.used and remote it.
  SmallVector<Constant*, 2> UsedArray;
  SmallSet<GlobalValue*, 4> UsedGlobals;
  Type *UsedElementType = Type::getInt8Ty(M->getContext())->getPointerTo(0);
  GlobalVariable *Used = collectUsedGlobalVariables(*M, UsedGlobals, true);
  for (auto *GV : UsedGlobals) {
    if (GV->getName() != "llvm.embedded.module" &&
        GV->getName() != "llvm.cmdline")
      UsedArray.push_back(
          ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV, UsedElementType));
  }
  if (Used)
    Used->eraseFromParent();

  // Embed the bitcode for the llvm module.
  std::string Data;
  ArrayRef<uint8_t> ModuleData;
  Triple T(M->getTargetTriple());
  // Create a constant that contains the bitcode.
  // In case of embedding a marker, ignore the input Buf and use the empty
  // ArrayRef. It is also legal to create a bitcode marker even Buf is empty.
  if (CGOpts.getEmbedBitcode() != CodeGenOptions::Embed_Marker) {
    if (!isBitcode((const unsigned char *)Buf.getBufferStart(),
                   (const unsigned char *)Buf.getBufferEnd())) {
      // If the input is LLVM Assembly, bitcode is produced by serializing
      // the module. Use-lists order need to be perserved in this case.
      llvm::raw_string_ostream OS(Data);
      llvm::WriteBitcodeToFile(M, OS, /* ShouldPreserveUseListOrder */ true);
      ModuleData =
          ArrayRef<uint8_t>((const uint8_t *)OS.str().data(), OS.str().size());
    } else
      // If the input is LLVM bitcode, write the input byte stream directly.
      ModuleData = ArrayRef<uint8_t>((const uint8_t *)Buf.getBufferStart(),
                                     Buf.getBufferSize());
  }
  llvm::Constant *ModuleConstant =
      llvm::ConstantDataArray::get(M->getContext(), ModuleData);
  llvm::GlobalVariable *GV = new llvm::GlobalVariable(
      *M, ModuleConstant->getType(), true, llvm::GlobalValue::PrivateLinkage,
      ModuleConstant);
  GV->setSection(getSectionNameForBitcode(T));
  UsedArray.push_back(
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV, UsedElementType));
  if (llvm::GlobalVariable *Old =
          M->getGlobalVariable("llvm.embedded.module", true)) {
    assert(Old->hasOneUse() &&
           "llvm.embedded.module can only be used once in llvm.compiler.used");
    GV->takeName(Old);
    Old->eraseFromParent();
  } else {
    GV->setName("llvm.embedded.module");
  }

  // Skip if only bitcode needs to be embedded.
  if (CGOpts.getEmbedBitcode() != CodeGenOptions::Embed_Bitcode) {
    // Embed command-line options.
    ArrayRef<uint8_t> CmdData(const_cast<uint8_t *>(CGOpts.CmdArgs.data()),
                              CGOpts.CmdArgs.size());
    llvm::Constant *CmdConstant =
      llvm::ConstantDataArray::get(M->getContext(), CmdData);
    GV = new llvm::GlobalVariable(*M, CmdConstant->getType(), true,
                                  llvm::GlobalValue::PrivateLinkage,
                                  CmdConstant);
    GV->setSection(getSectionNameForCommandline(T));
    UsedArray.push_back(
        ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV, UsedElementType));
    if (llvm::GlobalVariable *Old =
            M->getGlobalVariable("llvm.cmdline", true)) {
      assert(Old->hasOneUse() &&
             "llvm.cmdline can only be used once in llvm.compiler.used");
      GV->takeName(Old);
      Old->eraseFromParent();
    } else {
      GV->setName("llvm.cmdline");
    }
  }

  if (UsedArray.empty())
    return;

  // Recreate llvm.compiler.used.
  ArrayType *ATy = ArrayType::get(UsedElementType, UsedArray.size());
  auto *NewUsed = new GlobalVariable(
      *M, ATy, false, llvm::GlobalValue::AppendingLinkage,
      llvm::ConstantArray::get(ATy, UsedArray), "llvm.compiler.used");
  NewUsed->setSection("llvm.metadata");
}
