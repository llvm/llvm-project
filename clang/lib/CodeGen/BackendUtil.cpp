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
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Scalar.h"
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

  mutable PassManager *CodeGenPasses;
  mutable PassManager *PerModulePasses;
  mutable FunctionPassManager *PerFunctionPasses;

private:
  PassManager *getCodeGenPasses() const {
    if (!CodeGenPasses) {
      CodeGenPasses = new PassManager();
      CodeGenPasses->add(new DataLayoutPass(TheModule));
      if (TM)
        TM->addAnalysisPasses(*CodeGenPasses);
    }
    return CodeGenPasses;
  }

  PassManager *getPerModulePasses() const {
    if (!PerModulePasses) {
      PerModulePasses = new PassManager();
      PerModulePasses->add(new DataLayoutPass(TheModule));
      if (TM)
        TM->addAnalysisPasses(*PerModulePasses);
    }
    return PerModulePasses;
  }

  FunctionPassManager *getPerFunctionPasses() const {
    if (!PerFunctionPasses) {
      PerFunctionPasses = new FunctionPassManager(TheModule);
      PerFunctionPasses->add(new DataLayoutPass(TheModule));
      if (TM)
        TM->addAnalysisPasses(*PerFunctionPasses);
    }
    return PerFunctionPasses;
  }

  void CreatePasses();

  /// CreateTargetMachine - Generates the TargetMachine.
  /// Returns Null if it is unable to create the target machine.
  /// Some of our clang tests specify triples which are not built
  /// into clang. This is okay because these tests check the generated
  /// IR, and they require DataLayout which depends on the triple.
  /// In this case, we allow this method to fail and not report an error.
  /// When MustCreateTM is used, we print an error if we are unable to load
  /// the requested target.
  TargetMachine *CreateTargetMachine(bool MustCreateTM);

  /// AddEmitPasses - Add passes necessary to emit assembly or LLVM IR.
  ///
  /// \return True on success.
  bool AddEmitPasses(BackendAction Action, formatted_raw_ostream &OS);

public:
  EmitAssemblyHelper(DiagnosticsEngine &_Diags,
                     const CodeGenOptions &CGOpts,
                     const clang::TargetOptions &TOpts,
                     const LangOptions &LOpts,
                     Module *M)
    : Diags(_Diags), CodeGenOpts(CGOpts), TargetOpts(TOpts), LangOpts(LOpts),
      TheModule(M), CodeGenerationTime("Code Generation Time"),
      CodeGenPasses(nullptr), PerModulePasses(nullptr),
      PerFunctionPasses(nullptr) {}

  ~EmitAssemblyHelper() {
    delete CodeGenPasses;
    delete PerModulePasses;
    delete PerFunctionPasses;
    if (CodeGenOpts.DisableFree)
      BuryPointer(TM.release());
  }

  std::unique_ptr<TargetMachine> TM;

  void EmitAssembly(BackendAction Action, raw_ostream *OS);
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

static void addSampleProfileLoaderPass(const PassManagerBuilder &Builder,
                                       PassManagerBase &PM) {
  const PassManagerBuilderWrapper &BuilderWrapper =
      static_cast<const PassManagerBuilderWrapper &>(Builder);
  const CodeGenOptions &CGOpts = BuilderWrapper.getCGOpts();
  PM.add(createSampleProfileLoaderPass(CGOpts.SampleProfileFile));
}

static void addAddDiscriminatorsPass(const PassManagerBuilder &Builder,
                                     PassManagerBase &PM) {
  PM.add(createAddDiscriminatorsPass());
}

static void addBoundsCheckingPass(const PassManagerBuilder &Builder,
                                    PassManagerBase &PM) {
  PM.add(createBoundsCheckingPass());
}

static void addAddressSanitizerPasses(const PassManagerBuilder &Builder,
                                      PassManagerBase &PM) {
  PM.add(createAddressSanitizerFunctionPass());
  PM.add(createAddressSanitizerModulePass());
}

static void addMemorySanitizerPass(const PassManagerBuilder &Builder,
                                   PassManagerBase &PM) {
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
                                   PassManagerBase &PM) {
  PM.add(createThreadSanitizerPass());
}

static void addDataFlowSanitizerPass(const PassManagerBuilder &Builder,
                                     PassManagerBase &PM) {
  const PassManagerBuilderWrapper &BuilderWrapper =
      static_cast<const PassManagerBuilderWrapper&>(Builder);
  const CodeGenOptions &CGOpts = BuilderWrapper.getCGOpts();
  PM.add(createDataFlowSanitizerPass(CGOpts.SanitizerBlacklistFile));
}

void EmitAssemblyHelper::CreatePasses() {
  unsigned OptLevel = CodeGenOpts.OptimizationLevel;
  CodeGenOptions::InliningMethod Inlining = CodeGenOpts.getInlining();

  // Handle disabling of LLVM optimization, where we want to preserve the
  // internal module before any optimization.
  if (CodeGenOpts.DisableLLVMOpts) {
    OptLevel = 0;
    Inlining = CodeGenOpts.NoInlining;
  }

  PassManagerBuilderWrapper PMBuilder(CodeGenOpts, LangOpts);
  PMBuilder.OptLevel = OptLevel;
  PMBuilder.SizeLevel = CodeGenOpts.OptimizeSize;
  PMBuilder.BBVectorize = CodeGenOpts.VectorizeBB;
  PMBuilder.SLPVectorize = CodeGenOpts.VectorizeSLP;
  PMBuilder.LoopVectorize = CodeGenOpts.VectorizeLoop;

  PMBuilder.DisableTailCalls = CodeGenOpts.DisableTailCalls;
  PMBuilder.DisableUnitAtATime = !CodeGenOpts.UnitAtATime;
  PMBuilder.DisableUnrollLoops = !CodeGenOpts.UnrollLoops;
  PMBuilder.RerollLoops = CodeGenOpts.RerollLoops;

  PMBuilder.addExtension(PassManagerBuilder::EP_EarlyAsPossible,
                         addAddDiscriminatorsPass);

  if (!CodeGenOpts.SampleProfileFile.empty())
    PMBuilder.addExtension(PassManagerBuilder::EP_EarlyAsPossible,
                           addSampleProfileLoaderPass);

  // In ObjC ARC mode, add the main ARC optimization passes.
  if (LangOpts.ObjCAutoRefCount) {
    PMBuilder.addExtension(PassManagerBuilder::EP_EarlyAsPossible,
                           addObjCARCExpandPass);
    PMBuilder.addExtension(PassManagerBuilder::EP_ModuleOptimizerEarly,
                           addObjCARCAPElimPass);
    PMBuilder.addExtension(PassManagerBuilder::EP_ScalarOptimizerLate,
                           addObjCARCOptPass);
  }

  if (LangOpts.Sanitize.LocalBounds) {
    PMBuilder.addExtension(PassManagerBuilder::EP_ScalarOptimizerLate,
                           addBoundsCheckingPass);
    PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                           addBoundsCheckingPass);
  }

  if (LangOpts.Sanitize.Address) {
    PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                           addAddressSanitizerPasses);
    PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                           addAddressSanitizerPasses);
  }

  if (LangOpts.Sanitize.Memory) {
    PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                           addMemorySanitizerPass);
    PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                           addMemorySanitizerPass);
  }

  if (LangOpts.Sanitize.Thread) {
    PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                           addThreadSanitizerPass);
    PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                           addThreadSanitizerPass);
  }

  if (LangOpts.Sanitize.DataFlow) {
    PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                           addDataFlowSanitizerPass);
    PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                           addDataFlowSanitizerPass);
  }

  // Figure out TargetLibraryInfo.
  Triple TargetTriple(TheModule->getTargetTriple());
  PMBuilder.LibraryInfo = new TargetLibraryInfo(TargetTriple);
  if (!CodeGenOpts.SimplifyLibCalls)
    PMBuilder.LibraryInfo->disableAllFunctions();

  switch (Inlining) {
  case CodeGenOptions::NoInlining: break;
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

  // Set up the per-function pass manager.
  FunctionPassManager *FPM = getPerFunctionPasses();
  if (CodeGenOpts.VerifyModule)
    FPM->add(createVerifierPass());
  PMBuilder.populateFunctionPassManager(*FPM);

  // Set up the per-module pass manager.
  PassManager *MPM = getPerModulePasses();
  if (CodeGenOpts.VerifyModule)
    MPM->add(createDebugInfoVerifierPass());

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
    MPM->add(createGCOVProfilerPass(Options));
    if (CodeGenOpts.getDebugInfo() == CodeGenOptions::NoDebugInfo)
      MPM->add(createStripSymbolsPass(true));
  }

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
  if (llvm::TimePassesIsEnabled)
    BackendArgs.push_back("-time-passes");
  for (unsigned i = 0, e = CodeGenOpts.BackendOptions.size(); i != e; ++i)
    BackendArgs.push_back(CodeGenOpts.BackendOptions[i].c_str());
  if (CodeGenOpts.NoGlobalMerge)
    BackendArgs.push_back("-enable-global-merge=false");
  BackendArgs.push_back(nullptr);
  llvm::cl::ParseCommandLineOptions(BackendArgs.size() - 1,
                                    BackendArgs.data());

  std::string FeaturesStr;
  if (TargetOpts.Features.size()) {
    SubtargetFeatures Features;
    for (std::vector<std::string>::const_iterator
           it = TargetOpts.Features.begin(),
           ie = TargetOpts.Features.end(); it != ie; ++it)
      Features.AddFeature(*it);
    FeaturesStr = Features.getString();
  }

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

  if (CodeGenOpts.DisableIntegratedAS)
    Options.DisableIntegratedAS = true;

  if (CodeGenOpts.CompressDebugSections)
    Options.CompressDebugSections = true;

  // Set frame pointer elimination mode.
  if (!CodeGenOpts.DisableFPElim) {
    Options.NoFramePointerElim = false;
  } else if (CodeGenOpts.OmitLeafFramePointer) {
    Options.NoFramePointerElim = false;
  } else {
    Options.NoFramePointerElim = true;
  }

  if (CodeGenOpts.UseInitArray)
    Options.UseInitArray = true;

  // Set float ABI type.
  if (CodeGenOpts.FloatABI == "soft" || CodeGenOpts.FloatABI == "softfp")
    Options.FloatABIType = llvm::FloatABI::Soft;
  else if (CodeGenOpts.FloatABI == "hard")
    Options.FloatABIType = llvm::FloatABI::Hard;
  else {
    assert(CodeGenOpts.FloatABI.empty() && "Invalid float abi!");
    Options.FloatABIType = llvm::FloatABI::Default;
  }

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

  Options.LessPreciseFPMADOption = CodeGenOpts.LessPreciseFPMAD;
  Options.NoInfsFPMath = CodeGenOpts.NoInfsFPMath;
  Options.NoNaNsFPMath = CodeGenOpts.NoNaNsFPMath;
  Options.NoZerosInBSS = CodeGenOpts.NoZeroInitializedInBSS;
  Options.UnsafeFPMath = CodeGenOpts.UnsafeFPMath;
  Options.UseSoftFloat = CodeGenOpts.SoftFloat;
  Options.StackAlignmentOverride = CodeGenOpts.StackAlignment;
  Options.DisableTailCalls = CodeGenOpts.DisableTailCalls;
  Options.TrapFuncName = CodeGenOpts.TrapFuncName;
  Options.PositionIndependentExecutable = LangOpts.PIELevel != 0;
  Options.FunctionSections = CodeGenOpts.FunctionSections;
  Options.DataSections = CodeGenOpts.DataSections;

  Options.MCOptions.MCRelaxAll = CodeGenOpts.RelaxAll;
  Options.MCOptions.MCSaveTempLabels = CodeGenOpts.SaveTempLabels;
  Options.MCOptions.MCUseDwarfDirectory = !CodeGenOpts.NoDwarfDirectoryAsm;
  Options.MCOptions.MCNoExecStack = CodeGenOpts.NoExecStack;
  Options.MCOptions.AsmVerbose = CodeGenOpts.AsmVerbose;

  TargetMachine *TM = TheTarget->createTargetMachine(Triple, TargetOpts.CPU,
                                                     FeaturesStr, Options,
                                                     RM, CM, OptLevel);

  return TM;
}

bool EmitAssemblyHelper::AddEmitPasses(BackendAction Action,
                                       formatted_raw_ostream &OS) {

  // Create the code generator passes.
  PassManager *PM = getCodeGenPasses();

  // Add LibraryInfo.
  llvm::Triple TargetTriple(TheModule->getTargetTriple());
  TargetLibraryInfo *TLI = new TargetLibraryInfo(TargetTriple);
  if (!CodeGenOpts.SimplifyLibCalls)
    TLI->disableAllFunctions();
  PM->add(TLI);

  // Add Target specific analysis passes.
  TM->addAnalysisPasses(*PM);

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
  if (LangOpts.ObjCAutoRefCount &&
      CodeGenOpts.OptimizationLevel > 0)
    PM->add(createObjCARCContractPass());

  if (TM->addPassesToEmitFile(*PM, OS, CGFT,
                              /*DisableVerify=*/!CodeGenOpts.VerifyModule)) {
    Diags.Report(diag::err_fe_unable_to_interface_with_target);
    return false;
  }

  return true;
}

void EmitAssemblyHelper::EmitAssembly(BackendAction Action, raw_ostream *OS) {
  TimeRegion Region(llvm::TimePassesIsEnabled ? &CodeGenerationTime : nullptr);
  llvm::formatted_raw_ostream FormattedOS;

  bool UsesCodeGen = (Action != Backend_EmitNothing &&
                      Action != Backend_EmitBC &&
                      Action != Backend_EmitLL);
  if (!TM)
    TM.reset(CreateTargetMachine(UsesCodeGen));

  if (UsesCodeGen && !TM) return;
  CreatePasses();

  switch (Action) {
  case Backend_EmitNothing:
    break;

  case Backend_EmitBC:
    getPerModulePasses()->add(createBitcodeWriterPass(*OS));
    break;

  case Backend_EmitLL:
    FormattedOS.setStream(*OS, formatted_raw_ostream::PRESERVE_STREAM);
    getPerModulePasses()->add(createPrintModulePass(FormattedOS));
    break;

  default:
    FormattedOS.setStream(*OS, formatted_raw_ostream::PRESERVE_STREAM);
    if (!AddEmitPasses(Action, FormattedOS))
      return;
  }

  // Before executing passes, print the final values of the LLVM options.
  cl::PrintOptionValues();

  // Run passes. For now we do all passes at once, but eventually we
  // would like to have the option of streaming code generation.

  if (PerFunctionPasses) {
    PrettyStackTraceString CrashInfo("Per-function optimization");

    PerFunctionPasses->doInitialization();
    for (Module::iterator I = TheModule->begin(),
           E = TheModule->end(); I != E; ++I)
      if (!I->isDeclaration())
        PerFunctionPasses->run(*I);
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
                              raw_ostream *OS) {
  EmitAssemblyHelper AsmHelper(Diags, CGOpts, TOpts, LOpts, M);

  AsmHelper.EmitAssembly(Action, OS);

  // If an optional clang TargetInfo description string was passed in, use it to
  // verify the LLVM TargetMachine's DataLayout.
  if (AsmHelper.TM && !TDesc.empty()) {
    std::string DLDesc =
        AsmHelper.TM->getDataLayout()->getStringRepresentation();
    if (DLDesc != TDesc) {
      unsigned DiagID = Diags.getCustomDiagID(
          DiagnosticsEngine::Error, "backend data layout '%0' does not match "
                                    "expected target description '%1'");
      Diags.Report(DiagID) << DLDesc << TDesc;
    }
  }
}
