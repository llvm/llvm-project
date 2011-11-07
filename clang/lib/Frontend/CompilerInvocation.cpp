//===--- CompilerInvocation.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/Version.h"
#include "clang/Basic/FileManager.h"
#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/CC1Options.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/OptTable.h"
#include "clang/Driver/Option.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/LangStandard.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
using namespace clang;

static const char *getAnalysisStoreName(AnalysisStores Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown analysis store!");
#define ANALYSIS_STORE(NAME, CMDFLAG, DESC, CREATFN) \
  case NAME##Model: return CMDFLAG;
#include "clang/Frontend/Analyses.def"
  }
}

static const char *getAnalysisConstraintName(AnalysisConstraints Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown analysis constraints!");
#define ANALYSIS_CONSTRAINTS(NAME, CMDFLAG, DESC, CREATFN) \
  case NAME##Model: return CMDFLAG;
#include "clang/Frontend/Analyses.def"
  }
}

static const char *getAnalysisDiagClientName(AnalysisDiagClients Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown analysis client!");
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATFN, AUTOCREATE) \
  case PD_##NAME: return CMDFLAG;
#include "clang/Frontend/Analyses.def"
  }
}

static const char *getAnalysisPurgeModeName(AnalysisPurgeMode Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown analysis client!");
#define ANALYSIS_PURGE(NAME, CMDFLAG, DESC) \
  case NAME: return CMDFLAG;
#include "clang/Frontend/Analyses.def"
  }
}

//===----------------------------------------------------------------------===//
// Serialization (to args)
//===----------------------------------------------------------------------===//

static void AnalyzerOptsToArgs(const AnalyzerOptions &Opts,
                               std::vector<std::string> &Res) {
  if (Opts.ShowCheckerHelp)
    Res.push_back("-analyzer-checker-help");
  if (Opts.AnalysisStoreOpt != RegionStoreModel) {
    Res.push_back("-analyzer-store");
    Res.push_back(getAnalysisStoreName(Opts.AnalysisStoreOpt));
  }
  if (Opts.AnalysisConstraintsOpt != RangeConstraintsModel) {
    Res.push_back("-analyzer-constraints");
    Res.push_back(getAnalysisConstraintName(Opts.AnalysisConstraintsOpt));
  }
  if (Opts.AnalysisDiagOpt != PD_HTML) {
    Res.push_back("-analyzer-output");
    Res.push_back(getAnalysisDiagClientName(Opts.AnalysisDiagOpt));
  }
  if (Opts.AnalysisPurgeOpt != PurgeStmt) {
    Res.push_back("-analyzer-purge");
    Res.push_back(getAnalysisPurgeModeName(Opts.AnalysisPurgeOpt));
  }
  if (!Opts.AnalyzeSpecificFunction.empty()) {
    Res.push_back("-analyze-function");
    Res.push_back(Opts.AnalyzeSpecificFunction);
  }
  if (Opts.AnalyzeAll)
    Res.push_back("-analyzer-opt-analyze-headers");
  if (Opts.AnalyzerDisplayProgress)
    Res.push_back("-analyzer-display-progress");
  if (Opts.AnalyzeNestedBlocks)
    Res.push_back("-analyzer-opt-analyze-nested-blocks");
  if (Opts.EagerlyAssume)
    Res.push_back("-analyzer-eagerly-assume");
  if (Opts.TrimGraph)
    Res.push_back("-trim-egraph");
  if (Opts.VisualizeEGDot)
    Res.push_back("-analyzer-viz-egraph-graphviz");
  if (Opts.VisualizeEGUbi)
    Res.push_back("-analyzer-viz-egraph-ubigraph");

  for (unsigned i = 0, e = Opts.CheckersControlList.size(); i != e; ++i) {
    const std::pair<std::string, bool> &opt = Opts.CheckersControlList[i];
    if (opt.second)
      Res.push_back("-analyzer-disable-checker");
    else
      Res.push_back("-analyzer-checker");
    Res.push_back(opt.first);
  }
}

static void CodeGenOptsToArgs(const CodeGenOptions &Opts,
                              std::vector<std::string> &Res) {
  if (Opts.DebugInfo)
    Res.push_back("-g");
  if (Opts.DisableLLVMOpts)
    Res.push_back("-disable-llvm-optzns");
  if (Opts.DisableRedZone)
    Res.push_back("-disable-red-zone");
  if (!Opts.DwarfDebugFlags.empty()) {
    Res.push_back("-dwarf-debug-flags");
    Res.push_back(Opts.DwarfDebugFlags);
  }
  if (Opts.ObjCRuntimeHasARC)
    Res.push_back("-fobjc-runtime-has-arc");
  if (Opts.ObjCRuntimeHasTerminate)
    Res.push_back("-fobjc-runtime-has-terminate");
  if (Opts.EmitGcovArcs)
    Res.push_back("-femit-coverage-data");
  if (Opts.EmitGcovNotes)
    Res.push_back("-femit-coverage-notes");
  if (!Opts.MergeAllConstants)
    Res.push_back("-fno-merge-all-constants");
  if (Opts.NoCommon)
    Res.push_back("-fno-common");
  if (Opts.ForbidGuardVariables)
    Res.push_back("-fforbid-guard-variables");
  if (Opts.UseRegisterSizedBitfieldAccess)
    Res.push_back("-fuse-register-sized-bitfield-access");
  if (Opts.NoImplicitFloat)
    Res.push_back("-no-implicit-float");
  if (Opts.OmitLeafFramePointer)
    Res.push_back("-momit-leaf-frame-pointer");
  if (Opts.OptimizeSize) {
    assert(Opts.OptimizationLevel == 2 && "Invalid options!");
    Opts.OptimizeSize == 1 ? Res.push_back("-Os") : Res.push_back("-Oz");
  } else if (Opts.OptimizationLevel != 0)
    Res.push_back("-O" + llvm::utostr(Opts.OptimizationLevel));
  if (!Opts.MainFileName.empty()) {
    Res.push_back("-main-file-name");
    Res.push_back(Opts.MainFileName);
  }
  // SimplifyLibCalls is only derived.
  // TimePasses is only derived.
  // UnitAtATime is unused.
  // Inlining is only derived.

  // UnrollLoops is derived, but also accepts an option, no
  // harm in pushing it back here.
  if (Opts.UnrollLoops)
    Res.push_back("-funroll-loops");
  if (Opts.DataSections)
    Res.push_back("-fdata-sections");
  if (Opts.FunctionSections)
    Res.push_back("-ffunction-sections");
  if (Opts.AsmVerbose)
    Res.push_back("-masm-verbose");
  if (!Opts.CodeModel.empty()) {
    Res.push_back("-mcode-model");
    Res.push_back(Opts.CodeModel);
  }
  if (Opts.CUDAIsDevice)
    Res.push_back("-fcuda-is-device");
  if (!Opts.CXAAtExit)
    Res.push_back("-fno-use-cxa-atexit");
  if (Opts.CXXCtorDtorAliases)
    Res.push_back("-mconstructor-aliases");
  if (Opts.ObjCAutoRefCountExceptions)
    Res.push_back("-fobjc-arc-eh");
  if (!Opts.DebugPass.empty()) {
    Res.push_back("-mdebug-pass");
    Res.push_back(Opts.DebugPass);
  }
  if (Opts.DisableFPElim)
    Res.push_back("-mdisable-fp-elim");
  if (!Opts.FloatABI.empty()) {
    Res.push_back("-mfloat-abi");
    Res.push_back(Opts.FloatABI);
  }
  if (!Opts.LimitFloatPrecision.empty()) {
    Res.push_back("-mlimit-float-precision");
    Res.push_back(Opts.LimitFloatPrecision);
  }
  if (Opts.NoZeroInitializedInBSS)
    Res.push_back("-mno-zero-initialized-bss");
  switch (Opts.getObjCDispatchMethod()) {
  case CodeGenOptions::Legacy:
    break;
  case CodeGenOptions::Mixed:
    Res.push_back("-fobjc-dispatch-method=mixed");
    break;
  case CodeGenOptions::NonLegacy:
    Res.push_back("-fobjc-dispatch-method=non-legacy");
    break;
  }
  if (Opts.NumRegisterParameters) {
    Res.push_back("-mregparm");
    Res.push_back(llvm::utostr(Opts.NumRegisterParameters));
  }
  if (Opts.NoGlobalMerge)
    Res.push_back("-mno-global-merge");
  if (Opts.NoExecStack)
    Res.push_back("-mnoexecstack");
  if (Opts.RelaxAll)
    Res.push_back("-mrelax-all");
  if (Opts.SaveTempLabels)
    Res.push_back("-msave-temp-labels");
  if (Opts.NoDwarf2CFIAsm)
    Res.push_back("-fno-dwarf2-cfi-asm");
  if (Opts.SoftFloat)
    Res.push_back("-msoft-float");
  if (Opts.UnwindTables)
    Res.push_back("-munwind-tables");
  if (Opts.RelocationModel != "pic") {
    Res.push_back("-mrelocation-model");
    Res.push_back(Opts.RelocationModel);
  }
  if (!Opts.VerifyModule)
    Res.push_back("-disable-llvm-verifier");
  for (unsigned i = 0, e = Opts.BackendOptions.size(); i != e; ++i) {
    Res.push_back("-backend-option");
    Res.push_back(Opts.BackendOptions[i]);
  }
}

static void DependencyOutputOptsToArgs(const DependencyOutputOptions &Opts,
                                       std::vector<std::string> &Res) {
  if (Opts.IncludeSystemHeaders)
    Res.push_back("-sys-header-deps");
  if (Opts.ShowHeaderIncludes)
    Res.push_back("-H");
  if (!Opts.HeaderIncludeOutputFile.empty()) {
    Res.push_back("-header-include-file");
    Res.push_back(Opts.HeaderIncludeOutputFile);
  }
  if (Opts.UsePhonyTargets)
    Res.push_back("-MP");
  if (!Opts.OutputFile.empty()) {
    Res.push_back("-dependency-file");
    Res.push_back(Opts.OutputFile);
  }
  for (unsigned i = 0, e = Opts.Targets.size(); i != e; ++i) {
    Res.push_back("-MT");
    Res.push_back(Opts.Targets[i]);
  }
}

static void DiagnosticOptsToArgs(const DiagnosticOptions &Opts,
                                 std::vector<std::string> &Res) {
  if (Opts.IgnoreWarnings)
    Res.push_back("-w");
  if (Opts.NoRewriteMacros)
    Res.push_back("-Wno-rewrite-macros");
  if (Opts.Pedantic)
    Res.push_back("-pedantic");
  if (Opts.PedanticErrors)
    Res.push_back("-pedantic-errors");
  if (!Opts.ShowColumn)
    Res.push_back("-fno-show-column");
  if (!Opts.ShowLocation)
    Res.push_back("-fno-show-source-location");
  if (!Opts.ShowCarets)
    Res.push_back("-fno-caret-diagnostics");
  if (!Opts.ShowFixits)
    Res.push_back("-fno-diagnostics-fixit-info");
  if (Opts.ShowSourceRanges)
    Res.push_back("-fdiagnostics-print-source-range-info");
  if (Opts.ShowParseableFixits)
    Res.push_back("-fdiagnostics-parseable-fixits");
  if (Opts.ShowColors)
    Res.push_back("-fcolor-diagnostics");
  if (Opts.VerifyDiagnostics)
    Res.push_back("-verify");
  if (Opts.ShowNames)
    Res.push_back("-fdiagnostics-show-name");
  if (Opts.ShowOptionNames)
    Res.push_back("-fdiagnostics-show-option");
  if (Opts.ShowCategories == 1)
    Res.push_back("-fdiagnostics-show-category=id");
  else if (Opts.ShowCategories == 2)
    Res.push_back("-fdiagnostics-show-category=name");
  switch (Opts.Format) {
  case DiagnosticOptions::Clang: 
    Res.push_back("-fdiagnostics-format=clang"); break;
  case DiagnosticOptions::Msvc:  
    Res.push_back("-fdiagnostics-format=msvc");  break;
  case DiagnosticOptions::Vi:    
    Res.push_back("-fdiagnostics-format=vi");    break;
  }
  if (Opts.ErrorLimit) {
    Res.push_back("-ferror-limit");
    Res.push_back(llvm::utostr(Opts.ErrorLimit));
  }
  if (!Opts.DiagnosticLogFile.empty()) {
    Res.push_back("-diagnostic-log-file");
    Res.push_back(Opts.DiagnosticLogFile);
  }
  if (Opts.MacroBacktraceLimit
                        != DiagnosticOptions::DefaultMacroBacktraceLimit) {
    Res.push_back("-fmacro-backtrace-limit");
    Res.push_back(llvm::utostr(Opts.MacroBacktraceLimit));
  }
  if (Opts.TemplateBacktraceLimit
                        != DiagnosticOptions::DefaultTemplateBacktraceLimit) {
    Res.push_back("-ftemplate-backtrace-limit");
    Res.push_back(llvm::utostr(Opts.TemplateBacktraceLimit));
  }

  if (Opts.TabStop != DiagnosticOptions::DefaultTabStop) {
    Res.push_back("-ftabstop");
    Res.push_back(llvm::utostr(Opts.TabStop));
  }
  if (Opts.MessageLength) {
    Res.push_back("-fmessage-length");
    Res.push_back(llvm::utostr(Opts.MessageLength));
  }
  if (!Opts.DumpBuildInformation.empty()) {
    Res.push_back("-dump-build-information");
    Res.push_back(Opts.DumpBuildInformation);
  }
  for (unsigned i = 0, e = Opts.Warnings.size(); i != e; ++i)
    Res.push_back("-W" + Opts.Warnings[i]);
}

static const char *getInputKindName(InputKind Kind) {
  switch (Kind) {
  case IK_None:              break;
  case IK_AST:               return "ast";
  case IK_Asm:               return "assembler-with-cpp";
  case IK_C:                 return "c";
  case IK_CXX:               return "c++";
  case IK_LLVM_IR:           return "ir";
  case IK_ObjC:              return "objective-c";
  case IK_ObjCXX:            return "objective-c++";
  case IK_OpenCL:            return "cl";
  case IK_CUDA:              return "cuda";
  case IK_PreprocessedC:     return "cpp-output";
  case IK_PreprocessedCXX:   return "c++-cpp-output";
  case IK_PreprocessedObjC:  return "objective-c-cpp-output";
  case IK_PreprocessedObjCXX:return "objective-c++-cpp-output";
  }

  llvm_unreachable("Unexpected language kind!");
  return 0;
}

static const char *getActionName(frontend::ActionKind Kind) {
  switch (Kind) {
  case frontend::PluginAction:
    llvm_unreachable("Invalid kind!");

  case frontend::ASTDump:                return "-ast-dump";
  case frontend::ASTDumpXML:             return "-ast-dump-xml";
  case frontend::ASTPrint:               return "-ast-print";
  case frontend::ASTView:                return "-ast-view";
  case frontend::DumpRawTokens:          return "-dump-raw-tokens";
  case frontend::DumpTokens:             return "-dump-tokens";
  case frontend::EmitAssembly:           return "-S";
  case frontend::EmitBC:                 return "-emit-llvm-bc";
  case frontend::EmitHTML:               return "-emit-html";
  case frontend::EmitLLVM:               return "-emit-llvm";
  case frontend::EmitLLVMOnly:           return "-emit-llvm-only";
  case frontend::EmitCodeGenOnly:        return "-emit-codegen-only";
  case frontend::EmitObj:                return "-emit-obj";
  case frontend::FixIt:                  return "-fixit";
  case frontend::GenerateModule:         return "-emit-module";
  case frontend::GeneratePCH:            return "-emit-pch";
  case frontend::GeneratePTH:            return "-emit-pth";
  case frontend::InitOnly:               return "-init-only";
  case frontend::ParseSyntaxOnly:        return "-fsyntax-only";
  case frontend::PrintDeclContext:       return "-print-decl-contexts";
  case frontend::PrintPreamble:          return "-print-preamble";
  case frontend::PrintPreprocessedInput: return "-E";
  case frontend::RewriteMacros:          return "-rewrite-macros";
  case frontend::RewriteObjC:            return "-rewrite-objc";
  case frontend::RewriteTest:            return "-rewrite-test";
  case frontend::RunAnalysis:            return "-analyze";
  case frontend::RunPreprocessorOnly:    return "-Eonly";
  }

  llvm_unreachable("Unexpected language kind!");
  return 0;
}

static void FileSystemOptsToArgs(const FileSystemOptions &Opts,
                                 std::vector<std::string> &Res) {
  if (!Opts.WorkingDir.empty()) {
    Res.push_back("-working-directory");
    Res.push_back(Opts.WorkingDir);
  }
}

static void FrontendOptsToArgs(const FrontendOptions &Opts,
                               std::vector<std::string> &Res) {
  if (Opts.DisableFree)
    Res.push_back("-disable-free");
  if (Opts.RelocatablePCH)
    Res.push_back("-relocatable-pch");
  if (Opts.ShowHelp)
    Res.push_back("-help");
  if (Opts.ShowMacrosInCodeCompletion)
    Res.push_back("-code-completion-macros");
  if (Opts.ShowCodePatternsInCodeCompletion)
    Res.push_back("-code-completion-patterns");
  if (!Opts.ShowGlobalSymbolsInCodeCompletion)
    Res.push_back("-no-code-completion-globals");
  if (Opts.ShowStats)
    Res.push_back("-print-stats");
  if (Opts.ShowTimers)
    Res.push_back("-ftime-report");
  if (Opts.ShowVersion)
    Res.push_back("-version");
  if (Opts.FixWhatYouCan)
    Res.push_back("-fix-what-you-can");
  switch (Opts.ARCMTAction) {
  case FrontendOptions::ARCMT_None:
    break;
  case FrontendOptions::ARCMT_Check:
    Res.push_back("-arcmt-check");
    break;
  case FrontendOptions::ARCMT_Modify:
    Res.push_back("-arcmt-modify");
    break;
  case FrontendOptions::ARCMT_Migrate:
    Res.push_back("-arcmt-migrate");
    break;
  }
  if (!Opts.ARCMTMigrateDir.empty()) {
    Res.push_back("-arcmt-migrate-directory");
    Res.push_back(Opts.ARCMTMigrateDir);
  }
  if (!Opts.ARCMTMigrateReportOut.empty()) {
    Res.push_back("-arcmt-migrate-report-output");
    Res.push_back(Opts.ARCMTMigrateReportOut);
  }
  if (Opts.ARCMTMigrateEmitARCErrors)
    Res.push_back("-arcmt-migrate-emit-errors");

  bool NeedLang = false;
  for (unsigned i = 0, e = Opts.Inputs.size(); i != e; ++i)
    if (FrontendOptions::getInputKindForExtension(Opts.Inputs[i].second) !=
        Opts.Inputs[i].first)
      NeedLang = true;
  if (NeedLang) {
    Res.push_back("-x");
    Res.push_back(getInputKindName(Opts.Inputs[0].first));
  }
  for (unsigned i = 0, e = Opts.Inputs.size(); i != e; ++i) {
    assert((!NeedLang || Opts.Inputs[i].first == Opts.Inputs[0].first) &&
           "Unable to represent this input vector!");
    Res.push_back(Opts.Inputs[i].second);
  }

  if (!Opts.OutputFile.empty()) {
    Res.push_back("-o");
    Res.push_back(Opts.OutputFile);
  }
  if (!Opts.CodeCompletionAt.FileName.empty()) {
    Res.push_back("-code-completion-at");
    Res.push_back(Opts.CodeCompletionAt.FileName + ":" +
                  llvm::utostr(Opts.CodeCompletionAt.Line) + ":" +
                  llvm::utostr(Opts.CodeCompletionAt.Column));
  }
  if (Opts.ProgramAction != frontend::PluginAction)
    Res.push_back(getActionName(Opts.ProgramAction));
  if (!Opts.ActionName.empty()) {
    Res.push_back("-plugin");
    Res.push_back(Opts.ActionName);
    for(unsigned i = 0, e = Opts.PluginArgs.size(); i != e; ++i) {
      Res.push_back("-plugin-arg-" + Opts.ActionName);
      Res.push_back(Opts.PluginArgs[i]);
    }
  }
  for (unsigned i = 0, e = Opts.Plugins.size(); i != e; ++i) {
    Res.push_back("-load");
    Res.push_back(Opts.Plugins[i]);
  }
  for (unsigned i = 0, e = Opts.AddPluginActions.size(); i != e; ++i) {
    Res.push_back("-add-plugin");
    Res.push_back(Opts.AddPluginActions[i]);
    for(unsigned ai = 0, ae = Opts.AddPluginArgs.size(); ai != ae; ++ai) {
      Res.push_back("-plugin-arg-" + Opts.AddPluginActions[i]);
      Res.push_back(Opts.AddPluginArgs[i][ai]);
    }
  }
  for (unsigned i = 0, e = Opts.ASTMergeFiles.size(); i != e; ++i) {
    Res.push_back("-ast-merge");
    Res.push_back(Opts.ASTMergeFiles[i]);
  }
  for (unsigned i = 0, e = Opts.LLVMArgs.size(); i != e; ++i) {
    Res.push_back("-mllvm");
    Res.push_back(Opts.LLVMArgs[i]);
  }
}

static void HeaderSearchOptsToArgs(const HeaderSearchOptions &Opts,
                                   std::vector<std::string> &Res) {
  if (Opts.Sysroot != "/") {
    Res.push_back("-isysroot");
    Res.push_back(Opts.Sysroot);
  }

  /// User specified include entries.
  for (unsigned i = 0, e = Opts.UserEntries.size(); i != e; ++i) {
    const HeaderSearchOptions::Entry &E = Opts.UserEntries[i];
    if (E.IsFramework && (E.Group != frontend::Angled || !E.IsUserSupplied))
      llvm::report_fatal_error("Invalid option set!");
    if (E.IsUserSupplied) {
      switch (E.Group) {
      case frontend::After:
        Res.push_back("-idirafter");
        break;
        
      case frontend::Quoted:
        Res.push_back("-iquote");
        break;
        
      case frontend::System:
        Res.push_back("-isystem");
        break;
        
      case frontend::IndexHeaderMap:
        Res.push_back("-index-header-map");
        Res.push_back(E.IsFramework? "-F" : "-I");
        break;
        
      case frontend::CSystem:
        Res.push_back("-c-isystem");
        break;

      case frontend::CXXSystem:
        Res.push_back("-cxx-isystem");
        break;

      case frontend::ObjCSystem:
        Res.push_back("-objc-isystem");
        break;

      case frontend::ObjCXXSystem:
        Res.push_back("-objcxx-isystem");
        break;
        
      case frontend::Angled:
        Res.push_back(E.IsFramework ? "-F" : "-I");
        break;
      }
    } else {
      if (E.IsInternal) {
        assert(E.Group == frontend::System && "Unexpected header search group");
        if (E.ImplicitExternC)
          Res.push_back("-internal-externc-isystem");
        else
          Res.push_back("-internal-isystem");
      } else {
        if (E.Group != frontend::Angled && E.Group != frontend::System)
          llvm::report_fatal_error("Invalid option set!");
        Res.push_back(E.Group == frontend::Angled ? "-iwithprefixbefore" :
                      "-iwithprefix");
      }
    }
    Res.push_back(E.Path);
  }

  if (!Opts.ResourceDir.empty()) {
    Res.push_back("-resource-dir");
    Res.push_back(Opts.ResourceDir);
  }
  if (!Opts.ModuleCachePath.empty()) {
    Res.push_back("-fmodule-cache-path");
    Res.push_back(Opts.ModuleCachePath);
  }
  if (!Opts.UseStandardSystemIncludes)
    Res.push_back("-nostdsysteminc");
  if (!Opts.UseStandardCXXIncludes)
    Res.push_back("-nostdinc++");
  if (Opts.UseLibcxx)
    Res.push_back("-stdlib=libc++");
  if (Opts.Verbose)
    Res.push_back("-v");
}

static void LangOptsToArgs(const LangOptions &Opts,
                           std::vector<std::string> &Res) {
  LangOptions DefaultLangOpts;

  // FIXME: Need to set -std to get all the implicit options.

  // FIXME: We want to only pass options relative to the defaults, which
  // requires constructing a target. :(
  //
  // It would be better to push the all target specific choices into the driver,
  // so that everything below that was more uniform.

  if (Opts.Trigraphs)
    Res.push_back("-trigraphs");
  // Implicit based on the input kind:
  //   AsmPreprocessor, CPlusPlus, ObjC1, ObjC2, OpenCL
  // Implicit based on the input language standard:
  //   BCPLComment, C99, CPlusPlus0x, Digraphs, GNUInline, ImplicitInt, GNUMode
  if (Opts.DollarIdents)
    Res.push_back("-fdollars-in-identifiers");
  if (Opts.GNUMode && !Opts.GNUKeywords)
    Res.push_back("-fno-gnu-keywords");
  if (!Opts.GNUMode && Opts.GNUKeywords)
    Res.push_back("-fgnu-keywords");
  if (Opts.MicrosoftExt)
    Res.push_back("-fms-extensions");
  if (Opts.MSCVersion != 0)
    Res.push_back("-fmsc-version=" + llvm::utostr(Opts.MSCVersion));
  if (Opts.Borland)
    Res.push_back("-fborland-extensions");
  if (!Opts.ObjCNonFragileABI)
    Res.push_back("-fobjc-fragile-abi");
  if (Opts.ObjCDefaultSynthProperties)
    Res.push_back("-fobjc-default-synthesize-properties");
  // NoInline is implicit.
  if (!Opts.CXXOperatorNames)
    Res.push_back("-fno-operator-names");
  if (Opts.PascalStrings)
    Res.push_back("-fpascal-strings");
  if (Opts.CatchUndefined)
    Res.push_back("-fcatch-undefined-behavior");
  if (Opts.WritableStrings)
    Res.push_back("-fwritable-strings");
  if (Opts.ConstStrings)
    Res.push_back("-fconst-strings");
  if (!Opts.LaxVectorConversions)
    Res.push_back("-fno-lax-vector-conversions");
  if (Opts.AltiVec)
    Res.push_back("-faltivec");
  if (Opts.Exceptions)
    Res.push_back("-fexceptions");
  if (Opts.ObjCExceptions)
    Res.push_back("-fobjc-exceptions");
  if (Opts.CXXExceptions)
    Res.push_back("-fcxx-exceptions");
  if (Opts.SjLjExceptions)
    Res.push_back("-fsjlj-exceptions");
  if (Opts.TraditionalCPP)
    Res.push_back("-traditional-cpp");
  if (!Opts.RTTI)
    Res.push_back("-fno-rtti");
  if (Opts.MSBitfields)
    Res.push_back("-mms-bitfields");
  if (!Opts.NeXTRuntime)
    Res.push_back("-fgnu-runtime");
  if (Opts.Freestanding)
    Res.push_back("-ffreestanding");
  if (Opts.NoBuiltin)
    Res.push_back("-fno-builtin");
  if (!Opts.AssumeSaneOperatorNew)
    Res.push_back("-fno-assume-sane-operator-new");
  if (!Opts.ThreadsafeStatics)
    Res.push_back("-fno-threadsafe-statics");
  if (Opts.POSIXThreads)
    Res.push_back("-pthread");
  if (Opts.Blocks)
    Res.push_back("-fblocks");
  if (Opts.BlocksRuntimeOptional)
    Res.push_back("-fblocks-runtime-optional");
  if (Opts.EmitAllDecls)
    Res.push_back("-femit-all-decls");
  if (Opts.MathErrno)
    Res.push_back("-fmath-errno");
  switch (Opts.getSignedOverflowBehavior()) {
  case LangOptions::SOB_Undefined: break;
  case LangOptions::SOB_Defined:   Res.push_back("-fwrapv"); break;
  case LangOptions::SOB_Trapping:
    Res.push_back("-ftrapv"); break;
    if (!Opts.OverflowHandler.empty()) {
      Res.push_back("-ftrapv-handler");
      Res.push_back(Opts.OverflowHandler);
    }
  }
  if (Opts.HeinousExtensions)
    Res.push_back("-fheinous-gnu-extensions");
  // Optimize is implicit.
  // OptimizeSize is implicit.
  if (Opts.Static)
    Res.push_back("-static-define");
  if (Opts.DumpRecordLayouts)
    Res.push_back("-fdump-record-layouts");
  if (Opts.DumpVTableLayouts)
    Res.push_back("-fdump-vtable-layouts");
  if (Opts.NoBitFieldTypeAlign)
    Res.push_back("-fno-bitfield-type-alignment");
  if (Opts.PICLevel) {
    Res.push_back("-pic-level");
    Res.push_back(llvm::utostr(Opts.PICLevel));
  }
  if (Opts.ObjCGCBitmapPrint)
    Res.push_back("-print-ivar-layout");
  if (Opts.NoConstantCFStrings)
    Res.push_back("-fno-constant-cfstrings");
  if (!Opts.AccessControl)
    Res.push_back("-fno-access-control");
  if (!Opts.CharIsSigned)
    Res.push_back("-fno-signed-char");
  if (Opts.ShortWChar)
    Res.push_back("-fshort-wchar");
  if (!Opts.ElideConstructors)
    Res.push_back("-fno-elide-constructors");
  if (Opts.getGC() != LangOptions::NonGC) {
    if (Opts.getGC() == LangOptions::HybridGC) {
      Res.push_back("-fobjc-gc");
    } else {
      assert(Opts.getGC() == LangOptions::GCOnly && "Invalid GC mode!");
      Res.push_back("-fobjc-gc-only");
    }
  }
  if (Opts.ObjCAutoRefCount)
    Res.push_back("-fobjc-arc");
  if (Opts.ObjCRuntimeHasWeak)
    Res.push_back("-fobjc-runtime-has-weak");
  if (!Opts.ObjCInferRelatedResultType)
    Res.push_back("-fno-objc-infer-related-result-type");
  
  if (Opts.AppleKext)
    Res.push_back("-fapple-kext");
  
  if (Opts.getVisibilityMode() != DefaultVisibility) {
    Res.push_back("-fvisibility");
    if (Opts.getVisibilityMode() == HiddenVisibility) {
      Res.push_back("hidden");
    } else {
      assert(Opts.getVisibilityMode() == ProtectedVisibility &&
             "Invalid visibility!");
      Res.push_back("protected");
    }
  }
  if (Opts.InlineVisibilityHidden)
    Res.push_back("-fvisibility-inlines-hidden");

  if (Opts.getStackProtector() != 0) {
    Res.push_back("-stack-protector");
    Res.push_back(llvm::utostr(Opts.getStackProtector()));
  }
  if (Opts.InstantiationDepth != DefaultLangOpts.InstantiationDepth) {
    Res.push_back("-ftemplate-depth");
    Res.push_back(llvm::utostr(Opts.InstantiationDepth));
  }
  if (!Opts.ObjCConstantStringClass.empty()) {
    Res.push_back("-fconstant-string-class");
    Res.push_back(Opts.ObjCConstantStringClass);
  }
  if (Opts.FakeAddressSpaceMap)
    Res.push_back("-ffake-address-space-map");
  if (Opts.ParseUnknownAnytype)
    Res.push_back("-funknown-anytype");
  if (Opts.DebuggerSupport)
    Res.push_back("-fdebugger-support");
  if (Opts.DelayedTemplateParsing)
    Res.push_back("-fdelayed-template-parsing");
  if (Opts.Deprecated)
    Res.push_back("-fdeprecated-macro");
}

static void PreprocessorOptsToArgs(const PreprocessorOptions &Opts,
                                   std::vector<std::string> &Res) {
  for (unsigned i = 0, e = Opts.Macros.size(); i != e; ++i)
    Res.push_back(std::string(Opts.Macros[i].second ? "-U" : "-D") +
                  Opts.Macros[i].first);
  for (unsigned i = 0, e = Opts.Includes.size(); i != e; ++i) {
    // FIXME: We need to avoid reincluding the implicit PCH and PTH includes.
    Res.push_back("-include");
    Res.push_back(Opts.Includes[i]);
  }
  for (unsigned i = 0, e = Opts.MacroIncludes.size(); i != e; ++i) {
    Res.push_back("-imacros");
    Res.push_back(Opts.MacroIncludes[i]);
  }
  if (!Opts.UsePredefines)
    Res.push_back("-undef");
  if (Opts.DetailedRecord)
    Res.push_back("-detailed-preprocessing-record");
  if (!Opts.ImplicitPCHInclude.empty()) {
    Res.push_back("-include-pch");
    Res.push_back(Opts.ImplicitPCHInclude);
  }
  if (!Opts.ImplicitPTHInclude.empty()) {
    Res.push_back("-include-pth");
    Res.push_back(Opts.ImplicitPTHInclude);
  }
  if (!Opts.TokenCache.empty()) {
    if (Opts.ImplicitPTHInclude.empty()) {
      Res.push_back("-token-cache");
      Res.push_back(Opts.TokenCache);
    } else
      assert(Opts.ImplicitPTHInclude == Opts.TokenCache &&
             "Unsupported option combination!");
  }
  for (unsigned i = 0, e = Opts.ChainedIncludes.size(); i != e; ++i) {
    Res.push_back("-chain-include");
    Res.push_back(Opts.ChainedIncludes[i]);
  }
  for (unsigned i = 0, e = Opts.RemappedFiles.size(); i != e; ++i) {
    Res.push_back("-remap-file");
    Res.push_back(Opts.RemappedFiles[i].first + ";" +
                  Opts.RemappedFiles[i].second);
  }
}

static void PreprocessorOutputOptsToArgs(const PreprocessorOutputOptions &Opts,
                                         std::vector<std::string> &Res) {
  if (!Opts.ShowCPP && !Opts.ShowMacros)
    llvm::report_fatal_error("Invalid option combination!");

  if (Opts.ShowCPP && Opts.ShowMacros)
    Res.push_back("-dD");
  else if (!Opts.ShowCPP && Opts.ShowMacros)
    Res.push_back("-dM");

  if (!Opts.ShowLineMarkers)
    Res.push_back("-P");
  if (Opts.ShowComments)
    Res.push_back("-C");
  if (Opts.ShowMacroComments)
    Res.push_back("-CC");
}

static void TargetOptsToArgs(const TargetOptions &Opts,
                             std::vector<std::string> &Res) {
  Res.push_back("-triple");
  Res.push_back(Opts.Triple);
  if (!Opts.CPU.empty()) {
    Res.push_back("-target-cpu");
    Res.push_back(Opts.CPU);
  }
  if (!Opts.ABI.empty()) {
    Res.push_back("-target-abi");
    Res.push_back(Opts.ABI);
  }
  if (!Opts.LinkerVersion.empty()) {
    Res.push_back("-target-linker-version");
    Res.push_back(Opts.LinkerVersion);
  }
  if (!Opts.CXXABI.empty()) {
    Res.push_back("-cxx-abi");
    Res.push_back(Opts.CXXABI);
  }
  for (unsigned i = 0, e = Opts.Features.size(); i != e; ++i) {
    Res.push_back("-target-feature");
    Res.push_back(Opts.Features[i]);
  }
}

void CompilerInvocation::toArgs(std::vector<std::string> &Res) {
  AnalyzerOptsToArgs(getAnalyzerOpts(), Res);
  CodeGenOptsToArgs(getCodeGenOpts(), Res);
  DependencyOutputOptsToArgs(getDependencyOutputOpts(), Res);
  DiagnosticOptsToArgs(getDiagnosticOpts(), Res);
  FileSystemOptsToArgs(getFileSystemOpts(), Res);
  FrontendOptsToArgs(getFrontendOpts(), Res);
  HeaderSearchOptsToArgs(getHeaderSearchOpts(), Res);
  LangOptsToArgs(getLangOpts(), Res);
  PreprocessorOptsToArgs(getPreprocessorOpts(), Res);
  PreprocessorOutputOptsToArgs(getPreprocessorOutputOpts(), Res);
  TargetOptsToArgs(getTargetOpts(), Res);
}

//===----------------------------------------------------------------------===//
// Deserialization (to args)
//===----------------------------------------------------------------------===//

using namespace clang::driver;
using namespace clang::driver::cc1options;

//

static unsigned getOptimizationLevel(ArgList &Args, InputKind IK,
                                     DiagnosticsEngine &Diags) {
  unsigned DefaultOpt = 0;
  if (IK == IK_OpenCL && !Args.hasArg(OPT_cl_opt_disable))
    DefaultOpt = 2;
  // -Os/-Oz implies -O2
  return (Args.hasArg(OPT_Os) || Args.hasArg (OPT_Oz)) ? 2 :
    Args.getLastArgIntValue(OPT_O, DefaultOpt, Diags);
}

static void ParseAnalyzerArgs(AnalyzerOptions &Opts, ArgList &Args,
                              DiagnosticsEngine &Diags) {
  using namespace cc1options;

  if (Arg *A = Args.getLastArg(OPT_analyzer_store)) {
    StringRef Name = A->getValue(Args);
    AnalysisStores Value = llvm::StringSwitch<AnalysisStores>(Name)
#define ANALYSIS_STORE(NAME, CMDFLAG, DESC, CREATFN) \
      .Case(CMDFLAG, NAME##Model)
#include "clang/Frontend/Analyses.def"
      .Default(NumStores);
    // FIXME: Error handling.
    if (Value == NumStores)
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << Name;
    else
      Opts.AnalysisStoreOpt = Value;
  }

  if (Arg *A = Args.getLastArg(OPT_analyzer_constraints)) {
    StringRef Name = A->getValue(Args);
    AnalysisConstraints Value = llvm::StringSwitch<AnalysisConstraints>(Name)
#define ANALYSIS_CONSTRAINTS(NAME, CMDFLAG, DESC, CREATFN) \
      .Case(CMDFLAG, NAME##Model)
#include "clang/Frontend/Analyses.def"
      .Default(NumConstraints);
    // FIXME: Error handling.
    if (Value == NumConstraints)
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << Name;
    else
      Opts.AnalysisConstraintsOpt = Value;
  }

  if (Arg *A = Args.getLastArg(OPT_analyzer_output)) {
    StringRef Name = A->getValue(Args);
    AnalysisDiagClients Value = llvm::StringSwitch<AnalysisDiagClients>(Name)
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATFN, AUTOCREAT) \
      .Case(CMDFLAG, PD_##NAME)
#include "clang/Frontend/Analyses.def"
      .Default(NUM_ANALYSIS_DIAG_CLIENTS);
    // FIXME: Error handling.
    if (Value == NUM_ANALYSIS_DIAG_CLIENTS)
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << Name;
    else
      Opts.AnalysisDiagOpt = Value;
  }

  if (Arg *A = Args.getLastArg(OPT_analyzer_purge)) {
    StringRef Name = A->getValue(Args);
    AnalysisPurgeMode Value = llvm::StringSwitch<AnalysisPurgeMode>(Name)
#define ANALYSIS_PURGE(NAME, CMDFLAG, DESC) \
      .Case(CMDFLAG, NAME)
#include "clang/Frontend/Analyses.def"
      .Default(NumPurgeModes);
    // FIXME: Error handling.
    if (Value == NumPurgeModes)
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << Name;
    else
      Opts.AnalysisPurgeOpt = Value;
  }

  Opts.ShowCheckerHelp = Args.hasArg(OPT_analyzer_checker_help);
  Opts.VisualizeEGDot = Args.hasArg(OPT_analyzer_viz_egraph_graphviz);
  Opts.VisualizeEGUbi = Args.hasArg(OPT_analyzer_viz_egraph_ubigraph);
  Opts.AnalyzeAll = Args.hasArg(OPT_analyzer_opt_analyze_headers);
  Opts.AnalyzerDisplayProgress = Args.hasArg(OPT_analyzer_display_progress);
  Opts.AnalyzeNestedBlocks =
    Args.hasArg(OPT_analyzer_opt_analyze_nested_blocks);
  Opts.EagerlyAssume = Args.hasArg(OPT_analyzer_eagerly_assume);
  Opts.AnalyzeSpecificFunction = Args.getLastArgValue(OPT_analyze_function);
  Opts.UnoptimizedCFG = Args.hasArg(OPT_analysis_UnoptimizedCFG);
  Opts.CFGAddImplicitDtors = Args.hasArg(OPT_analysis_CFGAddImplicitDtors);
  Opts.CFGAddInitializers = Args.hasArg(OPT_analysis_CFGAddInitializers);
  Opts.TrimGraph = Args.hasArg(OPT_trim_egraph);
  Opts.MaxNodes = Args.getLastArgIntValue(OPT_analyzer_max_nodes, 150000,Diags);
  Opts.MaxLoop = Args.getLastArgIntValue(OPT_analyzer_max_loop, 4, Diags);
  Opts.EagerlyTrimEGraph = !Args.hasArg(OPT_analyzer_no_eagerly_trim_egraph);
  Opts.InlineCall = Args.hasArg(OPT_analyzer_inline_call);

  Opts.CheckersControlList.clear();
  for (arg_iterator it = Args.filtered_begin(OPT_analyzer_checker,
                                             OPT_analyzer_disable_checker),
         ie = Args.filtered_end(); it != ie; ++it) {
    const Arg *A = *it;
    A->claim();
    bool enable = (A->getOption().getID() == OPT_analyzer_checker);
    // We can have a list of comma separated checker names, e.g:
    // '-analyzer-checker=cocoa,unix'
    StringRef checkerList = A->getValue(Args);
    SmallVector<StringRef, 4> checkers;
    checkerList.split(checkers, ",");
    for (unsigned i = 0, e = checkers.size(); i != e; ++i)
      Opts.CheckersControlList.push_back(std::make_pair(checkers[i], enable));
  }
}

static void ParseCodeGenArgs(CodeGenOptions &Opts, ArgList &Args, InputKind IK,
                             DiagnosticsEngine &Diags) {
  using namespace cc1options;

  Opts.OptimizationLevel = getOptimizationLevel(Args, IK, Diags);
  if (Opts.OptimizationLevel > 3) {
    Diags.Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_O)->getAsString(Args) << Opts.OptimizationLevel;
    Opts.OptimizationLevel = 3;
  }

  // We must always run at least the always inlining pass.
  Opts.Inlining = (Opts.OptimizationLevel > 1) ? CodeGenOptions::NormalInlining
    : CodeGenOptions::OnlyAlwaysInlining;

  Opts.DebugInfo = Args.hasArg(OPT_g);
  Opts.LimitDebugInfo = Args.hasArg(OPT_flimit_debug_info);
  Opts.DisableLLVMOpts = Args.hasArg(OPT_disable_llvm_optzns);
  Opts.DisableRedZone = Args.hasArg(OPT_disable_red_zone);
  Opts.ForbidGuardVariables = Args.hasArg(OPT_fforbid_guard_variables);
  Opts.UseRegisterSizedBitfieldAccess = Args.hasArg(
    OPT_fuse_register_sized_bitfield_access);
  Opts.RelaxedAliasing = Args.hasArg(OPT_relaxed_aliasing);
  Opts.DwarfDebugFlags = Args.getLastArgValue(OPT_dwarf_debug_flags);
  Opts.MergeAllConstants = !Args.hasArg(OPT_fno_merge_all_constants);
  Opts.NoCommon = Args.hasArg(OPT_fno_common);
  Opts.NoImplicitFloat = Args.hasArg(OPT_no_implicit_float);
  Opts.OptimizeSize = Args.hasArg(OPT_Os);
  Opts.OptimizeSize = Args.hasArg(OPT_Oz) ? 2 : Opts.OptimizeSize;
  Opts.SimplifyLibCalls = !(Args.hasArg(OPT_fno_builtin) ||
                            Args.hasArg(OPT_ffreestanding));
  Opts.UnrollLoops = Args.hasArg(OPT_funroll_loops) ||
                     (Opts.OptimizationLevel > 1 && !Opts.OptimizeSize);

  Opts.AsmVerbose = Args.hasArg(OPT_masm_verbose);
  Opts.ObjCAutoRefCountExceptions = Args.hasArg(OPT_fobjc_arc_exceptions);
  Opts.ObjCRuntimeHasARC = Args.hasArg(OPT_fobjc_runtime_has_arc);
  Opts.ObjCRuntimeHasTerminate = Args.hasArg(OPT_fobjc_runtime_has_terminate);
  Opts.CUDAIsDevice = Args.hasArg(OPT_fcuda_is_device);
  Opts.CXAAtExit = !Args.hasArg(OPT_fno_use_cxa_atexit);
  Opts.CXXCtorDtorAliases = Args.hasArg(OPT_mconstructor_aliases);
  Opts.CodeModel = Args.getLastArgValue(OPT_mcode_model);
  Opts.DebugPass = Args.getLastArgValue(OPT_mdebug_pass);
  Opts.DisableFPElim = Args.hasArg(OPT_mdisable_fp_elim);
  Opts.FloatABI = Args.getLastArgValue(OPT_mfloat_abi);
  Opts.HiddenWeakVTables = Args.hasArg(OPT_fhidden_weak_vtables);
  Opts.LessPreciseFPMAD = Args.hasArg(OPT_cl_mad_enable);
  Opts.LimitFloatPrecision = Args.getLastArgValue(OPT_mlimit_float_precision);
  Opts.NoInfsFPMath = Opts.NoNaNsFPMath = Args.hasArg(OPT_cl_finite_math_only)||
                                          Args.hasArg(OPT_cl_fast_relaxed_math);
  Opts.NoZeroInitializedInBSS = Args.hasArg(OPT_mno_zero_initialized_in_bss);
  Opts.BackendOptions = Args.getAllArgValues(OPT_backend_option);
  Opts.NumRegisterParameters = Args.getLastArgIntValue(OPT_mregparm, 0, Diags);
  Opts.NoGlobalMerge = Args.hasArg(OPT_mno_global_merge);
  Opts.NoExecStack = Args.hasArg(OPT_mno_exec_stack);
  Opts.RelaxAll = Args.hasArg(OPT_mrelax_all);
  Opts.OmitLeafFramePointer = Args.hasArg(OPT_momit_leaf_frame_pointer);
  Opts.SaveTempLabels = Args.hasArg(OPT_msave_temp_labels);
  Opts.NoDwarf2CFIAsm = Args.hasArg(OPT_fno_dwarf2_cfi_asm);
  Opts.SoftFloat = Args.hasArg(OPT_msoft_float);
  Opts.UnsafeFPMath = Args.hasArg(OPT_cl_unsafe_math_optimizations) ||
                      Args.hasArg(OPT_cl_fast_relaxed_math);
  Opts.UnwindTables = Args.hasArg(OPT_munwind_tables);
  Opts.RelocationModel = Args.getLastArgValue(OPT_mrelocation_model, "pic");

  Opts.FunctionSections = Args.hasArg(OPT_ffunction_sections);
  Opts.DataSections = Args.hasArg(OPT_fdata_sections);

  Opts.MainFileName = Args.getLastArgValue(OPT_main_file_name);
  Opts.VerifyModule = !Args.hasArg(OPT_disable_llvm_verifier);

  Opts.InstrumentFunctions = Args.hasArg(OPT_finstrument_functions);
  Opts.InstrumentForProfiling = Args.hasArg(OPT_pg);
  Opts.EmitGcovArcs = Args.hasArg(OPT_femit_coverage_data);
  Opts.EmitGcovNotes = Args.hasArg(OPT_femit_coverage_notes);
  Opts.CoverageFile = Args.getLastArgValue(OPT_coverage_file);

  if (Arg *A = Args.getLastArg(OPT_fobjc_dispatch_method_EQ)) {
    StringRef Name = A->getValue(Args);
    unsigned Method = llvm::StringSwitch<unsigned>(Name)
      .Case("legacy", CodeGenOptions::Legacy)
      .Case("non-legacy", CodeGenOptions::NonLegacy)
      .Case("mixed", CodeGenOptions::Mixed)
      .Default(~0U);
    if (Method == ~0U)
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Name;
    else
      Opts.ObjCDispatchMethod = Method;
  }
}

static void ParseDependencyOutputArgs(DependencyOutputOptions &Opts,
                                      ArgList &Args) {
  using namespace cc1options;
  Opts.OutputFile = Args.getLastArgValue(OPT_dependency_file);
  Opts.Targets = Args.getAllArgValues(OPT_MT);
  Opts.IncludeSystemHeaders = Args.hasArg(OPT_sys_header_deps);
  Opts.UsePhonyTargets = Args.hasArg(OPT_MP);
  Opts.ShowHeaderIncludes = Args.hasArg(OPT_H);
  Opts.HeaderIncludeOutputFile = Args.getLastArgValue(OPT_header_include_file);
  Opts.AddMissingHeaderDeps = Args.hasArg(OPT_MG);
}

static void ParseDiagnosticArgs(DiagnosticOptions &Opts, ArgList &Args,
                                DiagnosticsEngine &Diags) {
  using namespace cc1options;
  Opts.DiagnosticLogFile = Args.getLastArgValue(OPT_diagnostic_log_file);
  Opts.IgnoreWarnings = Args.hasArg(OPT_w);
  Opts.NoRewriteMacros = Args.hasArg(OPT_Wno_rewrite_macros);
  Opts.Pedantic = Args.hasArg(OPT_pedantic);
  Opts.PedanticErrors = Args.hasArg(OPT_pedantic_errors);
  Opts.ShowCarets = !Args.hasArg(OPT_fno_caret_diagnostics);
  Opts.ShowColors = Args.hasArg(OPT_fcolor_diagnostics);
  Opts.ShowColumn = Args.hasFlag(OPT_fshow_column,
                                 OPT_fno_show_column,
                                 /*Default=*/true);
  Opts.ShowFixits = !Args.hasArg(OPT_fno_diagnostics_fixit_info);
  Opts.ShowLocation = !Args.hasArg(OPT_fno_show_source_location);
  Opts.ShowNames = Args.hasArg(OPT_fdiagnostics_show_name);
  Opts.ShowOptionNames = Args.hasArg(OPT_fdiagnostics_show_option);

  // Default behavior is to not to show note include stacks.
  Opts.ShowNoteIncludeStack = false;
  if (Arg *A = Args.getLastArg(OPT_fdiagnostics_show_note_include_stack,
                               OPT_fno_diagnostics_show_note_include_stack))
    if (A->getOption().matches(OPT_fdiagnostics_show_note_include_stack))
      Opts.ShowNoteIncludeStack = true;

  StringRef ShowOverloads =
    Args.getLastArgValue(OPT_fshow_overloads_EQ, "all");
  if (ShowOverloads == "best")
    Opts.ShowOverloads = DiagnosticsEngine::Ovl_Best;
  else if (ShowOverloads == "all")
    Opts.ShowOverloads = DiagnosticsEngine::Ovl_All;
  else
    Diags.Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_fshow_overloads_EQ)->getAsString(Args)
      << ShowOverloads;

  StringRef ShowCategory =
    Args.getLastArgValue(OPT_fdiagnostics_show_category, "none");
  if (ShowCategory == "none")
    Opts.ShowCategories = 0;
  else if (ShowCategory == "id")
    Opts.ShowCategories = 1;
  else if (ShowCategory == "name")
    Opts.ShowCategories = 2;
  else
    Diags.Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_fdiagnostics_show_category)->getAsString(Args)
      << ShowCategory;

  StringRef Format =
    Args.getLastArgValue(OPT_fdiagnostics_format, "clang");
  if (Format == "clang")
    Opts.Format = DiagnosticOptions::Clang;
  else if (Format == "msvc") 
    Opts.Format = DiagnosticOptions::Msvc;
  else if (Format == "vi") 
    Opts.Format = DiagnosticOptions::Vi;
  else 
    Diags.Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_fdiagnostics_format)->getAsString(Args)
      << Format;
  
  Opts.ShowSourceRanges = Args.hasArg(OPT_fdiagnostics_print_source_range_info);
  Opts.ShowParseableFixits = Args.hasArg(OPT_fdiagnostics_parseable_fixits);
  Opts.VerifyDiagnostics = Args.hasArg(OPT_verify);
  Opts.ErrorLimit = Args.getLastArgIntValue(OPT_ferror_limit, 0, Diags);
  Opts.MacroBacktraceLimit
    = Args.getLastArgIntValue(OPT_fmacro_backtrace_limit,
                         DiagnosticOptions::DefaultMacroBacktraceLimit, Diags);
  Opts.TemplateBacktraceLimit
    = Args.getLastArgIntValue(OPT_ftemplate_backtrace_limit,
                         DiagnosticOptions::DefaultTemplateBacktraceLimit,
                         Diags);
  Opts.TabStop = Args.getLastArgIntValue(OPT_ftabstop,
                                    DiagnosticOptions::DefaultTabStop, Diags);
  if (Opts.TabStop == 0 || Opts.TabStop > DiagnosticOptions::MaxTabStop) {
    Diags.Report(diag::warn_ignoring_ftabstop_value)
      << Opts.TabStop << DiagnosticOptions::DefaultTabStop;
    Opts.TabStop = DiagnosticOptions::DefaultTabStop;
  }
  Opts.MessageLength = Args.getLastArgIntValue(OPT_fmessage_length, 0, Diags);
  Opts.DumpBuildInformation = Args.getLastArgValue(OPT_dump_build_information);
  Opts.Warnings = Args.getAllArgValues(OPT_W);
}

static void ParseFileSystemArgs(FileSystemOptions &Opts, ArgList &Args) {
  Opts.WorkingDir = Args.getLastArgValue(OPT_working_directory);
}

static InputKind ParseFrontendArgs(FrontendOptions &Opts, ArgList &Args,
                                   DiagnosticsEngine &Diags) {
  using namespace cc1options;
  Opts.ProgramAction = frontend::ParseSyntaxOnly;
  if (const Arg *A = Args.getLastArg(OPT_Action_Group)) {
    switch (A->getOption().getID()) {
    default:
      llvm_unreachable("Invalid option in group!");
    case OPT_ast_dump:
      Opts.ProgramAction = frontend::ASTDump; break;
    case OPT_ast_dump_xml:
      Opts.ProgramAction = frontend::ASTDumpXML; break;
    case OPT_ast_print:
      Opts.ProgramAction = frontend::ASTPrint; break;
    case OPT_ast_view:
      Opts.ProgramAction = frontend::ASTView; break;
    case OPT_dump_raw_tokens:
      Opts.ProgramAction = frontend::DumpRawTokens; break;
    case OPT_dump_tokens:
      Opts.ProgramAction = frontend::DumpTokens; break;
    case OPT_S:
      Opts.ProgramAction = frontend::EmitAssembly; break;
    case OPT_emit_llvm_bc:
      Opts.ProgramAction = frontend::EmitBC; break;
    case OPT_emit_html:
      Opts.ProgramAction = frontend::EmitHTML; break;
    case OPT_emit_llvm:
      Opts.ProgramAction = frontend::EmitLLVM; break;
    case OPT_emit_llvm_only:
      Opts.ProgramAction = frontend::EmitLLVMOnly; break;
    case OPT_emit_codegen_only:
      Opts.ProgramAction = frontend::EmitCodeGenOnly; break;
    case OPT_emit_obj:
      Opts.ProgramAction = frontend::EmitObj; break;
    case OPT_fixit_EQ:
      Opts.FixItSuffix = A->getValue(Args);
      // fall-through!
    case OPT_fixit:
      Opts.ProgramAction = frontend::FixIt; break;
    case OPT_emit_module:
      Opts.ProgramAction = frontend::GenerateModule; break;
    case OPT_emit_pch:
      Opts.ProgramAction = frontend::GeneratePCH; break;
    case OPT_emit_pth:
      Opts.ProgramAction = frontend::GeneratePTH; break;
    case OPT_init_only:
      Opts.ProgramAction = frontend::InitOnly; break;
    case OPT_fsyntax_only:
      Opts.ProgramAction = frontend::ParseSyntaxOnly; break;
    case OPT_print_decl_contexts:
      Opts.ProgramAction = frontend::PrintDeclContext; break;
    case OPT_print_preamble:
      Opts.ProgramAction = frontend::PrintPreamble; break;
    case OPT_E:
      Opts.ProgramAction = frontend::PrintPreprocessedInput; break;
    case OPT_rewrite_macros:
      Opts.ProgramAction = frontend::RewriteMacros; break;
    case OPT_rewrite_objc:
      Opts.ProgramAction = frontend::RewriteObjC; break;
    case OPT_rewrite_test:
      Opts.ProgramAction = frontend::RewriteTest; break;
    case OPT_analyze:
      Opts.ProgramAction = frontend::RunAnalysis; break;
    case OPT_Eonly:
      Opts.ProgramAction = frontend::RunPreprocessorOnly; break;
    }
  }

  if (const Arg* A = Args.getLastArg(OPT_plugin)) {
    Opts.Plugins.push_back(A->getValue(Args,0));
    Opts.ProgramAction = frontend::PluginAction;
    Opts.ActionName = A->getValue(Args);

    for (arg_iterator it = Args.filtered_begin(OPT_plugin_arg),
           end = Args.filtered_end(); it != end; ++it) {
      if ((*it)->getValue(Args, 0) == Opts.ActionName)
        Opts.PluginArgs.push_back((*it)->getValue(Args, 1));
    }
  }

  Opts.AddPluginActions = Args.getAllArgValues(OPT_add_plugin);
  Opts.AddPluginArgs.resize(Opts.AddPluginActions.size());
  for (int i = 0, e = Opts.AddPluginActions.size(); i != e; ++i) {
    for (arg_iterator it = Args.filtered_begin(OPT_plugin_arg),
           end = Args.filtered_end(); it != end; ++it) {
      if ((*it)->getValue(Args, 0) == Opts.AddPluginActions[i])
        Opts.AddPluginArgs[i].push_back((*it)->getValue(Args, 1));
    }
  }

  if (const Arg *A = Args.getLastArg(OPT_code_completion_at)) {
    Opts.CodeCompletionAt =
      ParsedSourceLocation::FromString(A->getValue(Args));
    if (Opts.CodeCompletionAt.FileName.empty())
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue(Args);
  }
  Opts.DisableFree = Args.hasArg(OPT_disable_free);

  Opts.OutputFile = Args.getLastArgValue(OPT_o);
  Opts.Plugins = Args.getAllArgValues(OPT_load);
  Opts.RelocatablePCH = Args.hasArg(OPT_relocatable_pch);
  Opts.ShowHelp = Args.hasArg(OPT_help);
  Opts.ShowMacrosInCodeCompletion = Args.hasArg(OPT_code_completion_macros);
  Opts.ShowCodePatternsInCodeCompletion
    = Args.hasArg(OPT_code_completion_patterns);
  Opts.ShowGlobalSymbolsInCodeCompletion
    = !Args.hasArg(OPT_no_code_completion_globals);
  Opts.ShowStats = Args.hasArg(OPT_print_stats);
  Opts.ShowTimers = Args.hasArg(OPT_ftime_report);
  Opts.ShowVersion = Args.hasArg(OPT_version);
  Opts.ASTMergeFiles = Args.getAllArgValues(OPT_ast_merge);
  Opts.LLVMArgs = Args.getAllArgValues(OPT_mllvm);
  Opts.FixWhatYouCan = Args.hasArg(OPT_fix_what_you_can);

  Opts.ARCMTAction = FrontendOptions::ARCMT_None;
  if (const Arg *A = Args.getLastArg(OPT_arcmt_check,
                                     OPT_arcmt_modify,
                                     OPT_arcmt_migrate)) {
    switch (A->getOption().getID()) {
    default:
      llvm_unreachable("missed a case");
    case OPT_arcmt_check:
      Opts.ARCMTAction = FrontendOptions::ARCMT_Check;
      break;
    case OPT_arcmt_modify:
      Opts.ARCMTAction = FrontendOptions::ARCMT_Modify;
      break;
    case OPT_arcmt_migrate:
      Opts.ARCMTAction = FrontendOptions::ARCMT_Migrate;
      break;
    }
  }
  Opts.ARCMTMigrateDir = Args.getLastArgValue(OPT_arcmt_migrate_directory);
  Opts.ARCMTMigrateReportOut
    = Args.getLastArgValue(OPT_arcmt_migrate_report_output);
  Opts.ARCMTMigrateEmitARCErrors
    = Args.hasArg(OPT_arcmt_migrate_emit_arc_errors);

  InputKind DashX = IK_None;
  if (const Arg *A = Args.getLastArg(OPT_x)) {
    DashX = llvm::StringSwitch<InputKind>(A->getValue(Args))
      .Case("c", IK_C)
      .Case("cl", IK_OpenCL)
      .Case("cuda", IK_CUDA)
      .Case("c++", IK_CXX)
      .Case("objective-c", IK_ObjC)
      .Case("objective-c++", IK_ObjCXX)
      .Case("cpp-output", IK_PreprocessedC)
      .Case("assembler-with-cpp", IK_Asm)
      .Case("c++-cpp-output", IK_PreprocessedCXX)
      .Case("objective-c-cpp-output", IK_PreprocessedObjC)
      .Case("objc-cpp-output", IK_PreprocessedObjC)
      .Case("objective-c++-cpp-output", IK_PreprocessedObjCXX)
      .Case("objc++-cpp-output", IK_PreprocessedObjCXX)
      .Case("c-header", IK_C)
      .Case("cl-header", IK_OpenCL)
      .Case("objective-c-header", IK_ObjC)
      .Case("c++-header", IK_CXX)
      .Case("objective-c++-header", IK_ObjCXX)
      .Case("ast", IK_AST)
      .Case("ir", IK_LLVM_IR)
      .Default(IK_None);
    if (DashX == IK_None)
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue(Args);
  }

  // '-' is the default input if none is given.
  std::vector<std::string> Inputs = Args.getAllArgValues(OPT_INPUT);
  Opts.Inputs.clear();
  if (Inputs.empty())
    Inputs.push_back("-");
  for (unsigned i = 0, e = Inputs.size(); i != e; ++i) {
    InputKind IK = DashX;
    if (IK == IK_None) {
      IK = FrontendOptions::getInputKindForExtension(
        StringRef(Inputs[i]).rsplit('.').second);
      // FIXME: Remove this hack.
      if (i == 0)
        DashX = IK;
    }
    Opts.Inputs.push_back(std::make_pair(IK, Inputs[i]));
  }

  return DashX;
}

std::string CompilerInvocation::GetResourcesPath(const char *Argv0,
                                                 void *MainAddr) {
  llvm::sys::Path P = llvm::sys::Path::GetMainExecutable(Argv0, MainAddr);

  if (!P.isEmpty()) {
    P.eraseComponent();  // Remove /clang from foo/bin/clang
    P.eraseComponent();  // Remove /bin   from foo/bin

    // Get foo/lib/clang/<version>/include
    P.appendComponent("lib");
    P.appendComponent("clang");
    P.appendComponent(CLANG_VERSION_STRING);
  }

  return P.str();
}

static void ParseHeaderSearchArgs(HeaderSearchOptions &Opts, ArgList &Args) {
  using namespace cc1options;
  Opts.Sysroot = Args.getLastArgValue(OPT_isysroot, "/");
  Opts.Verbose = Args.hasArg(OPT_v);
  Opts.UseBuiltinIncludes = !Args.hasArg(OPT_nobuiltininc);
  Opts.UseStandardSystemIncludes = !Args.hasArg(OPT_nostdsysteminc);
  Opts.UseStandardCXXIncludes = !Args.hasArg(OPT_nostdincxx);
  if (const Arg *A = Args.getLastArg(OPT_stdlib_EQ))
    Opts.UseLibcxx = (strcmp(A->getValue(Args), "libc++") == 0);
  Opts.ResourceDir = Args.getLastArgValue(OPT_resource_dir);
  Opts.ModuleCachePath = Args.getLastArgValue(OPT_fmodule_cache_path);
  Opts.DisableModuleHash = Args.hasArg(OPT_fdisable_module_hash);
  
  // Add -I..., -F..., and -index-header-map options in order.
  bool IsIndexHeaderMap = false;
  for (arg_iterator it = Args.filtered_begin(OPT_I, OPT_F, 
                                             OPT_index_header_map),
       ie = Args.filtered_end(); it != ie; ++it) {
    if ((*it)->getOption().matches(OPT_index_header_map)) {
      // -index-header-map applies to the next -I or -F.
      IsIndexHeaderMap = true;
      continue;
    }
        
    frontend::IncludeDirGroup Group 
      = IsIndexHeaderMap? frontend::IndexHeaderMap : frontend::Angled;
    
    Opts.AddPath((*it)->getValue(Args), Group, true,
                 /*IsFramework=*/ (*it)->getOption().matches(OPT_F), false);
    IsIndexHeaderMap = false;
  }

  // Add -iprefix/-iwith-prefix/-iwithprefixbefore options.
  StringRef Prefix = ""; // FIXME: This isn't the correct default prefix.
  for (arg_iterator it = Args.filtered_begin(OPT_iprefix, OPT_iwithprefix,
                                             OPT_iwithprefixbefore),
         ie = Args.filtered_end(); it != ie; ++it) {
    const Arg *A = *it;
    if (A->getOption().matches(OPT_iprefix))
      Prefix = A->getValue(Args);
    else if (A->getOption().matches(OPT_iwithprefix))
      Opts.AddPath(Prefix.str() + A->getValue(Args),
                   frontend::System, false, false, false);
    else
      Opts.AddPath(Prefix.str() + A->getValue(Args),
                   frontend::Angled, false, false, false);
  }

  for (arg_iterator it = Args.filtered_begin(OPT_idirafter),
         ie = Args.filtered_end(); it != ie; ++it)
    Opts.AddPath((*it)->getValue(Args), frontend::After, true, false, false);
  for (arg_iterator it = Args.filtered_begin(OPT_iquote),
         ie = Args.filtered_end(); it != ie; ++it)
    Opts.AddPath((*it)->getValue(Args), frontend::Quoted, true, false, false);
  for (arg_iterator it = Args.filtered_begin(OPT_isystem,
         OPT_iwithsysroot), ie = Args.filtered_end(); it != ie; ++it)
    Opts.AddPath((*it)->getValue(Args), frontend::System, true, false,
                 !(*it)->getOption().matches(OPT_iwithsysroot));

  // Add the paths for the various language specific isystem flags.
  for (arg_iterator it = Args.filtered_begin(OPT_c_isystem),
       ie = Args.filtered_end(); it != ie; ++it)
    Opts.AddPath((*it)->getValue(Args), frontend::CSystem, true, false, true);
  for (arg_iterator it = Args.filtered_begin(OPT_cxx_isystem),
       ie = Args.filtered_end(); it != ie; ++it)
    Opts.AddPath((*it)->getValue(Args), frontend::CXXSystem, true, false, true);
  for (arg_iterator it = Args.filtered_begin(OPT_objc_isystem),
       ie = Args.filtered_end(); it != ie; ++it)
    Opts.AddPath((*it)->getValue(Args), frontend::ObjCSystem, true, false,true);
  for (arg_iterator it = Args.filtered_begin(OPT_objcxx_isystem),
       ie = Args.filtered_end(); it != ie; ++it)
    Opts.AddPath((*it)->getValue(Args), frontend::ObjCXXSystem, true, false,
                 true);

  // Add the internal paths from a driver that detects standard include paths.
  for (arg_iterator I = Args.filtered_begin(OPT_internal_isystem,
                                            OPT_internal_externc_isystem),
                    E = Args.filtered_end();
       I != E; ++I)
    Opts.AddPath((*I)->getValue(Args), frontend::System,
                 false, false, /*IgnoreSysRoot=*/true, /*IsInternal=*/true,
                 (*I)->getOption().matches(OPT_internal_externc_isystem));
}

void CompilerInvocation::setLangDefaults(LangOptions &Opts, InputKind IK,
                                         LangStandard::Kind LangStd) {
  // Set some properties which depend solely on the input kind; it would be nice
  // to move these to the language standard, and have the driver resolve the
  // input kind + language standard.
  if (IK == IK_Asm) {
    Opts.AsmPreprocessor = 1;
  } else if (IK == IK_ObjC ||
             IK == IK_ObjCXX ||
             IK == IK_PreprocessedObjC ||
             IK == IK_PreprocessedObjCXX) {
    Opts.ObjC1 = Opts.ObjC2 = 1;
  }

  if (LangStd == LangStandard::lang_unspecified) {
    // Based on the base language, pick one.
    switch (IK) {
    case IK_None:
    case IK_AST:
    case IK_LLVM_IR:
      llvm_unreachable("Invalid input kind!");
    case IK_OpenCL:
      LangStd = LangStandard::lang_opencl;
      break;
    case IK_CUDA:
      LangStd = LangStandard::lang_cuda;
      break;
    case IK_Asm:
    case IK_C:
    case IK_PreprocessedC:
    case IK_ObjC:
    case IK_PreprocessedObjC:
      LangStd = LangStandard::lang_gnu99;
      break;
    case IK_CXX:
    case IK_PreprocessedCXX:
    case IK_ObjCXX:
    case IK_PreprocessedObjCXX:
      LangStd = LangStandard::lang_gnucxx98;
      break;
    }
  }

  const LangStandard &Std = LangStandard::getLangStandardForKind(LangStd);
  Opts.BCPLComment = Std.hasBCPLComments();
  Opts.C99 = Std.isC99();
  Opts.C1X = Std.isC1X();
  Opts.CPlusPlus = Std.isCPlusPlus();
  Opts.CPlusPlus0x = Std.isCPlusPlus0x();
  Opts.Digraphs = Std.hasDigraphs();
  Opts.GNUMode = Std.isGNUMode();
  Opts.GNUInline = !Std.isC99();
  Opts.HexFloats = Std.hasHexFloats();
  Opts.ImplicitInt = Std.hasImplicitInt();

  // OpenCL has some additional defaults.
  if (LangStd == LangStandard::lang_opencl) {
    Opts.OpenCL = 1;
    Opts.AltiVec = 0;
    Opts.CXXOperatorNames = 1;
    Opts.LaxVectorConversions = 0;
    Opts.DefaultFPContract = 1;
  }

  if (LangStd == LangStandard::lang_cuda)
    Opts.CUDA = 1;

  // OpenCL and C++ both have bool, true, false keywords.
  Opts.Bool = Opts.OpenCL || Opts.CPlusPlus;

  Opts.GNUKeywords = Opts.GNUMode;
  Opts.CXXOperatorNames = Opts.CPlusPlus;

  // Mimicing gcc's behavior, trigraphs are only enabled if -trigraphs
  // is specified, or -std is set to a conforming mode.
  Opts.Trigraphs = !Opts.GNUMode;

  Opts.DollarIdents = !Opts.AsmPreprocessor;
}

static void ParseLangArgs(LangOptions &Opts, ArgList &Args, InputKind IK,
                          DiagnosticsEngine &Diags) {
  // FIXME: Cleanup per-file based stuff.
  LangStandard::Kind LangStd = LangStandard::lang_unspecified;
  if (const Arg *A = Args.getLastArg(OPT_std_EQ)) {
    LangStd = llvm::StringSwitch<LangStandard::Kind>(A->getValue(Args))
#define LANGSTANDARD(id, name, desc, features) \
      .Case(name, LangStandard::lang_##id)
#include "clang/Frontend/LangStandards.def"
      .Default(LangStandard::lang_unspecified);
    if (LangStd == LangStandard::lang_unspecified)
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue(Args);
    else {
      // Valid standard, check to make sure language and standard are compatable.    
      const LangStandard &Std = LangStandard::getLangStandardForKind(LangStd);
      switch (IK) {
      case IK_C:
      case IK_ObjC:
      case IK_PreprocessedC:
      case IK_PreprocessedObjC:
        if (!(Std.isC89() || Std.isC99()))
          Diags.Report(diag::err_drv_argument_not_allowed_with)
            << A->getAsString(Args) << "C/ObjC";
        break;
      case IK_CXX:
      case IK_ObjCXX:
      case IK_PreprocessedCXX:
      case IK_PreprocessedObjCXX:
        if (!Std.isCPlusPlus())
          Diags.Report(diag::err_drv_argument_not_allowed_with)
            << A->getAsString(Args) << "C++/ObjC++";
        break;
      case IK_OpenCL:
        if (!Std.isC99())
          Diags.Report(diag::err_drv_argument_not_allowed_with)
            << A->getAsString(Args) << "OpenCL";
        break;
      case IK_CUDA:
        if (!Std.isCPlusPlus())
          Diags.Report(diag::err_drv_argument_not_allowed_with)
            << A->getAsString(Args) << "CUDA";
        break;
      default:
        break;
      }
    }
  }

  if (const Arg *A = Args.getLastArg(OPT_cl_std_EQ)) {
    if (strcmp(A->getValue(Args), "CL1.1") != 0) {
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue(Args);
    }
  }

  CompilerInvocation::setLangDefaults(Opts, IK, LangStd);

  // We abuse '-f[no-]gnu-keywords' to force overriding all GNU-extension
  // keywords. This behavior is provided by GCC's poorly named '-fasm' flag,
  // while a subset (the non-C++ GNU keywords) is provided by GCC's
  // '-fgnu-keywords'. Clang conflates the two for simplicity under the single
  // name, as it doesn't seem a useful distinction.
  Opts.GNUKeywords = Args.hasFlag(OPT_fgnu_keywords, OPT_fno_gnu_keywords,
                                  Opts.GNUKeywords);

  if (Args.hasArg(OPT_fno_operator_names))
    Opts.CXXOperatorNames = 0;

  if (Opts.ObjC1) {
    if (Args.hasArg(OPT_fobjc_gc_only))
      Opts.setGC(LangOptions::GCOnly);
    else if (Args.hasArg(OPT_fobjc_gc))
      Opts.setGC(LangOptions::HybridGC);
    else if (Args.hasArg(OPT_fobjc_arc)) {
      Opts.ObjCAutoRefCount = 1;
      if (Args.hasArg(OPT_fobjc_fragile_abi))
        Diags.Report(diag::err_arc_nonfragile_abi);
    }

    if (Args.hasArg(OPT_fobjc_runtime_has_weak))
      Opts.ObjCRuntimeHasWeak = 1;

    if (Args.hasArg(OPT_fno_objc_infer_related_result_type))
      Opts.ObjCInferRelatedResultType = 0;
  }
    
  if (Args.hasArg(OPT_fgnu89_inline))
    Opts.GNUInline = 1;

  if (Args.hasArg(OPT_fapple_kext)) {
    if (!Opts.CPlusPlus)
      Diags.Report(diag::warn_c_kext);
    else
      Opts.AppleKext = 1;
  }

  if (Args.hasArg(OPT_print_ivar_layout))
    Opts.ObjCGCBitmapPrint = 1;
  if (Args.hasArg(OPT_fno_constant_cfstrings))
    Opts.NoConstantCFStrings = 1;

  if (Args.hasArg(OPT_faltivec))
    Opts.AltiVec = 1;

  if (Args.hasArg(OPT_pthread))
    Opts.POSIXThreads = 1;

  if (Args.hasArg(OPT_fdelayed_template_parsing))
    Opts.DelayedTemplateParsing = 1;

  StringRef Vis = Args.getLastArgValue(OPT_fvisibility, "default");
  if (Vis == "default")
    Opts.setVisibilityMode(DefaultVisibility);
  else if (Vis == "hidden")
    Opts.setVisibilityMode(HiddenVisibility);
  else if (Vis == "protected")
    Opts.setVisibilityMode(ProtectedVisibility);
  else
    Diags.Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_fvisibility)->getAsString(Args) << Vis;

  if (Args.hasArg(OPT_fvisibility_inlines_hidden))
    Opts.InlineVisibilityHidden = 1;

  if (Args.hasArg(OPT_ftrapv)) {
    Opts.setSignedOverflowBehavior(LangOptions::SOB_Trapping);
    // Set the handler, if one is specified.
    Opts.OverflowHandler =
        Args.getLastArgValue(OPT_ftrapv_handler);
  }
  else if (Args.hasArg(OPT_fwrapv))
    Opts.setSignedOverflowBehavior(LangOptions::SOB_Defined);

  if (Args.hasArg(OPT_trigraphs))
    Opts.Trigraphs = 1;

  Opts.DollarIdents = Args.hasFlag(OPT_fdollars_in_identifiers,
                                   OPT_fno_dollars_in_identifiers,
                                   Opts.DollarIdents);
  Opts.PascalStrings = Args.hasArg(OPT_fpascal_strings);
  Opts.MicrosoftExt = Args.hasArg(OPT_fms_extensions);
  Opts.MicrosoftMode = Args.hasArg(OPT_fms_compatibility);
  Opts.MSCVersion = Args.getLastArgIntValue(OPT_fmsc_version, 0, Diags);
  Opts.Borland = Args.hasArg(OPT_fborland_extensions);
  Opts.WritableStrings = Args.hasArg(OPT_fwritable_strings);
  Opts.ConstStrings = Args.hasFlag(OPT_fconst_strings, OPT_fno_const_strings,
                                   Opts.ConstStrings);
  if (Args.hasArg(OPT_fno_lax_vector_conversions))
    Opts.LaxVectorConversions = 0;
  if (Args.hasArg(OPT_fno_threadsafe_statics))
    Opts.ThreadsafeStatics = 0;
  Opts.Exceptions = Args.hasArg(OPT_fexceptions);
  Opts.ObjCExceptions = Args.hasArg(OPT_fobjc_exceptions);
  Opts.CXXExceptions = Args.hasArg(OPT_fcxx_exceptions);
  Opts.SjLjExceptions = Args.hasArg(OPT_fsjlj_exceptions);
  Opts.TraditionalCPP = Args.hasArg(OPT_traditional_cpp);

  Opts.RTTI = !Args.hasArg(OPT_fno_rtti);
  Opts.Blocks = Args.hasArg(OPT_fblocks);
  Opts.BlocksRuntimeOptional = Args.hasArg(OPT_fblocks_runtime_optional);
  Opts.CharIsSigned = !Args.hasArg(OPT_fno_signed_char);
  Opts.ShortWChar = Args.hasArg(OPT_fshort_wchar);
  Opts.ShortEnums = Args.hasArg(OPT_fshort_enums);
  Opts.Freestanding = Args.hasArg(OPT_ffreestanding);
  Opts.NoBuiltin = Args.hasArg(OPT_fno_builtin) || Opts.Freestanding;
  Opts.AssumeSaneOperatorNew = !Args.hasArg(OPT_fno_assume_sane_operator_new);
  Opts.HeinousExtensions = Args.hasArg(OPT_fheinous_gnu_extensions);
  Opts.AccessControl = !Args.hasArg(OPT_fno_access_control);
  Opts.ElideConstructors = !Args.hasArg(OPT_fno_elide_constructors);
  Opts.MathErrno = Args.hasArg(OPT_fmath_errno);
  Opts.InstantiationDepth = Args.getLastArgIntValue(OPT_ftemplate_depth, 1024,
                                               Diags);
  Opts.DelayedTemplateParsing = Args.hasArg(OPT_fdelayed_template_parsing);
  Opts.NumLargeByValueCopy = Args.getLastArgIntValue(OPT_Wlarge_by_value_copy,
                                                    0, Diags);
  Opts.MSBitfields = Args.hasArg(OPT_mms_bitfields);
  Opts.NeXTRuntime = !Args.hasArg(OPT_fgnu_runtime);
  Opts.ObjCConstantStringClass =
    Args.getLastArgValue(OPT_fconstant_string_class);
  Opts.ObjCNonFragileABI = !Args.hasArg(OPT_fobjc_fragile_abi);
  if (Opts.ObjCNonFragileABI)
    Opts.ObjCNonFragileABI2 = true;
  Opts.ObjCDefaultSynthProperties =
    Args.hasArg(OPT_fobjc_default_synthesize_properties);
  Opts.CatchUndefined = Args.hasArg(OPT_fcatch_undefined_behavior);
  Opts.EmitAllDecls = Args.hasArg(OPT_femit_all_decls);
  Opts.PackStruct = Args.getLastArgIntValue(OPT_fpack_struct, 0, Diags);
  Opts.PICLevel = Args.getLastArgIntValue(OPT_pic_level, 0, Diags);
  Opts.Static = Args.hasArg(OPT_static_define);
  Opts.DumpRecordLayouts = Args.hasArg(OPT_fdump_record_layouts);
  Opts.DumpVTableLayouts = Args.hasArg(OPT_fdump_vtable_layouts);
  Opts.SpellChecking = !Args.hasArg(OPT_fno_spell_checking);
  Opts.NoBitFieldTypeAlign = Args.hasArg(OPT_fno_bitfield_type_align);
  Opts.SinglePrecisionConstants = Args.hasArg(OPT_cl_single_precision_constant);
  Opts.FastRelaxedMath = Args.hasArg(OPT_cl_fast_relaxed_math);
  Opts.OptimizeSize = 0;
  Opts.MRTD = Args.hasArg(OPT_mrtd);
  Opts.FakeAddressSpaceMap = Args.hasArg(OPT_ffake_address_space_map);
  Opts.ParseUnknownAnytype = Args.hasArg(OPT_funknown_anytype);
  Opts.DebuggerSupport = Args.hasArg(OPT_fdebugger_support);

  // Record whether the __DEPRECATED define was requested.
  Opts.Deprecated = Args.hasFlag(OPT_fdeprecated_macro,
                                 OPT_fno_deprecated_macro,
                                 Opts.Deprecated);

  // FIXME: Eliminate this dependency.
  unsigned Opt = getOptimizationLevel(Args, IK, Diags);
  Opts.Optimize = Opt != 0;

  // This is the __NO_INLINE__ define, which just depends on things like the
  // optimization level and -fno-inline, not actually whether the backend has
  // inlining enabled.
  //
  // FIXME: This is affected by other options (-fno-inline).
  Opts.NoInline = !Opt;

  unsigned SSP = Args.getLastArgIntValue(OPT_stack_protector, 0, Diags);
  switch (SSP) {
  default:
    Diags.Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_stack_protector)->getAsString(Args) << SSP;
    break;
  case 0: Opts.setStackProtector(LangOptions::SSPOff); break;
  case 1: Opts.setStackProtector(LangOptions::SSPOn);  break;
  case 2: Opts.setStackProtector(LangOptions::SSPReq); break;
  }
}

static void ParsePreprocessorArgs(PreprocessorOptions &Opts, ArgList &Args,
                                  FileManager &FileMgr,
                                  DiagnosticsEngine &Diags) {
  using namespace cc1options;
  Opts.ImplicitPCHInclude = Args.getLastArgValue(OPT_include_pch);
  Opts.ImplicitPTHInclude = Args.getLastArgValue(OPT_include_pth);
  if (const Arg *A = Args.getLastArg(OPT_token_cache))
      Opts.TokenCache = A->getValue(Args);
  else
    Opts.TokenCache = Opts.ImplicitPTHInclude;
  Opts.UsePredefines = !Args.hasArg(OPT_undef);
  Opts.DetailedRecord = Args.hasArg(OPT_detailed_preprocessing_record);
  Opts.AutoModuleImport = Args.hasArg(OPT_fauto_module_import);
  Opts.DisablePCHValidation = Args.hasArg(OPT_fno_validate_pch);

  Opts.DumpDeserializedPCHDecls = Args.hasArg(OPT_dump_deserialized_pch_decls);
  for (arg_iterator it = Args.filtered_begin(OPT_error_on_deserialized_pch_decl),
         ie = Args.filtered_end(); it != ie; ++it) {
    const Arg *A = *it;
    Opts.DeserializedPCHDeclsToErrorOn.insert(A->getValue(Args));
  }

  if (const Arg *A = Args.getLastArg(OPT_preamble_bytes_EQ)) {
    StringRef Value(A->getValue(Args));
    size_t Comma = Value.find(',');
    unsigned Bytes = 0;
    unsigned EndOfLine = 0;

    if (Comma == StringRef::npos ||
        Value.substr(0, Comma).getAsInteger(10, Bytes) ||
        Value.substr(Comma + 1).getAsInteger(10, EndOfLine))
      Diags.Report(diag::err_drv_preamble_format);
    else {
      Opts.PrecompiledPreambleBytes.first = Bytes;
      Opts.PrecompiledPreambleBytes.second = (EndOfLine != 0);
    }
  }

  // Add macros from the command line.
  for (arg_iterator it = Args.filtered_begin(OPT_D, OPT_U),
         ie = Args.filtered_end(); it != ie; ++it) {
    if ((*it)->getOption().matches(OPT_D))
      Opts.addMacroDef((*it)->getValue(Args));
    else
      Opts.addMacroUndef((*it)->getValue(Args));
  }

  Opts.MacroIncludes = Args.getAllArgValues(OPT_imacros);

  // Add the ordered list of -includes.
  for (arg_iterator it = Args.filtered_begin(OPT_include, OPT_include_pch,
                                             OPT_include_pth),
         ie = Args.filtered_end(); it != ie; ++it) {
    const Arg *A = *it;
    // PCH is handled specially, we need to extra the original include path.
    if (A->getOption().matches(OPT_include_pch)) {
      std::string OriginalFile =
        ASTReader::getOriginalSourceFile(A->getValue(Args), FileMgr, Diags);
      if (OriginalFile.empty())
        continue;

      Opts.Includes.push_back(OriginalFile);
    } else
      Opts.Includes.push_back(A->getValue(Args));
  }

  for (arg_iterator it = Args.filtered_begin(OPT_chain_include),
         ie = Args.filtered_end(); it != ie; ++it) {
    const Arg *A = *it;
    Opts.ChainedIncludes.push_back(A->getValue(Args));
  }

  // Include 'altivec.h' if -faltivec option present
  if (Args.hasArg(OPT_faltivec))
    Opts.Includes.push_back("altivec.h");

  for (arg_iterator it = Args.filtered_begin(OPT_remap_file),
         ie = Args.filtered_end(); it != ie; ++it) {
    const Arg *A = *it;
    std::pair<StringRef,StringRef> Split =
      StringRef(A->getValue(Args)).split(';');

    if (Split.second.empty()) {
      Diags.Report(diag::err_drv_invalid_remap_file) << A->getAsString(Args);
      continue;
    }

    Opts.addRemappedFile(Split.first, Split.second);
  }
  
  if (Arg *A = Args.getLastArg(OPT_fobjc_arc_cxxlib_EQ)) {
    StringRef Name = A->getValue(Args);
    unsigned Library = llvm::StringSwitch<unsigned>(Name)
      .Case("libc++", ARCXX_libcxx)
      .Case("libstdc++", ARCXX_libstdcxx)
      .Case("none", ARCXX_nolib)
      .Default(~0U);
    if (Library == ~0U)
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Name;
    else
      Opts.ObjCXXARCStandardLibrary = (ObjCXXARCStandardLibraryKind)Library;
  }
}

static void ParsePreprocessorOutputArgs(PreprocessorOutputOptions &Opts,
                                        ArgList &Args) {
  using namespace cc1options;
  Opts.ShowCPP = !Args.hasArg(OPT_dM);
  Opts.ShowComments = Args.hasArg(OPT_C);
  Opts.ShowLineMarkers = !Args.hasArg(OPT_P);
  Opts.ShowMacroComments = Args.hasArg(OPT_CC);
  Opts.ShowMacros = Args.hasArg(OPT_dM) || Args.hasArg(OPT_dD);
}

static void ParseTargetArgs(TargetOptions &Opts, ArgList &Args) {
  using namespace cc1options;
  Opts.ABI = Args.getLastArgValue(OPT_target_abi);
  Opts.CXXABI = Args.getLastArgValue(OPT_cxx_abi);
  Opts.CPU = Args.getLastArgValue(OPT_target_cpu);
  Opts.Features = Args.getAllArgValues(OPT_target_feature);
  Opts.LinkerVersion = Args.getLastArgValue(OPT_target_linker_version);
  Opts.Triple = llvm::Triple::normalize(Args.getLastArgValue(OPT_triple));

  // Use the host triple if unspecified.
  if (Opts.Triple.empty())
    Opts.Triple = llvm::sys::getHostTriple();
}

//

void CompilerInvocation::CreateFromArgs(CompilerInvocation &Res,
                                        const char *const *ArgBegin,
                                        const char *const *ArgEnd,
                                        DiagnosticsEngine &Diags) {
  // Parse the arguments.
  llvm::OwningPtr<OptTable> Opts(createCC1OptTable());
  unsigned MissingArgIndex, MissingArgCount;
  llvm::OwningPtr<InputArgList> Args(
    Opts->ParseArgs(ArgBegin, ArgEnd,MissingArgIndex, MissingArgCount));

  // Check for missing argument error.
  if (MissingArgCount)
    Diags.Report(diag::err_drv_missing_argument)
      << Args->getArgString(MissingArgIndex) << MissingArgCount;

  // Issue errors on unknown arguments.
  for (arg_iterator it = Args->filtered_begin(OPT_UNKNOWN),
         ie = Args->filtered_end(); it != ie; ++it)
    Diags.Report(diag::err_drv_unknown_argument) << (*it)->getAsString(*Args);

  ParseAnalyzerArgs(Res.getAnalyzerOpts(), *Args, Diags);
  ParseDependencyOutputArgs(Res.getDependencyOutputOpts(), *Args);
  ParseDiagnosticArgs(Res.getDiagnosticOpts(), *Args, Diags);
  ParseFileSystemArgs(Res.getFileSystemOpts(), *Args);
  // FIXME: We shouldn't have to pass the DashX option around here
  InputKind DashX = ParseFrontendArgs(Res.getFrontendOpts(), *Args, Diags);
  ParseCodeGenArgs(Res.getCodeGenOpts(), *Args, DashX, Diags);
  ParseHeaderSearchArgs(Res.getHeaderSearchOpts(), *Args);
  if (DashX != IK_AST && DashX != IK_LLVM_IR) {
    ParseLangArgs(Res.getLangOpts(), *Args, DashX, Diags);
    if (Res.getFrontendOpts().ProgramAction == frontend::RewriteObjC)
      Res.getLangOpts().ObjCExceptions = 1;
  }
  // FIXME: ParsePreprocessorArgs uses the FileManager to read the contents of
  // PCH file and find the original header name. Remove the need to do that in
  // ParsePreprocessorArgs and remove the FileManager 
  // parameters from the function and the "FileManager.h" #include.
  FileManager FileMgr(Res.getFileSystemOpts());
  ParsePreprocessorArgs(Res.getPreprocessorOpts(), *Args, FileMgr, Diags);
  ParsePreprocessorOutputArgs(Res.getPreprocessorOutputOpts(), *Args);
  ParseTargetArgs(Res.getTargetOpts(), *Args);
}

namespace {

  class ModuleSignature {
    llvm::SmallVector<uint64_t, 16> Data;
    unsigned CurBit;
    uint64_t CurValue;
    
  public:
    ModuleSignature() : CurBit(0), CurValue(0) { }
    
    void add(uint64_t Value, unsigned Bits);
    void add(StringRef Value);
    void flush();
    
    llvm::APInt getAsInteger() const;
  };
}

void ModuleSignature::add(uint64_t Value, unsigned int NumBits) {
  CurValue |= Value << CurBit;
  if (CurBit + NumBits < 64) {
    CurBit += NumBits;
    return;
  }
  
  // Add the current word.
  Data.push_back(CurValue);
  
  if (CurBit)
    CurValue = Value >> (64-CurBit);
  else
    CurValue = 0;
  CurBit = (CurBit+NumBits) & 63;
}

void ModuleSignature::flush() {
  if (CurBit == 0)
    return;
  
  Data.push_back(CurValue);
  CurBit = 0;
  CurValue = 0;
}

void ModuleSignature::add(StringRef Value) {
  for (StringRef::iterator I = Value.begin(), IEnd = Value.end(); I != IEnd;++I)
    add(*I, 8);
}

llvm::APInt ModuleSignature::getAsInteger() const {
  return llvm::APInt(Data.size() * 64, Data);
}

std::string CompilerInvocation::getModuleHash() const {
  ModuleSignature Signature;
  
  // Start the signature with the compiler version.
  Signature.add(getClangFullRepositoryVersion());
  
  // Extend the signature with the language options
#define LANGOPT(Name, Bits, Default, Description) \
  Signature.add(LangOpts.Name, Bits);
#define ENUM_LANGOPT(Name, Type, Bits, Default, Description) \
  Signature.add(static_cast<unsigned>(LangOpts.get##Name()), Bits);
#define BENIGN_LANGOPT(Name, Bits, Default, Description)
#define BENIGN_ENUM_LANGOPT(Name, Type, Bits, Default, Description)
#include "clang/Basic/LangOptions.def"
  
  // Extend the signature with the target triple
  llvm::Triple T(TargetOpts.Triple);
  Signature.add((unsigned)T.getArch(), 5);
  Signature.add((unsigned)T.getVendor(), 4);
  Signature.add((unsigned)T.getOS(), 5);
  Signature.add((unsigned)T.getEnvironment(), 4);

  // Extend the signature with preprocessor options.
  Signature.add(getPreprocessorOpts().UsePredefines, 1);
  Signature.add(getPreprocessorOpts().DetailedRecord, 1);
  
  // Hash the preprocessor defines.
  // FIXME: This is terrible. Use an MD5 sum of the preprocessor defines.
  std::vector<StringRef> MacroDefs;
  for (std::vector<std::pair<std::string, bool/*isUndef*/> >::const_iterator 
            I = getPreprocessorOpts().Macros.begin(),
         IEnd = getPreprocessorOpts().Macros.end();
       I != IEnd; ++I) {
    if (!I->second)
      MacroDefs.push_back(I->first);
  }
  llvm::array_pod_sort(MacroDefs.begin(), MacroDefs.end());
       
  unsigned PPHashResult = 0;
  for (unsigned I = 0, N = MacroDefs.size(); I != N; ++I)
    PPHashResult = llvm::HashString(MacroDefs[I], PPHashResult);
  Signature.add(PPHashResult, 32);
  
  // We've generated the signature. Treat it as one large APInt that we'll
  // encode in base-36 and return.
  Signature.flush();
  return Signature.getAsInteger().toString(36, /*Signed=*/false);
}
