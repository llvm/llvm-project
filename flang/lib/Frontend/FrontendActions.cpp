//===--- FrontendActions.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendActions.h"
#include "flang/Common/default-kinds.h"
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/FrontendOptions.h"
#include "flang/Frontend/PreprocessorOptions.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Support/Verifier.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/source.h"
#include "flang/Parser/unparse.h"
#include "flang/Semantics/runtime-type-info.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/unparse-with-symbols.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"
#include <clang/Basic/Diagnostic.h>
#include <memory>

using namespace Fortran::frontend;

//===----------------------------------------------------------------------===//
// Custom BeginSourceFileAction
//===----------------------------------------------------------------------===//
bool PrescanAction::BeginSourceFileAction() { return RunPrescan(); }

bool PrescanAndParseAction::BeginSourceFileAction() {
  return RunPrescan() && RunParse();
}

bool PrescanAndSemaAction::BeginSourceFileAction() {
  return RunPrescan() && RunParse() && RunSemanticChecks() &&
      GenerateRtTypeTables();
}

bool PrescanAndSemaDebugAction::BeginSourceFileAction() {
  // This is a "debug" action for development purposes. To facilitate this, the
  // semantic checks are made to succeed unconditionally to prevent this action
  // from exiting early (i.e. in the presence of semantic errors). We should
  // never do this in actions intended for end-users or otherwise regular
  // compiler workflows!
  return RunPrescan() && RunParse() && (RunSemanticChecks() || true) &&
      (GenerateRtTypeTables() || true);
}

bool CodeGenAction::BeginSourceFileAction() {
  bool res = RunPrescan() && RunParse() && RunSemanticChecks();
  if (!res)
    return res;

  CompilerInstance &ci = this->instance();

  // Load the MLIR dialects required by Flang
  mlir::DialectRegistry registry;
  mlirCtx = std::make_unique<mlir::MLIRContext>(registry);
  fir::support::registerNonCodegenDialects(registry);
  fir::support::loadNonCodegenDialects(*mlirCtx);

  // Create a LoweringBridge
  const common::IntrinsicTypeDefaultKinds &defKinds =
      ci.invocation().semanticsContext().defaultKinds();
  fir::KindMapping kindMap(mlirCtx.get(),
      llvm::ArrayRef<fir::KindTy>{fir::fromDefaultKinds(defKinds)});
  lower::LoweringBridge lb = Fortran::lower::LoweringBridge::create(*mlirCtx,
      defKinds, ci.invocation().semanticsContext().intrinsics(),
      ci.parsing().allCooked(), ci.invocation().targetOpts().triple, kindMap);

  // Create a parse tree and lower it to FIR
  Fortran::parser::Program &parseTree{*ci.parsing().parseTree()};
  lb.lower(parseTree, ci.invocation().semanticsContext());
  mlirModule = std::make_unique<mlir::ModuleOp>(lb.getModule());

  // Run the default passes.
  mlir::PassManager pm(mlirCtx.get(), mlir::OpPassManager::Nesting::Implicit);
  pm.enableVerifier(/*verifyPasses=*/true);
  pm.addPass(std::make_unique<Fortran::lower::VerifierPass>());

  if (mlir::failed(pm.run(*mlirModule))) {
    unsigned diagID =
        ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
            "verification of lowering to FIR failed");
    ci.diagnostics().Report(diagID);
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Custom ExecuteAction
//===----------------------------------------------------------------------===//
void InputOutputTestAction::ExecuteAction() {
  CompilerInstance &ci = instance();

  // Create a stream for errors
  std::string buf;
  llvm::raw_string_ostream error_stream{buf};

  // Read the input file
  Fortran::parser::AllSources &allSources{ci.allSources()};
  std::string path{GetCurrentFileOrBufferName()};
  const Fortran::parser::SourceFile *sf;
  if (path == "-")
    sf = allSources.ReadStandardInput(error_stream);
  else
    sf = allSources.Open(path, error_stream, std::optional<std::string>{"."s});
  llvm::ArrayRef<char> fileContent = sf->content();

  // Output file descriptor to receive the contents of the input file.
  std::unique_ptr<llvm::raw_ostream> os;

  // Copy the contents from the input file to the output file
  if (!ci.IsOutputStreamNull()) {
    // An output stream (outputStream_) was set earlier
    ci.WriteOutputStream(fileContent.data());
  } else {
    // No pre-set output stream - create an output file
    os = ci.CreateDefaultOutputFile(
        /*binary=*/true, GetCurrentFileOrBufferName(), "txt");
    if (!os)
      return;
    (*os) << fileContent.data();
  }
}

void PrintPreprocessedAction::ExecuteAction() {
  std::string buf;
  llvm::raw_string_ostream outForPP{buf};

  // Format or dump the prescanner's output
  CompilerInstance &ci = this->instance();
  if (ci.invocation().preprocessorOpts().noReformat) {
    ci.parsing().DumpCookedChars(outForPP);
  } else {
    ci.parsing().EmitPreprocessedSource(
        outForPP, !ci.invocation().preprocessorOpts().noLineDirectives);
  }

  // Print diagnostics from the prescanner
  ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

  // If a pre-defined output stream exists, dump the preprocessed content there
  if (!ci.IsOutputStreamNull()) {
    // Send the output to the pre-defined output buffer.
    ci.WriteOutputStream(outForPP.str());
    return;
  }

  // Create a file and save the preprocessed output there
  std::unique_ptr<llvm::raw_pwrite_stream> os{ci.CreateDefaultOutputFile(
      /*Binary=*/true, /*InFile=*/GetCurrentFileOrBufferName())};
  if (!os) {
    return;
  }

  (*os) << outForPP.str();
}

void DebugDumpProvenanceAction::ExecuteAction() {
  this->instance().parsing().DumpProvenance(llvm::outs());
}

void ParseSyntaxOnlyAction::ExecuteAction() {
}

void DebugUnparseNoSemaAction::ExecuteAction() {
  auto &invoc = this->instance().invocation();
  auto &parseTree{instance().parsing().parseTree()};

  // TODO: Options should come from CompilerInvocation
  Unparse(llvm::outs(), *parseTree,
      /*encoding=*/Fortran::parser::Encoding::UTF_8,
      /*capitalizeKeywords=*/true, /*backslashEscapes=*/false,
      /*preStatement=*/nullptr,
      invoc.useAnalyzedObjectsForUnparse() ? &invoc.asFortran() : nullptr);
}

void DebugUnparseAction::ExecuteAction() {
  auto &invoc = this->instance().invocation();
  auto &parseTree{instance().parsing().parseTree()};

  CompilerInstance &ci = this->instance();
  auto os{ci.CreateDefaultOutputFile(
      /*Binary=*/false, /*InFile=*/GetCurrentFileOrBufferName())};

  // TODO: Options should come from CompilerInvocation
  Unparse(*os, *parseTree,
      /*encoding=*/Fortran::parser::Encoding::UTF_8,
      /*capitalizeKeywords=*/true, /*backslashEscapes=*/false,
      /*preStatement=*/nullptr,
      invoc.useAnalyzedObjectsForUnparse() ? &invoc.asFortran() : nullptr);

  // Report fatal semantic errors
  reportFatalSemanticErrors();
}

void DebugUnparseWithSymbolsAction::ExecuteAction() {
  auto &parseTree{*instance().parsing().parseTree()};

  Fortran::semantics::UnparseWithSymbols(
      llvm::outs(), parseTree, /*encoding=*/Fortran::parser::Encoding::UTF_8);

  // Report fatal semantic errors
  reportFatalSemanticErrors();
}

void DebugDumpSymbolsAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  if (!ci.getRtTyTables().schemata) {
    unsigned DiagID =
        ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
            "could not find module file for __fortran_type_info");
    ci.diagnostics().Report(DiagID);
    llvm::errs() << "\n";
    return;
  }

  // Dump symbols
  ci.semantics().DumpSymbols(llvm::outs());
}

void DebugDumpAllAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  // Dump parse tree
  auto &parseTree{instance().parsing().parseTree()};
  llvm::outs() << "========================";
  llvm::outs() << " Flang: parse tree dump ";
  llvm::outs() << "========================\n";
  Fortran::parser::DumpTree(
      llvm::outs(), parseTree, &ci.invocation().asFortran());

  if (!ci.getRtTyTables().schemata) {
    unsigned DiagID =
        ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
            "could not find module file for __fortran_type_info");
    ci.diagnostics().Report(DiagID);
    llvm::errs() << "\n";
    return;
  }

  // Dump symbols
  llvm::outs() << "=====================";
  llvm::outs() << " Flang: symbols dump ";
  llvm::outs() << "=====================\n";
  ci.semantics().DumpSymbols(llvm::outs());
}

void DebugDumpParseTreeNoSemaAction::ExecuteAction() {
  auto &parseTree{instance().parsing().parseTree()};

  // Dump parse tree
  Fortran::parser::DumpTree(
      llvm::outs(), parseTree, &this->instance().invocation().asFortran());
}

void DebugDumpParseTreeAction::ExecuteAction() {
  auto &parseTree{instance().parsing().parseTree()};

  // Dump parse tree
  Fortran::parser::DumpTree(
      llvm::outs(), parseTree, &this->instance().invocation().asFortran());

  // Report fatal semantic errors
  reportFatalSemanticErrors();
}

void DebugMeasureParseTreeAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  // Parse. In case of failure, report and return.
  ci.parsing().Parse(llvm::outs());

  if (!ci.parsing().messages().empty() &&
      (ci.invocation().warnAsErr() ||
          ci.parsing().messages().AnyFatalError())) {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Could not parse %0");
    ci.diagnostics().Report(diagID) << GetCurrentFileOrBufferName();

    ci.parsing().messages().Emit(
        llvm::errs(), this->instance().allCookedSources());
    return;
  }

  // Report the diagnostics from parsing
  ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

  auto &parseTree{*ci.parsing().parseTree()};

  // Measure the parse tree
  MeasurementVisitor visitor;
  Fortran::parser::Walk(parseTree, visitor);
  llvm::outs() << "Parse tree comprises " << visitor.objects
               << " objects and occupies " << visitor.bytes
               << " total bytes.\n";
}

void DebugPreFIRTreeAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors()) {
    return;
  }

  auto &parseTree{*ci.parsing().parseTree()};

  // Dump pre-FIR tree
  if (auto ast{Fortran::lower::createPFT(
          parseTree, ci.invocation().semanticsContext())}) {
    Fortran::lower::dumpPFT(llvm::outs(), *ast);
  } else {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Pre FIR Tree is NULL.");
    ci.diagnostics().Report(diagID);
  }
}

void DebugDumpParsingLogAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  ci.parsing().Parse(llvm::errs());
  ci.parsing().DumpParsingLog(llvm::outs());
}

void GetDefinitionAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors()) {
    return;
  }

  parser::AllCookedSources &cs = ci.allCookedSources();
  unsigned diagID = ci.diagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Error, "Symbol not found");

  auto gdv = ci.invocation().frontendOpts().getDefVals;
  auto charBlock{cs.GetCharBlockFromLineAndColumns(
      gdv.line, gdv.startColumn, gdv.endColumn)};
  if (!charBlock) {
    ci.diagnostics().Report(diagID);
    return;
  }

  llvm::outs() << "String range: >" << charBlock->ToString() << "<\n";

  auto *symbol{ci.invocation()
                   .semanticsContext()
                   .FindScope(*charBlock)
                   .FindSymbol(*charBlock)};
  if (!symbol) {
    ci.diagnostics().Report(diagID);
    return;
  }

  llvm::outs() << "Found symbol name: " << symbol->name().ToString() << "\n";

  auto sourceInfo{cs.GetSourcePositionRange(symbol->name())};
  if (!sourceInfo) {
    llvm_unreachable(
        "Failed to obtain SourcePosition."
        "TODO: Please, write a test and replace this with a diagnostic!");
    return;
  }

  llvm::outs() << "Found symbol name: " << symbol->name().ToString() << "\n";
  llvm::outs() << symbol->name().ToString() << ": "
               << sourceInfo->first.file.path() << ", "
               << sourceInfo->first.line << ", " << sourceInfo->first.column
               << "-" << sourceInfo->second.column << "\n";
}

void GetSymbolsSourcesAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors()) {
    return;
  }

  ci.semantics().DumpSymbolsSources(llvm::outs());
}

#include "flang/Tools/CLOptions.inc"

// Lower the previously generated MLIR module into an LLVM IR module
void CodeGenAction::GenerateLLVMIR() {
  assert(mlirModule && "The MLIR module has not been generated yet.");

  CompilerInstance &ci = this->instance();

  fir::support::loadDialects(*mlirCtx);
  fir::support::registerLLVMTranslation(*mlirCtx);

  // Set-up the MLIR pass manager
  mlir::PassManager pm(mlirCtx.get(), mlir::OpPassManager::Nesting::Implicit);

  pm.addPass(std::make_unique<Fortran::lower::VerifierPass>());
  pm.enableVerifier(/*verifyPasses=*/true);

  // Create the pass pipeline
  fir::createMLIRToLLVMPassPipeline(pm);
  mlir::applyPassManagerCLOptions(pm);

  // Run the pass manager
  if (!mlir::succeeded(pm.run(*mlirModule))) {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Lowering to LLVM IR failed");
    ci.diagnostics().Report(diagID);
  }

  // Translate to LLVM IR
  llvm::Optional<llvm::StringRef> moduleName = mlirModule->getName();
  llvmCtx = std::make_unique<llvm::LLVMContext>();
  llvmModule = mlir::translateModuleToLLVMIR(
      *mlirModule, *llvmCtx, moduleName ? *moduleName : "FIRModule");

  if (!llvmModule) {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "failed to create the LLVM module");
    ci.diagnostics().Report(diagID);
    return;
  }
}

void EmitLLVMAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  GenerateLLVMIR();

  // If set, use the predefined outupt stream to print the generated module.
  if (!ci.IsOutputStreamNull()) {
    llvmModule->print(
        ci.GetOutputStream(), /*AssemblyAnnotationWriter=*/nullptr);
    return;
  }

  // No predefined output stream was set. Create an output file and dump the
  // generated module there.
  std::unique_ptr<llvm::raw_ostream> os = ci.CreateDefaultOutputFile(
      /*Binary=*/false, /*InFile=*/GetCurrentFileOrBufferName(), "ll");
  if (!os) {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "failed to create the output file");
    ci.diagnostics().Report(diagID);
    return;
  }
  llvmModule->print(*os, /*AssemblyAnnotationWriter=*/nullptr);
}

void EmitLLVMBitcodeAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  // Generate an LLVM module if it's not already present (it will already be
  // present if the input file is an LLVM IR/BC file).
  if (!llvmModule)
    GenerateLLVMIR();

  // Create and configure `Target`
  std::string error;
  std::string theTriple = llvmModule->getTargetTriple();
  const llvm::Target *theTarget =
      llvm::TargetRegistry::lookupTarget(theTriple, error);
  assert(theTarget && "Failed to create Target");

  // Create and configure `TargetMachine`
  std::unique_ptr<llvm::TargetMachine> TM(
      theTarget->createTargetMachine(theTriple, /*CPU=*/"",
          /*Features=*/"", llvm::TargetOptions(), llvm::None));
  assert(TM && "Failed to create TargetMachine");
  llvmModule->setDataLayout(TM->createDataLayout());

  // Generate an output file
  std::unique_ptr<llvm::raw_ostream> os = ci.CreateDefaultOutputFile(
      /*Binary=*/true, /*InFile=*/GetCurrentFileOrBufferName(), "bc");
  if (!os) {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "failed to create the output file");
    ci.diagnostics().Report(diagID);
    return;
  }

  // Set-up the pass manager
  llvm::ModulePassManager MPM;
  llvm::ModuleAnalysisManager MAM;
  llvm::PassBuilder PB(TM.get());
  PB.registerModuleAnalyses(MAM);
  MPM.addPass(llvm::BitcodeWriterPass(*os));

  // Run the passes
  MPM.run(*llvmModule, MAM);
}

void EmitMLIRAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  // Print the output. If a pre-defined output stream exists, dump the MLIR
  // content there.
  if (!ci.IsOutputStreamNull()) {
    mlirModule->print(ci.GetOutputStream());
    return;
  }

  // ... otherwise, print to a file.
  std::unique_ptr<llvm::raw_pwrite_stream> os{ci.CreateDefaultOutputFile(
      /*Binary=*/true, /*InFile=*/GetCurrentFileOrBufferName(), "mlir")};
  if (!os) {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "failed to create the output file");
    ci.diagnostics().Report(diagID);
    return;
  }

  mlirModule->print(*os);
}

void BackendAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  // Generate an LLVM module if it's not already present (it will already be
  // present if the input file is an LLVM IR/BC file).
  if (!llvmModule)
    GenerateLLVMIR();

  // Create `Target`
  std::string error;
  const std::string &theTriple = llvmModule->getTargetTriple();
  const llvm::Target *theTarget =
      llvm::TargetRegistry::lookupTarget(theTriple, error);
  // TODO: Make this a diagnostic once `flang-new` can consume LLVM IR files
  // (in which users could use unsupported triples)
  assert(theTarget && "Failed to create Target");

  // Create `TargetMachine`
  std::unique_ptr<llvm::TargetMachine> TM(
      theTarget->createTargetMachine(theTriple, /*CPU=*/"",
          /*Features=*/"", llvm::TargetOptions(), llvm::None));
  assert(TM && "Failed to create TargetMachine");
  llvmModule->setDataLayout(TM->createDataLayout());

  // If the output stream is a file, generate it and define the corresponding
  // output stream. If a pre-defined output stream is available, we will use
  // that instead.
  //
  // NOTE: `os` is a smart pointer that will be destroyed at the end of this
  // method. However, it won't be written to until `CodeGenPasses` is
  // destroyed. By defining `os` before `CodeGenPasses`, we make sure that the
  // output stream won't be destroyed before it is written to. This only
  // applies when an output file is used (i.e. there is no pre-defined output
  // stream).
  // TODO: Revisit once the new PM is ready (i.e. when `CodeGenPasses` is
  // updated to use it).
  std::unique_ptr<llvm::raw_pwrite_stream> os;
  if (ci.IsOutputStreamNull()) {
    // Get the output buffer/file
    switch (action) {
    case BackendActionTy::Backend_EmitAssembly:
      os = ci.CreateDefaultOutputFile(
          /*Binary=*/false, /*InFile=*/GetCurrentFileOrBufferName(), "s");
      break;
    case BackendActionTy::Backend_EmitObj:
      os = ci.CreateDefaultOutputFile(
          /*Binary=*/true, /*InFile=*/GetCurrentFileOrBufferName(), "o");
      break;
    }
    if (!os) {
      unsigned diagID = ci.diagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "failed to create the output file");
      ci.diagnostics().Report(diagID);
      return;
    }
  }

  // Create an LLVM code-gen pass pipeline. Currently only the legacy pass
  // manager is supported.
  // TODO: Switch to the new PM once it's available in the backend.
  llvm::legacy::PassManager CodeGenPasses;
  CodeGenPasses.add(
      createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));
  llvm::Triple triple(theTriple);

  std::unique_ptr<llvm::TargetLibraryInfoImpl> TLII =
      std::make_unique<llvm::TargetLibraryInfoImpl>(triple);
  assert(TLII && "Failed to create TargetLibraryInfo");
  CodeGenPasses.add(new llvm::TargetLibraryInfoWrapperPass(*TLII));

  llvm::CodeGenFileType cgft = (action == BackendActionTy::Backend_EmitAssembly)
      ? llvm::CodeGenFileType::CGFT_AssemblyFile
      : llvm::CodeGenFileType::CGFT_ObjectFile;
  if (TM->addPassesToEmitFile(CodeGenPasses,
          ci.IsOutputStreamNull() ? *os : ci.GetOutputStream(), nullptr,
          cgft)) {
    unsigned diagID =
        ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
            "emission of this file type is not supported");
    ci.diagnostics().Report(diagID);
    return;
  }

  // Run the code-gen passes
  CodeGenPasses.run(*llvmModule);
}

void InitOnlyAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  unsigned DiagID =
      ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Warning,
          "Use `-init-only` for testing purposes only");
  ci.diagnostics().Report(DiagID);
}

void PluginParseTreeAction::ExecuteAction() {}

void DebugDumpPFTAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  if (auto ast = Fortran::lower::createPFT(
          *ci.parsing().parseTree(), ci.semantics().context())) {
    Fortran::lower::dumpPFT(llvm::outs(), *ast);
    return;
  }

  unsigned DiagID = ci.diagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Error, "Pre FIR Tree is NULL.");
  ci.diagnostics().Report(DiagID);
}
