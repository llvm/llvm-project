//===--- FrontendActions.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendActions.h"
#include "flang/Common/default-kinds.h"
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/FrontendOptions.h"
#include "flang/Frontend/PreprocessorOptions.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Support/Verifier.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Passes/Pipelines.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/source.h"
#include "flang/Parser/unparse.h"
#include "flang/Semantics/runtime-type-info.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/unparse-with-symbols.h"
#include "flang/Tools/CrossToolHelpers.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/RISCVISAInfo.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <memory>
#include <system_error>

namespace llvm {
extern cl::opt<bool> PrintPipelinePasses;
} // namespace llvm

using namespace Fortran::frontend;

// Declare plugin extension function declarations.
#define HANDLE_EXTENSION(Ext)                                                  \
  llvm::PassPluginLibraryInfo get##Ext##PluginInfo();
#include "llvm/Support/Extension.def"

/// Save the given \c mlirModule to a temporary .mlir file, in a location
/// decided by the -save-temps flag. No files are produced if the flag is not
/// specified.
static bool saveMLIRTempFile(const CompilerInvocation &ci,
                             mlir::ModuleOp mlirModule,
                             llvm::StringRef inputFile,
                             llvm::StringRef outputTag) {
  if (!ci.getCodeGenOpts().SaveTempsDir.has_value())
    return true;

  const llvm::StringRef compilerOutFile = ci.getFrontendOpts().outputFile;
  const llvm::StringRef saveTempsDir = ci.getCodeGenOpts().SaveTempsDir.value();
  auto dir = llvm::StringSwitch<llvm::StringRef>(saveTempsDir)
                 .Case("cwd", "")
                 .Case("obj", llvm::sys::path::parent_path(compilerOutFile))
                 .Default(saveTempsDir);

  // Build path from the compiler output file name, triple, cpu and OpenMP
  // information
  llvm::SmallString<256> path(dir);
  llvm::sys::path::append(path, llvm::sys::path::stem(inputFile) + "-" +
                                    outputTag + ".mlir");

  std::error_code ec;
  llvm::ToolOutputFile out(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return false;

  mlirModule->print(out.os());
  out.os().close();
  out.keep();

  return true;
}

//===----------------------------------------------------------------------===//
// Custom BeginSourceFileAction
//===----------------------------------------------------------------------===//

bool PrescanAction::beginSourceFileAction() { return runPrescan(); }

bool PrescanAndParseAction::beginSourceFileAction() {
  return runPrescan() && runParse(/*emitMessages=*/true);
}

bool PrescanAndSemaAction::beginSourceFileAction() {
  return runPrescan() && runParse(/*emitMessages=*/false) &&
         runSemanticChecks() && generateRtTypeTables();
}

bool PrescanAndSemaDebugAction::beginSourceFileAction() {
  // This is a "debug" action for development purposes. To facilitate this, the
  // semantic checks are made to succeed unconditionally to prevent this action
  // from exiting early (i.e. in the presence of semantic errors). We should
  // never do this in actions intended for end-users or otherwise regular
  // compiler workflows!
  return runPrescan() && runParse(/*emitMessages=*/false) &&
         (runSemanticChecks() || true) && (generateRtTypeTables() || true);
}

static void addDependentLibs(mlir::ModuleOp &mlirModule, CompilerInstance &ci) {
  const std::vector<std::string> &libs =
      ci.getInvocation().getCodeGenOpts().DependentLibs;
  if (libs.empty()) {
    return;
  }
  // dependent-lib is currently only supported on Windows, so the list should be
  // empty on non-Windows platforms
  assert(
      llvm::Triple(ci.getInvocation().getTargetOpts().triple).isOSWindows() &&
      "--dependent-lib is only supported on Windows");
  // Add linker options specified by --dependent-lib
  auto builder = mlir::OpBuilder(mlirModule.getRegion());
  for (const std::string &lib : libs) {
    builder.create<mlir::LLVM::LinkerOptionsOp>(
        mlirModule.getLoc(), builder.getStrArrayAttr({"/DEFAULTLIB:" + lib}));
  }
}

// Add to MLIR code target specific items which are dependent on target
// configuration specified by the user.
// Clang equivalent function: AMDGPUTargetCodeGenInfo::emitTargetGlobals
static void addAMDGPUSpecificMLIRItems(mlir::ModuleOp &mlirModule,
                                       CompilerInstance &ci) {
  const TargetOptions &targetOpts = ci.getInvocation().getTargetOpts();
  const llvm::Triple triple(targetOpts.triple);
  const llvm::StringRef codeObjectVersionGlobalOpName = "__oclc_ABI_version";

  if (!triple.isAMDGPU()) {
    return;
  }
  const CodeGenOptions &codeGenOpts = ci.getInvocation().getCodeGenOpts();
  if (codeGenOpts.CodeObjectVersion == llvm::CodeObjectVersionKind::COV_None) {
    return;
  }

  mlir::IRRewriter builder(mlirModule.getContext());
  unsigned oclcABIVERsion = codeGenOpts.CodeObjectVersion;
  auto int32Type = builder.getI32Type();

  std::optional<mlir::LLVM::GlobalOp> originalGV;

  mlirModule.walk([&originalGV, codeObjectVersionGlobalOpName](
                      mlir::LLVM::GlobalOp globalOp) {
    if (globalOp.getName() == codeObjectVersionGlobalOpName)
      originalGV = globalOp;
  });
  if (originalGV.has_value()) {
    mlir::LLVM::GlobalOp originalGVOp = originalGV.value();
    if (originalGVOp.getLinkage() != mlir::LLVM::Linkage::External) {
      return;
    }
    // Update the variable if it is already present in MLIR but it was marked
    // as external linkage variable
    originalGVOp.setLinkage(mlir::LLVM::Linkage::WeakODR);
    originalGVOp.setValueAttr(
        builder.getIntegerAttr(int32Type, oclcABIVERsion));
    originalGVOp.setUnnamedAddr(mlir::LLVM::UnnamedAddr::Local);
    originalGVOp.setAddrSpace(llvm::AMDGPUAS::CONSTANT_ADDRESS);
    originalGVOp.setVisibility_(mlir::LLVM::Visibility::Hidden);
    return;
  }

  mlir::LLVM::GlobalOp covInfo = builder.create<mlir::LLVM::GlobalOp>(
      /* Location */ mlirModule.getLoc(), /* Type */ int32Type,
      /* IsConstant */ true, /* Linkage */ mlir::LLVM::Linkage::WeakODR,
      /* Name */ codeObjectVersionGlobalOpName,
      /* Value */ builder.getIntegerAttr(int32Type, oclcABIVERsion));
  covInfo.setUnnamedAddr(mlir::LLVM::UnnamedAddr::Local);
  covInfo.setAddrSpace(llvm::AMDGPUAS::CONSTANT_ADDRESS);
  covInfo.setVisibility_(mlir::LLVM::Visibility::Hidden);
  builder.setInsertionPointToStart(mlirModule.getBody());
  builder.insert(covInfo);
}

bool CodeGenAction::beginSourceFileAction() {
  llvmCtx = std::make_unique<llvm::LLVMContext>();
  CompilerInstance &ci = this->getInstance();

  // If the input is an LLVM file, just parse it and return.
  if (this->getCurrentInput().getKind().getLanguage() == Language::LLVM_IR) {
    llvm::SMDiagnostic err;
    llvmModule = llvm::parseIRFile(getCurrentInput().getFile(), err, *llvmCtx);
    if (!llvmModule || llvm::verifyModule(*llvmModule, &llvm::errs())) {
      err.print("flang-new", llvm::errs());
      unsigned diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "Could not parse IR");
      ci.getDiagnostics().Report(diagID);
      return false;
    }

    return true;
  }

  // Load the MLIR dialects required by Flang
  mlir::DialectRegistry registry;
  mlirCtx = std::make_unique<mlir::MLIRContext>(registry);
  fir::support::registerNonCodegenDialects(registry);
  fir::support::loadNonCodegenDialects(*mlirCtx);
  fir::support::loadDialects(*mlirCtx);
  fir::support::registerLLVMTranslation(*mlirCtx);

  const llvm::TargetMachine &targetMachine = ci.getTargetMachine();

  // If the input is an MLIR file, just parse it and return.
  if (this->getCurrentInput().getKind().getLanguage() == Language::MLIR) {
    llvm::SourceMgr sourceMgr;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(getCurrentInput().getFile());
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, mlirCtx.get());

    if (!module || mlir::failed(module->verifyInvariants())) {
      unsigned diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "Could not parse FIR");
      ci.getDiagnostics().Report(diagID);
      return false;
    }

    mlirModule = std::make_unique<mlir::ModuleOp>(module.release());
    const llvm::DataLayout &dl = targetMachine.createDataLayout();
    fir::support::setMLIRDataLayout(*mlirModule, dl);
    return true;
  }

  // Otherwise, generate an MLIR module from the input Fortran source
  if (getCurrentInput().getKind().getLanguage() != Language::Fortran) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Invalid input type - expecting a Fortran file");
    ci.getDiagnostics().Report(diagID);
    return false;
  }
  bool res = runPrescan() && runParse(/*emitMessages=*/false) &&
             runSemanticChecks() && generateRtTypeTables();
  if (!res)
    return res;

  // Create a LoweringBridge
  const common::IntrinsicTypeDefaultKinds &defKinds =
      ci.getSemanticsContext().defaultKinds();
  fir::KindMapping kindMap(mlirCtx.get(), llvm::ArrayRef<fir::KindTy>{
                                              fir::fromDefaultKinds(defKinds)});
  lower::LoweringBridge lb = Fortran::lower::LoweringBridge::create(
      *mlirCtx, ci.getSemanticsContext(), defKinds,
      ci.getSemanticsContext().intrinsics(),
      ci.getSemanticsContext().targetCharacteristics(),
      ci.getParsing().allCooked(), ci.getInvocation().getTargetOpts().triple,
      kindMap, ci.getInvocation().getLoweringOpts(),
      ci.getInvocation().getFrontendOpts().envDefaults,
      ci.getInvocation().getFrontendOpts().features, targetMachine,
      ci.getInvocation().getTargetOpts().cpuToTuneFor);

  // Fetch module from lb, so we can set
  mlirModule = std::make_unique<mlir::ModuleOp>(lb.getModule());

  if (ci.getInvocation().getFrontendOpts().features.IsEnabled(
          Fortran::common::LanguageFeature::OpenMP)) {
    setOffloadModuleInterfaceAttributes(*mlirModule,
                                        ci.getInvocation().getLangOpts());
    setOpenMPVersionAttribute(*mlirModule,
                              ci.getInvocation().getLangOpts().OpenMPVersion);
  }

  // Create a parse tree and lower it to FIR
  Fortran::parser::Program &parseTree{*ci.getParsing().parseTree()};
  lb.lower(parseTree, ci.getSemanticsContext());

  // Add target specific items like dependent libraries, target specific
  // constants etc.
  addDependentLibs(*mlirModule, ci);
  addAMDGPUSpecificMLIRItems(*mlirModule, ci);

  // run the default passes.
  mlir::PassManager pm((*mlirModule)->getName(),
                       mlir::OpPassManager::Nesting::Implicit);
  (void)mlir::applyPassManagerCLOptions(pm);
  // Add OpenMP-related passes
  // WARNING: These passes must be run immediately after the lowering to ensure
  // that the FIR is correct with respect to OpenMP operations/attributes.
  if (ci.getInvocation().getFrontendOpts().features.IsEnabled(
          Fortran::common::LanguageFeature::OpenMP)) {
    bool isDevice = false;
    if (auto offloadMod = llvm::dyn_cast<mlir::omp::OffloadModuleInterface>(
            mlirModule->getOperation()))
      isDevice = offloadMod.getIsTargetDevice();
    // WARNING: This pipeline must be run immediately after the lowering to
    // ensure that the FIR is correct with respect to OpenMP operations/
    // attributes.
    fir::createOpenMPFIRPassPipeline(pm, isDevice);
  }

  pm.enableVerifier(/*verifyPasses=*/true);
  pm.addPass(std::make_unique<Fortran::lower::VerifierPass>());

  if (mlir::failed(pm.run(*mlirModule))) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "verification of lowering to FIR failed");
    ci.getDiagnostics().Report(diagID);
    return false;
  }

  // Print initial full MLIR module, before lowering or transformations, if
  // -save-temps has been specified.
  if (!saveMLIRTempFile(ci.getInvocation(), *mlirModule, getCurrentFile(),
                        "fir")) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Saving MLIR temp file failed");
    ci.getDiagnostics().Report(diagID);
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Custom ExecuteAction
//===----------------------------------------------------------------------===//
void InputOutputTestAction::executeAction() {
  CompilerInstance &ci = getInstance();

  // Create a stream for errors
  std::string buf;
  llvm::raw_string_ostream errorStream{buf};

  // Read the input file
  Fortran::parser::AllSources &allSources{ci.getAllSources()};
  std::string path{getCurrentFileOrBufferName()};
  const Fortran::parser::SourceFile *sf;
  if (path == "-")
    sf = allSources.ReadStandardInput(errorStream);
  else
    sf = allSources.Open(path, errorStream, std::optional<std::string>{"."s});
  llvm::ArrayRef<char> fileContent = sf->content();

  // Output file descriptor to receive the contents of the input file.
  std::unique_ptr<llvm::raw_ostream> os;

  // Copy the contents from the input file to the output file
  if (!ci.isOutputStreamNull()) {
    // An output stream (outputStream_) was set earlier
    ci.writeOutputStream(fileContent.data());
  } else {
    // No pre-set output stream - create an output file
    os = ci.createDefaultOutputFile(
        /*binary=*/true, getCurrentFileOrBufferName(), "txt");
    if (!os)
      return;
    (*os) << fileContent.data();
  }
}

void PrintPreprocessedAction::executeAction() {
  std::string buf;
  llvm::raw_string_ostream outForPP{buf};

  // Format or dump the prescanner's output
  CompilerInstance &ci = this->getInstance();
  if (ci.getInvocation().getPreprocessorOpts().showMacros) {
    ci.getParsing().EmitPreprocessorMacros(outForPP);
  } else if (ci.getInvocation().getPreprocessorOpts().noReformat) {
    ci.getParsing().DumpCookedChars(outForPP);
  } else {
    ci.getParsing().EmitPreprocessedSource(
        outForPP, !ci.getInvocation().getPreprocessorOpts().noLineDirectives);
  }

  // Print getDiagnostics from the prescanner
  ci.getParsing().messages().Emit(llvm::errs(), ci.getAllCookedSources());

  // If a pre-defined output stream exists, dump the preprocessed content there
  if (!ci.isOutputStreamNull()) {
    // Send the output to the pre-defined output buffer.
    ci.writeOutputStream(buf);
    return;
  }

  // Create a file and save the preprocessed output there
  std::unique_ptr<llvm::raw_pwrite_stream> os{ci.createDefaultOutputFile(
      /*Binary=*/true, /*InFile=*/getCurrentFileOrBufferName())};
  if (!os) {
    return;
  }

  (*os) << buf;
}

void DebugDumpProvenanceAction::executeAction() {
  this->getInstance().getParsing().DumpProvenance(llvm::outs());
}

void ParseSyntaxOnlyAction::executeAction() {}

void DebugUnparseNoSemaAction::executeAction() {
  auto &invoc = this->getInstance().getInvocation();
  auto &parseTree{getInstance().getParsing().parseTree()};

  // TODO: Options should come from CompilerInvocation
  Unparse(llvm::outs(), *parseTree,
          /*encoding=*/Fortran::parser::Encoding::UTF_8,
          /*capitalizeKeywords=*/true, /*backslashEscapes=*/false,
          /*preStatement=*/nullptr,
          invoc.getUseAnalyzedObjectsForUnparse() ? &invoc.getAsFortran()
                                                  : nullptr);
}

void DebugUnparseAction::executeAction() {
  auto &invoc = this->getInstance().getInvocation();
  auto &parseTree{getInstance().getParsing().parseTree()};

  CompilerInstance &ci = this->getInstance();
  auto os{ci.createDefaultOutputFile(
      /*Binary=*/false, /*InFile=*/getCurrentFileOrBufferName())};

  // TODO: Options should come from CompilerInvocation
  Unparse(*os, *parseTree,
          /*encoding=*/Fortran::parser::Encoding::UTF_8,
          /*capitalizeKeywords=*/true, /*backslashEscapes=*/false,
          /*preStatement=*/nullptr,
          invoc.getUseAnalyzedObjectsForUnparse() ? &invoc.getAsFortran()
                                                  : nullptr);

  // Report fatal semantic errors
  reportFatalSemanticErrors();
}

void DebugUnparseWithSymbolsAction::executeAction() {
  auto &parseTree{*getInstance().getParsing().parseTree()};

  Fortran::semantics::UnparseWithSymbols(
      llvm::outs(), parseTree, /*encoding=*/Fortran::parser::Encoding::UTF_8);

  // Report fatal semantic errors
  reportFatalSemanticErrors();
}

void DebugUnparseWithModulesAction::executeAction() {
  auto &parseTree{*getInstance().getParsing().parseTree()};
  CompilerInstance &ci{getInstance()};
  Fortran::semantics::UnparseWithModules(
      llvm::outs(), ci.getSemantics().context(), parseTree,
      /*encoding=*/Fortran::parser::Encoding::UTF_8);
  reportFatalSemanticErrors();
}

void DebugDumpSymbolsAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  if (!ci.getRtTyTables().schemata) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "could not find module file for __fortran_type_info");
    ci.getDiagnostics().Report(diagID);
    llvm::errs() << "\n";
    return;
  }

  // Dump symbols
  ci.getSemantics().DumpSymbols(llvm::outs());
}

void DebugDumpAllAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  // Dump parse tree
  auto &parseTree{getInstance().getParsing().parseTree()};
  llvm::outs() << "========================";
  llvm::outs() << " Flang: parse tree dump ";
  llvm::outs() << "========================\n";
  Fortran::parser::DumpTree(llvm::outs(), parseTree,
                            &ci.getInvocation().getAsFortran());

  if (!ci.getRtTyTables().schemata) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "could not find module file for __fortran_type_info");
    ci.getDiagnostics().Report(diagID);
    llvm::errs() << "\n";
    return;
  }

  // Dump symbols
  llvm::outs() << "=====================";
  llvm::outs() << " Flang: symbols dump ";
  llvm::outs() << "=====================\n";
  ci.getSemantics().DumpSymbols(llvm::outs());
}

void DebugDumpParseTreeNoSemaAction::executeAction() {
  auto &parseTree{getInstance().getParsing().parseTree()};

  // Dump parse tree
  Fortran::parser::DumpTree(
      llvm::outs(), parseTree,
      &this->getInstance().getInvocation().getAsFortran());
}

void DebugDumpParseTreeAction::executeAction() {
  auto &parseTree{getInstance().getParsing().parseTree()};

  // Dump parse tree
  Fortran::parser::DumpTree(
      llvm::outs(), parseTree,
      &this->getInstance().getInvocation().getAsFortran());

  // Report fatal semantic errors
  reportFatalSemanticErrors();
}

void DebugMeasureParseTreeAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  // Parse. In case of failure, report and return.
  ci.getParsing().Parse(llvm::outs());

  if (!ci.getParsing().messages().empty() &&
      (ci.getInvocation().getWarnAsErr() ||
       ci.getParsing().messages().AnyFatalError())) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Could not parse %0");
    ci.getDiagnostics().Report(diagID) << getCurrentFileOrBufferName();

    ci.getParsing().messages().Emit(llvm::errs(),
                                    this->getInstance().getAllCookedSources());
    return;
  }

  // Report the getDiagnostics from parsing
  ci.getParsing().messages().Emit(llvm::errs(), ci.getAllCookedSources());

  auto &parseTree{*ci.getParsing().parseTree()};

  // Measure the parse tree
  MeasurementVisitor visitor;
  Fortran::parser::Walk(parseTree, visitor);
  llvm::outs() << "Parse tree comprises " << visitor.objects
               << " objects and occupies " << visitor.bytes
               << " total bytes.\n";
}

void DebugPreFIRTreeAction::executeAction() {
  CompilerInstance &ci = this->getInstance();
  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors()) {
    return;
  }

  auto &parseTree{*ci.getParsing().parseTree()};

  // Dump pre-FIR tree
  if (auto ast{
          Fortran::lower::createPFT(parseTree, ci.getSemanticsContext())}) {
    Fortran::lower::dumpPFT(llvm::outs(), *ast);
  } else {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Pre FIR Tree is NULL.");
    ci.getDiagnostics().Report(diagID);
  }
}

void DebugDumpParsingLogAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  ci.getParsing().Parse(llvm::errs());
  ci.getParsing().DumpParsingLog(llvm::outs());
}

void GetDefinitionAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors()) {
    return;
  }

  parser::AllCookedSources &cs = ci.getAllCookedSources();
  unsigned diagID = ci.getDiagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Error, "Symbol not found");

  auto gdv = ci.getInvocation().getFrontendOpts().getDefVals;
  auto charBlock{cs.GetCharBlockFromLineAndColumns(gdv.line, gdv.startColumn,
                                                   gdv.endColumn)};
  if (!charBlock) {
    ci.getDiagnostics().Report(diagID);
    return;
  }

  llvm::outs() << "String range: >" << charBlock->ToString() << "<\n";

  auto *symbol{
      ci.getSemanticsContext().FindScope(*charBlock).FindSymbol(*charBlock)};
  if (!symbol) {
    ci.getDiagnostics().Report(diagID);
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
  llvm::outs() << symbol->name().ToString() << ": " << sourceInfo->first.path
               << ", " << sourceInfo->first.line << ", "
               << sourceInfo->first.column << "-" << sourceInfo->second.column
               << "\n";
}

void GetSymbolsSourcesAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors()) {
    return;
  }

  ci.getSemantics().DumpSymbolsSources(llvm::outs());
}

//===----------------------------------------------------------------------===//
// CodeGenActions
//===----------------------------------------------------------------------===//

CodeGenAction::~CodeGenAction() = default;

static llvm::OptimizationLevel
mapToLevel(const Fortran::frontend::CodeGenOptions &opts) {
  switch (opts.OptimizationLevel) {
  default:
    llvm_unreachable("Invalid optimization level!");
  case 0:
    return llvm::OptimizationLevel::O0;
  case 1:
    return llvm::OptimizationLevel::O1;
  case 2:
    return llvm::OptimizationLevel::O2;
  case 3:
    return llvm::OptimizationLevel::O3;
  }
}

// Lower using HLFIR then run the FIR to HLFIR pipeline
void CodeGenAction::lowerHLFIRToFIR() {
  assert(mlirModule && "The MLIR module has not been generated yet.");

  CompilerInstance &ci = this->getInstance();
  auto opts = ci.getInvocation().getCodeGenOpts();
  llvm::OptimizationLevel level = mapToLevel(opts);

  fir::support::loadDialects(*mlirCtx);

  // Set-up the MLIR pass manager
  mlir::PassManager pm((*mlirModule)->getName(),
                       mlir::OpPassManager::Nesting::Implicit);

  pm.addPass(std::make_unique<Fortran::lower::VerifierPass>());
  pm.enableVerifier(/*verifyPasses=*/true);

  // Create the pass pipeline
  fir::createHLFIRToFIRPassPipeline(pm, level);
  (void)mlir::applyPassManagerCLOptions(pm);

  if (!mlir::succeeded(pm.run(*mlirModule))) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Lowering to FIR failed");
    ci.getDiagnostics().Report(diagID);
  }
}

static std::optional<std::pair<unsigned, unsigned>>
getAArch64VScaleRange(CompilerInstance &ci) {
  const auto &langOpts = ci.getInvocation().getLangOpts();

  if (langOpts.VScaleMin || langOpts.VScaleMax)
    return std::pair<unsigned, unsigned>(
        langOpts.VScaleMin ? langOpts.VScaleMin : 1, langOpts.VScaleMax);

  std::string featuresStr = ci.getTargetFeatures();
  if (featuresStr.find("+sve") != std::string::npos)
    return std::pair<unsigned, unsigned>(1, 16);

  return std::nullopt;
}

static std::optional<std::pair<unsigned, unsigned>>
getRISCVVScaleRange(CompilerInstance &ci) {
  const auto &langOpts = ci.getInvocation().getLangOpts();
  const auto targetOpts = ci.getInvocation().getTargetOpts();
  const llvm::Triple triple(targetOpts.triple);

  auto parseResult = llvm::RISCVISAInfo::parseFeatures(
      triple.isRISCV64() ? 64 : 32, targetOpts.featuresAsWritten);
  if (!parseResult) {
    std::string buffer;
    llvm::raw_string_ostream outputErrMsg(buffer);
    handleAllErrors(parseResult.takeError(), [&](llvm::StringError &errMsg) {
      outputErrMsg << errMsg.getMessage();
    });
    ci.getDiagnostics().Report(clang::diag::err_invalid_feature_combination)
        << buffer;
    return std::nullopt;
  }

  llvm::RISCVISAInfo *const isaInfo = parseResult->get();

  // RISCV::RVVBitsPerBlock is 64.
  unsigned vscaleMin = isaInfo->getMinVLen() / llvm::RISCV::RVVBitsPerBlock;

  if (langOpts.VScaleMin || langOpts.VScaleMax) {
    // Treat Zvl*b as a lower bound on vscale.
    vscaleMin = std::max(vscaleMin, langOpts.VScaleMin);
    unsigned vscaleMax = langOpts.VScaleMax;
    if (vscaleMax != 0 && vscaleMax < vscaleMin)
      vscaleMax = vscaleMin;
    return std::pair<unsigned, unsigned>(vscaleMin ? vscaleMin : 1, vscaleMax);
  }

  if (vscaleMin > 0) {
    unsigned vscaleMax = isaInfo->getMaxVLen() / llvm::RISCV::RVVBitsPerBlock;
    return std::make_pair(vscaleMin, vscaleMax);
  }

  return std::nullopt;
}

// TODO: We should get this from TargetInfo. However, that depends on
// too much of clang, so for now, replicate the functionality.
static std::optional<std::pair<unsigned, unsigned>>
getVScaleRange(CompilerInstance &ci) {
  const llvm::Triple triple(ci.getInvocation().getTargetOpts().triple);

  if (triple.isAArch64())
    return getAArch64VScaleRange(ci);
  if (triple.isRISCV())
    return getRISCVVScaleRange(ci);

  // All other architectures that don't support scalable vectors (i.e. don't
  // need vscale)
  return std::nullopt;
}

// Lower the previously generated MLIR module into an LLVM IR module
void CodeGenAction::generateLLVMIR() {
  assert(mlirModule && "The MLIR module has not been generated yet.");

  CompilerInstance &ci = this->getInstance();
  auto opts = ci.getInvocation().getCodeGenOpts();
  auto mathOpts = ci.getInvocation().getLoweringOpts().getMathOptions();
  llvm::OptimizationLevel level = mapToLevel(opts);

  fir::support::loadDialects(*mlirCtx);
  mlir::DialectRegistry registry;
  fir::support::registerNonCodegenDialects(registry);
  fir::support::addFIRExtensions(registry);
  mlirCtx->appendDialectRegistry(registry);
  fir::support::registerLLVMTranslation(*mlirCtx);

  // Set-up the MLIR pass manager
  mlir::PassManager pm((*mlirModule)->getName(),
                       mlir::OpPassManager::Nesting::Implicit);

  pm.addPass(std::make_unique<Fortran::lower::VerifierPass>());
  pm.enableVerifier(/*verifyPasses=*/true);

  MLIRToLLVMPassPipelineConfig config(level, opts, mathOpts);
  fir::registerDefaultInlinerPass(config);

  if (auto vsr = getVScaleRange(ci)) {
    config.VScaleMin = vsr->first;
    config.VScaleMax = vsr->second;
  }

  if (ci.getInvocation().getLoweringOpts().getNSWOnLoopVarInc())
    config.NSWOnLoopVarInc = true;

  // Create the pass pipeline
  fir::createMLIRToLLVMPassPipeline(pm, config, getCurrentFile());
  (void)mlir::applyPassManagerCLOptions(pm);

  // run the pass manager
  if (!mlir::succeeded(pm.run(*mlirModule))) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Lowering to LLVM IR failed");
    ci.getDiagnostics().Report(diagID);
  }

  // Print final MLIR module, just before translation into LLVM IR, if
  // -save-temps has been specified.
  if (!saveMLIRTempFile(ci.getInvocation(), *mlirModule, getCurrentFile(),
                        "llvmir")) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Saving MLIR temp file failed");
    ci.getDiagnostics().Report(diagID);
    return;
  }

  // Translate to LLVM IR
  std::optional<llvm::StringRef> moduleName = mlirModule->getName();
  llvmModule = mlir::translateModuleToLLVMIR(
      *mlirModule, *llvmCtx, moduleName ? *moduleName : "FIRModule");

  if (!llvmModule) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "failed to create the LLVM module");
    ci.getDiagnostics().Report(diagID);
    return;
  }

  // Set PIC/PIE level LLVM module flags.
  if (opts.PICLevel > 0) {
    llvmModule->setPICLevel(static_cast<llvm::PICLevel::Level>(opts.PICLevel));
    if (opts.IsPIE)
      llvmModule->setPIELevel(
          static_cast<llvm::PIELevel::Level>(opts.PICLevel));
  }

  // Set mcmodel level LLVM module flags
  std::optional<llvm::CodeModel::Model> cm = getCodeModel(opts.CodeModel);
  if (cm.has_value()) {
    const llvm::Triple triple(ci.getInvocation().getTargetOpts().triple);
    llvmModule->setCodeModel(*cm);
    if ((cm == llvm::CodeModel::Medium || cm == llvm::CodeModel::Large) &&
        triple.getArch() == llvm::Triple::x86_64) {
      llvmModule->setLargeDataThreshold(opts.LargeDataThreshold);
    }
  }
}

static std::unique_ptr<llvm::raw_pwrite_stream>
getOutputStream(CompilerInstance &ci, llvm::StringRef inFile,
                BackendActionTy action) {
  switch (action) {
  case BackendActionTy::Backend_EmitAssembly:
    return ci.createDefaultOutputFile(
        /*Binary=*/false, inFile, /*extension=*/"s");
  case BackendActionTy::Backend_EmitLL:
    return ci.createDefaultOutputFile(
        /*Binary=*/false, inFile, /*extension=*/"ll");
  case BackendActionTy::Backend_EmitFIR:
  case BackendActionTy::Backend_EmitHLFIR:
    return ci.createDefaultOutputFile(
        /*Binary=*/false, inFile, /*extension=*/"mlir");
  case BackendActionTy::Backend_EmitBC:
    return ci.createDefaultOutputFile(
        /*Binary=*/true, inFile, /*extension=*/"bc");
  case BackendActionTy::Backend_EmitObj:
    return ci.createDefaultOutputFile(
        /*Binary=*/true, inFile, /*extension=*/"o");
  }

  llvm_unreachable("Invalid action!");
}

/// Generate target-specific machine-code or assembly file from the input LLVM
/// module.
///
/// \param [in] diags Diagnostics engine for reporting errors
/// \param [in] tm Target machine to aid the code-gen pipeline set-up
/// \param [in] act Backend act to run (assembly vs machine-code generation)
/// \param [in] llvmModule LLVM module to lower to assembly/machine-code
/// \param [in] codeGenOpts options configuring codegen pipeline
/// \param [out] os Output stream to emit the generated code to
static void generateMachineCodeOrAssemblyImpl(clang::DiagnosticsEngine &diags,
                                              llvm::TargetMachine &tm,
                                              BackendActionTy act,
                                              llvm::Module &llvmModule,
                                              const CodeGenOptions &codeGenOpts,
                                              llvm::raw_pwrite_stream &os) {
  assert(((act == BackendActionTy::Backend_EmitObj) ||
          (act == BackendActionTy::Backend_EmitAssembly)) &&
         "Unsupported action");

  // Set-up the pass manager, i.e create an LLVM code-gen pass pipeline.
  // Currently only the legacy pass manager is supported.
  // TODO: Switch to the new PM once it's available in the backend.
  llvm::legacy::PassManager codeGenPasses;
  codeGenPasses.add(
      createTargetTransformInfoWrapperPass(tm.getTargetIRAnalysis()));

  llvm::Triple triple(llvmModule.getTargetTriple());
  llvm::TargetLibraryInfoImpl *tlii =
      llvm::driver::createTLII(triple, codeGenOpts.getVecLib());
  codeGenPasses.add(new llvm::TargetLibraryInfoWrapperPass(*tlii));

  llvm::CodeGenFileType cgft = (act == BackendActionTy::Backend_EmitAssembly)
                                   ? llvm::CodeGenFileType::AssemblyFile
                                   : llvm::CodeGenFileType::ObjectFile;
  if (tm.addPassesToEmitFile(codeGenPasses, os, nullptr, cgft)) {
    unsigned diagID =
        diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
                              "emission of this file type is not supported");
    diags.Report(diagID);
    return;
  }

  // Run the passes
  codeGenPasses.run(llvmModule);
}

void CodeGenAction::runOptimizationPipeline(llvm::raw_pwrite_stream &os) {
  auto opts = getInstance().getInvocation().getCodeGenOpts();
  auto &diags = getInstance().getDiagnostics();
  llvm::OptimizationLevel level = mapToLevel(opts);

  llvm::TargetMachine *targetMachine = &getInstance().getTargetMachine();
  // Create the analysis managers.
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  // Create the pass manager builder.
  llvm::PassInstrumentationCallbacks pic;
  llvm::PipelineTuningOptions pto;
  std::optional<llvm::PGOOptions> pgoOpt;
  llvm::StandardInstrumentations si(llvmModule->getContext(),
                                    opts.DebugPassManager);
  si.registerCallbacks(pic, &mam);
  llvm::PassBuilder pb(targetMachine, pto, pgoOpt, &pic);

  // Attempt to load pass plugins and register their callbacks with PB.
  for (auto &pluginFile : opts.LLVMPassPlugins) {
    auto passPlugin = llvm::PassPlugin::Load(pluginFile);
    if (passPlugin) {
      passPlugin->registerPassBuilderCallbacks(pb);
    } else {
      diags.Report(clang::diag::err_fe_unable_to_load_plugin)
          << pluginFile << passPlugin.takeError();
    }
  }
  // Register static plugin extensions.
#define HANDLE_EXTENSION(Ext)                                                  \
  get##Ext##PluginInfo().RegisterPassBuilderCallbacks(pb);
#include "llvm/Support/Extension.def"

  // Register the target library analysis directly and give it a customized
  // preset TLI depending on -fveclib
  llvm::Triple triple(llvmModule->getTargetTriple());
  llvm::TargetLibraryInfoImpl *tlii =
      llvm::driver::createTLII(triple, opts.getVecLib());
  fam.registerPass([&] { return llvm::TargetLibraryAnalysis(*tlii); });

  // Register all the basic analyses with the managers.
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  // Create the pass manager.
  llvm::ModulePassManager mpm;
  if (opts.PrepareForFullLTO)
    mpm = pb.buildLTOPreLinkDefaultPipeline(level);
  else if (opts.PrepareForThinLTO)
    mpm = pb.buildThinLTOPreLinkDefaultPipeline(level);
  else
    mpm = pb.buildPerModuleDefaultPipeline(level);

  if (action == BackendActionTy::Backend_EmitBC)
    mpm.addPass(llvm::BitcodeWriterPass(os));
  else if (action == BackendActionTy::Backend_EmitLL)
    mpm.addPass(llvm::PrintModulePass(os));

  // FIXME: This should eventually be replaced by a first-class driver option.
  // This should be done for both flang and clang simultaneously.
  // Print a textual, '-passes=' compatible, representation of pipeline if
  // requested. In this case, don't run the passes. This mimics the behavior of
  // clang.
  if (llvm::PrintPipelinePasses) {
    mpm.printPipeline(llvm::outs(), [&pic](llvm::StringRef className) {
      auto passName = pic.getPassNameForClassName(className);
      return passName.empty() ? className : passName;
    });
    llvm::outs() << "\n";
    return;
  }

  // Run the passes.
  mpm.run(*llvmModule, mam);
}

// This class handles optimization remark messages requested if
// any of -Rpass, -Rpass-analysis or -Rpass-missed flags were provided
class BackendRemarkConsumer : public llvm::DiagnosticHandler {

  const CodeGenOptions &codeGenOpts;
  clang::DiagnosticsEngine &diags;

public:
  BackendRemarkConsumer(clang::DiagnosticsEngine &diags,
                        const CodeGenOptions &codeGenOpts)
      : codeGenOpts(codeGenOpts), diags(diags) {}

  bool isAnalysisRemarkEnabled(llvm::StringRef passName) const override {
    return codeGenOpts.OptimizationRemarkAnalysis.patternMatches(passName);
  }
  bool isMissedOptRemarkEnabled(llvm::StringRef passName) const override {
    return codeGenOpts.OptimizationRemarkMissed.patternMatches(passName);
  }
  bool isPassedOptRemarkEnabled(llvm::StringRef passName) const override {
    return codeGenOpts.OptimizationRemark.patternMatches(passName);
  }

  bool isAnyRemarkEnabled() const override {
    return codeGenOpts.OptimizationRemarkAnalysis.hasValidPattern() ||
           codeGenOpts.OptimizationRemarkMissed.hasValidPattern() ||
           codeGenOpts.OptimizationRemark.hasValidPattern();
  }

  void
  emitOptimizationMessage(const llvm::DiagnosticInfoOptimizationBase &diagInfo,
                          unsigned diagID) {
    // We only support warnings and remarks.
    assert(diagInfo.getSeverity() == llvm::DS_Remark ||
           diagInfo.getSeverity() == llvm::DS_Warning);

    std::string msg;
    llvm::raw_string_ostream msgStream(msg);

    if (diagInfo.isLocationAvailable()) {
      // Clang contains a SourceManager class which handles loading
      // and caching of source files into memory and it can be used to
      // query SourceLocation data. The SourceLocation data is what is
      // needed here as it contains the full include stack which gives
      // line and column number as well as file name and location.
      // Since Flang doesn't have SourceManager, send file name and absolute
      // path through msgStream, to use for printing.
      msgStream << diagInfo.getLocationStr() << ";;"
                << diagInfo.getAbsolutePath() << ";;";
    }

    msgStream << diagInfo.getMsg();

    // Emit message.
    diags.Report(diagID) << clang::AddFlagValue(diagInfo.getPassName()) << msg;
  }

  void optimizationRemarkHandler(
      const llvm::DiagnosticInfoOptimizationBase &diagInfo) {
    auto passName = diagInfo.getPassName();
    if (diagInfo.isPassed()) {
      if (codeGenOpts.OptimizationRemark.patternMatches(passName))
        // Optimization remarks are active only if the -Rpass flag has a regular
        // expression that matches the name of the pass name in \p d.
        emitOptimizationMessage(
            diagInfo, clang::diag::remark_fe_backend_optimization_remark);

      return;
    }

    if (diagInfo.isMissed()) {
      if (codeGenOpts.OptimizationRemarkMissed.patternMatches(passName))
        // Missed optimization remarks are active only if the -Rpass-missed
        // flag has a regular expression that matches the name of the pass
        // name in \p d.
        emitOptimizationMessage(
            diagInfo,
            clang::diag::remark_fe_backend_optimization_remark_missed);

      return;
    }

    assert(diagInfo.isAnalysis() && "Unknown remark type");

    bool shouldAlwaysPrint = false;
    auto *ora = llvm::dyn_cast<llvm::OptimizationRemarkAnalysis>(&diagInfo);
    if (ora)
      shouldAlwaysPrint = ora->shouldAlwaysPrint();

    if (shouldAlwaysPrint ||
        codeGenOpts.OptimizationRemarkAnalysis.patternMatches(passName))
      emitOptimizationMessage(
          diagInfo,
          clang::diag::remark_fe_backend_optimization_remark_analysis);
  }

  bool handleDiagnostics(const llvm::DiagnosticInfo &di) override {
    switch (di.getKind()) {
    case llvm::DK_OptimizationRemark:
      optimizationRemarkHandler(llvm::cast<llvm::OptimizationRemark>(di));
      break;
    case llvm::DK_OptimizationRemarkMissed:
      optimizationRemarkHandler(llvm::cast<llvm::OptimizationRemarkMissed>(di));
      break;
    case llvm::DK_OptimizationRemarkAnalysis:
      optimizationRemarkHandler(
          llvm::cast<llvm::OptimizationRemarkAnalysis>(di));
      break;
    case llvm::DK_MachineOptimizationRemark:
      optimizationRemarkHandler(
          llvm::cast<llvm::MachineOptimizationRemark>(di));
      break;
    case llvm::DK_MachineOptimizationRemarkMissed:
      optimizationRemarkHandler(
          llvm::cast<llvm::MachineOptimizationRemarkMissed>(di));
      break;
    case llvm::DK_MachineOptimizationRemarkAnalysis:
      optimizationRemarkHandler(
          llvm::cast<llvm::MachineOptimizationRemarkAnalysis>(di));
      break;
    default:
      break;
    }
    return true;
  }
};

void CodeGenAction::embedOffloadObjects() {
  CompilerInstance &ci = this->getInstance();
  const auto &cgOpts = ci.getInvocation().getCodeGenOpts();

  for (llvm::StringRef offloadObject : cgOpts.OffloadObjects) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(offloadObject);
    if (std::error_code ec = objectOrErr.getError()) {
      auto diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "could not open '%0' for embedding");
      ci.getDiagnostics().Report(diagID) << offloadObject;
      return;
    }
    llvm::embedBufferInModule(
        *llvmModule, **objectOrErr, ".llvm.offloading",
        llvm::Align(llvm::object::OffloadBinary::getAlignment()));
  }
}

void CodeGenAction::linkBuiltinBCLibs() {
  auto options = clang::FileSystemOptions();
  clang::FileManager fileManager(options);
  CompilerInstance &ci = this->getInstance();
  const auto &cgOpts = ci.getInvocation().getCodeGenOpts();

  std::vector<std::unique_ptr<llvm::Module>> modules;

  // Load LLVM modules
  for (llvm::StringRef bcLib : cgOpts.BuiltinBCLibs) {
    auto BCBuf = fileManager.getBufferForFile(bcLib);
    if (!BCBuf) {
      auto diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "could not open '%0' for linking");
      ci.getDiagnostics().Report(diagID) << bcLib;
      return;
    }

    llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOrErr =
        getOwningLazyBitcodeModule(std::move(*BCBuf), *llvmCtx);
    if (!ModuleOrErr) {
      auto diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "error loading '%0' for linking");
      ci.getDiagnostics().Report(diagID) << bcLib;
      return;
    }
    modules.push_back(std::move(ModuleOrErr.get()));
  }

  // Link modules and internalize functions
  for (auto &module : modules) {
    bool Err;
    Err = llvm::Linker::linkModules(
        *llvmModule, std::move(module), llvm::Linker::Flags::LinkOnlyNeeded,
        [](llvm::Module &M, const llvm::StringSet<> &GVS) {
          llvm::internalizeModule(M, [&GVS](const llvm::GlobalValue &GV) {
            return !GV.hasName() || (GVS.count(GV.getName()) == 0);
          });
        });
    if (Err) {
      auto diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "link error when linking '%0'");
      ci.getDiagnostics().Report(diagID) << module->getSourceFileName();
      return;
    }
  }
}

static void reportOptRecordError(llvm::Error e, clang::DiagnosticsEngine &diags,
                                 const CodeGenOptions &codeGenOpts) {
  handleAllErrors(
      std::move(e),
      [&](const llvm::LLVMRemarkSetupFileError &e) {
        diags.Report(clang::diag::err_cannot_open_file)
            << codeGenOpts.OptRecordFile << e.message();
      },
      [&](const llvm::LLVMRemarkSetupPatternError &e) {
        diags.Report(clang::diag::err_drv_optimization_remark_pattern)
            << e.message() << codeGenOpts.OptRecordPasses;
      },
      [&](const llvm::LLVMRemarkSetupFormatError &e) {
        diags.Report(clang::diag::err_drv_optimization_remark_format)
            << codeGenOpts.OptRecordFormat;
      });
}

void CodeGenAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  clang::DiagnosticsEngine &diags = ci.getDiagnostics();
  const CodeGenOptions &codeGenOpts = ci.getInvocation().getCodeGenOpts();
  Fortran::lower::LoweringOptions &loweringOpts =
      ci.getInvocation().getLoweringOpts();

  // If the output stream is a file, generate it and define the corresponding
  // output stream. If a pre-defined output stream is available, we will use
  // that instead.
  //
  // NOTE: `os` is a smart pointer that will be destroyed at the end of this
  // method. However, it won't be written to until `codeGenPasses` is
  // destroyed. By defining `os` before `codeGenPasses`, we make sure that the
  // output stream won't be destroyed before it is written to. This only
  // applies when an output file is used (i.e. there is no pre-defined output
  // stream).
  // TODO: Revisit once the new PM is ready (i.e. when `codeGenPasses` is
  // updated to use it).
  std::unique_ptr<llvm::raw_pwrite_stream> os;
  if (ci.isOutputStreamNull()) {
    os = getOutputStream(ci, getCurrentFileOrBufferName(), action);

    if (!os) {
      unsigned diagID = diags.getCustomDiagID(
          clang::DiagnosticsEngine::Error, "failed to create the output file");
      diags.Report(diagID);
      return;
    }
  }

  if (action == BackendActionTy::Backend_EmitFIR) {
    if (loweringOpts.getLowerToHighLevelFIR()) {
      lowerHLFIRToFIR();
    }
    mlirModule->print(ci.isOutputStreamNull() ? *os : ci.getOutputStream());
    return;
  }

  if (action == BackendActionTy::Backend_EmitHLFIR) {
    assert(loweringOpts.getLowerToHighLevelFIR() &&
           "Lowering must have been configured to emit HLFIR");
    mlirModule->print(ci.isOutputStreamNull() ? *os : ci.getOutputStream());
    return;
  }

  // Generate an LLVM module if it's not already present (it will already be
  // present if the input file is an LLVM IR/BC file).
  if (!llvmModule)
    generateLLVMIR();

  // If generating the LLVM module failed, abort! No need for further error
  // reporting since generateLLVMIR() does this already.
  if (!llvmModule)
    return;

  // Set the triple based on the targetmachine (this comes compiler invocation
  // and the command-line target option if specified, or the default if not
  // given on the command-line).
  llvm::TargetMachine &targetMachine = ci.getTargetMachine();

  const std::string &theTriple = targetMachine.getTargetTriple().str();

  if (llvmModule->getTargetTriple() != theTriple) {
    diags.Report(clang::diag::warn_fe_override_module) << theTriple;
  }

  // Always set the triple and data layout, to make sure they match and are set.
  // Note that this overwrites any datalayout stored in the LLVM-IR. This avoids
  // an assert for incompatible data layout when the code-generation happens.
  llvmModule->setTargetTriple(theTriple);
  llvmModule->setDataLayout(targetMachine.createDataLayout());

  // Link in builtin bitcode libraries
  if (!codeGenOpts.BuiltinBCLibs.empty())
    linkBuiltinBCLibs();

  // Embed offload objects specified with -fembed-offload-object
  if (!codeGenOpts.OffloadObjects.empty())
    embedOffloadObjects();

  BackendRemarkConsumer remarkConsumer(diags, codeGenOpts);

  llvmModule->getContext().setDiagnosticHandler(
      std::make_unique<BackendRemarkConsumer>(remarkConsumer));

  // write optimization-record
  llvm::Expected<std::unique_ptr<llvm::ToolOutputFile>> optRecordFileOrErr =
      setupLLVMOptimizationRemarks(
          llvmModule->getContext(), codeGenOpts.OptRecordFile,
          codeGenOpts.OptRecordPasses, codeGenOpts.OptRecordFormat,
          /*DiagnosticsWithHotness=*/false,
          /*DiagnosticsHotnessThreshold=*/0);

  if (llvm::Error e = optRecordFileOrErr.takeError()) {
    reportOptRecordError(std::move(e), diags, codeGenOpts);
    return;
  }

  std::unique_ptr<llvm::ToolOutputFile> optRecordFile =
      std::move(*optRecordFileOrErr);

  if (optRecordFile) {
    optRecordFile->keep();
    optRecordFile->os().flush();
  }

  // Run LLVM's middle-end (i.e. the optimizer).
  runOptimizationPipeline(ci.isOutputStreamNull() ? *os : ci.getOutputStream());

  if (action == BackendActionTy::Backend_EmitLL ||
      action == BackendActionTy::Backend_EmitBC) {
    // This action has effectively been completed in runOptimizationPipeline.
    return;
  }

  // Run LLVM's backend and generate either assembly or machine code
  if (action == BackendActionTy::Backend_EmitAssembly ||
      action == BackendActionTy::Backend_EmitObj) {
    generateMachineCodeOrAssemblyImpl(
        diags, targetMachine, action, *llvmModule, codeGenOpts,
        ci.isOutputStreamNull() ? *os : ci.getOutputStream());
    return;
  }
}

void InitOnlyAction::executeAction() {
  CompilerInstance &ci = this->getInstance();
  unsigned diagID = ci.getDiagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Warning,
      "Use `-init-only` for testing purposes only");
  ci.getDiagnostics().Report(diagID);
}

void PluginParseTreeAction::executeAction() {}

void DebugDumpPFTAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  if (auto ast = Fortran::lower::createPFT(*ci.getParsing().parseTree(),
                                           ci.getSemantics().context())) {
    Fortran::lower::dumpPFT(llvm::outs(), *ast);
    return;
  }

  unsigned diagID = ci.getDiagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Error, "Pre FIR Tree is NULL.");
  ci.getDiagnostics().Report(diagID);
}

Fortran::parser::Parsing &PluginParseTreeAction::getParsing() {
  return getInstance().getParsing();
}

std::unique_ptr<llvm::raw_pwrite_stream>
PluginParseTreeAction::createOutputFile(llvm::StringRef extension = "") {

  std::unique_ptr<llvm::raw_pwrite_stream> os{
      getInstance().createDefaultOutputFile(
          /*Binary=*/false, /*InFile=*/getCurrentFileOrBufferName(),
          extension)};
  return os;
}
