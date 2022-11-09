//===- bbc.cpp - Burnside Bridge Compiler -----------------------*- C++ -*-===//
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
///
/// This is a tool for translating Fortran sources to the FIR dialect of MLIR.
///
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran-features.h"
#include "flang/Common/default-kinds.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Support/Verifier.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/unparse.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/runtime-type-info.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/unparse-with-symbols.h"
#include "flang/Version.inc"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

//===----------------------------------------------------------------------===//
// Some basic command-line options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::Required,
                                                llvm::cl::desc("<input file>"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Specify the output filename"),
                   llvm::cl::value_desc("filename"));

static llvm::cl::list<std::string>
    includeDirs("I", llvm::cl::desc("include module search paths"));

static llvm::cl::alias includeAlias("module-directory",
                                    llvm::cl::desc("module search directory"),
                                    llvm::cl::aliasopt(includeDirs));

static llvm::cl::list<std::string>
    intrinsicIncludeDirs("J", llvm::cl::desc("intrinsic module search paths"));

static llvm::cl::alias
    intrinsicIncludeAlias("intrinsic-module-directory",
                          llvm::cl::desc("intrinsic module directory"),
                          llvm::cl::aliasopt(intrinsicIncludeDirs));

static llvm::cl::opt<std::string>
    moduleDir("module", llvm::cl::desc("module output directory (default .)"),
              llvm::cl::init("."));

static llvm::cl::opt<std::string>
    moduleSuffix("module-suffix", llvm::cl::desc("module file suffix override"),
                 llvm::cl::init(".mod"));

static llvm::cl::opt<bool>
    emitFIR("emit-fir",
            llvm::cl::desc("Dump the FIR created by lowering and exit"),
            llvm::cl::init(false));

static llvm::cl::opt<bool> warnStdViolation("Mstandard",
                                            llvm::cl::desc("emit warnings"),
                                            llvm::cl::init(false));

static llvm::cl::opt<bool> warnIsError("Werror",
                                       llvm::cl::desc("warnings are errors"),
                                       llvm::cl::init(false));

static llvm::cl::opt<bool> dumpSymbols("dump-symbols",
                                       llvm::cl::desc("dump the symbol table"),
                                       llvm::cl::init(false));

static llvm::cl::opt<bool> pftDumpTest(
    "pft-test",
    llvm::cl::desc("parse the input, create a PFT, dump it, and exit"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> enableOpenMP("fopenmp",
                                        llvm::cl::desc("enable openmp"),
                                        llvm::cl::init(false));

static llvm::cl::opt<bool> enableOpenACC("fopenacc",
                                         llvm::cl::desc("enable openacc"),
                                         llvm::cl::init(false));

static llvm::cl::opt<bool> enablePolymorphic(
    "polymorphic-type",
    llvm::cl::desc("enable polymorphic type lowering (experimental)"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> useHLFIR("hlfir",
                                    llvm::cl::desc("Lower to high level FIR"),
                                    llvm::cl::init(false));

#define FLANG_EXCLUDE_CODEGEN
#include "flang/Tools/CLOptions.inc"

//===----------------------------------------------------------------------===//

using ProgramName = std::string;

// Print the module without the "module { ... }" wrapper.
static void printModule(mlir::ModuleOp mlirModule, llvm::raw_ostream &out) {
  for (auto &op : *mlirModule.getBody())
    out << op << '\n';
  out << '\n';
}

static void registerAllPasses() {
  fir::support::registerMLIRPassesForFortranTools();
  fir::registerOptTransformPasses();
}

//===----------------------------------------------------------------------===//
// Translate Fortran input to FIR, a dialect of MLIR.
//===----------------------------------------------------------------------===//

static mlir::LogicalResult convertFortranSourceToMLIR(
    std::string path, Fortran::parser::Options options,
    const ProgramName &programPrefix,
    Fortran::semantics::SemanticsContext &semanticsContext,
    const mlir::PassPipelineCLParser &passPipeline) {

  // prep for prescan and parse
  Fortran::parser::Parsing parsing{semanticsContext.allCookedSources()};
  parsing.Prescan(path, options);
  if (!parsing.messages().empty() && (parsing.messages().AnyFatalError())) {
    llvm::errs() << programPrefix << "could not scan " << path << '\n';
    parsing.messages().Emit(llvm::errs(), parsing.allCooked());
    return mlir::failure();
  }

  // parse the input Fortran
  parsing.Parse(llvm::outs());
  parsing.messages().Emit(llvm::errs(), parsing.allCooked());
  if (!parsing.consumedWholeFile()) {
    parsing.EmitMessage(llvm::errs(), parsing.finalRestingPlace(),
                        "parser FAIL (final position)",
                        "error: ", llvm::raw_ostream::RED);
    return mlir::failure();
  }
  if ((!parsing.messages().empty() && (parsing.messages().AnyFatalError())) ||
      !parsing.parseTree().has_value()) {
    llvm::errs() << programPrefix << "could not parse " << path << '\n';
    return mlir::failure();
  }

  // run semantics
  auto &parseTree = *parsing.parseTree();
  Fortran::semantics::Semantics semantics(semanticsContext, parseTree);
  semantics.Perform();
  semantics.EmitMessages(llvm::errs());
  if (semantics.AnyFatalError()) {
    llvm::errs() << programPrefix << "semantic errors in " << path << '\n';
    return mlir::failure();
  }
  Fortran::semantics::RuntimeDerivedTypeTables tables;
  if (!semantics.AnyFatalError()) {
    tables =
        Fortran::semantics::BuildRuntimeDerivedTypeTables(semanticsContext);
    if (!tables.schemata)
      llvm::errs() << programPrefix
                   << "could not find module file for __fortran_type_info\n";
  }

  if (dumpSymbols) {
    semantics.DumpSymbols(llvm::outs());
    return mlir::success();
  }

  if (pftDumpTest) {
    if (auto ast = Fortran::lower::createPFT(parseTree, semanticsContext)) {
      Fortran::lower::dumpPFT(llvm::outs(), *ast);
      return mlir::success();
    }
    llvm::errs() << "Pre FIR Tree is NULL.\n";
    return mlir::failure();
  }

  // translate to FIR dialect of MLIR
  mlir::DialectRegistry registry;
  fir::support::registerNonCodegenDialects(registry);
  mlir::MLIRContext ctx(registry);
  fir::support::loadNonCodegenDialects(ctx);
  auto &defKinds = semanticsContext.defaultKinds();
  fir::KindMapping kindMap(
      &ctx, llvm::ArrayRef<fir::KindTy>{fir::fromDefaultKinds(defKinds)});
  // Use default lowering options for bbc.
  Fortran::lower::LoweringOptions loweringOptions{};
  loweringOptions.setPolymorphicTypeImpl(enablePolymorphic);
  loweringOptions.setLowerToHighLevelFIR(useHLFIR);
  auto burnside = Fortran::lower::LoweringBridge::create(
      ctx, semanticsContext, defKinds, semanticsContext.intrinsics(),
      semanticsContext.targetCharacteristics(), parsing.allCooked(), "",
      kindMap, loweringOptions, {});
  burnside.lower(parseTree, semanticsContext);
  mlir::ModuleOp mlirModule = burnside.getModule();
  std::error_code ec;
  std::string outputName = outputFilename;
  if (!outputName.size())
    outputName = llvm::sys::path::stem(inputFilename).str().append(".mlir");
  llvm::raw_fd_ostream out(outputName, ec);
  if (ec)
    return mlir::emitError(mlir::UnknownLoc::get(&ctx),
                           "could not open output file ")
           << outputName;

  // Otherwise run the default passes.
  mlir::PassManager pm(mlirModule->getName(),
                       mlir::OpPassManager::Nesting::Implicit);
  pm.enableVerifier(/*verifyPasses=*/true);
  mlir::applyPassManagerCLOptions(pm);
  if (passPipeline.hasAnyOccurrences()) {
    // run the command-line specified pipeline
    (void)passPipeline.addToPipeline(pm, [&](const llvm::Twine &msg) {
      mlir::emitError(mlir::UnknownLoc::get(&ctx)) << msg;
      return mlir::failure();
    });
  } else if (emitFIR) {
    // --emit-fir: Build the IR, verify it, and dump the IR if the IR passes
    // verification. Use --dump-module-on-failure to dump invalid IR.
    pm.addPass(std::make_unique<Fortran::lower::VerifierPass>());
    if (mlir::failed(pm.run(mlirModule))) {
      llvm::errs() << "FATAL: verification of lowering to FIR failed";
      return mlir::failure();
    }
    printModule(mlirModule, out);
    return mlir::success();
  } else {
    // run the default canned pipeline
    pm.addPass(std::make_unique<Fortran::lower::VerifierPass>());

    // Add O2 optimizer pass pipeline.
    fir::createDefaultFIROptimizerPassPipeline(pm, llvm::OptimizationLevel::O2);
  }

  if (mlir::succeeded(pm.run(mlirModule))) {
    // Emit MLIR and do not lower to LLVM IR.
    printModule(mlirModule, out);
    return mlir::success();
  }
  // Something went wrong. Try to dump the MLIR module.
  llvm::errs() << "oops, pass manager reported failure\n";
  return mlir::failure();
}

int main(int argc, char **argv) {
  [[maybe_unused]] llvm::InitLLVM y(argc, argv);
  registerAllPasses();

  mlir::registerMLIRContextCLOptions();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipe("", "Compiler passes to run");
  llvm::cl::ParseCommandLineOptions(argc, argv, "Burnside Bridge Compiler\n");

  ProgramName programPrefix;
  programPrefix = argv[0] + ": "s;

  if (includeDirs.size() == 0) {
    includeDirs.push_back(".");
    // Default Fortran modules should be installed in include/flang (a sibling
    // to the bin) directory.
    intrinsicIncludeDirs.push_back(
        llvm::sys::path::parent_path(
            llvm::sys::path::parent_path(
                llvm::sys::fs::getMainExecutable(argv[0], nullptr)))
            .str() +
        "/include/flang");
  }

  Fortran::parser::Options options;
  options.predefinitions.emplace_back("__flang__"s, "1"s);
  options.predefinitions.emplace_back("__flang_major__"s,
                                      std::string{FLANG_VERSION_MAJOR_STRING});
  options.predefinitions.emplace_back("__flang_minor__"s,
                                      std::string{FLANG_VERSION_MINOR_STRING});
  options.predefinitions.emplace_back(
      "__flang_patchlevel__"s, std::string{FLANG_VERSION_PATCHLEVEL_STRING});

  // enable parsing of OpenMP
  if (enableOpenMP) {
    options.features.Enable(Fortran::common::LanguageFeature::OpenMP);
    options.predefinitions.emplace_back("_OPENMP", "201511");
  }

  // enable parsing of OpenACC
  if (enableOpenACC) {
    options.features.Enable(Fortran::common::LanguageFeature::OpenACC);
    options.predefinitions.emplace_back("_OPENACC", "201911");
  }

  Fortran::common::IntrinsicTypeDefaultKinds defaultKinds;
  Fortran::parser::AllSources allSources;
  Fortran::parser::AllCookedSources allCookedSources(allSources);
  Fortran::semantics::SemanticsContext semanticsContext{
      defaultKinds, options.features, allCookedSources};
  semanticsContext.set_moduleDirectory(moduleDir)
      .set_moduleFileSuffix(moduleSuffix)
      .set_searchDirectories(includeDirs)
      .set_intrinsicModuleDirectories(intrinsicIncludeDirs)
      .set_warnOnNonstandardUsage(warnStdViolation)
      .set_warningsAreErrors(warnIsError);

  llvm::Triple targetTriple{llvm::Triple(
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple()))};
  // FIXME: Handle real(3) ?
  if (targetTriple.getArch() != llvm::Triple::ArchType::x86 &&
      targetTriple.getArch() != llvm::Triple::ArchType::x86_64) {
    semanticsContext.targetCharacteristics().DisableType(
        Fortran::common::TypeCategory::Real, /*kind=*/10);
  }

  return mlir::failed(convertFortranSourceToMLIR(
      inputFilename, options, programPrefix, semanticsContext, passPipe));
}
