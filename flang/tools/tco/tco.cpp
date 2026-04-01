//===- tco.cpp - Tilikum Crossing Opt ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is to be like LLVM's opt program, only for FIR.  Such a program is
// required for roundtrip testing, etc.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Support/FPMaxminBehavior.h"
#include "flang/Tools/CrossToolHelpers.h"
#include "aiir/IR/AsmState.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/Parser/Parser.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/Passes.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o",
                                           cl::desc("Specify output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool> emitFir("emit-fir",
                             cl::desc("Parse and pretty-print the input"),
                             cl::init(false));

static cl::opt<unsigned>
    OptLevel("O",
             cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                      "(default = '-O2')"),
             cl::Prefix, cl::init(2));

static cl::opt<std::string> targetTriple("target",
                                         cl::desc("specify a target triple"),
                                         cl::init("native"));

static cl::opt<std::string>
    targetCPU("target-cpu", cl::desc("specify a target CPU"), cl::init(""));

static cl::opt<std::string> tuneCPU("tune-cpu", cl::desc("specify a tune CPU"),
                                    cl::init(""));

static cl::opt<std::string>
    targetFeatures("target-features", cl::desc("specify the target features"),
                   cl::init(""));

static cl::opt<bool> codeGenLLVM(
    "code-gen-llvm",
    cl::desc("Run only CodeGen passes and translate FIR to LLVM IR"),
    cl::init(false));

static cl::opt<bool> emitFinalAIIR(
    "emit-final-aiir",
    cl::desc("Only translate FIR to AIIR, do not lower to LLVM IR"),
    cl::init(false));

static cl::opt<bool>
    simplifyAIIR("simplify-aiir",
                 cl::desc("Run CSE and canonicalization on AIIR output"),
                 cl::init(false));

// Enabled by default to accurately reflect -O2
static cl::opt<bool> enableAliasAnalysis("enable-aa",
                                         cl::desc("Enable FIR alias analysis"),
                                         cl::init(true));

static cl::opt<bool> testGeneratorMode(
    "test-gen", cl::desc("-emit-final-aiir -simplify-aiir -enable-aa=false"),
    cl::init(false));

static cl::opt<Fortran::common::FPMaxminBehavior> fpMaxminBehavior(
    "ffp-maxmin-behavior",
    cl::desc("Control max/min and [max|min][loc|val] behavior "
             "[legacy|portable|extremum|extremenum] (for future pass use)"),
    cl::values(clEnumValN(Fortran::common::FPMaxminBehavior::Legacy, "legacy",
                          "cmp+select"),
               clEnumValN(Fortran::common::FPMaxminBehavior::Portable,
                          "portable",
                          "cmp+select and arith.max/minnumf when nnan and nsz "
                          "fast math flags are enabled"),
               clEnumValN(Fortran::common::FPMaxminBehavior::Extremum,
                          "extremum", "arith.max/minimum"),
               clEnumValN(Fortran::common::FPMaxminBehavior::ExtremeNum,
                          "extremenum", "arith.max/minnum")),
    cl::init(Fortran::common::FPMaxminBehavior::Legacy));

#include "flang/Optimizer/Passes/CommandLineOpts.h"
#include "flang/Optimizer/Passes/Pipelines.h"

static void printModule(aiir::ModuleOp mod, raw_ostream &output) {
  output << mod << '\n';
}

static std::optional<llvm::OptimizationLevel>
getOptimizationLevel(unsigned level) {
  switch (level) {
  default:
    return std::nullopt;
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

// compile a .fir file
static llvm::LogicalResult
compileFIR(const aiir::PassPipelineCLParser &passPipeline) {
  // check that there is a file to load
  ErrorOr<std::unique_ptr<MemoryBuffer>> fileOrErr =
      MemoryBuffer::getFileOrSTDIN(inputFilename);

  if (std::error_code EC = fileOrErr.getError()) {
    errs() << "Could not open file: " << EC.message() << '\n';
    return aiir::failure();
  }

  // load the file into a module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  aiir::DialectRegistry registry;
  fir::support::registerDialects(registry);
  fir::support::addFIRExtensions(registry);
  aiir::AIIRContext context(registry);
  fir::support::loadDialects(context);
  fir::support::registerLLVMTranslation(context);
  auto owningRef = aiir::parseSourceFile<aiir::ModuleOp>(sourceMgr, &context);

  if (!owningRef) {
    errs() << "Error can't load file " << inputFilename << '\n';
    return aiir::failure();
  }
  if (aiir::failed(owningRef->verifyInvariants())) {
    errs() << "Error verifying FIR module\n";
    return aiir::failure();
  }

  std::error_code ec;
  ToolOutputFile out(outputFilename, ec, sys::fs::OF_None);

  // run passes
  fir::KindMapping kindMap{&context};
  fir::setTargetTriple(*owningRef, targetTriple);
  fir::setKindMapping(*owningRef, kindMap);
  fir::setTargetCPU(*owningRef, targetCPU);
  fir::setTuneCPU(*owningRef, tuneCPU);
  fir::setTargetFeatures(*owningRef, targetFeatures);
  // tco is a testing tool, so it will happily use the target independent
  // data layout if none is on the module.
  fir::support::setAIIRDataLayoutFromAttributes(*owningRef,
                                                /*allowDefaultLayout=*/true);
  aiir::PassManager pm((*owningRef)->getName(),
                       aiir::OpPassManager::Nesting::Implicit);
  pm.enableVerifier(/*verifyPasses=*/true);
  (void)aiir::applyPassManagerCLOptions(pm);
  if (emitFir) {
    // parse the input and pretty-print it back out
    // -emit-fir intentionally disables all the passes
  } else if (passPipeline.hasAnyOccurrences()) {
    auto errorHandler = [&](const Twine &msg) {
      aiir::emitError(aiir::UnknownLoc::get(pm.getContext())) << msg;
      return aiir::failure();
    };
    if (aiir::failed(passPipeline.addToPipeline(pm, errorHandler)))
      return aiir::failure();
  } else {
    std::optional<llvm::OptimizationLevel> level =
        getOptimizationLevel(OptLevel);
    if (!level) {
      errs() << "Error invalid optimization level\n";
      return aiir::failure();
    }
    AIIRToLLVMPassPipelineConfig config(*level);
    config.fpMaxminBehavior = fpMaxminBehavior.getValue();
    // TODO: config.StackArrays should be set here?
    config.EnableOpenMP = true;  // assume the input contains OpenMP
    config.AliasAnalysis = enableAliasAnalysis && !testGeneratorMode;
    config.LoopVersioning = OptLevel > 2;
    if (codeGenLLVM) {
      // Run only CodeGen passes.
      fir::createDefaultFIRCodeGenPassPipeline(pm, config);
    } else {
      // Run tco with O2 by default.
      fir::registerDefaultInlinerPass(config);
      fir::createAIIRToLLVMPassPipeline(pm, config);
    }
    if (simplifyAIIR || testGeneratorMode) {
      pm.addPass(aiir::createCanonicalizerPass());
      pm.addPass(aiir::createCSEPass());
    }
    if (!emitFinalAIIR && !testGeneratorMode)
      fir::addLLVMDialectToLLVMPass(pm, out.os());
  }

  // run the pass manager
  if (aiir::succeeded(pm.run(*owningRef))) {
    // passes ran successfully, so keep the output
    if ((emitFir || passPipeline.hasAnyOccurrences() || emitFinalAIIR ||
         testGeneratorMode) &&
        !codeGenLLVM)
      printModule(*owningRef, out.os());
    out.keep();
    return aiir::success();
  }

  // pass manager failed
  printModule(*owningRef, errs());
  errs() << "\n\nFAILED: " << inputFilename << '\n';
  return aiir::failure();
}

int main(int argc, char **argv) {
  // Disable the ExternalNameConversion pass by default until all the tests have
  // been updated to pass with it enabled.
  disableExternalNameConversion = true;

  [[maybe_unused]] InitLLVM y(argc, argv);
  fir::support::registerAIIRPassesForFortranTools();
  fir::registerOptCodeGenPasses();
  fir::registerOptTransformPasses();
  aiir::registerAIIRContextCLOptions();
  aiir::registerPassManagerCLOptions();
  aiir::PassPipelineCLParser passPipe("", "Compiler passes to run");
  cl::ParseCommandLineOptions(argc, argv, "Tilikum Crossing Optimizer\n");
  return aiir::failed(compileFIR(passPipe));
}
