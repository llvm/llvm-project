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
#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
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

static cl::opt<std::string> targetTriple("target",
                                         cl::desc("specify a target triple"),
                                         cl::init("native"));

static cl::opt<bool> codeGenLLVM(
    "code-gen-llvm",
    cl::desc("Run only CodeGen passes and translate FIR to LLVM IR"),
    cl::init(false));

#include "flang/Tools/CLOptions.inc"

static void printModuleBody(mlir::ModuleOp mod, raw_ostream &output) {
  for (auto &op : *mod.getBody())
    output << op << '\n';
}

// compile a .fir file
static mlir::LogicalResult
compileFIR(const mlir::PassPipelineCLParser &passPipeline) {
  // check that there is a file to load
  ErrorOr<std::unique_ptr<MemoryBuffer>> fileOrErr =
      MemoryBuffer::getFileOrSTDIN(inputFilename);

  if (std::error_code EC = fileOrErr.getError()) {
    errs() << "Could not open file: " << EC.message() << '\n';
    return mlir::failure();
  }

  // load the file into a module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::DialectRegistry registry;
  fir::support::registerDialects(registry);
  mlir::MLIRContext context(registry);
  fir::support::loadDialects(context);
  fir::support::registerLLVMTranslation(context);
  auto owningRef = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

  if (!owningRef) {
    errs() << "Error can't load file " << inputFilename << '\n';
    return mlir::failure();
  }
  if (mlir::failed(owningRef->verifyInvariants())) {
    errs() << "Error verifying FIR module\n";
    return mlir::failure();
  }

  std::error_code ec;
  ToolOutputFile out(outputFilename, ec, sys::fs::OF_None);

  // run passes
  fir::KindMapping kindMap{&context};
  fir::setTargetTriple(*owningRef, targetTriple);
  fir::setKindMapping(*owningRef, kindMap);
  mlir::PassManager pm((*owningRef)->getName(),
                       mlir::OpPassManager::Nesting::Implicit);
  pm.enableVerifier(/*verifyPasses=*/true);
  mlir::applyPassManagerCLOptions(pm);
  if (emitFir) {
    // parse the input and pretty-print it back out
    // -emit-fir intentionally disables all the passes
  } else if (passPipeline.hasAnyOccurrences()) {
    auto errorHandler = [&](const Twine &msg) {
      mlir::emitError(mlir::UnknownLoc::get(pm.getContext())) << msg;
      return mlir::failure();
    };
    if (mlir::failed(passPipeline.addToPipeline(pm, errorHandler)))
      return mlir::failure();
  } else {
    if (codeGenLLVM) {
      // Run only CodeGen passes.
      fir::createDefaultFIRCodeGenPassPipeline(pm);
    } else {
      // Run tco with O2 by default.
      fir::createMLIRToLLVMPassPipeline(pm, llvm::OptimizationLevel::O2);
    }
    fir::addLLVMDialectToLLVMPass(pm, out.os());
  }

  // run the pass manager
  if (mlir::succeeded(pm.run(*owningRef))) {
    // passes ran successfully, so keep the output
    if ((emitFir || passPipeline.hasAnyOccurrences()) && !codeGenLLVM)
      printModuleBody(*owningRef, out.os());
    out.keep();
    return mlir::success();
  }

  // pass manager failed
  printModuleBody(*owningRef, errs());
  errs() << "\n\nFAILED: " << inputFilename << '\n';
  return mlir::failure();
}

int main(int argc, char **argv) {
  // Disable the ExternalNameConversion pass by default until all the tests have
  // been updated to pass with it enabled.
  disableExternalNameConversion = true;

  [[maybe_unused]] InitLLVM y(argc, argv);
  fir::support::registerMLIRPassesForFortranTools();
  fir::registerOptCodeGenPasses();
  fir::registerOptTransformPasses();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipe("", "Compiler passes to run");
  cl::ParseCommandLineOptions(argc, argv, "Tilikum Crossing Optimizer\n");
  return mlir::failed(compileFIR(passPipe));
}
