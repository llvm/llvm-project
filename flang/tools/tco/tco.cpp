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

#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
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

static void printModuleBody(mlir::ModuleOp mod, raw_ostream &output) {
  for (auto &op : mod.getBody()->without_terminator())
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
  auto owningRef = mlir::parseSourceFile(sourceMgr, &context);

  if (!owningRef) {
    errs() << "Error can't load file " << inputFilename << '\n';
    return mlir::failure();
  }
  if (mlir::failed(owningRef->verify())) {
    errs() << "Error verifying FIR module\n";
    return mlir::failure();
  }

  std::error_code ec;
  ToolOutputFile out(outputFilename, ec, sys::fs::OF_None);

  // run passes
  mlir::PassManager pm{&context};
  mlir::applyPassManagerCLOptions(pm);
  if (emitFir) {
    // parse the input and pretty-print it back out
    // -emit-fir intentionally disables all the passes
  } else {
    // TODO: Actually add passes when added to FIR code base
    // add all the passes
    // the user can disable them individually
  }

  // run the pass manager
  if (mlir::succeeded(pm.run(*owningRef))) {
    // passes ran successfully, so keep the output
    if (emitFir)
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
  fir::support::registerFIRPasses();
  [[maybe_unused]] InitLLVM y(argc, argv);
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipe("", "Compiler passes to run");
  cl::ParseCommandLineOptions(argc, argv, "Tilikum Crossing Optimizer\n");
  return mlir::failed(compileFIR(passPipe));
}
