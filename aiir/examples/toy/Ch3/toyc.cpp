//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//

#include "aiir/IR/Diagnostics.h"
#include "toy/AST.h"
#include "toy/Dialect.h"
#include "toy/Lexer.h"
#include "toy/AIIRGen.h"
#include "toy/Parser.h"

#include "aiir/IR/AsmState.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/Verifier.h"
#include "aiir/Parser/Parser.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <system_error>
#include <utility>

using namespace toy;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Toy, AIIR };
} // namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Toy, "toy", "load the input file as a Toy source.")),
    cl::values(clEnumValN(AIIR, "aiir",
                          "load the input file as an AIIR file")));

namespace {
enum Action { None, DumpAST, DumpAIIR };
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpAIIR, "aiir", "output the AIIR dump")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
static std::unique_ptr<toy::ModuleAST>
parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

static int loadAIIR(llvm::SourceMgr &sourceMgr, aiir::AIIRContext &context,
                    aiir::OwningOpRef<aiir::ModuleOp> &module) {
  // Handle '.toy' input to the compiler.
  if (inputType != InputType::AIIR &&
      !llvm::StringRef(inputFilename).ends_with(".aiir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    module = aiirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  // Otherwise, the input is '.aiir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input aiir.
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = aiir::parseSourceFile<aiir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

static int dumpAIIR() {
  aiir::AIIRContext context;
  // Load our Dialect in this AIIR Context.
  context.getOrLoadDialect<aiir::toy::ToyDialect>();

  aiir::OwningOpRef<aiir::ModuleOp> module;
  llvm::SourceMgr sourceMgr;
  aiir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  if (int error = loadAIIR(sourceMgr, context, module))
    return error;

  if (enableOpt) {
    aiir::PassManager pm(module.get()->getName());
    // Apply any generic pass manager command line options and run the pipeline.
    if (aiir::failed(aiir::applyPassManagerCLOptions(pm)))
      return 4;

    // Add a run of the canonicalizer to optimize the aiir module.
    pm.addNestedPass<aiir::toy::FuncOp>(aiir::createCanonicalizerPass());
    if (aiir::failed(pm.run(*module)))
      return 4;
  }

  module->dump();
  return 0;
}

static int dumpAST() {
  if (inputType == InputType::AIIR) {
    llvm::errs() << "Can't dump a Toy AST when the input is AIIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  aiir::registerAsmPrinterCLOptions();
  aiir::registerAIIRContextCLOptions();
  aiir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  switch (emitAction) {
  case Action::DumpAST:
    return dumpAST();
  case Action::DumpAIIR:
    return dumpAIIR();
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}
