//===- aiir-reduce.cpp - The AIIR reducer ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the general framework of the AIIR reducer tool. It
// parses the command line arguments, parses the initial AIIR test case and sets
// up the testing environment. It  outputs the most reduced test case variant
// after executing the reduction passes.
//
//===----------------------------------------------------------------------===//

#include "aiir/Tools/aiir-reduce/AiirReduceMain.h"
#include "aiir/Parser/Parser.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Reducer/Passes.h"
#include "aiir/Support/FileUtilities.h"
#include "aiir/Support/ToolUtilities.h"
#include "aiir/Tools/ParseUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace aiir;

LogicalResult aiir::aiirReduceMain(int argc, char **argv,
                                   AIIRContext &context) {
  // Override the default '-h' and use the default PrintHelpMessage() which
  // won't print options in categories.
  static llvm::cl::opt<bool> help("h", llvm::cl::desc("Alias for -help"),
                                  llvm::cl::Hidden);

  static llvm::cl::OptionCategory aiirReduceCategory("aiir-reduce options");

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::cat(aiirReduceCategory));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename for the reduced test case"),
      llvm::cl::init("-"), llvm::cl::cat(aiirReduceCategory));

  static llvm::cl::opt<bool> noImplicitModule{
      "no-implicit-module",
      llvm::cl::desc(
          "Disable implicit addition of a top-level module op during parsing"),
      llvm::cl::init(false)};

  static llvm::cl::opt<bool> allowUnregisteredDialects(
      "allow-unregistered-dialect",
      llvm::cl::desc("Allow operation with no registered dialects"),
      llvm::cl::init(false));

  static llvm::cl::opt<std::string> splitInputFile(
      "split-input-file", llvm::cl::ValueOptional,
      llvm::cl::callback([&](const std::string &str) {
        // Implicit value: use default marker if flag was used without
        // value.
        if (str.empty())
          splitInputFile.setValue(kDefaultSplitMarker);
      }),
      llvm::cl::desc("Split the input file into chunks using the given or "
                     "default marker and process each chunk independently"),
      llvm::cl::init(""));

  llvm::cl::HideUnrelatedOptions(aiirReduceCategory);

  llvm::InitLLVM y(argc, argv);

  registerReducerPasses();

  PassPipelineCLParser parser("", "Reduction Passes to Run");
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "AIIR test case reduction tool.\n");

  if (help) {
    llvm::cl::PrintHelpMessage();
    return success();
  }
  if (allowUnregisteredDialects)
    context.allowUnregisteredDialects();

  std::string errorMessage;

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output)
    return failure();

  std::unique_ptr<llvm::MemoryBuffer> input =
      openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto errorHandler = [&](const Twine &msg) {
    return emitError(UnknownLoc::get(&context)) << msg;
  };

  auto chunkFn = [&](std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
                     raw_ostream &os) {
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(chunkBuffer), SMLoc());
    OwningOpRef<Operation *> opRef =
        parseSourceFileForTool(sourceMgr, &context, !noImplicitModule);
    if (!opRef)
      return failure();
    // Reduction pass pipeline.
    PassManager pm(&context, opRef.get()->getName().getStringRef());
    if (failed(parser.addToPipeline(pm, errorHandler)))
      return failure();

    OwningOpRef<Operation *> op = opRef.get()->clone();

    if (failed(pm.run(op.get())))
      return failure();
    op.get()->print(output->os());
    output->keep();
    return success();
  };

  auto &splitInputFileDelimiter = splitInputFile.getValue();
  if (!splitInputFileDelimiter.empty())
    return splitAndProcessBuffer(std::move(input), chunkFn, output->os(),
                                 splitInputFileDelimiter,
                                 splitInputFileDelimiter);

  return chunkFn(std::move(input), output->os());
}
