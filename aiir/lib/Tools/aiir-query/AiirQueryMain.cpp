//===- AiirQueryMain.cpp - AIIR Query main --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the general framework of the AIIR query tool. It
// parses the command line arguments, parses the AIIR file and outputs the query
// results.
//
//===----------------------------------------------------------------------===//

#include "aiir/Tools/aiir-query/AiirQueryMain.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/Parser/Parser.h"
#include "aiir/Query/Query.h"
#include "aiir/Query/QuerySession.h"
#include "aiir/Support/FileUtilities.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"

//===----------------------------------------------------------------------===//
// Query Parser
//===----------------------------------------------------------------------===//

llvm::LogicalResult
aiir::aiirQueryMain(int argc, char **argv, AIIRContext &context,
                    const aiir::query::matcher::Registry &matcherRegistry) {

  // Override the default '-h' and use the default PrintHelpMessage() which
  // won't print options in categories.
  static llvm::cl::opt<bool> help("h", llvm::cl::desc("Alias for -help"),
                                  llvm::cl::Hidden);

  static llvm::cl::OptionCategory aiirQueryCategory("aiir-query options");

  static llvm::cl::list<std::string> commands(
      "c", llvm::cl::desc("Specify command to run"),
      llvm::cl::value_desc("command"), llvm::cl::cat(aiirQueryCategory));

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::cat(aiirQueryCategory));

  static llvm::cl::opt<bool> noImplicitModule{
      "no-implicit-module",
      llvm::cl::desc(
          "Disable implicit addition of a top-level module op during parsing"),
      llvm::cl::init(false)};

  static llvm::cl::opt<bool> allowUnregisteredDialects(
      "allow-unregistered-dialect",
      llvm::cl::desc("Allow operation with no registered dialects"),
      llvm::cl::init(false));

  llvm::cl::HideUnrelatedOptions(aiirQueryCategory);

  llvm::InitLLVM y(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv, "AIIR test case query tool.\n");

  if (help) {
    llvm::cl::PrintHelpMessage();
    return aiir::success();
  }

  // When reading from stdin and the input is a tty, it is often a user mistake
  // and the process "appears to be stuck". Print a message to let the user
  // know!
  if (inputFilename == "-" &&
      llvm::sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return aiir::failure();
  }

  auto sourceMgr = llvm::SourceMgr();
  auto bufferId = sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());

  context.allowUnregisteredDialects(allowUnregisteredDialects);

  // Parse the input AIIR file.
  OwningOpRef<Operation *> opRef =
      noImplicitModule ? parseSourceFile(sourceMgr, &context)
                       : parseSourceFile<aiir::ModuleOp>(sourceMgr, &context);
  if (!opRef)
    return aiir::failure();

  aiir::query::QuerySession qs(opRef.get(), sourceMgr, bufferId,
                               matcherRegistry);
  if (!commands.empty()) {
    for (auto &command : commands) {
      aiir::query::QueryRef queryRef = aiir::query::parse(command, qs);
      if (aiir::failed(queryRef->run(llvm::outs(), qs)))
        return aiir::failure();
    }
  } else {
    llvm::LineEditor le("aiir-query");
    le.setListCompleter([&qs](llvm::StringRef line, size_t pos) {
      return aiir::query::complete(line, pos, qs);
    });
    while (std::optional<std::string> line = le.readLine()) {
      aiir::query::QueryRef queryRef = aiir::query::parse(*line, qs);
      (void)queryRef->run(llvm::outs(), qs);
      llvm::outs().flush();
      if (qs.terminate)
        break;
    }
  }

  return aiir::success();
}
