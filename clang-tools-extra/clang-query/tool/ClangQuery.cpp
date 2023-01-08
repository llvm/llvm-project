//===---- ClangQuery.cpp - clang-query tool -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool is for interactive exploration of the Clang AST using AST matchers.
// It currently allows the user to enter a matcher at an interactive prompt and
// view the resulting bindings as diagnostics, AST pretty prints or AST dumps.
// Example session:
//
// $ cat foo.c
// void foo(void) {}
// $ clang-query foo.c --
// clang-query> match functionDecl()
//
// Match #1:
//
// foo.c:1:1: note: "root" binds here
// void foo(void) {}
// ^~~~~~~~~~~~~~~~~
// 1 match.
//
//===----------------------------------------------------------------------===//

#include "Query.h"
#include "QueryParser.h"
#include "QuerySession.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include <optional>
#include <string>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::ast_matchers::dynamic;
using namespace clang::query;
using namespace clang::tooling;
using namespace llvm;

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::OptionCategory ClangQueryCategory("clang-query options");

static cl::opt<bool>
    UseColor("use-color",
             cl::desc(
                 R"(Use colors in detailed AST output. If not set, colors
will be used if the terminal connected to
standard output supports colors.)"),
             cl::init(false), cl::cat(ClangQueryCategory));

static cl::list<std::string> Commands("c", cl::desc("Specify command to run"),
                                      cl::value_desc("command"),
                                      cl::cat(ClangQueryCategory));

static cl::list<std::string> CommandFiles("f",
                                          cl::desc("Read commands from file"),
                                          cl::value_desc("file"),
                                          cl::cat(ClangQueryCategory));

static cl::opt<std::string> PreloadFile(
    "preload",
    cl::desc("Preload commands from file and start interactive mode"),
    cl::value_desc("file"), cl::cat(ClangQueryCategory));

bool runCommandsInFile(const char *ExeName, std::string const &FileName,
                       QuerySession &QS) {
  auto Buffer = llvm::MemoryBuffer::getFile(FileName);
  if (!Buffer) {
    llvm::errs() << ExeName << ": cannot open " << FileName << ": "
                 << Buffer.getError().message() << "\n";
    return true;
  }

  StringRef FileContentRef(Buffer.get()->getBuffer());

  while (!FileContentRef.empty()) {
    QueryRef Q = QueryParser::parse(FileContentRef, QS);
    if (!Q->run(llvm::outs(), QS))
      return true;
    FileContentRef = Q->RemainingContent;
  }
  return false;
}

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  llvm::Expected<CommonOptionsParser> OptionsParser =
      CommonOptionsParser::create(argc, argv, ClangQueryCategory,
                                  llvm::cl::OneOrMore);

  if (!OptionsParser) {
    llvm::WithColor::error() << llvm::toString(OptionsParser.takeError());
    return 1;
  }

  if (!Commands.empty() && !CommandFiles.empty()) {
    llvm::errs() << argv[0] << ": cannot specify both -c and -f\n";
    return 1;
  }

  if ((!Commands.empty() || !CommandFiles.empty()) && !PreloadFile.empty()) {
    llvm::errs() << argv[0]
                 << ": cannot specify both -c or -f with --preload\n";
    return 1;
  }

  ClangTool Tool(OptionsParser->getCompilations(),
                 OptionsParser->getSourcePathList());

  if (UseColor.getNumOccurrences() > 0) {
    ArgumentsAdjuster colorAdjustor = [](const CommandLineArguments &Args, StringRef /*unused*/) {
      CommandLineArguments AdjustedArgs = Args;
      if (UseColor)
        AdjustedArgs.push_back("-fdiagnostics-color");
      else
        AdjustedArgs.push_back("-fno-diagnostics-color");
      return AdjustedArgs;
    };
    Tool.appendArgumentsAdjuster(colorAdjustor);
  }

  std::vector<std::unique_ptr<ASTUnit>> ASTs;
  int ASTStatus = 0;
  switch (Tool.buildASTs(ASTs)) {
  case 0:
    break;
  case 1: // Building ASTs failed.
    return 1;
  case 2:
    ASTStatus |= 1;
    llvm::errs() << "Failed to build AST for some of the files, "
                 << "results may be incomplete."
                 << "\n";
    break;
  default:
    llvm_unreachable("Unexpected status returned");
  }

  QuerySession QS(ASTs);

  if (!Commands.empty()) {
    for (auto &Command : Commands) {
      QueryRef Q = QueryParser::parse(Command, QS);
      if (!Q->run(llvm::outs(), QS))
        return 1;
    }
  } else if (!CommandFiles.empty()) {
    for (auto &CommandFile : CommandFiles) {
      if (runCommandsInFile(argv[0], CommandFile, QS))
        return 1;
    }
  } else {
    if (!PreloadFile.empty()) {
      if (runCommandsInFile(argv[0], PreloadFile, QS))
        return 1;
    }
    LineEditor LE("clang-query");
    LE.setListCompleter([&QS](StringRef Line, size_t Pos) {
      return QueryParser::complete(Line, Pos, QS);
    });
    while (std::optional<std::string> Line = LE.readLine()) {
      QueryRef Q = QueryParser::parse(*Line, QS);
      Q->run(llvm::outs(), QS);
      llvm::outs().flush();
      if (QS.Terminate)
        break;
    }
  }

  return ASTStatus;
}
