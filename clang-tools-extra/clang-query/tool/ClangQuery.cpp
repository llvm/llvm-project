//===---- ClangQuery.cpp - clang-query tool -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include <fstream>
#include <string>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::ast_matchers::dynamic;
using namespace clang::query;
using namespace clang::tooling;
using namespace llvm;

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::OptionCategory ClangQueryCategory("clang-query options");

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
  std::ifstream Input(FileName.c_str());
  if (!Input.is_open()) {
    llvm::errs() << ExeName << ": cannot open " << FileName << "\n";
    return 1;
  }
  while (Input.good()) {
    std::string Line;
    std::getline(Input, Line);

    QueryRef Q = QueryParser::parse(Line, QS);
    if (!Q->run(llvm::outs(), QS))
      return true;
  }
  return false;
}

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  CommonOptionsParser OptionsParser(argc, argv, ClangQueryCategory);

  if (!Commands.empty() && !CommandFiles.empty()) {
    llvm::errs() << argv[0] << ": cannot specify both -c and -f\n";
    return 1;
  }

  if ((!Commands.empty() || !CommandFiles.empty()) && !PreloadFile.empty()) {
    llvm::errs() << argv[0]
                 << ": cannot specify both -c or -f with --preload\n";
    return 1;
  }

  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());
  std::vector<std::unique_ptr<ASTUnit>> ASTs;
  int Status = Tool.buildASTs(ASTs);
  int ASTStatus = 0;
  if (Status == 1) {
    // Building ASTs failed.
    return 1;
  } else if (Status == 2) {
    ASTStatus |= 1;
    llvm::errs() << "Failed to build AST for some of the files, "
                 << "results may be incomplete."
                 << "\n";
  } else {
    assert(Status == 0 && "Unexpected status returned");
  }

  QuerySession QS(ASTs);

  if (!Commands.empty()) {
    for (auto I = Commands.begin(), E = Commands.end(); I != E; ++I) {
      QueryRef Q = QueryParser::parse(*I, QS);
      if (!Q->run(llvm::outs(), QS))
        return 1;
    }
  } else if (!CommandFiles.empty()) {
    for (auto I = CommandFiles.begin(), E = CommandFiles.end(); I != E; ++I) {
      if (runCommandsInFile(argv[0], *I, QS))
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
    while (llvm::Optional<std::string> Line = LE.readLine()) {
      QueryRef Q = QueryParser::parse(*Line, QS);
      Q->run(llvm::outs(), QS);
      llvm::outs().flush();
      if (QS.Terminate)
        break;
    }
  }

  return ASTStatus;
}
