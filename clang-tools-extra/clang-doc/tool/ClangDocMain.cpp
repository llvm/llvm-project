//===-- ClangDocMain.cpp - ClangDoc -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tool for generating C and C++ documenation from source code
// and comments. Generally, it runs a LibTooling FrontendAction on source files,
// mapping each declaration in those files to its USR and serializing relevant
// information into LLVM bitcode. It then runs a pass over the collected
// declaration information, reducing by USR. There is an option to dump this
// intermediate result to bitcode. Finally, it hands the reduced information
// off to a generator, which does the final parsing from the intermediate
// representation to the desired output format.
//
//===----------------------------------------------------------------------===//

#include "ClangDoc.h"
#include "clang/AST/AST.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Execution.h"
#include "clang/Tooling/StandaloneExecution.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace clang;

static llvm::cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static llvm::cl::OptionCategory ClangDocCategory("clang-doc options");

static llvm::cl::opt<std::string>
    OutDirectory("output",
                 llvm::cl::desc("Directory for outputting generated files."),
                 llvm::cl::init("docs"), llvm::cl::cat(ClangDocCategory));

static llvm::cl::opt<bool>
    DumpMapperResult("dump-mapper",
                     llvm::cl::desc("Dump mapper results to bitcode file."),
                     llvm::cl::init(false), llvm::cl::cat(ClangDocCategory));

static llvm::cl::opt<bool> DoxygenOnly(
    "doxygen",
    llvm::cl::desc("Use only doxygen-style comments to generate docs."),
    llvm::cl::init(false), llvm::cl::cat(ClangDocCategory));

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  std::error_code OK;

  auto Exec = clang::tooling::createExecutorFromCommandLineArgs(
      argc, argv, ClangDocCategory);

  if (!Exec) {
    llvm::errs() << toString(Exec.takeError()) << "\n";
    return 1;
  }

  ArgumentsAdjuster ArgAdjuster;
  if (!DoxygenOnly)
    ArgAdjuster = combineAdjusters(
        getInsertArgumentAdjuster("-fparse-all-comments",
                                  tooling::ArgumentInsertPosition::END),
        ArgAdjuster);

  // Mapping phase
  llvm::outs() << "Mapping decls...\n";
  auto Err = Exec->get()->execute(doc::newMapperActionFactory(
                                      Exec->get()->getExecutionContext()),
                                  ArgAdjuster);
  if (Err)
    llvm::errs() << toString(std::move(Err)) << "\n";

  if (DumpMapperResult) {
    Exec->get()->getToolResults()->forEachResult([&](StringRef Key,
                                                     StringRef Value) {
      SmallString<128> IRRootPath;
      llvm::sys::path::native(OutDirectory, IRRootPath);
      llvm::sys::path::append(IRRootPath, "bc");
      std::error_code DirectoryStatus =
          llvm::sys::fs::create_directories(IRRootPath);
      if (DirectoryStatus != OK) {
        llvm::errs() << "Unable to create documentation directories.\n";
        return;
      }
      llvm::sys::path::append(IRRootPath, Key + ".bc");
      std::error_code OutErrorInfo;
      llvm::raw_fd_ostream OS(IRRootPath, OutErrorInfo, llvm::sys::fs::F_None);
      if (OutErrorInfo != OK) {
        llvm::errs() << "Error opening documentation file.\n";
        return;
      }
      OS << Value;
      OS.close();
    });
  }

  return 0;
}
