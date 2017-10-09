//===--- ClangdMain.cpp - clangd server loop ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangdLSPServer.h"
#include "JSONRPCDispatcher.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include <iostream>
#include <memory>
#include <string>
#include <thread>

using namespace clang;
using namespace clang::clangd;

static llvm::cl::opt<Path> CompileCommandsDir(
    "compile-commands-dir",
    llvm::cl::desc("Specify a path to look for compile_commands.json. If path "
                   "is invalid, clangd will look in the current directory and "
                   "parent paths of each source file."));

static llvm::cl::opt<unsigned>
    WorkerThreadsCount("j",
                       llvm::cl::desc("Number of async workers used by clangd"),
                       llvm::cl::init(getDefaultAsyncThreadsCount()));

static llvm::cl::opt<bool> EnableSnippets(
    "enable-snippets",
    llvm::cl::desc(
        "Present snippet completions instead of plaintext completions"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> RunSynchronously(
    "run-synchronously",
    llvm::cl::desc("Parse on main thread. If set, -j is ignored"),
    llvm::cl::init(false), llvm::cl::Hidden);

static llvm::cl::opt<std::string>
    ResourceDir("resource-dir",
                llvm::cl::desc("Directory for system clang headers"),
                llvm::cl::init(""), llvm::cl::Hidden);

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "clangd");

  if (!RunSynchronously && WorkerThreadsCount == 0) {
    llvm::errs() << "A number of worker threads cannot be 0. Did you mean to "
                    "specify -run-synchronously?";
    return 1;
  }

  // Ignore -j option if -run-synchonously is used.
  // FIXME: a warning should be shown here.
  if (RunSynchronously)
    WorkerThreadsCount = 0;

  /// Validate command line arguments.
  llvm::raw_ostream &Outs = llvm::outs();
  llvm::raw_ostream &Logs = llvm::errs();
  JSONOutput Out(Outs, Logs);

  // If --compile-commands-dir arg was invoked, check value and override default
  // path.
  namespace path = llvm::sys::path;
  llvm::Optional<Path> CompileCommandsDirPath;

  if (CompileCommandsDir.empty()) {
    CompileCommandsDirPath = llvm::None;
  } else if (!llvm::sys::path::is_absolute(CompileCommandsDir) ||
             !llvm::sys::fs::exists(CompileCommandsDir)) {
    llvm::errs() << "Path specified by --compile-commands-dir either does not "
                    "exist or is not an absolute "
                    "path. The argument will be ignored.\n";
    CompileCommandsDirPath = llvm::None;
  } else {
    CompileCommandsDirPath = CompileCommandsDir;
  }

  llvm::Optional<StringRef> ResourceDirRef = None;
  if (!ResourceDir.empty())
    ResourceDirRef = ResourceDir;

  /// Change stdin to binary to not lose \r\n on windows.
  llvm::sys::ChangeStdinToBinary();

  /// Initialize and run ClangdLSPServer.
  ClangdLSPServer LSPServer(Out, WorkerThreadsCount, EnableSnippets,
                            ResourceDirRef, CompileCommandsDirPath);
  LSPServer.run(std::cin);
}
