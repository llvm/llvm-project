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
#include "Path.h"
#include "Trace.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <memory>
#include <string>
#include <thread>

using namespace clang;
using namespace clang::clangd;

namespace {
enum class PCHStorageFlag { Disk, Memory };
}

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
        "Present snippet completions instead of plaintext completions. "
        "This also enables code pattern results." /* FIXME: should it? */),
    llvm::cl::init(clangd::CodeCompleteOptions().EnableSnippets));

// FIXME: Flags are the wrong mechanism for user preferences.
// We should probably read a dotfile or similar.
static llvm::cl::opt<bool> IncludeIneligibleResults(
    "include-ineligible-results",
    llvm::cl::desc(
        "Include ineligible completion results (e.g. private members)"),
    llvm::cl::init(clangd::CodeCompleteOptions().IncludeIneligibleResults),
    llvm::cl::Hidden);

static llvm::cl::opt<bool>
    PrettyPrint("pretty", llvm::cl::desc("Pretty-print JSON output"),
                llvm::cl::init(false));

static llvm::cl::opt<PCHStorageFlag> PCHStorage(
    "pch-storage",
    llvm::cl::desc("Storing PCHs in memory increases memory usages, but may "
                   "improve performance"),
    llvm::cl::values(
        clEnumValN(PCHStorageFlag::Disk, "disk", "store PCHs on disk"),
        clEnumValN(PCHStorageFlag::Memory, "memory", "store PCHs in memory")),
    llvm::cl::init(PCHStorageFlag::Disk));

static llvm::cl::opt<bool> RunSynchronously(
    "run-synchronously",
    llvm::cl::desc("Parse on main thread. If set, -j is ignored"),
    llvm::cl::init(false), llvm::cl::Hidden);

static llvm::cl::opt<Path>
    ResourceDir("resource-dir",
                llvm::cl::desc("Directory for system clang headers"),
                llvm::cl::init(""), llvm::cl::Hidden);

static llvm::cl::opt<Path> InputMirrorFile(
    "input-mirror-file",
    llvm::cl::desc(
        "Mirror all LSP input to the specified file. Useful for debugging."),
    llvm::cl::init(""), llvm::cl::Hidden);

static llvm::cl::opt<Path> TraceFile(
    "trace",
    llvm::cl::desc(
        "Trace internal events and timestamps in chrome://tracing JSON format"),
    llvm::cl::init(""), llvm::cl::Hidden);

static llvm::cl::opt<bool> EnableIndexBasedCompletion(
    "enable-index-based-completion",
    llvm::cl::desc(
        "Enable index-based global code completion (experimental). Clangd will "
        "use index built from symbols in opened files"),
    llvm::cl::init(false), llvm::cl::Hidden);

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

  // Validate command line arguments.
  llvm::Optional<llvm::raw_fd_ostream> InputMirrorStream;
  if (!InputMirrorFile.empty()) {
    std::error_code EC;
    InputMirrorStream.emplace(InputMirrorFile, /*ref*/ EC, llvm::sys::fs::F_RW);
    if (EC) {
      InputMirrorStream.reset();
      llvm::errs() << "Error while opening an input mirror file: "
                   << EC.message();
    }
  }

  // Setup tracing facilities.
  llvm::Optional<llvm::raw_fd_ostream> TraceStream;
  std::unique_ptr<trace::EventTracer> Tracer;
  if (!TraceFile.empty()) {
    std::error_code EC;
    TraceStream.emplace(TraceFile, /*ref*/ EC, llvm::sys::fs::F_RW);
    if (EC) {
      TraceFile.reset();
      llvm::errs() << "Error while opening trace file: " << EC.message();
    } else {
      Tracer = trace::createJSONTracer(*TraceStream, PrettyPrint);
    }
  }

  llvm::Optional<trace::Session> TracingSession;
  if (Tracer)
    TracingSession.emplace(*Tracer);

  llvm::raw_ostream &Outs = llvm::outs();
  llvm::raw_ostream &Logs = llvm::errs();
  JSONOutput Out(Outs, Logs,
                 InputMirrorStream ? InputMirrorStream.getPointer() : nullptr,
                 PrettyPrint);

  clangd::LoggingSession LoggingSession(Out);

  // If --compile-commands-dir arg was invoked, check value and override default
  // path.
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

  bool StorePreamblesInMemory;
  switch (PCHStorage) {
  case PCHStorageFlag::Memory:
    StorePreamblesInMemory = true;
    break;
  case PCHStorageFlag::Disk:
    StorePreamblesInMemory = false;
    break;
  }

  llvm::Optional<StringRef> ResourceDirRef = None;
  if (!ResourceDir.empty())
    ResourceDirRef = ResourceDir;

  // Change stdin to binary to not lose \r\n on windows.
  llvm::sys::ChangeStdinToBinary();

  clangd::CodeCompleteOptions CCOpts;
  CCOpts.EnableSnippets = EnableSnippets;
  CCOpts.IncludeIneligibleResults = IncludeIneligibleResults;
  // Initialize and run ClangdLSPServer.
  ClangdLSPServer LSPServer(Out, WorkerThreadsCount, StorePreamblesInMemory,
                            CCOpts, ResourceDirRef, CompileCommandsDirPath,
                            EnableIndexBasedCompletion);
  constexpr int NoShutdownRequestErrorCode = 1;
  llvm::set_thread_name("clangd.main");
  return LSPServer.run(std::cin) ? 0 : NoShutdownRequestErrorCode;
}
