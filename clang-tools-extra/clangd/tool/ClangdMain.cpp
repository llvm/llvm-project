//===--- ClangdMain.cpp - clangd server loop ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangdLSPServer.h"
#include "Path.h"
#include "Trace.h"
#include "Transport.h"
#include "index/Serialization.h"
#include "clang/Basic/Version.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

using namespace llvm;
using namespace clang;
using namespace clang::clangd;

// FIXME: remove this option when Dex is cheap enough.
static cl::opt<bool> UseDex("use-dex-index",
                            cl::desc("Use experimental Dex dynamic index."),
                            cl::init(false), cl::Hidden);

static cl::opt<Path> CompileCommandsDir(
    "compile-commands-dir",
    cl::desc("Specify a path to look for compile_commands.json. If path "
             "is invalid, clangd will look in the current directory and "
             "parent paths of each source file."));

static cl::opt<unsigned>
    WorkerThreadsCount("j", cl::desc("Number of async workers used by clangd"),
                       cl::init(getDefaultAsyncThreadsCount()));

// FIXME: also support "plain" style where signatures are always omitted.
enum CompletionStyleFlag { Detailed, Bundled };
static cl::opt<CompletionStyleFlag> CompletionStyle(
    "completion-style", cl::desc("Granularity of code completion suggestions"),
    cl::values(
        clEnumValN(Detailed, "detailed",
                   "One completion item for each semantically distinct "
                   "completion, with full type information."),
        clEnumValN(Bundled, "bundled",
                   "Similar completion items (e.g. function overloads) are "
                   "combined. Type information shown where possible.")),
    cl::init(Detailed));

// FIXME: Flags are the wrong mechanism for user preferences.
// We should probably read a dotfile or similar.
static cl::opt<bool> IncludeIneligibleResults(
    "include-ineligible-results",
    cl::desc("Include ineligible completion results (e.g. private members)"),
    cl::init(clangd::CodeCompleteOptions().IncludeIneligibleResults),
    cl::Hidden);

static cl::opt<JSONStreamStyle> InputStyle(
    "input-style", cl::desc("Input JSON stream encoding"),
    cl::values(
        clEnumValN(JSONStreamStyle::Standard, "standard", "usual LSP protocol"),
        clEnumValN(JSONStreamStyle::Delimited, "delimited",
                   "messages delimited by --- lines, with # comment support")),
    cl::init(JSONStreamStyle::Standard));

static cl::opt<bool> PrettyPrint("pretty", cl::desc("Pretty-print JSON output"),
                                 cl::init(false));

static cl::opt<Logger::Level> LogLevel(
    "log", cl::desc("Verbosity of log messages written to stderr"),
    cl::values(clEnumValN(Logger::Error, "error", "Error messages only"),
               clEnumValN(Logger::Info, "info", "High level execution tracing"),
               clEnumValN(Logger::Debug, "verbose", "Low level details")),
    cl::init(Logger::Info));

static cl::opt<bool>
    Test("lit-test",
         cl::desc("Abbreviation for -input-style=delimited -pretty "
                  "-run-synchronously -enable-test-scheme. "
                  "Intended to simplify lit tests."),
         cl::init(false), cl::Hidden);

static cl::opt<bool> EnableTestScheme(
    "enable-test-uri-scheme",
    cl::desc("Enable 'test:' URI scheme. Only use in lit tests."),
    cl::init(false), cl::Hidden);

enum PCHStorageFlag { Disk, Memory };
static cl::opt<PCHStorageFlag> PCHStorage(
    "pch-storage",
    cl::desc("Storing PCHs in memory increases memory usages, but may "
             "improve performance"),
    cl::values(clEnumValN(PCHStorageFlag::Disk, "disk", "store PCHs on disk"),
               clEnumValN(PCHStorageFlag::Memory, "memory",
                          "store PCHs in memory")),
    cl::init(PCHStorageFlag::Disk));

static cl::opt<int>
    LimitResults("limit-results",
                 cl::desc("Limit the number of results returned by clangd. "
                          "0 means no limit."),
                 cl::init(100));

static cl::opt<bool>
    RunSynchronously("run-synchronously",
                     cl::desc("Parse on main thread. If set, -j is ignored"),
                     cl::init(false), cl::Hidden);

static cl::opt<Path> ResourceDir("resource-dir",
                                 cl::desc("Directory for system clang headers"),
                                 cl::init(""), cl::Hidden);

static cl::opt<Path> InputMirrorFile(
    "input-mirror-file",
    cl::desc(
        "Mirror all LSP input to the specified file. Useful for debugging."),
    cl::init(""), cl::Hidden);

static cl::opt<bool> EnableIndex(
    "index",
    cl::desc(
        "Enable index-based features. By default, clangd maintains an index "
        "built from symbols in opened files. Global index support needs to "
        "enabled separatedly."),
    cl::init(true), cl::Hidden);

static cl::opt<bool> AllScopesCompletion(
    "all-scopes-completion",
    cl::desc(
        "If set to true, code completion will include index symbols that are "
        "not defined in the scopes (e.g. "
        "namespaces) visible from the code completion point. Such completions "
        "can insert scope qualifiers."),
    cl::init(false), cl::Hidden);

static cl::opt<bool>
    ShowOrigins("debug-origin", cl::desc("Show origins of completion items"),
                cl::init(clangd::CodeCompleteOptions().ShowOrigins),
                cl::Hidden);

static cl::opt<bool> HeaderInsertionDecorators(
    "header-insertion-decorators",
    cl::desc("Prepend a circular dot or space before the completion "
             "label, depending on whether "
             "an include line will be inserted or not."),
    cl::init(true));

static cl::opt<Path> IndexFile(
    "index-file",
    cl::desc(
        "Index file to build the static index. The file must have been created "
        "by a compatible clangd-index.\n"
        "WARNING: This option is experimental only, and will be removed "
        "eventually. Don't rely on it."),
    cl::init(""), cl::Hidden);

static cl::opt<bool> EnableBackgroundIndex(
    "background-index",
    cl::desc("Index project code in the background and persist index on disk. "
             "Experimental"),
    cl::init(false), cl::Hidden);

enum CompileArgsFrom { LSPCompileArgs, FilesystemCompileArgs };
static cl::opt<CompileArgsFrom> CompileArgsFrom(
    "compile_args_from", cl::desc("The source of compile commands"),
    cl::values(clEnumValN(LSPCompileArgs, "lsp",
                          "All compile commands come from LSP and "
                          "'compile_commands.json' files are ignored"),
               clEnumValN(FilesystemCompileArgs, "filesystem",
                          "All compile commands come from the "
                          "'compile_commands.json' files")),
    cl::init(FilesystemCompileArgs), cl::Hidden);

static cl::opt<bool> EnableFunctionArgSnippets(
    "function-arg-placeholders",
    cl::desc("When disabled, completions contain only parentheses for "
             "function calls. When enabled, completions also contain "
             "placeholders for method parameters."),
    cl::init(clangd::CodeCompleteOptions().EnableFunctionArgSnippets));

namespace {

/// \brief Supports a test URI scheme with relaxed constraints for lit tests.
/// The path in a test URI will be combined with a platform-specific fake
/// directory to form an absolute path. For example, test:///a.cpp is resolved
/// C:\clangd-test\a.cpp on Windows and /clangd-test/a.cpp on Unix.
class TestScheme : public URIScheme {
public:
  Expected<std::string> getAbsolutePath(StringRef /*Authority*/, StringRef Body,
                                        StringRef /*HintPath*/) const override {
    using namespace llvm::sys;
    // Still require "/" in body to mimic file scheme, as we want lengths of an
    // equivalent URI in both schemes to be the same.
    if (!Body.startswith("/"))
      return make_error<StringError>(
          "Expect URI body to be an absolute path starting with '/': " + Body,
          inconvertibleErrorCode());
    Body = Body.ltrim('/');
    SmallVector<char, 16> Path(Body.begin(), Body.end());
    path::native(Path);
    auto Err = fs::make_absolute(TestScheme::TestDir, Path);
    if (Err)
      llvm_unreachable("Failed to make absolute path in test scheme.");
    return std::string(Path.begin(), Path.end());
  }

  Expected<URI> uriFromAbsolutePath(StringRef AbsolutePath) const override {
    StringRef Body = AbsolutePath;
    if (!Body.consume_front(TestScheme::TestDir)) {
      return make_error<StringError>("Path " + AbsolutePath +
                                         " doesn't start with root " + TestDir,
                                     inconvertibleErrorCode());
    }

    return URI("test", /*Authority=*/"", sys::path::convert_to_slash(Body));
  }

private:
  const static char TestDir[];
};

#ifdef _WIN32
const char TestScheme::TestDir[] = "C:\\clangd-test";
#else
const char TestScheme::TestDir[] = "/clangd-test";
#endif

}

int main(int argc, char *argv[]) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  cl::SetVersionPrinter([](raw_ostream &OS) {
    OS << clang::getClangToolFullVersion("clangd") << "\n";
  });
  cl::ParseCommandLineOptions(
      argc, argv,
      "clangd is a language server that provides IDE-like features to editors. "
      "\n\nIt should be used via an editor plugin rather than invoked "
      "directly. "
      "For more information, see:"
      "\n\thttps://clang.llvm.org/extra/clangd.html"
      "\n\thttps://microsoft.github.io/language-server-protocol/");
  if (Test) {
    RunSynchronously = true;
    InputStyle = JSONStreamStyle::Delimited;
    PrettyPrint = true;
    preventThreadStarvationInTests(); // Ensure background index makes progress.
  }
  if (Test || EnableTestScheme) {
    static URISchemeRegistry::Add<TestScheme> X(
        "test", "Test scheme for clangd lit tests.");
  }

  if (!RunSynchronously && WorkerThreadsCount == 0) {
    errs() << "A number of worker threads cannot be 0. Did you mean to "
              "specify -run-synchronously?";
    return 1;
  }

  if (RunSynchronously) {
    if (WorkerThreadsCount.getNumOccurrences())
      errs() << "Ignoring -j because -run-synchronously is set.\n";
    WorkerThreadsCount = 0;
  }

  // Validate command line arguments.
  Optional<raw_fd_ostream> InputMirrorStream;
  if (!InputMirrorFile.empty()) {
    std::error_code EC;
    InputMirrorStream.emplace(InputMirrorFile, /*ref*/ EC,
                              sys::fs::FA_Read | sys::fs::FA_Write);
    if (EC) {
      InputMirrorStream.reset();
      errs() << "Error while opening an input mirror file: " << EC.message();
    } else {
      InputMirrorStream->SetUnbuffered();
    }
  }

  // Setup tracing facilities if CLANGD_TRACE is set. In practice enabling a
  // trace flag in your editor's config is annoying, launching with
  // `CLANGD_TRACE=trace.json vim` is easier.
  Optional<raw_fd_ostream> TraceStream;
  std::unique_ptr<trace::EventTracer> Tracer;
  if (auto *TraceFile = getenv("CLANGD_TRACE")) {
    std::error_code EC;
    TraceStream.emplace(TraceFile, /*ref*/ EC,
                        sys::fs::FA_Read | sys::fs::FA_Write);
    if (EC) {
      TraceStream.reset();
      errs() << "Error while opening trace file " << TraceFile << ": "
             << EC.message();
    } else {
      Tracer = trace::createJSONTracer(*TraceStream, PrettyPrint);
    }
  }

  Optional<trace::Session> TracingSession;
  if (Tracer)
    TracingSession.emplace(*Tracer);

  // Use buffered stream to stderr (we still flush each log message). Unbuffered
  // stream can cause significant (non-deterministic) latency for the logger.
  errs().SetBuffered();
  StreamLogger Logger(errs(), LogLevel);
  clangd::LoggingSession LoggingSession(Logger);

  // If --compile-commands-dir arg was invoked, check value and override default
  // path.
  Optional<Path> CompileCommandsDirPath;
  if (!CompileCommandsDir.empty()) {
    if (sys::fs::exists(CompileCommandsDir)) {
      // We support passing both relative and absolute paths to the
      // --compile-commands-dir argument, but we assume the path is absolute in
      // the rest of clangd so we make sure the path is absolute before
      // continuing.
      SmallString<128> Path(CompileCommandsDir);
      if (std::error_code EC = sys::fs::make_absolute(Path)) {
        errs() << "Error while converting the relative path specified by "
                  "--compile-commands-dir to an absolute path: "
               << EC.message() << ". The argument will be ignored.\n";
      } else {
        CompileCommandsDirPath = Path.str();
      }
    } else {
      errs() << "Path specified by --compile-commands-dir does not exist. The "
                "argument will be ignored.\n";
    }
  }

  ClangdServer::Options Opts;
  switch (PCHStorage) {
  case PCHStorageFlag::Memory:
    Opts.StorePreamblesInMemory = true;
    break;
  case PCHStorageFlag::Disk:
    Opts.StorePreamblesInMemory = false;
    break;
  }
  if (!ResourceDir.empty())
    Opts.ResourceDir = ResourceDir;
  Opts.BuildDynamicSymbolIndex = EnableIndex;
  Opts.HeavyweightDynamicSymbolIndex = UseDex;
  Opts.BackgroundIndex = EnableBackgroundIndex;
  std::unique_ptr<SymbolIndex> StaticIdx;
  std::future<void> AsyncIndexLoad; // Block exit while loading the index.
  if (EnableIndex && !IndexFile.empty()) {
    // Load the index asynchronously. Meanwhile SwapIndex returns no results.
    SwapIndex *Placeholder;
    StaticIdx.reset(Placeholder = new SwapIndex(llvm::make_unique<MemIndex>()));
    AsyncIndexLoad = runAsync<void>([Placeholder] {
      if (auto Idx = loadIndex(IndexFile, /*UseDex=*/true))
        Placeholder->reset(std::move(Idx));
    });
    if (RunSynchronously)
      AsyncIndexLoad.wait();
  }
  Opts.StaticIndex = StaticIdx.get();
  Opts.AsyncThreadsCount = WorkerThreadsCount;

  clangd::CodeCompleteOptions CCOpts;
  CCOpts.IncludeIneligibleResults = IncludeIneligibleResults;
  CCOpts.Limit = LimitResults;
  CCOpts.BundleOverloads = CompletionStyle != Detailed;
  CCOpts.ShowOrigins = ShowOrigins;
  if (!HeaderInsertionDecorators) {
    CCOpts.IncludeIndicator.Insert.clear();
    CCOpts.IncludeIndicator.NoInsert.clear();
  }
  CCOpts.SpeculativeIndexRequest = Opts.StaticIndex;
  CCOpts.EnableFunctionArgSnippets = EnableFunctionArgSnippets;
  CCOpts.AllScopes = AllScopesCompletion;

  // Initialize and run ClangdLSPServer.
  // Change stdin to binary to not lose \r\n on windows.
  sys::ChangeStdinToBinary();
  auto Transport = newJSONTransport(
      stdin, outs(),
      InputMirrorStream ? InputMirrorStream.getPointer() : nullptr, PrettyPrint,
      InputStyle);
  ClangdLSPServer LSPServer(
      *Transport, CCOpts, CompileCommandsDirPath,
      /*UseDirBasedCDB=*/CompileArgsFrom == FilesystemCompileArgs, Opts);
  constexpr int NoShutdownRequestErrorCode = 1;
  set_thread_name("clangd.main");
  return LSPServer.run() ? 0 : NoShutdownRequestErrorCode;
}
