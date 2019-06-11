//===--- ClangdMain.cpp - clangd server loop ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangdLSPServer.h"
#include "CodeComplete.h"
#include "Features.inc"
#include "Path.h"
#include "Protocol.h"
#include "Trace.h"
#include "Transport.h"
#include "index/Background.h"
#include "index/Serialization.h"
#include "clang/Basic/Version.h"
#include "clang/Format/Format.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace clang {
namespace clangd {

static llvm::cl::opt<Path> CompileCommandsDir(
    "compile-commands-dir",
    llvm::cl::desc("Specify a path to look for compile_commands.json. If path "
                   "is invalid, clangd will look in the current directory and "
                   "parent paths of each source file"));

static llvm::cl::opt<unsigned>
    WorkerThreadsCount("j",
                       llvm::cl::desc("Number of async workers used by clangd"),
                       llvm::cl::init(getDefaultAsyncThreadsCount()));

// FIXME: also support "plain" style where signatures are always omitted.
enum CompletionStyleFlag { Detailed, Bundled };
static llvm::cl::opt<CompletionStyleFlag> CompletionStyle(
    "completion-style",
    llvm::cl::desc("Granularity of code completion suggestions"),
    llvm::cl::values(
        clEnumValN(Detailed, "detailed",
                   "One completion item for each semantically distinct "
                   "completion, with full type information"),
        clEnumValN(Bundled, "bundled",
                   "Similar completion items (e.g. function overloads) are "
                   "combined. Type information shown where possible")),
    llvm::cl::init(Detailed));

// FIXME: Flags are the wrong mechanism for user preferences.
// We should probably read a dotfile or similar.
static llvm::cl::opt<bool> IncludeIneligibleResults(
    "include-ineligible-results",
    llvm::cl::desc(
        "Include ineligible completion results (e.g. private members)"),
    llvm::cl::init(CodeCompleteOptions().IncludeIneligibleResults),
    llvm::cl::Hidden);

static llvm::cl::opt<JSONStreamStyle> InputStyle(
    "input-style", llvm::cl::desc("Input JSON stream encoding"),
    llvm::cl::values(
        clEnumValN(JSONStreamStyle::Standard, "standard", "usual LSP protocol"),
        clEnumValN(JSONStreamStyle::Delimited, "delimited",
                   "messages delimited by --- lines, with # comment support")),
    llvm::cl::init(JSONStreamStyle::Standard));

static llvm::cl::opt<bool>
    PrettyPrint("pretty", llvm::cl::desc("Pretty-print JSON output"),
                llvm::cl::init(false));

static llvm::cl::opt<Logger::Level> LogLevel(
    "log", llvm::cl::desc("Verbosity of log messages written to stderr"),
    llvm::cl::values(clEnumValN(Logger::Error, "error", "Error messages only"),
                     clEnumValN(Logger::Info, "info",
                                "High level execution tracing"),
                     clEnumValN(Logger::Debug, "verbose", "Low level details")),
    llvm::cl::init(Logger::Info));

static llvm::cl::opt<bool>
    Test("lit-test",
         llvm::cl::desc("Abbreviation for -input-style=delimited -pretty -sync "
                        "-enable-test-scheme -log=verbose."
                        "Intended to simplify lit tests"),
         llvm::cl::init(false), llvm::cl::Hidden);

static llvm::cl::opt<bool> EnableTestScheme(
    "enable-test-uri-scheme",
    llvm::cl::desc("Enable 'test:' URI scheme. Only use in lit tests"),
    llvm::cl::init(false), llvm::cl::Hidden);

enum PCHStorageFlag { Disk, Memory };
static llvm::cl::opt<PCHStorageFlag> PCHStorage(
    "pch-storage",
    llvm::cl::desc("Storing PCHs in memory increases memory usages, but may "
                   "improve performance"),
    llvm::cl::values(
        clEnumValN(PCHStorageFlag::Disk, "disk", "store PCHs on disk"),
        clEnumValN(PCHStorageFlag::Memory, "memory", "store PCHs in memory")),
    llvm::cl::init(PCHStorageFlag::Disk));

static llvm::cl::opt<int> LimitResults(
    "limit-results",
    llvm::cl::desc("Limit the number of results returned by clangd. "
                   "0 means no limit (default=100)"),
    llvm::cl::init(100));

static llvm::cl::opt<bool>
    Sync("sync", llvm::cl::desc("Parse on main thread. If set, -j is ignored"),
         llvm::cl::init(false), llvm::cl::Hidden);

static llvm::cl::opt<Path>
    ResourceDir("resource-dir",
                llvm::cl::desc("Directory for system clang headers"),
                llvm::cl::init(""), llvm::cl::Hidden);

static llvm::cl::opt<Path> InputMirrorFile(
    "input-mirror-file",
    llvm::cl::desc(
        "Mirror all LSP input to the specified file. Useful for debugging"),
    llvm::cl::init(""), llvm::cl::Hidden);

static llvm::cl::opt<bool> EnableIndex(
    "index",
    llvm::cl::desc(
        "Enable index-based features. By default, clangd maintains an index "
        "built from symbols in opened files. Global index support needs to "
        "enabled separatedly"),
    llvm::cl::init(true), llvm::cl::Hidden);

static llvm::cl::opt<bool> AllScopesCompletion(
    "all-scopes-completion",
    llvm::cl::desc(
        "If set to true, code completion will include index symbols that are "
        "not defined in the scopes (e.g. "
        "namespaces) visible from the code completion point. Such completions "
        "can insert scope qualifiers"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> ShowOrigins(
    "debug-origin", llvm::cl::desc("Show origins of completion items"),
    llvm::cl::init(CodeCompleteOptions().ShowOrigins), llvm::cl::Hidden);

static llvm::cl::opt<CodeCompleteOptions::IncludeInsertion> HeaderInsertion(
    "header-insertion",
    llvm::cl::desc("Add #include directives when accepting code completions"),
    llvm::cl::init(CodeCompleteOptions().InsertIncludes),
    llvm::cl::values(
        clEnumValN(CodeCompleteOptions::IWYU, "iwyu",
                   "Include what you use. "
                   "Insert the owning header for top-level symbols, unless the "
                   "header is already directly included or the symbol is "
                   "forward-declared"),
        clEnumValN(
            CodeCompleteOptions::NeverInsert, "never",
            "Never insert #include directives as part of code completion")));

static llvm::cl::opt<bool> HeaderInsertionDecorators(
    "header-insertion-decorators",
    llvm::cl::desc("Prepend a circular dot or space before the completion "
                   "label, depending on whether "
                   "an include line will be inserted or not"),
    llvm::cl::init(true));

static llvm::cl::opt<Path> IndexFile(
    "index-file",
    llvm::cl::desc(
        "Index file to build the static index. The file must have been created "
        "by a compatible clangd-indexer\n"
        "WARNING: This option is experimental only, and will be removed "
        "eventually. Don't rely on it"),
    llvm::cl::init(""), llvm::cl::Hidden);

static llvm::cl::opt<bool> EnableBackgroundIndex(
    "background-index",
    llvm::cl::desc(
        "Index project code in the background and persist index on disk. "
        "Experimental"),
    llvm::cl::init(false), llvm::cl::Hidden);

static llvm::cl::opt<int> BackgroundIndexRebuildPeriod(
    "background-index-rebuild-period",
    llvm::cl::desc(
        "If set to non-zero, the background index rebuilds the symbol index "
        "periodically every X milliseconds; otherwise, the "
        "symbol index will be updated for each indexed file"),
    llvm::cl::init(5000), llvm::cl::Hidden);

enum CompileArgsFrom { LSPCompileArgs, FilesystemCompileArgs };
static llvm::cl::opt<CompileArgsFrom> CompileArgsFrom(
    "compile_args_from", llvm::cl::desc("The source of compile commands"),
    llvm::cl::values(clEnumValN(LSPCompileArgs, "lsp",
                                "All compile commands come from LSP and "
                                "'compile_commands.json' files are ignored"),
                     clEnumValN(FilesystemCompileArgs, "filesystem",
                                "All compile commands come from the "
                                "'compile_commands.json' files")),
    llvm::cl::init(FilesystemCompileArgs), llvm::cl::Hidden);

static llvm::cl::opt<bool> EnableFunctionArgSnippets(
    "function-arg-placeholders",
    llvm::cl::desc("When disabled, completions contain only parentheses for "
                   "function calls. When enabled, completions also contain "
                   "placeholders for method parameters"),
    llvm::cl::init(CodeCompleteOptions().EnableFunctionArgSnippets));

static llvm::cl::opt<std::string> ClangTidyChecks(
    "clang-tidy-checks",
    llvm::cl::desc(
        "List of clang-tidy checks to run (this will override "
        ".clang-tidy files). Only meaningful when -clang-tidy flag is on"),
    llvm::cl::init(""));

static llvm::cl::opt<bool>
    EnableClangTidy("clang-tidy",
                    llvm::cl::desc("Enable clang-tidy diagnostics"),
                    llvm::cl::init(true));

static llvm::cl::opt<std::string>
    FallbackStyle("fallback-style",
                  llvm::cl::desc("clang-format style to apply by default when "
                                 "no .clang-format file is found"),
                  llvm::cl::init(clang::format::DefaultFallbackStyle));

static llvm::cl::opt<bool> SuggestMissingIncludes(
    "suggest-missing-includes",
    llvm::cl::desc("Attempts to fix diagnostic errors caused by missing "
                   "includes using index"),
    llvm::cl::init(true));

static llvm::cl::opt<OffsetEncoding> ForceOffsetEncoding(
    "offset-encoding",
    llvm::cl::desc("Force the offsetEncoding used for character positions. "
                   "This bypasses negotiation via client capabilities"),
    llvm::cl::values(clEnumValN(OffsetEncoding::UTF8, "utf-8",
                                "Offsets are in UTF-8 bytes"),
                     clEnumValN(OffsetEncoding::UTF16, "utf-16",
                                "Offsets are in UTF-16 code units")),
    llvm::cl::init(OffsetEncoding::UnsupportedEncoding));

static llvm::cl::opt<CodeCompleteOptions::CodeCompletionParse>
    CodeCompletionParse(
        "completion-parse",
        llvm::cl::desc("Whether the clang-parser is used for code-completion"),
        llvm::cl::values(clEnumValN(CodeCompleteOptions::AlwaysParse, "always",
                                    "Block until the parser can be used"),
                         clEnumValN(CodeCompleteOptions::ParseIfReady, "auto",
                                    "Use text-based completion if the parser "
                                    "is not ready"),
                         clEnumValN(CodeCompleteOptions::NeverParse, "never",
                                    "Always used text-based completion")),
        llvm::cl::init(CodeCompleteOptions().RunParser), llvm::cl::Hidden);

namespace {

/// \brief Supports a test URI scheme with relaxed constraints for lit tests.
/// The path in a test URI will be combined with a platform-specific fake
/// directory to form an absolute path. For example, test:///a.cpp is resolved
/// C:\clangd-test\a.cpp on Windows and /clangd-test/a.cpp on Unix.
class TestScheme : public URIScheme {
public:
  llvm::Expected<std::string>
  getAbsolutePath(llvm::StringRef /*Authority*/, llvm::StringRef Body,
                  llvm::StringRef /*HintPath*/) const override {
    using namespace llvm::sys;
    // Still require "/" in body to mimic file scheme, as we want lengths of an
    // equivalent URI in both schemes to be the same.
    if (!Body.startswith("/"))
      return llvm::make_error<llvm::StringError>(
          "Expect URI body to be an absolute path starting with '/': " + Body,
          llvm::inconvertibleErrorCode());
    Body = Body.ltrim('/');
    llvm::SmallVector<char, 16> Path(Body.begin(), Body.end());
    path::native(Path);
    fs::make_absolute(TestScheme::TestDir, Path);
    return std::string(Path.begin(), Path.end());
  }

  llvm::Expected<URI>
  uriFromAbsolutePath(llvm::StringRef AbsolutePath) const override {
    llvm::StringRef Body = AbsolutePath;
    if (!Body.consume_front(TestScheme::TestDir)) {
      return llvm::make_error<llvm::StringError>(
          "Path " + AbsolutePath + " doesn't start with root " + TestDir,
          llvm::inconvertibleErrorCode());
    }

    return URI("test", /*Authority=*/"",
               llvm::sys::path::convert_to_slash(Body));
  }

private:
  const static char TestDir[];
};

#ifdef _WIN32
const char TestScheme::TestDir[] = "C:\\clangd-test";
#else
const char TestScheme::TestDir[] = "/clangd-test";
#endif

} // namespace
} // namespace clangd
} // namespace clang

enum class ErrorResultCode : int {
  NoShutdownRequest = 1,
  CantRunAsXPCService = 2
};

int main(int argc, char *argv[]) {
  using namespace clang;
  using namespace clang::clangd;

  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::cl::SetVersionPrinter([](llvm::raw_ostream &OS) {
    OS << clang::getClangToolFullVersion("clangd") << "\n";
  });
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "clangd is a language server that provides IDE-like features to editors. "
      "\n\nIt should be used via an editor plugin rather than invoked "
      "directly. "
      "For more information, see:"
      "\n\thttps://clang.llvm.org/extra/clangd.html"
      "\n\thttps://microsoft.github.io/language-server-protocol/");
  if (Test) {
    Sync = true;
    InputStyle = JSONStreamStyle::Delimited;
    LogLevel = Logger::Verbose;
    PrettyPrint = true;
    // Ensure background index makes progress.
    BackgroundIndex::preventThreadStarvationInTests();
  }
  if (Test || EnableTestScheme) {
    static URISchemeRegistry::Add<TestScheme> X(
        "test", "Test scheme for clangd lit tests.");
  }

  if (!Sync && WorkerThreadsCount == 0) {
    llvm::errs() << "A number of worker threads cannot be 0. Did you mean to "
                    "specify -sync?";
    return 1;
  }

  if (Sync) {
    if (WorkerThreadsCount.getNumOccurrences())
      llvm::errs() << "Ignoring -j because -sync is set.\n";
    WorkerThreadsCount = 0;
  }
  if (FallbackStyle.getNumOccurrences())
    clang::format::DefaultFallbackStyle = FallbackStyle.c_str();

  // Validate command line arguments.
  llvm::Optional<llvm::raw_fd_ostream> InputMirrorStream;
  if (!InputMirrorFile.empty()) {
    std::error_code EC;
    InputMirrorStream.emplace(InputMirrorFile, /*ref*/ EC,
                              llvm::sys::fs::FA_Read | llvm::sys::fs::FA_Write);
    if (EC) {
      InputMirrorStream.reset();
      llvm::errs() << "Error while opening an input mirror file: "
                   << EC.message();
    } else {
      InputMirrorStream->SetUnbuffered();
    }
  }

  // Setup tracing facilities if CLANGD_TRACE is set. In practice enabling a
  // trace flag in your editor's config is annoying, launching with
  // `CLANGD_TRACE=trace.json vim` is easier.
  llvm::Optional<llvm::raw_fd_ostream> TraceStream;
  std::unique_ptr<trace::EventTracer> Tracer;
  if (auto *TraceFile = getenv("CLANGD_TRACE")) {
    std::error_code EC;
    TraceStream.emplace(TraceFile, /*ref*/ EC,
                        llvm::sys::fs::FA_Read | llvm::sys::fs::FA_Write);
    if (EC) {
      TraceStream.reset();
      llvm::errs() << "Error while opening trace file " << TraceFile << ": "
                   << EC.message();
    } else {
      Tracer = trace::createJSONTracer(*TraceStream, PrettyPrint);
    }
  }

  llvm::Optional<trace::Session> TracingSession;
  if (Tracer)
    TracingSession.emplace(*Tracer);

  // Use buffered stream to stderr (we still flush each log message). Unbuffered
  // stream can cause significant (non-deterministic) latency for the logger.
  llvm::errs().SetBuffered();
  StreamLogger Logger(llvm::errs(), LogLevel);
  LoggingSession LoggingSession(Logger);

  // If --compile-commands-dir arg was invoked, check value and override default
  // path.
  llvm::Optional<Path> CompileCommandsDirPath;
  if (!CompileCommandsDir.empty()) {
    if (llvm::sys::fs::exists(CompileCommandsDir)) {
      // We support passing both relative and absolute paths to the
      // --compile-commands-dir argument, but we assume the path is absolute in
      // the rest of clangd so we make sure the path is absolute before
      // continuing.
      llvm::SmallString<128> Path(CompileCommandsDir);
      if (std::error_code EC = llvm::sys::fs::make_absolute(Path)) {
        llvm::errs() << "Error while converting the relative path specified by "
                        "--compile-commands-dir to an absolute path: "
                     << EC.message() << ". The argument will be ignored.\n";
      } else {
        CompileCommandsDirPath = Path.str();
      }
    } else {
      llvm::errs()
          << "Path specified by --compile-commands-dir does not exist. The "
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
  Opts.BackgroundIndex = EnableBackgroundIndex;
  Opts.BackgroundIndexRebuildPeriodMs = BackgroundIndexRebuildPeriod;
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
    if (Sync)
      AsyncIndexLoad.wait();
  }
  Opts.StaticIndex = StaticIdx.get();
  Opts.AsyncThreadsCount = WorkerThreadsCount;

  clangd::CodeCompleteOptions CCOpts;
  CCOpts.IncludeIneligibleResults = IncludeIneligibleResults;
  CCOpts.Limit = LimitResults;
  CCOpts.BundleOverloads = CompletionStyle != Detailed;
  CCOpts.ShowOrigins = ShowOrigins;
  CCOpts.InsertIncludes = HeaderInsertion;
  if (!HeaderInsertionDecorators) {
    CCOpts.IncludeIndicator.Insert.clear();
    CCOpts.IncludeIndicator.NoInsert.clear();
  }
  CCOpts.SpeculativeIndexRequest = Opts.StaticIndex;
  CCOpts.EnableFunctionArgSnippets = EnableFunctionArgSnippets;
  CCOpts.AllScopes = AllScopesCompletion;
  CCOpts.RunParser = CodeCompletionParse;

  RealFileSystemProvider FSProvider;
  // Initialize and run ClangdLSPServer.
  // Change stdin to binary to not lose \r\n on windows.
  llvm::sys::ChangeStdinToBinary();

  std::unique_ptr<Transport> TransportLayer;
  if (getenv("CLANGD_AS_XPC_SERVICE")) {
#if CLANGD_BUILD_XPC
    TransportLayer = newXPCTransport();
#else
    llvm::errs() << "This clangd binary wasn't built with XPC support.\n";
    return (int)ErrorResultCode::CantRunAsXPCService;
#endif
  } else {
    TransportLayer = newJSONTransport(
        stdin, llvm::outs(),
        InputMirrorStream ? InputMirrorStream.getPointer() : nullptr,
        PrettyPrint, InputStyle);
  }

  // Create an empty clang-tidy option.
  std::mutex ClangTidyOptMu;
  std::unique_ptr<tidy::ClangTidyOptionsProvider>
      ClangTidyOptProvider; /*GUARDED_BY(ClangTidyOptMu)*/
  if (EnableClangTidy) {
    auto OverrideClangTidyOptions = tidy::ClangTidyOptions::getDefaults();
    OverrideClangTidyOptions.Checks = ClangTidyChecks;
    ClangTidyOptProvider = llvm::make_unique<tidy::FileOptionsProvider>(
        tidy::ClangTidyGlobalOptions(),
        /* Default */ tidy::ClangTidyOptions::getDefaults(),
        /* Override */ OverrideClangTidyOptions, FSProvider.getFileSystem());
    Opts.GetClangTidyOptions = [&](llvm::vfs::FileSystem &,
                                   llvm::StringRef File) {
      // This function must be thread-safe and tidy option providers are not.
      std::lock_guard<std::mutex> Lock(ClangTidyOptMu);
      // FIXME: use the FS provided to the function.
      return ClangTidyOptProvider->getOptions(File);
    };
  }
  Opts.SuggestMissingIncludes = SuggestMissingIncludes;
  llvm::Optional<OffsetEncoding> OffsetEncodingFromFlag;
  if (ForceOffsetEncoding != OffsetEncoding::UnsupportedEncoding)
    OffsetEncodingFromFlag = ForceOffsetEncoding;
  ClangdLSPServer LSPServer(
      *TransportLayer, FSProvider, CCOpts, CompileCommandsDirPath,
      /*UseDirBasedCDB=*/CompileArgsFrom == FilesystemCompileArgs,
      OffsetEncodingFromFlag, Opts);
  llvm::set_thread_name("clangd.main");
  return LSPServer.run() ? 0
                         : static_cast<int>(ErrorResultCode::NoShutdownRequest);
}
