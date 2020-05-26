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
#include "PathMapping.h"
#include "Protocol.h"
#include "Transport.h"
#include "index/Background.h"
#include "index/Serialization.h"
#include "refactor/Rename.h"
#include "support/Path.h"
#include "support/Shutdown.h"
#include "support/Trace.h"
#include "clang/Basic/Version.h"
#include "clang/Format/Format.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace clang {
namespace clangd {
namespace {

using llvm::cl::cat;
using llvm::cl::CommaSeparated;
using llvm::cl::desc;
using llvm::cl::Hidden;
using llvm::cl::init;
using llvm::cl::list;
using llvm::cl::opt;
using llvm::cl::OptionCategory;
using llvm::cl::values;

// All flags must be placed in a category, or they will be shown neither in
// --help, nor --help-hidden!
OptionCategory CompileCommands("clangd compilation flags options");
OptionCategory Features("clangd feature options");
OptionCategory Misc("clangd miscellaneous options");
OptionCategory Protocol("clangd protocol and logging options");
const OptionCategory *ClangdCategories[] = {&Features, &Protocol,
                                            &CompileCommands, &Misc};

enum CompileArgsFrom { LSPCompileArgs, FilesystemCompileArgs };
opt<CompileArgsFrom> CompileArgsFrom{
    "compile_args_from",
    cat(CompileCommands),
    desc("The source of compile commands"),
    values(clEnumValN(LSPCompileArgs, "lsp",
                      "All compile commands come from LSP and "
                      "'compile_commands.json' files are ignored"),
           clEnumValN(FilesystemCompileArgs, "filesystem",
                      "All compile commands come from the "
                      "'compile_commands.json' files")),
    init(FilesystemCompileArgs),
    Hidden,
};

opt<Path> CompileCommandsDir{
    "compile-commands-dir",
    cat(CompileCommands),
    desc("Specify a path to look for compile_commands.json. If path "
         "is invalid, clangd will look in the current directory and "
         "parent paths of each source file"),
};

opt<Path> ResourceDir{
    "resource-dir",
    cat(CompileCommands),
    desc("Directory for system clang headers"),
    init(""),
    Hidden,
};

list<std::string> QueryDriverGlobs{
    "query-driver",
    cat(CompileCommands),
    desc(
        "Comma separated list of globs for white-listing gcc-compatible "
        "drivers that are safe to execute. Drivers matching any of these globs "
        "will be used to extract system includes. e.g. "
        "/usr/bin/**/clang-*,/path/to/repo/**/g++-*"),
    CommaSeparated,
};

// FIXME: Flags are the wrong mechanism for user preferences.
// We should probably read a dotfile or similar.
opt<bool> AllScopesCompletion{
    "all-scopes-completion",
    cat(Features),
    desc("If set to true, code completion will include index symbols that are "
         "not defined in the scopes (e.g. "
         "namespaces) visible from the code completion point. Such completions "
         "can insert scope qualifiers"),
    init(true),
};

opt<bool> ShowOrigins{
    "debug-origin",
    cat(Features),
    desc("Show origins of completion items"),
    init(CodeCompleteOptions().ShowOrigins),
    Hidden,
};

opt<bool> EnableBackgroundIndex{
    "background-index",
    cat(Features),
    desc("Index project code in the background and persist index on disk."),
    init(true),
};

opt<bool> EnableClangTidy{
    "clang-tidy",
    cat(Features),
    desc("Enable clang-tidy diagnostics"),
    init(true),
};

opt<std::string> ClangTidyChecks{
    "clang-tidy-checks",
    cat(Features),
    desc("List of clang-tidy checks to run (this will override "
         ".clang-tidy files). Only meaningful when -clang-tidy flag is on"),
    init(""),
};

opt<CodeCompleteOptions::CodeCompletionParse> CodeCompletionParse{
    "completion-parse",
    cat(Features),
    desc("Whether the clang-parser is used for code-completion"),
    values(clEnumValN(CodeCompleteOptions::AlwaysParse, "always",
                      "Block until the parser can be used"),
           clEnumValN(CodeCompleteOptions::ParseIfReady, "auto",
                      "Use text-based completion if the parser "
                      "is not ready"),
           clEnumValN(CodeCompleteOptions::NeverParse, "never",
                      "Always used text-based completion")),
    init(CodeCompleteOptions().RunParser),
    Hidden,
};

// FIXME: also support "plain" style where signatures are always omitted.
enum CompletionStyleFlag { Detailed, Bundled };
opt<CompletionStyleFlag> CompletionStyle{
    "completion-style",
    cat(Features),
    desc("Granularity of code completion suggestions"),
    values(clEnumValN(Detailed, "detailed",
                      "One completion item for each semantically distinct "
                      "completion, with full type information"),
           clEnumValN(Bundled, "bundled",
                      "Similar completion items (e.g. function overloads) are "
                      "combined. Type information shown where possible")),
};

opt<std::string> FallbackStyle{
    "fallback-style",
    cat(Features),
    desc("clang-format style to apply by default when "
         "no .clang-format file is found"),
    init(clang::format::DefaultFallbackStyle),
};

opt<bool> EnableFunctionArgSnippets{
    "function-arg-placeholders",
    cat(Features),
    desc("When disabled, completions contain only parentheses for "
         "function calls. When enabled, completions also contain "
         "placeholders for method parameters"),
    init(CodeCompleteOptions().EnableFunctionArgSnippets),
    Hidden,
};

opt<CodeCompleteOptions::IncludeInsertion> HeaderInsertion{
    "header-insertion",
    cat(Features),
    desc("Add #include directives when accepting code completions"),
    init(CodeCompleteOptions().InsertIncludes),
    values(
        clEnumValN(CodeCompleteOptions::IWYU, "iwyu",
                   "Include what you use. "
                   "Insert the owning header for top-level symbols, unless the "
                   "header is already directly included or the symbol is "
                   "forward-declared"),
        clEnumValN(
            CodeCompleteOptions::NeverInsert, "never",
            "Never insert #include directives as part of code completion")),
};

opt<bool> HeaderInsertionDecorators{
    "header-insertion-decorators",
    cat(Features),
    desc("Prepend a circular dot or space before the completion "
         "label, depending on whether "
         "an include line will be inserted or not"),
    init(true),
};

opt<bool> HiddenFeatures{
    "hidden-features",
    cat(Features),
    desc("Enable hidden features mostly useful to clangd developers"),
    init(false),
    Hidden,
};

opt<bool> IncludeIneligibleResults{
    "include-ineligible-results",
    cat(Features),
    desc("Include ineligible completion results (e.g. private members)"),
    init(CodeCompleteOptions().IncludeIneligibleResults),
    Hidden,
};

opt<bool> EnableIndex{
    "index",
    cat(Features),
    desc("Enable index-based features. By default, clangd maintains an index "
         "built from symbols in opened files. Global index support needs to "
         "enabled separatedly"),
    init(true),
    Hidden,
};

opt<int> LimitResults{
    "limit-results",
    cat(Features),
    desc("Limit the number of results returned by clangd. "
         "0 means no limit (default=100)"),
    init(100),
};

opt<bool> SuggestMissingIncludes{
    "suggest-missing-includes",
    cat(Features),
    desc("Attempts to fix diagnostic errors caused by missing "
         "includes using index"),
    init(true),
};

list<std::string> TweakList{
    "tweaks",
    cat(Features),
    desc("Specify a list of Tweaks to enable (only for clangd developers)."),
    Hidden,
    CommaSeparated,
};

opt<bool> CrossFileRename{
    "cross-file-rename",
    cat(Features),
    desc("Enable cross-file rename feature."),
    init(true),
};

opt<bool> RecoveryAST{
    "recovery-ast",
    cat(Features),
    desc("Preserve expressions in AST for broken code (C++ only). Note that "
         "this feature is experimental and may lead to crashes"),
    init(false),
    Hidden,
};
opt<bool> RecoveryASTType{
    "recovery-ast-type",
    cat(Features),
    desc("Preserve the type for recovery AST. Note that "
         "this feature is experimental and may lead to crashes"),
    init(false),
    Hidden,
};

opt<unsigned> WorkerThreadsCount{
    "j",
    cat(Misc),
    desc("Number of async workers used by clangd. Background index also "
         "uses this many workers."),
    init(getDefaultAsyncThreadsCount()),
};

opt<Path> IndexFile{
    "index-file",
    cat(Misc),
    desc(
        "Index file to build the static index. The file must have been created "
        "by a compatible clangd-indexer\n"
        "WARNING: This option is experimental only, and will be removed "
        "eventually. Don't rely on it"),
    init(""),
    Hidden,
};

opt<bool> Test{
    "lit-test",
    cat(Misc),
    desc("Abbreviation for -input-style=delimited -pretty -sync "
         "-enable-test-scheme -log=verbose. "
         "Intended to simplify lit tests"),
    init(false),
    Hidden,
};

enum PCHStorageFlag { Disk, Memory };
opt<PCHStorageFlag> PCHStorage{
    "pch-storage",
    cat(Misc),
    desc("Storing PCHs in memory increases memory usages, but may "
         "improve performance"),
    values(
        clEnumValN(PCHStorageFlag::Disk, "disk", "store PCHs on disk"),
        clEnumValN(PCHStorageFlag::Memory, "memory", "store PCHs in memory")),
    init(PCHStorageFlag::Disk),
};

opt<bool> Sync{
    "sync",
    cat(Misc),
    desc("Handle client requests on main thread. Background index still uses "
         "its own thread."),
    init(false),
    Hidden,
};

opt<JSONStreamStyle> InputStyle{
    "input-style",
    cat(Protocol),
    desc("Input JSON stream encoding"),
    values(
        clEnumValN(JSONStreamStyle::Standard, "standard", "usual LSP protocol"),
        clEnumValN(JSONStreamStyle::Delimited, "delimited",
                   "messages delimited by --- lines, with # comment support")),
    init(JSONStreamStyle::Standard),
    Hidden,
};

opt<bool> EnableTestScheme{
    "enable-test-uri-scheme",
    cat(Protocol),
    desc("Enable 'test:' URI scheme. Only use in lit tests"),
    init(false),
    Hidden,
};

opt<std::string> PathMappingsArg{
    "path-mappings",
    cat(Protocol),
    desc(
        "Translates between client paths (as seen by a remote editor) and "
        "server paths (where clangd sees files on disk). "
        "Comma separated list of '<client_path>=<server_path>' pairs, the "
        "first entry matching a given path is used. "
        "e.g. /home/project/incl=/opt/include,/home/project=/workarea/project"),
    init(""),
};

opt<Path> InputMirrorFile{
    "input-mirror-file",
    cat(Protocol),
    desc("Mirror all LSP input to the specified file. Useful for debugging"),
    init(""),
    Hidden,
};

opt<Logger::Level> LogLevel{
    "log",
    cat(Protocol),
    desc("Verbosity of log messages written to stderr"),
    values(clEnumValN(Logger::Error, "error", "Error messages only"),
           clEnumValN(Logger::Info, "info", "High level execution tracing"),
           clEnumValN(Logger::Debug, "verbose", "Low level details")),
    init(Logger::Info),
};

opt<OffsetEncoding> ForceOffsetEncoding{
    "offset-encoding",
    cat(Protocol),
    desc("Force the offsetEncoding used for character positions. "
         "This bypasses negotiation via client capabilities"),
    values(
        clEnumValN(OffsetEncoding::UTF8, "utf-8", "Offsets are in UTF-8 bytes"),
        clEnumValN(OffsetEncoding::UTF16, "utf-16",
                   "Offsets are in UTF-16 code units"),
        clEnumValN(OffsetEncoding::UTF32, "utf-32",
                   "Offsets are in unicode codepoints")),
    init(OffsetEncoding::UnsupportedEncoding),
};

opt<bool> PrettyPrint{
    "pretty",
    cat(Protocol),
    desc("Pretty-print JSON output"),
    init(false),
};

/// Supports a test URI scheme with relaxed constraints for lit tests.
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

  llvm::InitializeAllTargetInfos();
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::sys::SetInterruptFunction(&requestShutdown);
  llvm::cl::SetVersionPrinter([](llvm::raw_ostream &OS) {
    OS << clang::getClangToolFullVersion("clangd") << "\n";
  });
  const char *FlagsEnvVar = "CLANGD_FLAGS";
  const char *Overview =
      R"(clangd is a language server that provides IDE-like features to editors.

It should be used via an editor plugin rather than invoked directly. For more information, see:
	https://clangd.llvm.org/
	https://microsoft.github.io/language-server-protocol/

clangd accepts flags on the commandline, and in the CLANGD_FLAGS environment variable.
)";
  llvm::cl::HideUnrelatedOptions(ClangdCategories);
  llvm::cl::ParseCommandLineOptions(argc, argv, Overview,
                                    /*Errs=*/nullptr, FlagsEnvVar);
  if (Test) {
    Sync = true;
    InputStyle = JSONStreamStyle::Delimited;
    LogLevel = Logger::Verbose;
    PrettyPrint = true;
    // Disable background index on lit tests by default to prevent disk writes.
    if (!EnableBackgroundIndex.getNumOccurrences())
      EnableBackgroundIndex = false;
    // Ensure background index makes progress.
    else if (EnableBackgroundIndex)
      BackgroundQueue::preventThreadStarvationInTests();
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
  llvm::Optional<llvm::raw_fd_ostream> TracerStream;
  std::unique_ptr<trace::EventTracer> Tracer;
  const char *JSONTraceFile = getenv("CLANGD_TRACE");
  const char *MetricsCSVFile = getenv("CLANGD_METRICS");
  const char *TracerFile = JSONTraceFile ? JSONTraceFile : MetricsCSVFile;
  if (TracerFile) {
    std::error_code EC;
    TracerStream.emplace(TracerFile, /*ref*/ EC,
                         llvm::sys::fs::FA_Read | llvm::sys::fs::FA_Write);
    if (EC) {
      TracerStream.reset();
      llvm::errs() << "Error while opening trace file " << TracerFile << ": "
                   << EC.message();
    } else {
      Tracer = (TracerFile == JSONTraceFile)
                   ? trace::createJSONTracer(*TracerStream, PrettyPrint)
                   : trace::createCSVMetricTracer(*TracerStream);
    }
  }

  llvm::Optional<trace::Session> TracingSession;
  if (Tracer)
    TracingSession.emplace(*Tracer);

  // If a user ran `clangd` in a terminal without redirecting anything,
  // it's somewhat likely they're confused about how to use clangd.
  // Show them the help overview, which explains.
  if (llvm::outs().is_displayed() && llvm::errs().is_displayed())
    llvm::errs() << Overview << "\n";
  // Use buffered stream to stderr (we still flush each log message). Unbuffered
  // stream can cause significant (non-deterministic) latency for the logger.
  llvm::errs().SetBuffered();
  StreamLogger Logger(llvm::errs(), LogLevel);
  LoggingSession LoggingSession(Logger);
  // Write some initial logs before we start doing any real work.
  log("{0}", clang::getClangToolFullVersion("clangd"));
  log("PID: {0}", llvm::sys::Process::getProcessId());
  {
    SmallString<128> CWD;
    if (auto Err = llvm::sys::fs::current_path(CWD))
      log("Working directory unknown: {0}", Err.message());
    else
      log("Working directory: {0}", CWD);
  }
  for (int I = 0; I < argc; ++I)
    log("argv[{0}]: {1}", I, argv[I]);
  if (auto EnvFlags = llvm::sys::Process::GetEnv(FlagsEnvVar))
    log("{0}: {1}", FlagsEnvVar, *EnvFlags);

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
        CompileCommandsDirPath = std::string(Path.str());
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
  std::unique_ptr<SymbolIndex> StaticIdx;
  std::future<void> AsyncIndexLoad; // Block exit while loading the index.
  if (EnableIndex && !IndexFile.empty()) {
    // Load the index asynchronously. Meanwhile SwapIndex returns no results.
    SwapIndex *Placeholder;
    StaticIdx.reset(Placeholder = new SwapIndex(std::make_unique<MemIndex>()));
    AsyncIndexLoad = runAsync<void>([Placeholder] {
      if (auto Idx = loadIndex(IndexFile, /*UseDex=*/true))
        Placeholder->reset(std::move(Idx));
    });
    if (Sync)
      AsyncIndexLoad.wait();
  }
  Opts.StaticIndex = StaticIdx.get();
  Opts.AsyncThreadsCount = WorkerThreadsCount;
  Opts.BuildRecoveryAST = RecoveryAST;
  Opts.PreserveRecoveryASTType = RecoveryASTType;

  clangd::CodeCompleteOptions CCOpts;
  CCOpts.IncludeIneligibleResults = IncludeIneligibleResults;
  CCOpts.Limit = LimitResults;
  if (CompletionStyle.getNumOccurrences())
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
    log("Starting LSP over XPC service");
    TransportLayer = newXPCTransport();
#else
    llvm::errs() << "This clangd binary wasn't built with XPC support.\n";
    return (int)ErrorResultCode::CantRunAsXPCService;
#endif
  } else {
    log("Starting LSP over stdin/stdout");
    TransportLayer = newJSONTransport(
        stdin, llvm::outs(),
        InputMirrorStream ? InputMirrorStream.getPointer() : nullptr,
        PrettyPrint, InputStyle);
  }
  if (!PathMappingsArg.empty()) {
    auto Mappings = parsePathMappings(PathMappingsArg);
    if (!Mappings) {
      elog("Invalid -path-mappings: {0}", Mappings.takeError());
      return 1;
    }
    TransportLayer = createPathMappingTransport(std::move(TransportLayer),
                                                std::move(*Mappings));
  }
  // Create an empty clang-tidy option.
  std::mutex ClangTidyOptMu;
  std::unique_ptr<tidy::ClangTidyOptionsProvider>
      ClangTidyOptProvider; /*GUARDED_BY(ClangTidyOptMu)*/
  if (EnableClangTidy) {
    auto EmptyDefaults = tidy::ClangTidyOptions::getDefaults();
    EmptyDefaults.Checks.reset(); // So we can tell if checks were ever set.
    tidy::ClangTidyOptions OverrideClangTidyOptions;
    if (!ClangTidyChecks.empty())
      OverrideClangTidyOptions.Checks = ClangTidyChecks;
    ClangTidyOptProvider = std::make_unique<tidy::FileOptionsProvider>(
        tidy::ClangTidyGlobalOptions(),
        /* Default */ EmptyDefaults,
        /* Override */ OverrideClangTidyOptions, FSProvider.getFileSystem());
    Opts.GetClangTidyOptions = [&](llvm::vfs::FileSystem &,
                                   llvm::StringRef File) {
      // This function must be thread-safe and tidy option providers are not.
      tidy::ClangTidyOptions Opts;
      {
        std::lock_guard<std::mutex> Lock(ClangTidyOptMu);
        // FIXME: use the FS provided to the function.
        Opts = ClangTidyOptProvider->getOptions(File);
      }
      if (!Opts.Checks) {
        // If the user hasn't configured clang-tidy checks at all, including
        // via .clang-tidy, give them a nice set of checks.
        // (This should be what the "default" options does, but it isn't...)
        //
        // These default checks are chosen for:
        //  - low false-positive rate
        //  - providing a lot of value
        //  - being reasonably efficient
        Opts.Checks = llvm::join_items(
            ",", "readability-misleading-indentation",
            "readability-deleted-default", "bugprone-integer-division",
            "bugprone-sizeof-expression", "bugprone-suspicious-missing-comma",
            "bugprone-unused-raii", "bugprone-unused-return-value",
            "misc-unused-using-decls", "misc-unused-alias-decls",
            "misc-definitions-in-headers");
      }
      return Opts;
    };
  }
  Opts.SuggestMissingIncludes = SuggestMissingIncludes;
  Opts.QueryDriverGlobs = std::move(QueryDriverGlobs);

  Opts.TweakFilter = [&](const Tweak &T) {
    if (T.hidden() && !HiddenFeatures)
      return false;
    if (TweakList.getNumOccurrences())
      return llvm::is_contained(TweakList, T.id());
    return true;
  };
  llvm::Optional<OffsetEncoding> OffsetEncodingFromFlag;
  if (ForceOffsetEncoding != OffsetEncoding::UnsupportedEncoding)
    OffsetEncodingFromFlag = ForceOffsetEncoding;

  clangd::RenameOptions RenameOpts;
  // Shall we allow to customize the file limit?
  RenameOpts.AllowCrossFile = CrossFileRename;

  ClangdLSPServer LSPServer(
      *TransportLayer, FSProvider, CCOpts, RenameOpts, CompileCommandsDirPath,
      /*UseDirBasedCDB=*/CompileArgsFrom == FilesystemCompileArgs,
      OffsetEncodingFromFlag, Opts);
  llvm::set_thread_name("clangd.main");
  int ExitCode = LSPServer.run()
                     ? 0
                     : static_cast<int>(ErrorResultCode::NoShutdownRequest);
  log("LSP finished, exiting with status {0}", ExitCode);

  // There may still be lingering background threads (e.g. slow requests
  // whose results will be dropped, background index shutting down).
  //
  // These should terminate quickly, and ~ClangdLSPServer blocks on them.
  // However if a bug causes them to run forever, we want to ensure the process
  // eventually exits. As clangd isn't directly user-facing, an editor can
  // "leak" clangd processes. Crashing in this case contains the damage.
  abortAfterTimeout(std::chrono::minutes(5));

  return ExitCode;
}
