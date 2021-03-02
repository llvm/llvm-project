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
#include "TidyProvider.h"
#include "Transport.h"
#include "index/Background.h"
#include "index/Index.h"
#include "index/Merge.h"
#include "index/ProjectAware.h"
#include "index/Serialization.h"
#include "index/remote/Client.h"
#include "refactor/Rename.h"
#include "support/Path.h"
#include "support/Shutdown.h"
#include "support/ThreadsafeFS.h"
#include "support/Trace.h"
#include "clang/Basic/Version.h"
#include "clang/Format/Format.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace clang {
namespace clangd {

// Implemented in Check.cpp.
bool check(const llvm::StringRef File, const ThreadsafeFS &TFS,
           const ClangdLSPServer::Options &Opts);

namespace {

using llvm::cl::cat;
using llvm::cl::CommaSeparated;
using llvm::cl::desc;
using llvm::cl::Hidden;
using llvm::cl::init;
using llvm::cl::list;
using llvm::cl::opt;
using llvm::cl::OptionCategory;
using llvm::cl::ValueOptional;
using llvm::cl::values;

// All flags must be placed in a category, or they will be shown neither in
// --help, nor --help-hidden!
OptionCategory CompileCommands("clangd compilation flags options");
OptionCategory Features("clangd feature options");
OptionCategory Misc("clangd miscellaneous options");
OptionCategory Protocol("clangd protocol and logging options");
OptionCategory Retired("clangd flags no longer in use");
const OptionCategory *ClangdCategories[] = {&Features, &Protocol,
                                            &CompileCommands, &Misc, &Retired};

template <typename T> class RetiredFlag {
  opt<T> Option;

public:
  RetiredFlag(llvm::StringRef Name)
      : Option(Name, cat(Retired), desc("Obsolete flag, ignored"), Hidden,
               llvm::cl::callback([Name](const T &) {
                 llvm::errs()
                     << "The flag `-" << Name << "` is obsolete and ignored.\n";
               })) {}
};

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

opt<CodeCompleteOptions::CodeCompletionRankingModel> RankingModel{
    "ranking-model",
    cat(Features),
    desc("Model to use to rank code-completion items"),
    values(clEnumValN(CodeCompleteOptions::Heuristics, "heuristics",
                      "Use hueristics to rank code completion items"),
           clEnumValN(CodeCompleteOptions::DecisionForest, "decision_forest",
                      "Use Decision Forest model to rank completion items")),
    init(CodeCompleteOptions().RankingModel),
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

RetiredFlag<bool> EnableIndex("index");
RetiredFlag<bool> SuggestMissingIncludes("suggest-missing-includes");
RetiredFlag<bool> RecoveryAST("recovery-ast");
RetiredFlag<bool> RecoveryASTType("recovery-ast-type");
RetiredFlag<bool> AsyncPreamble("async-preamble");
RetiredFlag<bool> CollectMainFileRefs("collect-main-file-refs");
RetiredFlag<bool> CrossFileRename("cross-file-rename");
RetiredFlag<std::string> ClangTidyChecks("clang-tidy-checks");

opt<int> LimitResults{
    "limit-results",
    cat(Features),
    desc("Limit the number of results returned by clangd. "
         "0 means no limit (default=100)"),
    init(100),
};

list<std::string> TweakList{
    "tweaks",
    cat(Features),
    desc("Specify a list of Tweaks to enable (only for clangd developers)."),
    Hidden,
    CommaSeparated,
};

opt<bool> FoldingRanges{
    "folding-ranges",
    cat(Features),
    desc("Enable preview of FoldingRanges feature"),
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
         "-enable-test-scheme -enable-config=0 -log=verbose. "
         "Intended to simplify lit tests"),
    init(false),
    Hidden,
};

opt<Path> CheckFile{
    "check",
    cat(Misc),
    desc("Parse one file in isolation instead of acting as a language server. "
         "Useful to investigate/reproduce crashes or configuration problems. "
         "With --check=<filename>, attempts to parse a particular file."),
    init(""),
    ValueOptional,
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

opt<bool> EnableConfig{
    "enable-config",
    cat(Misc),
    desc(
        "Read user and project configuration from YAML files.\n"
        "Project config is from a .clangd file in the project directory.\n"
        "User config is from clangd/config.yaml in the following directories:\n"
        "\tWindows: %USERPROFILE%\\AppData\\Local\n"
        "\tMac OS: ~/Library/Preferences/\n"
        "\tOthers: $XDG_CONFIG_HOME, usually ~/.config\n"
        "Configuration is documented at https://clangd.llvm.org/config.html"),
    init(true),
};

#if defined(__GLIBC__) && CLANGD_MALLOC_TRIM
opt<bool> EnableMallocTrim{
    "malloc-trim",
    cat(Misc),
    desc("Release memory periodically via malloc_trim(3)."),
    init(true),
};

std::function<void()> getMemoryCleanupFunction() {
  if (!EnableMallocTrim)
    return nullptr;
  // Leave a few MB at the top of the heap: it is insignificant
  // and will most likely be needed by the main thread
  constexpr size_t MallocTrimPad = 20'000'000;
  return []() {
    if (malloc_trim(MallocTrimPad))
      vlog("Released memory via malloc_trim");
  };
}
#else
std::function<void()> getMemoryCleanupFunction() { return nullptr; }
#endif

#if CLANGD_ENABLE_REMOTE
opt<std::string> RemoteIndexAddress{
    "remote-index-address",
    cat(Features),
    desc("Address of the remote index server"),
};

// FIXME(kirillbobyrev): Should this be the location of compile_commands.json?
opt<std::string> ProjectRoot{
    "project-root",
    cat(Features),
    desc("Path to the project root. Requires remote-index-address to be set."),
};
#endif

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
      return error(
          "Expect URI body to be an absolute path starting with '/': {0}",
          Body);
    Body = Body.ltrim('/');
    llvm::SmallString<16> Path(Body);
    path::native(Path);
    fs::make_absolute(TestScheme::TestDir, Path);
    return std::string(Path);
  }

  llvm::Expected<URI>
  uriFromAbsolutePath(llvm::StringRef AbsolutePath) const override {
    llvm::StringRef Body = AbsolutePath;
    if (!Body.consume_front(TestScheme::TestDir))
      return error("Path {0} doesn't start with root {1}", AbsolutePath,
                   TestDir);

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

std::unique_ptr<SymbolIndex>
loadExternalIndex(const Config::ExternalIndexSpec &External,
                  AsyncTaskRunner &Tasks) {
  switch (External.Kind) {
  case Config::ExternalIndexSpec::Server:
    log("Associating {0} with remote index at {1}.", External.MountPoint,
        External.Location);
    return remote::getClient(External.Location, External.MountPoint);
  case Config::ExternalIndexSpec::File:
    log("Associating {0} with monolithic index at {1}.", External.MountPoint,
        External.Location);
    auto NewIndex = std::make_unique<SwapIndex>(std::make_unique<MemIndex>());
    Tasks.runAsync("Load-index:" + External.Location,
                   [File = External.Location, PlaceHolder = NewIndex.get()] {
                     if (auto Idx = loadIndex(File, /*UseDex=*/true))
                       PlaceHolder->reset(std::move(Idx));
                   });
    return std::move(NewIndex);
  }
  llvm_unreachable("Invalid ExternalIndexKind.");
}
} // namespace
} // namespace clangd
} // namespace clang

enum class ErrorResultCode : int {
  NoShutdownRequest = 1,
  CantRunAsXPCService = 2,
  CheckFailed = 3
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
    // Disable config system by default to avoid external reads.
    if (!EnableConfig.getNumOccurrences())
      EnableConfig = false;
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
  if (llvm::outs().is_displayed() && llvm::errs().is_displayed() &&
      !CheckFile.getNumOccurrences())
    llvm::errs() << Overview << "\n";
  // Use buffered stream to stderr (we still flush each log message). Unbuffered
  // stream can cause significant (non-deterministic) latency for the logger.
  llvm::errs().SetBuffered();
  // Don't flush stdout when logging, this would be both slow and racy!
  llvm::errs().tie(nullptr);
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

  ClangdLSPServer::Options Opts;
  Opts.UseDirBasedCDB = (CompileArgsFrom == FilesystemCompileArgs);

  // If --compile-commands-dir arg was invoked, check value and override default
  // path.
  if (!CompileCommandsDir.empty()) {
    if (llvm::sys::fs::exists(CompileCommandsDir)) {
      // We support passing both relative and absolute paths to the
      // --compile-commands-dir argument, but we assume the path is absolute in
      // the rest of clangd so we make sure the path is absolute before
      // continuing.
      llvm::SmallString<128> Path(CompileCommandsDir);
      if (std::error_code EC = llvm::sys::fs::make_absolute(Path)) {
        elog("Error while converting the relative path specified by "
             "--compile-commands-dir to an absolute path: {0}. The argument "
             "will be ignored.",
             EC.message());
      } else {
        Opts.CompileCommandsDir = std::string(Path.str());
      }
    } else {
      elog("Path specified by --compile-commands-dir does not exist. The "
           "argument will be ignored.");
    }
  }

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
  Opts.BuildDynamicSymbolIndex = true;
  std::vector<std::unique_ptr<SymbolIndex>> IdxStack;
  std::unique_ptr<SymbolIndex> StaticIdx;
  std::future<void> AsyncIndexLoad; // Block exit while loading the index.
  if (!IndexFile.empty()) {
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
#if CLANGD_ENABLE_REMOTE
  if (RemoteIndexAddress.empty() != ProjectRoot.empty()) {
    llvm::errs() << "remote-index-address and project-path have to be "
                    "specified at the same time.";
    return 1;
  }
  if (!RemoteIndexAddress.empty()) {
    if (IndexFile.empty()) {
      log("Connecting to remote index at {0}", RemoteIndexAddress);
      StaticIdx = remote::getClient(RemoteIndexAddress, ProjectRoot);
      EnableBackgroundIndex = false;
    } else {
      elog("When enabling remote index, IndexFile should not be specified. "
           "Only one can be used at time. Remote index will ignored.");
    }
  }
#endif
  Opts.BackgroundIndex = EnableBackgroundIndex;
  auto PAI = createProjectAwareIndex(loadExternalIndex);
  if (StaticIdx) {
    IdxStack.emplace_back(std::move(StaticIdx));
    IdxStack.emplace_back(
        std::make_unique<MergedIndex>(PAI.get(), IdxStack.back().get()));
    Opts.StaticIndex = IdxStack.back().get();
  } else {
    Opts.StaticIndex = PAI.get();
  }
  Opts.AsyncThreadsCount = WorkerThreadsCount;
  Opts.FoldingRanges = FoldingRanges;
  Opts.MemoryCleanup = getMemoryCleanupFunction();

  Opts.CodeComplete.IncludeIneligibleResults = IncludeIneligibleResults;
  Opts.CodeComplete.Limit = LimitResults;
  if (CompletionStyle.getNumOccurrences())
    Opts.CodeComplete.BundleOverloads = CompletionStyle != Detailed;
  Opts.CodeComplete.ShowOrigins = ShowOrigins;
  Opts.CodeComplete.InsertIncludes = HeaderInsertion;
  if (!HeaderInsertionDecorators) {
    Opts.CodeComplete.IncludeIndicator.Insert.clear();
    Opts.CodeComplete.IncludeIndicator.NoInsert.clear();
  }
  Opts.CodeComplete.EnableFunctionArgSnippets = EnableFunctionArgSnippets;
  Opts.CodeComplete.AllScopes = AllScopesCompletion;
  Opts.CodeComplete.RunParser = CodeCompletionParse;
  Opts.CodeComplete.RankingModel = RankingModel;

  RealThreadsafeFS TFS;
  std::vector<std::unique_ptr<config::Provider>> ProviderStack;
  std::unique_ptr<config::Provider> Config;
  if (EnableConfig) {
    ProviderStack.push_back(
        config::Provider::fromAncestorRelativeYAMLFiles(".clangd", TFS));
    llvm::SmallString<256> UserConfig;
    if (llvm::sys::path::user_config_directory(UserConfig)) {
      llvm::sys::path::append(UserConfig, "clangd", "config.yaml");
      vlog("User config file is {0}", UserConfig);
      ProviderStack.push_back(
          config::Provider::fromYAMLFile(UserConfig, /*Directory=*/"", TFS));
    } else {
      elog("Couldn't determine user config file, not loading");
    }
    std::vector<const config::Provider *> ProviderPointers;
    for (const auto &P : ProviderStack)
      ProviderPointers.push_back(P.get());
    Config = config::Provider::combine(std::move(ProviderPointers));
    Opts.ConfigProvider = Config.get();
  }

  // Create an empty clang-tidy option.
  TidyProvider ClangTidyOptProvider;
  if (EnableClangTidy) {
    std::vector<TidyProvider> Providers;
    Providers.reserve(4 + EnableConfig);
    Providers.push_back(provideEnvironment());
    Providers.push_back(provideClangTidyFiles(TFS));
    if (EnableConfig)
      Providers.push_back(provideClangdConfig());
    Providers.push_back(provideDefaultChecks());
    Providers.push_back(disableUnusableChecks());
    ClangTidyOptProvider = combine(std::move(Providers));
    Opts.ClangTidyProvider = ClangTidyOptProvider;
  }
  Opts.QueryDriverGlobs = std::move(QueryDriverGlobs);
  Opts.TweakFilter = [&](const Tweak &T) {
    if (T.hidden() && !HiddenFeatures)
      return false;
    if (TweakList.getNumOccurrences())
      return llvm::is_contained(TweakList, T.id());
    return true;
  };
  if (ForceOffsetEncoding != OffsetEncoding::UnsupportedEncoding)
    Opts.Encoding = ForceOffsetEncoding;

  if (CheckFile.getNumOccurrences()) {
    llvm::SmallString<256> Path;
    llvm::sys::fs::real_path(CheckFile, Path, /*expand_tilde=*/true);
    log("Entering check mode (no LSP server)");
    return check(Path, TFS, Opts)
               ? 0
               : static_cast<int>(ErrorResultCode::CheckFailed);
  }

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
    return static_cast<int>(ErrorResultCode::CantRunAsXPCService);
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

  ClangdLSPServer LSPServer(*TransportLayer, TFS, Opts);
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
