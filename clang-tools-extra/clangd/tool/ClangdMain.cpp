//===--- ClangdMain.cpp - clangd server loop ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangdMain.h"

#include "Check.h"
#include "ClangdLSPServer.h"
#include "CodeComplete.h"
#include "Compiler.h"
#include "Config.h"
#include "ConfigProvider.h"
#include "Feature.h"
#include "Opts.inc"
#include "PathMapping.h"
#include "Protocol.h"
#include "TidyProvider.h"
#include "Transport.h"
#include "index/Background.h"
#include "index/Index.h"
#include "index/MemIndex.h"
#include "index/ProjectAware.h"
#include "index/remote/Client.h"
#include "support/Path.h"
#include "support/Shutdown.h"
#include "support/ThreadCrashReporter.h"
#include "support/ThreadsafeFS.h"
#include "support/Trace.h"
#include "clang/Basic/Stack.h"
#include "clang/Format/Format.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace {

#if defined(__GLIBC__) && CLANGD_MALLOC_TRIM
static constexpr unsigned MallocTrimVis = (1 << 8);
#else
static constexpr unsigned MallocTrimVis = 0;
#endif

#if CLANGD_ENABLE_REMOTE
// FIXME(kirillbobyrev): Should this be the location of compile_commands.json?
static constexpr unsigned RemoteVis = (1 << 9);
#else
static constexpr unsigned RemoteVis = 0;
#endif

using namespace llvm;
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE)                                                    \
  static constexpr StringLiteral NAME##_init[] = VALUE;                        \
  static constexpr ArrayRef<StringLiteral> NAME(NAME##_init,                   \
                                                std::size(NAME##_init) - 1);
#include "Opts.inc"
#undef PREFIX

using namespace llvm::opt;
static constexpr opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

class ClangdOptTable : public llvm::opt::GenericOptTable {
public:
  ClangdOptTable() : GenericOptTable(InfoTable) {
    setGroupedShortOptions(true);
  }
};

enum CompileArgsFrom { LSPCompileArgs, FilesystemCompileArgs };

// FIXME: also support "plain" style where signatures are always omitted.
enum CompletionStyleFlag { Detailed, Bundled, Invalid };
enum PCHStorageFlag { Disk, Memory };

} // namespace

namespace clang {
namespace clangd {

static void parseValueError(const StringRef ArgName, const StringRef Value) {
  llvm::errs() << "for the " << ArgName << " option: Cannot find option named "
               << Value;
  exit(EXIT_FAILURE);
}

template <typename T>
static void parseIntArg(const opt::InputArgList &Args, int ID, T &Value,
                        T Default) {
  if (const opt::Arg *A = Args.getLastArg(ID)) {
    StringRef V(A->getValue());
    if (!llvm::to_integer(V, Value, 0)) {
      errs() << A->getSpelling() + ": expected an integer, but got '" + V + "'";
      exit(1);
    }
  } else {
    Value = Default;
  }
}

static PCHStorageFlag parsePCHStorage(const opt::InputArgList &Args) {
  if (Args.hasArg(OPT_pch_storage_EQ)) {
    StringRef PCHStorageStr = Args.getLastArgValue(OPT_pch_storage_EQ);
    if (PCHStorageStr.equals("disk"))
      return PCHStorageFlag::Disk;
    if (PCHStorageStr.equals("memory"))
      return PCHStorageFlag::Memory;

    parseValueError(Args.getArgString(OPT_pch_storage_EQ), PCHStorageStr);
  }

  return PCHStorageFlag::Disk;
}

static JSONStreamStyle parseInputStyle(const opt::InputArgList &Args) {
  if (Args.hasArg(OPT_input_style_EQ)) {
    StringRef InputStyleStr = Args.getLastArgValue(OPT_input_style_EQ);
    if (InputStyleStr.equals("standard"))
      return JSONStreamStyle::Standard;
    if (InputStyleStr.equals("delimited"))
      return JSONStreamStyle::Delimited;

    parseValueError(Args.getArgString(OPT_input_style_EQ), InputStyleStr);
  }

  return JSONStreamStyle::Standard;
}

static Logger::Level parseLogLevel(const opt::InputArgList &Args) {
  if (Args.hasArg(OPT_log_EQ)) {
    StringRef LogLevelStr = Args.getLastArgValue(OPT_log_EQ);
    if (LogLevelStr.equals("error"))
      return Logger::Level::Error;
    if (LogLevelStr.equals("info"))
      return Logger::Level::Info;
    if (LogLevelStr.equals("verbose"))
      return Logger::Level::Verbose;

    parseValueError(Args.getArgString(OPT_log_EQ), LogLevelStr);
  }

  return Logger::Level::Info;
}

static CodeCompleteOptions::CodeCompletionRankingModel
parseRankingModel(const opt::InputArgList &Args) {
  if (Args.hasArg(OPT_ranking_model_EQ)) {
    StringRef RankingModelStr = Args.getLastArgValue(OPT_ranking_model_EQ);
    if (RankingModelStr.equals("heuristics"))
      return CodeCompleteOptions::Heuristics;
    if (RankingModelStr.equals("decision_forest"))
      return CodeCompleteOptions::DecisionForest;

    parseValueError(Args.getArgString(OPT_ranking_model_EQ), RankingModelStr);
  }

  return CodeCompleteOptions().RankingModel;
}

static CompileArgsFrom parseCompileArgsFrom(const opt::InputArgList &Args) {
  if (Args.hasArg(OPT_compile_args_from_EQ)) {
    StringRef CompileArgsFromStr =
        Args.getLastArgValue(OPT_compile_args_from_EQ);
    if (CompileArgsFromStr.equals("lsp"))
      return CompileArgsFrom::LSPCompileArgs;
    if (CompileArgsFromStr.equals("filesystem"))
      return CompileArgsFrom::FilesystemCompileArgs;

    parseValueError(Args.getArgString(OPT_compile_args_from_EQ),
                    CompileArgsFromStr);
  }

  return CompileArgsFrom::FilesystemCompileArgs;
}

static llvm::ThreadPriority
parseBackgroundIndexPriority(const opt::InputArgList &Args) {
  if (Args.hasArg(OPT_background_index_priority_EQ)) {
    StringRef BackgroundIndexPriorityStr =
        Args.getLastArgValue(OPT_background_index_priority_EQ);
    if (BackgroundIndexPriorityStr.equals("background"))
      return llvm::ThreadPriority::Background;
    if (BackgroundIndexPriorityStr.equals("normal"))
      return llvm::ThreadPriority::Default;
    if (BackgroundIndexPriorityStr.equals("low"))
      return llvm::ThreadPriority::Low;

    parseValueError(Args.getArgString(OPT_background_index_priority_EQ),
                    BackgroundIndexPriorityStr);
  }

  return llvm::ThreadPriority::Low;
}

static CompletionStyleFlag parseCompletionStyle(const opt::InputArgList &Args) {
  if (Args.hasArg(OPT_completion_style_EQ)) {
    StringRef CompletionStyleStr =
        Args.getLastArgValue(OPT_completion_style_EQ);
    if (CompletionStyleStr.equals("detailed"))
      return CompletionStyleFlag::Detailed;
    if (CompletionStyleStr.equals("bundled"))
      return CompletionStyleFlag::Bundled;

    parseValueError(Args.getArgString(OPT_completion_style_EQ),
                    CompletionStyleStr);
  }

  return CompletionStyleFlag::Invalid;
}

static CodeCompleteOptions::IncludeInsertion
parseHeaderInsertion(const opt::InputArgList &Args) {
  if (Args.hasArg(OPT_header_insertion_EQ)) {
    StringRef HeaderInsertionStr =
        Args.getLastArgValue(OPT_header_insertion_EQ);
    if (HeaderInsertionStr.equals("iwyu"))
      return CodeCompleteOptions::IWYU;
    if (HeaderInsertionStr.equals("never"))
      return CodeCompleteOptions::NeverInsert;

    parseValueError(Args.getArgString(OPT_header_insertion_EQ),
                    HeaderInsertionStr);
  }

  return CodeCompleteOptions().InsertIncludes;
}

static CodeCompleteOptions::CodeCompletionParse
parseCodeCompletionParse(const opt::InputArgList &Args) {
  if (Args.hasArg(OPT_completion_parse_EQ)) {
    StringRef CompletionParseStr =
        Args.getLastArgValue(OPT_completion_parse_EQ);
    if (CompletionParseStr.equals("always"))
      return CodeCompleteOptions::AlwaysParse;
    if (CompletionParseStr.equals("auto"))
      return CodeCompleteOptions::ParseIfReady;
    if (CompletionParseStr.equals("never"))
      return CodeCompleteOptions::NeverParse;

    parseValueError(Args.getArgString(OPT_completion_parse_EQ),
                    CompletionParseStr);
  }

  return CodeCompleteOptions().RunParser;
}

namespace {

#if defined(__GLIBC__) && CLANGD_MALLOC_TRIM
std::function<void()> getMemoryCleanupFunction(bool EnableMallocTrim) {
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
std::function<void()> getMemoryCleanupFunction(bool EnableMallocTrim) {
  return nullptr
}
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
    if (!Body.starts_with("/"))
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
                  AsyncTaskRunner *Tasks) {
  static const trace::Metric RemoteIndexUsed("used_remote_index",
                                             trace::Metric::Value, "address");
  switch (External.Kind) {
  case Config::ExternalIndexSpec::None:
    break;
  case Config::ExternalIndexSpec::Server:
    RemoteIndexUsed.record(1, External.Location);
    log("Associating {0} with remote index at {1}.", External.MountPoint,
        External.Location);
    return remote::getClient(External.Location, External.MountPoint);
  case Config::ExternalIndexSpec::File:
    log("Associating {0} with monolithic index at {1}.", External.MountPoint,
        External.Location);
    auto NewIndex = std::make_unique<SwapIndex>(std::make_unique<MemIndex>());
    auto IndexLoadTask = [File = External.Location,
                          PlaceHolder = NewIndex.get()] {
      if (auto Idx = loadIndex(File, SymbolOrigin::Static, /*UseDex=*/true))
        PlaceHolder->reset(std::move(Idx));
    };
    if (Tasks) {
      Tasks->runAsync("Load-index:" + External.Location,
                      std::move(IndexLoadTask));
    } else {
      IndexLoadTask();
    }
    return std::move(NewIndex);
  }
  llvm_unreachable("Invalid ExternalIndexKind.");
}

struct FlagsConfigProviderOpts {
  Path CompileCommandsDir;
  std::string IndexFile;
  bool EnableBackgroundIndex;
  std::optional<bool> AllScopesCompletion;
  bool LitTest;
};

class FlagsConfigProvider : public config::Provider {
private:
  config::CompiledFragment Frag;

  std::vector<config::CompiledFragment>
  getFragments(const config::Params &,
               config::DiagnosticCallback) const override {
    return {Frag};
  }

public:
  FlagsConfigProvider(FlagsConfigProviderOpts &Opts) {
    std::optional<Config::CDBSearchSpec> CDBSearch;
    std::optional<Config::ExternalIndexSpec> IndexSpec;
    std::optional<Config::BackgroundPolicy> BGPolicy;

    // If --compile-commands-dir arg was invoked, check value and override
    // default path.
    if (!Opts.CompileCommandsDir.empty()) {
      elog("I must be printed elog");
      log("I am being printed log.");
      if (llvm::sys::fs::exists(Opts.CompileCommandsDir)) {
        // We support passing both relative and absolute paths to the
        // --compile-commands-dir argument, but we assume the path is absolute
        // in the rest of clangd so we make sure the path is absolute before
        // continuing.
        llvm::SmallString<128> Path(Opts.CompileCommandsDir);
        if (std::error_code EC = llvm::sys::fs::make_absolute(Path)) {
          elog("Error while converting the relative path specified by "
               "--compile-commands-dir to an absolute path: {0}. The argument "
               "will be ignored.",
               EC.message());
        } else {
          CDBSearch = {Config::CDBSearchSpec::FixedDir, Path.str().str()};
        }
      } else {
        llvm::errs() << "Path does not exists 'compilecommandsdir'\n";
        elog("Path specified by --compile-commands-dir does not exist. The "
             "argument will be ignored.");
      }
    }
    if (!Opts.IndexFile.empty()) {
      Config::ExternalIndexSpec Spec;
      Spec.Kind = Spec.File;
      Spec.Location = Opts.IndexFile;
      IndexSpec = std::move(Spec);
    }
#if CLANGD_ENABLE_REMOTE
    if (!RemoteIndexAddress.empty()) {
      assert(!ProjectRoot.empty() && IndexFile.empty());
      Config::ExternalIndexSpec Spec;
      Spec.Kind = Spec.Server;
      Spec.Location = RemoteIndexAddress;
      Spec.MountPoint = ProjectRoot;
      IndexSpec = std::move(Spec);
      BGPolicy = Config::BackgroundPolicy::Skip;
    }
#endif
    if (!Opts.EnableBackgroundIndex) {
      BGPolicy = Config::BackgroundPolicy::Skip;
    }

    Frag = [=](const config::Params &, Config &C) {
      if (CDBSearch)
        C.CompileFlags.CDBSearch = *CDBSearch;
      if (IndexSpec)
        C.Index.External = *IndexSpec;
      if (BGPolicy)
        C.Index.Background = *BGPolicy;
      if (Opts.AllScopesCompletion)
        C.Completion.AllScopes = Opts.AllScopesCompletion.value();

      if (Opts.LitTest)
        C.Index.StandardLibrary = false;
      return true;
    };
  }
};
} // namespace

enum class ErrorResultCode : int {
  NoShutdownRequest = 1,
  CantRunAsXPCService = 2,
  CheckFailed = 3
};

int clangdMain(int argc, char *argv[]) {
  // Clang could run on the main thread. e.g., when the flag '-check' or '-sync'
  // is enabled.
  clang::noteBottomOfStack();
  llvm::InitLLVM X(argc, argv);
  llvm::InitializeAllTargetInfos();
  llvm::sys::AddSignalHandler(
      [](void *) {
        ThreadCrashReporter::runCrashHandlers();
        // Ensure ThreadCrashReporter and PrintStackTrace output is visible.
        llvm::errs().flush();
      },
      nullptr);
  llvm::sys::SetInterruptFunction(&requestShutdown);
  llvm::cl::SetVersionPrinter([](llvm::raw_ostream &OS) {
    OS << versionString() << "\n"
       << "Features: " << featureString() << "\n"
       << "Platform: " << platformString() << "\n";
  });
  const char *FlagsEnvVar = "CLANGD_FLAGS";
  const char *Overview =
      R"(clangd is a language server that provides IDE-like features to editors.

It should be used via an editor plugin rather than invoked directly. For more information, see:
	https://clangd.llvm.org/
	https://microsoft.github.io/language-server-protocol/

clangd accepts flags on the commandline, and in the CLANGD_FLAGS environment variable.
)";

  const StringRef ToolName = argv[0];
  BumpPtrAllocator A;
  StringSaver Saver(A);
  ClangdOptTable Tbl;
  Tbl.setInitialOptionsFromEnvironment(FlagsEnvVar);
  opt::InputArgList Args =
      Tbl.parseArgs(argc, argv, OPT_UNKNOWN, Saver, [&](StringRef Msg) {
        llvm::errs() << Msg << '\n';
        std::exit(1);
      });

  if (Args.hasArg(OPT_help) || Args.hasArg(OPT_help_hidden)) {
    Tbl.printHelp(
        llvm::outs(), (ToolName + " [options]").str().c_str(), Overview,
        Args.hasArg(OPT_help_hidden), false,
        llvm::opt::Visibility(MallocTrimVis | RemoteVis | DefaultVis));
    std::exit(0);
  }

  if (Args.hasArg(OPT_version)) {
    outs() << ToolName << '\n';
    cl::PrintVersionMessage();
    exit(0);
  }

  bool Sync = Args.hasFlag(OPT_sync, OPT_no_sync, Args.hasArg(OPT_lit_test));
  bool CrashPragmas = Args.hasArg(OPT_crash_pragmas);

  JSONStreamStyle InputStyle = parseInputStyle(Args);
  Logger::Level LogLevel = parseLogLevel(Args);

  bool PrettyPrint = Args.hasArg(OPT_pretty);
  bool EnableConfig =
      Args.hasFlag(OPT_enable_config, OPT_no_enable_config, true);

  bool EnableBackgroundIndex =
      Args.hasFlag(OPT_background_index, OPT_no_background_index, true);

  if (Args.hasArg(OPT_lit_test)) {
    CrashPragmas = true;
    InputStyle = JSONStreamStyle::Delimited;
    LogLevel = Logger::Verbose;
    PrettyPrint = true;
    // Disable config system by default to avoid external reads.
    if (!Args.hasArg(OPT_enable_config))
      EnableConfig = false;
    // Disable background index on lit tests by default to prevent disk writes.
    if (!Args.hasArg(OPT_background_index))
      EnableBackgroundIndex = false;
    // Ensure background index makes progress.
    else if (EnableBackgroundIndex)
      BackgroundQueue::preventThreadStarvationInTests();
  }
  if (Args.hasArg(OPT_lit_test) || Args.hasArg(OPT_enable_test_uri_scheme)) {
    static URISchemeRegistry::Add<TestScheme> X(
        "test", "Test scheme for clangd lit tests.");
  }
  if (CrashPragmas)
    allowCrashPragmasForTest();

  unsigned int WorkerThreadsCount;
  parseIntArg(Args, OPT_j_EQ, WorkerThreadsCount,
              getDefaultAsyncThreadsCount());

  if (!Sync && WorkerThreadsCount == 0) {
    llvm::errs() << "A number of worker threads cannot be 0. Did you mean to "
                    "specify -sync?";
    return 1;
  }

  if (Sync) {
    if (Args.hasArg(OPT_j_EQ))
      llvm::errs() << "Ignoring -j because -sync is set.\n";
    WorkerThreadsCount = 0;
  }

  if (Args.hasArg(OPT_fallback_style_EQ))
    clang::format::DefaultFallbackStyle =
        Args.getLastArgValue(OPT_fallback_style_EQ).str().c_str();

  // Validate command line arguments.
  std::optional<llvm::raw_fd_ostream> InputMirrorStream;
  const std::string InputMirrorFile(
      Args.getLastArgValue(OPT_input_mirror_file_EQ, ""));
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

  CodeCompleteOptions::CodeCompletionRankingModel RankingModel =
      parseRankingModel(Args);
#if !CLANGD_DECISION_FOREST
  if (RankingModel == clangd::CodeCompleteOptions::DecisionForest) {
    llvm::errs() << "Clangd was compiled without decision forest support.\n";
    return 1;
  }
#endif

  // Setup tracing facilities if CLANGD_TRACE is set. In practice enabling a
  // trace flag in your editor's config is annoying, launching with
  // `CLANGD_TRACE=trace.json vim` is easier.
  std::optional<llvm::raw_fd_ostream> TracerStream;
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

  std::optional<trace::Session> TracingSession;
  if (Tracer)
    TracingSession.emplace(*Tracer);

  // If a user ran `clangd` in a terminal without redirecting anything,
  // it's somewhat likely they're confused about how to use clangd.
  // Show them the help overview, which explains.
  if (llvm::outs().is_displayed() && llvm::errs().is_displayed() &&
      !Args.hasArg(OPT_check_EQ))
    llvm::errs() << Overview << "\n";
  // Use buffered stream to stderr (we still flush each log message). Unbuffered
  // stream can cause significant (non-deterministic) latency for the logger.
  llvm::errs().SetBuffered();
  // Don't flush stdout when logging, this would be both slow and racy!
  llvm::errs().tie(nullptr);
  StreamLogger Logger(llvm::errs(), LogLevel);
  LoggingSession LoggingSession(Logger);
  // Write some initial logs before we start doing any real work.
  log("{0}", versionString());
  log("Features: {0}", featureString());
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

  CompileArgsFrom CompileArgsFrom = parseCompileArgsFrom(Args);
  Opts.UseDirBasedCDB = (CompileArgsFrom == FilesystemCompileArgs);

  const PCHStorageFlag PCHStorage = parsePCHStorage(Args);
  switch (PCHStorage) {
  case PCHStorageFlag::Memory:
    Opts.StorePreamblesInMemory = true;
    break;
  case PCHStorageFlag::Disk:
    Opts.StorePreamblesInMemory = false;
    break;
  }

  StringRef ResourceDir = Args.getLastArgValue(OPT_resource_dir_EQ);
  if (!ResourceDir.empty())
    Opts.ResourceDir = ResourceDir;

  Opts.BuildDynamicSymbolIndex = true;
  std::vector<std::unique_ptr<SymbolIndex>> IdxStack;

  const StringRef IndexFile = Args.getLastArgValue(OPT_index_file_EQ, "");

#if !CLANGD_ENABLE_REMOTE
  const StringRef RemoteIndexAddress =
      Args.getLastArgValue(OPT_remote_index_address_EQ, "");
  const StringRef ProjectRoot =
      Args.getLastArgValue(OPT_remote_index_address_EQ, "");
  if (RemoteIndexAddress.empty() != ProjectRoot.empty()) {
    llvm::errs() << "remote-index-address and project-path have to be "
                    "specified at the same time.";
    return 1;
  }
  if (!RemoteIndexAddress.empty()) {
    if (IndexFile.empty()) {
      log("Connecting to remote index at {0}", RemoteIndexAddress);
    } else {
      elog("When enabling remote index, IndexFile should not be specified. "
           "Only one can be used at time. Remote index will ignored.");
    }
  }
#endif
  Opts.BackgroundIndex = EnableBackgroundIndex;
  Opts.BackgroundIndexPriority = parseBackgroundIndexPriority(Args);
  parseIntArg(Args, OPT_limit_references_EQ, Opts.ReferencesLimit,
              size_t(1000));
  parseIntArg(Args, OPT_rename_file_limit_EQ, Opts.Rename.LimitFiles,
              size_t(50));
  auto PAI = createProjectAwareIndex(loadExternalIndex, Sync);
  Opts.StaticIndex = PAI.get();
  Opts.AsyncThreadsCount = WorkerThreadsCount;
  Opts.MemoryCleanup = getMemoryCleanupFunction(
      Args.hasFlag(OPT_malloc_trim, OPT_no_malloc_trim, true));

  Opts.CodeComplete.IncludeIneligibleResults =
      Args.hasArg(OPT_include_ineligible_results);

  parseIntArg(Args, OPT_limit_results_EQ, Opts.CodeComplete.Limit, size_t(100));
  if (Args.hasArg(OPT_completion_style_EQ)) {
    CompletionStyleFlag CompletionStyle = parseCompletionStyle(Args);
    Opts.CodeComplete.BundleOverloads = CompletionStyle != Detailed;
  }
  Opts.CodeComplete.ShowOrigins = Args.hasArg(OPT_debug_origin);
  Opts.CodeComplete.InsertIncludes = parseHeaderInsertion(Args);
  Opts.CodeComplete.ImportInsertions =
      Args.hasFlag(OPT_import_insertions, OPT_no_import_insertions, true);

  if (!Args.hasFlag(OPT_header_insertion_decorators,
                    OPT_no_header_insertion_decorators, true)) {
    Opts.CodeComplete.IncludeIndicator.Insert.clear();
    Opts.CodeComplete.IncludeIndicator.NoInsert.clear();
  }
  Opts.CodeComplete.EnableFunctionArgSnippets = Args.hasFlag(
      OPT_function_arg_placeholders, OPT_no_function_arg_placeholders, true);
  Opts.CodeComplete.RunParser = parseCodeCompletionParse(Args);
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
      ProviderStack.push_back(config::Provider::fromYAMLFile(
          UserConfig, /*Directory=*/"", TFS, /*Trusted=*/true));
    } else {
      elog("Couldn't determine user config file, not loading");
    }
  }

  FlagsConfigProviderOpts FlagsProviderOpts{
      Args.getLastArgValue(OPT_compile_commands_dir_EQ).str(),
      Args.getLastArgValue(OPT_index_file_EQ).str(),
      Args.hasFlag(OPT_background_index, OPT_no_background_index, true),
      {},
      Args.hasArg(OPT_lit_test),
  };

  if (Args.hasArg(OPT_all_scopes_completion) ||
      Args.hasArg(OPT_no_all_scopes_completion)) {
    FlagsProviderOpts.AllScopesCompletion = std::make_optional(
        Args.hasFlag(OPT_all_scopes_completion, OPT_no_all_scopes_completion,
                     CodeCompleteOptions().AllScopes));
  }

  ProviderStack.push_back(
      std::make_unique<FlagsConfigProvider>(FlagsProviderOpts));
  std::vector<const config::Provider *> ProviderPointers;
  for (const auto &P : ProviderStack)
    ProviderPointers.push_back(P.get());
  Config = config::Provider::combine(std::move(ProviderPointers));
  Opts.ConfigProvider = Config.get();

  // Create an empty clang-tidy option.
  TidyProvider ClangTidyOptProvider;
  if (Args.hasFlag(OPT_clang_tidy, OPT_no_clang_tidy, true)) {
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
  Opts.UseDirtyHeaders =
      Args.hasFlag(OPT_use_dirty_headers, OPT_no_use_dirty_headers,
                   ClangdServer::Options().UseDirtyHeaders);

  Opts.PreambleParseForwardingFunctions = Args.hasFlag(
      OPT_parse_forwarding_functions, OPT_no_parse_forwarding_functions,
      ParseOptions().PreambleParseForwardingFunctions);

  Opts.ImportInsertions =
      Args.hasFlag(OPT_import_insertions, OPT_no_import_insertions, true);

  Opts.QueryDriverGlobs = Args.getAllArgValues(OPT_query_driver_EQ);

  const bool HiddenFeatures = Args.hasArg(OPT_hidden_features);
  std::vector<std::string> TweakList = Args.getAllArgValues(OPT_tweaks_EQ);

  Opts.TweakFilter = [&](const Tweak &T) {
    if (T.hidden() && !HiddenFeatures)
      return false;
    if (!TweakList.empty())
      return llvm::is_contained(TweakList, T.id());
    return true;
  };

  OffsetEncoding ForceOffsetEncoding =
      StringSwitch<OffsetEncoding>(
          Args.getLastArgValue(OPT_offset_encoding_EQ, ""))
          .Case("utf-8", OffsetEncoding::UTF8)
          .Case("utf-16", OffsetEncoding::UTF16)
          .Case("utf-32", OffsetEncoding::UTF32)
          .Default(OffsetEncoding::UnsupportedEncoding);

  if (ForceOffsetEncoding != OffsetEncoding::UnsupportedEncoding)
    Opts.Encoding = ForceOffsetEncoding;

  if (Args.hasArg(OPT_check_EQ)) {
    llvm::SmallString<256> Path;
    StringRef CheckFile = Args.getLastArgValue(OPT_check_EQ);
    if (auto Error =
            llvm::sys::fs::real_path(CheckFile, Path, /*expand_tilde=*/true)) {
      elog("Failed to resolve path {0}: {1}", CheckFile, Error.message());
      return 1;
    }
    log("Entering check mode (no LSP server)");

    ClangdCheckOptions CheckOpts;
    if (Args.hasArg(OPT_check_tidy_time_EQ))
      CheckOpts.CheckTidyTime =
          std::make_optional(Args.getLastArgValue(OPT_check_tidy_time_EQ));

    CheckOpts.CheckFileLines = Args.getLastArgValue(OPT_check_file_lines_EQ);
    CheckOpts.CheckCompletion = Args.hasArg(OPT_check_completion);
    CheckOpts.CheckLocations =
        Args.hasFlag(OPT_check_locations, OPT_no_check_locations, true);
    CheckOpts.CheckWarnings = Args.hasArg(OPT_check_warnings);

    return check(Path, TFS, Opts, CheckOpts)
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
        stdin, llvm::outs(), InputMirrorStream ? &*InputMirrorStream : nullptr,
        PrettyPrint, InputStyle);
  }
  if (Args.hasArg(OPT_path_mappings_EQ)) {
    auto Mappings =
        parsePathMappings(Args.getLastArgValue(OPT_path_mappings_EQ));
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

} // namespace clangd
} // namespace clang
