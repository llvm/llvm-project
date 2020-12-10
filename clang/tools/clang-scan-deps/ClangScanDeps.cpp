//===- ClangScanDeps.cpp - Implementation of clang-scan-deps --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Tooling/JSONCompilationDatabase.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include <mutex>
#include <thread>

using namespace clang;
using namespace tooling::dependencies;

namespace {

class SharedStream {
public:
  SharedStream(raw_ostream &OS) : OS(OS) {}
  void applyLocked(llvm::function_ref<void(raw_ostream &OS)> Fn) {
    std::unique_lock<std::mutex> LockGuard(Lock);
    Fn(OS);
    OS.flush();
  }

private:
  std::mutex Lock;
  raw_ostream &OS;
};

class ResourceDirectoryCache {
public:
  /// findResourceDir finds the resource directory relative to the clang
  /// compiler being used in Args, by running it with "-print-resource-dir"
  /// option and cache the results for reuse. \returns resource directory path
  /// associated with the given invocation command or empty string if the
  /// compiler path is NOT an absolute path.
  StringRef findResourceDir(const tooling::CommandLineArguments &Args) {
    if (Args.size() < 1)
      return "";

    const std::string &ClangBinaryPath = Args[0];
    if (!llvm::sys::path::is_absolute(ClangBinaryPath))
      return "";

    const std::string &ClangBinaryName =
        std::string(llvm::sys::path::filename(ClangBinaryPath));

    std::unique_lock<std::mutex> LockGuard(CacheLock);
    const auto &CachedResourceDir = Cache.find(ClangBinaryPath);
    if (CachedResourceDir != Cache.end())
      return CachedResourceDir->second;

    std::vector<StringRef> PrintResourceDirArgs{ClangBinaryName,
                                                "-print-resource-dir"};
    llvm::SmallString<64> OutputFile, ErrorFile;
    llvm::sys::fs::createTemporaryFile("print-resource-dir-output",
                                       "" /*no-suffix*/, OutputFile);
    llvm::sys::fs::createTemporaryFile("print-resource-dir-error",
                                       "" /*no-suffix*/, ErrorFile);
    llvm::FileRemover OutputRemover(OutputFile.c_str());
    llvm::FileRemover ErrorRemover(ErrorFile.c_str());
    llvm::Optional<StringRef> Redirects[] = {
        {""}, // Stdin
        StringRef(OutputFile),
        StringRef(ErrorFile),
    };
    if (const int RC = llvm::sys::ExecuteAndWait(
            ClangBinaryPath, PrintResourceDirArgs, {}, Redirects)) {
      auto ErrorBuf = llvm::MemoryBuffer::getFile(ErrorFile.c_str());
      llvm::errs() << ErrorBuf.get()->getBuffer();
      return "";
    }

    auto OutputBuf = llvm::MemoryBuffer::getFile(OutputFile.c_str());
    if (!OutputBuf)
      return "";
    StringRef Output = OutputBuf.get()->getBuffer().rtrim('\n');

    Cache[ClangBinaryPath] = Output.str();
    return Cache[ClangBinaryPath];
  }

private:
  std::map<std::string, std::string> Cache;
  std::mutex CacheLock;
};

llvm::cl::opt<bool> Help("h", llvm::cl::desc("Alias for -help"),
                         llvm::cl::Hidden);

llvm::cl::OptionCategory DependencyScannerCategory("Tool options");

static llvm::cl::opt<ScanningMode> ScanMode(
    "mode",
    llvm::cl::desc("The preprocessing mode used to compute the dependencies"),
    llvm::cl::values(
        clEnumValN(ScanningMode::MinimizedSourcePreprocessing,
                   "preprocess-minimized-sources",
                   "The set of dependencies is computed by preprocessing the "
                   "source files that were minimized to only include the "
                   "contents that might affect the dependencies"),
        clEnumValN(ScanningMode::CanonicalPreprocessing, "preprocess",
                   "The set of dependencies is computed by preprocessing the "
                   "unmodified source files")),
    llvm::cl::init(ScanningMode::MinimizedSourcePreprocessing),
    llvm::cl::cat(DependencyScannerCategory));

static llvm::cl::opt<ScanningOutputFormat> Format(
    "format", llvm::cl::desc("The output format for the dependencies"),
    llvm::cl::values(clEnumValN(ScanningOutputFormat::Make, "make",
                                "Makefile compatible dep file"),
                     clEnumValN(ScanningOutputFormat::Full, "experimental-full",
                                "Full dependency graph suitable"
                                " for explicitly building modules. This format "
                                "is experimental and will change.")),
    llvm::cl::init(ScanningOutputFormat::Make),
    llvm::cl::cat(DependencyScannerCategory));

static llvm::cl::opt<bool> FullCommandLine(
    "full-command-line",
    llvm::cl::desc("Include the full command lines to use to build modules"),
    llvm::cl::init(false), llvm::cl::cat(DependencyScannerCategory));

llvm::cl::opt<unsigned>
    NumThreads("j", llvm::cl::Optional,
               llvm::cl::desc("Number of worker threads to use (default: use "
                              "all concurrent threads)"),
               llvm::cl::init(0), llvm::cl::cat(DependencyScannerCategory));

llvm::cl::opt<std::string>
    CompilationDB("compilation-database",
                  llvm::cl::desc("Compilation database"), llvm::cl::Required,
                  llvm::cl::cat(DependencyScannerCategory));

llvm::cl::opt<bool> ReuseFileManager(
    "reuse-filemanager",
    llvm::cl::desc("Reuse the file manager and its cache between invocations."),
    llvm::cl::init(true), llvm::cl::cat(DependencyScannerCategory));

llvm::cl::opt<bool> SkipExcludedPPRanges(
    "skip-excluded-pp-ranges",
    llvm::cl::desc(
        "Use the preprocessor optimization that skips excluded conditionals by "
        "bumping the buffer pointer in the lexer instead of lexing the tokens  "
        "until reaching the end directive."),
    llvm::cl::init(true), llvm::cl::cat(DependencyScannerCategory));

llvm::cl::opt<bool> Verbose("v", llvm::cl::Optional,
                            llvm::cl::desc("Use verbose output."),
                            llvm::cl::init(false),
                            llvm::cl::cat(DependencyScannerCategory));

} // end anonymous namespace

/// \returns object-file path derived from source-file path.
static std::string getObjFilePath(StringRef SrcFile) {
  SmallString<128> ObjFileName(SrcFile);
  llvm::sys::path::replace_extension(ObjFileName, "o");
  return std::string(ObjFileName.str());
}

class SingleCommandCompilationDatabase : public tooling::CompilationDatabase {
public:
  SingleCommandCompilationDatabase(tooling::CompileCommand Cmd)
      : Command(std::move(Cmd)) {}

  std::vector<tooling::CompileCommand>
  getCompileCommands(StringRef FilePath) const override {
    return {Command};
  }

  std::vector<tooling::CompileCommand> getAllCompileCommands() const override {
    return {Command};
  }

private:
  tooling::CompileCommand Command;
};

/// Takes the result of a dependency scan and prints error / dependency files
/// based on the result.
///
/// \returns True on error.
static bool
handleMakeDependencyToolResult(const std::string &Input,
                               llvm::Expected<std::string> &MaybeFile,
                               SharedStream &OS, SharedStream &Errs) {
  if (!MaybeFile) {
    llvm::handleAllErrors(
        MaybeFile.takeError(), [&Input, &Errs](llvm::StringError &Err) {
          Errs.applyLocked([&](raw_ostream &OS) {
            OS << "Error while scanning dependencies for " << Input << ":\n";
            OS << Err.getMessage();
          });
        });
    return true;
  }
  OS.applyLocked([&](raw_ostream &OS) { OS << *MaybeFile; });
  return false;
}

static llvm::json::Array toJSONSorted(const llvm::StringSet<> &Set) {
  std::vector<llvm::StringRef> Strings;
  for (auto &&I : Set)
    Strings.push_back(I.getKey());
  llvm::sort(Strings);
  return llvm::json::Array(Strings);
}

static llvm::json::Array toJSONSorted(std::vector<ClangModuleDep> V) {
  llvm::sort(V, [](const ClangModuleDep &A, const ClangModuleDep &B) {
    return std::tie(A.ModuleName, A.ContextHash) <
           std::tie(B.ModuleName, B.ContextHash);
  });

  llvm::json::Array Ret;
  for (const ClangModuleDep &CMD : V)
    Ret.push_back(llvm::json::Object(
        {{"module-name", CMD.ModuleName}, {"context-hash", CMD.ContextHash}}));
  return Ret;
}

// Thread safe.
class FullDeps {
public:
  void mergeDeps(StringRef Input, FullDependenciesResult FDR,
                 size_t InputIndex) {
    const FullDependencies &FD = FDR.FullDeps;

    InputDeps ID;
    ID.FileName = std::string(Input);
    ID.ContextHash = std::move(FD.ContextHash);
    ID.FileDeps = std::move(FD.FileDeps);
    ID.ModuleDeps = std::move(FD.ClangModuleDeps);

    std::unique_lock<std::mutex> ul(Lock);
    for (const ModuleDeps &MD : FDR.DiscoveredModules) {
      auto I = Modules.find({MD.ContextHash, MD.ModuleName, 0});
      if (I != Modules.end()) {
        I->first.InputIndex = std::min(I->first.InputIndex, InputIndex);
        continue;
      }
      Modules.insert(
          I, {{MD.ContextHash, MD.ModuleName, InputIndex}, std::move(MD)});
    }

    if (FullCommandLine)
      ID.AdditonalCommandLine = FD.getAdditionalCommandLine(
          [&](ClangModuleDep CMD) { return lookupPCMPath(CMD); },
          [&](ClangModuleDep CMD) -> const ModuleDeps & {
            return lookupModuleDeps(CMD);
          });

    Inputs.push_back(std::move(ID));
  }

  void printFullOutput(raw_ostream &OS) {
    // Sort the modules by name to get a deterministic order.
    std::vector<ContextModulePair> ModuleNames;
    for (auto &&M : Modules)
      ModuleNames.push_back(M.first);
    llvm::sort(ModuleNames,
               [](const ContextModulePair &A, const ContextModulePair &B) {
                 return std::tie(A.ModuleName, A.InputIndex) <
                        std::tie(B.ModuleName, B.InputIndex);
               });

    llvm::sort(Inputs, [](const InputDeps &A, const InputDeps &B) {
      return A.FileName < B.FileName;
    });

    using namespace llvm::json;

    Array OutModules;
    for (auto &&ModName : ModuleNames) {
      auto &MD = Modules[ModName];
      Object O{
          {"name", MD.ModuleName},
          {"context-hash", MD.ContextHash},
          {"file-deps", toJSONSorted(MD.FileDeps)},
          {"clang-module-deps", toJSONSorted(MD.ClangModuleDeps)},
          {"clang-modulemap-file", MD.ClangModuleMapFile},
          {"command-line",
           FullCommandLine
               ? MD.getFullCommandLine(
                     [&](ClangModuleDep CMD) { return lookupPCMPath(CMD); },
                     [&](ClangModuleDep CMD) -> const ModuleDeps & {
                       return lookupModuleDeps(CMD);
                     })
               : MD.NonPathCommandLine},
      };
      OutModules.push_back(std::move(O));
    }

    Array TUs;
    for (auto &&I : Inputs) {
      Object O{
          {"input-file", I.FileName},
          {"clang-context-hash", I.ContextHash},
          {"file-deps", I.FileDeps},
          {"clang-module-deps", toJSONSorted(I.ModuleDeps)},
          {"command-line", I.AdditonalCommandLine},
      };
      TUs.push_back(std::move(O));
    }

    Object Output{
        {"modules", std::move(OutModules)},
        {"translation-units", std::move(TUs)},
    };

    OS << llvm::formatv("{0:2}\n", Value(std::move(Output)));
  }

private:
  StringRef lookupPCMPath(ClangModuleDep CMD) {
    return Modules[ContextModulePair{CMD.ContextHash, CMD.ModuleName, 0}]
        .ImplicitModulePCMPath;
  }

  const ModuleDeps &lookupModuleDeps(ClangModuleDep CMD) {
    auto I =
        Modules.find(ContextModulePair{CMD.ContextHash, CMD.ModuleName, 0});
    assert(I != Modules.end());
    return I->second;
  };

  struct ContextModulePair {
    std::string ContextHash;
    std::string ModuleName;
    mutable size_t InputIndex;

    bool operator==(const ContextModulePair &Other) const {
      return ContextHash == Other.ContextHash && ModuleName == Other.ModuleName;
    }
  };

  struct ContextModulePairHasher {
    std::size_t operator()(const ContextModulePair &CMP) const {
      using llvm::hash_combine;

      return hash_combine(CMP.ContextHash, CMP.ModuleName);
    }
  };

  struct InputDeps {
    std::string FileName;
    std::string ContextHash;
    std::vector<std::string> FileDeps;
    std::vector<ClangModuleDep> ModuleDeps;
    std::vector<std::string> AdditonalCommandLine;
  };

  std::mutex Lock;
  std::unordered_map<ContextModulePair, ModuleDeps, ContextModulePairHasher>
      Modules;
  std::vector<InputDeps> Inputs;
};

static bool handleFullDependencyToolResult(
    const std::string &Input,
    llvm::Expected<FullDependenciesResult> &MaybeFullDeps, FullDeps &FD,
    size_t InputIndex, SharedStream &OS, SharedStream &Errs) {
  if (!MaybeFullDeps) {
    llvm::handleAllErrors(
        MaybeFullDeps.takeError(), [&Input, &Errs](llvm::StringError &Err) {
          Errs.applyLocked([&](raw_ostream &OS) {
            OS << "Error while scanning dependencies for " << Input << ":\n";
            OS << Err.getMessage();
          });
        });
    return true;
  }
  FD.mergeDeps(Input, std::move(*MaybeFullDeps), InputIndex);
  return false;
}

int main(int argc, const char **argv) {
  llvm::InitLLVM X(argc, argv);
  llvm::cl::HideUnrelatedOptions(DependencyScannerCategory);
  if (!llvm::cl::ParseCommandLineOptions(argc, argv))
    return 1;

  std::string ErrorMessage;
  std::unique_ptr<tooling::JSONCompilationDatabase> Compilations =
      tooling::JSONCompilationDatabase::loadFromFile(
          CompilationDB, ErrorMessage,
          tooling::JSONCommandLineSyntax::AutoDetect);
  if (!Compilations) {
    llvm::errs() << "error: " << ErrorMessage << "\n";
    return 1;
  }

  llvm::cl::PrintOptionValues();

  // The command options are rewritten to run Clang in preprocessor only mode.
  auto AdjustingCompilations =
      std::make_unique<tooling::ArgumentsAdjustingCompilations>(
          std::move(Compilations));
  ResourceDirectoryCache ResourceDirCache;
  AdjustingCompilations->appendArgumentsAdjuster(
      [&ResourceDirCache](const tooling::CommandLineArguments &Args,
                          StringRef FileName) {
        std::string LastO = "";
        bool HasMT = false;
        bool HasMQ = false;
        bool HasMD = false;
        bool HasResourceDir = false;
        // We need to find the last -o value.
        if (!Args.empty()) {
          std::size_t Idx = Args.size() - 1;
          for (auto It = Args.rbegin(); It != Args.rend(); ++It) {
            StringRef Arg = Args[Idx];
            if (LastO.empty()) {
              if (Arg == "-o" && It != Args.rbegin())
                LastO = Args[Idx + 1];
              else if (Arg.startswith("-o"))
                LastO = Arg.drop_front(2).str();
            }
            if (Arg == "-MT")
              HasMT = true;
            if (Arg == "-MQ")
              HasMQ = true;
            if (Arg == "-MD")
              HasMD = true;
            if (Arg == "-resource-dir")
              HasResourceDir = true;
            --Idx;
          }
        }
        // If there's no -MT/-MQ Driver would add -MT with the value of the last
        // -o option.
        tooling::CommandLineArguments AdjustedArgs = Args;
        AdjustedArgs.push_back("-o");
        AdjustedArgs.push_back("/dev/null");
        if (!HasMT && !HasMQ) {
          AdjustedArgs.push_back("-M");
          AdjustedArgs.push_back("-MT");
          // We're interested in source dependencies of an object file.
          if (!HasMD) {
            // FIXME: We are missing the directory unless the -o value is an
            // absolute path.
            AdjustedArgs.push_back(!LastO.empty() ? LastO
                                                  : getObjFilePath(FileName));
          } else {
            AdjustedArgs.push_back(std::string(FileName));
          }
        }
        AdjustedArgs.push_back("-Xclang");
        AdjustedArgs.push_back("-Eonly");
        AdjustedArgs.push_back("-Xclang");
        AdjustedArgs.push_back("-sys-header-deps");
        AdjustedArgs.push_back("-Wno-error");

        if (!HasResourceDir) {
          StringRef ResourceDir =
              ResourceDirCache.findResourceDir(Args);
          if (!ResourceDir.empty()) {
            AdjustedArgs.push_back("-resource-dir");
            AdjustedArgs.push_back(std::string(ResourceDir));
          }
        }
        return AdjustedArgs;
      });
  AdjustingCompilations->appendArgumentsAdjuster(
      tooling::getClangStripSerializeDiagnosticAdjuster());

  SharedStream Errs(llvm::errs());
  // Print out the dependency results to STDOUT by default.
  SharedStream DependencyOS(llvm::outs());

  DependencyScanningService Service(ScanMode, Format, ReuseFileManager,
                                    SkipExcludedPPRanges);
  llvm::ThreadPool Pool(llvm::hardware_concurrency(NumThreads));
  std::vector<std::unique_ptr<DependencyScanningTool>> WorkerTools;
  for (unsigned I = 0; I < Pool.getThreadCount(); ++I)
    WorkerTools.push_back(std::make_unique<DependencyScanningTool>(Service));

  std::vector<SingleCommandCompilationDatabase> Inputs;
  for (tooling::CompileCommand Cmd :
       AdjustingCompilations->getAllCompileCommands())
    Inputs.emplace_back(Cmd);

  std::atomic<bool> HadErrors(false);
  FullDeps FD;
  std::mutex Lock;
  size_t Index = 0;

  if (Verbose) {
    llvm::outs() << "Running clang-scan-deps on " << Inputs.size()
                 << " files using " << Pool.getThreadCount() << " workers\n";
  }
  for (unsigned I = 0; I < Pool.getThreadCount(); ++I) {
    Pool.async([I, &Lock, &Index, &Inputs, &HadErrors, &FD, &WorkerTools,
                &DependencyOS, &Errs]() {
      llvm::StringSet<> AlreadySeenModules;
      while (true) {
        const SingleCommandCompilationDatabase *Input;
        std::string Filename;
        std::string CWD;
        size_t LocalIndex;
        // Take the next input.
        {
          std::unique_lock<std::mutex> LockGuard(Lock);
          if (Index >= Inputs.size())
            return;
          LocalIndex = Index;
          Input = &Inputs[Index++];
          tooling::CompileCommand Cmd = Input->getAllCompileCommands()[0];
          Filename = std::move(Cmd.Filename);
          CWD = std::move(Cmd.Directory);
        }
        // Run the tool on it.
        if (Format == ScanningOutputFormat::Make) {
          auto MaybeFile = WorkerTools[I]->getDependencyFile(*Input, CWD);
          if (handleMakeDependencyToolResult(Filename, MaybeFile, DependencyOS,
                                             Errs))
            HadErrors = true;
        } else {
          auto MaybeFullDeps = WorkerTools[I]->getFullDependencies(
              *Input, CWD, AlreadySeenModules);
          if (handleFullDependencyToolResult(Filename, MaybeFullDeps, FD,
                                             LocalIndex, DependencyOS, Errs))
            HadErrors = true;
        }
      }
    });
  }
  Pool.wait();

  if (Format == ScanningOutputFormat::Full)
    FD.printFullOutput(llvm::outs());

  return HadErrors;
}
