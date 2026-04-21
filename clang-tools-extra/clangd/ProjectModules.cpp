//===------------------ ProjectModules.cpp ---------  ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProjectModules.h"
#include "Compiler.h"
#include "support/Logger.h"
#include "clang/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanningTool.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Host.h"

namespace clang::clangd {
namespace {

llvm::SmallString<128> normalizePath(PathRef Path) {
  llvm::SmallString<128> Result(Path);
  llvm::sys::path::remove_dots(Result, /*remove_dot_dot=*/true);
  llvm::sys::path::native(Result, llvm::sys::path::Style::posix);
  return Result;
}

std::string normalizePath(PathRef Path, PathRef WorkingDir) {
  if (Path.empty())
    return {};

  llvm::SmallString<128> Result;
  if (llvm::sys::path::is_absolute(Path) || WorkingDir.empty())
    Result = Path;
  else {
    Result = WorkingDir;
    llvm::sys::path::append(Result, Path);
  }

  return normalizePath(Result).str().str();
}

/// The information related to modules parsed from compile commands.
/// Including the source file, the module file it produces (if it is a
/// producer), and the module and the corresponding module files it
/// requires (if it is a consumer)
struct ParsedCompileCommandInfo {
  std::string SourceFile;
  std::optional<std::string> OutputModuleFile;
  // Map from required module name to the module file path.
  llvm::StringMap<std::string> RequiredModuleFiles;
};

/// Get ParsedCompileCommandInfo by looking at the '--precompile',
/// '-fmodule-file=' and '-fmodule-file=' commands in the compile command.
std::optional<ParsedCompileCommandInfo>
parseCompileCommandInfo(tooling::CompileCommand Cmd, const ThreadsafeFS &TFS) {
  auto FS = TFS.view(std::nullopt);
  auto Tokenizer = llvm::Triple(llvm::sys::getProcessTriple()).isOSWindows()
                       ? llvm::cl::TokenizeWindowsCommandLine
                       : llvm::cl::TokenizeGNUCommandLine;
  tooling::addExpandedResponseFiles(Cmd.CommandLine, Cmd.Directory, Tokenizer,
                                    *FS);

  ParsedCompileCommandInfo Result;
  Result.SourceFile = normalizePath(Cmd.Filename, Cmd.Directory);

  bool SawPrecompile = false;
  for (size_t I = 1; I < Cmd.CommandLine.size(); ++I) {
    llvm::StringRef Arg = Cmd.CommandLine[I];
    if (Arg == "--precompile") {
      SawPrecompile = true;
      continue;
    }

    if (Arg.consume_front("-fmodule-output=")) {
      Result.OutputModuleFile = normalizePath(Arg, Cmd.Directory);
      continue;
    }
    if (Arg == "-fmodule-output" && I + 1 < Cmd.CommandLine.size()) {
      Result.OutputModuleFile =
          normalizePath(Cmd.CommandLine[++I], Cmd.Directory);
      continue;
    }
    if (SawPrecompile && Arg == "-o" && I + 1 < Cmd.CommandLine.size()) {
      Result.OutputModuleFile =
          normalizePath(Cmd.CommandLine[++I], Cmd.Directory);
      continue;
    }
    if (SawPrecompile && Arg.starts_with("-o") && Arg.size() > 2) {
      Result.OutputModuleFile = normalizePath(Arg.drop_front(2), Cmd.Directory);
      continue;
    }

    if (!Arg.consume_front("-fmodule-file="))
      continue;

    auto Sep = Arg.find('=');
    if (Sep == llvm::StringRef::npos || Sep == 0 || Sep + 1 == Arg.size())
      continue;

    Result.RequiredModuleFiles[Arg.take_front(Sep)] =
        normalizePath(Arg.drop_front(Sep + 1), Cmd.Directory);
  }

  return Result;
}

std::optional<tooling::CompileCommand>
getCompileCommandForFile(const clang::tooling::CompilationDatabase &CDB,
                         PathRef FilePath,
                         const ProjectModules::CommandMangler &Mangler) {
  auto Candidates = CDB.getCompileCommands(FilePath);
  if (Candidates.empty())
    return std::nullopt;

  // Choose the first candidates as the compile commands as the file.
  // Following the same logic with
  // DirectoryBasedGlobalCompilationDatabase::getCompileCommand.
  tooling::CompileCommand Cmd = std::move(Candidates.front());

  if (Mangler)
    Mangler(Cmd, FilePath);

  return Cmd;
}

/// A scanner to query the dependency information for C++20 Modules.
///
/// The scanner can scan a single file with `scan(PathRef)` member function
/// or scan the whole project with `globalScan(vector<PathRef>)` member
/// function. See the comments of `globalScan` to see the details.
///
/// The ModuleDependencyScanner can get the directly required module names for a
/// specific source file. Also the ModuleDependencyScanner can get the source
/// file declaring the primary module interface for a specific module name.
///
/// IMPORTANT NOTE: we assume that every module unit is only declared once in a
/// source file in the project. But the assumption is not strictly true even
/// besides the invalid projects. The language specification requires that every
/// module unit should be unique in a valid program. But a project can contain
/// multiple programs. Then it is valid that we can have multiple source files
/// declaring the same module in a project as long as these source files don't
/// interfere with each other.
class ModuleDependencyScanner {
public:
  ModuleDependencyScanner(
      std::shared_ptr<const clang::tooling::CompilationDatabase> CDB,
      const ThreadsafeFS &TFS)
      : CDB(CDB), Service([&TFS] {
          dependencies::DependencyScanningServiceOptions Opts;
          Opts.MakeVFS = [&] { return TFS.view(std::nullopt); };
          Opts.Mode = dependencies::ScanningMode::CanonicalPreprocessing;
          Opts.Format = dependencies::ScanningOutputFormat::P1689;
          return Opts;
        }()) {}

  /// The scanned modules dependency information for a specific source file.
  struct ModuleDependencyInfo {
    /// The name of the module if the file is a module unit.
    std::optional<std::string> ModuleName;
    /// A list of names for the modules that the file directly depends.
    std::vector<std::string> RequiredModules;
  };

  /// Scanning the single file specified by \param FilePath.
  std::optional<ModuleDependencyInfo>
  scan(PathRef FilePath, const ProjectModules::CommandMangler &Mangler);

  /// Scanning every source file in the current project to get the
  /// <module-name> to <module-unit-source> map.
  /// TODO: We should find an efficient method to get the <module-name>
  /// to <module-unit-source> map. We can make it either by providing
  /// a global module dependency scanner to monitor every file. Or we
  /// can simply require the build systems (or even the end users)
  /// to provide the map.
  void globalScan(const ProjectModules::CommandMangler &Mangler);

  /// Get the source file from the module name. Note that the language
  /// guarantees all the module names are unique in a valid program.
  /// This function should only be called after globalScan.
  ///
  /// TODO: We should handle the case that there are multiple source files
  /// declaring the same module.
  PathRef getSourceForModuleName(llvm::StringRef ModuleName) const;

  /// Return the direct required modules. Indirect required modules are not
  /// included.
  std::vector<std::string>
  getRequiredModules(PathRef File,
                     const ProjectModules::CommandMangler &Mangler);

private:
  std::shared_ptr<const clang::tooling::CompilationDatabase> CDB;

  // Whether the scanner has scanned the project globally.
  bool GlobalScanned = false;

  clang::dependencies::DependencyScanningService Service;

  // TODO: Add a scanning cache.

  // Map module name to source file path.
  llvm::StringMap<std::string> ModuleNameToSource;
};

std::optional<ModuleDependencyScanner::ModuleDependencyInfo>
ModuleDependencyScanner::scan(PathRef FilePath,
                              const ProjectModules::CommandMangler &Mangler) {
  auto Cmd = getCompileCommandForFile(*CDB, FilePath, Mangler);
  if (!Cmd)
    return std::nullopt;

  using namespace clang::tooling;

  DependencyScanningTool ScanningTool(Service);

  std::string S;
  llvm::raw_string_ostream OS(S);
  DiagnosticOptions DiagOpts;
  DiagOpts.ShowCarets = false;
  TextDiagnosticPrinter DiagConsumer(OS, DiagOpts);

  std::optional<P1689Rule> ScanningResult =
      ScanningTool.getP1689ModuleDependencyFile(*Cmd, Cmd->Directory,
                                                DiagConsumer);

  if (!ScanningResult) {
    elog("Scanning modules dependencies for {0} failed: {1}", FilePath, S);
    std::string Cmdline;
    for (auto &Arg : Cmd->CommandLine)
      Cmdline += Arg + " ";
    elog("The command line the scanning tool use is: {0}", Cmdline);
    return std::nullopt;
  }

  ModuleDependencyInfo Result;

  if (ScanningResult->Provides) {
    Result.ModuleName = ScanningResult->Provides->ModuleName;

    auto [Iter, Inserted] = ModuleNameToSource.try_emplace(
        ScanningResult->Provides->ModuleName, FilePath);

    if (!Inserted && Iter->second != FilePath) {
      elog("Detected multiple source files ({0}, {1}) declaring the same "
           "module: '{2}'. "
           "Now clangd may find the wrong source in such case.",
           Iter->second, FilePath, ScanningResult->Provides->ModuleName);
    }
  }

  for (auto &Required : ScanningResult->Requires)
    Result.RequiredModules.push_back(Required.ModuleName);

  return Result;
}

void ModuleDependencyScanner::globalScan(
    const ProjectModules::CommandMangler &Mangler) {
  if (GlobalScanned)
    return;

  for (auto &File : CDB->getAllFiles())
    scan(File, Mangler);

  GlobalScanned = true;
}

PathRef ModuleDependencyScanner::getSourceForModuleName(
    llvm::StringRef ModuleName) const {
  assert(
      GlobalScanned &&
      "We should only call getSourceForModuleName after calling globalScan()");

  if (auto It = ModuleNameToSource.find(ModuleName);
      It != ModuleNameToSource.end())
    return It->second;

  return {};
}

std::vector<std::string> ModuleDependencyScanner::getRequiredModules(
    PathRef File, const ProjectModules::CommandMangler &Mangler) {
  auto ScanningResult = scan(File, Mangler);
  if (!ScanningResult)
    return {};

  return ScanningResult->RequiredModules;
}
} // namespace

/// TODO: The existing `ScanningAllProjectModules` is not efficient. See the
/// comments in ModuleDependencyScanner for detail.
///
/// In the future, we wish the build system can provide a well design
/// compilation database for modules then we can query that new compilation
/// database directly. Or we need to have a global long-live scanner to detect
/// the state of each file.
class ScanningAllProjectModules : public ProjectModules {
public:
  ScanningAllProjectModules(
      std::shared_ptr<const clang::tooling::CompilationDatabase> CDB,
      const ThreadsafeFS &TFS)
      : Scanner(CDB, TFS) {}

  ~ScanningAllProjectModules() override = default;

  std::vector<std::string> getRequiredModules(PathRef File) override {
    return Scanner.getRequiredModules(File, Mangler);
  }

  void setCommandMangler(CommandMangler Mangler) override {
    this->Mangler = std::move(Mangler);
  }

  /// RequiredSourceFile is not used intentionally. See the comments of
  /// ModuleDependencyScanner for detail.
  std::string getSourceForModuleName(llvm::StringRef ModuleName,
                                     PathRef RequiredSourceFile) override {
    Scanner.globalScan(Mangler);
    return Scanner.getSourceForModuleName(ModuleName).str();
  }

  std::string getModuleNameForSource(PathRef File) override {
    auto ScanningResult = Scanner.scan(File, Mangler);
    if (!ScanningResult || !ScanningResult->ModuleName)
      return {};

    return *ScanningResult->ModuleName;
  }

  // Determining Unique/Multiple needs a global scan; return Unknown for cost
  // reasons. We will have other ProjectModules implementations can determine
  // this more efficiently.
  ModuleNameState getModuleNameState(llvm::StringRef /*ModuleName*/) override {
    return ModuleNameState::Unknown;
  }

private:
  ModuleDependencyScanner Scanner;
  CommandMangler Mangler;
};

/// Reads project module information directly from compile commands.
///
/// The key observation is that compile commands may already encode the mapping
/// between a TU, the module names it imports, and the BMI paths it uses:
/// - producers may spell the BMI path with `--precompile -o <bmi>` or
///   `-fmodule-output=<bmi>`
/// - consumers may spell the mapping from module name to BMI path with
///   `-fmodule-file=<module>=<bmi>`
///
/// When that information is present, we can answer
/// `getSourceForModuleName(ModuleName, RequiredSourceFile)` by first looking up
/// the BMI path the consumer TU uses for `ModuleName`, and then mapping that
/// BMI path back to the module unit source that produced it. This avoids the
/// older scanning-only approach of guessing the module unit from the module
/// name alone.
///
/// One subtle point is that producer commands alone do not reliably tell us the
/// module name associated with a BMI path. In practice this backend learns that
/// association from consumer `-fmodule-file=` entries, and then uses the BMI
/// path to recover the producer source file. That is why indexing is built from
/// both producer and consumer commands.
///
/// Note that compilation database can be stale, so results from this backend
/// should be treated as preferred hints rather than unquestionable truth.
/// The compound layer below validates or falls back when needed.
class CompileCommandsProjectModules : public ProjectModules {
public:
  CompileCommandsProjectModules(
      std::shared_ptr<const clang::tooling::CompilationDatabase> CDB,
      const ThreadsafeFS &TFS)
      : CDB(std::move(CDB)), TFS(TFS) {}

  std::vector<std::string> getRequiredModules(PathRef File) override {
    auto Parsed = parseFileCommand(File);
    if (!Parsed)
      return {};

    std::vector<std::string> Result;
    Result.reserve(Parsed->RequiredModuleFiles.size());
    for (const auto &Required : Parsed->RequiredModuleFiles)
      Result.push_back(Required.getKey().str());
    return Result;
  }

  std::string getModuleNameForSource(PathRef File) override {
    indexProducerCommands();
    auto It = SourceToModuleName.find(
        maybeCaseFoldPath(normalizePath(File, /*WorkingDir=*/{})));
    if (It == SourceToModuleName.end() || It->second.Ambiguous)
      return {};
    return It->second.Name;
  }

  ModuleNameState getModuleNameState(llvm::StringRef ModuleName) override {
    indexProducerCommands();
    auto It = ModuleNameToDistinctSources.find(ModuleName);
    if (It == ModuleNameToDistinctSources.end())
      return ModuleNameState::Unknown;
    return It->second.size() > 1 ? ModuleNameState::Multiple
                                 : ModuleNameState::Unique;
  }

  std::string getSourceForModuleName(llvm::StringRef ModuleName,
                                     PathRef RequiredSourceFile) override {
    auto Parsed = parseFileCommand(RequiredSourceFile);
    if (!Parsed)
      return {};

    auto It = Parsed->RequiredModuleFiles.find(ModuleName);
    if (It == Parsed->RequiredModuleFiles.end())
      return {};

    indexProducerCommands();
    auto SourceIt = PCMToSource.find(maybeCaseFoldPath(It->second));
    if (SourceIt == PCMToSource.end())
      return {};

    return SourceIt->second;
  }

  void setCommandMangler(CommandMangler Mangler) override {
    this->Mangler = std::move(Mangler);
    ProducerCommandsIndexed = false;
    PCMToSource.clear();
    ModuleNameToDistinctSources.clear();
    SourceToModuleName.clear();
  }

private:
  /// Parses the compile command for \p File into the module information
  /// encoded in the command line.
  std::optional<ParsedCompileCommandInfo> parseFileCommand(PathRef File) const {
    auto Cmd = getCompileCommandForFile(*CDB, File, Mangler);
    if (!Cmd)
      return std::nullopt;
    return parseCompileCommandInfo(std::move(*Cmd), TFS);
  }

  /// Builds indexes from producer and consumer compile commands.
  ///
  /// Compile commands are parsed once up front. The first pass records which
  /// source file produces each BMI path. The second pass walks consumer
  /// commands, uses `-fmodule-file=` information to associate module names with
  /// those BMI paths, and then records which producer source files are
  /// referenced for each module name.
  void indexProducerCommands() {
    if (ProducerCommandsIndexed)
      return;

    std::vector<ParsedCompileCommandInfo> ParsedCommands;
    auto AllFiles = CDB->getAllFiles();
    ParsedCommands.reserve(AllFiles.size());
    for (const auto &File : AllFiles) {
      auto Parsed = parseFileCommand(File);
      if (!Parsed)
        continue;

      if (Parsed->OutputModuleFile)
        PCMToSource[maybeCaseFoldPath(*Parsed->OutputModuleFile)] =
            Parsed->SourceFile;

      ParsedCommands.push_back(std::move(*Parsed));
    }

    for (const auto &Parsed : ParsedCommands) {
      for (const auto &Required : Parsed.RequiredModuleFiles) {
        auto SourceIt =
            PCMToSource.find(maybeCaseFoldPath(Required.getValue()));
        if (SourceIt == PCMToSource.end())
          continue;
        ModuleNameToDistinctSources[Required.getKey()].insert(
            maybeCaseFoldPath(SourceIt->second));

        auto &Recovered =
            SourceToModuleName[maybeCaseFoldPath(SourceIt->second)];
        if (Recovered.Name.empty())
          Recovered.Name = Required.getKey().str();
        else if (Recovered.Name != Required.getKey()) {
          if (!Recovered.Ambiguous) {
            elog("Detected conflicting module names ('{0}' and '{1}') for "
                 "the same module file {2} produced by source {3}",
                 Recovered.Name, Required.getKey(), Required.getValue(),
                 SourceIt->second);
          }
          Recovered.Ambiguous = true;
        }
      }
    }

    ProducerCommandsIndexed = true;
  }

  std::shared_ptr<const clang::tooling::CompilationDatabase> CDB;
  const ThreadsafeFS &TFS;
  CommandMangler Mangler;
  bool ProducerCommandsIndexed = false;

  llvm::StringMap<std::string> PCMToSource;

  using DistinctSourceSet = llvm::StringSet<>;
  llvm::StringMap<DistinctSourceSet> ModuleNameToDistinctSources;

  struct RecoveredModuleName {
    std::string Name;
    bool Ambiguous = false;
  };
  llvm::StringMap<RecoveredModuleName> SourceToModuleName;
};

/// Combines the compile-commands backend with the scanning backend.
///
/// For getSourceForModuleName, it prefers compile-command-derived results when
/// available to avoid scanning the whole project, but validates them against
/// scanning results to avoid returning stale information. For other queries,
/// it returns scanning results directly as scanning information is update to
/// date.
class CompoundProjectModules : public ProjectModules {
public:
  CompoundProjectModules(
      std::shared_ptr<const clang::tooling::CompilationDatabase> CDB,
      const ThreadsafeFS &TFS)
      : CompileCommands(
            std::make_unique<CompileCommandsProjectModules>(CDB, TFS)),
        Scanning(
            std::make_unique<ScanningAllProjectModules>(std::move(CDB), TFS)) {}

  std::vector<std::string> getRequiredModules(PathRef File) override {
    // Return scanning results directly as it is fast enough and up to date.
    return Scanning->getRequiredModules(File);
  }

  std::string getModuleNameForSource(PathRef File) override {
    // Return scanning results directly as it is fast enough and up to date.
    return Scanning->getModuleNameForSource(File);
  }

  std::string getSourceForModuleName(llvm::StringRef ModuleName,
                                     PathRef RequiredSourceFile) override {
    auto FromCompileCommands =
        CompileCommands->getSourceForModuleName(ModuleName, RequiredSourceFile);
    // Check if the source still declares the module.
    // This is to validate compile-command-derived results may be stale and
    // scan a single file is fast enough. We just don't want to scan the project
    // entirely.
    if (!FromCompileCommands.empty() &&
        Scanning->getModuleNameForSource(FromCompileCommands) == ModuleName)
      return FromCompileCommands;

    return Scanning->getSourceForModuleName(ModuleName, RequiredSourceFile);
  }

  ModuleNameState getModuleNameState(llvm::StringRef ModuleName) override {
    auto FromCompileCommands = CompileCommands->getModuleNameState(ModuleName);
    if (FromCompileCommands != ModuleNameState::Unknown)
      return FromCompileCommands;
    return Scanning->getModuleNameState(ModuleName);
  }

  void setCommandMangler(CommandMangler Mangler) override {
    this->Mangler = std::move(Mangler);
    auto ForwardMangler = [this](tooling::CompileCommand &Command,
                                 PathRef CommandPath) {
      if (this->Mangler)
        this->Mangler(Command, CommandPath);
    };
    CompileCommands->setCommandMangler(ForwardMangler);
    Scanning->setCommandMangler(std::move(ForwardMangler));
  }

private:
  std::unique_ptr<CompileCommandsProjectModules> CompileCommands;
  std::unique_ptr<ScanningAllProjectModules> Scanning;
  CommandMangler Mangler;
};

/// Creates the project-modules facade used by clangd.
///
/// The implementation is intentionally layered:
///
///         CompoundProjectModules
///            /              \
///           v                v
/// CompileCommands      ScanningAllProjectModules
///   ProjectModules               |
///      |                         v
///      |               ModuleDependencyScanner
///      |
///      +-- preferred specifically for recovering the source file for a module
///      |     name in the context of a consumer TU, because compile commands
///      |     encode `module name -> BMI -> producer source`
///      |
///      +-- scanning remains fallback/validation for stale or missing data
///
/// - `CompileCommandsProjectModules` reads module relationships that the build
///   system already made explicit in compile commands. In particular, it uses
///   producer-side BMI output paths together with consumer-side
///   `-fmodule-file=<module>=<bmi>` entries to recover the module unit source a
///   TU actually depends on. This is the preferred source because it can
///   distinguish different module producers for the same module name when
///   different translation units reference different BMIs.
/// - `ScanningAllProjectModules` derives module information by scanning source
///   files. It is more expensive, but it can still answer queries that are not
///   present in compile commands and validate compile-command-derived results.
/// - `CompoundProjectModules` arbitrates between the two backends on a
///   per-query basis. Compile commands are especially valuable for
///   `getSourceForModuleName()` because they preserve the consumer TU's actual
///   `module name -> BMI` choice. Other queries may still fall back to, or be
///   validated by, scanning because compile-command information may be
///   incomplete or stale.
///
/// This split keeps the logic simple: compile commands provide precision when
/// available, while scanning preserves compatibility with projects that have
/// incomplete module information in their compilation database.
std::unique_ptr<ProjectModules> getProjectModules(
    std::shared_ptr<const clang::tooling::CompilationDatabase> CDB,
    const ThreadsafeFS &TFS) {
  return std::make_unique<CompoundProjectModules>(std::move(CDB), TFS);
}

} // namespace clang::clangd
