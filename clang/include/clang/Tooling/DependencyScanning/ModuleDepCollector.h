//===- ModuleDepCollector.h - Callbacks to collect deps ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_MODULE_DEP_COLLECTOR_H
#define LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_MODULE_DEP_COLLECTOR_H

#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <unordered_map>

namespace clang {
namespace tooling {
namespace dependencies {

class DependencyConsumer;

/// Modular dependency that has already been built prior to the dependency scan.
struct PrebuiltModuleDep {
  std::string ModuleName;
  std::string PCMFile;
  std::string ModuleMapFile;

  explicit PrebuiltModuleDep(const Module *M)
      : ModuleName(M->getTopLevelModuleName()),
        PCMFile(M->getASTFile()->getName()),
        ModuleMapFile(M->PresumedModuleMapFile) {}
};

/// This is used to identify a specific module.
struct ModuleID {
  /// The name of the module. This may include `:` for C++20 module partitions,
  /// or a header-name for C++20 header units.
  std::string ModuleName;

  /// The context hash of a module represents the set of compiler options that
  /// may make one version of a module incompatible with another. This includes
  /// things like language mode, predefined macros, header search paths, etc...
  ///
  /// Modules with the same name but a different \c ContextHash should be
  /// treated as separate modules for the purpose of a build.
  std::string ContextHash;

  bool operator==(const ModuleID &Other) const {
    return ModuleName == Other.ModuleName && ContextHash == Other.ContextHash;
  }
};

struct ModuleIDHasher {
  std::size_t operator()(const ModuleID &MID) const {
    return llvm::hash_combine(MID.ModuleName, MID.ContextHash);
  }
};

struct ModuleDeps {
  /// The identifier of the module.
  ModuleID ID;

  /// Whether this is a "system" module.
  bool IsSystem;

  /// The path to the modulemap file which defines this module.
  ///
  /// This can be used to explicitly build this module. This file will
  /// additionally appear in \c FileDeps as a dependency.
  std::string ClangModuleMapFile;

  /// The path to where an implicit build would put the PCM for this module.
  std::string ImplicitModulePCMPath;

  /// A collection of absolute paths to files that this module directly depends
  /// on, not including transitive dependencies.
  llvm::StringSet<> FileDeps;

  /// A collection of prebuilt modular dependencies this module directly depends
  /// on, not including transitive dependencies.
  std::vector<PrebuiltModuleDep> PrebuiltModuleDeps;

  /// A list of module identifiers this module directly depends on, not
  /// including transitive dependencies.
  ///
  /// This may include modules with a different context hash when it can be
  /// determined that the differences are benign for this compilation.
  std::vector<ModuleID> ClangModuleDeps;

  // Used to track which modules that were discovered were directly imported by
  // the primary TU.
  bool ImportedByMainFile = false;

  /// Compiler invocation that can be used to build this module (without paths).
  CompilerInvocation Invocation;

  /// Gets the canonical command line suitable for passing to clang.
  ///
  /// \param LookupPCMPath This function is called to fill in "-fmodule-file="
  ///                      arguments and the "-o" argument. It needs to return
  ///                      a path for where the PCM for the given module is to
  ///                      be located.
  /// \param LookupModuleDeps This function is called to collect the full
  ///                         transitive set of dependencies for this
  ///                         compilation and fill in "-fmodule-map-file="
  ///                         arguments.
  std::vector<std::string> getCanonicalCommandLine(
      std::function<StringRef(ModuleID)> LookupPCMPath,
      std::function<const ModuleDeps &(ModuleID)> LookupModuleDeps) const;

  /// Gets the canonical command line suitable for passing to clang, excluding
  /// arguments containing modules-related paths: "-fmodule-file=", "-o",
  /// "-fmodule-map-file=".
  std::vector<std::string> getCanonicalCommandLineWithoutModulePaths() const;

  /// Gets additional arguments suitable for appending to the original Clang
  /// command line, excluding arguments containing modules-related paths:
  /// "-fmodule-file=", "-o", "-fmodule-map-file=".
  std::vector<std::string> getAdditionalArgsWithoutModulePaths() const;
};

namespace detail {
/// Collect the paths of PCM and module map files for the modules in \c Modules
/// transitively.
void collectPCMAndModuleMapPaths(
    llvm::ArrayRef<ModuleID> Modules,
    std::function<StringRef(ModuleID)> LookupPCMPath,
    std::function<const ModuleDeps &(ModuleID)> LookupModuleDeps,
    std::vector<std::string> &PCMPaths, std::vector<std::string> &ModMapPaths);
} // namespace detail

class ModuleDepCollector;

/// Callback that records textual includes and direct modular includes/imports
/// during preprocessing. At the end of the main file, it also collects
/// transitive modular dependencies and passes everything to the
/// \c DependencyConsumer of the parent \c ModuleDepCollector.
class ModuleDepCollectorPP final : public PPCallbacks {
public:
  ModuleDepCollectorPP(ModuleDepCollector &MDC) : MDC(MDC) {}

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override;
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override;
  void moduleImport(SourceLocation ImportLoc, ModuleIdPath Path,
                    const Module *Imported) override;

  void EndOfMainFile() override;

private:
  /// The parent dependency collector.
  ModuleDepCollector &MDC;
  /// Working set of direct modular dependencies.
  llvm::DenseSet<const Module *> DirectModularDeps;
  /// Working set of direct modular dependencies that have already been built.
  llvm::DenseSet<const Module *> DirectPrebuiltModularDeps;

  void handleImport(const Module *Imported);

  /// Adds direct modular dependencies that have already been built to the
  /// ModuleDeps instance.
  void
  addAllSubmodulePrebuiltDeps(const Module *M, ModuleDeps &MD,
                              llvm::DenseSet<const Module *> &SeenSubmodules);
  void addModulePrebuiltDeps(const Module *M, ModuleDeps &MD,
                             llvm::DenseSet<const Module *> &SeenSubmodules);

  /// Traverses the previously collected direct modular dependencies to discover
  /// transitive modular dependencies and fills the parent \c ModuleDepCollector
  /// with both.
  ModuleID handleTopLevelModule(const Module *M);
  void addAllSubmoduleDeps(const Module *M, ModuleDeps &MD,
                           llvm::DenseSet<const Module *> &AddedModules);
  void addModuleDep(const Module *M, ModuleDeps &MD,
                    llvm::DenseSet<const Module *> &AddedModules);
};

/// Collects modular and non-modular dependencies of the main file by attaching
/// \c ModuleDepCollectorPP to the preprocessor.
class ModuleDepCollector final : public DependencyCollector {
public:
  ModuleDepCollector(std::unique_ptr<DependencyOutputOptions> Opts,
                     CompilerInstance &ScanInstance, DependencyConsumer &C,
                     CompilerInvocation &&OriginalCI, bool OptimizeArgs);

  void attachToPreprocessor(Preprocessor &PP) override;
  void attachToASTReader(ASTReader &R) override;

private:
  friend ModuleDepCollectorPP;

  /// The compiler instance for scanning the current translation unit.
  CompilerInstance &ScanInstance;
  /// The consumer of collected dependency information.
  DependencyConsumer &Consumer;
  /// Path to the main source file.
  std::string MainFile;
  /// Hash identifying the compilation conditions of the current TU.
  std::string ContextHash;
  /// Non-modular file dependencies. This includes the main source file and
  /// textually included header files.
  std::vector<std::string> FileDeps;
  /// Direct and transitive modular dependencies of the main source file.
  std::unordered_map<const Module *, ModuleDeps> ModularDeps;
  /// Options that control the dependency output generation.
  std::unique_ptr<DependencyOutputOptions> Opts;
  /// The original Clang invocation passed to dependency scanner.
  CompilerInvocation OriginalInvocation;
  /// Whether to optimize the modules' command-line arguments.
  bool OptimizeArgs;

  /// Checks whether the module is known as being prebuilt.
  bool isPrebuiltModule(const Module *M);

  /// Constructs a CompilerInvocation that can be used to build the given
  /// module, excluding paths to discovered modular dependencies that are yet to
  /// be built.
  CompilerInvocation makeInvocationForModuleBuildWithoutPaths(
      const ModuleDeps &Deps,
      llvm::function_ref<void(CompilerInvocation &)> Optimize) const;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_MODULE_DEP_COLLECTOR_H
