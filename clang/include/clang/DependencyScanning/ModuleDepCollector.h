//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DEPENDENCYSCANNING_MODULEDEPCOLLECTOR_H
#define LLVM_CLANG_DEPENDENCYSCANNING_MODULEDEPCOLLECTOR_H

#include "clang/Basic/LLVM.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "clang/DependencyScanning/DependencyGraph.h"
#include "clang/DependencyScanning/DependencyScanningService.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace clang {
namespace dependencies {

class DependencyActionController;
class DependencyConsumer;
class PrebuiltModuleASTAttrs;

/// Attributes loaded from AST files of prebuilt modules collected prior to
/// ModuleDepCollector creation.
using PrebuiltModulesAttrsMap = llvm::StringMap<PrebuiltModuleASTAttrs>;
class PrebuiltModuleASTAttrs {
public:
  /// When a module is discovered to not be in stable directories, traverse &
  /// update all modules that depend on it.
  void
  updateDependentsNotInStableDirs(PrebuiltModulesAttrsMap &PrebuiltModulesMap);

  /// Read-only access to whether the module is made up of dependencies in
  /// stable directories.
  bool isInStableDir() const { return IsInStableDirs; }

  /// Read-only access to vfs map files.
  const llvm::StringSet<> &getVFS() const { return VFSMap; }

  /// Update the VFSMap to the one discovered from serializing the AST file.
  void setVFS(llvm::StringSet<> &&VFS) { VFSMap = std::move(VFS); }

  /// Add a direct dependent module file, so it can be updated if the current
  /// module is from stable directores.
  void addDependent(StringRef ModuleFile) {
    ModuleFileDependents.insert(ModuleFile);
  }

  /// Update whether the prebuilt module resolves entirely in a stable
  /// directories.
  void setInStableDir(bool V = false) {
    // Cannot reset attribute once it's false.
    if (!IsInStableDirs)
      return;
    IsInStableDirs = V;
  }

private:
  llvm::StringSet<> VFSMap;
  bool IsInStableDirs = true;
  std::set<StringRef> ModuleFileDependents;
};

/// An output from a module compilation, such as the path of the module file.
enum class ModuleOutputKind {
  /// The module file (.pcm). Required.
  ModuleFile,
  /// The path of the dependency file (.d), if any.
  DependencyFile,
  /// The null-separated list of names to use as the targets in the dependency
  /// file, if any. Defaults to the value of \c ModuleFile, as in the driver.
  DependencyTargets,
  /// The path of the serialized diagnostic file (.dia), if any.
  DiagnosticSerializationFile,
};

class ModuleDepCollector;

/// Callback that records textual includes and direct modular includes/imports
/// during preprocessing. At the end of the main file, it also collects
/// transitive modular dependencies and passes everything to the
/// \c DependencyConsumer of the parent \c ModuleDepCollector.
class ModuleDepCollectorPP final : public PPCallbacks {
public:
  ModuleDepCollectorPP(ModuleDepCollector &MDC) : MDC(MDC) {}

  void LexedFileChanged(FileID FID, LexedFileChangeReason Reason,
                        SrcMgr::CharacteristicKind FileType, FileID PrevFID,
                        SourceLocation Loc) override;
  void HasInclude(SourceLocation Loc, StringRef FileName, bool IsAngled,
                  OptionalFileEntryRef File,
                  SrcMgr::CharacteristicKind FileType) override;
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *SuggestedModule,
                          bool ModuleImported,
                          SrcMgr::CharacteristicKind FileType) override;
  void moduleImport(SourceLocation ImportLoc, ModuleIdPath Path,
                    const Module *Imported) override;

  void EndOfMainFile() override;

private:
  /// The parent dependency collector.
  ModuleDepCollector &MDC;

  void handleImport(const Module *Imported);

  /// Returns the ID or nothing if the dependency is spurious and is ignored.
  std::optional<ModuleID> handleTopLevelModule(serialization::ModuleFile *MF);

  /// Adds direct module dependencies to the ModuleDeps instance. This includes
  /// prebuilt module and implicitly-built modules.
  void addAllModuleDeps(serialization::ModuleFile &MF, ModuleDeps &MD);
};

/// Collects modular and non-modular dependencies of the main file by attaching
/// \c ModuleDepCollectorPP to the preprocessor.
class ModuleDepCollector final : public DependencyCollector {
public:
  ModuleDepCollector(DependencyScanningService &Service,
                     std::unique_ptr<DependencyOutputOptions> Opts,
                     CompilerInstance &ScanInstance, DependencyConsumer &C,
                     DependencyActionController &Controller,
                     CompilerInvocation OriginalCI,
                     const PrebuiltModulesAttrsMap PrebuiltModulesASTMap,
                     const ArrayRef<StringRef> StableDirs);

  void attachToPreprocessor(Preprocessor &PP) override;
  void attachToASTReader(ASTReader &R) override;

  PPCallbacks *getPPCallbacks() { return CollectorPPPtr; }

  /// Apply any changes implied by the discovered dependencies to the given
  /// invocation, (e.g. disable implicit modules, add explicit module paths).
  void applyDiscoveredDependencies(CompilerInvocation &CI);

private:
  friend ModuleDepCollectorPP;

  /// The parent dependency scanning service.
  DependencyScanningService &Service;
  /// The compiler instance for scanning the current translation unit.
  CompilerInstance &ScanInstance;
  /// The consumer of collected dependency information.
  DependencyConsumer &Consumer;
  /// Callbacks for computing dependency information.
  DependencyActionController &Controller;
  /// Mapping from prebuilt AST filepaths to their attributes referenced during
  /// dependency collecting.
  const PrebuiltModulesAttrsMap PrebuiltModulesASTMap;
  /// Directory paths known to be stable through an active development and build
  /// cycle.
  const ArrayRef<StringRef> StableDirs;
  /// Path to the main source file.
  std::string MainFile;
  /// Non-modular file dependencies. This includes the main source file and
  /// textually included header files.
  std::vector<std::string> FileDeps;
  /// Direct and transitive modular dependencies of the main source file.
  llvm::MapVector<serialization::ModuleFile *, std::unique_ptr<ModuleDeps>>
      ModularDeps;
  /// Secondary mapping for \c ModularDeps allowing lookup by ModuleID without
  /// a preprocessor. Storage owned by \c ModularDeps.
  llvm::DenseMap<ModuleID, ModuleDeps *> ModuleDepsByID;
  /// Direct modular dependencies that have already been built.
  llvm::MapVector<serialization::ModuleFile *, PrebuiltModuleDep>
      DirectPrebuiltModularDeps;
  /// Working set of direct modular dependencies.
  llvm::SetVector<serialization::ModuleFile *> DirectModularDeps;
  /// Working set of direct modular dependencies, as they were imported.
  llvm::SmallPtrSet<const Module *, 32> DirectImports;
  /// All direct and transitive visible modules.
  llvm::StringSet<> VisibleModules;

  /// Options that control the dependency output generation.
  std::unique_ptr<DependencyOutputOptions> Opts;
  /// A Clang invocation that's based on the original TU invocation and that has
  /// been partially transformed into one that can perform explicit build of
  /// a discovered modular dependency. Note that this still needs to be adjusted
  /// for each individual module.
  CowCompilerInvocation CommonInvocation;

  std::optional<P1689ModuleInfo> ProvidedStdCXXModule;
  std::vector<P1689ModuleInfo> RequiredStdCXXModules;

  /// A pointer to the preprocessor callback so we can invoke it directly
  /// if needed. The callback is created and added to a Preprocessor instance by
  /// attachToPreprocessor and the Preprocessor instance owns it.
  ModuleDepCollectorPP *CollectorPPPtr = nullptr;

  /// Checks whether the module is known as being prebuilt.
  bool isPrebuiltModule(const serialization::ModuleFile *MF);

  /// Computes all visible modules resolved from direct imports.
  void addVisibleModules();

  /// Adds \p Path to \c FileDeps, making it absolute if necessary.
  void addFileDep(StringRef Path);
  /// Adds \p Path to \c MD.FileDeps, making it absolute if necessary.
  void addFileDep(ModuleDeps &MD, StringRef Path);

  /// Get a Clang invocation adjusted to build the given modular dependency.
  /// This excludes paths that are yet-to-be-provided by the build system.
  CowCompilerInvocation getInvocationAdjustedForModuleBuildWithoutOutputs(
      const ModuleDeps &Deps,
      llvm::function_ref<void(CowCompilerInvocation &)> Optimize) const;

  /// Collect module map files for given modules.
  llvm::DenseSet<const FileEntry *>
  collectModuleMapFiles(ArrayRef<ModuleID> ClangModuleDeps) const;

  /// Add module map files to the invocation, if needed.
  void addModuleMapFiles(CompilerInvocation &CI,
                         ArrayRef<ModuleID> ClangModuleDeps) const;
  /// Add module files (pcm) to the invocation, if needed.
  void addModuleFiles(CompilerInvocation &CI,
                      ArrayRef<ModuleID> ClangModuleDeps) const;
  void addModuleFiles(CowCompilerInvocation &CI,
                      ArrayRef<ModuleID> ClangModuleDeps) const;

  /// Add paths that require looking up outputs to the given dependencies.
  void addOutputPaths(CowCompilerInvocation &CI, ModuleDeps &Deps);

  /// Compute the context hash for \p Deps, and create the mapping
  /// \c ModuleDepsByID[Deps.ID] = &Deps.
  void associateWithContextHash(const CowCompilerInvocation &CI,
                                ModuleDeps &Deps);
};

/// Resets codegen options that don't affect modules/PCH.
void resetBenignCodeGenOptions(frontend::ActionKind ProgramAction,
                               const LangOptions &LangOpts,
                               CodeGenOptions &CGOpts);

/// Determine if \c Input can be resolved within a stable directory.
///
/// \param Directories Paths known to be in a stable location. e.g. Sysroot.
/// \param Input Path to evaluate.
bool isPathInStableDir(const ArrayRef<StringRef> Directories,
                       const StringRef Input);

/// Determine if options collected from a module's
/// compilation can safely be considered as stable.
///
/// \param Directories Paths known to be in a stable location. e.g. Sysroot.
/// \param HSOpts Header search options derived from the compiler invocation.
bool areOptionsInStableDir(const ArrayRef<StringRef> Directories,
                           const HeaderSearchOptions &HSOpts);

} // end namespace dependencies
} // end namespace clang

#endif // LLVM_CLANG_DEPENDENCYSCANNING_MODULEDEPCOLLECTOR_H
