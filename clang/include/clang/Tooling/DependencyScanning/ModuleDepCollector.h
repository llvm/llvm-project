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

/// This is used to refer to a specific module.
///
/// See \c ModuleDeps for details about what these members mean.
struct ClangModuleDep {
  std::string ModuleName;
  std::string ContextHash;
};

struct ModuleDeps {
  /// The name of the module. This may include `:` for C++20 module partitons,
  /// or a header-name for C++20 header units.
  std::string ModuleName;

  /// The context hash of a module represents the set of compiler options that
  /// may make one version of a module incompatible with another. This includes
  /// things like language mode, predefined macros, header search paths, etc...
  ///
  /// Modules with the same name but a different \c ContextHash should be
  /// treated as separate modules for the purpose of a build.
  std::string ContextHash;

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

  /// A list of modules this module directly depends on, not including
  /// transitive dependencies.
  ///
  /// This may include modules with a different context hash when it can be
  /// determined that the differences are benign for this compilation.
  std::vector<ClangModuleDep> ClangModuleDeps;

  /// A partial command line that can be used to build this module.
  ///
  /// Call \c getFullCommandLine() to get a command line suitable for passing to
  /// clang.
  std::vector<std::string> NonPathCommandLine;

  // Used to track which modules that were discovered were directly imported by
  // the primary TU.
  bool ImportedByMainFile = false;

  /// Gets the full command line suitable for passing to clang.
  ///
  /// \param LookupPCMPath this function is called to fill in `-fmodule-file=`
  ///                      flags and for the `-o` flag. It needs to return a
  ///                      path for where the PCM for the given module is to
  ///                      be located.
  /// \param LookupModuleDeps this fucntion is called to collect the full
  ///                         transitive set of dependencies for this
  ///                         compilation.
  std::vector<std::string> getFullCommandLine(
      std::function<StringRef(ClangModuleDep)> LookupPCMPath,
      std::function<const ModuleDeps &(ClangModuleDep)> LookupModuleDeps) const;
};

namespace detail {
/// Append the `-fmodule-file=` and `-fmodule-map-file=` arguments for the
/// modules in \c Modules transitively, along with other needed arguments to
/// use explicitly built modules.
void appendCommonModuleArguments(
    llvm::ArrayRef<ClangModuleDep> Modules,
    std::function<StringRef(ClangModuleDep)> LookupPCMPath,
    std::function<const ModuleDeps &(ClangModuleDep)> LookupModuleDeps,
    std::vector<std::string> &Result);
} // namespace detail

class ModuleDepCollector;

class ModuleDepCollectorPP final : public PPCallbacks {
public:
  ModuleDepCollectorPP(CompilerInstance &I, ModuleDepCollector &MDC)
      : Instance(I), MDC(MDC) {}

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
  CompilerInstance &Instance;
  ModuleDepCollector &MDC;
  llvm::DenseSet<const Module *> DirectDeps;

  void handleImport(const Module *Imported);
  void handleTopLevelModule(const Module *M);
  void addAllSubmoduleDeps(const Module *M, ModuleDeps &MD,
                           llvm::DenseSet<const Module *> &AddedModules);
  void addModuleDep(const Module *M, ModuleDeps &MD,
                    llvm::DenseSet<const Module *> &AddedModules);
};

class ModuleDepCollector final : public DependencyCollector {
public:
  ModuleDepCollector(std::unique_ptr<DependencyOutputOptions> Opts,
                     CompilerInstance &I, DependencyConsumer &C);

  void attachToPreprocessor(Preprocessor &PP) override;
  void attachToASTReader(ASTReader &R) override;

private:
  friend ModuleDepCollectorPP;

  CompilerInstance &Instance;
  DependencyConsumer &Consumer;
  std::string MainFile;
  std::string ContextHash;
  std::vector<std::string> MainDeps;
  std::unordered_map<std::string, ModuleDeps> Deps;
  std::unique_ptr<DependencyOutputOptions> Opts;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_MODULE_DEP_COLLECTOR_H
