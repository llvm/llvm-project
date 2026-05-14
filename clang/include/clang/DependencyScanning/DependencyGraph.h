//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYGRAPH_H
#define LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYGRAPH_H

#include "clang/Basic/LLVM.h"
#include "clang/Basic/Module.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <variant>
#include <vector>

namespace clang::dependencies {
/// Modular dependency that has already been built prior to the dependency scan.
struct PrebuiltModuleDep {
  std::string ModuleName;
  std::string PCMFile;
  std::string ModuleMapFile;
};

/// This is used to identify a specific module.
struct ModuleID {
  /// The name of the module. This may include `:` for C++20 module partitions,
  /// or a header-name for C++20 header units.
  std::string ModuleName;

  /// The context hash of a module represents the compiler options that affect
  /// the resulting command-line invocation.
  ///
  /// Modules with the same name and ContextHash but different invocations could
  /// cause non-deterministic build results.
  ///
  /// Modules with the same name but a different \c ContextHash should be
  /// treated as separate modules for the purpose of a build.
  std::string ContextHash;

  bool operator==(const ModuleID &Other) const {
    return std::tie(ModuleName, ContextHash) ==
           std::tie(Other.ModuleName, Other.ContextHash);
  }

  bool operator<(const ModuleID &Other) const {
    return std::tie(ModuleName, ContextHash) <
           std::tie(Other.ModuleName, Other.ContextHash);
  }
};

/// P1689ModuleInfo - Represents the needed information of standard C++20
/// modules for P1689 format.
struct P1689ModuleInfo {
  /// The name of the module. This may include `:` for partitions.
  std::string ModuleName;

  /// Optional. The source path to the module.
  std::string SourcePath;

  /// If this module is a standard c++ interface unit.
  bool IsStdCXXModuleInterface = true;

  enum class ModuleType {
    NamedCXXModule
    // To be supported
    // AngleHeaderUnit,
    // QuoteHeaderUnit
  };
  ModuleType Type = ModuleType::NamedCXXModule;
};

struct ModuleDeps {
  /// The identifier of the module.
  ModuleID ID;

  /// Whether this is a "system" module.
  bool IsSystem;

  /// Whether this module is fully composed of file & module inputs from
  /// locations likely to stay the same across the active development and build
  /// cycle. For example, when all those input paths only resolve in Sysroot.
  ///
  /// External paths, as opposed to virtual file paths, are always used
  /// for computing this value.
  bool IsInStableDirectories;

  /// Whether current working directory is ignored.
  bool IgnoreCWD;

  /// The path to the modulemap file which defines this module.
  ///
  /// This can be used to explicitly build this module. This file will
  /// additionally appear in \c FileDeps as a dependency.
  std::string ClangModuleMapFile;

  /// A collection of absolute paths to module map files that this module needs
  /// to know about. The ordering is significant.
  std::vector<std::string> ModuleMapFileDeps;

  /// A collection of prebuilt modular dependencies this module directly depends
  /// on, not including transitive dependencies.
  std::vector<PrebuiltModuleDep> PrebuiltModuleDeps;

  /// A list of module identifiers this module directly depends on, not
  /// including transitive dependencies.
  ///
  /// This may include modules with a different context hash when it can be
  /// determined that the differences are benign for this compilation.
  std::vector<ModuleID> ClangModuleDeps;

  /// The set of libraries or frameworks to link against when
  /// an entity from this module is used.
  llvm::SmallVector<Module::LinkLibrary, 2> LinkLibraries;

  /// Invokes \c Cb for all file dependencies of this module. Each provided
  /// \c StringRef is only valid within the individual callback invocation.
  void forEachFileDep(llvm::function_ref<void(StringRef)> Cb) const;

  /// Get (or compute) the compiler invocation that can be used to build this
  /// module. Does not include argv[0].
  const std::vector<std::string> &getBuildArguments() const;

private:
  friend class ModuleDepCollector;
  friend class ModuleDepCollectorPP;

  /// The absolute directory path that is the base for relative paths
  /// in \c FileDeps.
  std::string FileDepsBaseDir;

  /// A collection of paths to files that this module directly depends on, not
  /// including transitive dependencies.
  std::vector<std::string> FileDeps;

  mutable std::variant<std::monostate, CowCompilerInvocation,
                       std::vector<std::string>>
      BuildInfo;
};

/// A command-line tool invocation that is part of building a TU.
///
/// \see TranslationUnitDeps::Commands.
struct Command {
  std::string Executable;
  std::vector<std::string> Arguments;
};

/// Graph of modular dependencies.
using ModuleDepsGraph = std::vector<ModuleDeps>;

/// The full dependencies and module graph for a specific input.
struct TranslationUnitDeps {
  /// The graph of direct and transitive modular dependencies.
  ModuleDepsGraph ModuleGraph;

  /// The identifier of the C++20 module this translation unit exports.
  ///
  /// If the translation unit is not a module then \c ID.ModuleName is empty.
  ModuleID ID;

  /// A collection of absolute paths to files that this translation unit
  /// directly depends on, not including transitive dependencies.
  std::vector<std::string> FileDeps;

  /// A collection of prebuilt modules this translation unit directly depends
  /// on, not including transitive dependencies.
  std::vector<PrebuiltModuleDep> PrebuiltModuleDeps;

  /// A list of modules this translation unit directly depends on, not including
  /// transitive dependencies.
  ///
  /// This may include modules with a different context hash when it can be
  /// determined that the differences are benign for this compilation.
  std::vector<ModuleID> ClangModuleDeps;

  /// A list of module names that are visible to this translation unit. This
  /// includes both direct and transitive module dependencies.
  std::vector<std::string> VisibleModules;

  /// A list of the C++20 named modules this translation unit depends on.
  std::vector<std::string> NamedModuleDeps;

  /// The sequence of commands required to build the translation unit. Commands
  /// should be executed in order.
  ///
  /// FIXME: If we add support for multi-arch builds in clang-scan-deps, we
  /// should make the dependencies between commands explicit to enable parallel
  /// builds of each architecture.
  std::vector<Command> Commands;

  /// Deprecated driver command-line. This will be removed in a future version.
  std::vector<std::string> DriverCommandLine;
};
} // namespace clang::dependencies

namespace llvm {
inline hash_code hash_value(const clang::dependencies::ModuleID &ID) {
  return hash_combine(ID.ModuleName, ID.ContextHash);
}

template <> struct DenseMapInfo<clang::dependencies::ModuleID> {
  using ModuleID = clang::dependencies::ModuleID;
  static inline ModuleID getEmptyKey() { return ModuleID{"", ""}; }
  static inline ModuleID getTombstoneKey() {
    return ModuleID{"~", "~"}; // ~ is not a valid module name or context hash
  }
  static unsigned getHashValue(const ModuleID &ID) { return hash_value(ID); }
  static bool isEqual(const ModuleID &LHS, const ModuleID &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

#endif // LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYGRAPH_H
