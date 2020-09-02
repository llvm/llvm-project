//===- HeaderSearchOptions.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_HEADERSEARCHOPTIONS_H
#define LLVM_CLANG_LEX_HEADERSEARCHOPTIONS_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace clang {

namespace frontend {

/// IncludeDirGroup - Identifies the group an include Entry belongs to,
/// representing its relative positive in the search list.
/// \#include directives whose paths are enclosed by string quotes ("")
/// start searching at the Quoted group (specified by '-iquote'),
/// then search the Angled group, then the System group, etc.
enum IncludeDirGroup {
  /// '\#include ""' paths, added by 'gcc -iquote'.
  Quoted = 0,

  /// Paths for '\#include <>' added by '-I'.
  Angled,

  /// Like Angled, but marks header maps used when building frameworks.
  IndexHeaderMap,

  /// Like Angled, but marks system directories.
  System,

  /// Like System, but headers are implicitly wrapped in extern "C".
  ExternCSystem,

  /// Like System, but only used for C.
  CSystem,

  /// Like System, but only used for C++.
  CXXSystem,

  /// Like System, but only used for ObjC.
  ObjCSystem,

  /// Like System, but only used for ObjC++.
  ObjCXXSystem,

  /// Like System, but searched after the system directories.
  After
};

} // namespace frontend

/// HeaderSearchOptions - Helper class for storing options related to the
/// initialization of the HeaderSearch object.
class HeaderSearchOptions {
public:
  struct Entry {
    std::string Path;
    frontend::IncludeDirGroup Group;
    unsigned IsFramework : 1;

    /// IgnoreSysRoot - This is false if an absolute path should be treated
    /// relative to the sysroot, or true if it should always be the absolute
    /// path.
    unsigned IgnoreSysRoot : 1;

    Entry(StringRef path, frontend::IncludeDirGroup group, bool isFramework,
          bool ignoreSysRoot)
        : Path(path), Group(group), IsFramework(isFramework),
          IgnoreSysRoot(ignoreSysRoot) {}
  };

  struct SystemHeaderPrefix {
    /// A prefix to be matched against paths in \#include directives.
    std::string Prefix;

    /// True if paths beginning with this prefix should be treated as system
    /// headers.
    bool IsSystemHeader;

    SystemHeaderPrefix(StringRef Prefix, bool IsSystemHeader)
        : Prefix(Prefix), IsSystemHeader(IsSystemHeader) {}
  };

  using PrebuiltModuleFilesTy = std::map<std::string, std::string, std::less<>>;

  using ModulesIgnoreMacrosTy =
      llvm::SmallSetVector<llvm::CachedHashString, 16>;

#define HEADERSEARCHOPT(Name, Bits, Description) unsigned Name : Bits;
#define TYPED_HEADERSEARCHOPT(Type, Name, Description) Type Name;
#include "clang/Lex/HeaderSearchOptions.def"

  HeaderSearchOptions(StringRef _Sysroot = "/")
      : Sysroot(_Sysroot), ModuleFormat("raw"), DisableModuleHash(false),
        ImplicitModuleMaps(false), ModuleMapFileHomeIsCwd(false),
        UseBuiltinIncludes(true), UseStandardSystemIncludes(true),
        UseStandardCXXIncludes(true), UseLibcxx(false), Verbose(false),
        ModulesValidateOncePerBuildSession(false),
        ModulesValidateSystemHeaders(false),
        ValidateASTInputFilesContent(false), UseDebugInfo(false),
        ModulesValidateDiagnosticOptions(true), ModulesHashContent(false),
        ModulesStrictContextHash(false),
        ModuleCachePruneInterval(7 * 24 * 60 * 60),
        ModuleCachePruneAfter(31 * 24 * 60 * 60), BuildSessionTimestamp(0) {}

  /// AddPath - Add the \p Path path to the specified \p Group list.
  void AddPath(StringRef Path, frontend::IncludeDirGroup Group,
               bool IsFramework, bool IgnoreSysRoot) {
    UserEntries.emplace_back(Path, Group, IsFramework, IgnoreSysRoot);
  }

  /// AddSystemHeaderPrefix - Override whether \#include directives naming a
  /// path starting with \p Prefix should be considered as naming a system
  /// header.
  void AddSystemHeaderPrefix(StringRef Prefix, bool IsSystemHeader) {
    SystemHeaderPrefixes.emplace_back(Prefix, IsSystemHeader);
  }

  void AddVFSOverlayFile(StringRef Name) {
    VFSOverlayFiles.push_back(std::string(Name));
  }

  void AddPrebuiltModulePath(StringRef Name) {
    PrebuiltModulePaths.push_back(std::string(Name));
  }
};

inline llvm::hash_code hash_value(const HeaderSearchOptions::Entry &E) {
  return llvm::hash_combine(E.Path, E.Group, E.IsFramework, E.IgnoreSysRoot);
}

inline llvm::hash_code
hash_value(const HeaderSearchOptions::SystemHeaderPrefix &SHP) {
  return llvm::hash_combine(SHP.Prefix, SHP.IsSystemHeader);
}

} // namespace clang

#endif // LLVM_CLANG_LEX_HEADERSEARCHOPTIONS_H
