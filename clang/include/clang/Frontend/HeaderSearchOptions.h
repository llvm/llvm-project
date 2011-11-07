//===--- HeaderSearchOptions.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_HEADERSEARCHOPTIONS_H
#define LLVM_CLANG_FRONTEND_HEADERSEARCHOPTIONS_H

#include "llvm/ADT/StringRef.h"
#include <vector>

namespace clang {

namespace frontend {
  /// IncludeDirGroup - Identifiers the group a include entry belongs to, which
  /// represents its relative positive in the search list.  A #include of a ""
  /// path starts at the -iquote group, then searches the Angled group, then
  /// searches the system group, etc.
  enum IncludeDirGroup {
    Quoted = 0,     ///< '#include ""' paths, added by'gcc -iquote'.
    Angled,         ///< Paths for '#include <>' added by '-I'.
    IndexHeaderMap, ///< Like Angled, but marks header maps used when
                       ///  building frameworks.
    System,         ///< Like Angled, but marks system directories.
    CSystem,        ///< Like System, but only used for C.
    CXXSystem,      ///< Like System, but only used for C++.
    ObjCSystem,     ///< Like System, but only used for ObjC.
    ObjCXXSystem,   ///< Like System, but only used for ObjC++.
    After           ///< Like System, but searched after the system directories.
  };
}

/// HeaderSearchOptions - Helper class for storing options related to the
/// initialization of the HeaderSearch object.
class HeaderSearchOptions {
public:
  struct Entry {
    std::string Path;
    frontend::IncludeDirGroup Group;
    unsigned IsUserSupplied : 1;
    unsigned IsFramework : 1;
    
    /// IgnoreSysRoot - This is false if an absolute path should be treated
    /// relative to the sysroot, or true if it should always be the absolute
    /// path.
    unsigned IgnoreSysRoot : 1;

    /// \brief True if this entry is an internal search path.
    ///
    /// This typically indicates that users didn't directly provide it, but
    /// instead it was provided by a compatibility layer for a particular
    /// system. This isn't redundant with IsUserSupplied (even though perhaps
    /// it should be) because that is false for user provided '-iwithprefix'
    /// header search entries.
    unsigned IsInternal : 1;

    /// \brief True if this entry's headers should be wrapped in extern "C".
    unsigned ImplicitExternC : 1;

    Entry(StringRef path, frontend::IncludeDirGroup group,
          bool isUserSupplied, bool isFramework, bool ignoreSysRoot,
          bool isInternal, bool implicitExternC)
      : Path(path), Group(group), IsUserSupplied(isUserSupplied),
        IsFramework(isFramework), IgnoreSysRoot(ignoreSysRoot),
        IsInternal(isInternal), ImplicitExternC(implicitExternC) {}
  };

  /// If non-empty, the directory to use as a "virtual system root" for include
  /// paths.
  std::string Sysroot;

  /// User specified include entries.
  std::vector<Entry> UserEntries;

  /// The directory which holds the compiler resource files (builtin includes,
  /// etc.).
  std::string ResourceDir;

  /// \brief The directory used for the module cache.
  std::string ModuleCachePath;
  
  /// \brief Whether we should disable the use of the hash string within the
  /// module cache.
  ///
  /// Note: Only used for testing!
  unsigned DisableModuleHash : 1;
  
  /// Include the compiler builtin includes.
  unsigned UseBuiltinIncludes : 1;

  /// Include the system standard include search directories.
  unsigned UseStandardSystemIncludes : 1;

  /// Include the system standard C++ library include search directories.
  unsigned UseStandardCXXIncludes : 1;

  /// Use libc++ instead of the default libstdc++.
  unsigned UseLibcxx : 1;

  /// Whether header search information should be output as for -v.
  unsigned Verbose : 1;

public:
  HeaderSearchOptions(StringRef _Sysroot = "/")
    : Sysroot(_Sysroot), DisableModuleHash(0), UseBuiltinIncludes(true),
      UseStandardSystemIncludes(true), UseStandardCXXIncludes(true),
      UseLibcxx(false), Verbose(false) {}

  /// AddPath - Add the \arg Path path to the specified \arg Group list.
  void AddPath(StringRef Path, frontend::IncludeDirGroup Group,
               bool IsUserSupplied, bool IsFramework, bool IgnoreSysRoot,
               bool IsInternal = false, bool ImplicitExternC = false) {
    UserEntries.push_back(Entry(Path, Group, IsUserSupplied, IsFramework,
                                IgnoreSysRoot, IsInternal, ImplicitExternC));
  }
};

} // end namespace clang

#endif
