//===--- IndexUnitWriter.h - Index unit serialization ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_INDEXUNITWRITER_H
#define LLVM_CLANG_INDEX_INDEXUNITWRITER_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include <string>
#include <vector>

namespace llvm {
  class BitstreamWriter;
}

namespace clang {
  class FileEntry;
  class FileManager;
  class PathRemapper;

namespace index {

namespace writer {
/// An opaque pointer to a module used by the IndexUnitWriter to associate
/// record and file dependencies with a module, and as a token for getting
/// information about the module from the caller.
typedef const void *OpaqueModule;

/// Module info suitable for serialization.
///
/// This is used for top-level modules and sub-modules.
struct ModuleInfo {
  /// Full, dot-separate, module name.
  StringRef Name;
};

typedef llvm::function_ref<ModuleInfo(OpaqueModule, SmallVectorImpl<char> &Scratch)>
    ModuleInfoWriterCallback;
} // end namespace writer

class IndexUnitWriter {
  FileManager &FileMgr;
  SmallString<64> UnitsPath;
  std::string ProviderIdentifier;
  std::string ProviderVersion;
  std::string OutputFile;
  std::string ModuleName;
  const FileEntry *MainFile;
  bool IsSystemUnit;
  bool IsModuleUnit;
  bool IsDebugCompilation;
  std::string TargetTriple;
  std::string WorkDir;
  std::string SysrootPath;
  const PathRemapper &Remapper;
  std::function<writer::ModuleInfo(writer::OpaqueModule,
                            SmallVectorImpl<char> &Scratch)> GetInfoForModuleFn;
  struct FileInclude {
    int Index;
    unsigned Line;
  };
  struct FileEntryData {
    const FileEntry *File;
    bool IsSystem;
    int ModuleIndex;
    std::vector<FileInclude> Includes;
  };
  std::vector<FileEntryData> Files;
  std::vector<writer::OpaqueModule> Modules;
  llvm::DenseMap<const FileEntry *, int> IndexByFile;
  llvm::DenseMap<writer::OpaqueModule, int> IndexByModule;
  llvm::DenseSet<const FileEntry *> SeenASTFiles;
  struct RecordOrUnitData {
    std::string Name;
    int FileIndex;
    int ModuleIndex;
    bool IsSystem;
  };
  std::vector<RecordOrUnitData> Records;
  std::vector<RecordOrUnitData> ASTFileUnits;

public:
  /// \param MainFile the main file for a compiled source file. This should be
  /// null for PCH and module units.
  /// \param IsSystem true for system module units, false otherwise.
  /// \param Remapper Remapper to use to standardize file paths to make them
  /// hermetic/reproducible. This applies to all paths emitted in the unit file.
  IndexUnitWriter(FileManager &FileMgr,
                  StringRef StorePath,
                  StringRef ProviderIdentifier, StringRef ProviderVersion,
                  StringRef OutputFile,
                  StringRef ModuleName,
                  const FileEntry *MainFile,
                  bool IsSystem,
                  bool IsModuleUnit,
                  bool IsDebugCompilation,
                  StringRef TargetTriple,
                  StringRef SysrootPath,
                  const PathRemapper &Remapper,
                  writer::ModuleInfoWriterCallback GetInfoForModule);
  ~IndexUnitWriter();

  int addFileDependency(const FileEntry *File, bool IsSystem,
                        writer::OpaqueModule Mod);
  void addRecordFile(StringRef RecordFile, const FileEntry *File, bool IsSystem,
                     writer::OpaqueModule Mod);
  void addASTFileDependency(const FileEntry *File, bool IsSystem,
                            writer::OpaqueModule Mod, bool withoutUnitName = false);
  void addUnitDependency(StringRef UnitFile, const FileEntry *File, bool IsSystem,
                         writer::OpaqueModule Mod);
  bool addInclude(const FileEntry *Source, unsigned Line, const FileEntry *Target);

  bool write(std::string &Error);

  void getUnitNameForOutputFile(StringRef FilePath, SmallVectorImpl<char> &Str);
  void getUnitPathForOutputFile(StringRef FilePath, SmallVectorImpl<char> &Str);
  /// If the unit file exists and \p timeCompareFilePath is provided, it will
  /// return true if \p timeCompareFilePath is older than the unit file.
  Optional<bool> isUnitUpToDateForOutputFile(StringRef FilePath,
                                             Optional<StringRef> TimeCompareFilePath,
                                             std::string &Error);
  static void getUnitNameForAbsoluteOutputFile(StringRef FilePath, SmallVectorImpl<char> &Str,
                                               const PathRemapper &Remapper);
  static bool initIndexDirectory(StringRef StorePath, std::string &Error);

private:
  class PathStorage;
  int addModule(writer::OpaqueModule Mod);
  void writeUnitInfo(llvm::BitstreamWriter &Stream, PathStorage &PathStore);
  void writeDependencies(llvm::BitstreamWriter &Stream, PathStorage &PathStore);
  void writeIncludes(llvm::BitstreamWriter &Stream, PathStorage &PathStore);
  void writePaths(llvm::BitstreamWriter &Stream, PathStorage &PathStore);
  void writeModules(llvm::BitstreamWriter &Stream);
};

} // end namespace index
} // end namespace clang

#endif
