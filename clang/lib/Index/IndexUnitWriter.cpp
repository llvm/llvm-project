//===--- IndexUnitWriter.cpp - Index unit serialization -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Index/IndexUnitWriter.h"
#include "IndexDataStoreUtils.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/PathRemapper.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::index;
using namespace clang::index::store;
using namespace llvm;

class IndexUnitWriter::PathStorage {
  FileManager &FileMgr;
  std::string WorkDir;
  std::string SysrootPath;
  SmallString<512> PathsBuf;
  StringMap<DirBitPath, BumpPtrAllocator> Dirs;
  std::vector<FileBitPath> FileBitPaths;
  DenseMap<const FileEntry *, size_t> FileToIndex;
  const PathRemapper &Remapper;

public:
  PathStorage(FileManager &fileMgr, StringRef workDir, StringRef sysrootPath,
      const PathRemapper &remapper) : FileMgr(fileMgr), Remapper(remapper) {
    WorkDir = std::string(workDir);
    if (sysrootPath == "/")
      sysrootPath = StringRef();
    SysrootPath = std::string(sysrootPath);
  }

  StringRef getPathsBuffer() const { return PathsBuf.str(); }

  ArrayRef<FileBitPath> getBitPaths() const { return FileBitPaths; }

  int getPathIndex(const FileEntry *FE) {
    if (!FE)
      return -1;
    auto Pair = FileToIndex.insert(std::make_pair(FE, FileBitPaths.size()));
    bool IsNew = Pair.second;
    size_t Index = Pair.first->getSecond();

    if (IsNew) {
      StringRef Filename = sys::path::filename(FE->getName());
      SmallString<256> AbsDirPath(sys::path::parent_path(FE->getName()));
      FileMgr.makeAbsolutePath(AbsDirPath);
      DirBitPath Dir = getDirBitPath(AbsDirPath.str());
      FileBitPaths.emplace_back(Dir.PrefixKind, Dir.Dir,
                                BitPathComponent(getPathOffset(Filename),
                                                 Filename.size()));
    }
    return Index;
  }

  size_t getPathOffset(StringRef Path) {
    if (Path.empty())
      return 0;
    size_t offset = PathsBuf.size();
    PathsBuf += Path;
    return offset;
  }
  
private:
  DirBitPath getDirBitPath(StringRef dirStr) {
    auto pair = Dirs.insert(std::make_pair(dirStr, DirBitPath()));
    bool isNew = pair.second;
    auto &dirPath = pair.first->second;

    if (isNew) {
      if (isPathInDir(SysrootPath, dirStr)) {
        dirPath.PrefixKind = UNIT_PATH_PREFIX_SYSROOT;
        dirStr = dirStr.drop_front(SysrootPath.size());
        while (!dirStr.empty() && dirStr[0] == '/')
          dirStr = dirStr.drop_front();
      } else if (isPathInDir(WorkDir, dirStr)) {
        dirPath.PrefixKind = UNIT_PATH_PREFIX_WORKDIR;
        dirStr = dirStr.drop_front(WorkDir.size());
        while (!dirStr.empty() && dirStr[0] == '/')
          dirStr = dirStr.drop_front();
      }

      if (dirPath.PrefixKind != UNIT_PATH_PREFIX_NONE) {
        // No need to remap since it's already relative to another directory.
        dirPath.Dir.Offset = getPathOffset(dirStr);
        dirPath.Dir.Size = dirStr.size();
      } else {  // Remap the path before storing.
        std::string Remapped = Remapper.remapPath(dirStr);
        dirPath.Dir.Offset = getPathOffset(Remapped);
        dirPath.Dir.Size = Remapped.size();
      }
    }
    return dirPath;
  }

  static bool isPathInDir(StringRef dir, StringRef path) {
    if (dir.empty() || !path.startswith(dir))
      return false;
    StringRef rest = path.drop_front(dir.size());
    return !rest.empty() && sys::path::is_separator(rest.front());
  }
};

IndexUnitWriter::IndexUnitWriter(FileManager &FileMgr,
                                 StringRef StorePath,
                                 StringRef ProviderIdentifier,
                                 StringRef ProviderVersion,
                                 StringRef OutputFile,
                                 StringRef ModuleName,
                                 const FileEntry *MainFile,
                                 bool IsSystem,
                                 bool IsModuleUnit,
                                 bool IsDebugCompilation,
                                 StringRef TargetTriple,
                                 StringRef SysrootPath,
                                 const PathRemapper &Remapper,
                                 writer::ModuleInfoWriterCallback GetInfoForModule)
: FileMgr(FileMgr), Remapper(Remapper) {
  this->UnitsPath = StorePath;
  store::appendUnitSubDir(this->UnitsPath);
  this->ProviderIdentifier = std::string(ProviderIdentifier);
  this->ProviderVersion = std::string(ProviderVersion);
  SmallString<256> AbsOutputFile(OutputFile);
  if (OutputFile != "-")  // Can't make stdout absolute, should stay as "-".
    FileMgr.makeAbsolutePath(AbsOutputFile);
  this->OutputFile = std::string(AbsOutputFile.str());
  this->ModuleName = std::string(ModuleName);
  this->MainFile = MainFile;
  this->IsSystemUnit = IsSystem;
  this->IsModuleUnit = IsModuleUnit;
  this->IsDebugCompilation = IsDebugCompilation;
  this->TargetTriple = std::string(TargetTriple);
  SmallString<256> AbsSysroot(SysrootPath);
  FileMgr.makeAbsolutePath(AbsSysroot);
  this->SysrootPath = std::string(AbsSysroot.str());
  this->GetInfoForModuleFn = GetInfoForModule;
}

IndexUnitWriter::~IndexUnitWriter() {}

int IndexUnitWriter::addModule(writer::OpaqueModule Mod) {
  if (!Mod)
    return -1;

  auto Pair = IndexByModule.insert(std::make_pair(Mod, Modules.size()));
  bool WasInserted = Pair.second;
  if (WasInserted) {
    Modules.push_back(Mod);
  }
  return Pair.first->second;
}

int IndexUnitWriter::addFileDependency(const FileEntry *File, bool IsSystem,
                                       writer::OpaqueModule Mod) {
  assert(File);
  auto Pair = IndexByFile.insert(std::make_pair(File, Files.size()));
  bool WasInserted = Pair.second;
  if (WasInserted) {
    Files.push_back(FileEntryData{File, IsSystem, addModule(Mod), {}});
  }
  return Pair.first->second;
}

void IndexUnitWriter::addRecordFile(StringRef RecordFile, const FileEntry *File,
                                    bool IsSystem, writer::OpaqueModule Mod) {
  int Dep = File ? addFileDependency(File, IsSystem, /*module=*/nullptr) : -1;
  Records.push_back(RecordOrUnitData{std::string(RecordFile), Dep, addModule(Mod), IsSystem});
}

void IndexUnitWriter::addASTFileDependency(const FileEntry *File, bool IsSystem,
                                           writer::OpaqueModule Mod,
                                           bool withoutUnitName) {
  assert(File);
  if (!SeenASTFiles.insert(File).second)
    return;

  SmallString<64> UnitName;
  if (!withoutUnitName)
    getUnitNameForOutputFile(File->getName(), UnitName);
  addUnitDependency(UnitName.str(), File, IsSystem, Mod);
}

void IndexUnitWriter::addUnitDependency(StringRef UnitFile,
                                        const FileEntry *File, bool IsSystem,
                                        writer::OpaqueModule Mod) {
  int Dep = File ? addFileDependency(File, IsSystem, /*module=*/nullptr) : -1;
  ASTFileUnits.emplace_back(RecordOrUnitData{std::string(UnitFile), Dep, addModule(Mod), IsSystem});
}

bool IndexUnitWriter::addInclude(const FileEntry *Source, unsigned Line,
                                 const FileEntry *Target) {
  // FIXME: This will ignore includes of headers that resolve to module imports
  // because the 'target' header has not been added as a file dependency earlier
  // so it is missing from \c IndexByFile.

  auto It = IndexByFile.find(Source);
  if (It == IndexByFile.end())
    return false;
  int SourceIndex = It->getSecond();
  It = IndexByFile.find(Target);
  if (It == IndexByFile.end())
    return false;
  int TargetIndex = It->getSecond();
  Files[SourceIndex].Includes.emplace_back(FileInclude{TargetIndex, Line});
  return true;
}

void IndexUnitWriter::getUnitNameForOutputFile(StringRef FilePath,
                                               SmallVectorImpl<char> &Str) {
  SmallString<256> AbsPath(FilePath);
  FileMgr.makeAbsolutePath(AbsPath);
  return getUnitNameForAbsoluteOutputFile(AbsPath, Str, Remapper);
}

void IndexUnitWriter::getUnitPathForOutputFile(StringRef FilePath,
                                               SmallVectorImpl<char> &Str) {
  Str.append(UnitsPath.begin(), UnitsPath.end());
  Str.push_back('/');
  return getUnitNameForOutputFile(FilePath, Str);
}

Optional<bool> IndexUnitWriter::isUnitUpToDateForOutputFile(StringRef FilePath,
                                                            Optional<StringRef> TimeCompareFilePath,
                                                            std::string &Error) {
  SmallString<256> UnitPath;
  getUnitPathForOutputFile(FilePath, UnitPath);

  llvm::sys::fs::file_status UnitStat;
  if (std::error_code EC = llvm::sys::fs::status(UnitPath.c_str(), UnitStat)) {
    if (EC != llvm::errc::no_such_file_or_directory) {
      llvm::raw_string_ostream Err(Error);
      Err << "could not access path '" << UnitPath
          << "': " << EC.message();
      return None;
    }
    return false;
  }

  if (!TimeCompareFilePath.hasValue())
    return true;

  llvm::sys::fs::file_status CompareStat;
  if (std::error_code EC = llvm::sys::fs::status(*TimeCompareFilePath, CompareStat)) {
    if (EC != llvm::errc::no_such_file_or_directory) {
      llvm::raw_string_ostream Err(Error);
      Err << "could not access path '" << *TimeCompareFilePath
          << "': " << EC.message();
      return None;
    }
    return true;
  }

  // Return true (unit is up-to-date) if the file to compare is older than the
  // unit file.
  return CompareStat.getLastModificationTime() <= UnitStat.getLastModificationTime();
}

void IndexUnitWriter::getUnitNameForAbsoluteOutputFile(StringRef FilePath,
                                                   SmallVectorImpl<char> &Str,
                                                 const PathRemapper &Remapper) {
  StringRef Fname = sys::path::filename(FilePath);
  Str.append(Fname.begin(), Fname.end());
  Str.push_back('-');
  // Need to be sure we use the remapped path to keep things hermetic.
  std::string RemappedPath = Remapper.remapPath(FilePath);
  llvm::hash_code PathHashVal = llvm::hash_value(RemappedPath);
  llvm::APInt(64, PathHashVal).toString(Str, 36, /*Signed=*/false);
}

static void writeBlockInfo(BitstreamWriter &Stream) {
  RecordData Record;

  Stream.EnterBlockInfoBlock();
#define BLOCK(X) emitBlockID(X ## _ID, #X, Stream, Record)
#define RECORD(X) emitRecordID(X, #X, Stream, Record)

  BLOCK(UNIT_VERSION_BLOCK);
  RECORD(UNIT_VERSION);

  BLOCK(UNIT_INFO_BLOCK);
  RECORD(UNIT_INFO);

  BLOCK(UNIT_DEPENDENCIES_BLOCK);
  RECORD(UNIT_DEPENDENCY);

  BLOCK(UNIT_INCLUDES_BLOCK);
  RECORD(UNIT_INCLUDE);

  BLOCK(UNIT_PATHS_BLOCK);
  RECORD(UNIT_PATH);
  RECORD(UNIT_PATH_BUFFER);

  BLOCK(UNIT_MODULES_BLOCK);
  RECORD(UNIT_MODULE);
  RECORD(UNIT_MODULE_BUFFER);

#undef RECORD
#undef BLOCK
  Stream.ExitBlock();
}

static void writeVersionInfo(BitstreamWriter &Stream) {
  using namespace llvm::sys;

  Stream.EnterSubblock(UNIT_VERSION_BLOCK_ID, 3);

  auto Abbrev = std::make_shared<BitCodeAbbrev>();
  Abbrev->Add(BitCodeAbbrevOp(UNIT_VERSION));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Store format version
  unsigned AbbrevCode = Stream.EmitAbbrev(std::move(Abbrev));

  RecordData Record;
  Record.push_back(UNIT_VERSION);
  Record.push_back(STORE_FORMAT_VERSION);
  Stream.EmitRecordWithAbbrev(AbbrevCode, Record);

  Stream.ExitBlock();
}

bool IndexUnitWriter::write(std::string &Error) {
  using namespace llvm::sys;

  // Determine the working directory.
  SmallString<128> CWDPath;
  if (!FileMgr.getFileSystemOpts().WorkingDir.empty()) {
    CWDPath = FileMgr.getFileSystemOpts().WorkingDir;
    if (!path::is_absolute(CWDPath)) {
      fs::make_absolute(CWDPath);
    }
  } else {
    std::error_code EC = sys::fs::current_path(CWDPath);
    if (EC) {
      llvm::raw_string_ostream Err(Error);
      Err << "failed to determine current working directory: " << EC.message();
      return true;
    }
  }
  WorkDir = std::string(CWDPath.str());

  SmallString<512> Buffer;
  BitstreamWriter Stream(Buffer);
  Stream.Emit('I', 8);
  Stream.Emit('D', 8);
  Stream.Emit('X', 8);
  Stream.Emit('U', 8);

  PathStorage PathStore(FileMgr, WorkDir, SysrootPath, Remapper);

  writeBlockInfo(Stream);
  writeVersionInfo(Stream);
  writeUnitInfo(Stream, PathStore);
  writeDependencies(Stream, PathStore);
  writeIncludes(Stream, PathStore);
  writePaths(Stream, PathStore);
  writeModules(Stream);

  SmallString<256> UnitPath;
  getUnitPathForOutputFile(OutputFile, UnitPath);

  SmallString<128> TempPath;
  TempPath = path::parent_path(UnitsPath);
  TempPath += '/';
  TempPath += path::filename(UnitPath);
  TempPath += "-%%%%%%%%";
  int TempFD;
  if (llvm::sys::fs::createUniqueFile(TempPath.str(), TempFD, TempPath)) {
    llvm::raw_string_ostream Err(Error);
    Err << "failed to create temporary file: " << TempPath;
    return true;
  }

  raw_fd_ostream OS(TempFD, /*shouldClose=*/true);
  OS.write(Buffer.data(), Buffer.size());
  OS.close();

  if (OS.has_error()) {
    llvm::raw_string_ostream Err(Error);
    Err << "failed to write '" << TempPath << "': " << OS.error().message();
    OS.clear_error();
    return true;
  }

  std::error_code EC = fs::rename(/*from=*/TempPath.c_str(), /*to=*/UnitPath.c_str());
  if (EC) {
    llvm::raw_string_ostream Err(Error);
    Err << "failed to rename '" << TempPath << "' to '" << UnitPath << "': " << EC.message();
    return true;
  }

  return false;
}

void IndexUnitWriter::writeUnitInfo(llvm::BitstreamWriter &Stream,
                                    PathStorage &PathStore) {
  Stream.EnterSubblock(UNIT_INFO_BLOCK_ID, 3);

  auto Abbrev = std::make_shared<BitCodeAbbrev>();
  Abbrev->Add(BitCodeAbbrevOp(UNIT_INFO));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // IsSystemUnit
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 10)); // WorkDir offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // WorkDir size
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 10)); // OutputFile offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // OutputFile size
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 10)); // Sysroot offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Sysroot size
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 10)); // Main path id
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // IsDebugCompilation
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // IsModuleUnit
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 5)); // Module name size
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 5)); // ProviderIdentifier size
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 5)); // ProviderVersion size
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 5)); // ProviderDataVersion
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Module name + ProviderIdentifier + ProviderVersion + target triple
  unsigned AbbrevCode = Stream.EmitAbbrev(std::move(Abbrev));

  RecordData Record;
  Record.push_back(UNIT_INFO);
  Record.push_back(IsSystemUnit);
  std::string RemappedWorkDir = Remapper.remapPath(WorkDir);
  Record.push_back(PathStore.getPathOffset(RemappedWorkDir));
  Record.push_back(RemappedWorkDir.size());
  std::string RemappedOutputFile = Remapper.remapPath(OutputFile);
  Record.push_back(PathStore.getPathOffset(RemappedOutputFile));
  Record.push_back(RemappedOutputFile.size());
  std::string RemappedSysrootPath = Remapper.remapPath(SysrootPath);
  Record.push_back(PathStore.getPathOffset(RemappedSysrootPath));
  Record.push_back(RemappedSysrootPath.size());
  Record.push_back(PathStore.getPathIndex(MainFile) + 1); // Make 1-based with 0=invalid
  Record.push_back(IsDebugCompilation);
  Record.push_back(IsModuleUnit);
  Record.push_back(ModuleName.size());
  Record.push_back(ProviderIdentifier.size());
  Record.push_back(ProviderVersion.size());
  // ProviderDataVersion is reserved. Not sure it is a good to idea to have
  // clients consider the specifics of a 'provider data version', but reserving
  // to avoid store format version change in case there is a use case in the
  // future.
  Record.push_back(0); // ProviderDataVersion
  SmallString<128> InfoStrings;
  InfoStrings += ModuleName;
  InfoStrings += ProviderIdentifier;
  InfoStrings += ProviderVersion;
  InfoStrings += TargetTriple;
  Stream.EmitRecordWithBlob(AbbrevCode, Record, InfoStrings);

  Stream.ExitBlock();
}

void IndexUnitWriter::writeDependencies(llvm::BitstreamWriter &Stream,
                                        PathStorage &PathStore) {
  std::vector<bool> FileUsedForRecordOrUnit;
  FileUsedForRecordOrUnit.resize(Files.size());

  Stream.EnterSubblock(UNIT_DEPENDENCIES_BLOCK_ID, 3);

  auto Abbrev = std::make_shared<BitCodeAbbrev>();
  Abbrev->Add(BitCodeAbbrevOp(UNIT_DEPENDENCY));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, UnitDependencyKindBitNum)); // Dependency kind
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // IsSystem
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 10)); // PathIndex (1-based, 0 = none)
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // ModuleIndex (1-based, 0 = none)
  // Reserved. These used to be time_t & file size but we decided against
  // writing these in order to get reproducible build products (index data
  // output being the same with the same inputs). Keep these reserved for the
  // future, for coming up with a better scheme to track state of dependencies
  // without using modification time.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 0)); // Reserved
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 0)); // Reserved
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Name
  unsigned AbbrevCode = Stream.EmitAbbrev(std::move(Abbrev));

  RecordData Record;

  auto addRecordOrUnitData = [&](UnitDependencyKind K, const RecordOrUnitData &Data) {
    Record.push_back(UNIT_DEPENDENCY);
    Record.push_back(K);
    Record.push_back(Data.IsSystem);
    if (Data.FileIndex != -1) {
      Record.push_back(PathStore.getPathIndex(Files[Data.FileIndex].File) + 1);
      FileUsedForRecordOrUnit[Data.FileIndex] = true;
    } else {
      Record.push_back(0);
    }
    if (Data.ModuleIndex != -1) {
      Record.push_back(Data.ModuleIndex + 1);
    } else {
      Record.push_back(0);
    }
    Record.push_back(0); // Reserved.
    Record.push_back(0); // Reserved.
    Stream.EmitRecordWithBlob(AbbrevCode, Record, Data.Name);
  };

  for (auto &ASTData : ASTFileUnits) {
    Record.clear();
    addRecordOrUnitData(UNIT_DEPEND_KIND_UNIT, ASTData);
  }
  for (auto &recordData : Records) {
    Record.clear();
    addRecordOrUnitData(UNIT_DEPEND_KIND_RECORD, recordData);
  }
  size_t FileIndex = 0;
  for (auto &File : Files) {
    if (FileUsedForRecordOrUnit[FileIndex++])
      continue;
    Record.clear();
    Record.push_back(UNIT_DEPENDENCY);
    Record.push_back(UNIT_DEPEND_KIND_FILE);
    Record.push_back(File.IsSystem);
    Record.push_back(PathStore.getPathIndex(File.File) + 1);
    if (File.ModuleIndex != -1) {
      Record.push_back(File.ModuleIndex + 1);
    } else {
      Record.push_back(0);
    }
    Record.push_back(0); // Reserved.
    Record.push_back(0); // Reserved.
    Stream.EmitRecordWithBlob(AbbrevCode, Record, StringRef());
  }

  Stream.ExitBlock();
}

void IndexUnitWriter::writeIncludes(llvm::BitstreamWriter &Stream,
                                    PathStorage &PathStore) {
  Stream.EnterSubblock(UNIT_INCLUDES_BLOCK_ID, 3);

  auto Abbrev = std::make_shared<BitCodeAbbrev>();
  Abbrev->Add(BitCodeAbbrevOp(UNIT_INCLUDE));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 10)); // source path index (1-based, 0 = no path)
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 12)); // source include line
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 10)); // target path index (1-based, 0 = no path)
  unsigned AbbrevCode = Stream.EmitAbbrev(std::move(Abbrev));

  RecordData Record;

  for (auto &Including : Files) {
    for(auto &Included: Including.Includes) {
      Record.clear();
      Record.push_back(UNIT_INCLUDE);
      Record.push_back(PathStore.getPathIndex(Including.File) + 1);
      Record.push_back(Included.Line);
      Record.push_back(PathStore.getPathIndex(Files[Included.Index].File) + 1);
      Stream.EmitRecordWithAbbrev(AbbrevCode, Record);
    }
  }
  Stream.ExitBlock();
}

void IndexUnitWriter::writePaths(llvm::BitstreamWriter &Stream,
                                 PathStorage &PathStore) {
  Stream.EnterSubblock(UNIT_PATHS_BLOCK_ID, 3);

  auto PathAbbrev = std::make_shared<BitCodeAbbrev>();
  PathAbbrev->Add(BitCodeAbbrevOp(UNIT_PATH));
  PathAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, UnitFilePathPrefixKindBitNum)); // Path prefix kind
  PathAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 10)); // DirPath offset
  PathAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // DirPath size
  PathAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 10)); // Filename offset
  PathAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Filename size
  unsigned PathAbbrevCode = Stream.EmitAbbrev(std::move(PathAbbrev));

  auto PathBufferAbbrev = std::make_shared<BitCodeAbbrev>();
  PathBufferAbbrev->Add(BitCodeAbbrevOp(UNIT_PATH_BUFFER));
  PathBufferAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Paths buffer
  unsigned PathBufferAbbrevCode = Stream.EmitAbbrev(PathBufferAbbrev);

  RecordData Record;
  for(auto &BitPath: PathStore.getBitPaths()) {
    Record.push_back(UNIT_PATH);
    Record.push_back(BitPath.PrefixKind);
    Record.push_back(BitPath.Dir.Offset);
    Record.push_back(BitPath.Dir.Size);
    Record.push_back(BitPath.Filename.Offset);
    Record.push_back(BitPath.Filename.Size);
    Stream.EmitRecordWithAbbrev(PathAbbrevCode, Record);
    Record.clear();
  }

  Record.push_back(UNIT_PATH_BUFFER);
  Stream.EmitRecordWithBlob(PathBufferAbbrevCode, Record, PathStore.getPathsBuffer());

  Stream.ExitBlock();
}

void IndexUnitWriter::writeModules(llvm::BitstreamWriter &Stream) {
  Stream.EnterSubblock(UNIT_MODULES_BLOCK_ID, 3);

  auto Abbrev = std::make_shared<BitCodeAbbrev>();
  Abbrev->Add(BitCodeAbbrevOp(UNIT_MODULE));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 9)); // Module name offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Module name size
  unsigned AbbrevCode = Stream.EmitAbbrev(std::move(Abbrev));

  auto BufferAbbrev = std::make_shared<BitCodeAbbrev>();
  BufferAbbrev->Add(BitCodeAbbrevOp(UNIT_MODULE_BUFFER));
  BufferAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Module names buffer
  unsigned BufferAbbrevCode = Stream.EmitAbbrev(BufferAbbrev);

  SmallString<512> ModuleNamesBuf;

  RecordData Record;
  for (auto &Mod : Modules) {
    SmallString<64> ModuleName;
    StringRef name = GetInfoForModuleFn(Mod, ModuleName).Name;
    size_t offset = ModuleNamesBuf.size();
    ModuleNamesBuf += name;

    Record.push_back(UNIT_MODULE);
    Record.push_back(offset);
    Record.push_back(name.size());
    Stream.EmitRecordWithAbbrev(AbbrevCode, Record);
    Record.clear();
  }

  Record.push_back(UNIT_MODULE_BUFFER);
  Stream.EmitRecordWithBlob(BufferAbbrevCode, Record, ModuleNamesBuf.str());

  Stream.ExitBlock();
}

bool IndexUnitWriter::initIndexDirectory(StringRef StorePath,
                                         std::string &Error) {
  using namespace llvm::sys;
  SmallString<128> SubPath = StorePath;
  store::appendRecordSubDir(SubPath);
  std::error_code EC = fs::create_directories(SubPath);
  if (EC) {
    llvm::raw_string_ostream Err(Error);
    Err << "failed to create directory '" << SubPath << "': " << EC.message();
    return true;
  }

  SubPath = StorePath;
  store::appendUnitSubDir(SubPath);
  EC = fs::create_directory(SubPath);
  if (EC) {
    llvm::raw_string_ostream Err(Error);
    Err << "failed to create directory '" << SubPath << "': " << EC.message();
    return true;
  }

  return false;
}
