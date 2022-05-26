//===--- IndexUnitReader.cpp - Index unit deserialization -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Index/IndexUnitReader.h"
#include "IndexDataStoreUtils.h"
#include "BitstreamVisitor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::index;
using namespace clang::index::store;
using namespace llvm;

namespace {

typedef function_ref<bool(const IndexUnitReader::DependencyInfo &)> DependencyReceiver;
typedef function_ref<bool(const IndexUnitReader::IncludeInfo &)> IncludeReceiver;

class IndexUnitReaderImpl {
  sys::TimePoint<> ModTime;
  std::unique_ptr<MemoryBuffer> MemBuf;

public:
  StringRef ProviderIdentifier;
  StringRef ProviderVersion;
  llvm::BitstreamCursor DependCursor;
  llvm::BitstreamCursor IncludeCursor;
  bool IsSystemUnit;
  bool IsModuleUnit;
  bool IsDebugCompilation;
  std::string WorkingDir;
  std::string OutputFile;
  std::string SysrootPath;
  StringRef ModuleName;
  SmallString<128> MainFilePath;
  StringRef Target;
  std::vector<FileBitPath> Paths;
  StringRef PathsBuffer;
  const PathRemapper &Remapper;

  struct ModuleInfo {
    unsigned NameOffset;
    unsigned NameSize;
  };
  std::vector<ModuleInfo> Modules;
  StringRef ModuleNamesBuffer;

  IndexUnitReaderImpl(const PathRemapper &remapper) : Remapper(remapper) {}
  bool init(std::unique_ptr<MemoryBuffer> Buf, sys::TimePoint<> ModTime,
            std::string &Error);

  StringRef getProviderIdentifier() const { return ProviderIdentifier; }
  StringRef getProviderVersion() const { return ProviderVersion; }

  sys::TimePoint<> getModificationTime() const { return ModTime; }
  StringRef getWorkingDirectory() const { return WorkingDir; }
  StringRef getOutputFile() const { return OutputFile; }
  StringRef getSysrootPath() const { return SysrootPath; }
  StringRef getTarget() const { return Target; }

  StringRef getModuleName() const { return ModuleName; }
  StringRef getMainFilePath() const { return MainFilePath.str(); }
  bool hasMainFile() const { return !MainFilePath.empty(); }
  bool isSystemUnit() const { return IsSystemUnit; }
  bool isModuleUnit() const { return IsModuleUnit; }
  bool isDebugCompilation() const { return IsDebugCompilation; }

  /// Unit dependencies are provided ahead of record ones, record ones
  /// ahead of the file ones.
  bool foreachDependency(DependencyReceiver Receiver);

  bool foreachInclude(IncludeReceiver Receiver);

  StringRef getPathFromBuffer(size_t Offset, size_t Size) {
    return PathsBuffer.substr(Offset, Size);
  }

  std::string getAndRemapPathFromBuffer(size_t Offset, size_t Size) {
    return Remapper.remapPath(getPathFromBuffer(Offset, Size));
  }

  void constructFilePath(SmallVectorImpl<char> &Path, int PathIndex);

  StringRef getModuleName(int ModuleIndex);
};

class IndexUnitBitstreamVisitor : public BitstreamVisitor<IndexUnitBitstreamVisitor> {
  IndexUnitReaderImpl &Reader;
  size_t WorkDirOffset;
  size_t WorkDirSize;
  size_t OutputFileOffset;
  size_t OutputFileSize;
  size_t SysrootOffset;
  size_t SysrootSize;
  int MainPathIndex;

public:
  IndexUnitBitstreamVisitor(llvm::BitstreamCursor &Stream,
                            IndexUnitReaderImpl &Reader)
    : BitstreamVisitor(Stream), Reader(Reader) {}

  StreamVisit visitBlock(unsigned ID) {
    switch ((UnitBitBlock)ID) {
    case UNIT_VERSION_BLOCK_ID:
    case UNIT_INFO_BLOCK_ID:
    case UNIT_PATHS_BLOCK_ID:
    case UNIT_MODULES_BLOCK_ID:
      return StreamVisit::Continue;

    case UNIT_DEPENDENCIES_BLOCK_ID:
      Reader.DependCursor = Stream;
      if (Reader.DependCursor.EnterSubBlock(ID)) {
        *Error = "malformed unit dependencies block record";
        return StreamVisit::Abort;
      }
      if (llvm::Error Err = readBlockAbbrevs(Reader.DependCursor)) {
        *Error = toString(std::move(Err));
        return StreamVisit::Abort;
      }
      return StreamVisit::Skip;
    case UNIT_INCLUDES_BLOCK_ID:
      Reader.IncludeCursor = Stream;
      if (Reader.IncludeCursor.EnterSubBlock(ID)) {
        *Error = "malformed unit includes block record";
        return StreamVisit::Abort;
      }
      if (llvm::Error Err = readBlockAbbrevs(Reader.IncludeCursor)) {
        *Error = toString(std::move(Err));
        return StreamVisit::Abort;
      }
      return StreamVisit::Skip;
    }

    // Some newly introduced block in a minor version update that we cannot
    // handle.
    return StreamVisit::Skip;
  }

  StreamVisit visitRecord(unsigned BlockID, unsigned RecID,
                          RecordDataImpl &Record, StringRef Blob) {
    switch (BlockID) {
    case UNIT_VERSION_BLOCK_ID: {
      unsigned StoreFormatVersion = Record[0];
      if (StoreFormatVersion != STORE_FORMAT_VERSION) {
        llvm::raw_string_ostream OS(*Error);
        OS << "Store format version mismatch: " << StoreFormatVersion;
        OS << " , expected: " << STORE_FORMAT_VERSION;
        return StreamVisit::Abort;
      }
      break;
    }

    case UNIT_INFO_BLOCK_ID: {
      assert(RecID == UNIT_INFO);
      unsigned I = 0;
      Reader.IsSystemUnit = Record[I++];

      // Save these to lookup them up after we get the paths buffer.
      WorkDirOffset = Record[I++];
      WorkDirSize = Record[I++];
      OutputFileOffset = Record[I++];
      OutputFileSize = Record[I++];
      SysrootOffset = Record[I++];
      SysrootSize = Record[I++];
      MainPathIndex = (int)Record[I++] - 1;
      Reader.IsDebugCompilation = Record[I++];
      Reader.IsModuleUnit = Record[I++];

      size_t moduleNameSize = Record[I++];
      size_t providerIdentifierSize = Record[I++];
      size_t providerVersionSize = Record[I++];
      I++; // Reserved for ProviderDataVersion.
      Reader.ModuleName = Blob.substr(0, moduleNameSize);
      Blob = Blob.drop_front(moduleNameSize);
      Reader.ProviderIdentifier = Blob.substr(0, providerIdentifierSize);
      Blob = Blob.drop_front(providerIdentifierSize);
      Reader.ProviderVersion = Blob.substr(0, providerVersionSize);
      Reader.Target = Blob.drop_front(providerVersionSize);
      break;
    }

    case UNIT_PATHS_BLOCK_ID:
      switch (RecID) {
      case UNIT_PATH:
        {
          unsigned I = 0;
          UnitFilePathPrefixKind Kind = (UnitFilePathPrefixKind)Record[I++];
          size_t DirOffset = Record[I++];
          size_t DirSize = Record[I++];
          size_t FilenameOffset = Record[I++];
          size_t FilenameSize = Record[I++];

          Reader.Paths.emplace_back(Kind, BitPathComponent(DirOffset, DirSize),
                                  BitPathComponent(FilenameOffset, FilenameSize));
        }
        break;
      case UNIT_PATH_BUFFER:
        Reader.PathsBuffer = Blob;
        Reader.WorkingDir = Reader.getAndRemapPathFromBuffer(WorkDirOffset, WorkDirSize);
        Reader.OutputFile = Reader.getAndRemapPathFromBuffer(OutputFileOffset, OutputFileSize);
        Reader.SysrootPath = Reader.getAndRemapPathFromBuffer(SysrootOffset, SysrootSize);

        // now we can populate the main file's path
        Reader.constructFilePath(Reader.MainFilePath, MainPathIndex);
        break;
      default:
          llvm_unreachable("shouldn't visit this record");
      }
      break;

    case UNIT_MODULES_BLOCK_ID:
      switch (RecID) {
      case UNIT_MODULE:
        {
          unsigned I = 0;
          unsigned NameOffset = Record[I++];
          unsigned NameSize = Record[I++];
          Reader.Modules.push_back({NameOffset, NameSize});
        }
        break;
      case UNIT_MODULE_BUFFER:
        Reader.ModuleNamesBuffer = Blob;
        break;
      default:
          llvm_unreachable("shouldn't visit this record");
      }
      break;

    case UNIT_DEPENDENCIES_BLOCK_ID:
    case UNIT_INCLUDES_BLOCK_ID:
      llvm_unreachable("shouldn't visit this block'");
    }
    return StreamVisit::Continue;
  }
};

typedef std::function<bool(RecordDataImpl& Record, StringRef Blob)>
  BlockVisitorCallback;

class IndexUnitBlockBitstreamVisitor : public BitstreamVisitor<IndexUnitBlockBitstreamVisitor> {
  unsigned RecID;
  BlockVisitorCallback Visit;

public:
  IndexUnitBlockBitstreamVisitor(unsigned RecID,
                                 llvm::BitstreamCursor &BlockStream,
                                 BlockVisitorCallback Visit)
  : BitstreamVisitor(BlockStream), RecID(RecID), Visit(std::move(Visit)) {}

  StreamVisit visitRecord(unsigned BlockID, unsigned RecID,
                          RecordDataImpl &Record, StringRef Blob) {
    if (RecID != this->RecID)
      llvm_unreachable("shouldn't be called with this RecID");

    if (Visit(Record, Blob))
      return StreamVisit::Continue;
    return StreamVisit::Abort;
  }
};

} // anonymous namespace

bool IndexUnitReaderImpl::init(std::unique_ptr<MemoryBuffer> Buf,
                               sys::TimePoint<> ModTime,
                               std::string &Error) {
  this->ModTime = ModTime;
  this->MemBuf = std::move(Buf);
  llvm::BitstreamCursor Stream(*MemBuf);

  if (Stream.AtEndOfStream()) {
    Error = "empty file";
    return true;
  }

  // Sniff for the signature.
  for (unsigned char C : {'I', 'D', 'X', 'U'}) {
    if (Expected<llvm::SimpleBitstreamCursor::word_t> Res = Stream.Read(8)) {
      if (Res.get() == C)
        continue;
    } else {
      Error = toString(Res.takeError());
      return true;
    }
    Error = "not a serialized index unit file";
    return true;
  }

  IndexUnitBitstreamVisitor BitVisitor(Stream, *this);
  return !BitVisitor.visit(Error);
}

/// Unit dependencies are provided ahead of record ones, record ones
/// ahead of the file ones.
bool IndexUnitReaderImpl::foreachDependency(DependencyReceiver Receiver) {
  store::SavedStreamPosition SavedDepPosition(DependCursor);
  IndexUnitBlockBitstreamVisitor Visitor(UNIT_DEPENDENCY, DependCursor,
  [&](RecordDataImpl& Record, StringRef Blob) {
    unsigned I = 0;
    UnitDependencyKind UnitDepKind = (UnitDependencyKind)Record[I++];
    bool IsSystem = Record[I++];
    int PathIndex = (int)Record[I++] - 1;
    int ModuleIndex = (int)Record[I++] - 1;
    I++; // Reserved field.
    I++; // Reserved field.
    StringRef Name = Blob;

    IndexUnitReader::DependencyKind DepKind;
    switch (UnitDepKind) {
      case UNIT_DEPEND_KIND_UNIT:
        DepKind = IndexUnitReader::DependencyKind::Unit; break;
      case UNIT_DEPEND_KIND_RECORD:
        DepKind = IndexUnitReader::DependencyKind::Record; break;
      case UNIT_DEPEND_KIND_FILE:
        DepKind = IndexUnitReader::DependencyKind::File; break;
    }

    SmallString<512> PathBuf;
    this->constructFilePath(PathBuf, PathIndex);
    StringRef ModuleName = this->getModuleName(ModuleIndex);

    return Receiver(IndexUnitReader::DependencyInfo{DepKind, IsSystem, Name,
      PathBuf.str(), ModuleName});
  });

  std::string Error;
  return Visitor.visit(Error);
}

bool IndexUnitReaderImpl::foreachInclude(IncludeReceiver Receiver) {
  store::SavedStreamPosition SavedIncPosition(IncludeCursor);
  IndexUnitBlockBitstreamVisitor Visitor(UNIT_INCLUDE, IncludeCursor,
  [&](RecordDataImpl& Record, StringRef Blob) {
    unsigned I = 0;
    int SourcePathIndex = (int)Record[I++] - 1;
    unsigned Line = Record[I++];
    int TargetPathIndex = (int)Record[I++] - 1;

    SmallString<512> SourceBuf, TargetBuf;
    this->constructFilePath(SourceBuf, SourcePathIndex);
    this->constructFilePath(TargetBuf, TargetPathIndex);
    return Receiver(IndexUnitReader::IncludeInfo{SourceBuf.str(), Line, TargetBuf.str()});
  });

  std::string Error;
  return Visitor.visit(Error);
}


void IndexUnitReaderImpl::constructFilePath(SmallVectorImpl<char> &PathBuf,
                       int PathIndex) {

  if (PathIndex < 0) return;
  FileBitPath &Path = Paths[PathIndex];
  StringRef Prefix;
  switch (Path.PrefixKind) {
  case UNIT_PATH_PREFIX_NONE:
    break;
  case UNIT_PATH_PREFIX_WORKDIR:
    Prefix = getWorkingDirectory();
    break;
  case UNIT_PATH_PREFIX_SYSROOT:
    Prefix = getSysrootPath();
    break;
  }
  PathBuf.append(Prefix.begin(), Prefix.end());
  sys::path::append(PathBuf,
                    getPathFromBuffer(Path.Dir.Offset, Path.Dir.Size),
                    getPathFromBuffer(Path.Filename.Offset, Path.Filename.Size));
  if (Path.PrefixKind == UNIT_PATH_PREFIX_NONE && !Remapper.empty())
    Remapper.remapPath(PathBuf);
}

StringRef IndexUnitReaderImpl::getModuleName(int ModuleIndex) {
  if (ModuleIndex < 0 || ModuleNamesBuffer.empty())
    return StringRef();
  auto &ModInfo = Modules[ModuleIndex];
  return ModuleNamesBuffer.substr(ModInfo.NameOffset, ModInfo.NameSize);
}


//===----------------------------------------------------------------------===//
// IndexUnitReader
//===----------------------------------------------------------------------===//

std::unique_ptr<IndexUnitReader>
IndexUnitReader::createWithUnitFilename(StringRef UnitFilename,
                                        StringRef StorePath,
                                        const PathRemapper &Remapper,
                                        std::string &Error) {
  SmallString<128> PathBuf = StorePath;
  appendUnitSubDir(PathBuf);
  sys::path::append(PathBuf, UnitFilename);
  return createWithFilePath(PathBuf.str(), Remapper, Error);
}

std::unique_ptr<IndexUnitReader>
IndexUnitReader::createWithFilePath(StringRef FilePath,
                                    const PathRemapper &Remapper,
                                    std::string &Error) {
  int FD;
  std::error_code EC = sys::fs::openFileForRead(FilePath, FD);
  if (EC) {
    raw_string_ostream(Error) << "Failed opening '" << FilePath << "': "
      << EC.message();
    return nullptr;
  }

  assert(FD != -1);
  struct AutoFDClose {
    int FD;
    AutoFDClose(int FD) : FD(FD) {}
    ~AutoFDClose() {
        llvm::sys::Process::SafelyCloseFileDescriptor(FD);
    }
  } AutoFDClose(FD);

  sys::fs::file_status FileStat;
  EC = sys::fs::status(FD, FileStat);
  if (EC) {
    Error = EC.message();
    return nullptr;
  }

  auto ErrOrBuf = MemoryBuffer::getOpenFile(sys::fs::convertFDToNativeFile(FD),
                                            FilePath, /*FileSize=*/-1,
                                            /*RequiresNullTerminator=*/false);
  if (!ErrOrBuf) {
    raw_string_ostream(Error) << "Failed opening '" << FilePath << "': "
      << ErrOrBuf.getError().message();
    return nullptr;
  }

  auto Impl = std::make_unique<IndexUnitReaderImpl>(Remapper);
  bool Err = Impl->init(std::move(*ErrOrBuf), FileStat.getLastModificationTime(),
                        Error);
  if (Err)
    return nullptr;

  std::unique_ptr<IndexUnitReader> Reader;
  Reader.reset(new IndexUnitReader(Impl.release()));
  return Reader;
}

Optional<sys::TimePoint<>>
IndexUnitReader::getModificationTimeForUnit(StringRef UnitFilename,
                                            StringRef StorePath,
                                            std::string &Error) {
  SmallString<128> PathBuf = StorePath;
  appendUnitSubDir(PathBuf);
  sys::path::append(PathBuf, UnitFilename);

  sys::fs::file_status FileStat;
  std::error_code EC = sys::fs::status(PathBuf.str(), FileStat);
  if (EC) {
    Error = EC.message();
    return None;
  }
  return FileStat.getLastModificationTime();
}

#define IMPL static_cast<IndexUnitReaderImpl*>(Impl)

IndexUnitReader::~IndexUnitReader() {
  delete IMPL;
}

StringRef IndexUnitReader::getProviderIdentifier() const {
  return IMPL->getProviderIdentifier();
}

StringRef IndexUnitReader::getProviderVersion() const {
  return IMPL->getProviderVersion();
}

llvm::sys::TimePoint<> IndexUnitReader::getModificationTime() const {
  return IMPL->getModificationTime();
}

StringRef IndexUnitReader::getWorkingDirectory() const {
  return IMPL->getWorkingDirectory();
}

StringRef IndexUnitReader::getOutputFile() const {
  return IMPL->getOutputFile();
}

StringRef IndexUnitReader::getSysrootPath() const {
  return IMPL->getSysrootPath();
}

StringRef IndexUnitReader::getMainFilePath() const {
  return IMPL->getMainFilePath();
}

StringRef IndexUnitReader::getModuleName() const {
  return IMPL->getModuleName();
}

StringRef IndexUnitReader::getTarget() const {
  return IMPL->getTarget();
}

bool IndexUnitReader::hasMainFile() const {
  return IMPL->hasMainFile();
}

bool IndexUnitReader::isSystemUnit() const {
  return IMPL->isSystemUnit();
}

bool IndexUnitReader::isModuleUnit() const {
  return IMPL->isModuleUnit();
}

bool IndexUnitReader::isDebugCompilation() const {
  return IMPL->isDebugCompilation();
}

/// \c Index is the index in the \c getDependencies array.
/// Unit dependencies are provided ahead of record ones.
bool IndexUnitReader::foreachDependency(DependencyReceiver Receiver) {
  return IMPL->foreachDependency(std::move(Receiver));
}

bool IndexUnitReader::foreachInclude(IncludeReceiver Receiver) {
  return IMPL->foreachInclude(std::move(Receiver));
}
