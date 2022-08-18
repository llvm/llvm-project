//===- IncludeTree.cpp - Include-tree CAS graph -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CAS/IncludeTree.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/EndianStream.h"

using namespace clang;
using namespace clang::cas;

template <typename NodeT>
Expected<NodeT> IncludeTreeBase<NodeT>::create(CASDB &DB,
                                               ArrayRef<ObjectRef> Refs,
                                               ArrayRef<char> Data) {
  // Using 4 chars for less chance that it will randomly match a wrong node and
  // makes the buffer 4 bytes "aligned".
  static_assert(NodeT::getNodeKind().size() == 4,
                "getNodeKind() should return 4 characters");
  SmallString<256> Buf{NodeT::getNodeKind()};
  Buf.reserve(Data.size() + NodeT::getNodeKind().size());
  Buf.append(Data.begin(), Data.end());
  auto Node = DB.store(Refs, Buf);
  if (!Node)
    return Node.takeError();
  return NodeT(ObjectProxy::load(DB, *Node));
}

Expected<IncludeFile> IncludeFile::create(CASDB &DB, StringRef Filename,
                                          ObjectRef Contents) {
  auto PathHandle = DB.storeFromString({}, Filename);
  if (!PathHandle)
    return PathHandle.takeError();
  std::array<ObjectRef, 2> Refs{DB.getReference(*PathHandle), Contents};
  return IncludeTreeBase::create(DB, Refs, {});
}

llvm::Error IncludeTree::forEachInclude(
    llvm::function_ref<llvm::Error(std::pair<IncludeTree, uint32_t>)>
        Callback) {
  size_t RefI = 0;
  return forEachReference([&](ObjectRef Ref) -> llvm::Error {
    if (RefI == 0) {
      ++RefI;
      return llvm::Error::success();
    }
    size_t IncludeI = RefI - 1;
    ++RefI;
    auto Include = getInclude(Ref);
    if (!Include)
      return Include.takeError();
    return Callback({*Include, getIncludeOffset(IncludeI)});
  });
}

Expected<IncludeTree>
IncludeTree::create(CASDB &DB, SrcMgr::CharacteristicKind FileCharacteristic,
                    ObjectRef BaseFile,
                    ArrayRef<std::pair<ObjectRef, uint32_t>> Includes,
                    llvm::SmallBitVector Checks) {
  // The data buffer is composed of
  // 1. `uint32_t` offsets of includes
  // 2. 1 byte for `CharacteristicKind`
  // 3. variable number of bitset bytes for `Checks`.

  char Kind = FileCharacteristic;
  assert(Kind == FileCharacteristic && "SrcMgr::CharacteristicKind too big!");
  assert(IncludeFile::isValid(DB, BaseFile));
  SmallVector<ObjectRef, 16> Refs;
  Refs.reserve(Includes.size() + 1);
  Refs.push_back(BaseFile);
  SmallString<64> Buffer;
  Buffer.reserve(Includes.size() * sizeof(uint32_t) + 1);

  llvm::raw_svector_ostream BufOS(Buffer);
  llvm::support::endian::Writer Writer(BufOS, llvm::support::little);

  for (const auto &Include : Includes) {
    ObjectRef FileRef = Include.first;
    uint32_t Offset = Include.second;
    assert(IncludeTree::isValid(DB, FileRef));
    Refs.push_back(FileRef);
    Writer.write(Offset);
  }

  Buffer += Kind;

  uintptr_t Store;
  ArrayRef<uintptr_t> BitWords = Checks.getData(Store);
  size_t RemainingBitsCount = Checks.size();
  while (RemainingBitsCount > 0) {
    if (BitWords.size() > 1) {
      Writer.write(BitWords.front());
      BitWords = BitWords.drop_front();
      RemainingBitsCount -= sizeof(uintptr_t) * CHAR_BIT;
      continue;
    }
    assert(RemainingBitsCount <= sizeof(uintptr_t) * CHAR_BIT);
    uintptr_t LastWord = BitWords.front();
    unsigned BytesNum = RemainingBitsCount / CHAR_BIT;
    if (RemainingBitsCount % CHAR_BIT != 0)
      ++BytesNum;
    while (BytesNum--) {
      Buffer.push_back(LastWord & 0xFF);
      LastWord >>= CHAR_BIT;
    }
    break;
  }

  return IncludeTreeBase::create(DB, Refs, Buffer);
}

Expected<IncludeTree> IncludeTree::get(CASDB &DB, ObjectRef Ref) {
  auto Node = DB.getProxy(Ref);
  if (!Node)
    return Node.takeError();
  if (!isValid(*Node))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "not a IncludeTree node kind");
  return IncludeTree(std::move(*Node));
}

uint32_t IncludeTree::getIncludeOffset(size_t I) const {
  assert(I < getNumIncludes());
  StringRef Data = getData();
  assert(Data.size() >= (I + 1) * sizeof(uint32_t));
  uint32_t Offset =
      llvm::support::endian::read<uint32_t, llvm::support::little>(
          Data.data() + I * sizeof(uint32_t));
  return Offset;
}

bool IncludeTree::getCheckResult(size_t I) const {
  // Skip include offsets and CharacteristicKind.
  StringRef Data = dataSkippingIncludes().drop_front();
  unsigned ByteIndex = I / CHAR_BIT;
  size_t RemainingIndex = I % CHAR_BIT;
  uint8_t Bits = Data[ByteIndex];
  return Bits & (1 << RemainingIndex);
}

bool IncludeTree::isValid(const ObjectProxy &Node) {
  if (!IncludeTreeBase::isValid(Node))
    return false;
  IncludeTreeBase Base(Node);
  if (Base.getNumReferences() == 0)
    return false;
  unsigned NumIncludes = Base.getNumReferences() - 1;
  return Base.getData().size() >= NumIncludes * sizeof(uint32_t) + 1;
}

IncludeFileList::FileSizeTy IncludeFileList::getFileSize(size_t I) const {
  assert(I < getNumFiles());
  StringRef Data = getData();
  assert(Data.size() >= (I + 1) * sizeof(FileSizeTy));
  return llvm::support::endian::read<FileSizeTy, llvm::support::little>(
      Data.data() + I * sizeof(FileSizeTy));
}

llvm::Error IncludeFileList::forEachFile(
    llvm::function_ref<llvm::Error(IncludeFile, FileSizeTy)> Callback) {
  size_t I = 0;
  return forEachReference([&](ObjectRef Ref) -> llvm::Error {
    auto Include = getFile(Ref);
    if (!Include)
      return Include.takeError();
    return Callback(std::move(*Include), getFileSize(I++));
  });
}

Expected<IncludeFileList> IncludeFileList::create(CASDB &DB,
                                                  ArrayRef<FileEntry> Files) {
  SmallVector<ObjectRef, 16> Refs;
  Refs.reserve(Files.size());
  SmallString<256> Buffer;
  Buffer.reserve(Files.size() * sizeof(FileSizeTy));

  llvm::raw_svector_ostream BufOS(Buffer);
  llvm::support::endian::Writer Writer(BufOS, llvm::support::little);

  for (const FileEntry &Entry : Files) {
    assert(IncludeFile::isValid(DB, Entry.FileRef));
    Refs.push_back(Entry.FileRef);
    Writer.write(Entry.Size);
  }
  return IncludeTreeBase::create(DB, Refs, Buffer);
}

Expected<IncludeFileList> IncludeFileList::get(CASDB &DB, ObjectRef Ref) {
  auto Node = DB.getProxy(Ref);
  if (!Node)
    return Node.takeError();
  if (!isValid(*Node))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "not a IncludeFileList node kind");
  return IncludeFileList(std::move(*Node));
}

bool IncludeFileList::isValid(const ObjectProxy &Node) {
  if (!IncludeTreeBase::isValid(Node))
    return false;
  IncludeTreeBase Base(Node);
  unsigned NumFiles = Base.getNumReferences();
  return NumFiles != 0 &&
         Base.getData().size() == NumFiles * sizeof(FileSizeTy);
}

Expected<IncludeTreeRoot>
IncludeTreeRoot::create(CASDB &DB, ObjectRef MainFileTree, ObjectRef FileList) {
  assert(IncludeTree::isValid(DB, MainFileTree));
  assert(IncludeFileList::isValid(DB, FileList));
  return IncludeTreeBase::create(DB, {MainFileTree, FileList}, {});
}

Expected<IncludeTreeRoot> IncludeTreeRoot::get(CASDB &DB, ObjectRef Ref) {
  auto Node = DB.getProxy(Ref);
  if (!Node)
    return Node.takeError();
  if (!isValid(*Node))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "not a IncludeTreeRoot node kind");
  return IncludeTreeRoot(std::move(*Node));
}

llvm::Error IncludeFile::print(llvm::raw_ostream &OS, unsigned Indent) {
  auto Filename = getFilename();
  if (!Filename)
    return Filename.takeError();
  OS.indent(Indent) << Filename->getData() << ' ';
  CAS->getID(getContentsRef()).print(OS);
  OS << '\n';
  return llvm::Error::success();
}

llvm::Error IncludeTree::print(llvm::raw_ostream &OS, unsigned Indent) {
  auto IncludeBy = getBaseFile();
  if (!IncludeBy)
    return IncludeBy.takeError();
  if (llvm::Error E = IncludeBy->print(OS))
    return E;

  llvm::SourceMgr SM;
  auto Blob = IncludeBy->getContents();
  if (!Blob)
    return Blob.takeError();
  auto MemBuf = llvm::MemoryBuffer::getMemBuffer(Blob->getData());
  unsigned BufID = SM.AddNewSourceBuffer(std::move(MemBuf), llvm::SMLoc());

  return forEachInclude(
      [&](std::pair<cas::IncludeTree, uint32_t> Include) -> llvm::Error {
        llvm::SMLoc Loc = llvm::SMLoc::getFromPointer(
            SM.getMemoryBuffer(BufID)->getBufferStart() + Include.second);
        auto LineCol = SM.getLineAndColumn(Loc);
        OS.indent(Indent) << LineCol.first << ':' << LineCol.second << ' ';
        return Include.first.print(OS, Indent + 2);
      });
}

llvm::Error IncludeFileList::print(llvm::raw_ostream &OS, unsigned Indent) {
  return forEachFile([&](cas::IncludeFile File, FileSizeTy) -> llvm::Error {
    return File.print(OS, Indent);
  });
}

llvm::Error IncludeTreeRoot::print(llvm::raw_ostream &OS, unsigned Indent) {
  Optional<cas::IncludeTree> MainTree;
  if (llvm::Error E = getMainFileTree().moveInto(MainTree))
    return E;
  if (llvm::Error E = MainTree->print(OS.indent(Indent), Indent))
    return E;
  OS.indent(Indent) << "Files:\n";
  Optional<cas::IncludeFileList> List;
  if (llvm::Error E = getFileList().moveInto(List))
    return E;
  return List->print(OS, Indent);
}

namespace {
/// An implementation of a \p vfs::FileSystem that supports the simple queries
/// of the preprocessor, for creating \p FileEntries using a file path, while
/// "replaying" an \p IncludeTreeRoot. It is not intended to be a complete
/// implementation of a file system.
class IncludeTreeFileSystem : public llvm::vfs::FileSystem {
  llvm::cas::CASDB &CAS;

public:
  class IncludeTreeFile : public llvm::vfs::File {
    llvm::vfs::Status Stat;
    StringRef Contents;
    cas::ObjectRef ContentsRef;

  public:
    IncludeTreeFile(llvm::vfs::Status Stat, StringRef Contents,
                    cas::ObjectRef ContentsRef)
        : Stat(std::move(Stat)), Contents(Contents),
          ContentsRef(std::move(ContentsRef)) {}

    llvm::ErrorOr<llvm::vfs::Status> status() override { return Stat; }

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
    getBuffer(const Twine &Name, int64_t FileSize, bool RequiresNullTerminator,
              bool IsVolatile) override {
      return llvm::MemoryBuffer::getMemBuffer(Contents);
    }

    llvm::ErrorOr<Optional<cas::ObjectRef>> getObjectRefForContent() override {
      return ContentsRef;
    }

    std::error_code close() override { return std::error_code(); }
  };

  explicit IncludeTreeFileSystem(llvm::cas::CASDB &CAS) : CAS(CAS) {}

  struct FileEntry {
    cas::ObjectRef ContentsRef;
    IncludeFileList::FileSizeTy Size;
    llvm::sys::fs::UniqueID UniqueID;
  };

  struct MaterializedFile : FileEntry {
    StringRef Contents;

    MaterializedFile(StringRef Contents, FileEntry FE)
        : FileEntry(std::move(FE)), Contents(Contents) {}
  };

  llvm::StringMap<FileEntry> Files;
  llvm::StringMap<llvm::sys::fs::UniqueID> Directories;

  llvm::ErrorOr<llvm::vfs::Status> status(const Twine &Path) override {
    SmallString<128> FilenameBuffer;
    StringRef Filename = Path.toStringRef(FilenameBuffer);

    auto FileEntry = Files.find(Filename);
    if (FileEntry != Files.end()) {
      return makeStatus(Filename, FileEntry->second.Size,
                        FileEntry->second.UniqueID,
                        llvm::sys::fs::file_type::regular_file);
    }

    // Also check whether this is a parent directory status query.
    auto DirEntry = Directories.find(Filename);
    if (DirEntry != Directories.end()) {
      return makeStatus(Filename, /*Size*/ 0, DirEntry->second,
                        llvm::sys::fs::file_type::directory_file);
    }

    return llvm::errorToErrorCode(fileError(Filename));
  }

  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
  openFileForRead(const Twine &Path) override {
    SmallString<128> FilenameBuffer;
    StringRef Filename = Path.toStringRef(FilenameBuffer);
    auto MaterializedFile = materialize(Filename);
    if (!MaterializedFile)
      return llvm::errorToErrorCode(MaterializedFile.takeError());
    llvm::vfs::Status Stat = makeStatus(
        Filename, MaterializedFile->Contents.size(), MaterializedFile->UniqueID,
        llvm::sys::fs::file_type::regular_file);
    return std::make_unique<IncludeTreeFile>(
        std::move(Stat), MaterializedFile->Contents,
        std::move(MaterializedFile->ContentsRef));
  }

  Expected<MaterializedFile> materialize(StringRef Filename) {
    auto Entry = Files.find(Filename);
    if (Entry == Files.end())
      return fileError(Filename);
    auto ContentsBlob = CAS.getProxy(Entry->second.ContentsRef);
    if (!ContentsBlob)
      return ContentsBlob.takeError();

    return MaterializedFile{ContentsBlob->getData(), Entry->second};
  }

  static llvm::vfs::Status makeStatus(StringRef Filename, uint64_t Size,
                                      llvm::sys::fs::UniqueID UniqueID,
                                      llvm::sys::fs::file_type Type) {
    const llvm::sys::fs::perms Permissions =
        llvm::sys::fs::perms::all_read | llvm::sys::fs::perms::owner_write;
    return llvm::vfs::Status(Filename, UniqueID, llvm::sys::TimePoint<>(),
                             /*User=*/0,
                             /*Group=*/0, Size, Type, Permissions);
  }

  static llvm::Error fileError(StringRef Filename) {
    return llvm::createFileError(
        Filename,
        llvm::createStringError(std::errc::no_such_file_or_directory,
                                "filename not part of include tree list"));
  }

  llvm::vfs::directory_iterator dir_begin(const Twine &Dir,
                                          std::error_code &EC) override {
    EC = llvm::errc::operation_not_permitted;
    return llvm::vfs::directory_iterator();
  }
  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const override {
    return llvm::errc::operation_not_permitted;
  }
  std::error_code setCurrentWorkingDirectory(const Twine &Path) override {
    return llvm::errc::operation_not_permitted;
  }
};
} // namespace

Expected<IntrusiveRefCntPtr<llvm::vfs::FileSystem>>
cas::createIncludeTreeFileSystem(IncludeTreeRoot &Root) {
  auto FileList = Root.getFileList();
  if (!FileList)
    return FileList.takeError();

  IntrusiveRefCntPtr<IncludeTreeFileSystem> IncludeTreeFS =
      new IncludeTreeFileSystem(Root.getCAS());
  llvm::Error E = FileList->forEachFile(
      [&](IncludeFile File, IncludeFileList::FileSizeTy Size) -> llvm::Error {
        auto FilenameBlob = File.getFilename();
        if (!FilenameBlob)
          return FilenameBlob.takeError();
        StringRef Filename = FilenameBlob->getData();

        StringRef DirName = llvm::sys::path::parent_path(Filename);
        if (DirName.empty())
          DirName = ".";
        auto &DirEntry = IncludeTreeFS->Directories[DirName];
        if (DirEntry == llvm::sys::fs::UniqueID()) {
          DirEntry = llvm::vfs::getNextVirtualUniqueID();
        }

        IncludeTreeFS->Files.insert(
            std::make_pair(Filename, IncludeTreeFileSystem::FileEntry{
                                         File.getContentsRef(), Size,
                                         llvm::vfs::getNextVirtualUniqueID()}));
        return llvm::Error::success();
      });
  if (E)
    return std::move(E);

  return IncludeTreeFS;
}
