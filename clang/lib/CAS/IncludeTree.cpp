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
Expected<NodeT> IncludeTreeBase<NodeT>::create(ObjectStore &DB,
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
  auto Proxy = DB.getProxy(*Node);
  if (!Proxy)
    return Proxy.takeError();
  return NodeT(*Proxy);
}

Expected<IncludeTree::File> IncludeTree::File::create(ObjectStore &DB,
                                                      StringRef Filename,
                                                      ObjectRef Contents) {
  auto PathRef = DB.storeFromString({}, Filename);
  if (!PathRef)
    return PathRef.takeError();
  std::array<ObjectRef, 2> Refs{*PathRef, Contents};
  return IncludeTreeBase::create(DB, Refs, {});
}

Expected<IncludeTree::File> IncludeTree::getBaseFile() {
  auto Node = getCAS().getProxy(getBaseFileRef());
  if (!Node)
    return Node.takeError();
  return File(std::move(*Node));
}

Expected<IncludeTree::FileInfo> IncludeTree::getBaseFileInfo() {
  auto File = getBaseFile();
  if (!File)
    return File.takeError();
  return File->getFileInfo();
}

llvm::Error IncludeTree::forEachInclude(
    llvm::function_ref<llvm::Error(std::pair<Node, uint32_t>)> Callback) {
  size_t RefI = 0;
  return forEachReference([&](ObjectRef Ref) -> llvm::Error {
    if (RefI == 0) {
      ++RefI;
      return llvm::Error::success();
    }
    size_t IncludeI = RefI - 1;
    ++RefI;
    auto Include = getIncludeNode(Ref, getIncludeKind(IncludeI));
    if (!Include)
      return Include.takeError();
    return Callback({*Include, getIncludeOffset(IncludeI)});
  });
}

Expected<IncludeTree>
IncludeTree::create(ObjectStore &DB,
                    SrcMgr::CharacteristicKind FileCharacteristic,
                    ObjectRef BaseFile, ArrayRef<IncludeInfo> Includes,
                    llvm::SmallBitVector Checks) {
  // The data buffer is composed of
  // 1. `uint32_t` offset and `uint8_t` kind for each includes
  // 2. 1 byte for `CharacteristicKind`
  // 3. variable number of bitset bytes for `Checks`.

  char Kind = FileCharacteristic;
  assert(Kind == FileCharacteristic && "SrcMgr::CharacteristicKind too big!");
  assert(File::isValid(DB, BaseFile));
  SmallVector<ObjectRef, 16> Refs;
  Refs.reserve(Includes.size() + 1);
  Refs.push_back(BaseFile);
  SmallString<64> Buffer;
  Buffer.reserve(Includes.size() * sizeof(uint32_t) + 1);

  llvm::raw_svector_ostream BufOS(Buffer);
  llvm::support::endian::Writer Writer(BufOS, llvm::support::little);

  for (const auto &Include : Includes) {
    assert((Include.Kind == NodeKind::Tree &&
            IncludeTree::isValid(DB, Include.Ref)) ||
           (Include.Kind == NodeKind::ModuleImport &&
            ModuleImport::isValid(DB, Include.Ref)));
    Refs.push_back(Include.Ref);
    Writer.write(Include.Offset);
    static_assert(sizeof(uint8_t) == sizeof(Kind));
    Writer.write(static_cast<uint8_t>(Include.Kind));
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

Expected<IncludeTree> IncludeTree::get(ObjectStore &DB, ObjectRef Ref) {
  auto Node = DB.getProxy(Ref);
  if (!Node)
    return Node.takeError();
  if (!isValid(*Node))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "not a IncludeTree node kind");
  return IncludeTree(std::move(*Node));
}

IncludeTree::NodeKind IncludeTree::getIncludeKind(size_t I) const {
  assert(I < getNumIncludes());
  StringRef Data = getData();
  assert(Data.size() >= (I + 1) * (sizeof(uint32_t) + 1));
  uint8_t K = *(Data.data() + I * (sizeof(uint32_t) + 1) + sizeof(uint32_t));
  return NodeKind(K);
}

uint32_t IncludeTree::getIncludeOffset(size_t I) const {
  assert(I < getNumIncludes());
  StringRef Data = getData();
  assert(Data.size() >= (I + 1) * sizeof(uint32_t));
  uint32_t Offset =
      llvm::support::endian::read<uint32_t, llvm::support::little>(
          Data.data() + I * (sizeof(uint32_t) + 1));
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

Expected<IncludeTree::Node> IncludeTree::getIncludeNode(size_t I) {
  return getIncludeNode(getIncludeRef(I), getIncludeKind(I));
}

Expected<IncludeTree::Node> IncludeTree::getIncludeNode(ObjectRef Ref,
                                                        NodeKind K) {
  auto N = getCAS().getProxy(Ref);
  if (!N)
    return N.takeError();
  return Node(std::move(*N), K);
}

bool IncludeTree::isValid(const ObjectProxy &Node) {
  if (!IncludeTreeBase::isValid(Node))
    return false;
  IncludeTreeBase Base(Node);
  if (Base.getNumReferences() == 0)
    return false;
  unsigned NumIncludes = Base.getNumReferences() - 1;
  return Base.getData().size() >= NumIncludes * (sizeof(uint32_t) + 1) + 1;
}

Expected<IncludeTree::ModuleImport>
IncludeTree::ModuleImport::create(ObjectStore &DB, StringRef ModuleName) {
  return IncludeTreeBase::create(DB, {},
                                 llvm::arrayRefFromStringRef<char>(ModuleName));
}

IncludeTree::FileList::FileSizeTy
IncludeTree::FileList::getFileSize(size_t I) const {
  assert(I < getNumFiles());
  StringRef Data = getData();
  assert(Data.size() >= (I + 1) * sizeof(FileSizeTy));
  return llvm::support::endian::read<FileSizeTy, llvm::support::little>(
      Data.data() + I * sizeof(FileSizeTy));
}

llvm::Error IncludeTree::FileList::forEachFile(
    llvm::function_ref<llvm::Error(File, FileSizeTy)> Callback) {
  size_t I = 0;
  return forEachReference([&](ObjectRef Ref) -> llvm::Error {
    auto Include = getFile(Ref);
    if (!Include)
      return Include.takeError();
    return Callback(std::move(*Include), getFileSize(I++));
  });
}

Expected<IncludeTree::FileList>
IncludeTree::FileList::create(ObjectStore &DB, ArrayRef<FileEntry> Files) {
  SmallVector<ObjectRef, 16> Refs;
  Refs.reserve(Files.size());
  SmallString<256> Buffer;
  Buffer.reserve(Files.size() * sizeof(FileSizeTy));

  llvm::raw_svector_ostream BufOS(Buffer);
  llvm::support::endian::Writer Writer(BufOS, llvm::support::little);

  for (const FileEntry &Entry : Files) {
    assert(File::isValid(DB, Entry.FileRef));
    Refs.push_back(Entry.FileRef);
    Writer.write(Entry.Size);
  }
  return IncludeTreeBase::create(DB, Refs, Buffer);
}

Expected<IncludeTree::FileList> IncludeTree::FileList::get(ObjectStore &DB,
                                                           ObjectRef Ref) {
  auto Node = DB.getProxy(Ref);
  if (!Node)
    return Node.takeError();
  if (!isValid(*Node))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "not a IncludeFileList node kind");
  return FileList(std::move(*Node));
}

bool IncludeTree::FileList::isValid(const ObjectProxy &Node) {
  if (!IncludeTreeBase::isValid(Node))
    return false;
  IncludeTreeBase Base(Node);
  unsigned NumFiles = Base.getNumReferences();
  return NumFiles != 0 &&
         Base.getData().size() == NumFiles * sizeof(FileSizeTy);
}

static constexpr char HasPCH = 0x01;
static constexpr char HasModuleMap = 0x02;

Expected<IncludeTreeRoot>
IncludeTreeRoot::create(ObjectStore &DB, ObjectRef MainFileTree,
                        ObjectRef FileList, std::optional<ObjectRef> PCHRef,
                        std::optional<ObjectRef> ModuleMapRef) {
  assert(IncludeTree::isValid(DB, MainFileTree));
  assert(IncludeTree::FileList::isValid(DB, FileList));
  assert(!ModuleMapRef || IncludeTree::File::isValid(DB, *ModuleMapRef));

  std::array<char, 1> Data = {0};
  if (PCHRef)
    Data[0] |= HasPCH;
  if (ModuleMapRef)
    Data[0] |= HasModuleMap;

  SmallVector<ObjectRef> Refs = {MainFileTree, FileList};
  if (PCHRef)
    Refs.push_back(*PCHRef);
  if (ModuleMapRef)
    Refs.push_back(*ModuleMapRef);

  return IncludeTreeBase::create(DB, Refs, Data);
}

Expected<IncludeTreeRoot> IncludeTreeRoot::get(ObjectStore &DB, ObjectRef Ref) {
  auto Node = DB.getProxy(Ref);
  if (!Node)
    return Node.takeError();
  if (!isValid(*Node))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "not a IncludeTreeRoot node kind");
  return IncludeTreeRoot(std::move(*Node));
}

std::optional<unsigned> IncludeTreeRoot::getPCHRefIndex() const {
  if (getData()[0] & HasPCH)
    return 2;
  return std::nullopt;
}
std::optional<unsigned> IncludeTreeRoot::getModuleMapRefIndex() const {
  if (getData()[0] & HasModuleMap)
    return (getData()[0] & HasPCH) ? 3u : 2u;
  return std::nullopt;
}

llvm::Error IncludeTree::File::print(llvm::raw_ostream &OS, unsigned Indent) {
  auto Filename = getFilename();
  if (!Filename)
    return Filename.takeError();
  OS.indent(Indent) << Filename->getData() << ' ';
  getCAS().getID(getContentsRef()).print(OS);
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

  return forEachInclude([&](std::pair<Node, uint32_t> Include) -> llvm::Error {
    llvm::SMLoc Loc = llvm::SMLoc::getFromPointer(
        SM.getMemoryBuffer(BufID)->getBufferStart() + Include.second);
    auto LineCol = SM.getLineAndColumn(Loc);
    OS.indent(Indent) << LineCol.first << ':' << LineCol.second << ' ';
    return Include.first.print(OS, Indent + 2);
  });
}

llvm::Error IncludeTree::FileList::print(llvm::raw_ostream &OS,
                                         unsigned Indent) {
  return forEachFile([&](File File, FileSizeTy) -> llvm::Error {
    return File.print(OS, Indent);
  });
}

llvm::Error IncludeTree::ModuleImport::print(llvm::raw_ostream &OS,
                                             unsigned Indent) {
  OS << "(Module) " << getModuleName() << '\n';
  return llvm::Error::success();
}

llvm::Error IncludeTree::Node::print(llvm::raw_ostream &OS, unsigned Indent) {
  switch (K) {
  case NodeKind::Tree:
    return getIncludeTree().print(OS, Indent);
  case NodeKind::ModuleImport:
    return getModuleImport().print(OS, Indent);
  }
}

llvm::Error IncludeTreeRoot::print(llvm::raw_ostream &OS, unsigned Indent) {
  if (std::optional<ObjectRef> PCHRef = getPCHRef()) {
    OS.indent(Indent) << "(PCH) ";
    getCAS().getID(*PCHRef).print(OS);
    OS << '\n';
  }
  std::optional<cas::IncludeTree> MainTree;
  if (llvm::Error E = getMainFileTree().moveInto(MainTree))
    return E;
  if (llvm::Error E = MainTree->print(OS.indent(Indent), Indent))
    return E;
  std::optional<IncludeTree::File> ModuleMap;
  if (llvm::Error E = getModuleMapFile().moveInto(ModuleMap))
    return E;
  if (ModuleMap) {
    OS.indent(Indent) << "Module Map: ";
    if (llvm::Error E = ModuleMap->print(OS, Indent))
      return E;
  }
  OS.indent(Indent) << "Files:\n";
  std::optional<IncludeTree::FileList> List;
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
  llvm::cas::ObjectStore &CAS;

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
      SmallString<256> NameBuf;
      return llvm::MemoryBuffer::getMemBuffer(Contents,
                                              Name.toStringRef(NameBuf));
    }

    llvm::ErrorOr<std::optional<cas::ObjectRef>>
    getObjectRefForContent() override {
      return ContentsRef;
    }

    std::error_code close() override { return std::error_code(); }
  };

  explicit IncludeTreeFileSystem(llvm::cas::ObjectStore &CAS) : CAS(CAS) {}

  struct FileEntry {
    cas::ObjectRef ContentsRef;
    IncludeTree::FileList::FileSizeTy Size;
    llvm::sys::fs::UniqueID UniqueID;
  };

  struct MaterializedFile : FileEntry {
    StringRef Contents;

    MaterializedFile(StringRef Contents, FileEntry FE)
        : FileEntry(std::move(FE)), Contents(Contents) {}
  };

  llvm::BumpPtrAllocator Alloc;
  llvm::StringMap<FileEntry, llvm::BumpPtrAllocator &> Files{Alloc};
  llvm::StringMap<llvm::sys::fs::UniqueID, llvm::BumpPtrAllocator &>
      Directories{Alloc};

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
      [&](IncludeTree::File File,
          IncludeTree::FileList::FileSizeTy Size) -> llvm::Error {
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
