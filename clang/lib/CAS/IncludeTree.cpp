//===- IncludeTree.cpp - Include-tree CAS graph -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CAS/IncludeTree.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Error.h"
#include <utility>

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
  const size_t IncludeEnd = getNumIncludes();
  return forEachReference([&](ObjectRef Ref) -> llvm::Error {
    if (RefI == 0) {
      ++RefI;
      return llvm::Error::success();
    }
    size_t IncludeI = RefI - 1;
    if (IncludeI >= IncludeEnd)
      return llvm::Error::success();
    ++RefI;
    auto Include = getIncludeNode(Ref, getIncludeKind(IncludeI));
    if (!Include)
      return Include.takeError();
    return Callback({*Include, getIncludeOffset(IncludeI)});
  });
}

/// Write the bitset \p Bits to \p Writer, filling the final byte with zeros for
/// any unused values. Note: this does not store the size of the bitset.
static void writeBitSet(llvm::support::endian::Writer &Writer,
                        const llvm::SmallBitVector &Bits) {
  uintptr_t Store;
  ArrayRef<uintptr_t> BitWords = Bits.getData(Store);
  size_t RemainingBitsCount = Bits.size();
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
      Writer.write(static_cast<uint8_t>(LastWord & 0xFF));
      LastWord >>= CHAR_BIT;
    }
    break;
  }
}

Expected<IncludeTree> IncludeTree::create(
    ObjectStore &DB, SrcMgr::CharacteristicKind FileCharacteristic,
    ObjectRef BaseFile, ArrayRef<IncludeInfo> Includes,
    std::optional<ObjectRef> SubmoduleName, llvm::SmallBitVector Checks) {
  // The data buffer is composed of
  // 1. 1 byte for `CharacteristicKind` and IsSubmodule
  // 2. `uint32_t` offset and `uint8_t` kind for each includes
  // 3. variable number of bitset bytes for `Checks`.

  char Kind = FileCharacteristic;
  assert(Kind == FileCharacteristic && (Kind & IsSubmoduleBit) == 0 &&
         "SrcMgr::CharacteristicKind too big!");
  if (SubmoduleName)
    Kind |= IsSubmoduleBit;

  assert(File::isValid(DB, BaseFile));
  SmallVector<ObjectRef, 16> Refs;
  Refs.reserve(Includes.size() + 2);
  Refs.push_back(BaseFile);
  SmallString<64> Buffer;
  Buffer.reserve(Includes.size() * sizeof(uint32_t) + 1);

  Buffer += Kind;

  llvm::raw_svector_ostream BufOS(Buffer);
  llvm::support::endian::Writer Writer(BufOS, llvm::endianness::little);

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

  if (SubmoduleName)
    Refs.push_back(*SubmoduleName);

  writeBitSet(Writer, Checks);

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
  StringRef Data = dataSkippingFlags();
  assert(Data.size() >= (I + 1) * (sizeof(uint32_t) + 1));
  uint8_t K = *(Data.data() + I * (sizeof(uint32_t) + 1) + sizeof(uint32_t));
  return NodeKind(K);
}

uint32_t IncludeTree::getIncludeOffset(size_t I) const {
  assert(I < getNumIncludes());
  StringRef Data = dataSkippingFlags();
  assert(Data.size() >= (I + 1) * sizeof(uint32_t));
  uint32_t Offset =
      llvm::support::endian::read<uint32_t, llvm::endianness::little>(
          Data.data() + I * (sizeof(uint32_t) + 1));
  return Offset;
}

bool IncludeTree::getCheckResult(size_t I) const {
  // Skip include offsets and CharacteristicKind.
  StringRef Data = dataSkippingIncludes();
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
  if (Base.getNumReferences() == 0 || Base.getData().empty())
    return false;
  unsigned NumIncludes = Base.getNumReferences() - 1;
  if (Base.getData().front() & IsSubmoduleBit)
    NumIncludes -= 1;
  return Base.getData().size() >= NumIncludes * (sizeof(uint32_t) + 1) + 1;
}

Expected<IncludeTree::ModuleImport>
IncludeTree::ModuleImport::create(ObjectStore &DB, StringRef ModuleName,
                                  bool VisibilityOnly) {
  SmallString<64> Buffer;
  Buffer.push_back((char)VisibilityOnly);
  Buffer.append(ModuleName);
  return IncludeTreeBase::create(DB, {}, Buffer);
}

size_t IncludeTree::FileList::getNumFilesCurrentList() const {
  return llvm::support::endian::read<uint32_t, llvm::endianness::little>(
      getData().data());
}

IncludeTree::FileList::FileSizeTy
IncludeTree::FileList::getFileSize(size_t I) const {
  assert(I < getNumFilesCurrentList());
  StringRef Data = getData().drop_front(sizeof(uint32_t));
  assert(Data.size() >= (I + 1) * sizeof(FileSizeTy));
  return llvm::support::endian::read<FileSizeTy, llvm::endianness::little>(
      Data.data() + I * sizeof(FileSizeTy));
}

llvm::Error IncludeTree::FileList::forEachFileImpl(
    llvm::DenseSet<ObjectRef> &Seen,
    llvm::function_ref<llvm::Error(File, FileSizeTy)> Callback) {
  size_t Next = 0;
  size_t FileCount = getNumFilesCurrentList();
  return forEachReference([&](ObjectRef Ref) -> llvm::Error {
    size_t Index = Next++;
    if (!Seen.insert(Ref).second)
      return llvm::Error::success();

    if (Index < FileCount) {
      auto Include = getFile(Ref);
      if (!Include)
        return Include.takeError();
      return Callback(std::move(*Include), getFileSize(Index));
    }

    // Otherwise, it's a chained FileList.
    auto Proxy = getCAS().getProxy(Ref);
    if (!Proxy)
      return Proxy.takeError();
    FileList FL(std::move(*Proxy));
    return FL.forEachFileImpl(Seen, Callback);
  });
}

llvm::Error IncludeTree::FileList::forEachFile(
    llvm::function_ref<llvm::Error(File, FileSizeTy)> Callback) {
  llvm::DenseSet<ObjectRef> Seen;
  return forEachFileImpl(Seen, Callback);
}

Expected<IncludeTree::FileList>
IncludeTree::FileList::create(ObjectStore &DB, ArrayRef<FileEntry> Files,
                              ArrayRef<ObjectRef> FileLists) {
  SmallVector<ObjectRef, 16> Refs;
  Refs.reserve(Files.size() + FileLists.size());
  SmallString<256> Buffer;
  Buffer.reserve(sizeof(uint32_t) + Files.size() * sizeof(FileSizeTy));

  llvm::raw_svector_ostream BufOS(Buffer);
  llvm::support::endian::Writer Writer(BufOS, llvm::endianness::little);
  Writer.write(static_cast<uint32_t>(Files.size()));

  for (const FileEntry &Entry : Files) {
    assert(File::isValid(DB, Entry.FileRef));
    Refs.push_back(Entry.FileRef);
    Writer.write(Entry.Size);
  }

  Refs.append(FileLists.begin(), FileLists.end());

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
  StringRef Data = Base.getData();
  if (Data.size() < sizeof(uint32_t))
    return false;
  unsigned NumFiles =
      llvm::support::endian::read<uint32_t, llvm::endianness::little>(Data.data());
  return NumFiles != 0 && NumFiles <= Base.getNumReferences() &&
         Data.size() == sizeof(uint32_t) + NumFiles * sizeof(FileSizeTy);
}

static constexpr uint16_t ModuleFlagFramework = 1 << 0;
static constexpr uint16_t ModuleFlagExplicit = 1 << 1;
static constexpr uint16_t ModuleFlagExternC = 1 << 2;
static constexpr uint16_t ModuleFlagSystem = 1 << 3;
static constexpr uint16_t ModuleFlagInferSubmodules = 1 << 4;
static constexpr uint16_t ModuleFlagInferExplicitSubmodules = 1 << 5;
static constexpr uint16_t ModuleFlagInferInferExportWildcard = 1 << 6;
static constexpr uint16_t ModuleFlagHasExports = 1 << 7;
static constexpr uint16_t ModuleFlagHasLinkLibraries = 1 << 8;

IncludeTree::Module::ModuleFlags IncludeTree::Module::getFlags() const {
  uint16_t Raw = rawFlags();
  ModuleFlags Flags;
  Flags.IsFramework = Raw & ModuleFlagFramework;
  Flags.IsExplicit = Raw & ModuleFlagExplicit;
  Flags.IsExternC = Raw & ModuleFlagExternC;
  Flags.IsSystem = Raw & ModuleFlagSystem;
  Flags.InferSubmodules = Raw & ModuleFlagInferSubmodules;
  Flags.InferExplicitSubmodules = Raw & ModuleFlagInferExplicitSubmodules;
  Flags.InferExportWildcard = Raw & ModuleFlagInferInferExportWildcard;
  return Flags;
}

size_t IncludeTree::Module::getNumSubmodules() const {
  size_t Count = getNumReferences();
  if (hasExports())
    Count -= 1;
  if (hasLinkLibraries())
    Count -= 1;
  return Count;
}

llvm::Error IncludeTree::Module::forEachSubmodule(
    llvm::function_ref<llvm::Error(Module)> CB) {
  size_t Count = getNumSubmodules();
  return forEachReference([&](ObjectRef Ref) -> llvm::Error {
    if (Count == 0)
      return llvm::Error::success();
    Count -= 1;
    auto Node = getCAS().getProxy(Ref);
    if (!Node)
      return Node.takeError();
    return CB(Module(*Node));
  });
}

Expected<IncludeTree::Module>
IncludeTree::Module::create(ObjectStore &DB, StringRef ModuleName,
                            ModuleFlags Flags, ArrayRef<ObjectRef> Submodules,
                            std::optional<ObjectRef> ExportList,
                            std::optional<ObjectRef> LinkLibraries) {
  // Data:
  // - 2 bytes for Flags
  // - ModuleName (String)
  // Refs:
  // - Submodules (IncludeTreeModule)
  // - (optional) ExportList
  // - (optional) LinkLibaryList

  uint16_t RawFlags = 0;
  if (Flags.IsFramework)
    RawFlags |= ModuleFlagFramework;
  if (Flags.IsExplicit)
    RawFlags |= ModuleFlagExplicit;
  if (Flags.IsExternC)
    RawFlags |= ModuleFlagExternC;
  if (Flags.IsSystem)
    RawFlags |= ModuleFlagSystem;
  if (Flags.InferSubmodules)
    RawFlags |= ModuleFlagInferSubmodules;
  if (Flags.InferExplicitSubmodules)
    RawFlags |= ModuleFlagInferExplicitSubmodules;
  if (Flags.InferExportWildcard)
    RawFlags |= ModuleFlagInferInferExportWildcard;
  if (ExportList)
    RawFlags |= ModuleFlagHasExports;
  if (LinkLibraries)
    RawFlags |= ModuleFlagHasLinkLibraries;

  SmallString<64> Buffer;
  llvm::raw_svector_ostream BufOS(Buffer);
  llvm::support::endian::Writer Writer(BufOS, llvm::endianness::little);
  Writer.write(RawFlags);

  Buffer.append(ModuleName);

  SmallVector<ObjectRef> Refs(Submodules);
  if (ExportList)
    Refs.push_back(*ExportList);
  if (LinkLibraries)
    Refs.push_back(*LinkLibraries);

  return IncludeTreeBase::create(DB, Refs, Buffer);
}

uint16_t IncludeTree::Module::rawFlags() const {
  return llvm::support::endian::read<uint16_t, llvm::endianness::little>(
      getData().data());
}

bool IncludeTree::Module::hasExports() const {
  return rawFlags() & ModuleFlagHasExports;
}
bool IncludeTree::Module::hasLinkLibraries() const {
  return rawFlags() & ModuleFlagHasLinkLibraries;
}

std::optional<unsigned> IncludeTree::Module::getExportsIndex() const {
  if (hasExports())
    return getNumReferences() - (hasLinkLibraries() ? 2 : 1);
  return std::nullopt;
}
std::optional<unsigned> IncludeTree::Module::getLinkLibrariesIndex() const {
  if (hasLinkLibraries())
    return getNumReferences() - 1;
  return std::nullopt;
}

Expected<std::optional<IncludeTree::Module::ExportList>>
IncludeTree::Module::getExports() {
  if (auto Ref = getExportsRef()) {
    auto N = getCAS().getProxy(*Ref);
    if (!N)
      return N.takeError();
    return ExportList(std::move(*N));
  }
  return std::nullopt;
}

/// The list of modules that this submodule re-exports.
Expected<std::optional<IncludeTree::Module::LinkLibraryList>>
IncludeTree::Module::getLinkLibraries() {
  if (auto Ref = getLinkLibrariesRef()) {
    auto N = getCAS().getProxy(*Ref);
    if (!N)
      return N.takeError();
    return LinkLibraryList(std::move(*N));
  }
  return std::nullopt;
}

bool IncludeTree::Module::ExportList::hasGlobalWildcard() const {
  // The bit after explicit exports is global.
  return exportHasWildcard(getNumExplicitExports());
}
bool IncludeTree::Module::ExportList::exportHasWildcard(size_t I) const {
  assert(I < getNumExplicitExports() + 1);
  unsigned ByteIndex = I / CHAR_BIT;
  size_t RemainingIndex = I % CHAR_BIT;
  uint8_t Bits = getData()[ByteIndex];
  return Bits & (1 << RemainingIndex);
}
Expected<IncludeTree::Module::ExportList::Export>
IncludeTree::Module::ExportList::getExplicitExport(size_t I) {
  Expected<ObjectProxy> Name = getCAS().getProxy(getReference(I));
  if (!Name)
    return Name.takeError();
  return Export{Name->getData(), exportHasWildcard(I)};
}
llvm::Error IncludeTree::Module::ExportList::forEachExplicitExport(
    llvm::function_ref<llvm::Error(Export)> CB) {
  size_t ExportI = 0;
  return forEachReference([&](ObjectRef Ref) {
    Expected<ObjectProxy> Name = getCAS().getProxy(Ref);
    if (!Name)
      return Name.takeError();
    return CB(Export{Name->getData(), exportHasWildcard(ExportI)});
  });
}
Expected<IncludeTree::Module::ExportList>
IncludeTree::Module::ExportList::create(ObjectStore &DB,
                                        ArrayRef<Export> Exports,
                                        bool GlobalWildcard) {
  // Data:
  // - 1 bit per explicit export for wildcard
  // - 1 bit for global wildcard
  // Refs: export names
  SmallString<64> Buffer;
  llvm::raw_svector_ostream BufOS(Buffer);
  llvm::support::endian::Writer Writer(BufOS, llvm::endianness::little);
  SmallVector<ObjectRef> Refs;
  llvm::SmallBitVector WildcardBits;
  for (Export E : Exports) {
    auto Ref = DB.storeFromString({}, E.ModuleName);
    if (!Ref)
      return Ref.takeError();
    Refs.push_back(*Ref);
    WildcardBits.push_back(E.Wildcard);
  }
  WildcardBits.push_back(GlobalWildcard);
  writeBitSet(Writer, WildcardBits);

  return IncludeTreeBase::create(DB, Refs, Buffer);
}

bool IncludeTree::Module::LinkLibraryList::isFramework(size_t I) const {
  assert(I < getNumLibraries());
  unsigned ByteIndex = I / CHAR_BIT;
  size_t RemainingIndex = I % CHAR_BIT;
  uint8_t Bits = getData()[ByteIndex];
  return Bits & (1 << RemainingIndex);
}
llvm::Error IncludeTree::Module::LinkLibraryList::forEachLinkLibrary(
    llvm::function_ref<llvm::Error(LinkLibrary)> CB) {
  size_t I = 0;
  return forEachReference([&](ObjectRef Ref) {
    auto Name = getCAS().getProxy(getLibraryNameRef(I));
    if (!Name)
      return Name.takeError();
    return CB({Name->getData(), isFramework(I++)});
  });
}
Expected<IncludeTree::Module::LinkLibraryList>
IncludeTree::Module::LinkLibraryList::create(ObjectStore &DB,
                                             ArrayRef<LinkLibrary> Libraries) {
  // Data:
  // - 1 bit per library for IsFramework
  // Refs: library names
  SmallString<64> Buffer;
  llvm::raw_svector_ostream BufOS(Buffer);
  llvm::support::endian::Writer Writer(BufOS, llvm::endianness::little);
  SmallVector<ObjectRef> Refs;
  llvm::SmallBitVector FrameworkBits;
  for (LinkLibrary L : Libraries) {
    auto Ref = DB.storeFromString({}, L.Library);
    if (!Ref)
      return Ref.takeError();
    Refs.push_back(*Ref);
    FrameworkBits.push_back(L.IsFramework);
  }
  writeBitSet(Writer, FrameworkBits);

  return IncludeTreeBase::create(DB, Refs, Buffer);
}

Expected<IncludeTree::ModuleMap>
IncludeTree::ModuleMap::create(ObjectStore &DB, ArrayRef<ObjectRef> Modules) {
  return IncludeTreeBase::create(DB, Modules, {});
}

llvm::Error IncludeTree::ModuleMap::forEachModule(
    llvm::function_ref<llvm::Error(Module)> CB) {
  return forEachReference([&](ObjectRef Ref) {
    auto N = getCAS().getProxy(Ref);
    if (!N)
      return N.takeError();
    return CB(Module(std::move(*N)));
  });
}

static constexpr char HasPCH = 1;
static constexpr char HasModuleMap = 1 << 1;
static constexpr char HasAPINotes = 1 << 2;

Expected<IncludeTreeRoot>
IncludeTreeRoot::create(ObjectStore &DB, ObjectRef MainFileTree,
                        ObjectRef FileList, std::optional<ObjectRef> PCHRef,
                        std::optional<ObjectRef> ModuleMapRef,
                        std::optional<ObjectRef> APINotesRef) {
  assert(IncludeTree::isValid(DB, MainFileTree));
  assert(IncludeTree::FileList::isValid(DB, FileList));
  assert(!ModuleMapRef || IncludeTree::ModuleMap::isValid(DB, *ModuleMapRef));

  std::array<char, 1> Data = {0};
  if (PCHRef)
    Data[0] |= HasPCH;
  if (ModuleMapRef)
    Data[0] |= HasModuleMap;
  if (APINotesRef)
    Data[0] |= HasAPINotes;

  SmallVector<ObjectRef> Refs = {MainFileTree, FileList};
  if (PCHRef)
    Refs.push_back(*PCHRef);
  if (ModuleMapRef)
    Refs.push_back(*ModuleMapRef);
  if (APINotesRef)
    Refs.push_back(*APINotesRef);

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
std::optional<unsigned> IncludeTreeRoot::getAPINotesRefIndex() const {
  if (getData()[0] & HasAPINotes)
    return 2 + (getPCHRefIndex() ? 1 : 0) + (getModuleMapRefIndex() ? 1 : 0);
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

  auto Submodule = getSubmoduleName();
  if (!Submodule)
    return Submodule.takeError();
  if (*Submodule)
    OS.indent(Indent) << "Submodule: " << **Submodule << '\n';

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
  if (visibilityOnly())
    OS << "(Module for visibility only) ";
  else
    OS << "(Module) ";
  OS << getModuleName() << '\n';
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

llvm::Error IncludeTree::Module::print(llvm::raw_ostream &OS, unsigned Indent) {
  OS.indent(Indent) << getName();
  ModuleFlags Flags = getFlags();
  if (Flags.IsFramework)
    OS << " (framework)";
  if (Flags.IsExplicit)
    OS << " (explicit)";
  if (Flags.IsExternC)
    OS << " (extern_c)";
  if (Flags.IsSystem)
    OS << " (system)";
  OS << '\n';
  if (Flags.InferSubmodules) {
    if (Flags.InferExplicitSubmodules)
      OS << "  explicit module *";
    else
      OS << "  module *";
    if (Flags.InferExportWildcard)
      OS << " { export * }";
    OS << '\n';
  }
  auto ExportList = getExports();
  if (!ExportList)
    return ExportList.takeError();
  if (*ExportList)
    if (llvm::Error E = (*ExportList)->print(OS, Indent + 2))
      return E;
  auto LinkLibraries = getLinkLibraries();
  if (!LinkLibraries)
    return LinkLibraries.takeError();
  if (*LinkLibraries)
    if (llvm::Error E = (*LinkLibraries)->print(OS, Indent + 2))
      return E;
  return forEachSubmodule(
      [&](Module Sub) { return Sub.print(OS, Indent + 2); });
}
llvm::Error IncludeTree::Module::ExportList::print(llvm::raw_ostream &OS,
                                                   unsigned Indent) {
  if (hasGlobalWildcard())
    OS.indent(Indent) << "export *\n";
  return forEachExplicitExport([&](Export E) {
    OS.indent(Indent) << "export " << E.ModuleName;
    if (E.Wildcard)
      OS << ".*";
    OS << '\n';
    return llvm::Error::success();
  });
}

llvm::Error IncludeTree::Module::LinkLibraryList::print(llvm::raw_ostream &OS,
                                                        unsigned Indent) {
  return forEachLinkLibrary([&](LinkLibrary E) {
    OS.indent(Indent) << "link " << E.Library;
    if (E.IsFramework)
      OS << " (framework)";
    OS << '\n';
    return llvm::Error::success();
  });
}

llvm::Error IncludeTree::ModuleMap::print(llvm::raw_ostream &OS,
                                          unsigned Indent) {
  return forEachModule([&](Module M) { return M.print(OS, Indent); });
}

llvm::Expected<IncludeTree::APINotes>
IncludeTree::APINotes::create(ObjectStore &DB,
                              ArrayRef<ObjectRef> APINoteList) {
  assert(APINoteList.size() < 2 && "Too many APINotes added");
  return IncludeTreeBase::create(DB, APINoteList, {});
}

llvm::Expected<IncludeTree::APINotes>
IncludeTree::APINotes::get(ObjectStore &DB, ObjectRef Ref) {
  auto Node = DB.getProxy(Ref);
  if (!Node)
    return Node.takeError();
  if (!isValid(*Node))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "not an APINodes node kind");
  return APINotes(std::move(*Node));
}

llvm::Error IncludeTree::APINotes::print(llvm::raw_ostream &OS,
                                         unsigned Indent) {
  return forEachReference([&](ObjectRef Ref) -> llvm::Error {
    auto Node = getCAS().getProxy(Ref);
    if (!Node)
      return Node.takeError();
    OS.indent(Indent) << Node->getID() << "\n";
    OS.indent(Indent) << Node->getData() << "\n";
    return llvm::Error::success();
  });
}

llvm::Error IncludeTree::APINotes::forEachAPINotes(
    llvm::function_ref<llvm::Error(StringRef)> CB) {
  return forEachReference([&](ObjectRef Ref) {
    auto N = getCAS().getProxy(Ref);
    if (!N)
      return N.takeError();
    return CB(N->getData());
  });
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
  std::optional<IncludeTree::ModuleMap> ModuleMap;
  if (llvm::Error E = getModuleMap().moveInto(ModuleMap))
    return E;
  if (ModuleMap) {
    OS.indent(Indent) << "Module Map:\n";
    if (llvm::Error E = ModuleMap->print(OS, Indent))
      return E;
  }
  OS.indent(Indent) << "Files:\n";
  std::optional<IncludeTree::FileList> List;
  if (llvm::Error E = getFileList().moveInto(List))
    return E;
  std::optional<IncludeTree::APINotes> APINotes;
  if (llvm::Error E = getAPINotes().moveInto(APINotes))
    return E;
  if (APINotes) {
    OS.indent(Indent) << "APINotes:\n";
    if (llvm::Error E = APINotes->print(OS, Indent))
      return E;
  }
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
    SmallString<128> Filename;
    getPath(Path, Filename);
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
    SmallString<128> Filename;
    getPath(Path, Filename);
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

  /// Produce a filename compatible with our StringMaps. See comment in
  /// \c createIncludeTreeFileSystem.
  void getPath(const Twine &Path, SmallVectorImpl<char> &Out) {
    Path.toVector(Out);
    // Strip dots, but do not eliminate a path consisting only of '.'
    if (Out.size() != 1)
      llvm::sys::path::remove_dots(Out);
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

static llvm::Error diagnoseFileChange(IncludeTree::File F, ObjectRef Content) {
  auto FilenameBlob = F.getFilename();
  if (!FilenameBlob)
    return FilenameBlob.takeError();
  cas::ObjectStore &DB = F.getCAS();
  std::string Filename(FilenameBlob->getData());
  std::string OldID = DB.getID(Content).toString();
  std::string NewID = DB.getID(F.getContentsRef()).toString();
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "file '%s' changed during build; include-tree "
                                 "contents changed from %s to %s",
                                 Filename.c_str(), OldID.c_str(),
                                 NewID.c_str());
}

Expected<IntrusiveRefCntPtr<llvm::vfs::FileSystem>>
cas::createIncludeTreeFileSystem(IncludeTreeRoot &Root) {
  auto FileList = Root.getFileList();
  if (!FileList)
    return FileList.takeError();

  // Map from FilenameRef to ContentsRef.
  llvm::DenseMap<ObjectRef, ObjectRef> SeenContents;

  IntrusiveRefCntPtr<IncludeTreeFileSystem> IncludeTreeFS =
      new IncludeTreeFileSystem(Root.getCAS());
  llvm::Error E = FileList->forEachFile(
      [&](IncludeTree::File File,
          IncludeTree::FileList::FileSizeTy Size) -> llvm::Error {
        auto InsertPair = SeenContents.insert(
            std::make_pair(File.getFilenameRef(), File.getContentsRef()));
        if (!InsertPair.second) {
          if (InsertPair.first->second != File.getContentsRef())
            return diagnoseFileChange(File, InsertPair.first->second);
          return llvm::Error::success();
        }

        auto FilenameBlob = File.getFilename();
        if (!FilenameBlob)
          return FilenameBlob.takeError();

        SmallString<128> Filename(FilenameBlob->getData());
        // Strip './' in the filename to match the behaviour of ASTWriter; we
        // also strip './' in IncludeTreeFileSystem::getPath.
        assert(Filename != ".");
        llvm::sys::path::remove_dots(Filename);

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
