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

Expected<IncludeTreeRoot> IncludeTreeRoot::create(CASDB &DB,
                                                  ObjectRef MainFileTree) {
  assert(IncludeTree::isValid(DB, MainFileTree));
  return IncludeTreeBase::create(DB, MainFileTree, {});
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

llvm::Error IncludeTreeRoot::print(llvm::raw_ostream &OS, unsigned Indent) {
  Optional<cas::IncludeTree> MainTree;
  if (llvm::Error E = getMainFileTree().moveInto(MainTree))
    return E;
  return MainTree->print(OS, Indent);
}
