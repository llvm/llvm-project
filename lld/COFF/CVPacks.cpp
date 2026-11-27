//===- CVPacks.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CVPacks.h"
#include "COFFLinkerContext.h"
#include "Chunks.h"
#include "Config.h"
#include "Driver.h"
#include "InputFiles.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/CVPack/FunctionDebug.h"
#include "llvm/DebugInfo/CodeView/TypeHashing.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeIndexDiscovery.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include <limits>
#include <map>
#include <mutex>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::cvpack;
using namespace lld::coff;

static Error packError(const Twine &message) {
  return createStringError(inconvertibleErrorCode(), message);
}

CVPackCache::CVPackCache(COFFLinkerContext &ctx) : ctx(ctx) {}

CVPackCache::~CVPackCache() = default;

Expected<LoadedCVPack *> CVPackCache::load(const PackId &id, ObjFile *file) {
  std::lock_guard<std::mutex> lock(mutex);
  auto [it, inserted] = packs.try_emplace(id);
  if (!inserted && it->second)
    return it->second.get();
  // The directory of the object or of its archive, that of the output file,
  // then the library search paths.
  StringRef objPath =
      file->parentName.empty() ? file->getName() : file->parentName;
  SmallVector<StringRef, 16> directories;
  directories.push_back(sys::path::parent_path(objPath));
  directories.push_back(sys::path::parent_path(ctx.config.outputFile));
  append_range(directories, ctx.driver.getSearchPaths());
  auto resolved = resolver.get(id, /*VerifyContentId=*/true, directories);
  if (!resolved) {
    packs.erase(it);
    return packError(toString(resolved.takeError()) + "; required by " +
                     toString(file));
  }
  // In a reproducer, the pack goes to the directory that the reproduced link
  // runs from, which that link searches wherever this one found the pack.
  if (ctx.driver.tar) {
    const MemoryBuffer &buffer = *(*resolved)->Buffer;
    ctx.driver.tar->append(sys::path::filename(buffer.getBufferIdentifier()),
                           buffer.getBuffer());
  }
  auto pack = std::make_unique<LoadedCVPack>();
  pack->pack = *resolved;
  pack->used = std::vector<std::atomic<uint8_t>>(pack->view().Types.size());
  pack->gHashes.resize(pack->view().Types.size());
  it->second = std::move(pack);
  return it->second.get();
}

Expected<LoadedCVPack *>
lld::coff::loadCVPack(COFFLinkerContext &ctx, const PackId &id, ObjFile *file) {
  return ctx.cvPackCache->load(id, file);
}

/// Hash the used records of one pack. A record's level is one more than the
/// highest level among the records it references, so every record of a level
/// can be hashed in parallel once the previous levels are done.
static Error computeOnePackGHashes(LoadedCVPack &pack) {
  size_t count = pack.view().Types.size();
  // A pack may carry the hashes of its records.
  ArrayRef<uint8_t> stored = pack.view().GlobalHashes;
  if (!stored.empty() && stored.size() == 8 * count) {
    parallelFor(0, count, [&](size_t index) {
      if (pack.used[index].load(std::memory_order_relaxed))
        std::copy_n(stored.data() + 8 * index, 8,
                    pack.gHashes[index].Hash.begin());
    });
    return Error::success();
  }
  std::vector<uint32_t> level(count, 0);
  std::vector<uint32_t> levelSize;
  for (size_t index = 0; index != count; ++index) {
    if (!pack.used[index].load(std::memory_order_relaxed))
      continue;
    ArrayRef<uint8_t> record = pack.view().Types[index].RecordData;
    uint32_t myLevel = 0;
    if (Error e = forEachTypeIndex(
            record, false, [&](uint32_t offset, TiRefKind) -> Error {
              uint32_t value =
                  support::endian::read32le(record.data() + offset);
              if (value < TypeIndex::FirstNonSimpleIndex)
                return Error::success();
              uint32_t referenced = value - TypeIndex::FirstNonSimpleIndex;
              if (referenced >= index)
                return packError(
                    "CodeView pack record has a forward type reference");
              if (!pack.used[referenced].load(std::memory_order_relaxed))
                return packError("a linked object uses a CodeView pack record "
                                 "without the records it references");
              myLevel = std::max(myLevel, level[referenced] + 1);
              return Error::success();
            }))
      return e;
    level[index] = myLevel;
    if (levelSize.size() <= myLevel)
      levelSize.resize(myLevel + 1, 0);
    ++levelSize[myLevel];
  }

  // Bucket the used records by level.
  std::vector<uint32_t> levelStart(levelSize.size() + 1, 0);
  for (size_t l = 0; l != levelSize.size(); ++l)
    levelStart[l + 1] = levelStart[l] + levelSize[l];
  std::vector<uint32_t> order(levelStart.back());
  std::vector<uint32_t> fill(levelStart.begin(), levelStart.end() - 1);
  for (size_t index = 0; index != count; ++index)
    if (pack.used[index].load(std::memory_order_relaxed))
      order[fill[level[index]]++] = index;

  std::atomic<bool> failed{false};
  for (size_t l = 0; l != levelSize.size(); ++l) {
    parallelFor(levelStart[l], levelStart[l + 1], [&](size_t position) {
      uint32_t index = order[position];
      pack.gHashes[index] = GloballyHashedType::hashType(
          pack.view().Types[index], pack.gHashes, pack.gHashes);
      if (pack.gHashes[index].empty())
        failed = true;
    });
  }
  if (failed)
    return packError("unresolved type reference while hashing CodeView pack");
  return Error::success();
}

Error lld::coff::computeCVPackGHashes(ArrayRef<LoadedCVPack *> packs) {
  for (LoadedCVPack *pack : packs)
    if (Error e = computeOnePackGHashes(*pack))
      return e;
  return Error::success();
}

Error lld::coff::expandPackedTypeStream(ArrayRef<uint8_t> stream,
                                        LoadedCVPack &pack, bool materialize,
                                        CVPackLogicalTypes &logicalTypes,
                                        std::vector<uint8_t> &expandedTypes,
                                        IndexRemap &packTypeIndexMap) {
  logicalTypes.clear();
  expandedTypes.clear();
  CVTypeArray records;
  BinaryStreamReader reader(stream, llvm::endianness::little);
  if (Error e = reader.readArray(records, reader.getLength()))
    return e;
  if (records.begin() == records.end() ||
      records.begin()->kind() != TypeLeafKind::LF_CVPACK_LIST)
    return packError("packed type stream has no leading LF_CVPACK_LIST");

  auto id = decodePackList(*records.begin());
  if (!id)
    return id.takeError();
  if (*id != pack.view().ContentId)
    return packError("loaded CodeView pack does not match LF_CVPACK_LIST");

  std::vector<CVType> mapRecords;
  std::vector<CVType> inlineRecords;
  bool sawOrigin = false;
  bool sawInline = false;
  auto it = records.begin();
  ++it;
  for (; it != records.end(); ++it) {
    // A link does not use the origin record, which only restores the object.
    if (it->kind() == TypeLeafKind::LF_CVPACK_ORIGIN && !sawOrigin &&
        mapRecords.empty() && !sawInline) {
      sawOrigin = true;
      continue;
    }
    if (it->kind() == TypeLeafKind::LF_CVPACK_MAP && !sawInline) {
      mapRecords.push_back(*it);
      continue;
    }
    if (it->kind() == TypeLeafKind::LF_CVPACK_LIST ||
        it->kind() == TypeLeafKind::LF_CVPACK_ORIGIN ||
        it->kind() == TypeLeafKind::LF_CVPACK_MAP)
      return packError(
          "misplaced or duplicate CodeView pack bookkeeping record");
    sawInline = true;
    inlineRecords.push_back(*it);
  }

  auto map = decodeTypeMap(mapRecords);
  if (!map)
    return map.takeError();
  auto segments = expandRangeMap(
      *map, TypeIndex::FirstNonSimpleIndex + pack.view().Types.size(),
      inlineRecords.size(), TypeIndex::FirstNonSimpleIndex);
  if (!segments)
    return segments.takeError();

  IndexRemap indexMap;
  uint64_t localIndex = TypeIndex::FirstNonSimpleIndex;
  for (const RangeMapSegment &segment : *segments) {
    if (segment.IsPackReference)
      indexMap.add(static_cast<uint32_t>(segment.FirstIndex),
                   static_cast<uint32_t>(segment.Count),
                   static_cast<uint32_t>(localIndex));
    localIndex += segment.Count;
  }
  if (Error e = indexMap.finalize())
    return e;

  if (!materialize) {
    std::vector<uint32_t> &offsets = logicalTypes.getInlineOffsets();
    offsets.reserve(inlineRecords.size());
    for (const CVType &type : inlineRecords)
      offsets.push_back(
          static_cast<uint32_t>(type.RecordData.data() - stream.data()));
  }
  size_t inlineIndex = 0;
  for (const RangeMapSegment &segment : *segments) {
    if (!segment.IsPackReference) {
      if (!materialize) {
        logicalTypes.addInline(static_cast<uint32_t>(segment.Count),
                               static_cast<uint32_t>(inlineIndex));
        inlineIndex += segment.Count;
        continue;
      }
      for (uint64_t i = 0; i != segment.Count; ++i) {
        ArrayRef<uint8_t> data = inlineRecords[inlineIndex++].RecordData;
        expandedTypes.insert(expandedTypes.end(), data.begin(), data.end());
      }
      continue;
    }
    if (!materialize)
      logicalTypes.addPacked(
          static_cast<uint32_t>(segment.FirstIndex -
                                TypeIndex::FirstNonSimpleIndex),
          static_cast<uint32_t>(segment.Count));
    for (uint64_t i = 0; i != segment.Count; ++i) {
      uint64_t sourceIndex = segment.FirstIndex + i;
      uint32_t arrayIndex =
          static_cast<uint32_t>(sourceIndex - TypeIndex::FirstNonSimpleIndex);
      ArrayRef<uint8_t> recordData = pack.view().Types[arrayIndex].RecordData;
      if (materialize) {
        std::vector<uint8_t> copy(recordData.begin(), recordData.end());
        if (Error e = remapTypeIndices(copy, false, indexMap))
          return e;
        expandedTypes.insert(expandedTypes.end(), copy.begin(), copy.end());
        continue;
      }
      pack.used[arrayIndex].store(1, std::memory_order_relaxed);
    }
  }
  logicalTypes.shrinkToFit();
  packTypeIndexMap = std::move(indexMap);
  return Error::success();
}

Error lld::coff::expandCVPackFunctionDebug(ObjFile *file) {
  struct Stub {
    SectionChunk *chunk;
    FunctionRef ref;
  };
  std::vector<Stub> stubs;
  uint64_t total = 0;
  for (SectionChunk *c : file->getDebugChunks()) {
    if (!c->live || !c->isCOMDAT() || c->getSectionName() != ".debug$S")
      continue;
    if (std::optional<FunctionRef> ref = decodeFunctionRef(c->getContents())) {
      stubs.push_back({c, *ref});
      total += alignTo(ref->OriginalSize, 4);
    }
  }
  if (stubs.empty())
    return Error::success();

  std::vector<ArrayRef<uint8_t>> mainSections;
  for (SectionChunk *c : file->getDebugChunks())
    if (!c->isCOMDAT() && c->getSectionName() == ".debug$S")
      mainSections.push_back(c->getContents());
  auto files = ObjectFileTable::create(mainSections);
  if (!files)
    return files.takeError();

  file->cvPackFunctionStorage.resize(total);
  uint8_t *out = file->cvPackFunctionStorage.data();
  DenseMap<const SectionChunk *, ArrayRef<uint8_t>> expanded;
  expanded.reserve(stubs.size());
  // pack file -> checksum offset in this object.
  DenseMap<uint32_t, uint32_t> fileOffsets;
  std::vector<uint8_t> scratch;
  for (const Stub &s : stubs) {
    const FunctionRef &ref = s.ref;
    if (ref.Slot != 0 || ref.Ordinal >= file->cvPack->view().Functions.size())
      return createStringError("function reference names a missing pack entry");
    const PackFileView &view = file->cvPack->view();
    ArrayRef<uint8_t> blob = view.Functions[ref.Ordinal];
    const IndexRemap &map = file->packTypeIndexMap;
    scratch.clear();
    if (Error e = transformFunctionDebug(
            blob, scratch,
            [&](uint32_t index, uint32_t) -> Expected<uint32_t> {
              if (std::optional<uint32_t> local = map.lookup(index))
                return *local;
              return createStringError(
                  "pack type reference is absent from map");
            },
            [&](uint32_t packFile, uint32_t) -> Expected<uint32_t> {
              auto it = fileOffsets.find(packFile);
              if (it != fileOffsets.end())
                return it->second;
              std::optional<uint32_t> offset;
              if (packFile < view.Files.size())
                offset = files->lookup(view.Files[packFile]);
              if (!offset)
                return createStringError("pack file is absent from the object");
              fileOffsets.try_emplace(packFile, *offset);
              return *offset;
            },
            FileOperandEncoding::Minimal))
      return e;
    if (uint64_t(scratch.size()) + 4 != ref.OriginalSize)
      return createStringError("function reference has the wrong size");
    support::endian::write32le(out, COFF::DEBUG_SECTION_MAGIC);
    memcpy(out + 4, scratch.data(), scratch.size());
    expanded.try_emplace(s.chunk, ArrayRef<uint8_t>(out, ref.OriginalSize));
    out += alignTo(ref.OriginalSize, 4);
  }
  file->cvPackFunctionDebug = std::move(expanded);
  return Error::success();
}

ArrayRef<uint8_t> lld::coff::getDebugContents(const SectionChunk *c) {
  const auto &restored = c->file->cvPackFunctionDebug;
  if (LLVM_UNLIKELY(!restored.empty())) {
    auto it = restored.find(c);
    if (it != restored.end())
      return it->second;
  }
  return c->getContents();
}

std::optional<std::vector<uint8_t>>
lld::coff::restoreCVPackFunctionLines(const SectionChunk *c) {
  if (!c->isCOMDAT() || c->getSectionName() != ".debug$S")
    return std::nullopt;
  std::optional<FunctionRef> ref = decodeFunctionRef(getDebugContents(c));
  if (!ref)
    return std::nullopt;
  ObjFile *file = c->file;
  LoadedCVPack *pack = file->cvPack;
  if (!pack) {
    // A link without /debug does not load the packs.
    SectionChunk *types =
        SectionChunk::findByName(file->getDebugChunks(), ".debug$T");
    if (!types || !isPackedTypeStream(types->getContents()))
      return std::nullopt;
    ArrayRef<uint8_t> stream =
        types->getContents().drop_front(sizeof(uint32_t));
    size_t listSize = size_t(support::endian::read16le(stream.data())) + 2;
    if (listSize > stream.size())
      return std::nullopt;
    auto id = decodePackList(CVType(stream.take_front(listSize)));
    if (!id) {
      consumeError(id.takeError());
      return std::nullopt;
    }
    auto loaded = loadCVPack(file->symtab.ctx, *id, file);
    if (!loaded) {
      consumeError(loaded.takeError());
      return std::nullopt;
    }
    pack = *loaded;
  }
  if (ref->Slot != 0 || ref->Ordinal >= pack->view().Functions.size())
    return std::nullopt;
  const PackFileView &view = pack->view();

  std::vector<ArrayRef<uint8_t>> mainSections;
  for (SectionChunk *dbgC : file->getDebugChunks())
    if (!dbgC->isCOMDAT() && dbgC->getSectionName() == ".debug$S")
      mainSections.push_back(dbgC->getContents());
  auto files = ObjectFileTable::create(mainSections);
  if (!files) {
    consumeError(files.takeError());
    return std::nullopt;
  }
  std::vector<uint8_t> out;
  out.reserve(ref->OriginalSize);
  out.resize(4);
  support::endian::write32le(out.data(), COFF::DEBUG_SECTION_MAGIC);
  Error e = transformFunctionDebug(
      view.Functions[ref->Ordinal], out,
      [](uint32_t index, uint32_t) -> Expected<uint32_t> { return index; },
      [&](uint32_t packFile, uint32_t) -> Expected<uint32_t> {
        std::optional<uint32_t> offset;
        if (packFile < view.Files.size())
          offset = files->lookup(view.Files[packFile]);
        if (!offset)
          return createStringError("pack file is absent from the object");
        return *offset;
      },
      FileOperandEncoding::Minimal);
  if (e || out.size() != ref->OriginalSize) {
    consumeError(std::move(e));
    return std::nullopt;
  }
  return out;
}
