//===- CVPacks.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_CVPACKS_H
#define LLD_COFF_CVPACKS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/CVPack/IndexRemap.h"
#include "llvm/DebugInfo/CVPack/PackFile.h"
#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/TypeHashing.h"
#include "llvm/Support/Endian.h"
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace lld::coff {
class COFFLinkerContext;
class ObjFile;
class SectionChunk;

/// A pack of the link, and the state of type merging for its records.
struct LoadedCVPack {
  const llvm::codeview::cvpack::ResolvedCVPack *pack = nullptr;
  /// Set for every record referenced by a linked object. Only those records
  /// are hashed.
  std::vector<std::atomic<uint8_t>> used;
  std::vector<llvm::codeview::GloballyHashedType> gHashes;

  const llvm::codeview::cvpack::PackFileView &view() const {
    return pack->View;
  }
};

/// The logical type stream of a packed object, in local type index order, as
/// runs of the object's own (inline) records and of pack records. Objects
/// share long runs of pack records, so this is much smaller than a per-record
/// table: lookups binary-search the runs, and passes walk them in order.
class CVPackLogicalTypes {
public:
  struct Segment {
    /// Zero-based local index of the first record.
    uint32_t localFirst;
    uint32_t count;
    /// The zero-based pack index, or the index of the first inline record.
    uint32_t first;
    bool packed;
    bool isPacked() const { return packed; }
  };
  /// Where a logical record lives.
  struct Location {
    /// The zero-based pack index, or the inline record index.
    uint32_t index;
    bool packed;
    bool isPacked() const { return packed; }
  };

  void clear() {
    segments.clear();
    inlineOffsets.clear();
    numTypes = 0;
  }
  void addInline(uint32_t count, uint32_t firstInline) {
    add(false, firstInline, count);
  }
  void addPacked(uint32_t firstPackIndex, uint32_t count) {
    add(true, firstPackIndex, count);
  }
  /// The offsets of the inline records in the object's type stream.
  std::vector<uint32_t> &getInlineOffsets() { return inlineOffsets; }
  void shrinkToFit() {
    segments.shrink_to_fit();
    inlineOffsets.shrink_to_fit();
  }

  size_t size() const { return numTypes; }
  bool empty() const { return numTypes == 0; }
  size_t numInline() const { return inlineOffsets.size(); }
  llvm::ArrayRef<Segment> getSegments() const { return segments; }

  Location locate(uint32_t localIndex) const {
    auto it = llvm::upper_bound(
        segments, localIndex,
        [](uint32_t i, const Segment &s) { return i < s.localFirst; });
    assert(it != segments.begin());
    --it;
    assert(localIndex - it->localFirst < it->count);
    return {it->first + (localIndex - it->localFirst), it->packed};
  }

  /// The inline record at the given inline index, from the object's type
  /// stream.
  llvm::codeview::CVType inlineType(llvm::ArrayRef<uint8_t> stream,
                                    uint32_t inlineIndex) const {
    uint32_t offset = inlineOffsets[inlineIndex];
    uint16_t length =
        llvm::support::endian::read16le(stream.data() + offset) + 2;
    return llvm::codeview::CVType(stream.slice(offset, length));
  }

private:
  void add(bool packed, uint32_t first, uint32_t count) {
    if (count == 0)
      return;
    // Merge with the previous run when contiguous.
    if (!segments.empty()) {
      Segment &last = segments.back();
      if (last.packed == packed && last.first + last.count == first) {
        last.count += count;
        numTypes += count;
        return;
      }
    }
    segments.push_back({numTypes, count, first, packed});
    numTypes += count;
  }

  std::vector<Segment> segments;
  std::vector<uint32_t> inlineOffsets;
  uint32_t numTypes = 0;
};

/// The packs of a link. Thread-safe; a pack is loaded once.
class CVPackCache {
public:
  explicit CVPackCache(COFFLinkerContext &ctx);
  ~CVPackCache();

  /// Load the pack that file names by its content id. No option says where
  /// packs are: as the PDB of a type server, a pack is searched next to the
  /// object or the archive it comes from, then next to the output file. It is
  /// then searched as a library is, in the current directory and in those of
  /// /libpath and LIB.
  llvm::Expected<LoadedCVPack *> load(const llvm::codeview::cvpack::PackId &id,
                                      ObjFile *file);

private:
  COFFLinkerContext &ctx;
  std::mutex mutex;
  llvm::codeview::cvpack::CVPackResolver resolver;
  std::map<llvm::codeview::cvpack::PackId, std::unique_ptr<LoadedCVPack>> packs;
};

llvm::Expected<LoadedCVPack *>
loadCVPack(COFFLinkerContext &ctx, const llvm::codeview::cvpack::PackId &id,
           ObjFile *file);

/// Compute the global hashes of every used record of the given packs. Records
/// are hashed level by level, each after the records it references, and
/// each level in parallel.
llvm::Error computeCVPackGHashes(llvm::ArrayRef<LoadedCVPack *> packs);

/// Decode an object's packed type stream. A reference of a packed record that
/// the object does not map is rejected here when materializing, and otherwise
/// later, when the record is merged.
llvm::Error
expandPackedTypeStream(llvm::ArrayRef<uint8_t> stream, LoadedCVPack &pack,
                       bool materialize, CVPackLogicalTypes &logicalTypes,
                       std::vector<uint8_t> &expandedTypes,
                       llvm::codeview::cvpack::IndexRemap &packTypeIndexMap);

/// The record of a logical type of a packed object, in pack indices if it is
/// a pack record.
inline llvm::codeview::CVType
getCVPackLogicalType(const CVPackLogicalTypes &types,
                     llvm::ArrayRef<uint8_t> stream, const LoadedCVPack &pack,
                     CVPackLogicalTypes::Location l) {
  if (!l.isPacked())
    return types.inlineType(stream, l.index);
  return pack.view().Types[l.index];
}

/// Restore the live COMDAT .debug$S sections of a packed object whose function
/// debug info moved to a pack. getDebugContents then returns the original
/// bytes. Needs the object's pack index map.
llvm::Error expandCVPackFunctionDebug(ObjFile *file);

/// The contents of a debug section of an object, restored if its function
/// debug info moved to a pack and expandCVPackFunctionDebug ran.
llvm::ArrayRef<uint8_t> getDebugContents(const SectionChunk *c);

/// For diagnostics before the PDB is built: the content of a COMDAT .debug$S
/// section whose function debug info moved to a pack, with its line tables
/// restored. The type indices of its symbols stay those of the pack. Loads the
/// pack if the link did not. Returns nullopt for any other section, or if the
/// pack cannot be read.
std::optional<std::vector<uint8_t>>
restoreCVPackFunctionLines(const SectionChunk *c);
} // namespace lld::coff

#endif
