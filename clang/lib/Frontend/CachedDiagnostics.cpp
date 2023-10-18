//===- CachedDiagnostics.h - Serializing diagnostics for caching ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a mechanism for caching diagnostics with full source location
// fidelity. It serializes the subset of SourceManager state for the diagnostic
// source locations (FileIDs, macro expansions, etc.) and replays diagnostics by
// materializing their SourceLocations; that way the diagnostic consumers have
// full access to the include stack, expanded macros, etc.
//
// This allows us to only cache the diagnostic information that is essential for
// the compilation and ignore the various ways that diagnostics can be rendered.
//
//===----------------------------------------------------------------------===//

#include "CachedDiagnostics.h"
#include "clang/Basic/DiagnosticCAS.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/YAMLTraits.h"
#include <variant>

using namespace clang;
using namespace clang::cas;
using namespace llvm;

namespace {

namespace cached_diagnostics {

struct Location {
  unsigned SLocIdx;
  unsigned Offset;
};

struct Range {
  std::optional<Location> Begin;
  std::optional<Location> End;
  bool IsTokenRange;
};

struct FixItHint {
  std::optional<Range> RemoveRange;
  std::optional<Range> InsertFromRange;
  std::string CodeToInsert;
  bool BeforePreviousInsertions;
};

struct SLocEntry {
  struct FileInfo {
    std::string Filename;
    // Using \c shared_ptr instead of \c unique_ptr to accomodate the YAML
    // [de]serialization functions.
    std::shared_ptr<MemoryBuffer> Buffer;
    std::optional<Location> IncludeLoc;

    // Only used during compilation.
    bool IsScratchBuffer = false;
    // If this is for a scratch buffer, records the highest offset that a
    // converted location resolved inside it. Only used during compilation.
    unsigned HighestScratchBufferOffset = 0;
  };

  struct ExpansionInfo {
    std::optional<Location> SpellingLoc;
    std::optional<Location> ExpansionStartLoc;
    std::optional<Location> ExpansionEndLoc;
    unsigned Length;
    bool IsTokenRange;
  };

  std::variant<FileInfo, ExpansionInfo> Data = FileInfo();

  bool isFileInfo() const { return std::holds_alternative<FileInfo>(Data); }

  FileInfo &getAsFileInfo() {
    assert(isFileInfo());
    return std::get<FileInfo>(Data);
  }
  const FileInfo &getAsFileInfo() const {
    assert(isFileInfo());
    return std::get<FileInfo>(Data);
  }

  ExpansionInfo &getAsExpansionInfo() {
    assert(!isFileInfo());
    return std::get<ExpansionInfo>(Data);
  }
  const ExpansionInfo &getAsExpansionInfo() const {
    assert(!isFileInfo());
    return std::get<ExpansionInfo>(Data);
  }
};

struct Diagnostic {
  unsigned ID;
  DiagnosticsEngine::Level Level;
  std::string Message;
  std::optional<Location> Loc;
  std::vector<Range> Ranges;
  std::vector<FixItHint> FixIts;
};

struct Diagnostics {
  std::vector<SLocEntry> SLocEntries;
  std::vector<Diagnostic> Diags;

  size_t getNumDiags() const { return Diags.size(); }

  void clear() {
    SLocEntries.clear();
    Diags.clear();
  }
};

} // namespace cached_diagnostics

/// Converts diagnostics from/to the \p cached_diagnostics structures, and
/// de/serializes diagnostics from/to a buffer. It "materializes" source
/// locations using its own \p SourceManager.
struct CachedDiagnosticSerializer {
  PrefixMapper &Mapper;
  DiagnosticsEngine DiagEngine;
  SourceManager SourceMgr;
  /// Diagnostics either emitted during compilation or deserialized from a
  /// buffer.
  cached_diagnostics::Diagnostics CachedDiags;
  /// Associates a \p FileID with the index of a
  /// \p cached_diagnostics::SLocEntry object.
  SmallDenseMap<FileID, unsigned> FileIDToCachedSLocIdx;
  /// Associates the indices of \p cached_diagnostics::SLocEntry objects with
  /// their "materialized" \p FileID.
  SmallVector<FileID> FileIDBySlocIdx;
  /// Keeps track of \p MemoryBuffers for ownership purposes.
  SmallVector<std::unique_ptr<MemoryBuffer>> FileBuffers;
  BumpPtrAllocator Alloc;
  StringSaver Saver{Alloc};

  CachedDiagnosticSerializer(PrefixMapper &Mapper, FileManager &FileMgr)
      : Mapper(Mapper),
        DiagEngine(new DiagnosticIDs(), new DiagnosticOptions()),
        SourceMgr(DiagEngine, FileMgr) {}

  size_t getNumDiags() const { return CachedDiags.getNumDiags(); }
  bool empty() const { return getNumDiags() == 0; }
  void clear() { CachedDiags.clear(); }

  /// \returns the index of the created \p cached_diagnostics::Diagnostic
  /// object.
  unsigned addDiag(const StoredDiagnostic &Diag);
  StoredDiagnostic getDiag(unsigned Idx);

  std::optional<cached_diagnostics::Location>
  convertLoc(const FullSourceLoc &Loc);
  FullSourceLoc convertCachedLoc(
      const std::optional<cached_diagnostics::Location> &CachedLoc);

  std::optional<cached_diagnostics::Range>
  convertRange(const CharSourceRange &Range, const SourceManager &SM);
  CharSourceRange convertCachedRange(
      const std::optional<cached_diagnostics::Range> &CachedRange);

  cached_diagnostics::FixItHint convertFixIt(const FixItHint &FixIt,
                                             const SourceManager &SM);
  FixItHint
  convertCachedFixIt(const cached_diagnostics::FixItHint &CachedFixIt);

  unsigned convertFileID(FileID FID, const SourceManager &SM);
  FileID convertCachedSLocEntry(unsigned Idx);

  FileID getExistingFileIDForCachedSLocEntry(unsigned Idx) const {
    if (Idx >= FileIDBySlocIdx.size())
      return FileID();
    return FileIDBySlocIdx[Idx];
  }

  /// \returns a serialized buffer of the currently recorded
  /// \p cached_diagnostics::Diagnostics, or \p std::nullopt if there's no
  /// diagnostic. The buffer can be passed to \p deserializeCachedDiagnostics to
  /// get back the same diagnostics.
  ///
  /// There is no stability guarantee for the format of the buffer, the
  /// expectation is that the buffer will be deserialized only by the same
  /// compiler version that produced it. The format can change without
  /// restrictions.
  ///
  /// The intended use is as implementation detail of compilation caching, where
  /// the diagnostic output is associated with a compilation cache key. A
  /// different compiler version will create different cache keys, which ensures
  /// that the diagnostics buffer will only be read by the same compiler that
  /// produced it.
  std::optional<std::string> serializeEmittedDiagnostics();
  Error deserializeCachedDiagnostics(StringRef Buffer);
};

} // anonymous namespace

unsigned CachedDiagnosticSerializer::addDiag(const StoredDiagnostic &Diag) {
  cached_diagnostics::Diagnostic CachedDiag;
  CachedDiag.ID = Diag.getID();
  CachedDiag.Level = Diag.getLevel();
  CachedDiag.Message = Diag.getMessage();
  CachedDiag.Loc = convertLoc(Diag.getLocation());
  if (Diag.getLocation().isValid()) {
    const SourceManager &SM = Diag.getLocation().getManager();
    for (const CharSourceRange &Range : Diag.getRanges()) {
      if (std::optional<cached_diagnostics::Range> CachedRange =
              convertRange(Range, SM))
        CachedDiag.Ranges.push_back(std::move(*CachedRange));
    }
    for (const FixItHint &FixIt : Diag.getFixIts()) {
      CachedDiag.FixIts.push_back(convertFixIt(FixIt, SM));
    }
  }

  unsigned Idx = CachedDiags.Diags.size();
  CachedDiags.Diags.push_back(std::move(CachedDiag));
  return Idx;
}

StoredDiagnostic CachedDiagnosticSerializer::getDiag(unsigned Idx) {
  assert(Idx < getNumDiags());
  const cached_diagnostics::Diagnostic &CachedDiag = CachedDiags.Diags[Idx];
  FullSourceLoc Loc = convertCachedLoc(CachedDiag.Loc);
  SmallVector<CharSourceRange> Ranges;
  for (const cached_diagnostics::Range &CachedRange : CachedDiag.Ranges) {
    Ranges.push_back(convertCachedRange(CachedRange));
  }
  SmallVector<FixItHint> FixIts;
  for (const cached_diagnostics::FixItHint &CachedFixIt : CachedDiag.FixIts) {
    FixIts.push_back(convertCachedFixIt(CachedFixIt));
  }

  StoredDiagnostic Diag(CachedDiag.Level, CachedDiag.ID, CachedDiag.Message,
                        Loc, Ranges, FixIts);
  return Diag;
}

std::optional<cached_diagnostics::Location>
CachedDiagnosticSerializer::convertLoc(const FullSourceLoc &Loc) {
  if (Loc.isInvalid())
    return std::nullopt;

  FileID FID;
  unsigned Offset;
  std::tie(FID, Offset) = Loc.getDecomposedLoc();
  cached_diagnostics::Location CachedLoc;
  CachedLoc.SLocIdx = convertFileID(FID, Loc.getManager());
  CachedLoc.Offset = Offset;

  auto &SLocEntry = CachedDiags.SLocEntries[CachedLoc.SLocIdx];
  if (SLocEntry.isFileInfo()) {
    auto &FI = SLocEntry.getAsFileInfo();
    if (FI.IsScratchBuffer) {
      if (FI.HighestScratchBufferOffset < Offset)
        FI.HighestScratchBufferOffset = Offset;
      FileID CachedScratchFID =
          getExistingFileIDForCachedSLocEntry(CachedLoc.SLocIdx);
      if (CachedScratchFID.isValid()) {
        // Since the scratch buffer updates while parsing, use the line cache
        // from the compilation's source manager, to make sure it is up-to-date.
        auto LineMap = Loc.getManager()
                           .getSLocEntry(FID)
                           .getFile()
                           .getContentCache()
                           .SourceLineCache;
        SourceMgr.getSLocEntry(CachedScratchFID)
            .getFile()
            .getContentCache()
            .SourceLineCache = LineMap;
      }
    }
  }

  return CachedLoc;
}

FullSourceLoc CachedDiagnosticSerializer::convertCachedLoc(
    const std::optional<cached_diagnostics::Location> &CachedLoc) {
  if (!CachedLoc)
    return FullSourceLoc();

  FileID FID = convertCachedSLocEntry(CachedLoc->SLocIdx);
  SourceLocation Loc = SourceMgr.getComposedLoc(FID, CachedLoc->Offset);
  return FullSourceLoc(Loc, SourceMgr);
}

std::optional<cached_diagnostics::Range>
CachedDiagnosticSerializer::convertRange(const CharSourceRange &Range,
                                         const SourceManager &SM) {
  if (Range.isInvalid())
    return std::nullopt;

  cached_diagnostics::Range CachedRange;
  CachedRange.Begin = convertLoc(FullSourceLoc(Range.getBegin(), SM));
  CachedRange.End = convertLoc(FullSourceLoc(Range.getEnd(), SM));
  CachedRange.IsTokenRange = Range.isTokenRange();
  return CachedRange;
}

CharSourceRange CachedDiagnosticSerializer::convertCachedRange(
    const std::optional<cached_diagnostics::Range> &CachedRange) {
  if (!CachedRange)
    return CharSourceRange();

  FullSourceLoc Begin = convertCachedLoc(CachedRange->Begin);
  FullSourceLoc End = convertCachedLoc(CachedRange->End);
  return CharSourceRange(SourceRange(Begin, End), CachedRange->IsTokenRange);
}

cached_diagnostics::FixItHint
CachedDiagnosticSerializer::convertFixIt(const FixItHint &FixIt,
                                         const SourceManager &SM) {
  cached_diagnostics::FixItHint CachedFixIt;
  CachedFixIt.RemoveRange = convertRange(FixIt.RemoveRange, SM);
  CachedFixIt.InsertFromRange = convertRange(FixIt.InsertFromRange, SM);
  CachedFixIt.CodeToInsert = FixIt.CodeToInsert;
  CachedFixIt.BeforePreviousInsertions = FixIt.BeforePreviousInsertions;
  return CachedFixIt;
}

FixItHint CachedDiagnosticSerializer::convertCachedFixIt(
    const cached_diagnostics::FixItHint &CachedFixIt) {
  FixItHint FixIt;
  FixIt.RemoveRange = convertCachedRange(CachedFixIt.RemoveRange);
  FixIt.InsertFromRange = convertCachedRange(CachedFixIt.InsertFromRange);
  FixIt.CodeToInsert = CachedFixIt.CodeToInsert;
  FixIt.BeforePreviousInsertions = CachedFixIt.BeforePreviousInsertions;
  return FixIt;
}

unsigned CachedDiagnosticSerializer::convertFileID(FileID FID,
                                                   const SourceManager &SM) {
  auto Found = FileIDToCachedSLocIdx.find(FID);
  if (Found != FileIDToCachedSLocIdx.end())
    return Found->second;

  cached_diagnostics::SLocEntry CachedEntry;
  const SrcMgr::SLocEntry &Entry = SM.getSLocEntry(FID);
  if (Entry.isFile()) {
    const SrcMgr::FileInfo &FI = Entry.getFile();
    cached_diagnostics::SLocEntry::FileInfo CachedFI;
    if (OptionalFileEntryRef FE = FI.getContentCache().ContentsEntry) {
      CachedFI.Filename = FE->getName();
    } else {
      CachedFI.Filename = FI.getName();
      CachedFI.Buffer =
          MemoryBuffer::getMemBuffer(*FI.getContentCache().getBufferIfLoaded());
    }
    CachedFI.IncludeLoc = convertLoc(FullSourceLoc(FI.getIncludeLoc(), SM));
    CachedFI.IsScratchBuffer =
        SM.isWrittenInScratchSpace(SM.getLocForStartOfFile(FID));
    CachedEntry.Data = std::move(CachedFI);
  } else {
    const SrcMgr::ExpansionInfo &EI = Entry.getExpansion();
    cached_diagnostics::SLocEntry::ExpansionInfo CachedEI;
    CachedEI.Length = SM.getFileIDSize(FID);
    CachedEI.SpellingLoc = convertLoc(FullSourceLoc(EI.getSpellingLoc(), SM));
    CachedEI.ExpansionStartLoc =
        convertLoc(FullSourceLoc(EI.getExpansionLocStart(), SM));
    CachedEI.ExpansionEndLoc =
        convertLoc(FullSourceLoc(EI.getUnderlyingExpansionLocEnd(), SM));
    CachedEI.IsTokenRange = EI.isExpansionTokenRange();
    CachedEntry.Data = std::move(CachedEI);
  }

  unsigned Idx = CachedDiags.SLocEntries.size();
  CachedDiags.SLocEntries.push_back(std::move(CachedEntry));
  FileIDToCachedSLocIdx[FID] = Idx;
  return Idx;
}

FileID CachedDiagnosticSerializer::convertCachedSLocEntry(unsigned Idx) {
  if (Idx >= FileIDBySlocIdx.size())
    FileIDBySlocIdx.resize(Idx + 1);
  if (FileIDBySlocIdx[Idx].isValid())
    return FileIDBySlocIdx[Idx];

  const cached_diagnostics::SLocEntry &CachedSLocEntry =
      CachedDiags.SLocEntries[Idx];
  FileID FID;
  if (CachedSLocEntry.isFileInfo()) {
    const auto &FI = CachedSLocEntry.getAsFileInfo();
    FullSourceLoc IncludeLoc = convertCachedLoc(FI.IncludeLoc);
    if (FI.Buffer) {
      FID = SourceMgr.createFileID(FI.Buffer->getMemBufferRef(), SrcMgr::C_User,
                                   /*LoadedID*/ 0, /*LoadedOffset*/ 0,
                                   IncludeLoc);
    } else {
      auto MemBufOrErr =
          SourceMgr.getFileManager().getBufferForFile(FI.Filename);
      if (!MemBufOrErr)
        report_fatal_error(
            createFileError(FI.Filename, MemBufOrErr.getError()));
      SmallString<128> PathBuf;
      Mapper.map(FI.Filename, PathBuf);
      if (PathBuf.str() != FI.Filename) {
        // The file path was remapped. Keep the original buffer and pass a new
        // buffer using the remapped file path.
        FileBuffers.push_back(std::move(*MemBufOrErr));
        MemoryBufferRef NewBuffer(FileBuffers.back()->getBuffer(),
                                  Saver.save(PathBuf.str()));
        // Using \p SrcMgr::C_User as default since \p
        // SrcMgr::CharacteristicKind is irrelevant for diagnostic consumers; if
        // it becomes relevant we need to serialize this value as well.
        FID =
            SourceMgr.createFileID(NewBuffer, SrcMgr::C_User, 0, 0, IncludeLoc);
      } else {
        FID = SourceMgr.createFileID(std::move(*MemBufOrErr), SrcMgr::C_User, 0,
                                     0, IncludeLoc);
      }
    }
  } else {
    const auto &EI = CachedSLocEntry.getAsExpansionInfo();
    FullSourceLoc SpellingLoc = convertCachedLoc(EI.SpellingLoc);
    FullSourceLoc ExpansionStartLoc = convertCachedLoc(EI.ExpansionStartLoc);
    FullSourceLoc ExpansionEndLoc = convertCachedLoc(EI.ExpansionEndLoc);
    SourceLocation ExpansionLoc = SourceMgr.createExpansionLoc(
        SpellingLoc, ExpansionStartLoc, ExpansionEndLoc, EI.Length,
        EI.IsTokenRange);
    FID = SourceMgr.getDecomposedLoc(ExpansionLoc).first;
  }

  assert(FID.isValid());
  FileIDBySlocIdx[Idx] = FID;
  return FID;
}

namespace llvm::yaml {
template <> struct MappingTraits<cached_diagnostics::SLocEntry> {
  static void mapping(IO &io, cached_diagnostics::SLocEntry &s) {
    if (io.outputting()) {
      if (s.isFileInfo()) {
        io.mapRequired("file", s.getAsFileInfo());
      } else {
        io.mapRequired("expansion", s.getAsExpansionInfo());
      }
    } else {
      std::optional<cached_diagnostics::SLocEntry::FileInfo> FI;
      io.mapOptional("file", FI);
      if (FI) {
        s.Data = std::move(*FI);
      } else {
        cached_diagnostics::SLocEntry::ExpansionInfo EI;
        io.mapRequired("expansion", EI);
        s.Data = std::move(EI);
      }
    }
  }
};

template <> struct MappingTraits<cached_diagnostics::SLocEntry::FileInfo> {
  static void mapping(IO &io, cached_diagnostics::SLocEntry::FileInfo &s) {
    io.mapRequired("filename", s.Filename);
    io.mapOptional("include_loc", s.IncludeLoc);
    if (io.outputting()) {
      if (s.Buffer) {
        StringRef Contents = s.Buffer->getBuffer();
        if (s.IsScratchBuffer) {
          // The allocated buffer is large (~4K) but commonly a very small part
          // of that is used, so we truncate to the useful part only.
          // There's a '\0' between each addition in the scratch space, look for
          // the end of the highest referenced section.
          size_t EndIdx =
              Contents.find('\0', /*From=*/s.HighestScratchBufferOffset);
          Contents = Contents.substr(0, EndIdx);
        }
        std::string EncodedContents = encodeBase64(Contents);
        io.mapRequired("buffer", EncodedContents);
      }
    } else {
      std::optional<std::string> EncodedContents;
      io.mapOptional("buffer", EncodedContents);
      if (EncodedContents) {
        std::vector<char> Decoded;
        cantFail(decodeBase64(*EncodedContents, Decoded));
        s.Buffer = MemoryBuffer::getMemBufferCopy(
            StringRef(Decoded.data(), Decoded.size()), s.Filename);
      }
    }
  }
};

template <> struct MappingTraits<cached_diagnostics::SLocEntry::ExpansionInfo> {
  static void mapping(IO &io, cached_diagnostics::SLocEntry::ExpansionInfo &s) {
    io.mapRequired("length", s.Length);
    io.mapOptional("spelling_loc", s.SpellingLoc);
    io.mapOptional("expansion_start_loc", s.ExpansionStartLoc);
    io.mapOptional("expansion_end_loc", s.ExpansionEndLoc);
    io.mapRequired("is_token_range", s.IsTokenRange);
  }
};

template <> struct MappingTraits<cached_diagnostics::Location> {
  static void mapping(IO &io, cached_diagnostics::Location &s) {
    io.mapRequired("sloc_idx", s.SLocIdx);
    io.mapRequired("offset", s.Offset);
  }
};

template <> struct MappingTraits<cached_diagnostics::Range> {
  static void mapping(IO &io, cached_diagnostics::Range &s) {
    io.mapOptional("begin", s.Begin);
    io.mapOptional("end", s.End);
    io.mapRequired("is_token_range", s.IsTokenRange);
  }
};

template <> struct MappingTraits<cached_diagnostics::FixItHint> {
  static void mapping(IO &io, cached_diagnostics::FixItHint &s) {
    io.mapOptional("remove_range", s.RemoveRange);
    io.mapOptional("insert_range", s.InsertFromRange);
    io.mapOptional("code", s.CodeToInsert);
    io.mapRequired("before_previous", s.BeforePreviousInsertions);
  }
};

template <> struct MappingTraits<cached_diagnostics::Diagnostic> {
  static void mapping(IO &io, cached_diagnostics::Diagnostic &s) {
    io.mapRequired("id", s.ID);
    io.mapRequired("level", (unsigned &)s.Level);
    io.mapRequired("message", s.Message);
    io.mapOptional("loc", s.Loc);
    io.mapOptional("ranges", s.Ranges);
    io.mapOptional("fixits", s.FixIts);
  }
};

template <> struct MappingTraits<cached_diagnostics::Diagnostics> {
  static void mapping(IO &io, cached_diagnostics::Diagnostics &s) {
    io.mapRequired("sloc_entries", s.SLocEntries);
    io.mapRequired("diagnostics", s.Diags);
  }
};
} // namespace llvm::yaml

LLVM_YAML_IS_SEQUENCE_VECTOR(cached_diagnostics::SLocEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(cached_diagnostics::Diagnostic)
LLVM_YAML_IS_SEQUENCE_VECTOR(cached_diagnostics::Range)
LLVM_YAML_IS_SEQUENCE_VECTOR(cached_diagnostics::FixItHint)

std::optional<std::string>
CachedDiagnosticSerializer::serializeEmittedDiagnostics() {
  if (empty())
    return std::nullopt;

  SmallString<512> Buf;
  raw_svector_ostream OS(Buf);
  yaml::Output YOut(OS);
  YOut << CachedDiags;
  StringRef YamlContents = OS.str();

  // Use compression to reduce the size of the yaml output. Note that we don't
  // need to track whether compression was used or not, see doc-comments of \p
  // serializeEmittedDiagnostics().
  SmallVector<uint8_t, 512> CompressedBuffer;
  if (compression::zstd::isAvailable()) {
    compression::zstd::compress(arrayRefFromStringRef(YamlContents),
                                CompressedBuffer);

  } else if (compression::zlib::isAvailable()) {
    compression::zlib::compress(arrayRefFromStringRef(YamlContents),
                                CompressedBuffer);
  }
  if (!CompressedBuffer.empty()) {
    raw_svector_ostream BufOS((SmallVectorImpl<char> &)CompressedBuffer);
    support::endian::Writer Writer(BufOS, endianness::little);
    Writer.write(uint32_t(YamlContents.size()));
    return toStringRef(CompressedBuffer).str();
  }

  return YamlContents.str();
}

Error CachedDiagnosticSerializer::deserializeCachedDiagnostics(
    StringRef Buffer) {

  StringRef YamlContents;
  SmallVector<uint8_t, 512> UncompressedBuffer;
  if (compression::zstd::isAvailable() || compression::zlib::isAvailable()) {
    uint32_t UncompressedSize =
        support::endian::read<uint32_t, llvm::endianness::little>(
            Buffer.data() + Buffer.size() - sizeof(uint32_t));
    StringRef CompressedData = Buffer.drop_back(sizeof(uint32_t));
    if (compression::zstd::isAvailable()) {
      if (Error E = compression::zstd::decompress(
              arrayRefFromStringRef(CompressedData), UncompressedBuffer,
              UncompressedSize)) {
        return E;
      }
    } else {
      if (Error E = compression::zlib::decompress(
              arrayRefFromStringRef(CompressedData), UncompressedBuffer,
              UncompressedSize)) {
        return E;
      }
    }
    YamlContents = toStringRef(UncompressedBuffer);
  } else {
    YamlContents = Buffer;
  }

  CachedDiags.clear();
  yaml::Input YIn(YamlContents);
  YIn >> CachedDiags;
  if (YIn.error())
    return createStringError(YIn.error(),
                             "failed deserializing cached diagnostics");
  return Error::success();
}

/// Captures diagnostics emitted during compilation while also passing them
/// along to the original consumer.
struct CachingDiagnosticsProcessor::DiagnosticsConsumer
    : public DiagnosticConsumer {

  CachedDiagnosticSerializer Serializer;
  DiagnosticConsumer *OrigConsumer;
  bool EngineOwnedOrigConsumer;

  DiagnosticsConsumer(PrefixMapper &Mapper, FileManager &FileMgr,
                      DiagnosticConsumer *OrigConsumer, bool EngineOwnedOrig)
      : Serializer(Mapper, FileMgr), OrigConsumer(OrigConsumer),
        EngineOwnedOrigConsumer(EngineOwnedOrig) {
    // Using the new DiagnosticsEngine as a way to propagate converted
    // StoredDiagnostics to the original consumer. The new DiagnosticsEngine
    // contains the SourceManager that the SourceLocations of the converted
    // StoredDiagnostics came from.
    Serializer.DiagEngine.setClient(OrigConsumer, /*ShouldOwnClient*/ false);
  }

  ~DiagnosticsConsumer() {
    if (OrigConsumer && EngineOwnedOrigConsumer)
      delete OrigConsumer;
  }

  void clearConsumer() {
    OrigConsumer = nullptr;
    EngineOwnedOrigConsumer = false;
    Serializer.DiagEngine.setClient(nullptr);
  }

  void BeginSourceFile(const LangOptions &LangOpts,
                       const Preprocessor *PP) override {
    return OrigConsumer->BeginSourceFile(LangOpts, PP);
  }

  void EndSourceFile() override { return OrigConsumer->EndSourceFile(); }

  void finish() override { return OrigConsumer->finish(); }

  void HandleDiagnostic(DiagnosticsEngine::Level Level,
                        const Diagnostic &Info) override {
    if (shouldCacheDiagnostic(Level, Info)) {
      unsigned DiagIdx = Serializer.addDiag(StoredDiagnostic(Level, Info));
      StoredDiagnostic NewDiag = Serializer.getDiag(DiagIdx);
      // Pass the converted diagnostic to the original consumer. We do this
      // because:
      // 1. It ensures that the rendered diagnostics will use the same
      // diagnostics source for both a cache miss or a cache hit.
      // 2. If path prefixing is enabled, we'll pass locations with
      // de-canonicalized filenames during compilation (the original diagnostic
      // uses canonical paths).
      assert(Serializer.DiagEngine.getClient() == OrigConsumer);
      Serializer.DiagEngine.Report(NewDiag);
    } else {
      OrigConsumer->HandleDiagnostic(Level, Info);
    }
    // Update stats.
    NumWarnings = OrigConsumer->getNumWarnings();
    NumErrors = OrigConsumer->getNumErrors();
  }

  bool shouldCacheDiagnostic(DiagnosticsEngine::Level Level,
                             const Diagnostic &Info) {
    if (Level < DiagnosticsEngine::Note)
      return false;

#ifndef NDEBUG
    // These are intended for caching introspection, they should not be cached.
    // If the \p CachingDiagnosticsProcessor::DiagnosticsConsumer received one
    // of these it means that the diagnostic was emitted in-between a normal
    // compilation starting & finishing, which should not be happening.
    switch (Info.getID()) {
    case diag::remark_compile_job_cache_hit:
    case diag::remark_compile_job_cache_miss:
    case diag::remark_compile_job_cache_miss_result_not_found:
    case diag::remark_compile_job_cache_backend_output_not_found:
    case diag::remark_compile_job_cache_skipped:
    case diag::remark_compile_job_cache_timing_backend_key_query:
    case diag::remark_compile_job_cache_timing_backend_key_update:
    case diag::remark_compile_job_cache_timing_backend_load:
    case diag::remark_compile_job_cache_timing_backend_store:
      assert(0 && "unexpected caching remark diagnostic!");
    }
#endif

    return true;
  }

  void clear() override {
    Serializer.clear();
    DiagnosticConsumer::clear();
    OrigConsumer->clear();
  }

  bool IncludeInDiagnosticCounts() const override {
    return OrigConsumer->IncludeInDiagnosticCounts();
  }
};

CachingDiagnosticsProcessor::CachingDiagnosticsProcessor(PrefixMapper Mapper,
                                                         FileManager &FileMgr)
    : Mapper(std::move(Mapper)), FileMgr(FileMgr) {}

CachingDiagnosticsProcessor::~CachingDiagnosticsProcessor() = default;

void CachingDiagnosticsProcessor::insertDiagConsumer(DiagnosticsEngine &Diags) {
  assert(!DiagConsumer && "already called insertDiagConsumer?");
  DiagConsumer.reset(new DiagnosticsConsumer(Mapper, FileMgr, Diags.getClient(),
                                             Diags.ownsClient()));
  Diags.takeClient().release(); // DiagnosticsConsumer accepted ownership.
  Diags.setClient(DiagConsumer.get(), /*ShouldOwnClient*/ false);
}

void CachingDiagnosticsProcessor::removeDiagConsumer(DiagnosticsEngine &Diags) {
  assert(DiagConsumer && "didn't call insertDiagConsumer?");
  assert(DiagConsumer->OrigConsumer && "already called removeDiagConsumer?");
  assert(Diags.getClient() == DiagConsumer.get());
  Diags.setClient(DiagConsumer->OrigConsumer,
                  /*ShouldOwnClient*/ DiagConsumer->EngineOwnedOrigConsumer);
  DiagConsumer->clearConsumer();
}

Expected<std::optional<std::string>>
CachingDiagnosticsProcessor::serializeEmittedDiagnostics() {
  return DiagConsumer->Serializer.serializeEmittedDiagnostics();
}

Error CachingDiagnosticsProcessor::replayCachedDiagnostics(
    StringRef Buffer, DiagnosticConsumer &Consumer) {

  // A compilation that only includes "#import <Foundation/Foundation.h>" and
  // passes "-Weverything -Wsystem-headers" generated 3780 warnings and when
  // replaying them:
  //   * Deserialization of diagnostics: ~40ms
  //   * Rendering the diagnostics: ~240ms
  // FIXME: Look into improving performance of diagnostic rendering.

  CachedDiagnosticSerializer Serializer(Mapper, FileMgr);
  if (Error E = Serializer.deserializeCachedDiagnostics(Buffer))
    return E;
  Serializer.DiagEngine.setClient(&Consumer, /*ShouldOwnClient*/ false);
  for (unsigned I = 0, E = Serializer.getNumDiags(); I != E; I++) {
    Serializer.DiagEngine.Report(Serializer.getDiag(I));
  }
  return Error::success();
}
