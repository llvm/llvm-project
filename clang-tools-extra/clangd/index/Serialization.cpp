//===-- Serialization.cpp - Binary serialization of index data ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Serialization.h"
#include "Headers.h"
#include "RIFF.h"
#include "SymbolLocation.h"
#include "SymbolOrigin.h"
#include "dex/Dex.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <vector>

namespace clang {
namespace clangd {
namespace {

// IO PRIMITIVES
// We use little-endian 32 bit ints, sometimes with variable-length encoding.
//
// Variable-length int encoding (varint) uses the bottom 7 bits of each byte
// to encode the number, and the top bit to indicate whether more bytes follow.
// e.g. 9a 2f means [0x1a and keep reading, 0x2f and stop].
// This represents 0x1a | 0x2f<<7 = 6042.
// A 32-bit integer takes 1-5 bytes to encode; small numbers are more compact.

// Reads binary data from a StringRef, and keeps track of position.
class Reader {
  const char *Begin, *End;
  bool Err = false;

public:
  Reader(llvm::StringRef Data) : Begin(Data.begin()), End(Data.end()) {}
  // The "error" bit is set by reading past EOF or reading invalid data.
  // When in an error state, reads may return zero values: callers should check.
  bool err() const { return Err; }
  // Did we read all the data, or encounter an error?
  bool eof() const { return Begin == End || Err; }
  // All the data we didn't read yet.
  llvm::StringRef rest() const { return llvm::StringRef(Begin, End - Begin); }

  uint8_t consume8() {
    if (LLVM_UNLIKELY(Begin == End)) {
      Err = true;
      return 0;
    }
    return *Begin++;
  }

  uint32_t consume32() {
    if (LLVM_UNLIKELY(Begin + 4 > End)) {
      Err = true;
      return 0;
    }
    auto Ret = llvm::support::endian::read32le(Begin);
    Begin += 4;
    return Ret;
  }

  llvm::StringRef consume(int N) {
    if (LLVM_UNLIKELY(Begin + N > End)) {
      Err = true;
      return llvm::StringRef();
    }
    llvm::StringRef Ret(Begin, N);
    Begin += N;
    return Ret;
  }

  uint32_t consumeVar() {
    constexpr static uint8_t More = 1 << 7;

    // Use a 32 bit unsigned here to prevent promotion to signed int (unless int
    // is wider than 32 bits).
    uint32_t B = consume8();
    if (LLVM_LIKELY(!(B & More)))
      return B;
    uint32_t Val = B & ~More;
    for (int Shift = 7; B & More && Shift < 32; Shift += 7) {
      B = consume8();
      // 5th byte of a varint can only have lowest 4 bits set.
      assert((Shift != 28 || B == (B & 0x0f)) && "Invalid varint encoding");
      Val |= (B & ~More) << Shift;
    }
    return Val;
  }

  llvm::StringRef consumeString(llvm::ArrayRef<llvm::StringRef> Strings) {
    auto StringIndex = consumeVar();
    if (LLVM_UNLIKELY(StringIndex >= Strings.size())) {
      Err = true;
      return llvm::StringRef();
    }
    return Strings[StringIndex];
  }

  SymbolID consumeID() {
    llvm::StringRef Raw = consume(SymbolID::RawSize); // short if truncated.
    return LLVM_UNLIKELY(err()) ? SymbolID() : SymbolID::fromRaw(Raw);
  }

  // Read a varint (as consumeVar) and resize the container accordingly.
  // If the size is invalid, return false and mark an error.
  // (The caller should abort in this case).
  template <typename T> LLVM_NODISCARD bool consumeSize(T &Container) {
    auto Size = consumeVar();
    // Conservatively assume each element is at least one byte.
    if (Size > (End - Begin)) {
      Err = true;
      return false;
    }
    Container.resize(Size);
    return true;
  }
};

void write32(uint32_t I, llvm::raw_ostream &OS) {
  char Buf[4];
  llvm::support::endian::write32le(Buf, I);
  OS.write(Buf, sizeof(Buf));
}

void writeVar(uint32_t I, llvm::raw_ostream &OS) {
  constexpr static uint8_t More = 1 << 7;
  if (LLVM_LIKELY(I < 1 << 7)) {
    OS.write(I);
    return;
  }
  for (;;) {
    OS.write(I | More);
    I >>= 7;
    if (I < 1 << 7) {
      OS.write(I);
      return;
    }
  }
}

// STRING TABLE ENCODING
// Index data has many string fields, and many strings are identical.
// We store each string once, and refer to them by index.
//
// The string table's format is:
//   - UncompressedSize : uint32 (or 0 for no compression)
//   - CompressedData   : byte[CompressedSize]
//
// CompressedData is a zlib-compressed byte[UncompressedSize].
// It contains a sequence of null-terminated strings, e.g. "foo\0bar\0".
// These are sorted to improve compression.

// Maps each string to a canonical representation.
// Strings remain owned externally (e.g. by SymbolSlab).
class StringTableOut {
  llvm::DenseSet<llvm::StringRef> Unique;
  std::vector<llvm::StringRef> Sorted;
  // Since strings are interned, look up can be by pointer.
  llvm::DenseMap<std::pair<const char *, size_t>, unsigned> Index;

public:
  StringTableOut() {
    // Ensure there's at least one string in the table.
    // Table size zero is reserved to indicate no compression.
    Unique.insert("");
  }
  // Add a string to the table. Overwrites S if an identical string exists.
  void intern(llvm::StringRef &S) { S = *Unique.insert(S).first; };
  // Finalize the table and write it to OS. No more strings may be added.
  void finalize(llvm::raw_ostream &OS) {
    Sorted = {Unique.begin(), Unique.end()};
    llvm::sort(Sorted);
    for (unsigned I = 0; I < Sorted.size(); ++I)
      Index.try_emplace({Sorted[I].data(), Sorted[I].size()}, I);

    std::string RawTable;
    for (llvm::StringRef S : Sorted) {
      RawTable.append(std::string(S));
      RawTable.push_back(0);
    }
    if (llvm::zlib::isAvailable()) {
      llvm::SmallString<1> Compressed;
      llvm::cantFail(llvm::zlib::compress(RawTable, Compressed));
      write32(RawTable.size(), OS);
      OS << Compressed;
    } else {
      write32(0, OS); // No compression.
      OS << RawTable;
    }
  }
  // Get the ID of an string, which must be interned. Table must be finalized.
  unsigned index(llvm::StringRef S) const {
    assert(!Sorted.empty() && "table not finalized");
    assert(Index.count({S.data(), S.size()}) && "string not interned");
    return Index.find({S.data(), S.size()})->second;
  }
};

struct StringTableIn {
  llvm::BumpPtrAllocator Arena;
  std::vector<llvm::StringRef> Strings;
};

llvm::Expected<StringTableIn> readStringTable(llvm::StringRef Data) {
  Reader R(Data);
  size_t UncompressedSize = R.consume32();
  if (R.err())
    return error("Truncated string table");

  llvm::StringRef Uncompressed;
  llvm::SmallString<1> UncompressedStorage;
  if (UncompressedSize == 0) // No compression
    Uncompressed = R.rest();
  else if (llvm::zlib::isAvailable()) {
    if (llvm::Error E = llvm::zlib::uncompress(R.rest(), UncompressedStorage,
                                               UncompressedSize))
      return std::move(E);
    Uncompressed = UncompressedStorage;
  } else
    return error("Compressed string table, but zlib is unavailable");

  StringTableIn Table;
  llvm::StringSaver Saver(Table.Arena);
  R = Reader(Uncompressed);
  for (Reader R(Uncompressed); !R.eof();) {
    auto Len = R.rest().find(0);
    if (Len == llvm::StringRef::npos)
      return error("Bad string table: not null terminated");
    Table.Strings.push_back(Saver.save(R.consume(Len)));
    R.consume8();
  }
  if (R.err())
    return error("Truncated string table");
  return std::move(Table);
}

// SYMBOL ENCODING
// Each field of clangd::Symbol is encoded in turn (see implementation).
//  - StringRef fields encode as varint (index into the string table)
//  - enums encode as the underlying type
//  - most numbers encode as varint

void writeLocation(const SymbolLocation &Loc, const StringTableOut &Strings,
                   llvm::raw_ostream &OS) {
  writeVar(Strings.index(Loc.FileURI), OS);
  for (const auto &Endpoint : {Loc.Start, Loc.End}) {
    writeVar(Endpoint.line(), OS);
    writeVar(Endpoint.column(), OS);
  }
}

SymbolLocation readLocation(Reader &Data,
                            llvm::ArrayRef<llvm::StringRef> Strings) {
  SymbolLocation Loc;
  Loc.FileURI = Data.consumeString(Strings).data();
  for (auto *Endpoint : {&Loc.Start, &Loc.End}) {
    Endpoint->setLine(Data.consumeVar());
    Endpoint->setColumn(Data.consumeVar());
  }
  return Loc;
}

IncludeGraphNode readIncludeGraphNode(Reader &Data,
                                      llvm::ArrayRef<llvm::StringRef> Strings) {
  IncludeGraphNode IGN;
  IGN.Flags = static_cast<IncludeGraphNode::SourceFlag>(Data.consume8());
  IGN.URI = Data.consumeString(Strings);
  llvm::StringRef Digest = Data.consume(IGN.Digest.size());
  std::copy(Digest.bytes_begin(), Digest.bytes_end(), IGN.Digest.begin());
  if (!Data.consumeSize(IGN.DirectIncludes))
    return IGN;
  for (llvm::StringRef &Include : IGN.DirectIncludes)
    Include = Data.consumeString(Strings);
  return IGN;
}

void writeIncludeGraphNode(const IncludeGraphNode &IGN,
                           const StringTableOut &Strings,
                           llvm::raw_ostream &OS) {
  OS.write(static_cast<uint8_t>(IGN.Flags));
  writeVar(Strings.index(IGN.URI), OS);
  llvm::StringRef Hash(reinterpret_cast<const char *>(IGN.Digest.data()),
                       IGN.Digest.size());
  OS << Hash;
  writeVar(IGN.DirectIncludes.size(), OS);
  for (llvm::StringRef Include : IGN.DirectIncludes)
    writeVar(Strings.index(Include), OS);
}

void writeSymbol(const Symbol &Sym, const StringTableOut &Strings,
                 llvm::raw_ostream &OS) {
  OS << Sym.ID.raw(); // TODO: once we start writing xrefs and posting lists,
                      // symbol IDs should probably be in a string table.
  OS.write(static_cast<uint8_t>(Sym.SymInfo.Kind));
  OS.write(static_cast<uint8_t>(Sym.SymInfo.Lang));
  writeVar(Strings.index(Sym.Name), OS);
  writeVar(Strings.index(Sym.Scope), OS);
  writeVar(Strings.index(Sym.TemplateSpecializationArgs), OS);
  writeLocation(Sym.Definition, Strings, OS);
  writeLocation(Sym.CanonicalDeclaration, Strings, OS);
  writeVar(Sym.References, OS);
  OS.write(static_cast<uint8_t>(Sym.Flags));
  OS.write(static_cast<uint8_t>(Sym.Origin));
  writeVar(Strings.index(Sym.Signature), OS);
  writeVar(Strings.index(Sym.CompletionSnippetSuffix), OS);
  writeVar(Strings.index(Sym.Documentation), OS);
  writeVar(Strings.index(Sym.ReturnType), OS);
  writeVar(Strings.index(Sym.Type), OS);

  auto WriteInclude = [&](const Symbol::IncludeHeaderWithReferences &Include) {
    writeVar(Strings.index(Include.IncludeHeader), OS);
    writeVar(Include.References, OS);
  };
  writeVar(Sym.IncludeHeaders.size(), OS);
  for (const auto &Include : Sym.IncludeHeaders)
    WriteInclude(Include);
}

Symbol readSymbol(Reader &Data, llvm::ArrayRef<llvm::StringRef> Strings) {
  Symbol Sym;
  Sym.ID = Data.consumeID();
  Sym.SymInfo.Kind = static_cast<index::SymbolKind>(Data.consume8());
  Sym.SymInfo.Lang = static_cast<index::SymbolLanguage>(Data.consume8());
  Sym.Name = Data.consumeString(Strings);
  Sym.Scope = Data.consumeString(Strings);
  Sym.TemplateSpecializationArgs = Data.consumeString(Strings);
  Sym.Definition = readLocation(Data, Strings);
  Sym.CanonicalDeclaration = readLocation(Data, Strings);
  Sym.References = Data.consumeVar();
  Sym.Flags = static_cast<Symbol::SymbolFlag>(Data.consume8());
  Sym.Origin = static_cast<SymbolOrigin>(Data.consume8());
  Sym.Signature = Data.consumeString(Strings);
  Sym.CompletionSnippetSuffix = Data.consumeString(Strings);
  Sym.Documentation = Data.consumeString(Strings);
  Sym.ReturnType = Data.consumeString(Strings);
  Sym.Type = Data.consumeString(Strings);
  if (!Data.consumeSize(Sym.IncludeHeaders))
    return Sym;
  for (auto &I : Sym.IncludeHeaders) {
    I.IncludeHeader = Data.consumeString(Strings);
    I.References = Data.consumeVar();
  }
  return Sym;
}

// REFS ENCODING
// A refs section has data grouped by Symbol. Each symbol has:
//  - SymbolID: 8 bytes
//  - NumRefs: varint
//  - Ref[NumRefs]
// Fields of Ref are encoded in turn, see implementation.

void writeRefs(const SymbolID &ID, llvm::ArrayRef<Ref> Refs,
               const StringTableOut &Strings, llvm::raw_ostream &OS) {
  OS << ID.raw();
  writeVar(Refs.size(), OS);
  for (const auto &Ref : Refs) {
    OS.write(static_cast<unsigned char>(Ref.Kind));
    writeLocation(Ref.Location, Strings, OS);
    OS << Ref.Container.raw();
  }
}

std::pair<SymbolID, std::vector<Ref>>
readRefs(Reader &Data, llvm::ArrayRef<llvm::StringRef> Strings) {
  std::pair<SymbolID, std::vector<Ref>> Result;
  Result.first = Data.consumeID();
  if (!Data.consumeSize(Result.second))
    return Result;
  for (auto &Ref : Result.second) {
    Ref.Kind = static_cast<RefKind>(Data.consume8());
    Ref.Location = readLocation(Data, Strings);
    Ref.Container = Data.consumeID();
  }
  return Result;
}

// RELATIONS ENCODING
// A relations section is a flat list of relations. Each relation has:
//  - SymbolID (subject): 8 bytes
//  - relation kind (predicate): 1 byte
//  - SymbolID (object): 8 bytes
// In the future, we might prefer a packed representation if the need arises.

void writeRelation(const Relation &R, llvm::raw_ostream &OS) {
  OS << R.Subject.raw();
  OS.write(static_cast<uint8_t>(R.Predicate));
  OS << R.Object.raw();
}

Relation readRelation(Reader &Data) {
  SymbolID Subject = Data.consumeID();
  RelationKind Predicate = static_cast<RelationKind>(Data.consume8());
  SymbolID Object = Data.consumeID();
  return {Subject, Predicate, Object};
}

struct InternedCompileCommand {
  llvm::StringRef Directory;
  std::vector<llvm::StringRef> CommandLine;
};

void writeCompileCommand(const InternedCompileCommand &Cmd,
                         const StringTableOut &Strings,
                         llvm::raw_ostream &CmdOS) {
  writeVar(Strings.index(Cmd.Directory), CmdOS);
  writeVar(Cmd.CommandLine.size(), CmdOS);
  for (llvm::StringRef C : Cmd.CommandLine)
    writeVar(Strings.index(C), CmdOS);
}

InternedCompileCommand
readCompileCommand(Reader CmdReader, llvm::ArrayRef<llvm::StringRef> Strings) {
  InternedCompileCommand Cmd;
  Cmd.Directory = CmdReader.consumeString(Strings);
  if (!CmdReader.consumeSize(Cmd.CommandLine))
    return Cmd;
  for (llvm::StringRef &C : Cmd.CommandLine)
    C = CmdReader.consumeString(Strings);
  return Cmd;
}

// FILE ENCODING
// A file is a RIFF chunk with type 'CdIx'.
// It contains the sections:
//   - meta: version number
//   - srcs: information related to include graph
//   - stri: string table
//   - symb: symbols
//   - refs: references to symbols

// The current versioning scheme is simple - non-current versions are rejected.
// If you make a breaking change, bump this version number to invalidate stored
// data. Later we may want to support some backward compatibility.
constexpr static uint32_t Version = 14;

llvm::Expected<IndexFileIn> readRIFF(llvm::StringRef Data) {
  auto RIFF = riff::readFile(Data);
  if (!RIFF)
    return RIFF.takeError();
  if (RIFF->Type != riff::fourCC("CdIx"))
    return error("wrong RIFF filetype: {0}", riff::fourCCStr(RIFF->Type));
  llvm::StringMap<llvm::StringRef> Chunks;
  for (const auto &Chunk : RIFF->Chunks)
    Chunks.try_emplace(llvm::StringRef(Chunk.ID.data(), Chunk.ID.size()),
                       Chunk.Data);

  if (!Chunks.count("meta"))
    return error("missing meta chunk");
  Reader Meta(Chunks.lookup("meta"));
  auto SeenVersion = Meta.consume32();
  if (SeenVersion != Version)
    return error("wrong version: want {0}, got {1}", Version, SeenVersion);

  // meta chunk is checked above, as we prefer the "version mismatch" error.
  for (llvm::StringRef RequiredChunk : {"stri"})
    if (!Chunks.count(RequiredChunk))
      return error("missing required chunk {0}", RequiredChunk);

  auto Strings = readStringTable(Chunks.lookup("stri"));
  if (!Strings)
    return Strings.takeError();

  IndexFileIn Result;
  if (Chunks.count("srcs")) {
    Reader SrcsReader(Chunks.lookup("srcs"));
    Result.Sources.emplace();
    while (!SrcsReader.eof()) {
      auto IGN = readIncludeGraphNode(SrcsReader, Strings->Strings);
      auto Entry = Result.Sources->try_emplace(IGN.URI).first;
      Entry->getValue() = std::move(IGN);
      // We change all the strings inside the structure to point at the keys in
      // the map, since it is the only copy of the string that's going to live.
      Entry->getValue().URI = Entry->getKey();
      for (auto &Include : Entry->getValue().DirectIncludes)
        Include = Result.Sources->try_emplace(Include).first->getKey();
    }
    if (SrcsReader.err())
      return error("malformed or truncated include uri");
  }

  if (Chunks.count("symb")) {
    Reader SymbolReader(Chunks.lookup("symb"));
    SymbolSlab::Builder Symbols;
    while (!SymbolReader.eof())
      Symbols.insert(readSymbol(SymbolReader, Strings->Strings));
    if (SymbolReader.err())
      return error("malformed or truncated symbol");
    Result.Symbols = std::move(Symbols).build();
  }
  if (Chunks.count("refs")) {
    Reader RefsReader(Chunks.lookup("refs"));
    RefSlab::Builder Refs;
    while (!RefsReader.eof()) {
      auto RefsBundle = readRefs(RefsReader, Strings->Strings);
      for (const auto &Ref : RefsBundle.second) // FIXME: bulk insert?
        Refs.insert(RefsBundle.first, Ref);
    }
    if (RefsReader.err())
      return error("malformed or truncated refs");
    Result.Refs = std::move(Refs).build();
  }
  if (Chunks.count("rela")) {
    Reader RelationsReader(Chunks.lookup("rela"));
    RelationSlab::Builder Relations;
    while (!RelationsReader.eof())
      Relations.insert(readRelation(RelationsReader));
    if (RelationsReader.err())
      return error("malformed or truncated relations");
    Result.Relations = std::move(Relations).build();
  }
  if (Chunks.count("cmdl")) {
    Reader CmdReader(Chunks.lookup("cmdl"));
    InternedCompileCommand Cmd =
        readCompileCommand(CmdReader, Strings->Strings);
    if (CmdReader.err())
      return error("malformed or truncated commandline section");
    Result.Cmd.emplace();
    Result.Cmd->Directory = std::string(Cmd.Directory);
    Result.Cmd->CommandLine.reserve(Cmd.CommandLine.size());
    for (llvm::StringRef C : Cmd.CommandLine)
      Result.Cmd->CommandLine.emplace_back(C);
  }
  return std::move(Result);
}

template <class Callback>
void visitStrings(IncludeGraphNode &IGN, const Callback &CB) {
  CB(IGN.URI);
  for (llvm::StringRef &Include : IGN.DirectIncludes)
    CB(Include);
}

void writeRIFF(const IndexFileOut &Data, llvm::raw_ostream &OS) {
  assert(Data.Symbols && "An index file without symbols makes no sense!");
  riff::File RIFF;
  RIFF.Type = riff::fourCC("CdIx");

  llvm::SmallString<4> Meta;
  {
    llvm::raw_svector_ostream MetaOS(Meta);
    write32(Version, MetaOS);
  }
  RIFF.Chunks.push_back({riff::fourCC("meta"), Meta});

  StringTableOut Strings;
  std::vector<Symbol> Symbols;
  for (const auto &Sym : *Data.Symbols) {
    Symbols.emplace_back(Sym);
    visitStrings(Symbols.back(),
                 [&](llvm::StringRef &S) { Strings.intern(S); });
  }
  std::vector<IncludeGraphNode> Sources;
  if (Data.Sources)
    for (const auto &Source : *Data.Sources) {
      Sources.push_back(Source.getValue());
      visitStrings(Sources.back(),
                   [&](llvm::StringRef &S) { Strings.intern(S); });
    }

  std::vector<std::pair<SymbolID, std::vector<Ref>>> Refs;
  if (Data.Refs) {
    for (const auto &Sym : *Data.Refs) {
      Refs.emplace_back(Sym);
      for (auto &Ref : Refs.back().second) {
        llvm::StringRef File = Ref.Location.FileURI;
        Strings.intern(File);
        Ref.Location.FileURI = File.data();
      }
    }
  }

  std::vector<Relation> Relations;
  if (Data.Relations) {
    for (const auto &Relation : *Data.Relations) {
      Relations.emplace_back(Relation);
      // No strings to be interned in relations.
    }
  }

  InternedCompileCommand InternedCmd;
  if (Data.Cmd) {
    InternedCmd.CommandLine.reserve(Data.Cmd->CommandLine.size());
    InternedCmd.Directory = Data.Cmd->Directory;
    Strings.intern(InternedCmd.Directory);
    for (llvm::StringRef C : Data.Cmd->CommandLine) {
      InternedCmd.CommandLine.emplace_back(C);
      Strings.intern(InternedCmd.CommandLine.back());
    }
  }

  std::string StringSection;
  {
    llvm::raw_string_ostream StringOS(StringSection);
    Strings.finalize(StringOS);
  }
  RIFF.Chunks.push_back({riff::fourCC("stri"), StringSection});

  std::string SymbolSection;
  {
    llvm::raw_string_ostream SymbolOS(SymbolSection);
    for (const auto &Sym : Symbols)
      writeSymbol(Sym, Strings, SymbolOS);
  }
  RIFF.Chunks.push_back({riff::fourCC("symb"), SymbolSection});

  std::string RefsSection;
  if (Data.Refs) {
    {
      llvm::raw_string_ostream RefsOS(RefsSection);
      for (const auto &Sym : Refs)
        writeRefs(Sym.first, Sym.second, Strings, RefsOS);
    }
    RIFF.Chunks.push_back({riff::fourCC("refs"), RefsSection});
  }

  std::string RelationSection;
  if (Data.Relations) {
    {
      llvm::raw_string_ostream RelationOS{RelationSection};
      for (const auto &Relation : Relations)
        writeRelation(Relation, RelationOS);
    }
    RIFF.Chunks.push_back({riff::fourCC("rela"), RelationSection});
  }

  std::string SrcsSection;
  {
    {
      llvm::raw_string_ostream SrcsOS(SrcsSection);
      for (const auto &SF : Sources)
        writeIncludeGraphNode(SF, Strings, SrcsOS);
    }
    RIFF.Chunks.push_back({riff::fourCC("srcs"), SrcsSection});
  }

  std::string CmdlSection;
  if (Data.Cmd) {
    {
      llvm::raw_string_ostream CmdOS(CmdlSection);
      writeCompileCommand(InternedCmd, Strings, CmdOS);
    }
    RIFF.Chunks.push_back({riff::fourCC("cmdl"), CmdlSection});
  }

  OS << RIFF;
}

} // namespace

// Defined in YAMLSerialization.cpp.
void writeYAML(const IndexFileOut &, llvm::raw_ostream &);
llvm::Expected<IndexFileIn> readYAML(llvm::StringRef);

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const IndexFileOut &O) {
  switch (O.Format) {
  case IndexFileFormat::RIFF:
    writeRIFF(O, OS);
    break;
  case IndexFileFormat::YAML:
    writeYAML(O, OS);
    break;
  }
  return OS;
}

llvm::Expected<IndexFileIn> readIndexFile(llvm::StringRef Data) {
  if (Data.startswith("RIFF")) {
    return readRIFF(Data);
  } else if (auto YAMLContents = readYAML(Data)) {
    return std::move(*YAMLContents);
  } else {
    return error("Not a RIFF file and failed to parse as YAML: {0}",
                 YAMLContents.takeError());
  }
}

std::unique_ptr<SymbolIndex> loadIndex(llvm::StringRef SymbolFilename,
                                       bool UseDex) {
  trace::Span OverallTracer("LoadIndex");
  auto Buffer = llvm::MemoryBuffer::getFile(SymbolFilename);
  if (!Buffer) {
    elog("Can't open {0}: {1}", SymbolFilename, Buffer.getError().message());
    return nullptr;
  }

  SymbolSlab Symbols;
  RefSlab Refs;
  RelationSlab Relations;
  {
    trace::Span Tracer("ParseIndex");
    if (auto I = readIndexFile(Buffer->get()->getBuffer())) {
      if (I->Symbols)
        Symbols = std::move(*I->Symbols);
      if (I->Refs)
        Refs = std::move(*I->Refs);
      if (I->Relations)
        Relations = std::move(*I->Relations);
    } else {
      elog("Bad index file: {0}", I.takeError());
      return nullptr;
    }
  }

  size_t NumSym = Symbols.size();
  size_t NumRefs = Refs.numRefs();
  size_t NumRelations = Relations.size();

  trace::Span Tracer("BuildIndex");
  auto Index = UseDex ? dex::Dex::build(std::move(Symbols), std::move(Refs),
                                        std::move(Relations))
                      : MemIndex::build(std::move(Symbols), std::move(Refs),
                                        std::move(Relations));
  vlog("Loaded {0} from {1} with estimated memory usage {2} bytes\n"
       "  - number of symbols: {3}\n"
       "  - number of refs: {4}\n"
       "  - number of relations: {5}",
       UseDex ? "Dex" : "MemIndex", SymbolFilename,
       Index->estimateMemoryUsage(), NumSym, NumRefs, NumRelations);
  return Index;
}

} // namespace clangd
} // namespace clang
