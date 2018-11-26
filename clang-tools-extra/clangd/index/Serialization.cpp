//===-- Serialization.cpp - Binary serialization of index data ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Serialization.h"
#include "Index.h"
#include "Logger.h"
#include "RIFF.h"
#include "Trace.h"
#include "dex/Dex.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

using namespace llvm;
namespace clang {
namespace clangd {
namespace {
Error makeError(const Twine &Msg) {
  return make_error<StringError>(Msg, inconvertibleErrorCode());
}

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
  Reader(StringRef Data) : Begin(Data.begin()), End(Data.end()) {}
  // The "error" bit is set by reading past EOF or reading invalid data.
  // When in an error state, reads may return zero values: callers should check.
  bool err() const { return Err; }
  // Did we read all the data, or encounter an error?
  bool eof() const { return Begin == End || Err; }
  // All the data we didn't read yet.
  StringRef rest() const { return StringRef(Begin, End - Begin); }

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
    auto Ret = support::endian::read32le(Begin);
    Begin += 4;
    return Ret;
  }

  StringRef consume(int N) {
    if (LLVM_UNLIKELY(Begin + N > End)) {
      Err = true;
      return StringRef();
    }
    StringRef Ret(Begin, N);
    Begin += N;
    return Ret;
  }

  uint32_t consumeVar() {
    constexpr static uint8_t More = 1 << 7;
    uint8_t B = consume8();
    if (LLVM_LIKELY(!(B & More)))
      return B;
    uint32_t Val = B & ~More;
    for (int Shift = 7; B & More && Shift < 32; Shift += 7) {
      B = consume8();
      Val |= (B & ~More) << Shift;
    }
    return Val;
  }

  StringRef consumeString(ArrayRef<StringRef> Strings) {
    auto StringIndex = consumeVar();
    if (LLVM_UNLIKELY(StringIndex >= Strings.size())) {
      Err = true;
      return StringRef();
    }
    return Strings[StringIndex];
  }

  SymbolID consumeID() {
    StringRef Raw = consume(SymbolID::RawSize); // short if truncated.
    return LLVM_UNLIKELY(err()) ? SymbolID() : SymbolID::fromRaw(Raw);
  }
};

void write32(uint32_t I, raw_ostream &OS) {
  char buf[4];
  support::endian::write32le(buf, I);
  OS.write(buf, sizeof(buf));
}

void writeVar(uint32_t I, raw_ostream &OS) {
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
  DenseSet<StringRef> Unique;
  std::vector<StringRef> Sorted;
  // Since strings are interned, look up can be by pointer.
  DenseMap<std::pair<const char *, size_t>, unsigned> Index;

public:
  StringTableOut() {
    // Ensure there's at least one string in the table.
    // Table size zero is reserved to indicate no compression.
    Unique.insert("");
  }
  // Add a string to the table. Overwrites S if an identical string exists.
  void intern(StringRef &S) { S = *Unique.insert(S).first; };
  // Finalize the table and write it to OS. No more strings may be added.
  void finalize(raw_ostream &OS) {
    Sorted = {Unique.begin(), Unique.end()};
    llvm::sort(Sorted);
    for (unsigned I = 0; I < Sorted.size(); ++I)
      Index.try_emplace({Sorted[I].data(), Sorted[I].size()}, I);

    std::string RawTable;
    for (StringRef S : Sorted) {
      RawTable.append(S);
      RawTable.push_back(0);
    }
    if (zlib::isAvailable()) {
      SmallString<1> Compressed;
      cantFail(zlib::compress(RawTable, Compressed));
      write32(RawTable.size(), OS);
      OS << Compressed;
    } else {
      write32(0, OS); // No compression.
      OS << RawTable;
    }
  }
  // Get the ID of an string, which must be interned. Table must be finalized.
  unsigned index(StringRef S) const {
    assert(!Sorted.empty() && "table not finalized");
    assert(Index.count({S.data(), S.size()}) && "string not interned");
    return Index.find({S.data(), S.size()})->second;
  }
};

struct StringTableIn {
  BumpPtrAllocator Arena;
  std::vector<StringRef> Strings;
};

Expected<StringTableIn> readStringTable(StringRef Data) {
  Reader R(Data);
  size_t UncompressedSize = R.consume32();
  if (R.err())
    return makeError("Truncated string table");

  StringRef Uncompressed;
  SmallString<1> UncompressedStorage;
  if (UncompressedSize == 0) // No compression
    Uncompressed = R.rest();
  else {
    if (Error E = llvm::zlib::uncompress(R.rest(), UncompressedStorage,
                                         UncompressedSize))
      return std::move(E);
    Uncompressed = UncompressedStorage;
  }

  StringTableIn Table;
  StringSaver Saver(Table.Arena);
  R = Reader(Uncompressed);
  for (Reader R(Uncompressed); !R.eof();) {
    auto Len = R.rest().find(0);
    if (Len == StringRef::npos)
      return makeError("Bad string table: not null terminated");
    Table.Strings.push_back(Saver.save(R.consume(Len)));
    R.consume8();
  }
  if (R.err())
    return makeError("Truncated string table");
  return std::move(Table);
}

// SYMBOL ENCODING
// Each field of clangd::Symbol is encoded in turn (see implementation).
//  - StringRef fields encode as varint (index into the string table)
//  - enums encode as the underlying type
//  - most numbers encode as varint

void writeLocation(const SymbolLocation &Loc, const StringTableOut &Strings,
                   raw_ostream &OS) {
  writeVar(Strings.index(Loc.FileURI), OS);
  for (const auto &Endpoint : {Loc.Start, Loc.End}) {
    writeVar(Endpoint.line(), OS);
    writeVar(Endpoint.column(), OS);
  }
}

SymbolLocation readLocation(Reader &Data, ArrayRef<StringRef> Strings) {
  SymbolLocation Loc;
  Loc.FileURI = Data.consumeString(Strings).data();
  for (auto *Endpoint : {&Loc.Start, &Loc.End}) {
    Endpoint->setLine(Data.consumeVar());
    Endpoint->setColumn(Data.consumeVar());
  }
  return Loc;
}

void writeSymbol(const Symbol &Sym, const StringTableOut &Strings,
                 raw_ostream &OS) {
  OS << Sym.ID.raw(); // TODO: once we start writing xrefs and posting lists,
                      // symbol IDs should probably be in a string table.
  OS.write(static_cast<uint8_t>(Sym.SymInfo.Kind));
  OS.write(static_cast<uint8_t>(Sym.SymInfo.Lang));
  writeVar(Strings.index(Sym.Name), OS);
  writeVar(Strings.index(Sym.Scope), OS);
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

Symbol readSymbol(Reader &Data, ArrayRef<StringRef> Strings) {
  Symbol Sym;
  Sym.ID = Data.consumeID();
  Sym.SymInfo.Kind = static_cast<index::SymbolKind>(Data.consume8());
  Sym.SymInfo.Lang = static_cast<index::SymbolLanguage>(Data.consume8());
  Sym.Name = Data.consumeString(Strings);
  Sym.Scope = Data.consumeString(Strings);
  Sym.Definition = readLocation(Data, Strings);
  Sym.CanonicalDeclaration = readLocation(Data, Strings);
  Sym.References = Data.consumeVar();
  Sym.Flags = static_cast<Symbol::SymbolFlag>(Data.consumeVar());
  Sym.Origin = static_cast<SymbolOrigin>(Data.consumeVar());
  Sym.Signature = Data.consumeString(Strings);
  Sym.CompletionSnippetSuffix = Data.consumeString(Strings);
  Sym.Documentation = Data.consumeString(Strings);
  Sym.ReturnType = Data.consumeString(Strings);
  Sym.Type = Data.consumeString(Strings);
  Sym.IncludeHeaders.resize(Data.consumeVar());
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

void writeRefs(const SymbolID &ID, ArrayRef<Ref> Refs,
               const StringTableOut &Strings, raw_ostream &OS) {
  OS << ID.raw();
  writeVar(Refs.size(), OS);
  for (const auto &Ref : Refs) {
    OS.write(static_cast<unsigned char>(Ref.Kind));
    writeLocation(Ref.Location, Strings, OS);
  }
}

std::pair<SymbolID, std::vector<Ref>> readRefs(Reader &Data,
                                               ArrayRef<StringRef> Strings) {
  std::pair<SymbolID, std::vector<Ref>> Result;
  Result.first = Data.consumeID();
  Result.second.resize(Data.consumeVar());
  for (auto &Ref : Result.second) {
    Ref.Kind = static_cast<RefKind>(Data.consume8());
    Ref.Location = readLocation(Data, Strings);
  }
  return Result;
}

// FILE ENCODING
// A file is a RIFF chunk with type 'CdIx'.
// It contains the sections:
//   - meta: version number
//   - srcs: checksum of the source file
//   - stri: string table
//   - symb: symbols
//   - refs: references to symbols

// The current versioning scheme is simple - non-current versions are rejected.
// If you make a breaking change, bump this version number to invalidate stored
// data. Later we may want to support some backward compatibility.
constexpr static uint32_t Version = 8;

Expected<IndexFileIn> readRIFF(StringRef Data) {
  auto RIFF = riff::readFile(Data);
  if (!RIFF)
    return RIFF.takeError();
  if (RIFF->Type != riff::fourCC("CdIx"))
    return makeError("wrong RIFF type");
  StringMap<StringRef> Chunks;
  for (const auto &Chunk : RIFF->Chunks)
    Chunks.try_emplace(StringRef(Chunk.ID.data(), Chunk.ID.size()), Chunk.Data);

  for (StringRef RequiredChunk : {"meta", "stri"})
    if (!Chunks.count(RequiredChunk))
      return makeError("missing required chunk " + RequiredChunk);

  Reader Meta(Chunks.lookup("meta"));
  if (Meta.consume32() != Version)
    return makeError("wrong version");

  auto Strings = readStringTable(Chunks.lookup("stri"));
  if (!Strings)
    return Strings.takeError();

  IndexFileIn Result;
  if (Chunks.count("srcs")) {
    Reader Hash(Chunks.lookup("srcs"));
    Result.Digest.emplace();
    llvm::StringRef Digest = Hash.consume(Result.Digest->size());
    std::copy(Digest.bytes_begin(), Digest.bytes_end(), Result.Digest->begin());
  }

  if (Chunks.count("symb")) {
    Reader SymbolReader(Chunks.lookup("symb"));
    SymbolSlab::Builder Symbols;
    while (!SymbolReader.eof())
      Symbols.insert(readSymbol(SymbolReader, Strings->Strings));
    if (SymbolReader.err())
      return makeError("malformed or truncated symbol");
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
      return makeError("malformed or truncated refs");
    Result.Refs = std::move(Refs).build();
  }
  return std::move(Result);
}

void writeRIFF(const IndexFileOut &Data, raw_ostream &OS) {
  assert(Data.Symbols && "An index file without symbols makes no sense!");
  riff::File RIFF;
  RIFF.Type = riff::fourCC("CdIx");

  SmallString<4> Meta;
  {
    raw_svector_ostream MetaOS(Meta);
    write32(Version, MetaOS);
  }
  RIFF.Chunks.push_back({riff::fourCC("meta"), Meta});

  if (Data.Digest) {
    llvm::StringRef Hash(reinterpret_cast<const char *>(Data.Digest->data()),
                         Data.Digest->size());
    RIFF.Chunks.push_back({riff::fourCC("srcs"), Hash});
  }

  StringTableOut Strings;
  std::vector<Symbol> Symbols;
  for (const auto &Sym : *Data.Symbols) {
    Symbols.emplace_back(Sym);
    visitStrings(Symbols.back(), [&](StringRef &S) { Strings.intern(S); });
  }
  std::vector<std::pair<SymbolID, std::vector<Ref>>> Refs;
  if (Data.Refs) {
    for (const auto &Sym : *Data.Refs) {
      Refs.emplace_back(Sym);
      for (auto &Ref : Refs.back().second) {
        StringRef File = Ref.Location.FileURI;
        Strings.intern(File);
        Ref.Location.FileURI = File.data();
      }
    }
  }

  std::string StringSection;
  {
    raw_string_ostream StringOS(StringSection);
    Strings.finalize(StringOS);
  }
  RIFF.Chunks.push_back({riff::fourCC("stri"), StringSection});

  std::string SymbolSection;
  {
    raw_string_ostream SymbolOS(SymbolSection);
    for (const auto &Sym : Symbols)
      writeSymbol(Sym, Strings, SymbolOS);
  }
  RIFF.Chunks.push_back({riff::fourCC("symb"), SymbolSection});

  std::string RefsSection;
  if (Data.Refs) {
    {
      raw_string_ostream RefsOS(RefsSection);
      for (const auto &Sym : Refs)
        writeRefs(Sym.first, Sym.second, Strings, RefsOS);
    }
    RIFF.Chunks.push_back({riff::fourCC("refs"), RefsSection});
  }

  OS << RIFF;
}

} // namespace

// Defined in YAMLSerialization.cpp.
void writeYAML(const IndexFileOut &, raw_ostream &);
Expected<IndexFileIn> readYAML(StringRef);

raw_ostream &operator<<(raw_ostream &OS, const IndexFileOut &O) {
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

Expected<IndexFileIn> readIndexFile(StringRef Data) {
  if (Data.startswith("RIFF")) {
    return readRIFF(Data);
  } else if (auto YAMLContents = readYAML(Data)) {
    return std::move(*YAMLContents);
  } else {
    return makeError("Not a RIFF file and failed to parse as YAML: " +
                     toString(YAMLContents.takeError()));
  }
}

std::unique_ptr<SymbolIndex> loadIndex(StringRef SymbolFilename, bool UseDex) {
  trace::Span OverallTracer("LoadIndex");
  auto Buffer = MemoryBuffer::getFile(SymbolFilename);
  if (!Buffer) {
    errs() << "Can't open " << SymbolFilename << "\n";
    return nullptr;
  }

  SymbolSlab Symbols;
  RefSlab Refs;
  {
    trace::Span Tracer("ParseIndex");
    if (auto I = readIndexFile(Buffer->get()->getBuffer())) {
      if (I->Symbols)
        Symbols = std::move(*I->Symbols);
      if (I->Refs)
        Refs = std::move(*I->Refs);
    } else {
      errs() << "Bad Index: " << toString(I.takeError()) << "\n";
      return nullptr;
    }
  }

  size_t NumSym = Symbols.size();
  size_t NumRefs = Refs.numRefs();

  trace::Span Tracer("BuildIndex");
  auto Index = UseDex ? dex::Dex::build(std::move(Symbols), std::move(Refs))
                      : MemIndex::build(std::move(Symbols), std::move(Refs));
  vlog("Loaded {0} from {1} with estimated memory usage {2} bytes\n"
       "  - number of symbols: {3}\n"
       "  - number of refs: {4}\n",
       UseDex ? "Dex" : "MemIndex", SymbolFilename,
       Index->estimateMemoryUsage(), NumSym, NumRefs);
  return Index;
}

} // namespace clangd
} // namespace clang
