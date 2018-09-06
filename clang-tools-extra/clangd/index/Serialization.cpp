//===-- Serialization.cpp - Binary serialization of index data ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "Serialization.h"
#include "../RIFF.h"
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

StringRef consume(StringRef &Data, int N) {
  StringRef Ret = Data.take_front(N);
  Data = Data.drop_front(N);
  return Ret;
}

uint8_t consume8(StringRef &Data) {
  uint8_t Ret = Data.front();
  Data = Data.drop_front();
  return Ret;
}

uint32_t consume32(StringRef &Data) {
  auto Ret = support::endian::read32le(Data.bytes_begin());
  Data = Data.drop_front(4);
  return Ret;
}

void write32(uint32_t I, raw_ostream &OS) {
  char buf[4];
  support::endian::write32le(buf, I);
  OS.write(buf, sizeof(buf));
}

// Variable-length int encoding (varint) uses the bottom 7 bits of each byte
// to encode the number, and the top bit to indicate whether more bytes follow.
// e.g. 9a 2f means [0x1a and keep reading, 0x2f and stop].
// This represents 0x1a | 0x2f<<7 = 6042.
// A 32-bit integer takes 1-5 bytes to encode; small numbers are more compact.
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

uint32_t consumeVar(StringRef &Data) {
  constexpr static uint8_t More = 1 << 7;
  uint8_t B = consume8(Data);
  if (LLVM_LIKELY(!(B & More)))
    return B;
  uint32_t Val = B & ~More;
  for (int Shift = 7; B & More && Shift < 32; Shift += 7) {
    B = consume8(Data);
    Val |= (B & ~More) << Shift;
  }
  return Val;
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
    std::sort(Sorted.begin(), Sorted.end());
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
  if (Data.size() < 4)
    return makeError("Bad string table: not enough metadata");
  size_t UncompressedSize = consume32(Data);

  StringRef Uncompressed;
  SmallString<1> UncompressedStorage;
  if (UncompressedSize == 0) // No compression
    Uncompressed = Data;
  else {
    if (Error E =
            llvm::zlib::uncompress(Data, UncompressedStorage, UncompressedSize))
      return std::move(E);
    Uncompressed = UncompressedStorage;
  }

  StringTableIn Table;
  StringSaver Saver(Table.Arena);
  for (StringRef Rest = Uncompressed; !Rest.empty();) {
    auto Len = Rest.find(0);
    if (Len == StringRef::npos)
      return makeError("Bad string table: not null terminated");
    Table.Strings.push_back(Saver.save(consume(Rest, Len)));
    Rest = Rest.drop_front();
  }
  return std::move(Table);
}

// SYMBOL ENCODING
// Each field of clangd::Symbol is encoded in turn (see implementation).
//  - StringRef fields encode as varint (index into the string table)
//  - enums encode as the underlying type
//  - most numbers encode as varint

// It's useful to the implementation to assume symbols have a bounded size.
constexpr size_t SymbolSizeBound = 512;
// To ensure the bounded size, restrict the number of include headers stored.
constexpr unsigned MaxIncludes = 50;

void writeSymbol(const Symbol &Sym, const StringTableOut &Strings,
                 raw_ostream &OS) {
  auto StartOffset = OS.tell();
  OS << Sym.ID.raw(); // TODO: once we start writing xrefs and posting lists,
                      // symbol IDs should probably be in a string table.
  OS.write(static_cast<uint8_t>(Sym.SymInfo.Kind));
  OS.write(static_cast<uint8_t>(Sym.SymInfo.Lang));
  writeVar(Strings.index(Sym.Name), OS);
  writeVar(Strings.index(Sym.Scope), OS);
  for (const auto &Loc : {Sym.Definition, Sym.CanonicalDeclaration}) {
    writeVar(Strings.index(Loc.FileURI), OS);
    for (const auto &Endpoint : {Loc.Start, Loc.End}) {
      writeVar(Endpoint.Line, OS);
      writeVar(Endpoint.Column, OS);
    }
  }
  writeVar(Sym.References, OS);
  OS.write(Sym.IsIndexedForCodeCompletion);
  OS.write(static_cast<uint8_t>(Sym.Origin));
  writeVar(Strings.index(Sym.Signature), OS);
  writeVar(Strings.index(Sym.CompletionSnippetSuffix), OS);
  writeVar(Strings.index(Sym.Documentation), OS);
  writeVar(Strings.index(Sym.ReturnType), OS);

  auto WriteInclude = [&](const Symbol::IncludeHeaderWithReferences &Include) {
    writeVar(Strings.index(Include.IncludeHeader), OS);
    writeVar(Include.References, OS);
  };
  // There are almost certainly few includes, so we can just write them.
  if (LLVM_LIKELY(Sym.IncludeHeaders.size() <= MaxIncludes)) {
    writeVar(Sym.IncludeHeaders.size(), OS);
    for (const auto &Include : Sym.IncludeHeaders)
      WriteInclude(Include);
  } else {
    // If there are too many, make sure we truncate the least important.
    using Pointer = const Symbol::IncludeHeaderWithReferences *;
    std::vector<Pointer> Pointers;
    for (const auto &Include : Sym.IncludeHeaders)
      Pointers.push_back(&Include);
    std::sort(Pointers.begin(), Pointers.end(), [](Pointer L, Pointer R) {
      return L->References > R->References;
    });
    Pointers.resize(MaxIncludes);

    writeVar(MaxIncludes, OS);
    for (Pointer P : Pointers)
      WriteInclude(*P);
  }

  assert(OS.tell() - StartOffset < SymbolSizeBound && "Symbol length unsafe!");
  (void)StartOffset; // Unused in NDEBUG;
}

Expected<Symbol> readSymbol(StringRef &Data, const StringTableIn &Strings) {
  // Usually we can skip bounds checks because the buffer is huge.
  // Near the end of the buffer, this would be unsafe. In this rare case, copy
  // the data into a bigger buffer so we can again skip the checks.
  if (LLVM_UNLIKELY(Data.size() < SymbolSizeBound)) {
    std::string Buf(Data);
    Buf.resize(SymbolSizeBound);
    StringRef ExtendedData = Buf;
    auto Ret = readSymbol(ExtendedData, Strings);
    unsigned BytesRead = Buf.size() - ExtendedData.size();
    if (BytesRead > Data.size())
      return makeError("read past end of data");
    Data = Data.drop_front(BytesRead);
    return Ret;
  }

#define READ_STRING(Field)                                                     \
  do {                                                                         \
    auto StringIndex = consumeVar(Data);                                       \
    if (LLVM_UNLIKELY(StringIndex >= Strings.Strings.size()))                  \
      return makeError("Bad string index");                                    \
    Field = Strings.Strings[StringIndex];                                      \
  } while (0)

  Symbol Sym;
  Sym.ID = SymbolID::fromRaw(consume(Data, 20));
  Sym.SymInfo.Kind = static_cast<index::SymbolKind>(consume8(Data));
  Sym.SymInfo.Lang = static_cast<index::SymbolLanguage>(consume8(Data));
  READ_STRING(Sym.Name);
  READ_STRING(Sym.Scope);
  for (SymbolLocation *Loc : {&Sym.Definition, &Sym.CanonicalDeclaration}) {
    READ_STRING(Loc->FileURI);
    for (auto &Endpoint : {&Loc->Start, &Loc->End}) {
      Endpoint->Line = consumeVar(Data);
      Endpoint->Column = consumeVar(Data);
    }
  }
  Sym.References = consumeVar(Data);
  Sym.IsIndexedForCodeCompletion = consume8(Data);
  Sym.Origin = static_cast<SymbolOrigin>(consume8(Data));
  READ_STRING(Sym.Signature);
  READ_STRING(Sym.CompletionSnippetSuffix);
  READ_STRING(Sym.Documentation);
  READ_STRING(Sym.ReturnType);
  unsigned IncludeHeaderN = consumeVar(Data);
  if (IncludeHeaderN > MaxIncludes)
    return makeError("too many IncludeHeaders");
  Sym.IncludeHeaders.resize(IncludeHeaderN);
  for (auto &I : Sym.IncludeHeaders) {
    READ_STRING(I.IncludeHeader);
    I.References = consumeVar(Data);
  }

#undef READ_STRING
  return std::move(Sym);
}

} // namespace

// FILE ENCODING
// A file is a RIFF chunk with type 'CdIx'.
// It contains the sections:
//   - meta: version number
//   - stri: string table
//   - symb: symbols

// The current versioning scheme is simple - non-current versions are rejected.
// If you make a breaking change, bump this version number to invalidate stored
// data. Later we may want to support some backward compatibility.
constexpr static uint32_t Version = 2;

Expected<IndexFileIn> readIndexFile(StringRef Data) {
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

  StringRef Meta = Chunks.lookup("meta");
  if (Meta.size() < 4 || consume32(Meta) != Version)
    return makeError("wrong version");

  auto Strings = readStringTable(Chunks.lookup("stri"));
  if (!Strings)
    return Strings.takeError();

  IndexFileIn Result;
  if (Chunks.count("symb")) {
    StringRef SymbolData = Chunks.lookup("symb");
    SymbolSlab::Builder Symbols;
    while (!SymbolData.empty())
      if (auto Sym = readSymbol(SymbolData, *Strings))
        Symbols.insert(*Sym);
      else
        return Sym.takeError();
    Result.Symbols = std::move(Symbols).build();
  }
  return std::move(Result);
}

raw_ostream &operator<<(raw_ostream &OS, const IndexFileOut &Data) {
  assert(Data.Symbols && "An index file without symbols makes no sense!");
  riff::File RIFF;
  RIFF.Type = riff::fourCC("CdIx");

  SmallString<4> Meta;
  {
    raw_svector_ostream MetaOS(Meta);
    write32(Version, MetaOS);
  }
  RIFF.Chunks.push_back({riff::fourCC("meta"), Meta});

  StringTableOut Strings;
  std::vector<Symbol> Symbols;
  for (const auto &Sym : *Data.Symbols) {
    Symbols.emplace_back(Sym);
    visitStrings(Symbols.back(), [&](StringRef &S) { Strings.intern(S); });
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

  return OS << RIFF;
}

} // namespace clangd
} // namespace clang
