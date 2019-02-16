//===--- PostingList.cpp - Symbol identifiers storage interface -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PostingList.h"
#include "Iterator.h"
#include "Token.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
namespace clang {
namespace clangd {
namespace dex {
namespace {

/// Implements iterator of PostingList chunks. This requires iterating over two
/// levels: the first level iterator iterates over the chunks and decompresses
/// them on-the-fly when the contents of chunk are to be seen.
class ChunkIterator : public Iterator {
public:
  explicit ChunkIterator(const Token *Tok, ArrayRef<Chunk> Chunks)
      : Tok(Tok), Chunks(Chunks), CurrentChunk(Chunks.begin()) {
    if (!Chunks.empty()) {
      DecompressedChunk = CurrentChunk->decompress();
      CurrentID = DecompressedChunk.begin();
    }
  }

  bool reachedEnd() const override { return CurrentChunk == Chunks.end(); }

  /// Advances cursor to the next item.
  void advance() override {
    assert(!reachedEnd() &&
           "Posting List iterator can't advance() at the end.");
    ++CurrentID;
    normalizeCursor();
  }

  /// Applies binary search to advance cursor to the next item with DocID
  /// equal or higher than the given one.
  void advanceTo(DocID ID) override {
    assert(!reachedEnd() &&
           "Posting List iterator can't advance() at the end.");
    if (ID <= peek())
      return;
    advanceToChunk(ID);
    // Try to find ID within current chunk.
    CurrentID = std::lower_bound(CurrentID, std::end(DecompressedChunk), ID);
    normalizeCursor();
  }

  DocID peek() const override {
    assert(!reachedEnd() && "Posting List iterator can't peek() at the end.");
    return *CurrentID;
  }

  float consume() override {
    assert(!reachedEnd() &&
           "Posting List iterator can't consume() at the end.");
    return 1;
  }

  size_t estimateSize() const override {
    return Chunks.size() * ApproxEntriesPerChunk;
  }

private:
  raw_ostream &dump(raw_ostream &OS) const override {
    if (Tok != nullptr)
      return OS << *Tok;
    OS << '[';
    const char *Sep = "";
    for (const Chunk &C : Chunks)
      for (const DocID Doc : C.decompress()) {
        OS << Sep << Doc;
        Sep = " ";
      }
    return OS << ']';
  }

  /// If the cursor is at the end of a chunk, place it at the start of the next
  /// chunk.
  void normalizeCursor() {
    // Invariant is already established if examined chunk is not exhausted.
    if (CurrentID != std::end(DecompressedChunk))
      return;
    // Advance to next chunk if current one is exhausted.
    ++CurrentChunk;
    if (CurrentChunk == Chunks.end()) // Reached the end of PostingList.
      return;
    DecompressedChunk = CurrentChunk->decompress();
    CurrentID = DecompressedChunk.begin();
  }

  /// Advances CurrentChunk to the chunk which might contain ID.
  void advanceToChunk(DocID ID) {
    if ((CurrentChunk != Chunks.end() - 1) &&
        ((CurrentChunk + 1)->Head <= ID)) {
      // Find the next chunk with Head >= ID.
      CurrentChunk = std::lower_bound(
          CurrentChunk + 1, Chunks.end(), ID,
          [](const Chunk &C, const DocID ID) { return C.Head <= ID; });
      --CurrentChunk;
      DecompressedChunk = CurrentChunk->decompress();
      CurrentID = DecompressedChunk.begin();
    }
  }

  const Token *Tok;
  ArrayRef<Chunk> Chunks;
  /// Iterator over chunks.
  /// If CurrentChunk is valid, then DecompressedChunk is
  /// CurrentChunk->decompress() and CurrentID is a valid (non-end) iterator
  /// into it.
  decltype(Chunks)::const_iterator CurrentChunk;
  SmallVector<DocID, Chunk::PayloadSize + 1> DecompressedChunk;
  /// Iterator over DecompressedChunk.
  decltype(DecompressedChunk)::iterator CurrentID;

  static constexpr size_t ApproxEntriesPerChunk = 15;
};

static constexpr size_t BitsPerEncodingByte = 7;

/// Writes a variable length DocID into the buffer and updates the buffer size.
/// If it doesn't fit, returns false and doesn't write to the buffer.
bool encodeVByte(DocID Delta, MutableArrayRef<uint8_t> &Payload) {
  assert(Delta != 0 && "0 is not a valid PostingList delta.");
  // Calculate number of bytes Delta encoding would take by examining the
  // meaningful bits.
  unsigned Width = 1 + findLastSet(Delta) / BitsPerEncodingByte;
  if (Width > Payload.size())
    return false;

  do {
    uint8_t Encoding = Delta & 0x7f;
    Delta >>= 7;
    Payload.front() = Delta ? Encoding | 0x80 : Encoding;
    Payload = Payload.drop_front();
  } while (Delta != 0);
  return true;
}

/// Use Variable-length Byte (VByte) delta encoding to compress sorted list of
/// DocIDs. The compression stores deltas (differences) between subsequent
/// DocIDs and encodes these deltas utilizing the least possible number of
/// bytes.
///
/// Each encoding byte consists of two parts: the first bit (continuation bit)
/// indicates whether this is the last byte (0 if this byte is the last) of
/// current encoding and seven bytes a piece of DocID (payload). DocID contains
/// 32 bits and therefore it takes up to 5 bytes to encode it (4 full 7-bit
/// payloads and one 4-bit payload), but in practice it is expected that gaps
/// (deltas) between subsequent DocIDs are not large enough to require 5 bytes.
/// In very dense posting lists (with average gaps less than 128) this
/// representation would be 4 times more efficient than raw DocID array.
///
/// PostingList encoding example:
///
/// DocIDs    42            47        7000
/// gaps                    5         6958
/// Encoding  (raw number)  00000101  10110110 00101110
std::vector<Chunk> encodeStream(ArrayRef<DocID> Documents) {
  assert(!Documents.empty() && "Can't encode empty sequence.");
  std::vector<Chunk> Result;
  Result.emplace_back();
  DocID Last = Result.back().Head = Documents.front();
  MutableArrayRef<uint8_t> RemainingPayload = Result.back().Payload;
  for (DocID Doc : Documents.drop_front()) {
    if (!encodeVByte(Doc - Last, RemainingPayload)) { // didn't fit, flush chunk
      Result.emplace_back();
      Result.back().Head = Doc;
      RemainingPayload = Result.back().Payload;
    }
    Last = Doc;
  }
  return std::vector<Chunk>(Result); // no move, shrink-to-fit
}

/// Reads variable length DocID from the buffer and updates the buffer size. If
/// the stream is terminated, return None.
Optional<DocID> readVByte(ArrayRef<uint8_t> &Bytes) {
  if (Bytes.front() == 0 || Bytes.empty())
    return None;
  DocID Result = 0;
  bool HasNextByte = true;
  for (size_t Length = 0; HasNextByte && !Bytes.empty(); ++Length) {
    assert(Length <= 5 && "Malformed VByte encoding sequence.");
    // Write meaningful bits to the correct place in the document decoding.
    Result |= (Bytes.front() & 0x7f) << (BitsPerEncodingByte * Length);
    if ((Bytes.front() & 0x80) == 0)
      HasNextByte = false;
    Bytes = Bytes.drop_front();
  }
  return Result;
}

} // namespace

SmallVector<DocID, Chunk::PayloadSize + 1> Chunk::decompress() const {
  SmallVector<DocID, Chunk::PayloadSize + 1> Result{Head};
  ArrayRef<uint8_t> Bytes(Payload);
  DocID Delta;
  for (DocID Current = Head; !Bytes.empty(); Current += Delta) {
    auto MaybeDelta = readVByte(Bytes);
    if (!MaybeDelta)
      break;
    Delta = *MaybeDelta;
    Result.push_back(Current + Delta);
  }
  return SmallVector<DocID, Chunk::PayloadSize + 1>{Result};
}

PostingList::PostingList(ArrayRef<DocID> Documents)
    : Chunks(encodeStream(Documents)) {}

std::unique_ptr<Iterator> PostingList::iterator(const Token *Tok) const {
  return llvm::make_unique<ChunkIterator>(Tok, Chunks);
}

} // namespace dex
} // namespace clangd
} // namespace clang
