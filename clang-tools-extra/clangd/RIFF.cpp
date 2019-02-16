//===--- RIFF.cpp - Binary container file format --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RIFF.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
namespace clang {
namespace clangd {
namespace riff {

static Error makeError(const char *Msg) {
  return createStringError(inconvertibleErrorCode(), Msg);
}

Expected<Chunk> readChunk(StringRef &Stream) {
  if (Stream.size() < 8)
    return makeError("incomplete chunk header");
  Chunk C;
  std::copy(Stream.begin(), Stream.begin() + 4, C.ID.begin());
  Stream = Stream.drop_front(4);
  uint32_t Len = support::endian::read32le(Stream.take_front(4).begin());
  Stream = Stream.drop_front(4);
  if (Stream.size() < Len)
    return makeError("truncated chunk");
  C.Data = Stream.take_front(Len);
  Stream = Stream.drop_front(Len);
  if (Len % 2 & !Stream.empty()) { // Skip padding byte.
    if (Stream.front())
      return makeError("nonzero padding byte");
    Stream = Stream.drop_front();
  }
  return std::move(C);
}

raw_ostream &operator<<(raw_ostream &OS, const Chunk &C) {
  OS.write(C.ID.data(), C.ID.size());
  char Size[4];
  support::endian::write32le(Size, C.Data.size());
  OS.write(Size, sizeof(Size));
  OS << C.Data;
  if (C.Data.size() % 2)
    OS.write(0);
  return OS;
}

Expected<File> readFile(StringRef Stream) {
  auto RIFF = readChunk(Stream);
  if (!RIFF)
    return RIFF.takeError();
  if (RIFF->ID != fourCC("RIFF"))
    return makeError("not a RIFF container");
  if (RIFF->Data.size() < 4)
    return makeError("RIFF chunk too short");
  File F;
  std::copy(RIFF->Data.begin(), RIFF->Data.begin() + 4, F.Type.begin());
  for (StringRef Body = RIFF->Data.drop_front(4); !Body.empty();)
    if (auto Chunk = readChunk(Body)) {
      F.Chunks.push_back(*Chunk);
    } else
      return Chunk.takeError();
  return std::move(F);
}

raw_ostream &operator<<(raw_ostream &OS, const File &F) {
  // To avoid copies, we serialize the outer RIFF chunk "by hand".
  size_t DataLen = 4; // Predict length of RIFF chunk data.
  for (const auto &C : F.Chunks)
    DataLen += 4 + 4 + C.Data.size() + (C.Data.size() % 2);
  OS << "RIFF";
  char Size[4];
  support::endian::write32le(Size, DataLen);
  OS.write(Size, sizeof(Size));
  OS.write(F.Type.data(), F.Type.size());
  for (const auto &C : F.Chunks)
    OS << C;
  return OS;
}

} // namespace riff
} // namespace clangd
} // namespace clang
