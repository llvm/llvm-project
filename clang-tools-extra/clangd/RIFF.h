//===--- RIFF.h - Binary container file format -------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tools for reading and writing data in RIFF containers.
//
// A chunk consists of:
//   - ID      : char[4]
//   - Length  : uint32
//   - Data    : byte[Length]
//   - Padding : byte[Length % 2]
// The semantics of a chunk's Data are determined by its ID.
// The format makes it easy to skip over uninteresting or unknown chunks.
//
// A RIFF file is a single chunk with ID "RIFF". Its Data is:
//   - Type    : char[4]
//   - Chunks  : chunk[]
//
// This means that a RIFF file consists of:
//   - "RIFF"          : char[4]
//   - File length - 8 : uint32
//   - File type       : char[4]
//   - Chunks          : chunk[]
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_RIFF_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_RIFF_H
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"
#include <array>

namespace clang {
namespace clangd {
namespace riff {

// A FourCC identifies a chunk in a file, or the type of file itself.
using FourCC = std::array<char, 4>;
// Get a FourCC from a string literal, e.g. fourCC("RIFF").
inline constexpr FourCC fourCC(const char (&Literal)[5]) {
  return FourCC{{Literal[0], Literal[1], Literal[2], Literal[3]}};
}
// A chunk is a section in a RIFF container.
struct Chunk {
  FourCC ID;
  llvm::StringRef Data;
};
inline bool operator==(const Chunk &L, const Chunk &R) {
  return std::tie(L.ID, L.Data) == std::tie(R.ID, R.Data);
}
// A File is a RIFF container, which is a typed chunk sequence.
struct File {
  FourCC Type;
  std::vector<Chunk> Chunks;
};
inline bool operator==(const File &L, const File &R) {
  return std::tie(L.Type, L.Chunks) == std::tie(R.Type, R.Chunks);
}

// Reads a single chunk from the start of Stream.
// Stream is updated to exclude the consumed chunk.
llvm::Expected<Chunk> readChunk(llvm::StringRef &Stream);

// Serialize a single chunk to OS.
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Chunk &);

// Parses a RIFF file consisting of a single RIFF chunk.
llvm::Expected<File> readFile(llvm::StringRef Stream);

// Serialize a RIFF file (i.e. a single RIFF chunk) to OS.
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const File &);

} // namespace riff
} // namespace clangd
} // namespace clang
#endif
