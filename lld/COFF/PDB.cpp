//===- PDB.cpp ------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Error.h"
#include "Symbols.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileOutputBuffer.h"
#include <memory>

using namespace llvm;
using namespace llvm::support;
using namespace llvm::support::endian;

const int PageSize = 4096;
const uint8_t Magic[32] = "Microsoft C/C++ MSF 7.00\r\n\032DS\0\0";

namespace {
struct PDBHeader {
  uint8_t Magic[32];
  ulittle32_t PageSize;
  ulittle32_t FpmPage;
  ulittle32_t PageCount;
  ulittle32_t RootSize;
  ulittle32_t Reserved;
  ulittle32_t RootPointer;
};
}

void lld::coff::createPDB(StringRef Path) {
  // Create a file.
  size_t FileSize = PageSize * 3;
  ErrorOr<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
      FileOutputBuffer::create(Path, FileSize);
  error(BufferOrErr, Twine("failed to open ") + Path);
  std::unique_ptr<FileOutputBuffer> Buffer = std::move(*BufferOrErr);

  // Write the file header.
  uint8_t *Buf = Buffer->getBufferStart();
  auto *Hdr = reinterpret_cast<PDBHeader *>(Buf);
  memcpy(Hdr->Magic, Magic, sizeof(Magic));
  Hdr->PageSize = PageSize;
  // I don't know what FpmPage field means, but it must not be 0.
  Hdr->FpmPage = 1;
  Hdr->PageCount = FileSize / PageSize;
  // Root directory is empty, containing only the length field.
  Hdr->RootSize = 4;
  // Root directory is on page 1.
  Hdr->RootPointer = 1;

  // Write the root directory. Root stream is on page 2.
  write32le(Buf + PageSize, 2);
  Buffer->commit();
}
