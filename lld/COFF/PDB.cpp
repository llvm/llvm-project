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
#include "llvm/Support/FileOutputBuffer.h"
#include <memory>

using namespace llvm;

const int PageSize = 4096;
const uint8_t Magic[32] = "Microsoft C/C++ MSF 7.00\r\n\032DS\0\0";

void lld::coff::createPDB(StringRef Path) {
  // Create a file.
  size_t FileSize = PageSize * 3;
  ErrorOr<std::unique_ptr<FileOutputBuffer>> BufOrErr =
      FileOutputBuffer::create(Path, FileSize);
  error(BufOrErr, Twine("failed to open ") + Path);
  std::unique_ptr<FileOutputBuffer> Buf = std::move(*BufOrErr);

  // Write the file magic.
  uint8_t *P = Buf->getBufferStart();
  memcpy(P, Magic, sizeof(Magic));
}
