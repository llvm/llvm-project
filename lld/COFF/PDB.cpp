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
#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileOutputBuffer.h"
#include <memory>

using namespace llvm;
using namespace llvm::support;
using namespace llvm::support::endian;

const int BlockSize = 4096;

void lld::coff::createPDB(StringRef Path) {
  // Create a file.
  size_t FileSize = BlockSize * 3;
  ErrorOr<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
      FileOutputBuffer::create(Path, FileSize);
  if (auto EC = BufferOrErr.getError())
    fatal(EC, "failed to open " + Path);
  std::unique_ptr<FileOutputBuffer> Buffer = std::move(*BufferOrErr);

  // Write the file header.
  uint8_t *Buf = Buffer->getBufferStart();
  auto *SB = reinterpret_cast<msf::SuperBlock *>(Buf);
  memcpy(SB->MagicBytes, msf::Magic, sizeof(msf::Magic));
  SB->BlockSize = BlockSize;

  // FreeBlockMap is a page number containing free page map bitmap.
  // Set a dummy value for now.
  SB->FreeBlockMapBlock = 1;

  SB->NumBlocks = FileSize / BlockSize;

  // Root directory is empty, containing only the length field.
  SB->NumDirectoryBytes = 4;

  // Root directory is on page 1.
  SB->BlockMapAddr = 1;

  // Write the root directory. Root stream is on page 2.
  write32le(Buf + BlockSize, 2);
  Buffer->commit();
}
