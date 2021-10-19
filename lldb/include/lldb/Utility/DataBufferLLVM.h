//===--- DataBufferLLVM.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_DATABUFFERLLVM_H
#define LLDB_UTILITY_DATABUFFERLLVM_H

#include "lldb/Utility/DataBuffer.h"
#include "lldb/lldb-types.h"

#include <cstdint>
#include <memory>

namespace llvm {
class WritableMemoryBuffer;
class Twine;
}

namespace lldb_private {

class FileSystem;
class DataBufferLLVM : public DataBuffer {
public:
  ~DataBufferLLVM() override;

  uint8_t *GetBytes() override;
  const uint8_t *GetBytes() const override;
  lldb::offset_t GetByteSize() const override;

  char *GetChars() { return reinterpret_cast<char *>(GetBytes()); }

private:
  friend FileSystem;
  /// Construct a DataBufferLLVM from \p Buffer.  \p Buffer must be a valid
  /// pointer.
  explicit DataBufferLLVM(std::unique_ptr<llvm::WritableMemoryBuffer> Buffer);

  std::unique_ptr<llvm::WritableMemoryBuffer> Buffer;
};
}

#endif
