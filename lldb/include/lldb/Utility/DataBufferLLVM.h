//===--- DataBufferLLVM.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_DATABUFFERLLVM_H
#define LLDB_CORE_DATABUFFERLLVM_H

#include "lldb/Utility/DataBuffer.h"

#include <memory>

namespace llvm {
class MemoryBuffer;
class Twine;
}

namespace lldb_private {

class DataBufferLLVM : public DataBuffer {
public:
  ~DataBufferLLVM();

  static std::shared_ptr<DataBufferLLVM>
  CreateSliceFromPath(const llvm::Twine &Path, uint64_t Size, uint64_t Offset, bool Private = false);

  static std::shared_ptr<DataBufferLLVM>
  CreateFromPath(const llvm::Twine &Path, bool NullTerminate = false, bool Private = false);

  uint8_t *GetBytes() override;
  const uint8_t *GetBytes() const override;
  lldb::offset_t GetByteSize() const override;

  char *GetChars() { return reinterpret_cast<char *>(GetBytes()); }

private:
  /// \brief Construct a DataBufferLLVM from \p Buffer.  \p Buffer must be a
  /// valid pointer.
  explicit DataBufferLLVM(std::unique_ptr<llvm::MemoryBuffer> Buffer);
  const uint8_t *GetBuffer() const;

  std::unique_ptr<llvm::MemoryBuffer> Buffer;
};
}

#endif
