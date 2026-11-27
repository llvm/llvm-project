//===-- PipeBase.h -----------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_PIPEBASE_H
#define LLDB_HOST_PIPEBASE_H

#include "lldb/Utility/Status.h"
#include "lldb/Utility/Timeout.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace lldb_private {
class PipeBase {
public:
  virtual ~PipeBase();

  virtual Status CreateNew() = 0;
  virtual Status CreateNew(llvm::StringRef name) = 0;
  virtual Status CreateWithUniqueName(llvm::StringRef prefix,
                                      llvm::SmallVectorImpl<char> &name) = 0;

  virtual Status OpenAsReader(llvm::StringRef name) = 0;

  virtual llvm::Error OpenAsWriter(llvm::StringRef name,
                                   const Timeout<std::micro> &timeout) = 0;

  virtual bool CanRead() const = 0;
  virtual bool CanWrite() const = 0;

  virtual lldb::pipe_t GetReadPipe() const = 0;
  virtual lldb::pipe_t GetWritePipe() const = 0;

  virtual int GetReadFileDescriptor() const = 0;
  virtual int GetWriteFileDescriptor() const = 0;
  virtual int ReleaseReadFileDescriptor() = 0;
  virtual int ReleaseWriteFileDescriptor() = 0;
  virtual void CloseReadFileDescriptor() = 0;
  virtual void CloseWriteFileDescriptor() = 0;

  // Close both descriptors
  virtual void Close() = 0;

  // Delete named pipe.
  virtual Status Delete(llvm::StringRef name) = 0;

  virtual llvm::Expected<size_t>
  Write(const void *buf, size_t size,
        const Timeout<std::micro> &timeout = std::nullopt) = 0;

  virtual llvm::Expected<size_t>
  Read(void *buf, size_t size,
       const Timeout<std::micro> &timeout = std::nullopt) = 0;
};
}

#endif
