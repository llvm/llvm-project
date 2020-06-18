//===-- FDReader.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FDReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cassert>
#include <cstring>
#include <unistd.h>

namespace __llvm_libc {
namespace testutils {

FDReader::FDReader() {
  if (::pipe(pipefd))
    llvm::report_fatal_error("pipe(2) failed");
}

FDReader::~FDReader() {
  ::close(pipefd[0]);
  ::close(pipefd[1]);
}

bool FDReader::matchWritten(const char *str) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufOrErr =
      llvm::MemoryBuffer::getOpenFile(pipefd[0], "<pipe>",
                                      /* FileSize (irrelevant) */ 0);
  if (!bufOrErr) {
    assert(0 && "Error reading from pipe");
    return false;
  }
  const llvm::MemoryBuffer &buf = **bufOrErr;
  return !std::strncmp(buf.getBufferStart(), str, buf.getBufferSize());
}

} // namespace testutils
} // namespace __llvm_libc
