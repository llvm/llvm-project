//===-- FDReader.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FDReader.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <unistd.h>

namespace __llvm_libc {
namespace testutils {

FDReader::FDReader() {
  if (::pipe(pipefd)) {
    std::cerr << "pipe(2) failed";
    abort();
  }
}

FDReader::~FDReader() {
  ::close(pipefd[0]);
  ::close(pipefd[1]);
}

bool FDReader::matchWritten(const char *str) {

  ::close(pipefd[1]);

  constexpr ssize_t ChunkSize = 4096 * 4;

  char Buffer[ChunkSize];
  std::string PipeStr;
  std::string InputStr(str);

  for (int BytesRead; (BytesRead = ::read(pipefd[0], Buffer, ChunkSize));) {
    if (BytesRead > 0) {
      PipeStr.insert(PipeStr.size(), Buffer, BytesRead);
    } else {
      assert(0 && "Error reading from pipe");
      return false;
    }
  }

  return PipeStr == InputStr;
}

} // namespace testutils
} // namespace __llvm_libc
