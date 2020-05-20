//===-- FDReader.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_TESTUTILS_FDREADER_H
#define LLVM_LIBC_UTILS_TESTUTILS_FDREADER_H

namespace __llvm_libc {
namespace testutils {

class FDReader {
  int pipefd[2];

public:
  FDReader();
  ~FDReader();

  int getWriteFD() { return pipefd[1]; }
  bool matchWritten(const char *);
};

} // namespace testutils
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_TESTUTILS_FDREADER_H
