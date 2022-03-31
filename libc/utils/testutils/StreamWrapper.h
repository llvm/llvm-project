//===-- StreamWrapper.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_TESTUTILS_STREAMWRAPPER_H
#define LLVM_LIBC_UTILS_TESTUTILS_STREAMWRAPPER_H

namespace __llvm_libc {
namespace testutils {

// StreamWrapper is necessary because llvm/Support/raw_ostream.h includes
// standard headers so we must provide streams through indirection to not
// expose the system libc headers.
class StreamWrapper {
protected:
  void *os;

public:
  StreamWrapper(void *OS) : os(OS) {}

  template <typename T> StreamWrapper &operator<<(T t);
};

StreamWrapper outs();

class OutputFileStream : public StreamWrapper {
public:
  explicit OutputFileStream(const char *FN);
  ~OutputFileStream();
};

} // namespace testutils
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_TESTUTILS_STREAMWRAPPER_H
