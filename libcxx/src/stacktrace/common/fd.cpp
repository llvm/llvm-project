//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fd.h"
#include "failed.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

fd_streambuf::~fd_streambuf() = default;
fd_istream::~fd_istream()     = default;

int fd_streambuf::underflow() {
  int bytesRead = ::read(fd_, buf_, size_);
  if (bytesRead < 0) {
    throw failed("I/O error reading from child process", errno);
  }
  if (bytesRead == 0) {
    return traits_type::eof();
  }
  setg(buf_, buf_, buf_ + bytesRead);
  return int(*buf_);
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD
