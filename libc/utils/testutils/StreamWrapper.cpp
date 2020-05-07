//===-- StreamWrapper.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StreamWrapper.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <memory>
#include <string>

namespace __llvm_libc {
namespace testutils {

StreamWrapper outs() { return {std::addressof(llvm::outs())}; }

template <typename T> StreamWrapper &StreamWrapper::operator<<(T t) {
  assert(OS);
  llvm::raw_ostream &Stream = *reinterpret_cast<llvm::raw_ostream *>(OS);
  Stream << t;
  return *this;
}

template StreamWrapper &StreamWrapper::operator<<<void *>(void *t);
template StreamWrapper &StreamWrapper::operator<<<const char *>(const char *t);
template StreamWrapper &StreamWrapper::operator<<<char *>(char *t);
template StreamWrapper &StreamWrapper::operator<<<char>(char t);
template StreamWrapper &StreamWrapper::operator<<<short>(short t);
template StreamWrapper &StreamWrapper::operator<<<int>(int t);
template StreamWrapper &StreamWrapper::operator<<<long>(long t);
template StreamWrapper &StreamWrapper::operator<<<long long>(long long t);
template StreamWrapper &
    StreamWrapper::operator<<<unsigned char>(unsigned char t);
template StreamWrapper &
    StreamWrapper::operator<<<unsigned short>(unsigned short t);
template StreamWrapper &StreamWrapper::operator<<<unsigned int>(unsigned int t);
template StreamWrapper &
    StreamWrapper::operator<<<unsigned long>(unsigned long t);
template StreamWrapper &
    StreamWrapper::operator<<<unsigned long long>(unsigned long long t);
template StreamWrapper &StreamWrapper::operator<<<bool>(bool t);
template StreamWrapper &StreamWrapper::operator<<<std::string>(std::string t);

} // namespace testutils
} // namespace __llvm_libc
