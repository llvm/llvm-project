//===-- Implementation for jmpbuf checksum ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/checksum.h"
#include "src/__support/OSUtil/io.h"
#include "src/stdlib/abort.h"
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {
namespace jmpbuf {
// random bytes from https://www.random.org/cgi-bin/randbyte?nbytes=8&format=h
// the cookie should not be zero otherwise it will be a bad seed as a multiplier
__UINTPTR_TYPE__ value_mask =
    static_cast<__UINTPTR_TYPE__>(0x3899'f0d3'5005'd953ull);
__UINTPTR_TYPE__ checksum_cookie =
    static_cast<__UINTPTR_TYPE__>(0xc7d9'd341'6afc'33f2ull);

// initialize the checksum state
void initialize() {
  union {
    struct {
      __UINTPTR_TYPE__ entropy0;
      __UINTPTR_TYPE__ entropy1;
    };
    char buffer[sizeof(__UINTPTR_TYPE__) * 2];
  };
  syscall_impl<long>(SYS_getrandom, buffer, sizeof(buffer), 0);
  // add in additional entropy
  jmpbuf::value_mask ^= entropy0;
  jmpbuf::checksum_cookie ^= entropy1;
}

extern "C" [[gnu::cold, noreturn]] void __libc_jmpbuf_corruption() {
  write_to_stderr("invalid checksum detected in longjmp\n");
  abort();
}

} // namespace jmpbuf
} // namespace LIBC_NAMESPACE_DECL
