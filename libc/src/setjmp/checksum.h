//===-- Implementation header for jmpbuf checksum ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SETJMP_CHECKSUM_H
#define LLVM_LIBC_SRC_SETJMP_CHECKSUM_H

#ifndef LIBC_COPT_SETJMP_ENABLE_FORTIFICATION
#define LIBC_COPT_SETJMP_ENABLE_FORTIFICATION 1
#endif

#if LIBC_COPT_SETJMP_ENABLE_FORTIFICATION
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {
namespace jmpbuf {
// random bytes from https://www.random.org/cgi-bin/randbyte?nbytes=8&format=h
LIBC_INLINE __UINTPTR_TYPE__ value_mask = 0x3899'f0d3'5005'd953;
LIBC_INLINE __UINT64_TYPE__ checksum_cookie = 0xc7d9'd341'6afc'33f2;
// abitrary prime number
LIBC_INLINE constexpr __UINT64_TYPE__ ROTATION = 13;
// initialize the checksum state
LIBC_INLINE void initialize() {
  union {
    struct {
      __UINTPTR_TYPE__ entropy0;
      __UINT64_TYPE__ entropy1;
    };
    char buffer[sizeof(__UINTPTR_TYPE__) + sizeof(__UINT64_TYPE__)];
  };
  syscall_impl<long>(SYS_getrandom, buffer, sizeof(buffer), 0);
  // add in additional entropy
  jmpbuf::value_mask ^= entropy0;
  jmpbuf::checksum_cookie ^= entropy1;
}
} // namespace jmpbuf
} // namespace LIBC_NAMESPACE_DECL
#endif // LIBC_SETJMP_ENABLE_FORTIFICATION

#endif // LLVM_LIBC_SRC_SETJMP_CHECKSUM_H
