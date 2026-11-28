//===-- report_linux.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.h"

#if SCUDO_LINUX || SCUDO_TRUSTY

#include "common.h"
#include "internal_defs.h"
#include "report.h"
#include "report_linux.h"
#include "string_utils.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

// __GLIBC_PREREQ is only defined by glibc, so it can't be used in the same #if
// as the defined(__GLIBC__) check.
#if defined(__GLIBC__)
#if __GLIBC_PREREQ(2, 32)
#define SCUDO_HAS_STRERRORDESC_NP 1
#endif
#endif

namespace scudo {

// glibc's strerror() translates the message through gettext, which may call
// malloc(). If that happens while the allocator is being initialized (e.g. the
// initial reservation fails because of RLIMIT_AS), malloc() re-enters the
// allocator and deadlocks on the initialization mutex already held by the same
// thread. glibc's strerrordesc_np() is documented as async-signal-safe (so it
// can't allocate) and returns the untranslated message.
const char *getErrorDescription(int Err) {
#if defined(SCUDO_HAS_STRERRORDESC_NP)
  const char *Desc = strerrordesc_np(Err);
  return Desc ? Desc : "Unknown error";
#else
  // Other C libraries (e.g. musl, bionic) don't allocate in strerror().
  return strerror(Err);
#endif
}

// Fatal internal map() error (potentially OOM related).
void NORETURN reportMapError(uptr SizeIfOOM) {
  const int Err = errno;
  ScopedString Error;
  Error.append("Scudo ERROR: internal map failure (error desc=%s)",
               getErrorDescription(Err));
  if (SizeIfOOM)
    Error.append(" requesting %zuKB", SizeIfOOM >> 10);
  Error.append("\n");
  reportRawError(Error.data());
}

void NORETURN reportMapFixedError(uptr ExpectedAddr, uptr RequestedAddr) {
  ScopedString Error;
  Error.append("Scudo ERROR: internal map failure using fixed address "
               "(expected: 0x%zx requested: 0x%zx)\n",
               ExpectedAddr, RequestedAddr);
  reportRawError(Error.data());
}

void NORETURN reportUnmapError(uptr Addr, uptr Size) {
  const int Err = errno;
  ScopedString Error;
  Error.append("Scudo ERROR: internal unmap failure (error desc=%s) Addr 0x%zx "
               "Size %zu\n",
               getErrorDescription(Err), Addr, Size);
  reportRawError(Error.data());
}

void NORETURN reportProtectError(uptr Addr, uptr Size, int Prot) {
  const int Err = errno;
  ScopedString Error;
  Error.append(
      "Scudo ERROR: internal protect failure (error desc=%s) Addr 0x%zx "
      "Size %zu Prot %x\n",
      getErrorDescription(Err), Addr, Size, Prot);
  reportRawError(Error.data());
}

} // namespace scudo

#endif // SCUDO_LINUX || SCUDO_TRUSTY
