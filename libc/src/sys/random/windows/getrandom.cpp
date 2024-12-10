//===-- Windows implementation of getrandom -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/random/getrandom.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/errno/libc_errno.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <bcrypt.h>
#include <ntstatus.h>
#pragma comment(lib, "bcrypt.lib")

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, getrandom,
                   (void *buf, size_t buflen,
                    [[maybe_unused]] unsigned int flags)) {
  // https://learn.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptgenrandom
  // BCRYPT_USE_SYSTEM_PREFERRED_RNG
  // Use the system-preferred random number generator algorithm. The hAlgorithm
  // parameter must be NULL.

  // flags are ignored as Windows does not distinguish between urandom/random.
  // size_t is larger than ULONG. Linux API allows getrandom to return fewer
  // bytes than required. Hence, we trancate the size_t to ULONG. If user really
  // needs huge amount of bytes (which is highly unlikely), they can call
  // getrandom multiple times in a loop. This is also the common pattern in
  // Linux.

  // https://learn.microsoft.com/en-us/windows-hardware/drivers/gettingstarted/virtual-address-spaces
  // A 64-bit process on 64-bit Windows has a virtual address space within the
  // 128-terabyte range 0x000'00000000 through 0x7FFF'FFFFFFFF.
  if (buf == nullptr || cpp::bit_cast<INT_PTR>(buf) < 0) {
    libc_errno = EFAULT;
    return -1;
  }

  constexpr size_t PARAM_LIMIT =
      static_cast<size_t>(cpp::numeric_limits<ULONG>::max());
  constexpr size_t RETURN_LIMIT =
      static_cast<size_t>(cpp::numeric_limits<ssize_t>::max());
  buflen = buflen > PARAM_LIMIT ? PARAM_LIMIT : buflen;
  buflen = buflen > RETURN_LIMIT ? RETURN_LIMIT : buflen;
  NTSTATUS result = ::BCryptGenRandom(nullptr, static_cast<PUCHAR>(buf),
                                      static_cast<ULONG>(buflen),
                                      BCRYPT_USE_SYSTEM_PREFERRED_RNG);

  // not possible to overflow as we have truncated the limit.
  if (LIBC_LIKELY(result == STATUS_SUCCESS))
    return static_cast<ssize_t>(buflen);

  libc_errno = EINVAL;
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
