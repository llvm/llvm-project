//===-- Windows implementation of getentropy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getentropy.h"
#include "hdr/errno_macros.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <bcrypt.h>
#include <ntstatus.h>
#pragma comment(lib, "bcrypt.lib")

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getentropy, (void *buffer, size_t length)) {
  __try {
    // check the length limit
    if (length > 256)
      __leave;

    NTSTATUS result = ::BCryptGenRandom(nullptr, static_cast<PUCHAR>(buffer),
                                        static_cast<ULONG>(length),
                                        BCRYPT_USE_SYSTEM_PREFERRED_RNG);

    if (result == STATUS_SUCCESS)
      return 0;

  } __except (EXCEPTION_EXECUTE_HANDLER) {
    // no need to handle exceptions specially
  }

  libc_errno = EIO;
  return -1;
}
} // namespace LIBC_NAMESPACE_DECL
