//===-- Implementation of bzero -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/bzero.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/memset_utils.h"

namespace __llvm_libc {

void LLVM_LIBC_ENTRYPOINT(bzero)(void *ptr, size_t count) {
  GeneralPurposeMemset(reinterpret_cast<char *>(ptr), 0, count);
}

} // namespace __llvm_libc
