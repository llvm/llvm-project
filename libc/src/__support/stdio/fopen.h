//===-- Common implementation of fopen ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_STDIO_FOPEN_H
#define LLVM_LIBC_SRC___SUPPORT_STDIO_FOPEN_H

#include "hdr/types/FILE.h"
#include "src/__support/File/file.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

namespace stdio_internal {

LIBC_INLINE static constexpr ::FILE *fopen(const char *__restrict name,
                                           const char *__restrict mode) {
  auto result = LIBC_NAMESPACE::openfile(name, mode);
  if (!result.has_value()) {
    libc_errno = result.error();
    return nullptr;
  }
  return reinterpret_cast<::FILE *>(result.value());
}

} // namespace stdio_internal

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_STDIO_FOPEN_H
