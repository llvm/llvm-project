//===-- Baremetal implementation header of fwrite ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_BAREMETAL_FWRITE_INTERNAL_H
#define LLVM_LIBC_SRC_STDIO_BAREMETAL_FWRITE_INTERNAL_H

#include "hdr/types/FILE.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE size_t fwrite_internal(const void *buffer, size_t size, size_t nmemb,
                                   ::FILE *stream) {
  __llvm_libc_stdio_write(stream, reinterpret_cast<const char *>(buffer), size * nmemb);
  return nmemb;
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_BAREMETAL_FWRITE_INTERNAL_H
