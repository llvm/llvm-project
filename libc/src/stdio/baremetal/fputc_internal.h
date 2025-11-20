//===-- Baremetal implementation header of fputc ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_BAREMETAL_FPUTC_INTERNAL_H
#define LLVM_LIBC_SRC_STDIO_BAREMETAL_FPUTC_INTERNAL_H

#include "hdr/types/FILE.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE int fputc_internal(int c, ::FILE *stream) {
  char ch = static_cast<char>(c);
  cpp::string_view str_view(&ch, 1);
  __llvm_libc_stdio_write(stream, str_view.data(), str_view.size());
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_BAREMETAL_FPUTC_INTERNAL_H
