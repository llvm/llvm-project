//===-- Baremetal implementation header of fputs ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_BAREMETAL_FPUTS_INTERNAL_H
#define LLVM_LIBC_SRC_STDIO_BAREMETAL_FPUTS_INTERNAL_H

#include "hdr/types/FILE.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE int fputs_internal(const char *str, ::FILE *stream) {
  cpp::string_view str_view(str);
  __llvm_libc_stdio_write(stream, str_view.data(), str_view.size());
  __llvm_libc_stdio_write(stream, "\n", 1);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_BAREMETAL_FPUTS_INTERNAL_H
