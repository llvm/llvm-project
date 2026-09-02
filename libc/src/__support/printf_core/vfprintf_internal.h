//===-- Internal implementation header of vfprintf --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_VFPRINTF_INTERNAL_H
#define LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_VFPRINTF_INTERNAL_H

#include "src/__support/File/file.h"
#include "src/__support/arg_list.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/attributes.h" // For LIBC_INLINE
#include "src/__support/macros/config.h"
#include "src/__support/printf_core/core_structs.h"
#include "src/__support/printf_core/make_file_writer.h"
#include "src/__support/printf_core/printf_main.h"

#include "hdr/types/FILE.h"

namespace LIBC_NAMESPACE_DECL {
namespace printf_core {

LIBC_INLINE ErrorOr<size_t> vfprintf_internal(::FILE *__restrict stream,
                                              const char *__restrict format,
                                              internal::ArgList &args) {
  constexpr size_t BUFF_SIZE = 1024;
  char buffer[BUFF_SIZE];
  Writer writer = make_file_writer(buffer, BUFF_SIZE, stream);
  internal::flockfile(stream);
  auto retval = printf_main(&writer, format, args);
  if (!retval.has_value()) {
    internal::funlockfile(stream);
    return retval;
  }
  WriteBuffer<char> &wb = writer.get_write_buffer();
  int flushval = write_to_file_unlocked({wb.buff, wb.buff_cur}, stream);
  if (flushval != WRITE_OK)
    retval = Error(-flushval);
  internal::funlockfile(stream);
  return retval;
}

} // namespace printf_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_VFPRINTF_INTERNAL_H
