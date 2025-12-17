//===-- Implementation header of vfprintf for baremetal ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_BAREMETAL_VFPRINTF_INTERNAL_H
#define LLVM_LIBC_SRC_STDIO_BAREMETAL_VFPRINTF_INTERNAL_H

#include "hdr/types/FILE.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/arg_list.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/error_mapper.h"
#include "src/stdio/printf_core/printf_main.h"
#include "src/stdio/printf_core/writer.h"

namespace LIBC_NAMESPACE_DECL {

namespace internal {

LIBC_INLINE int write_hook(cpp::string_view str_view, void *cookie) {
  auto result =
      __llvm_libc_stdio_write(cookie, str_view.data(), str_view.size());
  if (result <= 0)
    return result;
  if (static_cast<size_t>(result) != str_view.size())
    return printf_core::FILE_WRITE_ERROR;
  return printf_core::WRITE_OK;
}

} // namespace internal

LIBC_INLINE int vfprintf_internal(::FILE *__restrict stream,
                                  const char *__restrict format,
                                  internal::ArgList &args) {
  static constexpr size_t BUFF_SIZE = 1024;
  char buffer[BUFF_SIZE];

  printf_core::FlushingBuffer wb(buffer, BUFF_SIZE, &internal::write_hook,
                                 stream);
  printf_core::Writer writer(wb);

  auto retval = printf_core::printf_main(&writer, format, args);
  if (!retval.has_value()) {
    libc_errno = printf_core::internal_error_to_errno(retval.error());
    return -1;
  }

  int flushval = wb.flush_to_stream();
  if (flushval != printf_core::WRITE_OK) {
    libc_errno = printf_core::internal_error_to_errno(-flushval);
    return -1;
  }

  if (retval.value() > static_cast<size_t>(cpp::numeric_limits<int>::max())) {
    libc_errno =
        printf_core::internal_error_to_errno(-printf_core::OVERFLOW_ERROR);
    return -1;
  }

  return static_cast<int>(retval.value());
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_BAREMETAL_VFPRINTF_INTERNAL_H
