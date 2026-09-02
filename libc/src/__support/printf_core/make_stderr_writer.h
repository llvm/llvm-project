//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation helpers for Writer<OverflowMode::FLUSH_TO_STDERR, CharT>.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_MAKE_STDERR_WRITER_H
#define LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_MAKE_STDERR_WRITER_H

#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/macros/config.h"
#include "src/__support/printf_core/writer.h"

namespace LIBC_NAMESPACE_DECL {
namespace printf_core {

template <typename CharT>
LIBC_INLINE Writer<Mode<OverflowMode::FLUSH_TO_STDERR>::value, CharT>
make_stderr_writer(CharT *buffer, size_t buffer_len) {
  return Writer(
      buffer, buffer_len,
      make_overflow_writer<OverflowMode::FLUSH_TO_STDERR, CharT>(nullptr));
}

LIBC_INLINE void flush_to_stderr(WriteBuffer<char> &wb) {
  if (wb.buff_cur == 0)
    return;

  write_to_stderr({wb.buff, wb.buff_cur});
  wb.buff_cur = 0;
}

// Handles overflow by flushing the current contents of `wb` and `new_str` to
// stderr.
template <typename CharT>
LIBC_INLINE int
overflow_write_flush_to_stderr(WriteBuffer<CharT> &wb,
                               cpp::basic_string_view<CharT> new_str, void *) {
  flush_to_stderr(wb);
  if (new_str.size() > 0)
    write_to_stderr(new_str);
  return WRITE_OK;
}

} // namespace printf_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_MAKE_STDERR_WRITER_H
