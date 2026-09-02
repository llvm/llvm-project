//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation helpers for Writer<OverflowMode::RESIZE_BUFFER, CharT>.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_MAKE_RESIZING_WRITER_H
#define LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_MAKE_RESIZING_WRITER_H

#include "hdr/func/free.h"
#include "hdr/func/malloc.h"
#include "hdr/func/realloc.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/printf_core/writer.h"

namespace LIBC_NAMESPACE_DECL {
namespace printf_core {

template <typename CharT>
LIBC_INLINE Writer<Mode<OverflowMode::RESIZE_BUFFER>::value, CharT>
make_resizing_writer(CharT *buffer, size_t buffer_len, bool buffer_on_stack) {
  return Writer(buffer, buffer_len,
                make_overflow_writer<OverflowMode::RESIZE_BUFFER, CharT>(
                    buffer_on_stack ? buffer : nullptr));
}

// Handles overflow by resizing `wb` to have enough capacity for `new_str`, and
// then copying the `new_str` contents into the buffer.
template <typename CharT>
LIBC_INLINE int
overflow_write_resize_buffer(WriteBuffer<CharT> &wb,
                             cpp::basic_string_view<CharT> new_str,
                             void *initial_stack_buffer) {
  size_t new_size = new_str.size() + wb.buff_cur;
  const bool is_on_stack = (wb.buff == initial_stack_buffer);
  char *new_buff = static_cast<char *>(
      is_on_stack ? malloc(new_size + 1)
                  : realloc(wb.buff, new_size + 1)); // +1 for null
  if (new_buff == nullptr) {
    if (!is_on_stack)
      free(wb.buff);
    return ALLOCATION_ERROR;
  }
  if (is_on_stack)
    inline_memcpy(new_buff, wb.buff, wb.buff_cur);
  wb.buff = new_buff;
  inline_memcpy(wb.buff + wb.buff_cur, new_str.data(), new_str.size());
  wb.buff_cur = new_size;
  wb.buff_len = new_size;
  return WRITE_OK;
}

} // namespace printf_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_MAKE_RESIZING_WRITER_H
