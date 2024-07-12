//===-- Writer definition for printf ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "writer.h"
#include "src/__support/CPP/string_view.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/string/memory_utils/inline_memset.h"
#include <stddef.h>

namespace LIBC_NAMESPACE {
namespace printf_core {

int Writer::pad(char new_char, size_t length) {
  // First, fill as much of the buffer as possible with the padding char.
  size_t written = 0;
  const size_t buff_space = wb->buff_len - wb->buff_cur;
  // ASSERT: length > buff_space
  if (buff_space > 0) {
    inline_memset(wb->buff + wb->buff_cur, new_char, buff_space);
    wb->buff_cur += buff_space;
    written = buff_space;
  }

  // Next, overflow write the rest of length using the mini_buff.
  constexpr size_t MINI_BUFF_SIZE = 64;
  char mini_buff[MINI_BUFF_SIZE];
  inline_memset(mini_buff, new_char, MINI_BUFF_SIZE);
  cpp::string_view mb_string_view(mini_buff, MINI_BUFF_SIZE);
  while (written + MINI_BUFF_SIZE < length) {
    int result = wb->overflow_write(mb_string_view);
    if (result != WRITE_OK)
      return result;
    written += MINI_BUFF_SIZE;
  }
  cpp::string_view mb_substr = mb_string_view.substr(0, length - written);
  return wb->overflow_write(mb_substr);
}

} // namespace printf_core
} // namespace LIBC_NAMESPACE
