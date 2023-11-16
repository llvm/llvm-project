//===-- Reader definition for scanf -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/scanf_core/reader.h"
#include <stddef.h>

namespace LIBC_NAMESPACE {
namespace scanf_core {

void Reader::ungetc(char c) {
  --cur_chars_read;
  if (rb != nullptr && rb->buff_cur > 0) {
    // While technically c should be written back to the buffer, in scanf we
    // always write the character that was already there. Additionally, the
    // buffer is most likely to contain a string that isn't part of a file,
    // which may not be writable.
    --(rb->buff_cur);
    return;
  }
  stream_ungetc(static_cast<int>(c), input_stream);
}
} // namespace scanf_core
} // namespace LIBC_NAMESPACE
