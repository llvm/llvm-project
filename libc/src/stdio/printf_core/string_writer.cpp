//===-- String Writer implementation for printf -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/printf_core/string_writer.h"
#include "src/string/memory_utils/memcpy_implementations.h"
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

void StringWriter::write(const char *__restrict to_write, size_t len) {
  if (len > available_capacity)
    len = available_capacity;

  if (len > 0) {
    inline_memcpy(cur_buffer, to_write, len);
    cur_buffer += len;
    available_capacity -= len;
  }
}

int write_to_string(void *raw_pointer, const char *__restrict to_write,
                    size_t len) {
  StringWriter *string_writer = reinterpret_cast<StringWriter *>(raw_pointer);
  string_writer->write(to_write, len);
  return 0;
}

} // namespace printf_core
} // namespace __llvm_libc
