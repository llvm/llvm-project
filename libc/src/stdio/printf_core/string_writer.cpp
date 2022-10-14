//===-- String Writer implementation for printf -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/printf_core/string_writer.h"
#include "src/__support/CPP/string_view.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/string/memory_utils/memcpy_implementations.h"
#include "src/string/memory_utils/memset_implementations.h"
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

void StringWriter::write(cpp::string_view new_string) {
  size_t len = new_string.size();
  if (len > available_capacity)
    len = available_capacity;

  if (len > 0) {
    inline_memcpy(cur_buffer, new_string.data(), len);
    cur_buffer += len;
    available_capacity -= len;
  }
}

void StringWriter::write(char new_char, size_t len) {
  if (len > available_capacity)
    len = available_capacity;

  if (len > 0) {
    inline_memset(cur_buffer, static_cast<uint8_t>(new_char), len);
    cur_buffer += len;
    available_capacity -= len;
  }
}

void StringWriter::write(char new_char) {
  if (1 > available_capacity)
    return;

  cur_buffer[0] = new_char;
  ++cur_buffer;
  available_capacity -= 1;
}

int StringWriter::write_str(void *raw_pointer, cpp::string_view new_string) {
  StringWriter *string_writer = reinterpret_cast<StringWriter *>(raw_pointer);
  string_writer->write(new_string);
  return WRITE_OK;
}

int StringWriter::write_chars(void *raw_pointer, char new_char, size_t len) {
  StringWriter *string_writer = reinterpret_cast<StringWriter *>(raw_pointer);
  string_writer->write(new_char, len);
  return WRITE_OK;
}

int StringWriter::write_char(void *raw_pointer, char new_char) {
  StringWriter *string_writer = reinterpret_cast<StringWriter *>(raw_pointer);
  string_writer->write(new_char);
  return WRITE_OK;
}

} // namespace printf_core
} // namespace __llvm_libc
