//===-- Writer definition for printf ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "writer.h"
#include "src/__support/CPP/string_view.h"
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

int Writer::write(cpp::string_view new_string) {
  chars_written += new_string.size();
  return str_write(output, new_string);
}

int Writer::write(char new_char, size_t length) {
  chars_written += length;
  return chars_write(output, new_char, length);
}

int Writer::write(char new_char) {
  chars_written += 1;
  return char_write(output, new_char);
}

} // namespace printf_core
} // namespace __llvm_libc
