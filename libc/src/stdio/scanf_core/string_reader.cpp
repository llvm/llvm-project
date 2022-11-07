//===-- String Reader implementation for scanf ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/scanf_core/string_reader.h"
#include <stddef.h>

namespace __llvm_libc {
namespace scanf_core {

char StringReader::get_char() {
  char cur_char = string[cur_index];
  ++cur_index;
  return cur_char;
}

void StringReader::unget_char() { --cur_index; }

} // namespace scanf_core
} // namespace __llvm_libc
