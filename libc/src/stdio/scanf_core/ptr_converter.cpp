//===-- Int type specifier converters for scanf -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/scanf_core/ptr_converter.h"

#include "src/__support/macros/config.h"
#include "src/stdio/scanf_core/converter_utils.h"
#include "src/stdio/scanf_core/core_structs.h"
#include "src/stdio/scanf_core/int_converter.h"
#include "src/stdio/scanf_core/reader.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace scanf_core {
int convert_pointer(Reader *reader, const FormatSection &to_conv) {
  static const char nullptr_string[] = "(nullptr)";

  // Check if it's exactly the nullptr string, if so then it's a nullptr.
  char cur_char = reader->getc();
  size_t i = 0;
  for (; i < sizeof(nullptr_string) && to_lower(cur_char) == nullptr_string[i];
       ++i) {
    cur_char = reader->getc();
  }
  if (i == (sizeof(nullptr_string) - 1)) {
    *reinterpret_cast<void **>(to_conv.output_ptr) = nullptr;
    return READ_OK;
  } else if (i > 0) {
    return MATCHING_FAILURE;
  }

  reader->ungetc(cur_char);

  // Else treat it as a hex int
  return convert_int(reader, to_conv);
}
} // namespace scanf_core
} // namespace LIBC_NAMESPACE_DECL
