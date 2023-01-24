//===-- Format specifier converter implmentation for scanf -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/scanf_core/converter.h"

#include "src/__support/ctype_utils.h"
#include "src/stdio/scanf_core/core_structs.h"
#include "src/stdio/scanf_core/reader.h"

#ifndef LLVM_LIBC_SCANF_DISABLE_FLOAT
#include "src/stdio/scanf_core/float_converter.h"
#endif // LLVM_LIBC_SCANF_DISABLE_FLOAT
#include "src/stdio/scanf_core/current_pos_converter.h"
#include "src/stdio/scanf_core/int_converter.h"
#include "src/stdio/scanf_core/string_converter.h"

#include <stddef.h>

namespace __llvm_libc {
namespace scanf_core {

int convert(Reader *reader, const FormatSection &to_conv) {
  int ret_val = 0;
  switch (to_conv.conv_name) {
  case '%':
    return raw_match(reader, "%");
  case 's':
    ret_val = raw_match(reader, " ");
    if (ret_val != READ_OK)
      return ret_val;
    return convert_string(reader, to_conv);
  case 'c':
  case '[':
    return convert_string(reader, to_conv);
  case 'd':
  case 'i':
  case 'u':
  case 'o':
  case 'x':
  case 'X':
    ret_val = raw_match(reader, " ");
    if (ret_val != READ_OK)
      return ret_val;
    return convert_int(reader, to_conv);
#ifndef LLVM_LIBC_SCANF_DISABLE_FLOAT
  case 'f':
  case 'F':
  case 'e':
  case 'E':
  case 'a':
  case 'A':
  case 'g':
  case 'G':
    ret_val = raw_match(reader, " ");
    if (ret_val != READ_OK)
      return ret_val;
    return convert_float(reader, to_conv);
#endif // LLVM_LIBC_SCANF_DISABLE_FLOAT
  case 'n':
    return convert_current_pos(reader, to_conv);
    //   case 'p':
    //     ret_val = raw_match(reader, " ");
    //     if (ret_val != READ_OK)
    //       return ret_val;
    //     return convert_pointer(reader, to_conv);
  default:
    return raw_match(reader, to_conv.raw_string);
  }
  return -1;
}

// raw_string is assumed to have a positive size.
int raw_match(Reader *reader, cpp::string_view raw_string) {
  char cur_char = reader->getc();
  int ret_val = READ_OK;
  for (size_t i = 0; i < raw_string.size(); ++i) {
    // Any space character matches any number of space characters.
    if (internal::isspace(raw_string[i])) {
      while (internal::isspace(cur_char)) {
        cur_char = reader->getc();
      }
    } else {
      if (raw_string[i] == cur_char) {
        cur_char = reader->getc();
      } else {
        ret_val = MATCHING_FAILURE;
        break;
      }
    }
  }
  reader->ungetc(cur_char);
  return ret_val;
}

} // namespace scanf_core
} // namespace __llvm_libc
