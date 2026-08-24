//===-- String type specifier converters for scanf --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_SCANF_CORE_STRING_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_SCANF_CORE_STRING_CONVERTER_H

#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/new.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/macros/config.h"
#include "src/stdio/scanf_core/core_structs.h"
#include "src/stdio/scanf_core/reader.h"
#include "src/string/memory_utils/inline_memcpy.h"

#include <stddef.h>

constexpr int ALLOCATION_SCALE = 2;
constexpr int ALLOCATION_BASE = 32;

namespace LIBC_NAMESPACE_DECL {
namespace scanf_core {

template <typename T>
int convert_string(Reader<T> *reader, const FormatSection &to_conv) {
  // %s "Matches a sequence of non-white-space characters"

  // %c "Matches a sequence of characters of exactly the number specified by the
  // field width (1 if no field width is present in the directive)"

  // %[ "Matches a nonempty sequence of characters from a set of expected
  // characters (the scanset)."
  size_t max_width = 0;
  if (to_conv.max_width > 0) {
    max_width = to_conv.max_width;
  } else {
    if (to_conv.conv_name == 'c') {
      max_width = 1;
    } else {
      max_width = cpp::numeric_limits<size_t>::max();
    }
  }

  char *output;
 AllocChecker ac;
#ifndef LIBC_COPT_SCANF_DISABLE_ALLOCATION
  size_t alloc_size;
  if ((to_conv.flags & NO_WRITE) == 0 && (to_conv.flags & ALLOCATE) != 0) {
    if (to_conv.conv_name == 'c')
      alloc_size = max_width + 1;
    else
      alloc_size =
          (max_width < ALLOCATION_BASE) ? max_width + 1 : ALLOCATION_BASE;
    output = new (ac) char[alloc_size];
    if (!ac) {
      return ALLOCATION_FAILURE;
    }
  } else {
    output = reinterpret_cast<char *>(to_conv.output_ptr);
  }
#else
  output = reinterpret_cast<char *>(to_conv.output_ptr);
#endif
  char cur_char = reader->getc();
  size_t i = 0;
  for (; i < max_width && cur_char != '\0'; ++i) {
    // If this is %s and we've hit a space, or if this is %[] and we've found
    // something not in the scanset.
    if ((to_conv.conv_name == 's' && internal::isspace(cur_char)) ||
        (to_conv.conv_name == '[' && !to_conv.scan_set.test(cur_char))) {
      break;
    }
    // if the NO_WRITE flag is not set, write to the output.
    if ((to_conv.flags & NO_WRITE) == 0) {
      output[i] = cur_char;
#ifndef LIBC_COPT_SCANF_DISABLE_ALLOCATION
      if (((to_conv.flags & ALLOCATE) != 0) && (i + 1) == alloc_size &&
          alloc_size < max_width) {
        alloc_size *= ALLOCATION_SCALE;
        if (alloc_size > max_width + 1)
          alloc_size = max_width + 1;
        char *tmp = new (ac) char[alloc_size];
        if (!ac) {
          delete[] output;
          reader->ungetc(cur_char);
          return ALLOCATION_FAILURE;
        }
        inline_memcpy(tmp, output, i + 1);
        delete[] output;
        output = tmp;
      }
#endif
    }
    cur_char = reader->getc();
  }

  // We always read one more character than will be used, so we have to put the
  // last one back.
  reader->ungetc(cur_char);

  bool null_terminate =
      (to_conv.conv_name != 'c') || ((to_conv.flags & ALLOCATE) != 0);

  // If this is %s or %[]
  if (null_terminate && (to_conv.flags & NO_WRITE) == 0) {
    // Always null terminate the string. This may cause a write to the
    // (max_width + 1) byte, which is correct. The max width describes the max
    // number of characters read from the input string, and doesn't necessarily
    // correspond to the output.
    output[i] = '\0';
  }

  if (i == 0) {
#ifndef LIBC_COPT_SCANF_DISABLE_ALLOCATION
    if ((to_conv.flags & ALLOCATE) != 0 && output)
      delete[] output;
#endif
    return MATCHING_FAILURE;
  }

#ifndef LIBC_COPT_SCANF_DISABLE_ALLOCATION
  if ((to_conv.flags & ALLOCATE) != 0) {
    *reinterpret_cast<char **>(to_conv.output_ptr) = output;
  }
#endif

  return READ_OK;
}

} // namespace scanf_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_SCANF_CORE_STRING_CONVERTER_H
