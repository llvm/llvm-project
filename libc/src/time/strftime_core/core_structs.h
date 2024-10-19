//===-- Core Structures for printf ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CORE_STRUCTS_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CORE_STRUCTS_H

#include "src/__support/CPP/string_view.h"
#include <time.h>

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

struct FormatSection {
  bool has_conv{false};
  bool isE{false};
  bool isO{false};
  cpp::string_view raw_string{};
  char conv_name;
  const struct tm *time;
  int min_width{0};
  char padding;
};

#define RET_IF_RESULT_NEGATIVE(func)                                           \
  {                                                                            \
    int result = (func);                                                       \
    if (result < 0)                                                            \
      return result;                                                           \
  }

} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CORE_STRUCTS_H
