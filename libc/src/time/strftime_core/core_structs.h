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

// This is the value to be returned by conversions when no error has occurred.
constexpr int WRITE_OK = 0;
// These are the printf return values for when an error has occurred. They are
// all negative, and should be distinct.
constexpr int FILE_WRITE_ERROR = -1;
constexpr int FILE_STATUS_ERROR = -2;
constexpr int NULLPTR_WRITE_ERROR = -3;
constexpr int INT_CONVERSION_ERROR = -4;
constexpr int FIXED_POINT_CONVERSION_ERROR = -5;
constexpr int ALLOCATION_ERROR = -6;
} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_CORE_STRUCTS_H
