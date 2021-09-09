//===-- string_utils.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_STRING_UTILS_H_
#define SCUDO_STRING_UTILS_H_

#include "internal_defs.h"
#include "vector.h"

#include <stdarg.h>

namespace scudo {

class ScopedString {
public:
  explicit ScopedString() { String.push_back('\0'); }
  uptr length() { return String.size() - 1; }
  const char *data() { return String.data(); }
  void clear() {
    String.clear();
    String.push_back('\0');
  }
  void append(const char *Format, va_list Args);
  void append(const char *Format, ...) FORMAT(2, 3);
  void output() const { outputRaw(String.data()); }

private:
  Vector<char> String;
};

int formatString(char *Buffer, uptr BufferLength, const char *Format, ...)
    FORMAT(3, 4);
void Printf(const char *Format, ...) FORMAT(1, 2);

} // namespace scudo

#endif // SCUDO_STRING_UTILS_H_
