//===-- Standalone implementation std::string_view --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_CPP_STRINGVIEW_H
#define LLVM_LIBC_UTILS_CPP_STRINGVIEW_H

#include <stddef.h>

namespace __llvm_libc {
namespace cpp {

// This is very simple alternate of the std::string_view class. There is no
// bounds check performed in any of the methods. The callers are expected to
// do the checks before invoking the methods.
//
// This class will be extended as needed in future.
class StringView {
private:
  const char *Data;
  size_t Len;

public:
  StringView() : Data(nullptr), Len(0) {}

  // Assumes Str is a null-terminated string. The length of the string does
  // not include the terminating null character.
  explicit StringView(const char *Str) : Data(Str), Len(0) {
    if (Str == nullptr)
      return;
    for (const char *D = Data; *D != '\0'; ++D, ++Len)
      ;
    if (Len == 0)
      Data = nullptr;
  }

  explicit StringView(const char *Str, size_t N)
      : Data(N ? Str : nullptr), Len(Str == nullptr ? 0 : N) {}

  const char *data() const { return Data; }

  size_t size() { return Len; }

  StringView remove_prefix(size_t N) const {
    if (N >= Len)
      return StringView();
    return StringView(Data + N, Len - N);
  }

  StringView remove_suffix(size_t N) const {
    if (N >= Len)
      return StringView();
    return StringView(Data, Len - N);
  }

  // An equivalent method is not available in std::string_view.
  StringView trim(char C) const {
    if (Len == 0)
      return StringView();

    const char *NewStart = Data;
    size_t PrefixLen = 0;
    for (; PrefixLen < Len; ++NewStart, ++PrefixLen) {
      if (*NewStart != C)
        break;
    }

    size_t SuffixLen = 0;
    const char *NewEnd = Data + Len - 1;
    for (; SuffixLen < Len; --NewEnd, ++SuffixLen) {
      if (*NewEnd != C)
        break;
    }

    return remove_prefix(PrefixLen).remove_suffix(SuffixLen);
  }

  // An equivalent method is not available in std::string_view.
  bool equals(StringView Other) const {
    if (Len != Other.Len)
      return false;
    for (size_t I = 0; I < Len; ++I) {
      if (Data[I] != Other.Data[I])
        return false;
    }
    return true;
  }
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_CPP_STRINGVIEW_H
