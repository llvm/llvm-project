//===-- Definition of a class for mbstate_t and conversion -----*-- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_STRING_CONVERTER_H
#define LLVM_LIBC_SRC___SUPPORT_STRING_CONVERTER_H

#include "hdr/types/char32_t.h"
#include "hdr/types/char8_t.h"
#include "hdr/types/size_t.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/wchar/character_converter.h"
#include "src/__support/wchar/mbstate.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

template <typename T> class StringConverter {
private:
  CharacterConverter cr;
  const T *src;
  size_t src_len;
  size_t src_idx;

  int pushFullCharacter() {
    if (!cr.isEmpty())
      return 0;

    int original_idx = src_idx;
    while (!cr.isFull() && src_idx < src_len) {
      int err = cr.push(src[src_idx++]);
      if (err != 0) {
        // point to the beginning of the invalid sequence
        src_idx = original_idx;
        return err;
      }
    }

    if (src_idx == src_len && !cr.isFull()) {
      // src points to the beginning of the character
      src_idx = original_idx;
      return -1;
    }

    return 0;
  }

public:
  StringConverter(const T *s, mbstate *ps)
      : cr(ps), src(s), src_len(SIZE_MAX), src_idx(0) {}
  StringConverter(const T *s, size_t len, mbstate *ps)
      : cr(ps), src(s), src_len(len), src_idx(0) {}

  ErrorOr<char32_t> popUTF32() {
    int err = pushFullCharacter();
    if (err != 0)
      return Error(err);

    auto out = cr.pop_utf32();
    if (out.has_value() && out.value() == L'\0')
      src_len = src_idx;
    
    return out;
  }

  ErrorOr<char8_t> popUTF8() {
    int err = pushFullCharacter();
    if (err != 0)
      return Error(err);

    auto out = cr.pop_utf8();
    if (out.has_value() && out.value() == '\0')
      src_len = src_idx;
    
    return out;
  }
};

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_STRING_CONVERTER_H
