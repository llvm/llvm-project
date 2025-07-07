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
  size_t num_pushed;
  size_t num_to_write;

  int pushFullCharacter() {
    for (num_pushed = 0; !cr.isFull() && src_idx + num_pushed < src_len;
         ++num_pushed) {
      int err = cr.push(src[src_idx + num_pushed]);
      if (err != 0)
        return err;
    }

    // if we aren't able to read a full character from the source string
    if (src_idx + num_pushed == src_len && !cr.isFull()) {
      src_idx += num_pushed;
      return -1;
    }

    return 0;
  }

public:
  StringConverter(const T *s, size_t srclen, size_t dstlen, mbstate *ps)
      : cr(ps), src(s), src_len(srclen), src_idx(0), num_pushed(0),
        num_to_write(dstlen) {
    pushFullCharacter();
  }

  StringConverter(const T *s, size_t dstlen, mbstate *ps)
      : StringConverter(s, SIZE_MAX, dstlen, ps) {}

  ErrorOr<char32_t> popUTF32() {
    if (cr.isEmpty()) {
      int err = pushFullCharacter();
      if (err != 0)
        return Error(err);

      if (cr.sizeAsUTF32() > num_to_write) {
        cr.clear();
        return Error(-1);
      }
    }

    auto out = cr.pop_utf32();
    if (cr.isEmpty())
      src_idx += num_pushed;

    if (out.has_value() && out.value() == L'\0')
      src_len = src_idx;

    num_to_write--;

    return out;
  }

  ErrorOr<char8_t> popUTF8() {
    if (cr.isEmpty()) {
      int err = pushFullCharacter();
      if (err != 0)
        return Error(err);

      if (cr.sizeAsUTF8() > num_to_write) {
        cr.clear();
        return Error(-1);
      }
    }

    auto out = cr.pop_utf8();
    if (cr.isEmpty())
      src_idx += num_pushed;

    if (out.has_value() && out.value() == '\0')
      src_len = src_idx;

    num_to_write--;

    return out;
  }

  size_t getSourceIndex() { return src_idx; }
};

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_STRING_CONVERTER_H
