//===-- Definition of a class for mbstate_t and conversion -----*-- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CHARACTER_CONVERTER_H
#define LLVM_LIBC_SRC___SUPPORT_CHARACTER_CONVERTER_H

#include "hdr/types/char32_t.h"
#include "hdr/types/char8_t.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/wchar/mbstate.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

class CharacterConverter {
private:
  mbstate *state;

public:
  CharacterConverter(mbstate *mbstate);

  void clear();
  bool isFull();
  bool isEmpty();
  bool isValidState();

  template <typename CharType> size_t sizeAs();
  template <> size_t sizeAs<char8_t>() { return state->total_bytes; }
  template <> size_t sizeAs<char32_t>() { return 1; }

  int push(char8_t utf8_byte);
  int push(char32_t utf32);

  ErrorOr<char8_t> pop_utf8();
  ErrorOr<char32_t> pop_utf32();
  template <typename CharType> ErrorOr<CharType> pop();
  template <> ErrorOr<char8_t> pop() { return pop_utf8(); }
  template <> ErrorOr<char32_t> pop() { return pop_utf32(); }
};

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_CHARACTER_CONVERTER_H
