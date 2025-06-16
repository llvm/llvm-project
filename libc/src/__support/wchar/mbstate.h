//===-- Definition of mbstate-----------------------------------*-- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MBSTATE_H
#define LLVM_LIBC_SRC___SUPPORT_MBSTATE_H

#include "hdr/types/char32_t.h"
#include "src/__support/common.h"
#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace internal {

struct mbstate {
  // store a partial codepoint (in UTF-32)
  char32_t partial;

  /*
  Progress towards a conversion
    For utf8  -> utf32, increases with each CharacterConverter::push(utf8_byte)
    For utf32 ->  utf8, increases with each CharacterConverter::pop_utf8()
  */
  uint8_t bytes_processed;

  // Total number of bytes that will be needed to represent this character
  uint8_t total_bytes;
};

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MBSTATE_H
