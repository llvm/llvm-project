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
  char32_t partial;
  uint8_t bytes_processed;
  uint8_t total_bytes;
};

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MBSTATE_H
