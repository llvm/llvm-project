//===-- Definition of mbstate_t -------------------------- -----*-- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MBSTATE_H
#define LLVM_LIBC_SRC___SUPPORT_MBSTATE_H

#include "hdr/types/wchar_t.h"

namespace LIBC_NAMESPACE_DECL {

struct mbstate_t {
  wchar_t partial;
  unsigned char bits_processed;
  unsigned char total_bytes;
}; 

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MBSTATE_H

