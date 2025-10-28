//===-- Implementation of dl_iterate_phdr --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "dl_iterate_phdr.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, dl_iterate_phdr,
                   (__dl_iterate_phdr_callback_t callback, void *arg)) {
  // FIXME: For pure static linking, this can report just the executable with
  // info from __ehdr_start or AT_{PHDR,PHNUM} decoding, and its PT_TLS; and it
  // could report the vDSO.
  (void)callback, (void)arg;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
