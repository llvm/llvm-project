//===-- Implementation of dl_iterate_phdr ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dl_iterate_phdr.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

// TODO(@izaakschroeder): https://github.com/llvm/llvm-project/issues/97928
LLVM_LIBC_FUNCTION(int, dl_iterate_phdr,
                   (__dl_iterate_phdr_callback_t, void *)) {
  return -1;
}

} // namespace LIBC_NAMESPACE
