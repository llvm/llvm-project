//===-- Implementation of at_quick_exit -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/at_quick_exit.h"
#include "hdr/types/atexithandler_t.h"
#include "src/__support/common.h"
#include "src/stdlib/exit_handler.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, at_quick_exit, (__atexithandler_t callback)) {
  return add_atexit_unit(
      at_quick_exit_callbacks,
      {&stdc_at_exit_func, reinterpret_cast<void *>(callback)});
}

} // namespace LIBC_NAMESPACE
