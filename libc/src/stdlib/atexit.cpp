//===-- Implementation of atexit ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/atexit.h"
#include "hdr/types/atexithandler_t.h"
#include "src/__support/common.h"
#include "src/stdlib/exit_handler.h"

namespace LIBC_NAMESPACE {

extern "C" {

int __cxa_atexit(AtExitCallback *callback, void *payload, void *) {
  return add_atexit_unit(atexit_callbacks, {callback, payload});
}

void __cxa_finalize(void *dso) {
  if (!dso)
    call_exit_callbacks(atexit_callbacks);
}

} // extern "C"

LLVM_LIBC_FUNCTION(int, atexit, (__atexithandler_t callback)) {
  return add_atexit_unit(
      atexit_callbacks,
      {&stdc_at_exit_func, reinterpret_cast<void *>(callback)});
}

} // namespace LIBC_NAMESPACE
