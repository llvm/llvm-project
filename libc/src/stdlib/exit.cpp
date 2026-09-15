//===-- Implementation of exit --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/exit.h"
#include "src/__support/OSUtil/exit.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#ifdef LIBC_COPT_SUPPORT_THREADS
#include "src/__support/threads/thread.h"
#endif

namespace LIBC_NAMESPACE_DECL {

extern "C" void __cxa_finalize(void *);

// TODO: use recursive mutex to protect this routine.
[[noreturn]] LLVM_LIBC_FUNCTION(void, exit, (int status)) {
#ifdef LIBC_COPT_SUPPORT_THREADS
  // Call TLS destructors, if supported by the target.
  internal::call_atexit_callbacks(current_thread().attrib);
#endif
  __cxa_finalize(nullptr);
  internal::exit(status);
}

} // namespace LIBC_NAMESPACE_DECL
