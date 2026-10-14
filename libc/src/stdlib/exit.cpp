//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of exit.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/exit.h"
#include "src/__support/OSUtil/exit.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#ifdef LIBC_COPT_SUPPORT_THREADS
#include "src/__support/threads/thread.h"
#endif

#ifdef LIBC_COPT_EXIT_FLUSH_STREAMS
#include "src/stdio/fflush_internal.h"
#endif

namespace LIBC_NAMESPACE_DECL {

// Defined with the rest of the exit handlers. Visibility is declared here as
// well as there: the parts of the library are merged into one object before
// it becomes a shared one, and the narrower of the two would otherwise win
// and keep it in.
extern "C" LIBC_SHARED_INTERNAL void __cxa_finalize(void *);

// TODO: use recursive mutex to protect this routine.
[[noreturn]] LLVM_LIBC_FUNCTION(void, exit, (int status)) {
#ifdef LIBC_COPT_SUPPORT_THREADS
  // Call TLS destructors, if supported by the target.
  internal::call_atexit_callbacks();
#endif
  __cxa_finalize(nullptr);
#ifdef LIBC_COPT_EXIT_FLUSH_STREAMS
  // C23 7.24.4.4: streams are flushed after the atexit handlers have run.
  internal::flush_all_streams();
#endif
  internal::exit(status);
}

} // namespace LIBC_NAMESPACE_DECL
