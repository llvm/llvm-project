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

namespace LIBC_NAMESPACE_DECL {

extern "C" void __cxa_finalize(void *);

// exit() needs to clean up TLS and call associated destructors.
//
// The weak no-op implementation is in the same TU with its caller for the case
// where the real definition is not linked in. This is preferable over weak
// undefined symbol with a check since that requires a GOT slot.
//
// The strong implementation is in libc/src/__support/threads/thread.cpp
// but not every platform supports threads (e.g. baremetal) in which case the
// the no-op implementation is sufficient.
//
// TODO: Strictly speaking, it is not valid to call exit in overlay mode
//       as we have no way to ensure system libc will call the TLS destructors.
//       We should run exit related tests in hermetic mode but this is currently
//       blocked by https://github.com/llvm/llvm-project/issues/133925.
extern "C" [[gnu::weak]] void __cxa_thread_finalize() {}

// TODO: use recursive mutex to protect this routine.
[[noreturn]] LLVM_LIBC_FUNCTION(void, exit, (int status)) {
  __cxa_thread_finalize();
  __cxa_finalize(nullptr);
  internal::exit(status);
}

} // namespace LIBC_NAMESPACE_DECL
