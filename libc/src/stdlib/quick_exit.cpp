//===-- Implementation of quick_exit --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/quick_exit.h"
#include "src/__support/OSUtil/exit.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/exit_handler.h"

// extern "C" void __cxa_finalize(void *);
namespace LIBC_NAMESPACE_DECL {

extern ExitCallbackList at_quick_exit_callbacks;
[[gnu::weak]] extern void teardown_main_tls();

[[noreturn]] LLVM_LIBC_FUNCTION(void, quick_exit, (int status)) {
  call_exit_callbacks(at_quick_exit_callbacks);
  if (teardown_main_tls)
    teardown_main_tls();
  internal::exit(status);
}

} // namespace LIBC_NAMESPACE_DECL
