//===-- Implementation of dlerror -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dlerror.h"

#include "src/__support/CPP/mutex.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/dlfcn/dl_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, dlerror, ()) {
  cpp::lock_guard lock(dl::dl_mutex);
  // Reading the error clears it, so a second call with nothing in between
  // reports no error rather than repeating the last one.
  const char *message = dl::dl_last_error;
  dl::dl_last_error = nullptr;
  return const_cast<char *>(message);
}

} // namespace LIBC_NAMESPACE_DECL
