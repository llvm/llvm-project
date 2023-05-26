//===------------- Fuchsia implementation of IO utils -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_OSUTIL_FUCHSIA_IO_H
#define LLVM_LIBC_SRC_SUPPORT_OSUTIL_FUCHSIA_IO_H

#ifndef LIBC_COPT_TEST_USE_FUCHSIA
#error this file should only be used by tests
#endif

#include "src/__support/CPP/string_view.h"

#include <zircon/sanitizer.h>

namespace __llvm_libc {

LIBC_INLINE void write_to_stderr(cpp::string_view msg) {
  __sanitizer_log_write(msg.data(), msg.size());
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_OSUTIL_FUCHSIA_IO_H
