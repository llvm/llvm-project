//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of abort_utils for UEFI.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_UEFI_ABORT_UTILS_H
#define LLVM_LIBC_SRC_STDLIB_UEFI_ABORT_UTILS_H

#include "config/app.h"
#include "include/llvm-libc-types/EFI_STATUS.h"
#include "include/llvm-libc-types/EFI_SYSTEM_TABLE.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

namespace abort_utils {
[[noreturn]] LIBC_INLINE void abort() {
  // Hand control back to the firmware with an error status so that the
  // loader reports the failure. Boot services are gone after
  // ExitBootServices(), in which case all that is left is to trap.
  if (app.system_table != nullptr && app.system_table->BootServices != nullptr)
    app.system_table->BootServices->Exit(app.image_handle, EFI_ABORTED, 0,
                                         nullptr);
  __builtin_trap();
}
} // namespace abort_utils

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_UEFI_ABORT_UTILS_H
