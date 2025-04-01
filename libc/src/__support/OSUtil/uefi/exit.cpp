//===-------- UEFI implementation of an exit function ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "src/__support/OSUtil/exit.h"
#include "include/Uefi.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

[[noreturn]] void exit(int status) {
  efi_system_table->BootServices->Exit(efi_image_handle, status, 0, nullptr);
  __builtin_unreachable();
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
