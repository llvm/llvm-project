//===-- Implementation of crt for UEFI ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "config/app.h"
#include "include/llvm-libc-types/EFI_STATUS.h"
#include "src/__support/OSUtil/uefi/error.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
AppProperties app;
}

extern "C" {
EFI_HANDLE __llvm_libc_efi_image_handle;
EFI_SYSTEM_TABLE *__llvm_libc_efi_system_table;

int main(int argc, char **argv, char **envp);

EFI_STATUS EfiMain(EFI_HANDLE ImageHandle, EFI_SYSTEM_TABLE *SystemTable) {
  LIBC_NAMESPACE::app.image_handle = ImageHandle;
  LIBC_NAMESPACE::app.system_table = SystemTable;

  // TODO: we need the EFI_SHELL_PROTOCOL, malloc, free, and UTF16 -> UTF8
  // conversion.
  return LIBC_NAMESPACE::errno_to_uefi_status(main(0, nullptr, nullptr));
}
}
