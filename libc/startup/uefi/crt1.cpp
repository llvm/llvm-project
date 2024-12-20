//===-- Implementation of crt for UEFI ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

#include "include/llvm-libc-macros/stdlib-macros.h"
#include "include/llvm-libc-types/EFI_HANDLE.h"
#include "include/llvm-libc-types/EFI_STATUS.h"
#include "include/llvm-libc-types/EFI_SYSTEM_TABLE.h"
#include "src/__support/macros/config.h"

extern "C" int main(int argc, char **argv, char **envp);

extern "C" EFI_STATUS EfiMain(EFI_HANDLE ImageHandle,
                              EFI_SYSTEM_TABLE *SystemTable) {
  (void)ImageHandle;
  (void)SystemTable;
  main(0, NULL, NULL);
  // TODO: convert the return value of main to EFI_STATUS
  return 0; // TODO: EFI_SUCCESS
}
