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

namespace LIBC_NAMESPACE_DECL {

using InitCallback = void(void);
using FiniCallback = void(void);
extern "C" InitCallback *__CTOR_LIST__[];
extern "C" FiniCallback *__DTOR_LIST__[];

static void call_init_array_callbacks() {
  unsigned long nptrs = (unsigned long)__CTOR_LIST__[0];
  unsigned long i;

  if (nptrs == ~0ul) {
    for (nptrs = 0; __CTOR_LIST__[nptrs + 1] != 0; nptrs++)
      ;
  }

  for (i = nptrs; i >= 1; i--) {
    __CTOR_LIST__[i]();
  }
}

static void call_fini_array_callbacks() {
  unsigned long nptrs = 0;

  for (nptrs = 0; __DTOR_LIST__[nptrs + 1] != 0; nptrs++)
    ;

  for (unsigned long i = nptrs; i >= 1; i--) {
    __DTOR_LIST__[i]();
  }
}
} // namespace LIBC_NAMESPACE_DECL

EFI_HANDLE efi_image_handle;
EFI_SYSTEM_TABLE *efi_system_table;

extern "C" int main(int argc, char **argv, char **envp);

extern "C" EFI_STATUS EfiMain(EFI_HANDLE ImageHandle,
                              EFI_SYSTEM_TABLE *SystemTable) {
  efi_image_handle = ImageHandle;
  efi_system_table = SystemTable;

  LIBC_NAMESPACE::call_init_array_callbacks();

  main(0, NULL, NULL);

  LIBC_NAMESPACE::call_fini_array_callbacks();
  // TODO: convert the return value of main to EFI_STATUS
  return 0; // TODO: EFI_SUCCESS
}
