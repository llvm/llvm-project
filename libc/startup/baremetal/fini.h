//===-- Implementation header of __libc_fini_array ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"

extern "C" {
extern uintptr_t __fini_array_start[];
extern uintptr_t __fini_array_end[];

void __libc_fini_array(void);
} // extern "C"
