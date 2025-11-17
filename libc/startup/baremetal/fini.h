//===-- Implementation header of __libc_fini_array ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/config.h"

// NOTE: The namespace is necessary here to set the correct symbol visibility.
namespace LIBC_NAMESPACE_DECL {

extern "C" {
extern uintptr_t __fini_array_start[];
extern uintptr_t __fini_array_end[];

void __libc_fini_array(void);
} // extern "C"

} // namespace LIBC_NAMESPACE_DECL
