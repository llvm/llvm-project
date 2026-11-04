//===-- Implementation header for IRELATIVE relocations -------- *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_STARTUP_LINUX_IRELATIVE_H
#define LLVM_LIBC_STARTUP_LINUX_IRELATIVE_H

#include "hdr/link_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/macros/config.h"

extern "C" {
[[gnu::weak, gnu::visibility("hidden")]] extern const ElfW(Rela)
    __rela_iplt_start[]; // NOLINT
[[gnu::weak, gnu::visibility("hidden")]] extern const ElfW(Rela)
    __rela_iplt_end[]; // NOLINT
}

namespace LIBC_NAMESPACE_DECL {

// Process IRELATIVE relocations (ifunc resolvers).
// base is the load bias (actual load address − link-time address).  It is
// intptr_t (signed) because it is a difference; it is negative if the binary
// loaded below its link address. (unlikely but possible in principle)
void apply_irelative_relocs(intptr_t base, unsigned long hwcap,
                            unsigned long hwcap2);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_STARTUP_LINUX_IRELATIVE_H
