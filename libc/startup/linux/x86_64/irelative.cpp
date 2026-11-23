//===-- Implementation of apply_irelative_relocs (x86_64) -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "startup/linux/irelative.h"
#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

void apply_irelative_relocs(intptr_t base, unsigned long /*hwcap*/,
                            unsigned long /*hwcap2*/) {
  for (const ElfW(Rela) *rela = __rela_iplt_start; rela != __rela_iplt_end;
       ++rela) {
    if (ELF64_R_TYPE(rela->r_info) != R_X86_64_IRELATIVE)
      continue;

    // x86_64 resolvers take no arguments.
    // Use unsigned arithmetic to avoid undefined behavior on signed overflow,
    // which can occur with very large binaries or high load addresses.
    uintptr_t resolver_addr =
        static_cast<uintptr_t>(base) + static_cast<uintptr_t>(rela->r_addend);
    auto resolver = reinterpret_cast<uintptr_t (*)(void)>(resolver_addr);
    uintptr_t result = resolver();

    uintptr_t target_addr = static_cast<uintptr_t>(base) + rela->r_offset;
    *reinterpret_cast<uintptr_t *>(target_addr) = result;
  }
}

} // namespace LIBC_NAMESPACE_DECL
