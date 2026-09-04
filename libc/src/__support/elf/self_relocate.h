//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Relocating the loader itself
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_SELF_RELOCATE_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_SELF_RELOCATE_H

#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "src/__support/elf/relocation.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// Applies the loader's own relative relocations.
//
// This is the first thing that runs, before anything in the loader has a
// usable address. It must therefore not call through the GOT or the PLT, read
// a global, or touch thread local storage: none of those work until it has
// finished. Everything it needs is passed in, and it is marked always inline
// so there is not even a call to make.
//
// `dynamic` and `load_bias` come from the entry stub, which can compute both
// with pc-relative addressing before any relocation has been applied.
[[gnu::always_inline]] inline void self_relocate(const ElfW(Dyn) * dynamic,
                                                 ElfW(Addr) load_bias) {
  const ElfW(Rela) *rela = nullptr;
  size_t rela_size = 0;

  for (const ElfW(Dyn) *entry = dynamic; entry->d_tag != DT_NULL; ++entry) {
    switch (entry->d_tag) {
    case DT_RELA:
      rela =
          reinterpret_cast<const ElfW(Rela) *>(entry->d_un.d_ptr + load_bias);
      break;
    case DT_RELASZ:
      rela_size = entry->d_un.d_val;
      break;
    default:
      break;
    }
  }

  if (rela == nullptr || rela_size == 0)
    return;

  // Written out rather than reusing RelaTable so that nothing here depends on
  // a member function being callable before relocation.
  const size_t count = rela_size / sizeof(ElfW(Rela));
  for (size_t i = 0; i < count; ++i) {
    if (reloc_type(rela[i].r_info) != RELATIVE_RELOC)
      continue;
    auto *target = reinterpret_cast<ElfW(Addr) *>(load_bias + rela[i].r_offset);
    *target = load_bias + static_cast<ElfW(Addr)>(rela[i].r_addend);
  }
}

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_SELF_RELOCATE_H
