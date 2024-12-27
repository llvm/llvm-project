//===-- Implementation of dl_iterate_phdr ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/link/dl_iterate_phdr.h"
#include "config/linux/app.h"
#include "include/llvm-libc-macros/sys-auxv-macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

[[gnu::weak,
  gnu::visibility("hidden")]] extern const ElfW(Dyn) _DYNAMIC[]; // NOLINT

#define AUX_CNT 38

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, dl_iterate_phdr,
                   (__dl_iterate_phdr_callback_t callback, void *data)) {
  // dl_iterate_phdr implementation based on Musl source
  // "src/ldso/dl_iterate_phdr.c"
  size_t *auxv_ptr = reinterpret_cast<size_t *>(app.auxv_ptr);
  size_t aux[AUX_CNT] = {0};

  for (size_t i = 0; auxv_ptr[i]; i += 2) {
    if (auxv_ptr[i] < AUX_CNT) {
      aux[auxv_ptr[i]] = auxv_ptr[i + 1];
    }
  }

  void *p;
  size_t n;
  size_t base = 0;
  for (p = (void *)aux[AT_PHDR], n = aux[AT_PHNUM]; n;
       n--, p = reinterpret_cast<void *>((uintptr_t)p + aux[AT_PHENT])) {
    ElfW(Phdr) *phdr = (ElfW(Phdr) *)p;
    if (phdr->p_type == PT_PHDR)
      base = aux[AT_PHDR] - phdr->p_vaddr;
    if (phdr->p_type == PT_DYNAMIC && _DYNAMIC)
      base = (size_t)_DYNAMIC - phdr->p_vaddr;
  }

  struct dl_phdr_info info;
  info.dlpi_addr = base;
  info.dlpi_name = "/proc/self/exe";
  info.dlpi_phdr = (const ElfW(Phdr) *)aux[AT_PHDR];
  info.dlpi_phnum = (ElfW(Half))aux[AT_PHNUM];
  info.dlpi_adds = 0;
  info.dlpi_subs = 0;
  info.dlpi_tls_modid = 0;
  info.dlpi_tls_data = 0;
  return callback(&info, sizeof(struct dl_phdr_info), data);
}

} // namespace LIBC_NAMESPACE_DECL
