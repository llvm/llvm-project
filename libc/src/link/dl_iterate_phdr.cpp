//===-- Implementation of dl_iterate_phdr --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "dl_iterate_phdr.h"

#include "hdr/sys_auxv_macros.h"
#include "src/__support/OSUtil/linux/auxv.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include <elf.h>

extern "C" {
[[gnu::weak]] extern const char __ehdr_start;
[[gnu::weak]] extern const char __executable_start;
}

namespace LIBC_NAMESPACE_DECL {

namespace {

LIBC_INLINE bool is_valid_elf(const ElfW(Ehdr) *ehdr) {
  if (ehdr == nullptr)
    return false;

  return ehdr->e_ident[EI_MAG0] == ELFMAG0 &&
         ehdr->e_ident[EI_MAG1] == ELFMAG1 &&
         ehdr->e_ident[EI_MAG2] == ELFMAG2 &&
         ehdr->e_ident[EI_MAG3] == ELFMAG3;
}

LIBC_INLINE const ElfW(Ehdr) *get_executable_ehdr() {
  const ElfW(Ehdr) *ehdr = nullptr;

  if (&__ehdr_start != nullptr)
    ehdr = reinterpret_cast<const ElfW(Ehdr) *>(&__ehdr_start);
  else if (&__executable_start != nullptr)
    ehdr = reinterpret_cast<const ElfW(Ehdr) *>(&__executable_start);

  return is_valid_elf(ehdr) ? ehdr : nullptr;
}

LIBC_INLINE ElfW(Addr) get_executable_load_bias(const ElfW(Ehdr) *ehdr,
                                                const ElfW(Phdr) *phdr_table,
                                                ElfW(Half) phnum) {
  if (phdr_table != nullptr) {
    const uintptr_t runtime_phdr = reinterpret_cast<uintptr_t>(phdr_table);
    for (ElfW(Half) i = 0; i < phnum; ++i) {
      if (phdr_table[i].p_type == PT_PHDR)
        return static_cast<ElfW(Addr)>(runtime_phdr - phdr_table[i].p_vaddr);
    }
  }

  // For PIE binaries the ELF header is mapped at the load bias. ET_EXEC uses
  // absolute virtual addresses, so report zero there.
  if (ehdr != nullptr && ehdr->e_type == ET_DYN)
    return reinterpret_cast<ElfW(Addr)>(ehdr);

  return 0;
}

LIBC_INLINE int call_with_executable(__dl_iterate_phdr_callback_t callback,
                                     void *arg) {
  const ElfW(Ehdr) *ehdr = get_executable_ehdr();
  if (ehdr == nullptr)
    return -1;

  cpp::optional<unsigned long> aux_phdr_val = auxv::get(AT_PHDR);
  cpp::optional<unsigned long> aux_phnum_val = auxv::get(AT_PHNUM);
  auto *aux_phdr = reinterpret_cast<const ElfW(Phdr) *>(
      aux_phdr_val ? *aux_phdr_val : 0);
  const auto aux_phnum = static_cast<ElfW(Half)>(aux_phnum_val ? *aux_phnum_val
                                                               : 0);

  const ElfW(Phdr) *phdr =
      aux_phdr != nullptr ? aux_phdr
                          : reinterpret_cast<const ElfW(Phdr) *>(
                                reinterpret_cast<uintptr_t>(ehdr) + ehdr->e_phoff);
  const ElfW(Half) phnum = aux_phnum != 0 ? aux_phnum : ehdr->e_phnum;

  dl_phdr_info exe_info = {};
  exe_info.dlpi_addr = get_executable_load_bias(ehdr, phdr, phnum);
  exe_info.dlpi_name = nullptr;
  exe_info.dlpi_phdr = phdr;
  exe_info.dlpi_phnum = phnum;
  exe_info.dlpi_adds = 0;
  exe_info.dlpi_subs = 0;
  exe_info.dlpi_tls_modid = 0;
  exe_info.dlpi_tls_data = nullptr;
  return callback(&exe_info, sizeof(exe_info), arg);
}

LIBC_INLINE int call_with_vdso(__dl_iterate_phdr_callback_t callback,
                               void *arg) {
  cpp::optional<unsigned long> aux_vdso = auxv::get(AT_SYSINFO_EHDR);
  auto *ehdr_vdso =
      reinterpret_cast<const ElfW(Ehdr) *>(aux_vdso ? *aux_vdso : 0);
  if (!is_valid_elf(ehdr_vdso))
    return 0;

  auto *phdr = reinterpret_cast<const ElfW(Phdr) *>(
      reinterpret_cast<uintptr_t>(ehdr_vdso) + ehdr_vdso->e_phoff);

  dl_phdr_info vdso_info = {};
  vdso_info.dlpi_name = nullptr;
  vdso_info.dlpi_phdr = phdr;
  vdso_info.dlpi_phnum = ehdr_vdso->e_phnum;
  vdso_info.dlpi_adds = 0;
  vdso_info.dlpi_subs = 0;
  vdso_info.dlpi_tls_modid = 0;
  vdso_info.dlpi_tls_data = nullptr;

  for (ElfW(Half) i = 0; i < ehdr_vdso->e_phnum; ++i) {
    if (phdr[i].p_type == PT_LOAD) {
      vdso_info.dlpi_addr = reinterpret_cast<ElfW(Addr)>(ehdr_vdso) -
                            static_cast<ElfW(Addr)>(phdr[i].p_vaddr);
      break;
    }
  }

  return callback(&vdso_info, sizeof(vdso_info), arg);
}

} // namespace

LLVM_LIBC_FUNCTION(int, dl_iterate_phdr,
                   (__dl_iterate_phdr_callback_t callback, void *arg)) {
#if defined(LIBC_TARGET_ARCH_IS_X86_64) || defined(LIBC_TARGET_ARCH_IS_AARCH64)
  if (callback == nullptr)
    return -1;

  int rc = call_with_executable(callback, arg);
  if (rc != 0)
    return rc;
  return call_with_vdso(callback, arg);
#else
  (void)callback;
  (void)arg;
  return 0;
#endif
}

} // namespace LIBC_NAMESPACE_DECL
