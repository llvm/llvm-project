//===-- Implementation of dl_iterate_phdr --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "dl_iterate_phdr.h"

#include "llvm-libc-macros/link-macros.h"
#include "src/__support/OSUtil/linux/auxv.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include <elf.h>

extern "C" void *__executable_start;

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, dl_iterate_phdr,
                   (__dl_iterate_phdr_callback_t callback, void *arg)) {
  ElfW(Ehdr) *executable_header =
      reinterpret_cast<ElfW(Ehdr) *>(&__executable_start);
  struct dl_phdr_info executable_info;
  executable_info.dlpi_addr = 0;
  executable_info.dlpi_name = nullptr;
  executable_info.dlpi_phdr = reinterpret_cast<ElfW(Phdr) *>(
      reinterpret_cast<uintptr_t>(executable_header) +
      executable_header->e_phoff);
  executable_info.dlpi_phnum = executable_header->e_phnum;
  executable_info.dlpi_adds = 0;
  executable_info.dlpi_subs = 0;
  executable_info.dlpi_tls_modid = 0;
  executable_info.dlpi_tls_data = nullptr;
  int executable_return_code =
      callback(&executable_info, sizeof(executable_info), arg);
  if (executable_return_code != 0)
    return executable_return_code;

  cpp::optional<unsigned long> vdso_start_address =
      LIBC_NAMESPACE::auxv::get(AT_SYSINFO_EHDR);
  if (!vdso_start_address)
    return 0;
  ElfW(Ehdr) *vdso_header = reinterpret_cast<ElfW(Ehdr) *>(*vdso_start_address);
  if (vdso_header == nullptr)
    return 0;
  struct dl_phdr_info vdso_info;
  vdso_info.dlpi_addr = 0;
  vdso_info.dlpi_phdr = reinterpret_cast<ElfW(Phdr) *>(
      reinterpret_cast<char *>(vdso_header) + vdso_header->e_phoff);
  vdso_info.dlpi_phnum = vdso_header->e_phnum;
  vdso_info.dlpi_adds = 0;
  vdso_info.dlpi_subs = 0;
  vdso_info.dlpi_tls_modid = 0;
  vdso_info.dlpi_tls_data = nullptr;
  for (size_t i = 0; i < vdso_info.dlpi_phnum; ++i) {
    if (vdso_info.dlpi_phdr[i].p_type == PT_LOAD) {
      vdso_info.dlpi_addr =
          (ElfW(Addr))vdso_header - vdso_info.dlpi_phdr[i].p_vaddr;
      break;
    }
  }
  return callback(&vdso_info, sizeof(vdso_info), arg);
}

} // namespace LIBC_NAMESPACE_DECL
