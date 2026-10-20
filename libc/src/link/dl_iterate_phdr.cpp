//===-- Implementation of dl_iterate_phdr --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
///
/// \file
/// The dl_iterate_phdr implementation.
///
//===----------------------------------------------------------------------===/

#include "dl_iterate_phdr.h"

#include "llvm-libc-macros/link-macros.h"
#include "src/__support/CPP/span.h"
#include "src/__support/OSUtil/linux/auxv.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include <elf.h>

extern "C" void *__ehdr_start;

namespace LIBC_NAMESPACE_DECL {

struct dl_phdr_info create_executable_info(ElfW(Ehdr) * executable_header) {
  // TODO: Calculate dlpi_addr in the PIE case and set dlpi_name for VDSO.
  struct dl_phdr_info to_return;
  to_return.dlpi_addr = 0;
  to_return.dlpi_name = "";
  to_return.dlpi_phdr = reinterpret_cast<ElfW(Phdr) *>(
      reinterpret_cast<uintptr_t>(executable_header) +
      executable_header->e_phoff);
  to_return.dlpi_phnum = executable_header->e_phnum;
  to_return.dlpi_adds = 0;
  to_return.dlpi_subs = 0;
  to_return.dlpi_tls_modid = 0;
  to_return.dlpi_tls_data = nullptr;
  return to_return;
}

LLVM_LIBC_FUNCTION(int, dl_iterate_phdr,
                   (__dl_iterate_phdr_callback_t callback, void *arg)) {
  ElfW(Ehdr) *executable_header = reinterpret_cast<ElfW(Ehdr) *>(&__ehdr_start);
  struct dl_phdr_info executable_info =
      create_executable_info(executable_header);
  int executable_return_code =
      callback(&executable_info, sizeof(executable_info), arg);
  if (executable_return_code != 0)
    return executable_return_code;

  cpp::optional<unsigned long> vdso_start_address = auxv::get(AT_SYSINFO_EHDR);
  if (!vdso_start_address)
    return 0;
  ElfW(Ehdr) *vdso_header = reinterpret_cast<ElfW(Ehdr) *>(*vdso_start_address);
  if (vdso_header == nullptr)
    return 0;
  struct dl_phdr_info vdso_info = create_executable_info(vdso_header);
  for (auto elf_headers :
       cpp::span<const ElfW(Phdr)>(vdso_info.dlpi_phdr, vdso_info.dlpi_phnum)) {
    if (elf_headers.p_type == PT_LOAD) {
      vdso_info.dlpi_addr =
          reinterpret_cast<ElfW(Addr)>(vdso_header) - elf_headers.p_vaddr;
      break;
    }
  }
  return callback(&vdso_info, sizeof(vdso_info), arg);
}

} // namespace LIBC_NAMESPACE_DECL
