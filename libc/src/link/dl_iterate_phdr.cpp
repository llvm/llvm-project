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
#include "src/__support/CPP/mutex.h"
#include "src/__support/OSUtil/linux/auxv.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include <elf.h>

extern "C" void *__ehdr_start;

namespace LIBC_NAMESPACE_DECL {

// The name the kernel gives the vDSO in the link map.
constexpr const char *VDSO_NAME = "linux-vdso.so.1";

// The load bias is the distance between where the first PT_LOAD segment was
// linked and where it actually ended up. It is zero for a non-PIE executable
// and non-zero for a PIE one or for the vDSO.
static ElfW(Addr) load_bias(ElfW(Ehdr) * header, const ElfW(Phdr) * phdrs) {
  for (ElfW(Half) i = 0; i < header->e_phnum; ++i)
    if (phdrs[i].p_type == PT_LOAD)
      return reinterpret_cast<ElfW(Addr)>(header) - phdrs[i].p_vaddr;
  return 0;
}

struct dl_phdr_info create_module_info(ElfW(Ehdr) * header, const char *name) {
  struct dl_phdr_info to_return;
  to_return.dlpi_phdr = reinterpret_cast<ElfW(Phdr) *>(
      reinterpret_cast<uintptr_t>(header) + header->e_phoff);
  to_return.dlpi_phnum = header->e_phnum;
  to_return.dlpi_addr = load_bias(header, to_return.dlpi_phdr);
  to_return.dlpi_name = name;
  to_return.dlpi_adds = 0;
  to_return.dlpi_subs = 0;
  to_return.dlpi_tls_modid = 0;
  to_return.dlpi_tls_data = nullptr;
  return to_return;
}

LLVM_LIBC_FUNCTION(int, dl_iterate_phdr,
                   (__dl_iterate_phdr_callback_t callback, void *arg)) {
  ElfW(Ehdr) *executable_header = reinterpret_cast<ElfW(Ehdr) *>(&__ehdr_start);
  // The main executable is always reported with an empty name.
  struct dl_phdr_info executable_info =
      create_module_info(executable_header, "");
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
  struct dl_phdr_info vdso_info = create_module_info(vdso_header, VDSO_NAME);
  return callback(&vdso_info, sizeof(vdso_info), arg);
}

} // namespace LIBC_NAMESPACE_DECL
