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
#include "src/__support/elf/passive_abi.h"
#include "src/__support/macros/config.h"
#include "src/dlfcn/dl_internal.h"

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

// What a module's own headers say about it, for one the loader recorded. It
// already knows where the headers are and where the module was put, so
// nothing has to be worked out again.
static struct dl_phdr_info info_for(const elf::Module &module,
                                    const elf::ModuleSet &set) {
  struct dl_phdr_info to_return;
  to_return.dlpi_phdr = module.phdrs();
  to_return.dlpi_phnum = module.phnum();
  to_return.dlpi_addr = module.load_bias();
  // The path it was loaded from, as glibc and musl report it. A caller such
  // as a JVM finds its own library's directory from this.
  const char *soname = module.soname();
  to_return.dlpi_name = module.name() != nullptr ? module.name() : soname;
  to_return.dlpi_adds = set.generation;
  to_return.dlpi_subs = 0;
  to_return.dlpi_tls_modid = 0;
  to_return.dlpi_tls_data = nullptr;
  return to_return;
}

LLVM_LIBC_FUNCTION(int, dl_iterate_phdr,
                   (__dl_iterate_phdr_callback_t callback, void *arg)) {
  // Where a loader linked the process, what it loaded is what there is to
  // report, and it left a record of all of it. An unwinder walks this to find
  // the frame information of whatever module a return address is in, so a
  // library left out of it is one nothing can unwind through.
  const elf::ModuleSet *modules = elf::process_modules();
  if (modules != nullptr && modules->linked) {
    // dlopen and dlclose change the set under this lock, and a library that
    // is unmapped or an array that is replaced part way through the walk
    // would be read after it was gone. The load lock is not used, as it is
    // held while the initialisers of a new library run, and one of them can
    // be waiting for a thread that is here to unwind.
    cpp::lock_guard lock(dl::dl_write_lock);
    for (size_t i = 0; i < modules->count; ++i) {
      struct dl_phdr_info info = info_for(modules->modules[i], *modules);
      // The main executable is reported with an empty name, which is how a
      // caller tells it from everything else. Its own headers cannot be
      // reached from here: the linker defined symbol naming them means this
      // library's, not the program's, so the record is what says where they
      // are.
      if (i == 0)
        info.dlpi_name = "";
      const int code = callback(&info, sizeof(info), arg);
      if (code != 0)
        return code;
    }
  } else {
    // Nothing linked this, so the executable is all there is besides what the
    // kernel mapped.
    ElfW(Ehdr) *executable_header =
        reinterpret_cast<ElfW(Ehdr) *>(&__ehdr_start);
    struct dl_phdr_info executable_info =
        create_module_info(executable_header, "");
    const int code = callback(&executable_info, sizeof(executable_info), arg);
    if (code != 0)
      return code;
  }

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
