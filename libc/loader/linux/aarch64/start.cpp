//===-- Implementation of crt for aarch64 ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "config/linux/app.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/string/memory_utils/memcpy_implementations.h"

#include <arm_acle.h>

#include <linux/auxvec.h>
#include <linux/elf.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/syscall.h>

extern "C" int main(int, char **, char **);

// Source documentation:
// https://github.com/ARM-software/abi-aa/tree/main/sysvabi64

namespace __llvm_libc {

#ifdef SYS_mmap2
static constexpr long MMAP_SYSCALL_NUMBER = SYS_mmap2;
#elif SYS_mmap
static constexpr long MMAP_SYSCALL_NUMBER = SYS_mmap;
#else
#error "Target platform does not have SYS_mmap or SYS_mmap2 defined"
#endif

AppProperties app;

void init_tls(TLSDescriptor &tls_descriptor) {
  if (app.tls.size == 0) {
    tls_descriptor.size = 0;
    tls_descriptor.tp = 0;
    return;
  }

  // aarch64 follows the variant 1 TLS layout:
  //
  // 1. First entry is the dynamic thread vector pointer
  // 2. Second entry is a 8-byte reserved word.
  // 3. Padding for alignment.
  // 4. The TLS data from the ELF image.
  //
  // The thread pointer points to the first entry.

  const uintptr_t size_of_pointers = 2 * sizeof(uintptr_t);
  uintptr_t padding = 0;
  const uintptr_t ALIGNMENT_MASK = app.tls.align - 1;
  uintptr_t diff = size_of_pointers & ALIGNMENT_MASK;
  if (diff != 0)
    padding += (ALIGNMENT_MASK - diff) + 1;

  uintptr_t alloc_size = size_of_pointers + padding + app.tls.size;

  // We cannot call the mmap function here as the functions set errno on
  // failure. Since errno is implemented via a thread local variable, we cannot
  // use errno before TLS is setup.
  long mmap_ret_val = __llvm_libc::syscall(MMAP_SYSCALL_NUMBER, nullptr,
                                           alloc_size, PROT_READ | PROT_WRITE,
                                           MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  // We cannot check the return value with MAP_FAILED as that is the return
  // of the mmap function and not the mmap syscall.
  if (mmap_ret_val < 0 && static_cast<uintptr_t>(mmap_ret_val) > -app.pageSize)
    __llvm_libc::syscall(SYS_exit, 1);
  uintptr_t thread_ptr = uintptr_t(reinterpret_cast<uintptr_t *>(mmap_ret_val));
  uintptr_t tls_addr = thread_ptr + size_of_pointers + padding;
  __llvm_libc::inline_memcpy(reinterpret_cast<char *>(tls_addr),
                             reinterpret_cast<const char *>(app.tls.address),
                             app.tls.init_size);
  tls_descriptor.size = alloc_size;
  tls_descriptor.addr = thread_ptr;
  tls_descriptor.tp = thread_ptr;
}

void cleanup_tls(uintptr_t addr, uintptr_t size) {
  if (size == 0)
    return;
  __llvm_libc::syscall(SYS_munmap, addr, size);
}

static void set_thread_ptr(uintptr_t val) { __arm_wsr64("tpidr_el0", val); }

} // namespace __llvm_libc

using __llvm_libc::app;

// TODO: Would be nice to use the aux entry structure from elf.h when available.
struct AuxEntry {
  uint64_t type;
  uint64_t value;
};

extern "C" void _start() {
  // Skip the Frame Pointer and the Link Register
  // https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst
  // Section 6.2.3
  app.args = reinterpret_cast<__llvm_libc::Args *>(
      reinterpret_cast<uintptr_t *>(__builtin_frame_address(0)) + 2);

  // After the argv array, is a 8-byte long NULL value before the array of env
  // values. The end of the env values is marked by another 8-byte long NULL
  // value. We step over it (the "+ 1" below) to get to the env values.
  uint64_t *env_ptr = app.args->argv + app.args->argc + 1;
  uint64_t *env_end_marker = env_ptr;
  while (*env_end_marker)
    ++env_end_marker;

  // After the env array, is the aux-vector. The end of the aux-vector is
  // denoted by an AT_NULL entry.
  Elf64_Phdr *programHdrTable = nullptr;
  uintptr_t programHdrCount;
  for (AuxEntry *aux_entry = reinterpret_cast<AuxEntry *>(env_end_marker + 1);
       aux_entry->type != AT_NULL; ++aux_entry) {
    switch (aux_entry->type) {
    case AT_PHDR:
      programHdrTable = reinterpret_cast<Elf64_Phdr *>(aux_entry->value);
      break;
    case AT_PHNUM:
      programHdrCount = aux_entry->value;
      break;
    case AT_PAGESZ:
      app.pageSize = aux_entry->value;
      break;
    default:
      break; // TODO: Read other useful entries from the aux vector.
    }
  }

  app.tls.size = 0;
  for (uintptr_t i = 0; i < programHdrCount; ++i) {
    Elf64_Phdr *phdr = programHdrTable + i;
    if (phdr->p_type != PT_TLS)
      continue;
    // TODO: p_vaddr value has to be adjusted for static-pie executables.
    app.tls.address = phdr->p_vaddr;
    app.tls.size = phdr->p_memsz;
    app.tls.init_size = phdr->p_filesz;
    app.tls.align = phdr->p_align;
  }

  __llvm_libc::TLSDescriptor tls;
  __llvm_libc::init_tls(tls);
  if (tls.size != 0)
    __llvm_libc::set_thread_ptr(tls.tp);

  int retval = main(app.args->argc, reinterpret_cast<char **>(app.args->argv),
                    reinterpret_cast<char **>(env_ptr));
  __llvm_libc::cleanup_tls(tls.addr, tls.size);
  __llvm_libc::syscall(SYS_exit, retval);
}
