//===-- Implementation of crt for x86_64 ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "config/linux/app.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/string/memcpy.h"

#include <asm/prctl.h>
#include <linux/auxvec.h>
#include <linux/elf.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/syscall.h>

extern "C" int main(int, char **, char **);

namespace __llvm_libc {

#ifdef SYS_mmap2
static constexpr long mmapSyscallNumber = SYS_mmap2;
#elif SYS_mmap
static constexpr long mmapSyscallNumber = SYS_mmap;
#else
#error "Target platform does not have SYS_mmap or SYS_mmap2 defined"
#endif

AppProperties app;

// TODO: The function is x86_64 specific. Move it to config/linux/app.h
// and generalize it. Also, dynamic loading is not handled currently.
void initTLS() {
  if (app.tls.size == 0)
    return;

  // We will assume the alignment is always a power of two.
  uintptr_t tlsSize = app.tls.size & -app.tls.align;
  if (tlsSize != app.tls.size)
    tlsSize += app.tls.align;

  // Per the x86_64 TLS ABI, the entry pointed to by the thread pointer is the
  // address of the TLS block. So, we add more size to accomodate this address
  // entry.
  size_t tlsSizeWithAddr = tlsSize + sizeof(uintptr_t);

  // We cannot call the mmap function here as the functions set errno on
  // failure. Since errno is implemented via a thread local variable, we cannot
  // use errno before TLS is setup.
  long mmapRetVal = __llvm_libc::syscall(
      mmapSyscallNumber, nullptr, tlsSizeWithAddr, PROT_READ | PROT_WRITE,
      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  // We cannot check the return value with MAP_FAILED as that is the return
  // of the mmap function and not the mmap syscall.
  if (mmapRetVal < 0 && static_cast<uintptr_t>(mmapRetVal) > -app.pageSize)
    __llvm_libc::syscall(SYS_exit, 1);
  uintptr_t *tlsAddr = reinterpret_cast<uintptr_t *>(mmapRetVal);

  // x86_64 TLS faces down from the thread pointer with the first entry
  // pointing to the address of the first real TLS byte.
  uintptr_t endPtr = reinterpret_cast<uintptr_t>(tlsAddr) + tlsSize;
  *reinterpret_cast<uintptr_t *>(endPtr) = endPtr;

  __llvm_libc::memcpy(tlsAddr, reinterpret_cast<const void *>(app.tls.address),
                      app.tls.size);
  if (__llvm_libc::syscall(SYS_arch_prctl, ARCH_SET_FS, endPtr) == -1)
    __llvm_libc::syscall(SYS_exit, 1);
}

} // namespace __llvm_libc

using __llvm_libc::app;

// TODO: Would be nice to use the aux entry structure from elf.h when available.
struct AuxEntry {
  uint64_t type;
  uint64_t value;
};

extern "C" void _start() {
  // This TU is compiled with -fno-omit-frame-pointer. Hence, the previous value
  // of the base pointer is pushed on to the stack. So, we step over it (the
  // "+ 1" below) to get to the args.
  app.args = reinterpret_cast<__llvm_libc::Args *>(
      reinterpret_cast<uintptr_t *>(__builtin_frame_address(0)) + 1);

  // The x86_64 ABI requires that the stack pointer is aligned to a 16-byte
  // boundary. We align it here but we cannot use any local variables created
  // before the following alignment. Best would be to not create any local
  // variables before the alignment. Also, note that we are aligning the stack
  // downwards as the x86_64 stack grows downwards. This ensures that we don't
  // tread on argc, argv etc.
  // NOTE: Compiler attributes for alignment do not help here as the stack
  // pointer on entry to this _start function is controlled by the OS. In fact,
  // compilers can generate code assuming the alignment as required by the ABI.
  // If the stack pointers as setup by the OS are already aligned, then the
  // following code is a NOP.
  __asm__ __volatile__("andq $0xfffffffffffffff0, %%rsp\n\t" ::: "%rsp");
  __asm__ __volatile__("andq $0xfffffffffffffff0, %%rbp\n\t" ::: "%rbp");

  // After the argv array, is a 8-byte long NULL value before the array of env
  // values. The end of the env values is marked by another 8-byte long NULL
  // value. We step over it (the "+ 1" below) to get to the env values.
  uint64_t *env_ptr = app.args->argv + app.args->argc + 1;
  uint64_t *env_end_marker = env_ptr;
  app.envPtr = env_ptr;
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

  for (uintptr_t i = 0; i < programHdrCount; ++i) {
    Elf64_Phdr *phdr = programHdrTable + i;
    if (phdr->p_type != PT_TLS)
      continue;
    // TODO: p_vaddr value has to be adjusted for static-pie executables.
    app.tls.address = phdr->p_vaddr;
    app.tls.size = phdr->p_memsz;
    app.tls.align = phdr->p_align;
  }

  __llvm_libc::initTLS();

  __llvm_libc::syscall(SYS_exit, main(app.args->argc,
                                      reinterpret_cast<char **>(app.args->argv),
                                      reinterpret_cast<char **>(env_ptr)));
}
