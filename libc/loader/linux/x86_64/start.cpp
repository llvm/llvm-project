//===-- Implementation of crt for x86_64 ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "config/linux/app.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/threads/thread.h"
#include "src/stdlib/atexit.h"
#include "src/stdlib/exit.h"
#include "src/string/memory_utils/memcpy_implementations.h"

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

static ThreadAttributes main_thread_attrib;

// TODO: The function is x86_64 specific. Move it to config/linux/app.h
// and generalize it. Also, dynamic loading is not handled currently.
void init_tls(TLSDescriptor &tls_descriptor) {
  if (app.tls.size == 0) {
    tls_descriptor.size = 0;
    tls_descriptor.tp = 0;
    return;
  }

  // We will assume the alignment is always a power of two.
  uintptr_t tlsSize = app.tls.size & -app.tls.align;
  if (tlsSize != app.tls.size)
    tlsSize += app.tls.align;

  // Per the x86_64 TLS ABI, the entry pointed to by the thread pointer is the
  // address of the TLS block. So, we add more size to accomodate this address
  // entry.
  uintptr_t tlsSizeWithAddr = tlsSize + sizeof(uintptr_t);

  // We cannot call the mmap function here as the functions set errno on
  // failure. Since errno is implemented via a thread local variable, we cannot
  // use errno before TLS is setup.
  long mmapRetVal = __llvm_libc::syscall_impl(
      mmapSyscallNumber, nullptr, tlsSizeWithAddr, PROT_READ | PROT_WRITE,
      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  // We cannot check the return value with MAP_FAILED as that is the return
  // of the mmap function and not the mmap syscall.
  if (mmapRetVal < 0 && static_cast<uintptr_t>(mmapRetVal) > -app.pageSize)
    __llvm_libc::syscall_impl(SYS_exit, 1);
  uintptr_t *tlsAddr = reinterpret_cast<uintptr_t *>(mmapRetVal);

  // x86_64 TLS faces down from the thread pointer with the first entry
  // pointing to the address of the first real TLS byte.
  uintptr_t endPtr = reinterpret_cast<uintptr_t>(tlsAddr) + tlsSize;
  *reinterpret_cast<uintptr_t *>(endPtr) = endPtr;

  __llvm_libc::inline_memcpy(reinterpret_cast<char *>(tlsAddr),
                             reinterpret_cast<const char *>(app.tls.address),
                             app.tls.init_size);

  tls_descriptor = {tlsSizeWithAddr, uintptr_t(tlsAddr), endPtr};
  return;
}

void cleanup_tls(uintptr_t addr, uintptr_t size) {
  if (size == 0)
    return;
  __llvm_libc::syscall_impl(SYS_munmap, addr, size);
}

// Sets the thread pointer to |val|. Returns true on success, false on failure.
static bool set_thread_ptr(uintptr_t val) {
  return __llvm_libc::syscall_impl(SYS_arch_prctl, ARCH_SET_FS, val) == -1
             ? false
             : true;
}

using InitCallback = void(int, char **, char **);
using FiniCallback = void(void);

extern "C" {
// These arrays are present in the .init_array and .fini_array sections.
// The symbols are inserted by linker when it sees references to them.
extern uintptr_t __preinit_array_start[];
extern uintptr_t __preinit_array_end[];
extern uintptr_t __init_array_start[];
extern uintptr_t __init_array_end[];
extern uintptr_t __fini_array_start[];
extern uintptr_t __fini_array_end[];
}

static void call_init_array_callbacks(int argc, char **argv, char **env) {
  size_t preinit_array_size = __preinit_array_end - __preinit_array_start;
  for (size_t i = 0; i < preinit_array_size; ++i)
    reinterpret_cast<InitCallback *>(__preinit_array_start[i])(argc, argv, env);
  size_t init_array_size = __init_array_end - __init_array_start;
  for (size_t i = 0; i < init_array_size; ++i)
    reinterpret_cast<InitCallback *>(__init_array_start[i])(argc, argv, env);
}

static void call_fini_array_callbacks() {
  size_t fini_array_size = __fini_array_end - __fini_array_start;
  for (size_t i = fini_array_size; i > 0; --i)
    reinterpret_cast<FiniCallback *>(__fini_array_start[i - 1])();
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

  auto tid = __llvm_libc::syscall_impl(SYS_gettid);
  if (tid <= 0)
    __llvm_libc::syscall_impl(SYS_exit, 1);
  __llvm_libc::main_thread_attrib.tid = tid;

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
  if (tls.size != 0 && !__llvm_libc::set_thread_ptr(tls.tp))
    __llvm_libc::syscall_impl(SYS_exit, 1);

  __llvm_libc::self.attrib = &__llvm_libc::main_thread_attrib;
  __llvm_libc::main_thread_attrib.atexit_callback_mgr =
      __llvm_libc::internal::get_thread_atexit_callback_mgr();

  // We want the fini array callbacks to be run after other atexit
  // callbacks are run. So, we register them before running the init
  // array callbacks as they can potentially register their own atexit
  // callbacks.
  __llvm_libc::atexit(&__llvm_libc::call_fini_array_callbacks);

  __llvm_libc::call_init_array_callbacks(
      app.args->argc, reinterpret_cast<char **>(app.args->argv),
      reinterpret_cast<char **>(env_ptr));

  int retval = main(app.args->argc, reinterpret_cast<char **>(app.args->argv),
                    reinterpret_cast<char **>(env_ptr));

  // TODO: TLS cleanup should be done after all other atexit callbacks
  // are run. So, register a cleanup callback for it with atexit before
  // everything else.
  __llvm_libc::cleanup_tls(tls.addr, tls.size);
  __llvm_libc::exit(retval);
}
