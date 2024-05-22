//===-- Implementation of crt for x86_64 ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "config/linux/app.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/threads/thread.h"
#include "src/stdlib/abort.h"
#include "src/stdlib/atexit.h"
#include "src/stdlib/exit.h"
#include "src/string/memory_utils/inline_memcpy.h"

#include <asm/prctl.h>
#include <linux/auxvec.h>
#include <linux/elf.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

extern "C" int main(int, char **, char **);

extern "C" void __stack_chk_fail() {
  LIBC_NAMESPACE::write_to_stderr("stack smashing detected");
  LIBC_NAMESPACE::abort();
}

namespace LIBC_NAMESPACE {

#ifdef SYS_mmap2
static constexpr long MMAP_SYSCALL_NUMBER = SYS_mmap2;
#elif SYS_mmap
static constexpr long MMAP_SYSCALL_NUMBER = SYS_mmap;
#else
#error "mmap and mmap2 syscalls not available."
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
  uintptr_t tls_size = app.tls.size & -app.tls.align;
  if (tls_size != app.tls.size)
    tls_size += app.tls.align;

  // Per the x86_64 TLS ABI, the entry pointed to by the thread pointer is the
  // address of the TLS block. So, we add more size to accomodate this address
  // entry.
  // We also need to include space for the stack canary. The canary is at
  // offset 0x28 (40) and is of size uintptr_t.
  uintptr_t tls_size_with_addr = tls_size + sizeof(uintptr_t) + 40;

  // We cannot call the mmap function here as the functions set errno on
  // failure. Since errno is implemented via a thread local variable, we cannot
  // use errno before TLS is setup.
  long mmap_retval = LIBC_NAMESPACE::syscall_impl<long>(
      MMAP_SYSCALL_NUMBER, nullptr, tls_size_with_addr, PROT_READ | PROT_WRITE,
      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  // We cannot check the return value with MAP_FAILED as that is the return
  // of the mmap function and not the mmap syscall.
  if (mmap_retval < 0 && static_cast<uintptr_t>(mmap_retval) > -app.page_size)
    LIBC_NAMESPACE::syscall_impl<long>(SYS_exit, 1);
  uintptr_t *tls_addr = reinterpret_cast<uintptr_t *>(mmap_retval);

  // x86_64 TLS faces down from the thread pointer with the first entry
  // pointing to the address of the first real TLS byte.
  uintptr_t end_ptr = reinterpret_cast<uintptr_t>(tls_addr) + tls_size;
  *reinterpret_cast<uintptr_t *>(end_ptr) = end_ptr;

  LIBC_NAMESPACE::inline_memcpy(reinterpret_cast<char *>(tls_addr),
                                reinterpret_cast<const char *>(app.tls.address),
                                app.tls.init_size);
  uintptr_t *stack_guard_addr = reinterpret_cast<uintptr_t *>(end_ptr + 40);
  // Setting the stack guard to a random value.
  // We cannot call the get_random function here as the function sets errno on
  // failure. Since errno is implemented via a thread local variable, we cannot
  // use errno before TLS is setup.
  ssize_t stack_guard_retval = LIBC_NAMESPACE::syscall_impl<ssize_t>(
      SYS_getrandom, reinterpret_cast<long>(stack_guard_addr), sizeof(uint64_t),
      0);
  if (stack_guard_retval < 0)
    LIBC_NAMESPACE::syscall_impl(SYS_exit, 1);

  tls_descriptor = {tls_size_with_addr, reinterpret_cast<uintptr_t>(tls_addr),
                    end_ptr};
  return;
}

void cleanup_tls(uintptr_t addr, uintptr_t size) {
  if (size == 0)
    return;
  LIBC_NAMESPACE::syscall_impl<long>(SYS_munmap, addr, size);
}

// Sets the thread pointer to |val|. Returns true on success, false on failure.
static bool set_thread_ptr(uintptr_t val) {
  return LIBC_NAMESPACE::syscall_impl(SYS_arch_prctl, ARCH_SET_FS, val) != -1;
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

} // namespace LIBC_NAMESPACE

using LIBC_NAMESPACE::app;

// TODO: Would be nice to use the aux entry structure from elf.h when available.
struct AuxEntry {
  uint64_t type;
  uint64_t value;
};

extern "C" void _start() {
  // This TU is compiled with -fno-omit-frame-pointer. Hence, the previous value
  // of the base pointer is pushed on to the stack. So, we step over it (the
  // "+ 1" below) to get to the args.
  app.args = reinterpret_cast<LIBC_NAMESPACE::Args *>(
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
  __asm__ __volatile__("andq $0xfffffffffffffff0, %rsp\n\t");
  __asm__ __volatile__("andq $0xfffffffffffffff0, %rbp\n\t");

  auto tid = LIBC_NAMESPACE::syscall_impl<long>(SYS_gettid);
  if (tid <= 0)
    LIBC_NAMESPACE::syscall_impl<long>(SYS_exit, 1);
  LIBC_NAMESPACE::main_thread_attrib.tid = static_cast<int>(tid);

  // After the argv array, is a 8-byte long NULL value before the array of env
  // values. The end of the env values is marked by another 8-byte long NULL
  // value. We step over it (the "+ 1" below) to get to the env values.
  uint64_t *env_ptr = app.args->argv + app.args->argc + 1;
  uint64_t *env_end_marker = env_ptr;
  app.env_ptr = env_ptr;
  while (*env_end_marker)
    ++env_end_marker;

  // Initialize the POSIX global declared in unistd.h
  environ = reinterpret_cast<char **>(env_ptr);

  // After the env array, is the aux-vector. The end of the aux-vector is
  // denoted by an AT_NULL entry.
  Elf64_Phdr *program_hdr_table = nullptr;
  uintptr_t program_hdr_count = 0;
  for (AuxEntry *aux_entry = reinterpret_cast<AuxEntry *>(env_end_marker + 1);
       aux_entry->type != AT_NULL; ++aux_entry) {
    switch (aux_entry->type) {
    case AT_PHDR:
      program_hdr_table = reinterpret_cast<Elf64_Phdr *>(aux_entry->value);
      break;
    case AT_PHNUM:
      program_hdr_count = aux_entry->value;
      break;
    case AT_PAGESZ:
      app.page_size = aux_entry->value;
      break;
    default:
      break; // TODO: Read other useful entries from the aux vector.
    }
  }

  app.tls.size = 0;
  for (uintptr_t i = 0; i < program_hdr_count; ++i) {
    Elf64_Phdr *phdr = program_hdr_table + i;
    if (phdr->p_type != PT_TLS)
      continue;
    // TODO: p_vaddr value has to be adjusted for static-pie executables.
    app.tls.address = phdr->p_vaddr;
    app.tls.size = phdr->p_memsz;
    app.tls.init_size = phdr->p_filesz;
    app.tls.align = phdr->p_align;
  }

  // This descriptor has to be static since its cleanup function cannot
  // capture the context.
  static LIBC_NAMESPACE::TLSDescriptor tls;
  LIBC_NAMESPACE::init_tls(tls);
  if (tls.size != 0 && !LIBC_NAMESPACE::set_thread_ptr(tls.tp))
    LIBC_NAMESPACE::syscall_impl<long>(SYS_exit, 1);

  LIBC_NAMESPACE::self.attrib = &LIBC_NAMESPACE::main_thread_attrib;
  LIBC_NAMESPACE::main_thread_attrib.atexit_callback_mgr =
      LIBC_NAMESPACE::internal::get_thread_atexit_callback_mgr();
  // We register the cleanup_tls function to be the last atexit callback to be
  // invoked. It will tear down the TLS. Other callbacks may depend on TLS (such
  // as the stack protector canary).
  LIBC_NAMESPACE::atexit(
      []() { LIBC_NAMESPACE::cleanup_tls(tls.tp, tls.size); });
  // We want the fini array callbacks to be run after other atexit
  // callbacks are run. So, we register them before running the init
  // array callbacks as they can potentially register their own atexit
  // callbacks.
  LIBC_NAMESPACE::atexit(&LIBC_NAMESPACE::call_fini_array_callbacks);

  LIBC_NAMESPACE::call_init_array_callbacks(
      static_cast<int>(app.args->argc),
      reinterpret_cast<char **>(app.args->argv),
      reinterpret_cast<char **>(env_ptr));

  int retval = main(static_cast<int>(app.args->argc),
                    reinterpret_cast<char **>(app.args->argv),
                    reinterpret_cast<char **>(env_ptr));

  LIBC_NAMESPACE::exit(retval);
}
