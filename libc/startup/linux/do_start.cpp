//===-- Implementation file of do_start -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "startup/linux/do_start.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/threads/thread.h"
#include "src/stdlib/atexit.h"
#include "src/stdlib/exit.h"
#include "src/unistd/environ.h"

#include <linux/auxvec.h>
#include <linux/elf.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/syscall.h>

extern "C" int main(int argc, char **argv, char **envp);

extern "C" {
// These arrays are present in the .init_array and .fini_array sections.
// The symbols are inserted by linker when it sees references to them.
extern uintptr_t __preinit_array_start[];
extern uintptr_t __preinit_array_end[];
extern uintptr_t __init_array_start[];
extern uintptr_t __init_array_end[];
extern uintptr_t __fini_array_start[];
extern uintptr_t __fini_array_end[];
// https://refspecs.linuxbase.org/elf/gabi4+/ch5.dynamic.html#dynamic_section
// This symbol is provided by the dynamic linker. It can be undefined depending
// on how the program is loaded exactly.
[[gnu::weak,
  gnu::visibility("hidden")]] extern const Elf64_Dyn _DYNAMIC[]; // NOLINT
}

namespace LIBC_NAMESPACE {
// TODO: this symbol will be moved to config.linux.app
AppProperties app;

using InitCallback = void(int, char **, char **);
using FiniCallback = void(void);

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

static ThreadAttributes main_thread_attrib;

[[noreturn]] void do_start() {
  auto tid = syscall_impl<long>(SYS_gettid);
  if (tid <= 0)
    syscall_impl<long>(SYS_exit, 1);
  main_thread_attrib.tid = static_cast<int>(tid);

  // After the argv array, is a 8-byte long NULL value before the array of env
  // values. The end of the env values is marked by another 8-byte long NULL
  // value. We step over it (the "+ 1" below) to get to the env values.
  ArgVEntryType *env_ptr = app.args->argv + app.args->argc + 1;
  ArgVEntryType *env_end_marker = env_ptr;
  app.env_ptr = env_ptr;
  while (*env_end_marker)
    ++env_end_marker;

  // Initialize the POSIX global declared in unistd.h
  environ = reinterpret_cast<char **>(env_ptr);

  // After the env array, is the aux-vector. The end of the aux-vector is
  // denoted by an AT_NULL entry.
  Elf64_Phdr *program_hdr_table = nullptr;
  uintptr_t program_hdr_count = 0;
  app.auxv_ptr = reinterpret_cast<AuxEntry *>(env_end_marker + 1);
  for (auto *aux_entry = app.auxv_ptr; aux_entry->id != AT_NULL; ++aux_entry) {
    switch (aux_entry->id) {
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

  ptrdiff_t base = 0;
  app.tls.size = 0;
  Elf64_Phdr *tls_phdr = nullptr;

  for (uintptr_t i = 0; i < program_hdr_count; ++i) {
    Elf64_Phdr &phdr = program_hdr_table[i];
    if (phdr.p_type == PT_PHDR)
      base = reinterpret_cast<ptrdiff_t>(program_hdr_table) - phdr.p_vaddr;
    if (phdr.p_type == PT_DYNAMIC && _DYNAMIC)
      base = reinterpret_cast<ptrdiff_t>(_DYNAMIC) - phdr.p_vaddr;
    if (phdr.p_type == PT_TLS)
      tls_phdr = &phdr;
    // TODO: adjust PT_GNU_STACK
  }

  app.tls.address = tls_phdr->p_vaddr + base;
  app.tls.size = tls_phdr->p_memsz;
  app.tls.init_size = tls_phdr->p_filesz;
  app.tls.align = tls_phdr->p_align;

  // This descriptor has to be static since its cleanup function cannot
  // capture the context.
  static TLSDescriptor tls;
  init_tls(tls);
  if (tls.size != 0 && !set_thread_ptr(tls.tp))
    syscall_impl<long>(SYS_exit, 1);

  self.attrib = &main_thread_attrib;
  main_thread_attrib.atexit_callback_mgr =
      internal::get_thread_atexit_callback_mgr();
  // We register the cleanup_tls function to be the last atexit callback to be
  // invoked. It will tear down the TLS. Other callbacks may depend on TLS (such
  // as the stack protector canary).
  atexit([]() { cleanup_tls(tls.tp, tls.size); });
  // We want the fini array callbacks to be run after other atexit
  // callbacks are run. So, we register them before running the init
  // array callbacks as they can potentially register their own atexit
  // callbacks.
  atexit(&call_fini_array_callbacks);

  call_init_array_callbacks(static_cast<int>(app.args->argc),
                            reinterpret_cast<char **>(app.args->argv),
                            reinterpret_cast<char **>(env_ptr));

  int retval = main(static_cast<int>(app.args->argc),
                    reinterpret_cast<char **>(app.args->argv),
                    reinterpret_cast<char **>(env_ptr));

  exit(retval);
}

} // namespace LIBC_NAMESPACE
