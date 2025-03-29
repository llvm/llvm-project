//===-- Implementation of tls for aarch64 ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/thread.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "startup/linux/do_start.h"

#include <arm_acle.h>
#include <sys/mman.h>
#include <sys/syscall.h>

// Source documentation:
// https://github.com/ARM-software/abi-aa/tree/main/sysvabi64

namespace LIBC_NAMESPACE_DECL {

#ifdef SYS_mmap2
static constexpr long MMAP_SYSCALL_NUMBER = SYS_mmap2;
#elif SYS_mmap
static constexpr long MMAP_SYSCALL_NUMBER = SYS_mmap;
#else
#error "mmap and mmap2 syscalls not available."
#endif

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
  long mmap_ret_val = syscall_impl<long>(MMAP_SYSCALL_NUMBER, nullptr,
                                         alloc_size, PROT_READ | PROT_WRITE,
                                         MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  // We cannot check the return value with MAP_FAILED as that is the return
  // of the mmap function and not the mmap syscall.
  if (mmap_ret_val < 0 && static_cast<uintptr_t>(mmap_ret_val) > -app.page_size)
    syscall_impl<long>(SYS_exit, 1);
  uintptr_t thread_ptr = uintptr_t(reinterpret_cast<uintptr_t *>(mmap_ret_val));
  uintptr_t tls_addr = thread_ptr + size_of_pointers + padding;
  inline_memcpy(reinterpret_cast<char *>(tls_addr),
                reinterpret_cast<const char *>(app.tls.address),
                app.tls.init_size);
  tls_descriptor.size = alloc_size;
  tls_descriptor.addr = thread_ptr;
  tls_descriptor.tp = thread_ptr;
}

void cleanup_tls(uintptr_t addr, uintptr_t size) {
  if (size == 0)
    return;
  syscall_impl<long>(SYS_munmap, addr, size);
}

bool set_thread_ptr(uintptr_t val) {
// The PR for __arm_wsr64 support in GCC was merged on Dec 6, 2023, and it is
// not yet usable in 13.3.0
// https://github.com/gcc-mirror/gcc/commit/fc42900d21abd5eacb7537c3c8ffc5278d510195
#if __has_builtin(__builtin_arm_wsr64)
  __builtin_arm_wsr64("tpidr_el0", val);
#elif __has_builtin(__builtin_aarch64_wsr)
  __builtin_aarch64_wsr("tpidr_el0", val);
#elif defined(__GNUC__)
  asm volatile("msr tpidr_el0, %0" ::"r"(val));
#else
#error "Unsupported compiler"
#endif
  return true;
}
} // namespace LIBC_NAMESPACE_DECL
