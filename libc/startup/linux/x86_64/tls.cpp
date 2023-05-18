//===-- Implementation of tls for x86_64 ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "startup/linux/do_start.h"

#include <asm/prctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>

namespace LIBC_NAMESPACE {

#ifdef SYS_mmap2
static constexpr long MMAP_SYSCALL_NUMBER = SYS_mmap2;
#elif SYS_mmap
static constexpr long MMAP_SYSCALL_NUMBER = SYS_mmap;
#else
#error "mmap and mmap2 syscalls not available."
#endif

// TODO: Also generalize this routine and handle dynamic loading properly.
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
  long mmap_retval = syscall_impl<long>(
      MMAP_SYSCALL_NUMBER, nullptr, tls_size_with_addr, PROT_READ | PROT_WRITE,
      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  // We cannot check the return value with MAP_FAILED as that is the return
  // of the mmap function and not the mmap syscall.
  if (mmap_retval < 0 && static_cast<uintptr_t>(mmap_retval) > -app.page_size)
    syscall_impl<long>(SYS_exit, 1);
  uintptr_t *tls_addr = reinterpret_cast<uintptr_t *>(mmap_retval);

  // x86_64 TLS faces down from the thread pointer with the first entry
  // pointing to the address of the first real TLS byte.
  uintptr_t end_ptr = reinterpret_cast<uintptr_t>(tls_addr) + tls_size;
  *reinterpret_cast<uintptr_t *>(end_ptr) = end_ptr;

  inline_memcpy(reinterpret_cast<char *>(tls_addr),
                reinterpret_cast<const char *>(app.tls.address),
                app.tls.init_size);
  uintptr_t *stack_guard_addr = reinterpret_cast<uintptr_t *>(end_ptr + 40);
  // Setting the stack guard to a random value.
  // We cannot call the get_random function here as the function sets errno on
  // failure. Since errno is implemented via a thread local variable, we cannot
  // use errno before TLS is setup.
  long stack_guard_retval =
      syscall_impl(SYS_getrandom, reinterpret_cast<long>(stack_guard_addr),
                   sizeof(uint64_t), 0);
  if (stack_guard_retval < 0)
    syscall_impl(SYS_exit, 1);

  tls_descriptor = {tls_size_with_addr, reinterpret_cast<uintptr_t>(tls_addr),
                    end_ptr};
  return;
}

void cleanup_tls(uintptr_t addr, uintptr_t size) {
  if (size == 0)
    return;
  syscall_impl<long>(SYS_munmap, addr, size);
}

// Sets the thread pointer to |val|. Returns true on success, false on failure.
bool set_thread_ptr(uintptr_t val) {
  return syscall_impl(SYS_arch_prctl, ARCH_SET_FS, val) != -1;
}
} // namespace LIBC_NAMESPACE
