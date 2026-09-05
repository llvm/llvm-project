//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Installing the thread pointer
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_THREAD_POINTER_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_THREAD_POINTER_H

#include "hdr/stdint_proxy.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"

#if defined(LIBC_TARGET_ARCH_IS_X86_64)
#include <asm/prctl.h>
#include <sys/syscall.h>
#endif

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// Points the thread register at a block prepared by the loader. Everything
// thread local in every module is addressed relative to this, so nothing that
// touches thread local storage may run before it.
LIBC_INLINE bool set_thread_pointer(uintptr_t value) {
#if defined(LIBC_TARGET_ARCH_IS_X86_64)
  // The register is not writable from user mode, so the kernel sets it.
  return syscall_impl(SYS_arch_prctl, ARCH_SET_FS, value) != -1;
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#if __has_builtin(__builtin_arm_wsr64)
  __builtin_arm_wsr64("tpidr_el0", value);
#else
  __asm__ __volatile__("msr tpidr_el0, %0" : : "r"(value));
#endif
  return true;
#elif defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
  __asm__ __volatile__("mv tp, %0\n\t" : : "r"(value));
  return true;
#else
  // No loader runs here, so nothing asks for the thread pointer to be set.
  (void)value;
  return false;
#endif
}

// Where the thread's blocks are, which is what a thread local's offset is
// measured from.
LIBC_INLINE uintptr_t thread_pointer() {
#if __has_builtin(__builtin_thread_pointer)
  return reinterpret_cast<uintptr_t>(__builtin_thread_pointer());
#elif defined(LIBC_TARGET_ARCH_IS_X86_64)
  // The register reads back through itself; the first word of the thread
  // control block holds its own address so that this works.
  uintptr_t value;
  __asm__ __volatile__("mov %%fs:0, %0" : "=r"(value));
  return value;
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  uintptr_t value;
  __asm__ __volatile__("mrs %0, tpidr_el0" : "=r"(value));
  return value;
#elif defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
  uintptr_t value;
  __asm__ __volatile__("mv %0, tp" : "=r"(value));
  return value;
#else
  return 0;
#endif
}

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_THREAD_POINTER_H
