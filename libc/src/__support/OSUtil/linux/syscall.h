//===----------------------- Linux syscalls ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_H

#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/macros/properties/architectures.h"

#ifdef LIBC_TARGET_ARCH_IS_X86_64
#include "x86_64/syscall.h"
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "aarch64/syscall.h"
#elif defined(LIBC_TARGET_ARCH_IS_ARM)
#include "arm/syscall.h"
#elif defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
#include "riscv/syscall.h"
#endif

namespace LIBC_NAMESPACE {

template <typename R, typename... Ts>
LIBC_INLINE R syscall_impl(long __number, Ts... ts) {
  static_assert(sizeof...(Ts) <= 6, "Too many arguments for syscall");
  return cpp::bit_or_static_cast<R>(syscall_impl(__number, (long)ts...));
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_H
