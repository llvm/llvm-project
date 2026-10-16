//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines Linux architecture-level unmap_and_exit dispatcher.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_UNMAP_AND_EXIT_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_UNMAP_AND_EXIT_H

#include "src/__support/macros/properties/architectures.h"

#if defined(LIBC_TARGET_ARCH_IS_X86_64)
#include "src/__support/threads/linux/x86_64/unmap_and_exit.h"
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "src/__support/threads/linux/aarch64/unmap_and_exit.h"
#elif defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
#include "src/__support/threads/linux/riscv/unmap_and_exit.h"
#else
#error "Unsupported architecture for Linux unmap_and_exit"
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_UNMAP_AND_EXIT_H
