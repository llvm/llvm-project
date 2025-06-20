//===--- Futex Utilities ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===--- Futex Wrapper ------------------------------------------*- C++ -*-===//
// We take the name "futex" from Linux system. This library provides a general
// wrapper for waiting and notifying on atomic words. Various platforms support
// futex-like operations.
// - Windows: WaitOnAddress and WakeByAddressSingle/WakeByAddressAll
//   (Windows futex cannot be used in inter-process synchronization)
// - MacOS: os_sync_wait_on_address or __ulock_wait/__ulock_wake
// - FreeBSD: _umtx_op

#ifndef LLVM_LIBC_SRC_SUPPORT_THREADS_FUTEX_UTILS_H
#define LLVM_LIBC_SRC_SUPPORT_THREADS_FUTEX_UTILS_H

// TODO: implement futex for other platforms.
#ifdef __linux__
#include "src/__support/threads/linux/futex_utils.h"
#else
#error "Unsupported platform"
#endif

#endif // LLVM_LIBC_SRC_SUPPORT_THREADS_FUTEX_UTILS_H
