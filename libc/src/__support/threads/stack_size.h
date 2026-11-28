//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// OS-level default thread stack size dispatcher.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_STACK_SIZE_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_STACK_SIZE_H

#include "src/__support/macros/properties/os.h"

#if defined(LIBC_TARGET_OS_IS_LINUX)
#include "src/__support/threads/linux/stack_size.h"
#else
#error "Unsupported OS for thread stack size"
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_STACK_SIZE_H
