//===-- Windows implementation of an exit function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"

// On Windows we cannot make direct syscalls since Microsoft changes system call
// IDs periodically. We must rely on functions exported from ntdll.dll or
// kernel32.dll to invoke system service procedures.
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

namespace LIBC_NAMESPACE_DECL {
namespace internal {

[[noreturn]] void exit(int status) { ::ExitProcess(status); }

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
