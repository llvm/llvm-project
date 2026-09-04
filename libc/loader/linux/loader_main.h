//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Entry point of the startup dynamic linker
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_LOADER_LINUX_LOADER_MAIN_H
#define LLVM_LIBC_LOADER_LINUX_LOADER_MAIN_H

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// Links the process and returns the address to transfer control to. The entry
// stub jumps there with the stack exactly as the kernel left it, which is what
// the program's own startup code expects to find.
uintptr_t loader_link(void *stack_pointer);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_LOADER_LINUX_LOADER_MAIN_H
