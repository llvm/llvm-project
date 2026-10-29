//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A thread's blocks for modules opened later
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_DYNAMIC_BLOCKS_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_DYNAMIC_BLOCKS_H

#include "hdr/types/size_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// The calling thread's block for `module_index`, allocated and filled in from
// the module's initial image if this is the first time it has asked. Null if
// there is no room to be had or the module has no thread local storage.
void *dynamic_block_for(size_t module_index);

// Gives back what the calling thread allocated for modules opened after it
// started. Called as the thread ends.
void release_dynamic_thread_blocks();

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_DYNAMIC_BLOCKS_H
