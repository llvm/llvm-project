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

#include "src/__support/elf/dynamic_blocks.h"

#include "src/__support/elf/dtv.h"
#include "src/__support/elf/passive_abi.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

namespace {

// It is itself a thread local of libc, which was loaded at startup, so it
// lives in the block every thread already has and costs a thread that never
// opens anything nothing at all.
LIBC_THREAD_LOCAL DynamicThreadVector blocks;

} // anonymous namespace

void *dynamic_block_for(size_t module_index) {
  return blocks.block_for(loaded_modules(), module_index);
}

void release_dynamic_thread_blocks() { blocks.release(); }

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL
