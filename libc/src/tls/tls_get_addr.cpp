//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of __tls_get_addr
///
//===----------------------------------------------------------------------===//

#include "src/tls/tls_get_addr.h"

#include "hdr/stdint_proxy.h"
#include "src/__support/common.h"
#include "src/__support/elf/dynamic_blocks.h"
#include "src/__support/elf/passive_abi.h"
#include "src/__support/elf/thread_pointer.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// Where a thread local lives for the thread which is asking.
//
// A shared library cannot know at link time where in a thread's blocks its
// own will sit, so its thread locals are compiled to name the module and an
// offset within that module's block and to call this.
//
// A module loaded at startup has a place in the one block each thread is
// given, at an offset the loader worked out and left behind in its record, so
// the answer is a sum. One opened later has no such place, because the
// threads that already exist were given blocks with no room for it, and takes
// a block of its own that this thread allocates the first time it asks.
LLVM_LIBC_FUNCTION(void *, __tls_get_addr, (TlsIndex * index)) {
  if (index == nullptr)
    return nullptr;

  const elf::ModuleSet &modules = elf::loaded_modules();
  if (index->ti_module >= elf::module_count(modules))
    return nullptr;

  if (index->ti_module < modules.static_count) {
    const intptr_t *offsets =
        __atomic_load_n(&modules.tls_offsets, __ATOMIC_ACQUIRE);
    if (offsets == nullptr)
      return nullptr;
    const uintptr_t base = elf::thread_pointer();
    const intptr_t offset = offsets[index->ti_module];
    return reinterpret_cast<void *>(base + static_cast<uintptr_t>(offset) +
                                    index->ti_offset);
  }

  void *block = elf::dynamic_block_for(index->ti_module);
  if (block == nullptr)
    return nullptr;
  return reinterpret_cast<unsigned char *>(block) + index->ti_offset;
}

} // namespace LIBC_NAMESPACE_DECL
