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
#include "src/__support/elf/passive_abi.h"
#include "src/__support/elf/thread_pointer.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// Where a thread local lives for the thread which is asking.
//
// A shared library cannot know at link time where in a thread's blocks its
// own will sit, so its thread locals are compiled to name the module and an
// offset within that module's block and to call this. The module's offset
// from the thread pointer is what the loader worked out when it laid the
// blocks out, and it is in the record the loader left behind.
//
// Every module here is one loaded at startup, so they all have a place in
// the one block each thread is given and the answer is a sum. A module
// brought in later by dlopen would need a block of its own, allocated the
// first time each thread asked for it; that is not supported yet, and such
// a module's index is not in the record, so this reports it as nothing
// rather than reading past the end.
LLVM_LIBC_FUNCTION(void *, __tls_get_addr, (TlsIndex * index)) {
  if (index == nullptr)
    return nullptr;

  const elf::ModuleSet &modules = elf::loaded_modules();
  if (modules.tls_offsets == nullptr || index->ti_module >= modules.count)
    return nullptr;

  const uintptr_t base = elf::thread_pointer();
  const intptr_t offset = modules.tls_offsets[index->ti_module];
  return reinterpret_cast<void *>(base + static_cast<uintptr_t>(offset) +
                                  index->ti_offset);
}

} // namespace LIBC_NAMESPACE_DECL
