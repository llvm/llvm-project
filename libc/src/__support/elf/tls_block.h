//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Allocating a thread's static TLS block
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_TLS_BLOCK_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_TLS_BLOCK_H

#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/errno_macros.h"
#include "hdr/link_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/sys_mman_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getrandom.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/mmap.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/munmap.h"
#include "src/__support/elf/module.h"
#include "src/__support/elf/tls_layout.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// A thread's static TLS, with the thread pointer the ABI expects.
struct TlsBlock {
  void *storage = nullptr;
  size_t size = 0;
  uintptr_t thread_pointer = 0;
};

// Allocates and initialises the static TLS for one thread, covering every
// module in the list that has a PT_TLS segment.
//
// Sizing this from the main executable alone is the bug that makes a shared
// libc unusable: libc.so carries a PT_TLS segment of its own, and its thread
// local accesses land outside a block that did not account for it.
//
// Each module's .tdata is copied in and the .tbss remainder is left zero,
// which mmap already guarantees for fresh anonymous memory.
// `offsets`, when given, receives each module's offset from the thread
// pointer, indexed the same as `modules`. A module without TLS gets zero. A
// caller building a DTV needs these, and recomputing them would mean keeping
// the layout rules in two places.
// Copies each module's thread local initialisation image into a block laid
// out by allocate_tls_block.
//
// This is separate because the images are part of the modules' data and so
// are subject to their relocations: a thread local initialised with the
// address of anything is written by one. Copying before they are applied
// takes the value the linker left, which is not the address of anything.
LIBC_INLINE void copy_tls_images(const Module *modules, size_t count,
                                 uintptr_t tp, const intptr_t *offsets) {
  for (size_t i = 0; i < count; ++i) {
    const ElfW(Phdr) *tls = modules[i].tls();
    if (tls == nullptr || tls->p_filesz == 0)
      continue;
    auto *dest = reinterpret_cast<unsigned char *>(static_cast<intptr_t>(tp) +
                                                   offsets[i]);
    const void *image =
        reinterpret_cast<const void *>(modules[i].load_bias() + tls->p_vaddr);
    inline_memcpy(dest, image, tls->p_filesz);
    // The rest is .tbss and is already zero from the fresh mapping.
  }
}

LIBC_INLINE ErrorOr<TlsBlock> allocate_tls_block(const Module *modules,
                                                 size_t count,
                                                 intptr_t *offsets = nullptr,
                                                 bool copy_images = true) {
  TlsLayout layout;
  for (size_t i = 0; i < count; ++i) {
    const ElfW(Phdr) *tls = modules[i].tls();
    if (tls != nullptr)
      layout.add(tls->p_memsz, tls->p_align);
  }

  // Under variant 2 an empty layout still reports the control block size, so
  // the module count is what says whether anything needs allocating.
  if (layout.module_count() == 0) {
    for (size_t i = 0; i < count; ++i)
      if (offsets != nullptr)
        offsets[i] = 0;
    return TlsBlock{};
  }
  // Each module's offset is a multiple of its own alignment, which only
  // places it correctly if the thread pointer is a multiple of the strictest
  // one. Under variant 2 the pointer sits at the top, so the blocks below it
  // are padded until it does; the mapping itself is page aligned.
  const size_t below_tp =
      TLS_VARIANT_2
          ? round_up(layout.size() - TLS_TCB_RESERVE, layout.alignment())
          : 0;
  const size_t total =
      TLS_VARIANT_2 ? below_tp + TLS_TCB_RESERVE : layout.size();

  auto storage = linux_syscalls::mmap(nullptr, total, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (!storage.has_value())
    return Error(storage.error());

  const uintptr_t base = reinterpret_cast<uintptr_t>(storage.value());
  // Variant 2 puts the thread control block at the top with the module blocks
  // below it; variant 1 puts it at the bottom with the modules above.
  const uintptr_t tp = TLS_VARIANT_2 ? base + below_tp : base;

  // The ABI requires the first word of the thread control block to point at
  // the thread pointer itself. Compiler generated thread local accesses read
  // it to find their own base, so leaving it zero makes every one of them
  // dereference an offset from address zero.
  *reinterpret_cast<uintptr_t *>(tp) = tp;

  // Code built with -fstack-protector reads its guard from a fixed offset
  // from the thread pointer, so every thread needs one before any such code
  // runs. A failed read leaves the value at zero, which still compares equal
  // to itself and so costs correctness nothing.
  if (TLS_VARIANT_2) {
    auto *guard = reinterpret_cast<uintptr_t *>(tp + TLS_STACK_GUARD_OFFSET);
    if (!linux_syscalls::getrandom(guard, sizeof(uintptr_t), 0).has_value())
      *guard = 0;
  }

  TlsLayout again;
  for (size_t i = 0; i < count; ++i) {
    const ElfW(Phdr) *tls = modules[i].tls();
    if (tls == nullptr) {
      if (offsets != nullptr)
        offsets[i] = 0;
      continue;
    }
    const intptr_t offset = again.add(tls->p_memsz, tls->p_align);
    if (offsets != nullptr)
      offsets[i] = offset;
    if (!copy_images || tls->p_filesz == 0)
      continue;
    auto *dest =
        reinterpret_cast<unsigned char *>(static_cast<intptr_t>(tp) + offset);
    const void *image =
        reinterpret_cast<const void *>(modules[i].load_bias() + tls->p_vaddr);
    inline_memcpy(dest, image, tls->p_filesz);
    // The rest is .tbss and is already zero from the fresh mapping.
  }

  TlsBlock block;
  block.storage = storage.value();
  block.size = total;
  block.thread_pointer = tp;
  return block;
}

LIBC_INLINE void free_tls_block(const TlsBlock &block) {
  if (block.storage != nullptr)
    linux_syscalls::munmap(block.storage, block.size);
}

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_TLS_BLOCK_H
