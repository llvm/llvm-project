//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A thread's dynamic thread vector
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_DTV_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_DTV_H

#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/sys_mman_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/mmap.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/munmap.h"
#include "src/__support/elf/module.h"
#include "src/__support/elf/passive_abi.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// Where a thread keeps the blocks for modules that were not there when it was
// given its own.
//
// Everything loaded at startup has a place in the one block each thread gets,
// at an offset fixed when the process was linked. A module opened later has
// no such place: the threads that already exist were given blocks with no
// room for it. So each thread grows this instead, a block at a time, the
// first time it reaches for one of that module's thread locals.
//
// A block stays where it is for as long as its module is loaded. The address
// of a thread local can be kept anywhere, a destructor registered to run as
// the thread ends among them, so the only block given back early is one made
// for a module that has since been closed and had its index given to another.
//
// The storage comes from the kernel rather than from malloc. A thread gives
// these back as it ends, which is also when the allocator is taking its own
// per thread state apart, and the order of the two is not something either
// should have to know about.
class DynamicThreadVector {
public:
  // The block for `module_index` in this thread, allocating and filling it in
  // from the module's initial image if this is the first time. Null if there
  // is no room to be had or the module has no thread local storage.
  LIBC_INLINE void *block_for(const ModuleSet &modules, size_t module_index) {
    if (module_index >= module_count(modules))
      return nullptr;
    if (module_index < count_) {
      Entry &entry = entries_[module_index];
      if (entry.storage != nullptr) {
        const MappedModule *mappings =
            __atomic_load_n(&modules.mappings, __ATOMIC_ACQUIRE);
        if (entry.generation >= mappings[module_index].generation)
          return entry.block;
        // Made before the module now at this index was opened, so for the
        // one that was there before it.
        linux_syscalls::munmap(entry.storage, entry.size);
        entry = Entry{};
      }
    }
    return make_block(modules, module_index);
  }

  // Gives back everything this thread took. Called as the thread ends.
  LIBC_INLINE void release() {
    for (size_t i = 0; i < count_; ++i)
      if (entries_[i].storage != nullptr)
        linux_syscalls::munmap(entries_[i].storage, entries_[i].size);
    if (entries_ != nullptr)
      linux_syscalls::munmap(entries_, capacity_ * sizeof(Entry));
    entries_ = nullptr;
    count_ = 0;
    capacity_ = 0;
  }

private:
  struct Entry {
    // What the kernel gave, and what it was asked for, so it can be given
    // back.
    void *storage;
    size_t size;
    // Where within it the module's thread locals begin, which is the first
    // point far enough in to have the alignment they need.
    void *block;
    // The generation of the module set when this was made.
    size_t generation;
  };

  LIBC_INLINE static void *map(size_t size) {
    auto result = linux_syscalls::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (!result.has_value())
      return nullptr;
    return result.value();
  }

  LIBC_INLINE bool reserve(size_t wanted) {
    if (wanted <= capacity_)
      return true;
    size_t next = capacity_ == 0 ? 16 : capacity_ * 2;
    while (next < wanted)
      next *= 2;
    auto *grown = static_cast<Entry *>(map(next * sizeof(Entry)));
    if (grown == nullptr)
      return false;
    // Fresh anonymous memory is already zero, so only what is in use has to
    // be carried over.
    for (size_t i = 0; i < count_; ++i)
      grown[i] = entries_[i];
    if (entries_ != nullptr)
      linux_syscalls::munmap(entries_, capacity_ * sizeof(Entry));
    entries_ = grown;
    capacity_ = next;
    return true;
  }

  LIBC_INLINE void *make_block(const ModuleSet &modules, size_t module_index) {
    if (module_index >= module_count(modules))
      return nullptr;
    const Module *loaded = __atomic_load_n(&modules.modules, __ATOMIC_ACQUIRE);
    const ElfW(Phdr) *tls = loaded[module_index].tls();
    if (tls == nullptr)
      return nullptr;

    if (!reserve(module_index + 1))
      return nullptr;
    if (module_index >= count_)
      count_ = module_index + 1;

    // What the kernel gives back is page aligned, which covers any alignment
    // a thread local can ask for, but the request is padded anyway so that a
    // larger one would still be met.
    const size_t alignment =
        tls->p_align < 1 ? 1 : static_cast<size_t>(tls->p_align);
    const size_t size = tls->p_memsz + alignment;
    void *storage = map(size);
    if (storage == nullptr)
      return nullptr;

    uintptr_t start = reinterpret_cast<uintptr_t>(storage);
    start = (start + alignment - 1) & ~(alignment - 1);
    auto *block = reinterpret_cast<unsigned char *>(start);

    // The .tdata is copied in and the .tbss remainder is left zero, which the
    // fresh mapping already is.
    if (tls->p_filesz != 0) {
      const void *image = reinterpret_cast<const void *>(
          loaded[module_index].load_bias() + tls->p_vaddr);
      inline_memcpy(block, image, tls->p_filesz);
    }

    entries_[module_index] =
        Entry{storage, size, block,
              __atomic_load_n(&modules.generation, __ATOMIC_ACQUIRE)};
    return block;
  }

  // Indexed the same as the module set, holding what this thread took for
  // each module that needed it.
  Entry *entries_ = nullptr;
  size_t count_ = 0;
  size_t capacity_ = 0;
};

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_DTV_H
