//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// What the loader publishes about the loaded modules
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_PASSIVE_ABI_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_PASSIVE_ABI_H

#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/elf/map_segments.h"
#include "src/__support/elf/module.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// How many objects the startup linker can load. It runs before there is an
// allocator to ask, so its storage is fixed and lives on the stack the
// kernel handed the process. dlopen grows the set past this onto the heap.
//
// A Qt program is already past sixty libraries before any of its own are
// counted, and a desktop built on it goes further (Spectacle links nearly
// three hundred), so the room here is for that rather than for the handful a
// small program needs.
constexpr size_t MAX_PROCESS_MODULES = 512;

// The description of the loaded module set.
//
// The startup linker fills this in and then stops being involved. Everything
// afterwards, dlopen included, reads and extends it. Keeping it as plain data
// rather than a call back into the loader is what lets the startup code
// finish and go away: there is no interface to keep alive, only a record of
// what was loaded.
//
// The storage lives in libc rather than in the loader, because the loader is
// gone by the time anything else wants to look.
struct ModuleSet {
  Module *modules;
  MappedModule *mappings;
  intptr_t *tls_offsets;
  // Reference counts, so dlclose only unmaps what nothing else is using.
  // Objects loaded at startup begin at one and are never released.
  size_t *references;
  size_t count;
  // How many of those were loaded at startup. Those have a place in the block
  // each thread is given; anything past this was opened later and takes a
  // block of its own per thread.
  size_t static_count;
  // Bumped whenever a module is added or dropped after startup. Each module
  // opened later records the value its opening reached, and a thread's block
  // for an index made before then was made for a module that is gone.
  size_t generation;
  size_t capacity;
  size_t page_size;
  // Set once the startup linker has finished, so a caller can tell a linked
  // process from one that was started some other way.
  bool linked;
};

// Given C linkage and a stable name so the loader can find it by symbol
// lookup. It cannot link against libc: libc is one of the things it loads,
// and the loader has to be self contained to run before anything else.
constexpr const char *MODULE_SET_SYMBOL = "__llvm_libc_process_modules";

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

// Declared weak because the startup files refer to it and are also linked
// into programs built against a static libc, where nothing defines it. Such a
// program has no loader and no module set, and the reference resolves to
// nothing rather than failing the link.
extern "C" [[gnu::weak, gnu::visibility("default")]]
LIBC_NAMESPACE::elf::ModuleSet __llvm_libc_process_modules;

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// The same object, under the name the rest of libc uses. Only for code that
// is always linked with the definition, which is everything inside libc and
// the loader.
LIBC_INLINE ModuleSet &loaded_modules() { return __llvm_libc_process_modules; }

// Some readers of the set do not hold a lock: __tls_get_addr and the creation
// of a thread cannot wait on the loader. What they may rely on is that an
// array they have seen is never freed, only replaced by a larger one (see
// make_room in dlopen), and that an entry is written before the count which
// covers it. These are the two halves of that, for the code that changes the
// set and the code that reads it unlocked.
LIBC_INLINE size_t module_count(const ModuleSet &set) {
  return __atomic_load_n(&set.count, __ATOMIC_ACQUIRE);
}
LIBC_INLINE void set_module_count(ModuleSet &set, size_t count) {
  __atomic_store_n(&set.count, count, __ATOMIC_RELEASE);
}

// The record, or null where it was not linked in at all. Callers that may run
// in a statically linked program have to ask this way.
LIBC_INLINE ModuleSet *process_modules() {
  return &__llvm_libc_process_modules;
}

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_PASSIVE_ABI_H
