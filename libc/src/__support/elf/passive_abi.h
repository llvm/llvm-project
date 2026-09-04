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

// How many objects a process may have loaded at once, startup and dlopen
// together.
constexpr size_t MAX_PROCESS_MODULES = 64;

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

extern "C" [[gnu::visibility("default")]]
LIBC_NAMESPACE::elf::ModuleSet __llvm_libc_process_modules;

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// The same object, under the name the rest of libc uses.
LIBC_INLINE ModuleSet &loaded_modules() { return __llvm_libc_process_modules; }

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_PASSIVE_ABI_H
