//===-- Implementation of dlsym -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dlsym.h"

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/mutex.h"
#include "src/__support/common.h"
#include "src/__support/elf/bind.h"
#include "src/__support/elf/passive_abi.h"
#include "src/__support/macros/config.h"
#include "src/dlfcn/dl_internal.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

// <dlfcn.h> states the two of these as values rather than in a header this
// can include, so they are named here rather than spelled out at the tests
// below.
constexpr void *DEFAULT_HANDLE = nullptr;
constexpr uintptr_t NEXT_HANDLE = static_cast<uintptr_t>(-1l);

// The first module at or after `from` which defines `name`. Returns nothing
// when none does.
void *search_from(const elf::ModuleSet &set, size_t from, const char *name) {
  for (size_t i = from; i < set.count; ++i) {
    const ElfW(Sym) *symbol = set.modules[i].symbols().lookup(name);
    if (symbol == nullptr || symbol->st_shndx == SHN_UNDEF)
      continue;
    return reinterpret_cast<void *>(set.modules[i].load_bias() +
                                    symbol->st_value);
  }
  return nullptr;
}

} // anonymous namespace

LLVM_LIBC_FUNCTION(void *, dlsym,
                   (void *__restrict handle, const char *__restrict name)) {
  // Which module the call is returning to, for RTLD_NEXT. Taken before
  // anything else, so nothing stands between this and the caller.
  const ElfW(Addr) caller =
      reinterpret_cast<ElfW(Addr)>(__builtin_return_address(0));

  elf::ModuleSet *modules = elf::process_modules();
  cpp::lock_guard lock(dl::dl_mutex);

  if (modules == nullptr || !modules->linked) {
    dl::set_error("dlsym is only available in a dynamically linked process");
    return nullptr;
  }
  elf::ModuleSet &set = *modules;
  if (name == nullptr) {
    dl::set_error("dlsym was given no symbol name");
    return nullptr;
  }

  // Every module, in the order they were loaded, which is the order a lookup
  // through the program's own scope goes in.
  if (handle == DEFAULT_HANDLE) {
    if (void *found = search_from(set, 0, name))
      return found;
    dl::set_error("symbol not found");
    return nullptr;
  }

  // Everything after whatever called, which is how a wrapper reaches the
  // definition it is standing in front of.
  if (reinterpret_cast<uintptr_t>(handle) == NEXT_HANDLE) {
    size_t after = set.count;
    for (size_t i = 0; i < set.count; ++i) {
      if (set.modules[i].contains(caller)) {
        after = i + 1;
        break;
      }
    }
    if (void *found = search_from(set, after, name))
      return found;
    dl::set_error("symbol not found");
    return nullptr;
  }

  size_t index = 0;
  if (!dl::index_for(handle, index)) {
    dl::set_error("dlsym was given a handle that dlopen did not return");
    return nullptr;
  }

  const elf::Module &module = set.modules[index];
  const ElfW(Sym) *symbol = module.symbols().lookup(name);
  // A symbol the object only references does not count as finding it.
  if (symbol == nullptr || symbol->st_shndx == SHN_UNDEF) {
    dl::set_error("symbol not found");
    return nullptr;
  }
  return reinterpret_cast<void *>(module.load_bias() + symbol->st_value);
}

} // namespace LIBC_NAMESPACE_DECL
