//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of dladdr
///
//===----------------------------------------------------------------------===//

#include "dladdr.h"

#include "src/__support/CPP/mutex.h"
#include "src/__support/common.h"
#include "src/__support/elf/passive_abi.h"
#include "src/__support/macros/config.h"
#include "src/dlfcn/dl_internal.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

// The symbol an address belongs to is the one defined closest below it. A
// symbol with a size covers only that much; one with none, which is what an
// assembly routine often has, covers up to whatever is defined next.
const ElfW(Sym) * enclosing_symbol(const elf::Module &module,
                                   ElfW(Addr) address, const char **name) {
  const elf::GnuSymbolTable table = module.symbols();
  const uint32_t count = table.count();
  const ElfW(Sym) *best = nullptr;
  ElfW(Addr) best_value = 0;

  for (uint32_t i = 0; i < count; ++i) {
    const ElfW(Sym) *symbol = table.symbol(i);
    if (symbol->st_shndx == SHN_UNDEF || symbol->st_name == 0)
      continue;
    const ElfW(Addr) value = module.load_bias() + symbol->st_value;
    if (value > address)
      continue;
    if (symbol->st_size != 0 && address >= value + symbol->st_size)
      continue;
    if (best == nullptr || value > best_value) {
      best = symbol;
      best_value = value;
      *name = table.name(i);
    }
  }
  return best;
}

} // anonymous namespace

// Says which object an address is in, and which symbol of it. Unlike the rest
// of dlfcn this reports failure with zero rather than with a message, and it
// leaves dlerror alone, which is what every other implementation does.
namespace internal {

int dladdr_impl(const void *__restrict addr, Dl_info *__restrict info) {
  if (info == nullptr)
    return 0;

  elf::ModuleSet *modules = elf::process_modules();
  cpp::lock_guard lock(dl::dl_mutex);
  if (modules == nullptr || !modules->linked)
    return 0;
  elf::ModuleSet &set = *modules;

  const ElfW(Addr) address = reinterpret_cast<ElfW(Addr)>(addr);
  const elf::Module *module = nullptr;
  for (size_t i = 0; i < set.count; ++i) {
    if (set.modules[i].contains(address)) {
      module = &set.modules[i];
      break;
    }
  }
  if (module == nullptr)
    return 0;

  info->dli_fname = module->name();
  info->dli_fbase = reinterpret_cast<void *>(module->load_bias());
  info->dli_sname = nullptr;
  info->dli_saddr = nullptr;

  const char *name = nullptr;
  if (const ElfW(Sym) *symbol = enclosing_symbol(*module, address, &name)) {
    info->dli_sname = name;
    info->dli_saddr =
        reinterpret_cast<void *>(module->load_bias() + symbol->st_value);
  }
  return 1;
}

} // namespace internal

LLVM_LIBC_FUNCTION(int, dladdr,
                   (const void *__restrict addr, Dl_info *__restrict info)) {
  return internal::dladdr_impl(addr, info);
}

} // namespace LIBC_NAMESPACE_DECL
