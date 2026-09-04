//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Binding a module's symbol relocations
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_BIND_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_BIND_H

#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/elf/elf_types.h"
#include "src/__support/elf/module.h"
#include "src/__support/elf/relocation.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// Relocation types that take a symbol value. Anything else is either relative,
// handled before this point, or unsupported.
#if defined(LIBC_TARGET_ARCH_IS_X86_64)
constexpr uint32_t RELOC_ABSOLUTE = R_X86_64_64;
constexpr uint32_t RELOC_GLOB_DAT = R_X86_64_GLOB_DAT;
constexpr uint32_t RELOC_JUMP_SLOT = R_X86_64_JUMP_SLOT;
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
constexpr uint32_t RELOC_ABSOLUTE = 257; // R_AARCH64_ABS64
constexpr uint32_t RELOC_GLOB_DAT = R_AARCH64_GLOB_DAT;
constexpr uint32_t RELOC_JUMP_SLOT = R_AARCH64_JUMP_SLOT;
#elif defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
constexpr uint32_t RELOC_ABSOLUTE = 2; // R_RISCV_64
constexpr uint32_t RELOC_GLOB_DAT = R_RISCV_JUMP_SLOT;
constexpr uint32_t RELOC_JUMP_SLOT = R_RISCV_JUMP_SLOT;
#elif defined(LIBC_TARGET_ARCH_IS_ARM)
constexpr uint32_t RELOC_ABSOLUTE = 2; // R_ARM_ABS32
constexpr uint32_t RELOC_GLOB_DAT = R_ARM_GLOB_DAT;
constexpr uint32_t RELOC_JUMP_SLOT = R_ARM_JUMP_SLOT;
#else
// No loader runs on this architecture, so nothing binds symbols. These match
// no relocation type.
constexpr uint32_t RELOC_ABSOLUTE = 0xffffffffu;
constexpr uint32_t RELOC_COPY = 0xfffffffeu;
constexpr uint32_t RELOC_IRELATIVE = 0xfffffffdu;
constexpr uint32_t RELOC_GLOB_DAT = 0xfffffffcu;
constexpr uint32_t RELOC_JUMP_SLOT = 0xfffffffbu;
constexpr uint32_t RELOC_TLS_OFFSET = 0xfffffffau;
constexpr uint32_t RELOC_TLS_MODULE = 0xfffffff9u;
constexpr uint32_t RELOC_TLS_MODULE_OFFSET = 0xfffffff8u;
constexpr uint32_t RELOC_TLS_DESCRIPTOR = 0xfffffff7u;
#endif

LIBC_INLINE constexpr unsigned char symbol_binding(unsigned char info) {
#ifdef __LP64__
  return static_cast<unsigned char>(ELF64_ST_BIND(info));
#else
  return static_cast<unsigned char>(ELF32_ST_BIND(info));
#endif
}

LIBC_INLINE constexpr uint32_t reloc_symbol(ElfW(Xword) info) {
#ifdef __LP64__
  return static_cast<uint32_t>(ELF64_R_SYM(info));
#else
  return static_cast<uint32_t>(ELF32_R_SYM(info));
#endif
}

// An ordered set of modules, searched in load order the way a dynamic loader
// does. The first definition found wins, which is what makes interposition
// work.
class SearchOrder {
public:
  LIBC_INLINE constexpr SearchOrder(const Module *modules, size_t count)
      : modules_(modules), count_(count) {}

  // Returns the runtime address of `name`, or nothing if no module defines it.
  LIBC_INLINE cpp::optional<ElfW(Addr)> resolve(const char *name) const {
    for (size_t i = 0; i < count_; ++i) {
      const ElfW(Sym) *symbol = modules_[i].symbols().lookup(name);
      if (symbol == nullptr)
        continue;
      // A symbol the module only references rather than defines has no
      // section, and must not satisfy the lookup.
      if (symbol->st_shndx == SHN_UNDEF)
        continue;
      return modules_[i].load_bias() + symbol->st_value;
    }
    return cpp::nullopt;
  }

private:
  const Module *modules_;
  size_t count_;
};

struct BindResult {
  size_t bound = 0;
  size_t unresolved = 0;
};

// Applies one table of symbol relocations against the search order.
//
// A strong symbol that cannot be found is counted rather than fatal, so the
// caller decides whether to fail the load.
LIBC_INLINE BindResult bind_relocations(const Module &target,
                                        const RelaTable &table,
                                        const SearchOrder &order) {
  BindResult result;
  const ElfW(Sym) *symtab = target.symtab();
  const char *strtab = target.strtab();
  if (symtab == nullptr || strtab == nullptr)
    return result;

  for (const ElfW(Rela) *rela = table.begin(); rela != table.end(); ++rela) {
    const uint32_t type = reloc_type(rela->r_info);
    if (type != RELOC_ABSOLUTE && type != RELOC_GLOB_DAT &&
        type != RELOC_JUMP_SLOT)
      continue;

    const ElfW(Sym) &symbol = symtab[reloc_symbol(rela->r_info)];
    const char *name = strtab + symbol.st_name;
    auto found = order.resolve(name);

    // A weak undefined symbol is allowed to stay unresolved and is written as
    // zero, which is what lets a module test for an optional dependency.
    ElfW(Addr) value = 0;
    if (found) {
      value = *found;
    } else if (symbol_binding(symbol.st_info) != STB_WEAK) {
      ++result.unresolved;
      continue;
    }

    auto *where =
        reinterpret_cast<ElfW(Addr) *>(target.load_bias() + rela->r_offset);
    // Only the absolute form carries an addend that matters here; the GOT and
    // PLT slots take the symbol value on its own.
    *where = (type == RELOC_ABSOLUTE)
                 ? value + static_cast<ElfW(Addr)>(rela->r_addend)
                 : value;
    ++result.bound;
  }
  return result;
}

// Binds both a module's ordinary and PLT relocations.
LIBC_INLINE BindResult bind_module(const Module &target,
                                   const SearchOrder &order) {
  BindResult a = bind_relocations(target, target.relocations(), order);
  BindResult b = bind_relocations(target, target.plt_relocations(), order);
  return BindResult{a.bound + b.bound, a.unresolved + b.unresolved};
}

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_BIND_H
