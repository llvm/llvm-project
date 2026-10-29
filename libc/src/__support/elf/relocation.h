//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Relative relocations for a loaded ELF module
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_RELOCATION_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_RELOCATION_H

#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/elf/elf_types.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// The relocation type that means "add the load bias to the addend and store
// it", which is the only kind a module can apply to itself before any symbol
// lookup is possible.
#if defined(LIBC_TARGET_ARCH_IS_X86_64)
constexpr uint32_t RELATIVE_RELOC = R_X86_64_RELATIVE;
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
constexpr uint32_t RELATIVE_RELOC = R_AARCH64_RELATIVE;
#elif defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
constexpr uint32_t RELATIVE_RELOC = R_RISCV_RELATIVE;
#elif defined(LIBC_TARGET_ARCH_IS_ARM)
constexpr uint32_t RELATIVE_RELOC = R_ARM_RELATIVE;
#else
// No loader runs on this architecture, so nothing applies relocations. The
// value matches no relocation type.
constexpr uint32_t RELATIVE_RELOC = 0xffffffffu;
#endif

// ELF64_R_TYPE and ELF32_R_TYPE differ, and ElfW() picks the class, so this
// keeps the choice in one place.
LIBC_INLINE constexpr uint32_t reloc_type(Xword info) {
#ifdef __LP64__
  return static_cast<uint32_t>(ELF64_R_TYPE(info));
#else
  return static_cast<uint32_t>(ELF32_R_TYPE(info));
#endif
}

// A view over a DT_RELA table, sized by DT_RELASZ.
class RelaTable {
public:
  LIBC_INLINE constexpr RelaTable() = default;
  LIBC_INLINE constexpr RelaTable(const ElfW(Rela) * entries, size_t size_bytes)
      : entries_(entries), count_(size_bytes / sizeof(ElfW(Rela))) {}

  LIBC_INLINE constexpr size_t size() const { return count_; }
  LIBC_INLINE constexpr bool empty() const { return count_ == 0; }
  LIBC_INLINE constexpr const ElfW(Rela) * begin() const { return entries_; }
  LIBC_INLINE constexpr const ElfW(Rela) * end() const {
    return entries_ + count_;
  }

  // Applies every relative relocation in the table, and returns how many were
  // applied. Other relocation types are left alone: they need a symbol table,
  // which is not available until this pass has run.
  //
  // This runs before the module is relocated, so it must not call anything
  // through the GOT or PLT.
  LIBC_INLINE size_t apply_relative(ElfW(Addr) load_bias) const {
    size_t applied = 0;
    for (size_t i = 0; i < count_; ++i) {
      const ElfW(Rela) &rela = entries_[i];
      if (reloc_type(rela.r_info) != RELATIVE_RELOC)
        continue;
      auto *target = reinterpret_cast<ElfW(Addr) *>(load_bias + rela.r_offset);
      *target = load_bias + static_cast<ElfW(Addr)>(rela.r_addend);
      ++applied;
    }
    return applied;
  }

private:
  const ElfW(Rela) *entries_ = nullptr;
  size_t count_ = 0;
};

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_RELOCATION_H
