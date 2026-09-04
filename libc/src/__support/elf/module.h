//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Description of one loaded ELF module
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_MODULE_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_MODULE_H

#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/elf/dynamic.h"
#include "src/__support/elf/elf_types.h"
#include "src/__support/elf/relocation.h"
#include "src/__support/elf/symbol.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// Everything about one loaded ELF object that a consumer needs in order to
// resolve a symbol in it or to describe it to a debugger.
//
// This is deliberately plain data with no ownership and no mutable state. It
// is built once from a module's program headers and then only read, so it can
// be published to libdl and to a debugger without any of them needing to call
// back into the code that produced it.
class Module {
public:
  LIBC_INLINE constexpr Module() = default;

  // Builds the description from a module's program headers. `load_bias` is the
  // distance between where the module was linked and where it was mapped.
  LIBC_INLINE Module(const ElfW(Phdr) * phdrs, ElfW(Half) phnum,
                     ElfW(Addr) load_bias, const char *name)
      : phdrs_(phdrs), phnum_(phnum), load_bias_(load_bias), name_(name) {
    for (ElfW(Half) i = 0; i < phnum; ++i) {
      switch (phdrs[i].p_type) {
      case PT_DYNAMIC:
        dynamic_ = DynamicTable(
            reinterpret_cast<const ElfW(Dyn) *>(load_bias + phdrs[i].p_vaddr),
            load_bias);
        break;
      case PT_TLS:
        tls_ = &phdrs[i];
        break;
      default:
        break;
      }
    }
  }

  LIBC_INLINE constexpr const char *name() const { return name_; }
  LIBC_INLINE constexpr ElfW(Addr) load_bias() const { return load_bias_; }
  LIBC_INLINE constexpr const ElfW(Phdr) * phdrs() const { return phdrs_; }
  LIBC_INLINE constexpr ElfW(Half) phnum() const { return phnum_; }
  LIBC_INLINE constexpr const DynamicTable &dynamic() const { return dynamic_; }

  // The PT_TLS header, or null if the module has no thread local storage.
  // A loader has to size the thread block from every module's PT_TLS, which
  // is why this is exposed rather than kept private.
  LIBC_INLINE constexpr const ElfW(Phdr) * tls() const { return tls_; }

  LIBC_INLINE const char *strtab() const {
    if (auto addr = dynamic_.address(DT_STRTAB))
      return reinterpret_cast<const char *>(*addr);
    return nullptr;
  }

  LIBC_INLINE const ElfW(Sym) * symtab() const {
    if (auto addr = dynamic_.address(DT_SYMTAB))
      return reinterpret_cast<const ElfW(Sym) *>(*addr);
    return nullptr;
  }

  // The module's SONAME, or null if it has none. An executable normally does
  // not carry one.
  LIBC_INLINE const char *soname() const {
    const char *strings = strtab();
    auto offset = dynamic_.value(DT_SONAME);
    if (strings == nullptr || !offset)
      return nullptr;
    return strings + *offset;
  }

  LIBC_INLINE GnuSymbolTable symbols() const {
    auto hash = dynamic_.address(DT_GNU_HASH);
    if (!hash)
      return GnuSymbolTable();
    return GnuSymbolTable(reinterpret_cast<const void *>(*hash), symtab(),
                          strtab());
  }

  // The DT_RELA table. DT_JMPREL is kept separate because a loader may bind
  // it lazily while the rest has to be applied up front.
  LIBC_INLINE RelaTable relocations() const {
    auto addr = dynamic_.address(DT_RELA);
    auto size = dynamic_.value(DT_RELASZ);
    if (!addr || !size)
      return RelaTable();
    return RelaTable(reinterpret_cast<const ElfW(Rela) *>(*addr), *size);
  }

  LIBC_INLINE RelaTable plt_relocations() const {
    auto addr = dynamic_.address(DT_JMPREL);
    auto size = dynamic_.value(DT_PLTRELSZ);
    // DT_PLTREL says which form DT_JMPREL takes; only RELA is supported.
    auto form = dynamic_.value(DT_PLTREL);
    if (!addr || !size || !form || *form != Xword(DT_RELA))
      return RelaTable();
    return RelaTable(reinterpret_cast<const ElfW(Rela) *>(*addr), *size);
  }

  // Calls callback(const char *) for each DT_NEEDED name, in the order the
  // linker recorded them, which is the order they must be loaded in.
  template <typename F> LIBC_INLINE void for_each_needed(F callback) const {
    const char *strings = strtab();
    if (strings == nullptr)
      return;
    dynamic_.for_each(DT_NEEDED, [strings, &callback](const ElfW(Dyn) &entry) {
      callback(strings + entry.d_un.d_val);
    });
  }

private:
  const ElfW(Phdr) *phdrs_ = nullptr;
  ElfW(Half) phnum_ = 0;
  ElfW(Addr) load_bias_ = 0;
  const char *name_ = nullptr;
  const ElfW(Phdr) *tls_ = nullptr;
  DynamicTable dynamic_;
};

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_MODULE_H
