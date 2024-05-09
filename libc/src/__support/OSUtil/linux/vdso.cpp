//===------------- Linux VDSO Implementation --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/__support/OSUtil/linux/vdso.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/threads/callonce.h"
#include "src/__support/threads/linux/futex_word.h"
#include "src/errno/libc_errno.h"
#include <linux/auxvec.h>
#include <linux/elf.h>

#ifndef ElfW
#if __POINTER_WIDTH__ == 32
#define ElfW(type) Elf32_##type
#else
#define ElfW(type) Elf64_##type
#endif
#endif

namespace LIBC_NAMESPACE {

// we don't include getauxval.h as it may forcibly pull in elf.h (via
// sys/auxv.h) in overlay mode instead, we provide a separate declaration for
// getauxval
unsigned long getauxval(unsigned long id);

namespace vdso {

// See https://refspecs.linuxfoundation.org/LSB_1.3.0/gLSB/gLSB/symverdefs.html
struct Verdaux {
  ElfW(Word) vda_name; /* Version or dependency names */
  ElfW(Word) vda_next; /* Offset in bytes to next verdaux
                          entry */
};
struct Verdef {
  ElfW(Half) vd_version; /* Version revision */
  ElfW(Half) vd_flags;   /* Version information */
  ElfW(Half) vd_ndx;     /* Version Index */
  ElfW(Half) vd_cnt;     /* Number of associated aux entries */
  ElfW(Word) vd_hash;    /* Version name hash value */
  ElfW(Word) vd_aux;     /* Offset in bytes to verdaux array */
  ElfW(Word) vd_next;    /* Offset in bytes to next verdef
                            entry */
  Verdef *next() const {
    if (vd_next == 0)
      return nullptr;
    return reinterpret_cast<Verdef *>(reinterpret_cast<uintptr_t>(this) +
                                      vd_next);
  }
  Verdaux *aux() const {
    return reinterpret_cast<Verdaux *>(reinterpret_cast<uintptr_t>(this) +
                                       vd_aux);
  }
};

// version search procedure specified by
// https://refspecs.linuxfoundation.org/LSB_1.3.0/gLSB/gLSB/symversion.html#SYMVERTBL
cpp::string_view find_version(Verdef *verdef, ElfW(Half) * versym,
                              const char *strtab, size_t idx) {
  static constexpr ElfW(Half) VER_FLG_BASE = 0x1;
  ElfW(Half) identifier = versym[idx] & 0x7FFF;
  // iterate through all version definitions
  for (Verdef *def = verdef; def != nullptr; def = def->next()) {
    // skip if this is a file-level version
    if (def->vd_flags & VER_FLG_BASE)
      continue;
    // check if the version identifier matches
    if ((def->vd_ndx & 0x7FFF) == identifier) {
      Verdaux *aux = def->aux();
      return strtab + aux->vda_name;
    }
  }
  return "";
}

using VDSOArray =
    cpp::array<void *, static_cast<size_t>(VDSOSym::VDSOSymCount)>;

static VDSOArray symbol_table;

void *get_symbol(VDSOSym sym) {
  // if sym is invalid, return nullptr
  const size_t index = static_cast<size_t>(sym);
  if (index >= symbol_table.size())
    return nullptr;

  static FutexWordType once_flag = 0;
  callonce(reinterpret_cast<CallOnceFlag *>(&once_flag), [] {
    // first clear the symbol table
    for (auto &i : symbol_table) {
      i = nullptr;
    }

    // get the address of the VDSO, protect errno since getauxval may change it
    int errno_backup = libc_errno;
    uintptr_t vdso_ehdr_addr = getauxval(AT_SYSINFO_EHDR);
    // Get the memory address of the vDSO ELF header.
    auto vdso_ehdr = reinterpret_cast<ElfW(Ehdr) *>(vdso_ehdr_addr);
    // leave the table unpopulated if we don't have vDSO
    if (vdso_ehdr == nullptr) {
      libc_errno = errno_backup;
      return;
    }

    // count entries
    size_t symbol_count = 0;
    // locate the section header inside the elf using the section header offset
    auto vdso_shdr =
        reinterpret_cast<ElfW(Shdr) *>(vdso_ehdr_addr + vdso_ehdr->e_shoff);
    // iterate all sections until we locate the dynamic symbol section
    for (size_t i = 0; i < vdso_ehdr->e_shnum; ++i) {
      if (vdso_shdr[i].sh_type == SHT_DYNSYM) {
        // dynamic symbol section is a table section
        // therefore, the number of entries can be computed as the ratio
        // of the section size to the size of a single entry
        symbol_count = vdso_shdr[i].sh_size / vdso_shdr[i].sh_entsize;
        break;
      }
    }

    // early return if no symbol is found
    if (symbol_count == 0)
      return;

    // We need to find both the loadable segment and the dynamic linking of the
    // vDSO.
    auto vdso_addr = static_cast<ElfW(Addr)>(-1);
    ElfW(Dyn) *vdso_dyn = nullptr;
    // compute vdso_phdr as the program header using the program header offset
    ElfW(Phdr) *vdso_phdr =
        reinterpret_cast<ElfW(Phdr) *>(vdso_ehdr_addr + vdso_ehdr->e_phoff);
    // iterate through all the program headers until we get the desired pieces
    for (size_t i = 0; i < vdso_ehdr->e_phnum; ++i) {
      if (vdso_phdr[i].p_type == PT_DYNAMIC)
        vdso_dyn = reinterpret_cast<ElfW(Dyn) *>(vdso_ehdr_addr +
                                                 vdso_phdr[i].p_offset);

      if (vdso_phdr[i].p_type == PT_LOAD)
        vdso_addr =
            vdso_ehdr_addr + vdso_phdr[i].p_offset - vdso_phdr[i].p_vaddr;

      if (vdso_addr && vdso_dyn)
        break;
    }
    // early return if either the dynamic linking or the loadable segment is not
    // found
    if (vdso_dyn == nullptr || vdso_addr == static_cast<ElfW(Addr)>(-1))
      return;

    // now, locate several more tables inside the dynmaic linking section
    const char *strtab = nullptr;
    ElfW(Sym) *symtab = nullptr;
    ElfW(Half) *versym = nullptr;
    Verdef *verdef = nullptr;
    for (ElfW(Dyn) *d = vdso_dyn; d->d_tag != DT_NULL; ++d) {
      switch (d->d_tag) {
      case DT_STRTAB:
        strtab = reinterpret_cast<const char *>(vdso_addr + d->d_un.d_ptr);
        break;
      case DT_SYMTAB:
        symtab = reinterpret_cast<ElfW(Sym) *>(vdso_addr + d->d_un.d_ptr);
        break;
      case DT_VERSYM:
        versym = reinterpret_cast<uint16_t *>(vdso_addr + d->d_un.d_ptr);
        break;
      case DT_VERDEF:
        verdef = reinterpret_cast<Verdef *>(vdso_addr + d->d_un.d_ptr);
        break;
      }
      if (strtab && symtab && versym && verdef) {
        break;
      }
    }
    if (strtab == nullptr || symtab == nullptr)
      return;

    for (size_t i = 0; i < symbol_table.size(); ++i) {
      for (size_t j = 0; j < symbol_count; ++j) {
        auto sym = static_cast<VDSOSym>(i);
        if (symbol_name(sym) == strtab + symtab[j].st_name) {
          // we find a symbol with desired name
          // now we need to check if it has the right version
          if (versym && verdef)
            if (symbol_version(sym) != find_version(verdef, versym, strtab, j))
              continue;

          // put the symbol address into the symbol table
          symbol_table[i] =
              reinterpret_cast<void *>(vdso_addr + symtab[j].st_value);
        }
      }
    }
  });

  return symbol_table[index];
}
} // namespace vdso
} // namespace LIBC_NAMESPACE
