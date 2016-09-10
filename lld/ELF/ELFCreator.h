//===- ELFCreator.h ---------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_ELF_CREATOR_H
#define LLD_ELF_ELF_CREATOR_H

#include "lld/Core/LLVM.h"

#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELFTypes.h"

namespace lld {
namespace elf {

template <class ELFT> class ELFCreator {
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Ehdr Elf_Ehdr;
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Sym Elf_Sym;

public:
  struct Section {
    Elf_Shdr *Header;
    std::size_t Index;
  };

  struct Symbol {
    Elf_Sym *Sym;
    std::size_t Index;
  };

  ELFCreator(std::uint16_t Type, std::uint16_t Machine);
  Section addSection(StringRef Name);
  Symbol addSymbol(StringRef Name);
  std::size_t layout();
  void write(uint8_t *Out);

private:
  Elf_Ehdr Header;
  std::vector<Elf_Shdr *> Sections;
  std::vector<Elf_Sym *> StaticSymbols;
  llvm::StringTableBuilder SecHdrStrTabBuilder{llvm::StringTableBuilder::ELF};
  llvm::StringTableBuilder StrTabBuilder{llvm::StringTableBuilder::ELF};
  llvm::BumpPtrAllocator Alloc;
  Elf_Shdr *ShStrTab;
  Elf_Shdr *StrTab;
  Elf_Shdr *SymTab;
};
}
}

#endif
