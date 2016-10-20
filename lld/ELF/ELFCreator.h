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
#include "llvm/Support/StringSaver.h"

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
    size_t Index;
  };

  ELFCreator(std::uint16_t Type, std::uint16_t Machine);
  Section addSection(StringRef Name);
  Elf_Sym *addSymbol(StringRef Name);
  size_t layout();
  void writeTo(uint8_t *Out);

private:
  Elf_Ehdr Header = {};
  std::vector<Elf_Shdr *> Sections;
  std::vector<Elf_Sym *> Symbols;
  llvm::StringTableBuilder ShStrTabBuilder{llvm::StringTableBuilder::ELF};
  llvm::StringTableBuilder StrTabBuilder{llvm::StringTableBuilder::ELF};
  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver Saver{Alloc};
  Elf_Shdr *ShStrTab;
  Elf_Shdr *StrTab;
  Elf_Shdr *SymTab;
};
}
}

#endif
