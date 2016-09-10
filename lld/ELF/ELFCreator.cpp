//===- ELFCreator.cpp -----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ELFCreator.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

using namespace lld;
using namespace lld::elf;

template <class ELFT>
ELFCreator<ELFT>::ELFCreator(std::uint16_t Type, std::uint16_t Machine) {
  std::memcpy(Header.e_ident, "\177ELF", 4);
  Header.e_ident[EI_CLASS] = ELFT::Is64Bits ? ELFCLASS64 : ELFCLASS32;
  Header.e_ident[EI_DATA] = ELFT::TargetEndianness == llvm::support::little
                                ? ELFDATA2LSB
                                : ELFDATA2MSB;
  Header.e_ident[EI_VERSION] = EV_CURRENT;
  Header.e_ident[EI_OSABI] = 0;
  Header.e_type = Type;
  Header.e_machine = Machine;
  Header.e_version = EV_CURRENT;
  Header.e_entry = 0;
  Header.e_phoff = 0;
  Header.e_flags = 0;
  Header.e_ehsize = sizeof(Elf_Ehdr);
  Header.e_phnum = 0;
  Header.e_shentsize = sizeof(Elf_Shdr);
  Header.e_shstrndx = 1;

  ShStrTab = addSection(".shstrtab").Header;
  ShStrTab->sh_type = SHT_STRTAB;
  ShStrTab->sh_addralign = 1;

  StrTab = addSection(".strtab").Header;
  StrTab->sh_type = SHT_STRTAB;
  StrTab->sh_addralign = 1;

  SymTab = addSection(".symtab").Header;
  SymTab->sh_type = SHT_SYMTAB;
  SymTab->sh_link = 2;
  SymTab->sh_info = 1;
  SymTab->sh_addralign = sizeof(uintX_t);
  SymTab->sh_entsize = sizeof(Elf_Sym);
}

template <class ELFT>
typename ELFCreator<ELFT>::Section
ELFCreator<ELFT>::addSection(StringRef Name) {
  std::size_t NameOff = SecHdrStrTabBuilder.add(Name);
  auto Shdr = new (Alloc) Elf_Shdr{};
  Shdr->sh_name = NameOff;
  Sections.push_back(Shdr);
  return {Shdr, Sections.size()};
}

template <class ELFT>
typename ELFCreator<ELFT>::Symbol ELFCreator<ELFT>::addSymbol(StringRef Name) {
  std::size_t NameOff = StrTabBuilder.add(Name);
  auto Sym = new (Alloc) Elf_Sym{};
  Sym->st_name = NameOff;
  StaticSymbols.push_back(Sym);
  return {Sym, StaticSymbols.size()};
}

template <class ELFT> std::size_t ELFCreator<ELFT>::layout() {
  SecHdrStrTabBuilder.finalizeInOrder();
  ShStrTab->sh_size = SecHdrStrTabBuilder.getSize();

  StrTabBuilder.finalizeInOrder();
  StrTab->sh_size = StrTabBuilder.getSize();

  SymTab->sh_size = (StaticSymbols.size() + 1) * sizeof(Elf_Sym);

  uintX_t Offset = sizeof(Elf_Ehdr);
  for (Elf_Shdr *Sec : Sections) {
    Offset = alignTo(Offset, Sec->sh_addralign);
    Sec->sh_offset = Offset;
    Offset += Sec->sh_size;
  }

  Offset = alignTo(Offset, sizeof(uintX_t));
  Header.e_shoff = Offset;
  Offset += (Sections.size() + 1) * sizeof(Elf_Shdr);
  Header.e_shnum = Sections.size() + 1;

  return Offset;
}

template <class ELFT> void ELFCreator<ELFT>::write(uint8_t *Out) {
  std::memcpy(Out, &Header, sizeof(Elf_Ehdr));
  std::copy(SecHdrStrTabBuilder.data().begin(),
            SecHdrStrTabBuilder.data().end(), Out + ShStrTab->sh_offset);
  std::copy(StrTabBuilder.data().begin(), StrTabBuilder.data().end(),
            Out + StrTab->sh_offset);

  Elf_Sym *Sym = reinterpret_cast<Elf_Sym *>(Out + SymTab->sh_offset);
  // Skip null.
  ++Sym;
  for (Elf_Sym *S : StaticSymbols)
    *Sym++ = *S;

  Elf_Shdr *Shdr = reinterpret_cast<Elf_Shdr *>(Out + Header.e_shoff);
  // Skip null.
  ++Shdr;
  for (Elf_Shdr *S : Sections)
    *Shdr++ = *S;
}

template class elf::ELFCreator<ELF32LE>;
template class elf::ELFCreator<ELF32BE>;
template class elf::ELFCreator<ELF64LE>;
template class elf::ELFCreator<ELF64BE>;
