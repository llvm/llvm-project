//===- ELFCreator.cpp -----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a class to create an ELF file in memory. This is
// supposed to be used for "-format binary" option.
//
//===----------------------------------------------------------------------===//

#include "ELFCreator.h"
#include "Config.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

using namespace lld;
using namespace lld::elf;

namespace {
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
  void addSymbol(StringRef Name, uintX_t SecIdx, uintX_t Value);
  size_t layout();
  void writeTo(uint8_t *Out);

private:
  Elf_Ehdr Header = {};
  std::vector<Elf_Shdr *> Sections;
  std::vector<Elf_Sym *> Symbols;
  StringTableBuilder StrTabBuilder{StringTableBuilder::ELF};
  BumpPtrAllocator Alloc;
  StringSaver Saver{Alloc};
  Elf_Shdr *StrTab;
  Elf_Shdr *SymTab;
};
}

template <class ELFT>
ELFCreator<ELFT>::ELFCreator(std::uint16_t Type, std::uint16_t Machine) {
  std::memcpy(Header.e_ident, "\177ELF", 4);
  Header.e_ident[EI_CLASS] = ELFT::Is64Bits ? ELFCLASS64 : ELFCLASS32;
  Header.e_ident[EI_DATA] =
      ELFT::TargetEndianness == support::little ? ELFDATA2LSB : ELFDATA2MSB;
  Header.e_ident[EI_VERSION] = EV_CURRENT;
  Header.e_type = Type;
  Header.e_machine = Machine;
  Header.e_version = EV_CURRENT;
  Header.e_ehsize = sizeof(Elf_Ehdr);
  Header.e_shentsize = sizeof(Elf_Shdr);
  Header.e_shstrndx = 1;

  StrTab = addSection(".strtab").Header;
  StrTab->sh_type = SHT_STRTAB;
  StrTab->sh_addralign = 1;

  SymTab = addSection(".symtab").Header;
  SymTab->sh_type = SHT_SYMTAB;
  SymTab->sh_link = 1;
  SymTab->sh_info = 1;
  SymTab->sh_addralign = sizeof(uintX_t);
  SymTab->sh_entsize = sizeof(Elf_Sym);
}

template <class ELFT>
typename ELFCreator<ELFT>::Section
ELFCreator<ELFT>::addSection(StringRef Name) {
  auto *Shdr = new (Alloc) Elf_Shdr{};
  Shdr->sh_name = StrTabBuilder.add(Saver.save(Name));
  Sections.push_back(Shdr);
  return {Shdr, Sections.size()};
}

template <class ELFT>
void ELFCreator<ELFT>::addSymbol(StringRef Name, uintX_t SecIdx,
                                 uintX_t Value) {
  auto *Sym = new (Alloc) Elf_Sym{};
  Sym->st_name = StrTabBuilder.add(Saver.save(Name));
  Sym->setBindingAndType(STB_GLOBAL, STT_OBJECT);
  Sym->st_shndx = SecIdx;
  Sym->st_value = Value;
  Symbols.push_back(Sym);
}

template <class ELFT> size_t ELFCreator<ELFT>::layout() {
  StrTabBuilder.finalizeInOrder();
  StrTab->sh_size = StrTabBuilder.getSize();
  SymTab->sh_size = (Symbols.size() + 1) * sizeof(Elf_Sym);

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

template <class ELFT> void ELFCreator<ELFT>::writeTo(uint8_t *Out) {
  std::memcpy(Out, &Header, sizeof(Elf_Ehdr));
  StrTabBuilder.write(Out + StrTab->sh_offset);

  Elf_Sym *Sym = reinterpret_cast<Elf_Sym *>(Out + SymTab->sh_offset);
  // Skip null.
  ++Sym;
  for (Elf_Sym *S : Symbols)
    *Sym++ = *S;

  Elf_Shdr *Shdr = reinterpret_cast<Elf_Shdr *>(Out + Header.e_shoff);
  // Skip null.
  ++Shdr;
  for (Elf_Shdr *S : Sections)
    *Shdr++ = *S;
}

template <class ELFT>
std::vector<uint8_t> elf::wrapBinaryWithElfHeader(ArrayRef<uint8_t> Blob,
                                                  std::string Filename) {
  // Fill the ELF file header.
  ELFCreator<ELFT> File(ET_REL, Config->EMachine);
  typename ELFCreator<ELFT>::Section Sec = File.addSection(".data");
  Sec.Header->sh_flags = SHF_ALLOC;
  Sec.Header->sh_size = Blob.size();
  Sec.Header->sh_type = SHT_PROGBITS;
  Sec.Header->sh_addralign = 8;

  // Replace non-alphanumeric characters with '_'.
  std::transform(Filename.begin(), Filename.end(), Filename.begin(),
                 [](char C) { return isalnum(C) ? C : '_'; });

  // Add _start, _end and _size symbols.
  File.addSymbol("_binary_" + Filename + "_start", Sec.Index, 0);
  File.addSymbol("_binary_" + Filename + "_end", Sec.Index, Blob.size());
  File.addSymbol("_binary_" + Filename + "_size", SHN_ABS, Blob.size());

  // Fix the ELF file layout and write it down to a uint8_t vector.
  size_t Size = File.layout();
  std::vector<uint8_t> Ret(Size);
  File.writeTo(Ret.data());

  // Fill .data section with actual data.
  memcpy(Ret.data() + Sec.Header->sh_offset, Blob.data(), Blob.size());
  return Ret;
}

template std::vector<uint8_t>
    elf::wrapBinaryWithElfHeader<ELF32LE>(ArrayRef<uint8_t>, std::string);
template std::vector<uint8_t>
    elf::wrapBinaryWithElfHeader<ELF32BE>(ArrayRef<uint8_t>, std::string);
template std::vector<uint8_t>
    elf::wrapBinaryWithElfHeader<ELF64LE>(ArrayRef<uint8_t>, std::string);
template std::vector<uint8_t>
    elf::wrapBinaryWithElfHeader<ELF64BE>(ArrayRef<uint8_t>, std::string);
