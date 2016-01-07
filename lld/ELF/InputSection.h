//===- InputSection.h -------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_INPUT_SECTION_H
#define LLD_ELF_INPUT_SECTION_H

#include "Config.h"
#include "lld/Core/LLVM.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf2 {

template <class ELFT> class ObjectFile;
template <class ELFT> class OutputSection;
template <class ELFT> class OutputSectionBase;

// This corresponds to a section of an input file.
template <class ELFT> class InputSectionBase {
protected:
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela Elf_Rela;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  const Elf_Shdr *Header;

  // The file this section is from.
  ObjectFile<ELFT> *File;

public:
  enum Kind { Regular, EHFrame, Merge, MipsReginfo };
  Kind SectionKind;

  InputSectionBase(ObjectFile<ELFT> *File, const Elf_Shdr *Header,
                   Kind SectionKind);
  OutputSectionBase<ELFT> *OutSec = nullptr;

  // Used for garbage collection.
  // Live bit makes sense only when Config->GcSections is true.
  bool isLive() const { return !Config->GcSections || Live; }
  bool Live = false;

  // Returns the size of this section (even if this is a common or BSS.)
  size_t getSize() const { return Header->sh_size; }

  static InputSectionBase<ELFT> Discarded;

  StringRef getSectionName() const;
  const Elf_Shdr *getSectionHdr() const { return Header; }
  ObjectFile<ELFT> *getFile() const { return File; }

  // The writer sets and uses the addresses.
  uintX_t getAlign() {
    // The ELF spec states that a value of 0 means the section has no alignment
    // constraits.
    return std::max<uintX_t>(Header->sh_addralign, 1);
  }

  uintX_t getOffset(const Elf_Sym &Sym);

  // Translate an offset in the input section to an offset in the output
  // section.
  uintX_t getOffset(uintX_t Offset);

  ArrayRef<uint8_t> getSectionData() const;

  // Returns a section that Rel is pointing to. Used by the garbage collector.
  InputSectionBase<ELFT> *getRelocTarget(const Elf_Rel &Rel);
  InputSectionBase<ELFT> *getRelocTarget(const Elf_Rela &Rel);

  template <bool isRela>
  using RelIteratorRange =
      llvm::iterator_range<const llvm::object::Elf_Rel_Impl<ELFT, isRela> *>;

  template <bool isRela>
  void relocate(uint8_t *Buf, uint8_t *BufEnd, RelIteratorRange<isRela> Rels);

private:
  template <bool isRela>
  uint8_t *findMipsPairedReloc(uint8_t *Buf, uint32_t SymIndex, uint32_t Type,
                               RelIteratorRange<isRela> Rels);
};

template <class ELFT>
InputSectionBase<ELFT>
    InputSectionBase<ELFT>::Discarded(nullptr, nullptr,
                                      InputSectionBase<ELFT>::Regular);

// Usually sections are copied to the output as atomic chunks of data,
// but some special types of sections are split into small pieces of data
// and each piece is copied to a different place in the output.
// This class represents such special sections.
template <class ELFT> class SplitInputSection : public InputSectionBase<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;

public:
  SplitInputSection(ObjectFile<ELFT> *File, const Elf_Shdr *Header,
                    typename InputSectionBase<ELFT>::Kind SectionKind);

  // For each piece of data, we maintain the offsets in the input section and
  // in the output section. The latter may be -1 if it is not assigned yet.
  std::vector<std::pair<uintX_t, uintX_t>> Offsets;

  std::pair<std::pair<uintX_t, uintX_t> *, uintX_t>
  getRangeAndSize(uintX_t Offset);
};

// This corresponds to a SHF_MERGE section of an input file.
template <class ELFT> class MergeInputSection : public SplitInputSection<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;

public:
  MergeInputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Header);
  static bool classof(const InputSectionBase<ELFT> *S);
  // Translate an offset in the input section to an offset in the output
  // section.
  uintX_t getOffset(uintX_t Offset);
};

// This corresponds to a .eh_frame section of an input file.
template <class ELFT> class EHInputSection : public SplitInputSection<ELFT> {
public:
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  EHInputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Header);
  static bool classof(const InputSectionBase<ELFT> *S);

  // Translate an offset in the input section to an offset in the output
  // section.
  uintX_t getOffset(uintX_t Offset);

  // Relocation section that refer to this one.
  const Elf_Shdr *RelocSection = nullptr;
};

// This corresponds to a non SHF_MERGE section of an input file.
template <class ELFT> class InputSection : public InputSectionBase<ELFT> {
  typedef InputSectionBase<ELFT> Base;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela Elf_Rela;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;

public:
  InputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Header);

  // Write this section to a mmap'ed file, assuming Buf is pointing to
  // beginning of the output section.
  void writeTo(uint8_t *Buf);

  // Relocation sections that refer to this one.
  SmallVector<const Elf_Shdr *, 1> RelocSections;

  // The offset from beginning of the output sections this section was assigned
  // to. The writer sets a value.
  uint64_t OutSecOff = 0;

  static bool classof(const InputSectionBase<ELFT> *S);
};

// MIPS .reginfo section provides information on the registers used by the code
// in the object file. Linker should collect this information and write a single
// .reginfo section in the output file. The output section contains a union of
// used registers masks taken from input .reginfo sections and final value
// of the `_gp` symbol.  For details: Chapter 4 / "Register Information" at
// ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
template <class ELFT>
class MipsReginfoInputSection : public InputSectionBase<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;

public:
  MipsReginfoInputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Hdr);
  static bool classof(const InputSectionBase<ELFT> *S);

  const llvm::object::Elf_Mips_RegInfo<ELFT> *Reginfo;
};

} // namespace elf2
} // namespace lld

#endif
