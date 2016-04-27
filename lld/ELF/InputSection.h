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
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf {

class SymbolBody;

template <class ELFT> class ICF;
template <class ELFT> class DefinedRegular;
template <class ELFT> class ObjectFile;
template <class ELFT> class OutputSection;
template <class ELFT> class OutputSectionBase;

enum RelExpr {
  R_ABS,
  R_GOT,
  R_GOTONLY_PC,
  R_GOTREL,
  R_GOT_OFF,
  R_GOT_FROM_END,
  R_GOT_PAGE_PC,
  R_GOT_PC,
  R_MIPS_GOT,
  R_MIPS_GOT_LOCAL,
  R_NEG_TLS,
  R_PAGE_PC,
  R_PC,
  R_PLT,
  R_PLT_PC,
  R_PPC_OPD,
  R_PPC_PLT_OPD,
  R_PPC_TOC,
  R_RELAX_TLS_GD_TO_IE,
  R_RELAX_TLS_GD_TO_IE_PC,
  R_RELAX_TLS_GD_TO_LE,
  R_RELAX_TLS_IE_TO_LE,
  R_RELAX_TLS_LD_TO_LE,
  R_SIZE,
  R_THUNK,
  R_TLS,
  R_TLSGD,
  R_TLSGD_PC,
  R_TLSLD,
  R_TLSLD_PC
};

inline bool refersToGotEntry(RelExpr Expr) {
  return Expr == R_GOT || Expr == R_GOT_OFF || Expr == R_MIPS_GOT ||
         Expr == R_MIPS_GOT_LOCAL || Expr == R_GOT_PAGE_PC ||
         Expr == R_GOT_PC || Expr == R_GOT_FROM_END;
}

struct Relocation {
  RelExpr Expr;
  uint32_t Type;
  uint64_t Offset;
  uint64_t Addend;
  SymbolBody *Sym;
};

// This corresponds to a section of an input file.
template <class ELFT> class InputSectionBase {
protected:
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Rela Elf_Rela;
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::uint uintX_t;
  const Elf_Shdr *Header;

  // The file this section is from.
  ObjectFile<ELFT> *File;

public:
  enum Kind { Regular, EHFrame, Merge, MipsReginfo };
  Kind SectionKind;

  InputSectionBase() : Repl(this) {}

  InputSectionBase(ObjectFile<ELFT> *File, const Elf_Shdr *Header,
                   Kind SectionKind);
  OutputSectionBase<ELFT> *OutSec = nullptr;
  uint32_t Align;

  // Used for garbage collection.
  bool Live;

  // This pointer points to the "real" instance of this instance.
  // Usually Repl == this. However, if ICF merges two sections,
  // Repl pointer of one section points to another section. So,
  // if you need to get a pointer to this instance, do not use
  // this but instead this->Repl.
  InputSectionBase<ELFT> *Repl;

  // Returns the size of this section (even if this is a common or BSS.)
  size_t getSize() const;

  static InputSectionBase<ELFT> Discarded;

  StringRef getSectionName() const;
  const Elf_Shdr *getSectionHdr() const { return Header; }
  ObjectFile<ELFT> *getFile() const { return File; }
  uintX_t getOffset(const DefinedRegular<ELFT> &Sym);

  // Translate an offset in the input section to an offset in the output
  // section.
  uintX_t getOffset(uintX_t Offset);

  ArrayRef<uint8_t> getSectionData() const;

  void relocate(uint8_t *Buf, uint8_t *BufEnd);
  std::vector<Relocation> Relocations;
};

template <class ELFT> InputSectionBase<ELFT> InputSectionBase<ELFT>::Discarded;

// Usually sections are copied to the output as atomic chunks of data,
// but some special types of sections are split into small pieces of data
// and each piece is copied to a different place in the output.
// This class represents such special sections.
template <class ELFT> class SplitInputSection : public InputSectionBase<ELFT> {
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::uint uintX_t;

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
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::Shdr Elf_Shdr;

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
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::uint uintX_t;
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
  friend ICF<ELFT>;
  typedef InputSectionBase<ELFT> Base;
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Rela Elf_Rela;
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::uint uintX_t;

public:
  InputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Header);

  // Write this section to a mmap'ed file, assuming Buf is pointing to
  // beginning of the output section.
  void writeTo(uint8_t *Buf);

  // Relocation sections that refer to this one.
  llvm::TinyPtrVector<const Elf_Shdr *> RelocSections;

  // The offset from beginning of the output sections this section was assigned
  // to. The writer sets a value.
  uint64_t OutSecOff = 0;

  static bool classof(const InputSectionBase<ELFT> *S);

  InputSectionBase<ELFT> *getRelocatedSection();

  // Register thunk related to the symbol. When the section is written
  // to a mmap'ed file, target is requested to write an actual thunk code.
  // Now thunks is supported for MIPS target only.
  void addThunk(SymbolBody &Body);

  // The offset of synthetic thunk code from beginning of this section.
  uint64_t getThunkOff() const;

  // Size of chunk with thunks code.
  uint64_t getThunksSize() const;

private:
  template <class RelTy>
  void copyRelocations(uint8_t *Buf, llvm::ArrayRef<RelTy> Rels);

  // Called by ICF to merge two input sections.
  void replace(InputSection<ELFT> *Other);

  // Used by ICF.
  uint64_t GroupId = 0;

  llvm::TinyPtrVector<const SymbolBody *> Thunks;
};

// MIPS .reginfo section provides information on the registers used by the code
// in the object file. Linker should collect this information and write a single
// .reginfo section in the output file. The output section contains a union of
// used registers masks taken from input .reginfo sections and final value
// of the `_gp` symbol.  For details: Chapter 4 / "Register Information" at
// ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
template <class ELFT>
class MipsReginfoInputSection : public InputSectionBase<ELFT> {
  typedef typename ELFT::Shdr Elf_Shdr;

public:
  MipsReginfoInputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Hdr);
  static bool classof(const InputSectionBase<ELFT> *S);

  const llvm::object::Elf_Mips_RegInfo<ELFT> *Reginfo;
};

} // namespace elf
} // namespace lld

#endif
