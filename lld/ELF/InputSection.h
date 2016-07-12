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
#include "Relocations.h"
#include "Thunks.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf {

template <class ELFT> bool isDiscarded(InputSectionBase<ELFT> *S);

class SymbolBody;

template <class ELFT> class ICF;
template <class ELFT> class DefinedRegular;
template <class ELFT> class ObjectFile;
template <class ELFT> class OutputSection;
template <class ELFT> class OutputSectionBase;

// This corresponds to a section of an input file.
template <class ELFT> class InputSectionBase {
protected:
  typedef typename ELFT::Chdr Elf_Chdr;
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Rela Elf_Rela;
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::uint uintX_t;
  const Elf_Shdr *Header;

  // The file this section is from.
  ObjectFile<ELFT> *File;

  // If a section is compressed, this vector has uncompressed section data.
  SmallVector<char, 0> Uncompressed;

public:
  enum Kind { Regular, EHFrame, Merge, MipsReginfo, MipsOptions };
  Kind SectionKind;

  InputSectionBase() : Repl(this) {}

  InputSectionBase(ObjectFile<ELFT> *File, const Elf_Shdr *Header,
                   Kind SectionKind);
  OutputSectionBase<ELFT> *OutSec = nullptr;
  uint32_t Alignment;

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
  uintX_t getOffset(const DefinedRegular<ELFT> &Sym) const;

  // Translate an offset in the input section to an offset in the output
  // section.
  uintX_t getOffset(uintX_t Offset) const;

  ArrayRef<uint8_t> getSectionData() const;

  void uncompress();

  void relocate(uint8_t *Buf, uint8_t *BufEnd);
  std::vector<Relocation<ELFT>> Relocations;

  bool Compressed;
};

template <class ELFT> InputSectionBase<ELFT> InputSectionBase<ELFT>::Discarded;

// SectionPiece represents a piece of splittable section contents.
struct SectionPiece {
  SectionPiece(size_t Off, ArrayRef<uint8_t> Data)
      : InputOff(Off), Data((const uint8_t *)Data.data()), Size(Data.size()),
        Live(!Config->GcSections) {}

  ArrayRef<uint8_t> data() { return {Data, Size}; }
  size_t size() const { return Size; }

  size_t InputOff;
  size_t OutputOff = -1;

private:
  // We use bitfields because SplitInputSection is accessed by
  // std::upper_bound very often.
  // We want to save bits to make it cache friendly.
  const uint8_t *Data;
  uint32_t Size : 31;

public:
  uint32_t Live : 1;
};

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

  // Splittable sections are handled as a sequence of data
  // rather than a single large blob of data.
  std::vector<SectionPiece> Pieces;

  // Returns the SectionPiece at a given input section offset.
  SectionPiece *getSectionPiece(uintX_t Offset);
  const SectionPiece *getSectionPiece(uintX_t Offset) const;
};

// This corresponds to a SHF_MERGE section of an input file.
template <class ELFT> class MergeInputSection : public SplitInputSection<ELFT> {
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::Shdr Elf_Shdr;

public:
  MergeInputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Header);
  static bool classof(const InputSectionBase<ELFT> *S);
  void splitIntoPieces();

  // Mark the piece at a given offset live. Used by GC.
  void markLiveAt(uintX_t Offset) { LiveOffsets.insert(Offset); }

  // Translate an offset in the input section to an offset
  // in the output section.
  uintX_t getOffset(uintX_t Offset) const;

  void finalizePieces();

private:
  llvm::DenseMap<uintX_t, uintX_t> OffsetMap;
  llvm::DenseSet<uintX_t> LiveOffsets;
};

// This corresponds to a .eh_frame section of an input file.
template <class ELFT> class EhInputSection : public SplitInputSection<ELFT> {
public:
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::uint uintX_t;
  EhInputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Header);
  static bool classof(const InputSectionBase<ELFT> *S);
  void split();

  // Translate an offset in the input section to an offset in the output
  // section.
  uintX_t getOffset(uintX_t Offset) const;

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
  // Now thunks is supported for MIPS and ARM target only.
  void addThunk(const Thunk<ELFT> *T);

  // The offset of synthetic thunk code from beginning of this section.
  uint64_t getThunkOff() const;

  // Size of chunk with thunks code.
  uint64_t getThunksSize() const;

  template <class RelTy>
  void relocateNonAlloc(uint8_t *Buf, llvm::ArrayRef<RelTy> Rels);

private:
  template <class RelTy>
  void copyRelocations(uint8_t *Buf, llvm::ArrayRef<RelTy> Rels);

  // Called by ICF to merge two input sections.
  void replace(InputSection<ELFT> *Other);

  // Used by ICF.
  uint64_t GroupId = 0;

  llvm::TinyPtrVector<const Thunk<ELFT> *> Thunks;
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

  const llvm::object::Elf_Mips_RegInfo<ELFT> *Reginfo = nullptr;
};

template <class ELFT>
class MipsOptionsInputSection : public InputSectionBase<ELFT> {
  typedef typename ELFT::Shdr Elf_Shdr;

public:
  MipsOptionsInputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Hdr);
  static bool classof(const InputSectionBase<ELFT> *S);

  const llvm::object::Elf_Mips_RegInfo<ELFT> *Reginfo = nullptr;
};

} // namespace elf
} // namespace lld

#endif
