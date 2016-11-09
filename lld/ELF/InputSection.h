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

class DefinedCommon;
class SymbolBody;
struct SectionPiece;

template <class ELFT> class ICF;
template <class ELFT> class DefinedRegular;
template <class ELFT> class ObjectFile;
template <class ELFT> class OutputSection;
template <class ELFT> class OutputSectionBase;

// We need non-template input section class to store symbol layout
// in linker script parser structures, where we do not have ELFT
// template parameter. For each scripted output section symbol we
// store pointer to preceding InputSectionData object or nullptr,
// if symbol should be placed at the very beginning of the output
// section
class InputSectionData {
public:
  enum Kind { Regular, EHFrame, Merge, MipsReginfo, MipsOptions, MipsAbiFlags };

  // The garbage collector sets sections' Live bits.
  // If GC is disabled, all sections are considered live by default.
  InputSectionData(Kind SectionKind, StringRef Name, ArrayRef<uint8_t> Data,
                   bool Compressed, bool Live)
      : SectionKind(SectionKind), Live(Live), Compressed(Compressed),
        Name(Name), Data(Data) {}

private:
  unsigned SectionKind : 3;

public:
  Kind kind() const { return (Kind)SectionKind; }

  unsigned Live : 1; // for garbage collection
  unsigned Compressed : 1;
  uint32_t Alignment;
  StringRef Name;
  ArrayRef<uint8_t> Data;

  template <typename T> llvm::ArrayRef<T> getDataAs() const {
    size_t S = Data.size();
    assert(S % sizeof(T) == 0);
    return llvm::makeArrayRef<T>((const T *)Data.data(), S / sizeof(T));
  }

  // If a section is compressed, this has the uncompressed section data.
  std::unique_ptr<uint8_t[]> UncompressedData;

  std::vector<Relocation> Relocations;
};

// This corresponds to a section of an input file.
template <class ELFT> class InputSectionBase : public InputSectionData {
protected:
  typedef typename ELFT::Chdr Elf_Chdr;
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Rela Elf_Rela;
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::uint uintX_t;

  // The file this section is from.
  ObjectFile<ELFT> *File;

public:
  // These corresponds to the fields in Elf_Shdr.
  uintX_t Flags;
  uintX_t Offset = 0;
  uintX_t Entsize;
  uint32_t Type;
  uint32_t Link;
  uint32_t Info;

  InputSectionBase()
      : InputSectionData(Regular, "", ArrayRef<uint8_t>(), false, false),
        Repl(this) {}

  InputSectionBase(ObjectFile<ELFT> *File, const Elf_Shdr *Header,
                   StringRef Name, Kind SectionKind);
  InputSectionBase(ObjectFile<ELFT> *File, uintX_t Flags, uint32_t Type,
                   uintX_t Entsize, uint32_t Link, uint32_t Info,
                   uintX_t Addralign, ArrayRef<uint8_t> Data, StringRef Name,
                   Kind SectionKind);
  OutputSectionBase<ELFT> *OutSec = nullptr;

  // This pointer points to the "real" instance of this instance.
  // Usually Repl == this. However, if ICF merges two sections,
  // Repl pointer of one section points to another section. So,
  // if you need to get a pointer to this instance, do not use
  // this but instead this->Repl.
  InputSectionBase<ELFT> *Repl;

  // Returns the size of this section (even if this is a common or BSS.)
  size_t getSize() const;

  ObjectFile<ELFT> *getFile() const { return File; }
  llvm::object::ELFFile<ELFT> getObj() const { return File->getObj(); }
  uintX_t getOffset(const DefinedRegular<ELFT> &Sym) const;
  InputSectionBase *getLinkOrderDep() const;
  // Translate an offset in the input section to an offset in the output
  // section.
  uintX_t getOffset(uintX_t Offset) const;

  void uncompress();

  void relocate(uint8_t *Buf, uint8_t *BufEnd);

private:
  std::pair<ArrayRef<uint8_t>, uint64_t>
  getElfCompressedData(ArrayRef<uint8_t> Data);

  std::pair<ArrayRef<uint8_t>, uint64_t>
  getRawCompressedData(ArrayRef<uint8_t> Data);
};

// SectionPiece represents a piece of splittable section contents.
// We allocate a lot of these and binary search on them. This means that they
// have to be as compact as possible, which is why we don't store the size (can
// be found by looking at the next one) and put the hash in a side table.
struct SectionPiece {
  SectionPiece(size_t Off, bool Live = false)
      : InputOff(Off), OutputOff(-1), Live(Live || !Config->GcSections) {}

  size_t InputOff;
  ssize_t OutputOff : 8 * sizeof(ssize_t) - 1;
  size_t Live : 1;
};
static_assert(sizeof(SectionPiece) == 2 * sizeof(size_t),
              "SectionPiece is too big");

// This corresponds to a SHF_MERGE section of an input file.
template <class ELFT> class MergeInputSection : public InputSectionBase<ELFT> {
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::Shdr Elf_Shdr;

public:
  MergeInputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Header,
                    StringRef Name);
  static bool classof(const InputSectionData *S);
  void splitIntoPieces();

  // Mark the piece at a given offset live. Used by GC.
  void markLiveAt(uintX_t Offset) {
    assert(this->Flags & llvm::ELF::SHF_ALLOC);
    LiveOffsets.insert(Offset);
  }

  // Translate an offset in the input section to an offset
  // in the output section.
  uintX_t getOffset(uintX_t Offset) const;

  void finalizePieces();

  // Splittable sections are handled as a sequence of data
  // rather than a single large blob of data.
  std::vector<SectionPiece> Pieces;
  ArrayRef<uint8_t> getData(std::vector<SectionPiece>::const_iterator I) const;
  std::vector<uint32_t> Hashes;

  // Returns the SectionPiece at a given input section offset.
  SectionPiece *getSectionPiece(uintX_t Offset);
  const SectionPiece *getSectionPiece(uintX_t Offset) const;

private:
  std::vector<SectionPiece> splitStrings(ArrayRef<uint8_t> A, size_t Size);
  std::vector<SectionPiece> splitNonStrings(ArrayRef<uint8_t> A, size_t Size);

  llvm::DenseMap<uintX_t, uintX_t> OffsetMap;
  llvm::DenseSet<uintX_t> LiveOffsets;
};

struct EhSectionPiece : public SectionPiece {
  EhSectionPiece(size_t Off, ArrayRef<uint8_t> Data, unsigned FirstRelocation)
      : SectionPiece(Off, false), Data(Data.data()), Size(Data.size()),
        FirstRelocation(FirstRelocation) {}
  const uint8_t *Data;
  uint32_t Size;
  uint32_t size() const { return Size; }

  ArrayRef<uint8_t> data() { return {Data, Size}; }
  unsigned FirstRelocation;
};

// This corresponds to a .eh_frame section of an input file.
template <class ELFT> class EhInputSection : public InputSectionBase<ELFT> {
public:
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::uint uintX_t;
  EhInputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Header, StringRef Name);
  static bool classof(const InputSectionData *S);
  void split();
  template <class RelTy> void split(ArrayRef<RelTy> Rels);

  // Splittable sections are handled as a sequence of data
  // rather than a single large blob of data.
  std::vector<EhSectionPiece> Pieces;

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
  InputSection();
  InputSection(uintX_t Flags, uint32_t Type, uintX_t Addralign,
               ArrayRef<uint8_t> Data, StringRef Name);
  InputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Header, StringRef Name);

  static InputSection<ELFT> Discarded;

  // Write this section to a mmap'ed file, assuming Buf is pointing to
  // beginning of the output section.
  void writeTo(uint8_t *Buf);

  // Relocation sections that refer to this one.
  llvm::TinyPtrVector<const Elf_Shdr *> RelocSections;

  // The offset from beginning of the output sections this section was assigned
  // to. The writer sets a value.
  uint64_t OutSecOff = 0;

  // InputSection that is dependent on us (reverse dependency for GC)
  InputSectionBase<ELFT> *DependentSection = nullptr;

  static bool classof(const InputSectionData *S);

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

template <class ELFT> InputSection<ELFT> InputSection<ELFT>::Discarded;

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
  MipsReginfoInputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Hdr,
                          StringRef Name);
  static bool classof(const InputSectionData *S);

  const llvm::object::Elf_Mips_RegInfo<ELFT> *Reginfo = nullptr;
};

template <class ELFT>
class MipsOptionsInputSection : public InputSectionBase<ELFT> {
  typedef typename ELFT::Shdr Elf_Shdr;

public:
  MipsOptionsInputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Hdr,
                          StringRef Name);
  static bool classof(const InputSectionData *S);

  const llvm::object::Elf_Mips_RegInfo<ELFT> *Reginfo = nullptr;
};

template <class ELFT>
class MipsAbiFlagsInputSection : public InputSectionBase<ELFT> {
  typedef typename ELFT::Shdr Elf_Shdr;

public:
  MipsAbiFlagsInputSection(ObjectFile<ELFT> *F, const Elf_Shdr *Hdr,
                           StringRef Name);
  static bool classof(const InputSectionData *S);

  const llvm::object::Elf_Mips_ABIFlags<ELFT> *Flags = nullptr;
};

} // namespace elf
} // namespace lld

#endif
