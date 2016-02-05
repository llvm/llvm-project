//===- OutputSections.h -----------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_OUTPUT_SECTIONS_H
#define LLD_ELF_OUTPUT_SECTIONS_H

#include "lld/Core/LLVM.h"

#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELF.h"

#include "Config.h"

#include <type_traits>

namespace lld {
namespace elf2 {

class SymbolBody;
template <class ELFT> class SymbolTable;
template <class ELFT> class SymbolTableSection;
template <class ELFT> class StringTableSection;
template <class ELFT> class EHInputSection;
template <class ELFT> class InputSection;
template <class ELFT> class InputSectionBase;
template <class ELFT> class MergeInputSection;
template <class ELFT> class MipsReginfoInputSection;
template <class ELFT> class OutputSection;
template <class ELFT> class ObjectFile;
template <class ELFT> class DefinedRegular;

template <class ELFT>
static inline typename llvm::object::ELFFile<ELFT>::uintX_t
getAddend(const typename llvm::object::ELFFile<ELFT>::Elf_Rel &Rel) {
  return 0;
}

template <class ELFT>
static inline typename llvm::object::ELFFile<ELFT>::uintX_t
getAddend(const typename llvm::object::ELFFile<ELFT>::Elf_Rela &Rel) {
  return Rel.r_addend;
}

template <class ELFT, bool IsRela>
typename llvm::object::ELFFile<ELFT>::uintX_t
getLocalRelTarget(const ObjectFile<ELFT> &File,
                  const llvm::object::Elf_Rel_Impl<ELFT, IsRela> &Rel,
                  typename llvm::object::ELFFile<ELFT>::uintX_t Addend);

bool canBePreempted(const SymbolBody *Body, bool NeedsGot);

// This represents a section in an output file.
// Different sub classes represent different types of sections. Some contain
// input sections, others are created by the linker.
// The writer creates multiple OutputSections and assign them unique,
// non-overlapping file offsets and VAs.
template <class ELFT> class OutputSectionBase {
public:
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;

  OutputSectionBase(StringRef Name, uint32_t sh_type, uintX_t sh_flags);
  void setVA(uintX_t VA) { Header.sh_addr = VA; }
  uintX_t getVA() const { return Header.sh_addr; }
  void setFileOffset(uintX_t Off) { Header.sh_offset = Off; }
  void setSHName(unsigned Val) { Header.sh_name = Val; }
  void writeHeaderTo(Elf_Shdr *SHdr);
  StringRef getName() { return Name; }

  virtual void addSection(InputSectionBase<ELFT> *C) {}

  unsigned SectionIndex;

  // Returns the size of the section in the output file.
  uintX_t getSize() const { return Header.sh_size; }
  void setSize(uintX_t Val) { Header.sh_size = Val; }
  uintX_t getFlags() { return Header.sh_flags; }
  uintX_t getFileOff() { return Header.sh_offset; }
  uintX_t getAlign() {
    // The ELF spec states that a value of 0 means the section has no alignment
    // constraits.
    return std::max<uintX_t>(Header.sh_addralign, 1);
  }
  uint32_t getType() { return Header.sh_type; }
  void updateAlign(uintX_t Align) {
    if (Align > Header.sh_addralign)
      Header.sh_addralign = Align;
  }

  virtual void finalize() {}
  virtual void writeTo(uint8_t *Buf) = 0;
  virtual ~OutputSectionBase() = default;

protected:
  StringRef Name;
  Elf_Shdr Header;
};

template <class ELFT> class GotSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;
  typedef typename Base::uintX_t uintX_t;

public:
  GotSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  void addEntry(SymbolBody *Sym);
  void addMipsLocalEntry();
  bool addDynTlsEntry(SymbolBody *Sym);
  bool addTlsIndex();
  bool empty() const { return MipsLocalEntries == 0 && Entries.empty(); }
  uintX_t getMipsLocalFullAddr(const SymbolBody &B);
  uintX_t getMipsLocalPageAddr(uintX_t Addr);
  uintX_t getGlobalDynAddr(const SymbolBody &B) const;
  uintX_t getNumEntries() const { return Entries.size(); }

  // Returns the symbol which corresponds to the first entry of the global part
  // of GOT on MIPS platform. It is required to fill up MIPS-specific dynamic
  // table properties.
  // Returns nullptr if the global part is empty.
  const SymbolBody *getMipsFirstGlobalEntry() const;

  // Returns the number of entries in the local part of GOT including
  // the number of reserved entries. This method is MIPS-specific.
  unsigned getMipsLocalEntriesNum() const;

  uintX_t getTlsIndexVA() { return Base::getVA() + TlsIndexOff; }

private:
  std::vector<const SymbolBody *> Entries;
  uint32_t TlsIndexOff = -1;
  uint32_t MipsLocalEntries = 0;
  llvm::DenseMap<uintX_t, size_t> MipsLocalGotPos;

  uintX_t getMipsLocalEntryAddr(uintX_t EntryValue);
};

template <class ELFT>
class GotPltSection final : public OutputSectionBase<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;

public:
  GotPltSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  void addEntry(SymbolBody *Sym);
  bool empty() const;

private:
  std::vector<const SymbolBody *> Entries;
};

template <class ELFT> class PltSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;
  typedef typename Base::uintX_t uintX_t;

public:
  PltSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  void addEntry(SymbolBody *Sym);
  bool empty() const { return Entries.empty(); }

private:
  std::vector<std::pair<const SymbolBody *, unsigned>> Entries;
};

template <class ELFT> struct DynamicReloc {
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  uint32_t Type;

  // Where the relocation is.
  enum OffsetKind {
    Off_Got,       // The got entry of Sym.
    Off_GotPlt,    // The got.plt entry of Sym.
    Off_Bss,       // The bss entry of Sym (copy reloc).
    Off_Sec,       // The final position of the given input section and offset.
    Off_LTlsIndex, // The local tls index.
    Off_GTlsIndex, // The global tls index of Sym.
    Off_GTlsOffset // The global tls offset of Sym.
  } OKind;

  SymbolBody *Sym = nullptr;
  InputSectionBase<ELFT> *OffsetSec = nullptr;
  uintX_t OffsetInSec = 0;
  bool UseSymVA = false;
  InputSectionBase<ELFT> *TargetSec = nullptr;
  uintX_t OffsetInTargetSec = 0;
  uintX_t Addend = 0;

  DynamicReloc(uint32_t Type, OffsetKind OKind, SymbolBody *Sym)
      : Type(Type), OKind(OKind), Sym(Sym) {}

  DynamicReloc(uint32_t Type, OffsetKind OKind, bool UseSymVA, SymbolBody *Sym)
      : Type(Type), OKind(OKind), Sym(Sym), UseSymVA(UseSymVA) {}

  DynamicReloc(uint32_t Type, InputSectionBase<ELFT> *OffsetSec,
               uintX_t OffsetInSec, bool UseSymVA, SymbolBody *Sym,
               uintX_t Addend)
      : Type(Type), OKind(Off_Sec), Sym(Sym), OffsetSec(OffsetSec),
        OffsetInSec(OffsetInSec), UseSymVA(UseSymVA), Addend(Addend) {}

  DynamicReloc(uint32_t Type, InputSectionBase<ELFT> *OffsetSec,
               uintX_t OffsetInSec, InputSectionBase<ELFT> *TargetSec,
               uintX_t OffsetInTargetSec, uintX_t Addend)
      : Type(Type), OKind(Off_Sec), OffsetSec(OffsetSec),
        OffsetInSec(OffsetInSec), TargetSec(TargetSec),
        OffsetInTargetSec(OffsetInTargetSec), Addend(Addend) {}
};

template <class ELFT>
class SymbolTableSection final : public OutputSectionBase<ELFT> {
public:
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  SymbolTableSection(SymbolTable<ELFT> &Table,
                     StringTableSection<ELFT> &StrTabSec);

  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  void addSymbol(SymbolBody *Body);
  StringTableSection<ELFT> &getStrTabSec() const { return StrTabSec; }
  unsigned getNumSymbols() const { return NumLocals + Symbols.size() + 1; }

  ArrayRef<std::pair<SymbolBody *, unsigned>> getSymbols() const {
    return Symbols;
  }

  unsigned NumLocals = 0;
  StringTableSection<ELFT> &StrTabSec;

private:
  void writeLocalSymbols(uint8_t *&Buf);
  void writeGlobalSymbols(uint8_t *Buf);

  static uint8_t getSymbolBinding(SymbolBody *Body);

  SymbolTable<ELFT> &Table;
  std::vector<std::pair<SymbolBody *, unsigned>> Symbols;
};

template <class ELFT>
class RelocationSection final : public OutputSectionBase<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela Elf_Rela;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;

public:
  RelocationSection(StringRef Name, bool IsRela);
  void addReloc(const DynamicReloc<ELFT> &Reloc) { Relocs.push_back(Reloc); }
  unsigned getRelocOffset();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  bool hasRelocs() const { return !Relocs.empty(); }
  bool isRela() const { return IsRela; }

  bool Static = false;

private:
  std::vector<DynamicReloc<ELFT>> Relocs;
  const bool IsRela;
};

template <class ELFT>
class OutputSection final : public OutputSectionBase<ELFT> {
public:
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela Elf_Rela;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  OutputSection(StringRef Name, uint32_t sh_type, uintX_t sh_flags);
  void addSection(InputSectionBase<ELFT> *C) override;
  void writeTo(uint8_t *Buf) override;

private:
  std::vector<InputSection<ELFT> *> Sections;
};

template <class ELFT>
class MergeOutputSection final : public OutputSectionBase<ELFT> {
  typedef typename OutputSectionBase<ELFT>::uintX_t uintX_t;

  bool shouldTailMerge() const;

public:
  MergeOutputSection(StringRef Name, uint32_t sh_type, uintX_t sh_flags);
  void addSection(InputSectionBase<ELFT> *S) override;
  void writeTo(uint8_t *Buf) override;
  unsigned getOffset(StringRef Val);
  void finalize() override;

private:
  llvm::StringTableBuilder Builder{llvm::StringTableBuilder::RAW};
};

// FDE or CIE
template <class ELFT> struct EHRegion {
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  EHRegion(EHInputSection<ELFT> *S, unsigned Index);
  StringRef data() const;
  EHInputSection<ELFT> *S;
  unsigned Index;
};

template <class ELFT> struct Cie : public EHRegion<ELFT> {
  Cie(EHInputSection<ELFT> *S, unsigned Index);
  std::vector<EHRegion<ELFT>> Fdes;
  uint8_t FdeEncoding;
};

template <class ELFT>
class EHOutputSection final : public OutputSectionBase<ELFT> {
public:
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela Elf_Rela;
  EHOutputSection(StringRef Name, uint32_t sh_type, uintX_t sh_flags);
  void writeTo(uint8_t *Buf) override;

  template <bool IsRela>
  void addSectionAux(
      EHInputSection<ELFT> *S,
      llvm::iterator_range<const llvm::object::Elf_Rel_Impl<ELFT, IsRela> *>
          Rels);

  void addSection(InputSectionBase<ELFT> *S) override;

private:
  uint8_t getFdeEncoding(ArrayRef<uint8_t> D);
  uintX_t readEntryLength(ArrayRef<uint8_t> D);

  std::vector<EHInputSection<ELFT> *> Sections;
  std::vector<Cie<ELFT>> Cies;

  // Maps CIE content + personality to a index in Cies.
  llvm::DenseMap<std::pair<StringRef, StringRef>, unsigned> CieMap;
};

template <class ELFT>
class InterpSection final : public OutputSectionBase<ELFT> {
public:
  InterpSection();
  void writeTo(uint8_t *Buf) override;
};

template <class ELFT>
class StringTableSection final : public OutputSectionBase<ELFT> {
public:
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  StringTableSection(StringRef Name, bool Dynamic);
  unsigned addString(StringRef S, bool HashIt = true);
  void writeTo(uint8_t *Buf) override;
  unsigned getSize() const { return Size; }
  void finalize() override { this->Header.sh_size = getSize(); }
  bool isDynamic() const { return Dynamic; }

private:
  const bool Dynamic;
  llvm::DenseMap<StringRef, unsigned> StringMap;
  std::vector<StringRef> Strings;
  unsigned Size = 1; // ELF string tables start with a NUL byte, so 1.
};

template <class ELFT>
class HashTableSection final : public OutputSectionBase<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Word Elf_Word;

public:
  HashTableSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
};

// Outputs GNU Hash section. For detailed explanation see:
// https://blogs.oracle.com/ali/entry/gnu_hash_elf_sections
template <class ELFT>
class GnuHashTableSection final : public OutputSectionBase<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Off Elf_Off;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Word Elf_Word;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;

public:
  GnuHashTableSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;

  // Adds symbols to the hash table.
  // Sorts the input to satisfy GNU hash section requirements.
  void addSymbols(std::vector<std::pair<SymbolBody *, unsigned>> &Symbols);

private:
  static unsigned calcNBuckets(unsigned NumHashed);
  static unsigned calcMaskWords(unsigned NumHashed);

  void writeHeader(uint8_t *&Buf);
  void writeBloomFilter(uint8_t *&Buf);
  void writeHashTable(uint8_t *Buf);

  struct HashedSymbolData {
    SymbolBody *Body;
    unsigned STName;
    uint32_t Hash;
  };

  std::vector<HashedSymbolData> HashedSymbols;

  unsigned MaskWords;
  unsigned NBuckets;
  unsigned Shift2;
};

template <class ELFT>
class DynamicSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Dyn Elf_Dyn;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela Elf_Rela;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;

  // The .dynamic section contains information for the dynamic linker.
  // The section consists of fixed size entries, which consist of
  // type and value fields. Value are one of plain integers, symbol
  // addresses, or section addresses. This struct represents the entry.
  struct Entry {
    int32_t Tag;
    union {
      OutputSectionBase<ELFT> *OutSec;
      uint64_t Val;
      const SymbolBody *Sym;
    };
    enum KindT { SecAddr, SymAddr, PlainInt } Kind;
    Entry(int32_t Tag, OutputSectionBase<ELFT> *OutSec)
        : Tag(Tag), OutSec(OutSec), Kind(SecAddr) {}
    Entry(int32_t Tag, uint64_t Val) : Tag(Tag), Val(Val), Kind(PlainInt) {}
    Entry(int32_t Tag, const SymbolBody *Sym)
        : Tag(Tag), Sym(Sym), Kind(SymAddr) {}
  };

  // finalize() fills this vector with the section contents. finalize()
  // cannot directly create final section contents because when the
  // function is called, symbol or section addresses are not fixed yet.
  std::vector<Entry> Entries;

public:
  DynamicSection(SymbolTable<ELFT> &SymTab);
  void finalize() override;
  void writeTo(uint8_t *Buf) override;

  OutputSectionBase<ELFT> *PreInitArraySec = nullptr;
  OutputSectionBase<ELFT> *InitArraySec = nullptr;
  OutputSectionBase<ELFT> *FiniArraySec = nullptr;

private:
  SymbolTable<ELFT> &SymTab;
};

template <class ELFT>
class MipsReginfoOutputSection final : public OutputSectionBase<ELFT> {
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;

public:
  MipsReginfoOutputSection();
  void writeTo(uint8_t *Buf) override;
  void addSection(InputSectionBase<ELFT> *S) override;

private:
  uint32_t GprMask = 0;
};

// --eh-frame-hdr option tells linker to construct a header for all the
// .eh_frame sections. This header is placed to a section named .eh_frame_hdr
// and also to a PT_GNU_EH_FRAME segment.
// At runtime the unwinder then can find all the PT_GNU_EH_FRAME segments by
// calling dl_iterate_phdr.
// This section contains a lookup table for quick binary search of FDEs.
// Detailed info about internals can be found in Ian Lance Taylor's blog:
// http://www.airs.com/blog/archives/460 (".eh_frame")
// http://www.airs.com/blog/archives/462 (".eh_frame_hdr")
template <class ELFT>
class EhFrameHeader final : public OutputSectionBase<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;

public:
  EhFrameHeader();
  void writeTo(uint8_t *Buf) override;

  void addFde(uint8_t Enc, size_t Off, uint8_t *PCRel);
  void assignEhFrame(EHOutputSection<ELFT> *Sec);
  void reserveFde();

  bool Live = false;

private:
  struct FdeData {
    uint8_t Enc;
    size_t Off;
    uint8_t *PCRel;
  };

  uintX_t getFdePc(uintX_t EhVA, const FdeData &F);

  EHOutputSection<ELFT> *Sec = nullptr;
  std::vector<FdeData> FdeList;
};

// All output sections that are hadnled by the linker specially are
// globally accessible. Writer initializes them, so don't use them
// until Writer is initialized.
template <class ELFT> struct Out {
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Phdr Elf_Phdr;
  static DynamicSection<ELFT> *Dynamic;
  static EhFrameHeader<ELFT> *EhFrameHdr;
  static GnuHashTableSection<ELFT> *GnuHashTab;
  static GotPltSection<ELFT> *GotPlt;
  static GotSection<ELFT> *Got;
  static HashTableSection<ELFT> *HashTab;
  static InterpSection<ELFT> *Interp;
  static OutputSection<ELFT> *Bss;
  static OutputSection<ELFT> *MipsRldMap;
  static OutputSectionBase<ELFT> *Opd;
  static uint8_t *OpdBuf;
  static PltSection<ELFT> *Plt;
  static RelocationSection<ELFT> *RelaDyn;
  static RelocationSection<ELFT> *RelaPlt;
  static StringTableSection<ELFT> *DynStrTab;
  static StringTableSection<ELFT> *ShStrTab;
  static StringTableSection<ELFT> *StrTab;
  static SymbolTableSection<ELFT> *DynSymTab;
  static SymbolTableSection<ELFT> *SymTab;
  static Elf_Phdr *TlsPhdr;
};

template <class ELFT> DynamicSection<ELFT> *Out<ELFT>::Dynamic;
template <class ELFT> EhFrameHeader<ELFT> *Out<ELFT>::EhFrameHdr;
template <class ELFT> GnuHashTableSection<ELFT> *Out<ELFT>::GnuHashTab;
template <class ELFT> GotPltSection<ELFT> *Out<ELFT>::GotPlt;
template <class ELFT> GotSection<ELFT> *Out<ELFT>::Got;
template <class ELFT> HashTableSection<ELFT> *Out<ELFT>::HashTab;
template <class ELFT> InterpSection<ELFT> *Out<ELFT>::Interp;
template <class ELFT> OutputSection<ELFT> *Out<ELFT>::Bss;
template <class ELFT> OutputSection<ELFT> *Out<ELFT>::MipsRldMap;
template <class ELFT> OutputSectionBase<ELFT> *Out<ELFT>::Opd;
template <class ELFT> uint8_t *Out<ELFT>::OpdBuf;
template <class ELFT> PltSection<ELFT> *Out<ELFT>::Plt;
template <class ELFT> RelocationSection<ELFT> *Out<ELFT>::RelaDyn;
template <class ELFT> RelocationSection<ELFT> *Out<ELFT>::RelaPlt;
template <class ELFT> StringTableSection<ELFT> *Out<ELFT>::DynStrTab;
template <class ELFT> StringTableSection<ELFT> *Out<ELFT>::ShStrTab;
template <class ELFT> StringTableSection<ELFT> *Out<ELFT>::StrTab;
template <class ELFT> SymbolTableSection<ELFT> *Out<ELFT>::DynSymTab;
template <class ELFT> SymbolTableSection<ELFT> *Out<ELFT>::SymTab;
template <class ELFT> typename Out<ELFT>::Elf_Phdr *Out<ELFT>::TlsPhdr;

} // namespace elf2
} // namespace lld

#endif // LLD_ELF_OUTPUT_SECTIONS_H
