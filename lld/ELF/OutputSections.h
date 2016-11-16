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

#include "Config.h"
#include "GdbIndex.h"
#include "Relocations.h"

#include "lld/Core/LLVM.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf {

class SymbolBody;
struct EhSectionPiece;
template <class ELFT> class SymbolTable;
template <class ELFT> class SymbolTableSection;
template <class ELFT> class StringTableSection;
template <class ELFT> class EhInputSection;
template <class ELFT> class InputSection;
template <class ELFT> class InputSectionBase;
template <class ELFT> class MergeInputSection;
template <class ELFT> class OutputSection;
template <class ELFT> class ObjectFile;
template <class ELFT> class SharedFile;
template <class ELFT> class SharedSymbol;
template <class ELFT> class DefinedRegular;

// This represents a section in an output file.
// Different sub classes represent different types of sections. Some contain
// input sections, others are created by the linker.
// The writer creates multiple OutputSections and assign them unique,
// non-overlapping file offsets and VAs.
class OutputSectionBase {
public:
  enum Kind {
    Base,
    EHFrame,
    EHFrameHdr,
    GnuHashTable,
    HashTable,
    Merge,
    Plt,
    Regular,
    Reloc,
    SymTable,
    VersDef,
    VersNeed,
    VersTable
  };

  OutputSectionBase(StringRef Name, uint32_t Type, uint64_t Flags);
  void setLMAOffset(uint64_t LMAOff) { LMAOffset = LMAOff; }
  uint64_t getLMA() const { return Addr + LMAOffset; }
  template <typename ELFT> void writeHeaderTo(typename ELFT::Shdr *SHdr);
  StringRef getName() const { return Name; }

  virtual void addSection(InputSectionData *C) {}
  virtual Kind getKind() const { return Base; }
  static bool classof(const OutputSectionBase *B) {
    return B->getKind() == Base;
  }

  unsigned SectionIndex;

  uint32_t getPhdrFlags() const;

  void updateAlignment(uint64_t Alignment) {
    if (Alignment > Addralign)
      Addralign = Alignment;
  }

  // If true, this section will be page aligned on disk.
  // Typically the first section of each PT_LOAD segment has this flag.
  bool PageAlign = false;

  // Pointer to the first section in PT_LOAD segment, which this section
  // also resides in. This field is used to correctly compute file offset
  // of a section. When two sections share the same load segment, difference
  // between their file offsets should be equal to difference between their
  // virtual addresses. To compute some section offset we use the following
  // formula: Off = Off_first + VA - VA_first.
  OutputSectionBase *FirstInPtLoad = nullptr;

  virtual void finalize() {}
  virtual void finalizePieces() {}
  virtual void assignOffsets() {}
  virtual void writeTo(uint8_t *Buf) {}
  virtual ~OutputSectionBase() = default;

  StringRef Name;

  // The following fields correspond to Elf_Shdr members.
  uint64_t Size = 0;
  uint64_t Entsize = 0;
  uint64_t Addralign = 0;
  uint64_t Offset = 0;
  uint64_t Flags = 0;
  uint64_t LMAOffset = 0;
  uint64_t Addr = 0;
  uint32_t ShName = 0;
  uint32_t Type = 0;
  uint32_t Info = 0;
  uint32_t Link = 0;
};

template <class ELFT> class GdbIndexSection final : public OutputSectionBase {
  typedef typename ELFT::uint uintX_t;

  const unsigned OffsetTypeSize = 4;
  const unsigned CuListOffset = 6 * OffsetTypeSize;
  const unsigned CompilationUnitSize = 16;
  const unsigned AddressEntrySize = 16 + OffsetTypeSize;
  const unsigned SymTabEntrySize = 2 * OffsetTypeSize;

public:
  GdbIndexSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;

  // Pairs of [CU Offset, CU length].
  std::vector<std::pair<uintX_t, uintX_t>> CompilationUnits;

private:
  void parseDebugSections();
  void readDwarf(InputSection<ELFT> *I);

  uint32_t CuTypesOffset;
};

template <class ELFT> class PltSection final : public OutputSectionBase {
  typedef typename ELFT::uint uintX_t;

public:
  PltSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  void addEntry(SymbolBody &Sym);
  bool empty() const { return Entries.empty(); }
  Kind getKind() const override { return Plt; }
  static bool classof(const OutputSectionBase *B) {
    return B->getKind() == Plt;
  }

private:
  std::vector<std::pair<const SymbolBody *, unsigned>> Entries;
};

template <class ELFT> class DynamicReloc {
  typedef typename ELFT::uint uintX_t;

public:
  DynamicReloc(uint32_t Type, const InputSectionBase<ELFT> *InputSec,
               uintX_t OffsetInSec, bool UseSymVA, SymbolBody *Sym,
               uintX_t Addend)
      : Type(Type), Sym(Sym), InputSec(InputSec), OffsetInSec(OffsetInSec),
        UseSymVA(UseSymVA), Addend(Addend) {}

  DynamicReloc(uint32_t Type, const OutputSectionBase *OutputSec,
               uintX_t OffsetInSec, bool UseSymVA, SymbolBody *Sym,
               uintX_t Addend)
      : Type(Type), Sym(Sym), OutputSec(OutputSec), OffsetInSec(OffsetInSec),
        UseSymVA(UseSymVA), Addend(Addend) {}

  uintX_t getOffset() const;
  uintX_t getAddend() const;
  uint32_t getSymIndex() const;
  const OutputSectionBase *getOutputSec() const { return OutputSec; }
  const InputSectionBase<ELFT> *getInputSec() const { return InputSec; }

  uint32_t Type;

private:
  SymbolBody *Sym;
  const InputSectionBase<ELFT> *InputSec = nullptr;
  const OutputSectionBase *OutputSec = nullptr;
  uintX_t OffsetInSec;
  bool UseSymVA;
  uintX_t Addend;
};

struct SymbolTableEntry {
  SymbolBody *Symbol;
  size_t StrTabOffset;
};

template <class ELFT>
class SymbolTableSection final : public OutputSectionBase {
  typedef OutputSectionBase Base;

public:
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::SymRange Elf_Sym_Range;
  typedef typename ELFT::uint uintX_t;
  SymbolTableSection(StringTableSection<ELFT> &StrTabSec);

  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  void addSymbol(SymbolBody *Body);
  StringTableSection<ELFT> &getStrTabSec() const { return StrTabSec; }
  unsigned getNumSymbols() const { return NumLocals + Symbols.size() + 1; }
  typename Base::Kind getKind() const override { return Base::SymTable; }
  static bool classof(const Base *B) { return B->getKind() == Base::SymTable; }

  ArrayRef<SymbolTableEntry> getSymbols() const { return Symbols; }

  unsigned NumLocals = 0;
  StringTableSection<ELFT> &StrTabSec;

private:
  void writeLocalSymbols(uint8_t *&Buf);
  void writeGlobalSymbols(uint8_t *Buf);

  const OutputSectionBase *getOutputSection(SymbolBody *Sym);

  // A vector of symbols and their string table offsets.
  std::vector<SymbolTableEntry> Symbols;
};

// For more information about .gnu.version and .gnu.version_r see:
// https://www.akkadia.org/drepper/symbol-versioning

// The .gnu.version_d section which has a section type of SHT_GNU_verdef shall
// contain symbol version definitions. The number of entries in this section
// shall be contained in the DT_VERDEFNUM entry of the .dynamic section.
// The section shall contain an array of Elf_Verdef structures, optionally
// followed by an array of Elf_Verdaux structures.
template <class ELFT>
class VersionDefinitionSection final : public OutputSectionBase {
  typedef typename ELFT::Verdef Elf_Verdef;
  typedef typename ELFT::Verdaux Elf_Verdaux;

public:
  VersionDefinitionSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  Kind getKind() const override { return VersDef; }
  static bool classof(const OutputSectionBase *B) {
    return B->getKind() == VersDef;
  }

private:
  void writeOne(uint8_t *Buf, uint32_t Index, StringRef Name, size_t NameOff);

  unsigned FileDefNameOff;
};

// The .gnu.version section specifies the required version of each symbol in the
// dynamic symbol table. It contains one Elf_Versym for each dynamic symbol
// table entry. An Elf_Versym is just a 16-bit integer that refers to a version
// identifier defined in the either .gnu.version_r or .gnu.version_d section.
// The values 0 and 1 are reserved. All other values are used for versions in
// the own object or in any of the dependencies.
template <class ELFT>
class VersionTableSection final : public OutputSectionBase {
  typedef typename ELFT::Versym Elf_Versym;

public:
  VersionTableSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  Kind getKind() const override { return VersTable; }
  static bool classof(const OutputSectionBase *B) {
    return B->getKind() == VersTable;
  }
};

// The .gnu.version_r section defines the version identifiers used by
// .gnu.version. It contains a linked list of Elf_Verneed data structures. Each
// Elf_Verneed specifies the version requirements for a single DSO, and contains
// a reference to a linked list of Elf_Vernaux data structures which define the
// mapping from version identifiers to version names.
template <class ELFT>
class VersionNeedSection final : public OutputSectionBase {
  typedef typename ELFT::Verneed Elf_Verneed;
  typedef typename ELFT::Vernaux Elf_Vernaux;

  // A vector of shared files that need Elf_Verneed data structures and the
  // string table offsets of their sonames.
  std::vector<std::pair<SharedFile<ELFT> *, size_t>> Needed;

  // The next available version identifier.
  unsigned NextIndex;

public:
  VersionNeedSection();
  void addSymbol(SharedSymbol<ELFT> *SS);
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  size_t getNeedNum() const { return Needed.size(); }
  Kind getKind() const override { return VersNeed; }
  static bool classof(const OutputSectionBase *B) {
    return B->getKind() == VersNeed;
  }
};

template <class ELFT> class RelocationSection final : public OutputSectionBase {
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Rela Elf_Rela;
  typedef typename ELFT::uint uintX_t;

public:
  RelocationSection(StringRef Name, bool Sort);
  void addReloc(const DynamicReloc<ELFT> &Reloc);
  unsigned getRelocOffset();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  bool hasRelocs() const { return !Relocs.empty(); }
  Kind getKind() const override { return Reloc; }
  size_t getRelativeRelocCount() const { return NumRelativeRelocs; }
  static bool classof(const OutputSectionBase *B) {
    return B->getKind() == Reloc;
  }

private:
  bool Sort;
  size_t NumRelativeRelocs = 0;
  std::vector<DynamicReloc<ELFT>> Relocs;
};

template <class ELFT> class OutputSection final : public OutputSectionBase {

public:
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Rela Elf_Rela;
  typedef typename ELFT::uint uintX_t;
  OutputSection(StringRef Name, uint32_t Type, uintX_t Flags);
  void addSection(InputSectionData *C) override;
  void sort(std::function<unsigned(InputSection<ELFT> *S)> Order);
  void sortInitFini();
  void sortCtorsDtors();
  void writeTo(uint8_t *Buf) override;
  void finalize() override;
  void assignOffsets() override;
  Kind getKind() const override { return Regular; }
  static bool classof(const OutputSectionBase *B) {
    return B->getKind() == Regular;
  }
  std::vector<InputSection<ELFT> *> Sections;
};

template <class ELFT>
class MergeOutputSection final : public OutputSectionBase {
  typedef typename ELFT::uint uintX_t;

public:
  MergeOutputSection(StringRef Name, uint32_t Type, uintX_t Flags,
                     uintX_t Alignment);
  void addSection(InputSectionData *S) override;
  void writeTo(uint8_t *Buf) override;
  unsigned getOffset(llvm::CachedHashStringRef Val);
  void finalize() override;
  void finalizePieces() override;
  bool shouldTailMerge() const;
  Kind getKind() const override { return Merge; }
  static bool classof(const OutputSectionBase *B) {
    return B->getKind() == Merge;
  }

private:
  llvm::StringTableBuilder Builder;
  std::vector<MergeInputSection<ELFT> *> Sections;
};

struct CieRecord {
  EhSectionPiece *Piece = nullptr;
  std::vector<EhSectionPiece *> FdePieces;
};

// Output section for .eh_frame.
template <class ELFT> class EhOutputSection final : public OutputSectionBase {
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Rela Elf_Rela;

public:
  EhOutputSection();
  void writeTo(uint8_t *Buf) override;
  void finalize() override;
  bool empty() const { return Sections.empty(); }

  void addSection(InputSectionData *S) override;
  Kind getKind() const override { return EHFrame; }
  static bool classof(const OutputSectionBase *B) {
    return B->getKind() == EHFrame;
  }

  size_t NumFdes = 0;

private:
  template <class RelTy>
  void addSectionAux(EhInputSection<ELFT> *S, llvm::ArrayRef<RelTy> Rels);

  template <class RelTy>
  CieRecord *addCie(EhSectionPiece &Piece, EhInputSection<ELFT> *Sec,
                    ArrayRef<RelTy> Rels);

  template <class RelTy>
  bool isFdeLive(EhSectionPiece &Piece, EhInputSection<ELFT> *Sec,
                 ArrayRef<RelTy> Rels);

  uintX_t getFdePc(uint8_t *Buf, size_t Off, uint8_t Enc);

  std::vector<EhInputSection<ELFT> *> Sections;
  std::vector<CieRecord *> Cies;

  // CIE records are uniquified by their contents and personality functions.
  llvm::DenseMap<std::pair<ArrayRef<uint8_t>, SymbolBody *>, CieRecord> CieMap;
};

template <class ELFT> class HashTableSection final : public OutputSectionBase {
  typedef typename ELFT::Word Elf_Word;

public:
  HashTableSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  Kind getKind() const override { return HashTable; }
  static bool classof(const OutputSectionBase *B) {
    return B->getKind() == HashTable;
  }
};

// Outputs GNU Hash section. For detailed explanation see:
// https://blogs.oracle.com/ali/entry/gnu_hash_elf_sections
template <class ELFT>
class GnuHashTableSection final : public OutputSectionBase {
  typedef typename ELFT::Off Elf_Off;
  typedef typename ELFT::Word Elf_Word;
  typedef typename ELFT::uint uintX_t;

public:
  GnuHashTableSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;

  // Adds symbols to the hash table.
  // Sorts the input to satisfy GNU hash section requirements.
  void addSymbols(std::vector<SymbolTableEntry> &Symbols);
  Kind getKind() const override { return GnuHashTable; }
  static bool classof(const OutputSectionBase *B) {
    return B->getKind() == GnuHashTable;
  }

private:
  static unsigned calcNBuckets(unsigned NumHashed);
  static unsigned calcMaskWords(unsigned NumHashed);

  void writeHeader(uint8_t *&Buf);
  void writeBloomFilter(uint8_t *&Buf);
  void writeHashTable(uint8_t *Buf);

  struct SymbolData {
    SymbolBody *Body;
    size_t STName;
    uint32_t Hash;
  };

  std::vector<SymbolData> Symbols;

  unsigned MaskWords;
  unsigned NBuckets;
  unsigned Shift2;
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
template <class ELFT> class EhFrameHeader final : public OutputSectionBase {
  typedef typename ELFT::uint uintX_t;

public:
  EhFrameHeader();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  void addFde(uint32_t Pc, uint32_t FdeVA);
  Kind getKind() const override { return EHFrameHdr; }
  static bool classof(const OutputSectionBase *B) {
    return B->getKind() == EHFrameHdr;
  }

private:
  struct FdeData {
    uint32_t Pc;
    uint32_t FdeVA;
  };

  std::vector<FdeData> Fdes;
};

// All output sections that are hadnled by the linker specially are
// globally accessible. Writer initializes them, so don't use them
// until Writer is initialized.
template <class ELFT> struct Out {
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Phdr Elf_Phdr;

  static uint8_t First;
  static EhFrameHeader<ELFT> *EhFrameHdr;
  static EhOutputSection<ELFT> *EhFrame;
  static GdbIndexSection<ELFT> *GdbIndex;
  static GnuHashTableSection<ELFT> *GnuHashTab;
  static HashTableSection<ELFT> *HashTab;
  static OutputSection<ELFT> *Bss;
  static OutputSection<ELFT> *MipsRldMap;
  static OutputSectionBase *Opd;
  static uint8_t *OpdBuf;
  static PltSection<ELFT> *Plt;
  static RelocationSection<ELFT> *RelaDyn;
  static RelocationSection<ELFT> *RelaPlt;
  static SymbolTableSection<ELFT> *DynSymTab;
  static SymbolTableSection<ELFT> *SymTab;
  static VersionDefinitionSection<ELFT> *VerDef;
  static VersionTableSection<ELFT> *VerSym;
  static VersionNeedSection<ELFT> *VerNeed;
  static Elf_Phdr *TlsPhdr;
  static OutputSectionBase *DebugInfo;
  static OutputSectionBase *ElfHeader;
  static OutputSectionBase *ProgramHeaders;
  static OutputSectionBase *PreinitArray;
  static OutputSectionBase *InitArray;
  static OutputSectionBase *FiniArray;
};

template <bool Is64Bits> struct SectionKey {
  typedef typename std::conditional<Is64Bits, uint64_t, uint32_t>::type uintX_t;
  StringRef Name;
  uint32_t Type;
  uintX_t Flags;
  uintX_t Alignment;
};

// This class knows how to create an output section for a given
// input section. Output section type is determined by various
// factors, including input section's sh_flags, sh_type and
// linker scripts.
template <class ELFT> class OutputSectionFactory {
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::uint uintX_t;
  typedef typename elf::SectionKey<ELFT::Is64Bits> Key;

public:
  std::pair<OutputSectionBase *, bool> create(InputSectionBase<ELFT> *C,
                                              StringRef OutsecName);
  std::pair<OutputSectionBase *, bool>
  create(const SectionKey<ELFT::Is64Bits> &Key, InputSectionBase<ELFT> *C);

private:
  llvm::SmallDenseMap<Key, OutputSectionBase *> Map;
};

template <class ELFT> uint64_t getHeaderSize() {
  if (Config->OFormatBinary)
    return 0;
  return Out<ELFT>::ElfHeader->Size + Out<ELFT>::ProgramHeaders->Size;
}

template <class ELFT> uint8_t Out<ELFT>::First;
template <class ELFT> EhFrameHeader<ELFT> *Out<ELFT>::EhFrameHdr;
template <class ELFT> EhOutputSection<ELFT> *Out<ELFT>::EhFrame;
template <class ELFT> GdbIndexSection<ELFT> *Out<ELFT>::GdbIndex;
template <class ELFT> GnuHashTableSection<ELFT> *Out<ELFT>::GnuHashTab;
template <class ELFT> HashTableSection<ELFT> *Out<ELFT>::HashTab;
template <class ELFT> OutputSection<ELFT> *Out<ELFT>::Bss;
template <class ELFT> OutputSection<ELFT> *Out<ELFT>::MipsRldMap;
template <class ELFT> OutputSectionBase *Out<ELFT>::Opd;
template <class ELFT> uint8_t *Out<ELFT>::OpdBuf;
template <class ELFT> PltSection<ELFT> *Out<ELFT>::Plt;
template <class ELFT> RelocationSection<ELFT> *Out<ELFT>::RelaDyn;
template <class ELFT> RelocationSection<ELFT> *Out<ELFT>::RelaPlt;
template <class ELFT> SymbolTableSection<ELFT> *Out<ELFT>::DynSymTab;
template <class ELFT> SymbolTableSection<ELFT> *Out<ELFT>::SymTab;
template <class ELFT> VersionDefinitionSection<ELFT> *Out<ELFT>::VerDef;
template <class ELFT> VersionTableSection<ELFT> *Out<ELFT>::VerSym;
template <class ELFT> VersionNeedSection<ELFT> *Out<ELFT>::VerNeed;
template <class ELFT> typename ELFT::Phdr *Out<ELFT>::TlsPhdr;
template <class ELFT> OutputSectionBase *Out<ELFT>::DebugInfo;
template <class ELFT> OutputSectionBase *Out<ELFT>::ElfHeader;
template <class ELFT> OutputSectionBase *Out<ELFT>::ProgramHeaders;
template <class ELFT> OutputSectionBase *Out<ELFT>::PreinitArray;
template <class ELFT> OutputSectionBase *Out<ELFT>::InitArray;
template <class ELFT> OutputSectionBase *Out<ELFT>::FiniArray;
} // namespace elf
} // namespace lld

namespace llvm {
template <bool Is64Bits> struct DenseMapInfo<lld::elf::SectionKey<Is64Bits>> {
  typedef typename lld::elf::SectionKey<Is64Bits> Key;

  static Key getEmptyKey();
  static Key getTombstoneKey();
  static unsigned getHashValue(const Key &Val);
  static bool isEqual(const Key &LHS, const Key &RHS);
};
}

#endif
