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
#include "Relocations.h"

#include "lld/Core/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/SHA1.h"

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
template <class ELFT> class MipsReginfoInputSection;
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
template <class ELFT> class OutputSectionBase {
public:
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Shdr Elf_Shdr;
  enum Kind {
    Base,
    BuildId,
    Dynamic,
    EHFrame,
    EHFrameHdr,
    GnuHashTable,
    Got,
    GotPlt,
    HashTable,
    Interp,
    Merge,
    MipsReginfo,
    MipsOptions,
    MipsAbiFlags,
    Plt,
    Regular,
    Reloc,
    StrTable,
    SymTable,
    VersDef,
    VersNeed,
    VersTable
  };

  OutputSectionBase(StringRef Name, uint32_t Type, uintX_t Flags);
  void setVA(uintX_t VA) { Header.sh_addr = VA; }
  uintX_t getVA() const { return Header.sh_addr; }
  void setFileOffset(uintX_t Off) { Header.sh_offset = Off; }
  void setSHName(unsigned Val) { Header.sh_name = Val; }
  void writeHeaderTo(Elf_Shdr *SHdr);
  StringRef getName() { return Name; }

  virtual void addSection(InputSectionBase<ELFT> *C) {}
  virtual Kind getKind() const { return Base; }
  static bool classof(const OutputSectionBase<ELFT> *B) {
    return B->getKind() == Base;
  }

  unsigned SectionIndex;

  // Returns the size of the section in the output file.
  uintX_t getSize() const { return Header.sh_size; }
  void setSize(uintX_t Val) { Header.sh_size = Val; }
  uintX_t getFlags() const { return Header.sh_flags; }
  uint32_t getPhdrFlags() const;
  uintX_t getFileOff() const { return Header.sh_offset; }
  uintX_t getAlignment() const { return Header.sh_addralign; }
  uint32_t getType() const { return Header.sh_type; }

  void updateAlignment(uintX_t Alignment) {
    if (Alignment > Header.sh_addralign)
      Header.sh_addralign = Alignment;
  }

  // If true, this section will be page aligned on disk.
  // Typically the first section of each PT_LOAD segment has this flag.
  bool PageAlign = false;

  virtual void finalize() {}
  virtual void finalizePieces() {}
  virtual void assignOffsets() {}
  virtual void writeTo(uint8_t *Buf) {}
  virtual ~OutputSectionBase() = default;

protected:
  StringRef Name;
  Elf_Shdr Header;
};

template <class ELFT> class GotSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;
  typedef typename ELFT::uint uintX_t;

public:
  GotSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  void addEntry(SymbolBody &Sym);
  void addMipsEntry(SymbolBody &Sym, uintX_t Addend, RelExpr Expr);
  bool addDynTlsEntry(SymbolBody &Sym);
  bool addTlsIndex();
  bool empty() const { return MipsPageEntries == 0 && Entries.empty(); }
  uintX_t getMipsLocalPageOffset(uintX_t Addr);
  uintX_t getMipsGotOffset(const SymbolBody &B, uintX_t Addend) const;
  uintX_t getGlobalDynAddr(const SymbolBody &B) const;
  uintX_t getGlobalDynOffset(const SymbolBody &B) const;
  uintX_t getNumEntries() const { return Entries.size(); }
  typename Base::Kind getKind() const override { return Base::Got; }
  static bool classof(const Base *B) { return B->getKind() == Base::Got; }

  // Returns the symbol which corresponds to the first entry of the global part
  // of GOT on MIPS platform. It is required to fill up MIPS-specific dynamic
  // table properties.
  // Returns nullptr if the global part is empty.
  const SymbolBody *getMipsFirstGlobalEntry() const;

  // Returns the number of entries in the local part of GOT including
  // the number of reserved entries. This method is MIPS-specific.
  unsigned getMipsLocalEntriesNum() const;

  // Returns offset of TLS part of the MIPS GOT table. This part goes
  // after 'local' and 'global' entries.
  uintX_t getMipsTlsOffset();

  uintX_t getTlsIndexVA() { return Base::getVA() + TlsIndexOff; }
  uint32_t getTlsIndexOff() { return TlsIndexOff; }

  // Flag to force GOT to be in output if we have relocations
  // that relies on its address.
  bool HasGotOffRel = false;

private:
  std::vector<const SymbolBody *> Entries;
  uint32_t TlsIndexOff = -1;
  uint32_t MipsPageEntries = 0;
  // Output sections referenced by MIPS GOT relocations.
  llvm::SmallPtrSet<const OutputSectionBase<ELFT> *, 10> MipsOutSections;
  llvm::DenseMap<uintX_t, size_t> MipsLocalGotPos;

  // MIPS ABI requires to create unique GOT entry for each Symbol/Addend
  // pairs. The `MipsGotMap` maps (S,A) pair to the GOT index in the `MipsLocal`
  // or `MipsGlobal` vectors. In general it does not have a sence to take in
  // account addend for preemptible symbols because the corresponding
  // GOT entries should have one-to-one mapping with dynamic symbols table.
  // But we use the same container's types for both kind of GOT entries
  // to handle them uniformly.
  typedef std::pair<const SymbolBody*, uintX_t> MipsGotEntry;
  typedef std::vector<MipsGotEntry> MipsGotEntries;
  llvm::DenseMap<MipsGotEntry, size_t> MipsGotMap;
  MipsGotEntries MipsLocal;
  MipsGotEntries MipsGlobal;

  // Write MIPS-specific parts of the GOT.
  void writeMipsGot(uint8_t *&Buf);
};

template <class ELFT>
class GotPltSection final : public OutputSectionBase<ELFT> {
  typedef typename ELFT::uint uintX_t;
  typedef OutputSectionBase<ELFT> Base;

public:
  GotPltSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  void addEntry(SymbolBody &Sym);
  bool empty() const;
  typename Base::Kind getKind() const override { return Base::GotPlt; }
  static bool classof(const Base *B) { return B->getKind() == Base::GotPlt; }

private:
  std::vector<const SymbolBody *> Entries;
};

template <class ELFT> class PltSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;
  typedef typename ELFT::uint uintX_t;

public:
  PltSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  void addEntry(SymbolBody &Sym);
  bool empty() const { return Entries.empty(); }
  typename Base::Kind getKind() const override { return Base::Plt; }
  static bool classof(const Base *B) { return B->getKind() == Base::Plt; }

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

  DynamicReloc(uint32_t Type, const OutputSectionBase<ELFT> *OutputSec,
               uintX_t OffsetInSec, bool UseSymVA, SymbolBody *Sym,
               uintX_t Addend)
      : Type(Type), Sym(Sym), OutputSec(OutputSec), OffsetInSec(OffsetInSec),
        UseSymVA(UseSymVA), Addend(Addend) {}

  uintX_t getOffset() const;
  uintX_t getAddend() const;
  uint32_t getSymIndex() const;
  const OutputSectionBase<ELFT> *getOutputSec() const { return OutputSec; }

  uint32_t Type;

private:
  SymbolBody *Sym;
  const InputSectionBase<ELFT> *InputSec = nullptr;
  const OutputSectionBase<ELFT> *OutputSec = nullptr;
  uintX_t OffsetInSec;
  bool UseSymVA;
  uintX_t Addend;
};

template <class ELFT>
class SymbolTableSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;

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

  ArrayRef<std::pair<SymbolBody *, size_t>> getSymbols() const {
    return Symbols;
  }

  unsigned NumLocals = 0;
  StringTableSection<ELFT> &StrTabSec;

private:
  void writeLocalSymbols(uint8_t *&Buf);
  void writeGlobalSymbols(uint8_t *Buf);

  const OutputSectionBase<ELFT> *getOutputSection(SymbolBody *Sym);

  // A vector of symbols and their string table offsets.
  std::vector<std::pair<SymbolBody *, size_t>> Symbols;
};

// For more information about .gnu.version and .gnu.version_r see:
// https://www.akkadia.org/drepper/symbol-versioning

// The .gnu.version_d section which has a section type of SHT_GNU_verdef shall
// contain symbol version definitions. The number of entries in this section
// shall be contained in the DT_VERDEFNUM entry of the .dynamic section.
// The section shall contain an array of Elf_Verdef structures, optionally
// followed by an array of Elf_Verdaux structures.
template <class ELFT>
class VersionDefinitionSection final : public OutputSectionBase<ELFT> {
  typedef typename ELFT::Verdef Elf_Verdef;
  typedef typename ELFT::Verdaux Elf_Verdaux;
  typedef OutputSectionBase<ELFT> Base;

public:
  VersionDefinitionSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  typename Base::Kind getKind() const override { return Base::VersDef; }
  static bool classof(const Base *B) { return B->getKind() == Base::VersDef; }

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
class VersionTableSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;
  typedef typename ELFT::Versym Elf_Versym;

public:
  VersionTableSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  typename Base::Kind getKind() const override { return Base::VersTable; }
  static bool classof(const Base *B) { return B->getKind() == Base::VersTable; }
};

// The .gnu.version_r section defines the version identifiers used by
// .gnu.version. It contains a linked list of Elf_Verneed data structures. Each
// Elf_Verneed specifies the version requirements for a single DSO, and contains
// a reference to a linked list of Elf_Vernaux data structures which define the
// mapping from version identifiers to version names.
template <class ELFT>
class VersionNeedSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;
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
  typename Base::Kind getKind() const override { return Base::VersNeed; }
  static bool classof(const Base *B) { return B->getKind() == Base::VersNeed; }
};

template <class ELFT>
class RelocationSection final : public OutputSectionBase<ELFT> {
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Rela Elf_Rela;
  typedef typename ELFT::uint uintX_t;
  typedef OutputSectionBase<ELFT> Base;

public:
  RelocationSection(StringRef Name, bool Sort);
  void addReloc(const DynamicReloc<ELFT> &Reloc);
  unsigned getRelocOffset();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  bool hasRelocs() const { return !Relocs.empty(); }
  typename Base::Kind getKind() const override { return Base::Reloc; }
  size_t getRelativeRelocCount() const { return NumRelativeRelocs; }
  static bool classof(const Base *B) { return B->getKind() == Base::Reloc; }

private:
  bool Sort;
  size_t NumRelativeRelocs = 0;
  std::vector<DynamicReloc<ELFT>> Relocs;
};

template <class ELFT>
class OutputSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;

public:
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Rela Elf_Rela;
  typedef typename ELFT::uint uintX_t;
  OutputSection(StringRef Name, uint32_t Type, uintX_t Flags);
  void addSection(InputSectionBase<ELFT> *C) override;
  void sortInitFini();
  void sortCtorsDtors();
  void writeTo(uint8_t *Buf) override;
  void finalize() override;
  void assignOffsets() override;
  typename Base::Kind getKind() const override { return Base::Regular; }
  static bool classof(const Base *B) { return B->getKind() == Base::Regular; }
  std::vector<InputSection<ELFT> *> Sections;
};

template <class ELFT>
class MergeOutputSection final : public OutputSectionBase<ELFT> {
  typedef typename ELFT::uint uintX_t;
  typedef OutputSectionBase<ELFT> Base;

public:
  MergeOutputSection(StringRef Name, uint32_t Type, uintX_t Flags,
                     uintX_t Alignment);
  void addSection(InputSectionBase<ELFT> *S) override;
  void writeTo(uint8_t *Buf) override;
  unsigned getOffset(StringRef Val);
  void finalize() override;
  void finalizePieces() override;
  bool shouldTailMerge() const;
  typename Base::Kind getKind() const override { return Base::Merge; }
  static bool classof(const Base *B) { return B->getKind() == Base::Merge; }

private:
  llvm::StringTableBuilder Builder;
  std::vector<MergeInputSection<ELFT> *> Sections;
};

struct CieRecord {
  EhSectionPiece *Piece = nullptr;
  std::vector<EhSectionPiece *> FdePieces;
};

// Output section for .eh_frame.
template <class ELFT>
class EhOutputSection final : public OutputSectionBase<ELFT> {
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Rela Elf_Rela;
  typedef OutputSectionBase<ELFT> Base;

public:
  EhOutputSection();
  void writeTo(uint8_t *Buf) override;
  void finalize() override;
  bool empty() const { return Sections.empty(); }

  void addSection(InputSectionBase<ELFT> *S) override;
  typename Base::Kind getKind() const override { return Base::EHFrame; }
  static bool classof(const Base *B) { return B->getKind() == Base::EHFrame; }

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

template <class ELFT>
class InterpSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;

public:
  InterpSection();
  void writeTo(uint8_t *Buf) override;
  typename Base::Kind getKind() const override { return Base::Interp; }
  static bool classof(const Base *B) { return B->getKind() == Base::Interp; }
};

template <class ELFT>
class StringTableSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;

public:
  typedef typename ELFT::uint uintX_t;
  StringTableSection(StringRef Name, bool Dynamic);
  unsigned addString(StringRef S, bool HashIt = true);
  void writeTo(uint8_t *Buf) override;
  unsigned getSize() const { return Size; }
  void finalize() override { this->Header.sh_size = getSize(); }
  bool isDynamic() const { return Dynamic; }
  typename Base::Kind getKind() const override { return Base::StrTable; }
  static bool classof(const Base *B) { return B->getKind() == Base::StrTable; }

private:
  const bool Dynamic;
  llvm::DenseMap<StringRef, unsigned> StringMap;
  std::vector<StringRef> Strings;
  unsigned Size = 1; // ELF string tables start with a NUL byte, so 1.
};

template <class ELFT>
class HashTableSection final : public OutputSectionBase<ELFT> {
  typedef typename ELFT::Word Elf_Word;
  typedef OutputSectionBase<ELFT> Base;

public:
  HashTableSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  typename Base::Kind getKind() const override { return Base::HashTable; }
  static bool classof(const Base *B) { return B->getKind() == Base::HashTable; }
};

// Outputs GNU Hash section. For detailed explanation see:
// https://blogs.oracle.com/ali/entry/gnu_hash_elf_sections
template <class ELFT>
class GnuHashTableSection final : public OutputSectionBase<ELFT> {
  typedef typename ELFT::Off Elf_Off;
  typedef typename ELFT::Word Elf_Word;
  typedef typename ELFT::uint uintX_t;
  typedef OutputSectionBase<ELFT> Base;

public:
  GnuHashTableSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;

  // Adds symbols to the hash table.
  // Sorts the input to satisfy GNU hash section requirements.
  void addSymbols(std::vector<std::pair<SymbolBody *, size_t>> &Symbols);
  typename Base::Kind getKind() const override { return Base::GnuHashTable; }
  static bool classof(const Base *B) {
    return B->getKind() == Base::GnuHashTable;
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

template <class ELFT>
class DynamicSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;
  typedef typename ELFT::Dyn Elf_Dyn;
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Rela Elf_Rela;
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::uint uintX_t;

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
    enum KindT { SecAddr, SecSize, SymAddr, PlainInt } Kind;
    Entry(int32_t Tag, OutputSectionBase<ELFT> *OutSec, KindT Kind = SecAddr)
        : Tag(Tag), OutSec(OutSec), Kind(Kind) {}
    Entry(int32_t Tag, uint64_t Val) : Tag(Tag), Val(Val), Kind(PlainInt) {}
    Entry(int32_t Tag, const SymbolBody *Sym)
        : Tag(Tag), Sym(Sym), Kind(SymAddr) {}
  };

  // finalize() fills this vector with the section contents. finalize()
  // cannot directly create final section contents because when the
  // function is called, symbol or section addresses are not fixed yet.
  std::vector<Entry> Entries;

public:
  explicit DynamicSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  typename Base::Kind getKind() const override { return Base::Dynamic; }
  static bool classof(const Base *B) { return B->getKind() == Base::Dynamic; }
};

template <class ELFT>
class MipsReginfoOutputSection final : public OutputSectionBase<ELFT> {
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;
  typedef OutputSectionBase<ELFT> Base;

public:
  MipsReginfoOutputSection();
  void writeTo(uint8_t *Buf) override;
  void addSection(InputSectionBase<ELFT> *S) override;
  typename Base::Kind getKind() const override { return Base::MipsReginfo; }
  static bool classof(const Base *B) {
    return B->getKind() == Base::MipsReginfo;
  }

private:
  uint32_t GprMask = 0;
};

template <class ELFT>
class MipsOptionsOutputSection final : public OutputSectionBase<ELFT> {
  typedef llvm::object::Elf_Mips_Options<ELFT> Elf_Mips_Options;
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;
  typedef OutputSectionBase<ELFT> Base;

public:
  MipsOptionsOutputSection();
  void writeTo(uint8_t *Buf) override;
  void addSection(InputSectionBase<ELFT> *S) override;
  typename Base::Kind getKind() const override { return Base::MipsOptions; }
  static bool classof(const Base *B) {
    return B->getKind() == Base::MipsOptions;
  }

private:
  uint32_t GprMask = 0;
};

template <class ELFT>
class MipsAbiFlagsOutputSection final : public OutputSectionBase<ELFT> {
  typedef llvm::object::Elf_Mips_ABIFlags<ELFT> Elf_Mips_ABIFlags;
  typedef OutputSectionBase<ELFT> Base;

public:
  MipsAbiFlagsOutputSection();
  void writeTo(uint8_t *Buf) override;
  void addSection(InputSectionBase<ELFT> *S) override;
  typename Base::Kind getKind() const override { return Base::MipsAbiFlags; }
  static bool classof(const Base *B) {
    return B->getKind() == Base::MipsAbiFlags;
  }

private:
  Elf_Mips_ABIFlags Flags;
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
  typedef typename ELFT::uint uintX_t;
  typedef OutputSectionBase<ELFT> Base;

public:
  EhFrameHeader();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  void addFde(uint32_t Pc, uint32_t FdeVA);
  typename Base::Kind getKind() const override { return Base::EHFrameHdr; }
  static bool classof(const Base *B) {
    return B->getKind() == Base::EHFrameHdr;
  }

private:
  struct FdeData {
    uint32_t Pc;
    uint32_t FdeVA;
  };

  std::vector<FdeData> Fdes;
};

template <class ELFT> class BuildIdSection : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;

public:
  void writeTo(uint8_t *Buf) override;
  virtual void writeBuildId(ArrayRef<ArrayRef<uint8_t>> Bufs) = 0;
  typename Base::Kind getKind() const override { return Base::BuildId; }
  static bool classof(const Base *B) { return B->getKind() == Base::BuildId; }

protected:
  BuildIdSection(size_t HashSize);
  size_t HashSize;
  uint8_t *HashBuf = nullptr;
};

template <class ELFT> class BuildIdFnv1 final : public BuildIdSection<ELFT> {
public:
  BuildIdFnv1() : BuildIdSection<ELFT>(8) {}
  void writeBuildId(ArrayRef<ArrayRef<uint8_t>> Bufs) override;
};

template <class ELFT> class BuildIdMd5 final : public BuildIdSection<ELFT> {
public:
  BuildIdMd5() : BuildIdSection<ELFT>(16) {}
  void writeBuildId(ArrayRef<ArrayRef<uint8_t>> Bufs) override;
};

template <class ELFT> class BuildIdSha1 final : public BuildIdSection<ELFT> {
public:
  BuildIdSha1() : BuildIdSection<ELFT>(20) {}
  void writeBuildId(ArrayRef<ArrayRef<uint8_t>> Bufs) override;
};

template <class ELFT> class BuildIdUuid final : public BuildIdSection<ELFT> {
public:
  BuildIdUuid() : BuildIdSection<ELFT>(16) {}
  void writeBuildId(ArrayRef<ArrayRef<uint8_t>> Bufs) override;
};

template <class ELFT>
class BuildIdHexstring final : public BuildIdSection<ELFT> {
public:
  BuildIdHexstring();
  void writeBuildId(ArrayRef<ArrayRef<uint8_t>> Bufs) override;
};

// All output sections that are hadnled by the linker specially are
// globally accessible. Writer initializes them, so don't use them
// until Writer is initialized.
template <class ELFT> struct Out {
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Phdr Elf_Phdr;
  static BuildIdSection<ELFT> *BuildId;
  static DynamicSection<ELFT> *Dynamic;
  static EhFrameHeader<ELFT> *EhFrameHdr;
  static EhOutputSection<ELFT> *EhFrame;
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
  static VersionDefinitionSection<ELFT> *VerDef;
  static VersionTableSection<ELFT> *VerSym;
  static VersionNeedSection<ELFT> *VerNeed;
  static Elf_Phdr *TlsPhdr;
  static OutputSectionBase<ELFT> *ElfHeader;
  static OutputSectionBase<ELFT> *ProgramHeaders;

  static OutputSectionBase<ELFT> *PreinitArray;
  static OutputSectionBase<ELFT> *InitArray;
  static OutputSectionBase<ELFT> *FiniArray;

  // This pool owns dynamically-allocated output sections.
  static std::vector<std::unique_ptr<OutputSectionBase<ELFT>>> Pool;
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
  std::pair<OutputSectionBase<ELFT> *, bool> create(InputSectionBase<ELFT> *C,
                                                    StringRef OutsecName);

private:
  Key createKey(InputSectionBase<ELFT> *C, StringRef OutsecName);

  llvm::SmallDenseMap<Key, OutputSectionBase<ELFT> *> Map;
};

template <class ELFT> BuildIdSection<ELFT> *Out<ELFT>::BuildId;
template <class ELFT> DynamicSection<ELFT> *Out<ELFT>::Dynamic;
template <class ELFT> EhFrameHeader<ELFT> *Out<ELFT>::EhFrameHdr;
template <class ELFT> EhOutputSection<ELFT> *Out<ELFT>::EhFrame;
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
template <class ELFT> VersionDefinitionSection<ELFT> *Out<ELFT>::VerDef;
template <class ELFT> VersionTableSection<ELFT> *Out<ELFT>::VerSym;
template <class ELFT> VersionNeedSection<ELFT> *Out<ELFT>::VerNeed;
template <class ELFT> typename ELFT::Phdr *Out<ELFT>::TlsPhdr;
template <class ELFT> OutputSectionBase<ELFT> *Out<ELFT>::ElfHeader;
template <class ELFT> OutputSectionBase<ELFT> *Out<ELFT>::ProgramHeaders;
template <class ELFT> OutputSectionBase<ELFT> *Out<ELFT>::PreinitArray;
template <class ELFT> OutputSectionBase<ELFT> *Out<ELFT>::InitArray;
template <class ELFT> OutputSectionBase<ELFT> *Out<ELFT>::FiniArray;

template <class ELFT>
std::vector<std::unique_ptr<OutputSectionBase<ELFT>>> Out<ELFT>::Pool;

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
