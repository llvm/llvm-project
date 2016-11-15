//===- SyntheticSection.h ---------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_SYNTHETIC_SECTION_H
#define LLD_ELF_SYNTHETIC_SECTION_H

#include "InputSection.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace lld {
namespace elf {

// .MIPS.abiflags section.
template <class ELFT>
class MipsAbiFlagsSection final : public InputSection<ELFT> {
  typedef llvm::object::Elf_Mips_ABIFlags<ELFT> Elf_Mips_ABIFlags;

public:
  MipsAbiFlagsSection();

private:
  Elf_Mips_ABIFlags Flags = {};
};

// .MIPS.options section.
template <class ELFT>
class MipsOptionsSection final : public InputSection<ELFT> {
  typedef llvm::object::Elf_Mips_Options<ELFT> Elf_Mips_Options;
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;

public:
  MipsOptionsSection();
  void finalize();

private:
  std::vector<uint8_t> Buf;

  Elf_Mips_Options *getOptions() {
    return reinterpret_cast<Elf_Mips_Options *>(Buf.data());
  }
};

// MIPS .reginfo section.
template <class ELFT>
class MipsReginfoSection final : public InputSection<ELFT> {
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;

public:
  MipsReginfoSection();
  void finalize();

private:
  Elf_Mips_RegInfo Reginfo = {};
};

template <class ELFT> class SyntheticSection : public InputSection<ELFT> {
  typedef typename ELFT::uint uintX_t;

public:
  SyntheticSection(uintX_t Flags, uint32_t Type, uintX_t Addralign,
                   StringRef Name)
      : InputSection<ELFT>(Flags, Type, Addralign, ArrayRef<uint8_t>(), Name,
                           InputSectionData::Synthetic) {
    this->Live = true;
  }

  virtual void writeTo(uint8_t *Buf) = 0;
  virtual size_t getSize() const { return this->Data.size(); }
  virtual void finalize() {}
  uintX_t getVA() const {
    return this->OutSec ? this->OutSec->Addr + this->OutSecOff : 0;
  }

  static bool classof(const InputSectionData *D) {
    return D->kind() == InputSectionData::Synthetic;
  }

protected:
  ~SyntheticSection() = default;
};

// .note.gnu.build-id section.
template <class ELFT> class BuildIdSection : public InputSection<ELFT> {
public:
  virtual void writeBuildId(llvm::MutableArrayRef<uint8_t> Buf) = 0;
  virtual ~BuildIdSection() = default;

  uint8_t *getOutputLoc(uint8_t *Start) const;

protected:
  BuildIdSection(size_t HashSize);
  std::vector<uint8_t> Buf;

  void
  computeHash(llvm::MutableArrayRef<uint8_t> Buf,
              std::function<void(ArrayRef<uint8_t> Arr, uint8_t *Hash)> Hash);

  size_t HashSize;
  // First 16 bytes are a header.
  static const unsigned HeaderSize = 16;
};

template <class ELFT>
class BuildIdFastHash final : public BuildIdSection<ELFT> {
public:
  BuildIdFastHash() : BuildIdSection<ELFT>(8) {}
  void writeBuildId(llvm::MutableArrayRef<uint8_t> Buf) override;
};

template <class ELFT> class BuildIdMd5 final : public BuildIdSection<ELFT> {
public:
  BuildIdMd5() : BuildIdSection<ELFT>(16) {}
  void writeBuildId(llvm::MutableArrayRef<uint8_t> Buf) override;
};

template <class ELFT> class BuildIdSha1 final : public BuildIdSection<ELFT> {
public:
  BuildIdSha1() : BuildIdSection<ELFT>(20) {}
  void writeBuildId(llvm::MutableArrayRef<uint8_t> Buf) override;
};

template <class ELFT> class BuildIdUuid final : public BuildIdSection<ELFT> {
public:
  BuildIdUuid() : BuildIdSection<ELFT>(16) {}
  void writeBuildId(llvm::MutableArrayRef<uint8_t> Buf) override;
};

template <class ELFT>
class BuildIdHexstring final : public BuildIdSection<ELFT> {
public:
  BuildIdHexstring();
  void writeBuildId(llvm::MutableArrayRef<uint8_t>) override;
};

template <class ELFT> class GotSection final : public SyntheticSection<ELFT> {
  typedef typename ELFT::uint uintX_t;

public:
  GotSection();
  void writeTo(uint8_t *Buf) override;
  size_t getSize() const override { return Size; }
  void finalize() override;
  void addEntry(SymbolBody &Sym);
  void addMipsEntry(SymbolBody &Sym, uintX_t Addend, RelExpr Expr);
  bool addDynTlsEntry(SymbolBody &Sym);
  bool addTlsIndex();
  bool empty() const { return MipsPageEntries == 0 && Entries.empty(); }
  uintX_t getMipsLocalPageOffset(uintX_t Addr);
  uintX_t getMipsGotOffset(const SymbolBody &B, uintX_t Addend) const;
  uintX_t getGlobalDynAddr(const SymbolBody &B) const;
  uintX_t getGlobalDynOffset(const SymbolBody &B) const;

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
  uintX_t getMipsTlsOffset() const;

  uintX_t getTlsIndexVA() { return this->getVA() + TlsIndexOff; }
  uint32_t getTlsIndexOff() const { return TlsIndexOff; }

  // Flag to force GOT to be in output if we have relocations
  // that relies on its address.
  bool HasGotOffRel = false;

private:
  std::vector<const SymbolBody *> Entries;
  uint32_t TlsIndexOff = -1;
  uint32_t MipsPageEntries = 0;
  uintX_t Size = 0;
  // Output sections referenced by MIPS GOT relocations.
  llvm::SmallPtrSet<const OutputSectionBase *, 10> MipsOutSections;
  llvm::DenseMap<uintX_t, size_t> MipsLocalGotPos;

  // MIPS ABI requires to create unique GOT entry for each Symbol/Addend
  // pairs. The `MipsGotMap` maps (S,A) pair to the GOT index in the `MipsLocal`
  // or `MipsGlobal` vectors. In general it does not have a sence to take in
  // account addend for preemptible symbols because the corresponding
  // GOT entries should have one-to-one mapping with dynamic symbols table.
  // But we use the same container's types for both kind of GOT entries
  // to handle them uniformly.
  typedef std::pair<const SymbolBody *, uintX_t> MipsGotEntry;
  typedef std::vector<MipsGotEntry> MipsGotEntries;
  llvm::DenseMap<MipsGotEntry, size_t> MipsGotMap;
  MipsGotEntries MipsLocal;
  MipsGotEntries MipsLocal32;
  MipsGotEntries MipsGlobal;

  // Write MIPS-specific parts of the GOT.
  void writeMipsGot(uint8_t *Buf);
};

template <class ELFT>
class GotPltSection final : public SyntheticSection<ELFT> {
  typedef typename ELFT::uint uintX_t;

public:
  GotPltSection();
  void addEntry(SymbolBody &Sym);
  bool empty() const;
  size_t getSize() const override;
  void writeTo(uint8_t *Buf) override;

private:
  std::vector<const SymbolBody *> Entries;
};

template <class ELFT>
class StringTableSection final : public SyntheticSection<ELFT> {
public:
  typedef typename ELFT::uint uintX_t;
  StringTableSection(StringRef Name, bool Dynamic);
  unsigned addString(StringRef S, bool HashIt = true);
  void writeTo(uint8_t *Buf) override;
  size_t getSize() const override { return Size; }
  bool isDynamic() const { return Dynamic; }

private:
  const bool Dynamic;

  // ELF string tables start with a NUL byte, so 1.
  uintX_t Size = 1;

  llvm::DenseMap<StringRef, unsigned> StringMap;
  std::vector<StringRef> Strings;
};

template <class ELFT>
class DynamicSection final : public SyntheticSection<ELFT> {
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
      OutputSectionBase *OutSec;
      InputSection<ELFT> *InSec;
      uint64_t Val;
      const SymbolBody *Sym;
    };
    enum KindT { SecAddr, SecSize, SymAddr, PlainInt, InSecAddr } Kind;
    Entry(int32_t Tag, OutputSectionBase *OutSec, KindT Kind = SecAddr)
        : Tag(Tag), OutSec(OutSec), Kind(Kind) {}
    Entry(int32_t Tag, InputSection<ELFT> *Sec)
        : Tag(Tag), InSec(Sec), Kind(InSecAddr) {}
    Entry(int32_t Tag, uint64_t Val) : Tag(Tag), Val(Val), Kind(PlainInt) {}
    Entry(int32_t Tag, const SymbolBody *Sym)
        : Tag(Tag), Sym(Sym), Kind(SymAddr) {}
  };

  // finalize() fills this vector with the section contents. finalize()
  // cannot directly create final section contents because when the
  // function is called, symbol or section addresses are not fixed yet.
  std::vector<Entry> Entries;

public:
  DynamicSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  size_t getSize() const override { return Size; }

private:
  void addEntries();
  void Add(Entry E) { Entries.push_back(E); }
  uintX_t Size = 0;
};

template <class ELFT> InputSection<ELFT> *createCommonSection();
template <class ELFT> InputSection<ELFT> *createInterpSection();
template <class ELFT> MergeInputSection<ELFT> *createCommentSection();

// Linker generated sections which can be used as inputs.
template <class ELFT> struct In {
  static BuildIdSection<ELFT> *BuildId;
  static InputSection<ELFT> *Common;
  static DynamicSection<ELFT> *Dynamic;
  static StringTableSection<ELFT> *DynStrTab;
  static GotSection<ELFT> *Got;
  static GotPltSection<ELFT> *GotPlt;
  static InputSection<ELFT> *Interp;
  static MipsAbiFlagsSection<ELFT> *MipsAbiFlags;
  static MipsOptionsSection<ELFT> *MipsOptions;
  static MipsReginfoSection<ELFT> *MipsReginfo;
  static StringTableSection<ELFT> *ShStrTab;
  static StringTableSection<ELFT> *StrTab;
};

template <class ELFT> BuildIdSection<ELFT> *In<ELFT>::BuildId;
template <class ELFT> InputSection<ELFT> *In<ELFT>::Common;
template <class ELFT> DynamicSection<ELFT> *In<ELFT>::Dynamic;
template <class ELFT> StringTableSection<ELFT> *In<ELFT>::DynStrTab;
template <class ELFT> GotSection<ELFT> *In<ELFT>::Got;
template <class ELFT> GotPltSection<ELFT> *In<ELFT>::GotPlt;
template <class ELFT> InputSection<ELFT> *In<ELFT>::Interp;
template <class ELFT> MipsAbiFlagsSection<ELFT> *In<ELFT>::MipsAbiFlags;
template <class ELFT> MipsOptionsSection<ELFT> *In<ELFT>::MipsOptions;
template <class ELFT> MipsReginfoSection<ELFT> *In<ELFT>::MipsReginfo;
template <class ELFT> StringTableSection<ELFT> *In<ELFT>::ShStrTab;
template <class ELFT> StringTableSection<ELFT> *In<ELFT>::StrTab;
} // namespace elf
} // namespace lld

#endif
