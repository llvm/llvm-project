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
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf {

struct PhdrEntry;
class SymbolBody;
struct EhSectionPiece;
template <class ELFT> class EhInputSection;
class InputSection;
class InputSectionBase;
template <class ELFT> class MergeInputSection;
class OutputSection;
template <class ELFT> class ObjectFile;
template <class ELFT> class SharedFile;
template <class ELFT> class SharedSymbol;
template <class ELFT> class DefinedRegular;

// This represents a section in an output file.
// It is composed of multiple InputSections.
// The writer creates multiple OutputSections and assign them unique,
// non-overlapping file offsets and VAs.
class OutputSection final {
public:
  OutputSection(StringRef Name, uint32_t Type, uint64_t Flags);

  uint64_t getLMA() const { return Addr + LMAOffset; }
  template <typename ELFT> void writeHeaderTo(typename ELFT::Shdr *SHdr);

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
  OutputSection *FirstInPtLoad = nullptr;

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

  void addSection(InputSectionBase *C);
  void sort(std::function<int(InputSectionBase *S)> Order);
  void sortInitFini();
  void sortCtorsDtors();
  template <class ELFT> void writeTo(uint8_t *Buf);
  template <class ELFT> void finalize();
  template <class ELFT> void assignOffsets();
  std::vector<InputSection *> Sections;

  // Location in the output buffer.
  uint8_t *Loc = nullptr;
};

// All output sections that are hadnled by the linker specially are
// globally accessible. Writer initializes them, so don't use them
// until Writer is initialized.
template <class ELFT> struct Out {
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Phdr Elf_Phdr;

  static uint8_t First;
  static OutputSection *Bss;
  static OutputSection *BssRelRo;
  static OutputSection *Opd;
  static uint8_t *OpdBuf;
  static PhdrEntry *TlsPhdr;
  static OutputSection *DebugInfo;
  static OutputSection *ElfHeader;
  static OutputSection *ProgramHeaders;
  static OutputSection *PreinitArray;
  static OutputSection *InitArray;
  static OutputSection *FiniArray;
};

struct SectionKey {
  StringRef Name;
  uint64_t Flags;
  uint64_t Alignment;
};
}
}
namespace llvm {
template <> struct DenseMapInfo<lld::elf::SectionKey> {
  static lld::elf::SectionKey getEmptyKey();
  static lld::elf::SectionKey getTombstoneKey();
  static unsigned getHashValue(const lld::elf::SectionKey &Val);
  static bool isEqual(const lld::elf::SectionKey &LHS,
                      const lld::elf::SectionKey &RHS);
};
}
namespace lld {
namespace elf {

// This class knows how to create an output section for a given
// input section. Output section type is determined by various
// factors, including input section's sh_flags, sh_type and
// linker scripts.
template <class ELFT> class OutputSectionFactory {
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::uint uintX_t;

public:
  OutputSectionFactory(std::vector<OutputSection *> &OutputSections);
  ~OutputSectionFactory();
  void addInputSec(InputSectionBase *IS, StringRef OutsecName);

private:
  llvm::SmallDenseMap<SectionKey, OutputSection *> Map;
  std::vector<OutputSection *> &OutputSections;
};

template <class ELFT> uint64_t getHeaderSize() {
  if (Config->OFormatBinary)
    return 0;
  return Out<ELFT>::ElfHeader->Size + Out<ELFT>::ProgramHeaders->Size;
}

template <class ELFT> uint8_t Out<ELFT>::First;
template <class ELFT> OutputSection *Out<ELFT>::Bss;
template <class ELFT> OutputSection *Out<ELFT>::BssRelRo;
template <class ELFT> OutputSection *Out<ELFT>::Opd;
template <class ELFT> uint8_t *Out<ELFT>::OpdBuf;
template <class ELFT> PhdrEntry *Out<ELFT>::TlsPhdr;
template <class ELFT> OutputSection *Out<ELFT>::DebugInfo;
template <class ELFT> OutputSection *Out<ELFT>::ElfHeader;
template <class ELFT> OutputSection *Out<ELFT>::ProgramHeaders;
template <class ELFT> OutputSection *Out<ELFT>::PreinitArray;
template <class ELFT> OutputSection *Out<ELFT>::InitArray;
template <class ELFT> OutputSection *Out<ELFT>::FiniArray;
} // namespace elf
} // namespace lld


#endif
