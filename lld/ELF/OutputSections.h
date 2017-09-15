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
#include "InputSection.h"
#include "LinkerScript.h"
#include "Relocations.h"

#include "lld/Core/LLVM.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf {

struct PhdrEntry;
class SymbolBody;
struct EhSectionPiece;
class EhInputSection;
class InputSection;
class InputSectionBase;
class MergeInputSection;
class OutputSection;
template <class ELFT> class ObjFile;
template <class ELFT> class SharedFile;
class SharedSymbol;
class DefinedRegular;

// This represents a section in an output file.
// It is composed of multiple InputSections.
// The writer creates multiple OutputSections and assign them unique,
// non-overlapping file offsets and VAs.
class OutputSection final : public BaseCommand, public SectionBase {
public:
  OutputSection(StringRef Name, uint32_t Type, uint64_t Flags);

  static bool classof(const SectionBase *S) {
    return S->kind() == SectionBase::Output;
  }
  static bool classof(const BaseCommand *C);

  uint64_t getLMA() const { return Addr + LMAOffset; }
  template <typename ELFT> void writeHeaderTo(typename ELFT::Shdr *SHdr);

  unsigned SectionIndex;
  unsigned SortRank;

  uint32_t getPhdrFlags() const;

  void updateAlignment(uint32_t Val) {
    if (Val > Alignment)
      Alignment = Val;
  }

  // Pointer to the PT_LOAD segment, which this section resides in. This field
  // is used to correctly compute file offset of a section. When two sections
  // share the same load segment, difference between their file offsets should
  // be equal to difference between their virtual addresses. To compute some
  // section offset we use the following formula: Off = Off_first + VA -
  // VA_first, where Off_first and VA_first is file offset and VA of first
  // section in PT_LOAD.
  PhdrEntry *PtLoad = nullptr;

  // Pointer to a relocation section for this section. Usually nullptr because
  // we consume relocations, but if --emit-relocs is specified (which is rare),
  // it may have a non-null value.
  OutputSection *RelocationSection = nullptr;

  // The following fields correspond to Elf_Shdr members.
  uint64_t Size = 0;
  uint64_t Offset = 0;
  uint64_t LMAOffset = 0;
  uint64_t Addr = 0;
  uint32_t ShName = 0;

  void addSection(InputSection *S);

  // Used for implementation of --compress-debug-sections option.
  std::vector<uint8_t> ZDebugHeader;
  llvm::SmallVector<char, 1> CompressedData;

  // Location in the output buffer.
  uint8_t *Loc = nullptr;

  // The following members are normally only used in linker scripts.
  MemoryRegion *MemRegion = nullptr;
  Expr AddrExpr;
  Expr AlignExpr;
  Expr LMAExpr;
  Expr SubalignExpr;
  std::vector<BaseCommand *> Commands;
  std::vector<StringRef> Phdrs;
  llvm::Optional<uint32_t> Filler;
  ConstraintKind Constraint = ConstraintKind::NoConstraint;
  std::string Location;
  std::string MemoryRegionName;
  bool Noload = false;

  template <class ELFT> void finalize();
  template <class ELFT> void writeTo(uint8_t *Buf);
  template <class ELFT> void maybeCompress();
  uint32_t getFiller();

  void sort(std::function<int(InputSectionBase *S)> Order);
  void sortInitFini();
  void sortCtorsDtors();
};

int getPriority(StringRef S);

// All output sections that are handled by the linker specially are
// globally accessible. Writer initializes them, so don't use them
// until Writer is initialized.
struct Out {
  static uint8_t First;
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
  uint32_t Alignment;
};
} // namespace elf
} // namespace lld
namespace llvm {
template <> struct DenseMapInfo<lld::elf::SectionKey> {
  static lld::elf::SectionKey getEmptyKey();
  static lld::elf::SectionKey getTombstoneKey();
  static unsigned getHashValue(const lld::elf::SectionKey &Val);
  static bool isEqual(const lld::elf::SectionKey &LHS,
                      const lld::elf::SectionKey &RHS);
};
} // namespace llvm
namespace lld {
namespace elf {

// This class knows how to create an output section for a given
// input section. Output section type is determined by various
// factors, including input section's sh_flags, sh_type and
// linker scripts.
class OutputSectionFactory {
public:
  OutputSectionFactory();
  ~OutputSectionFactory();

  void addInputSec(InputSectionBase *IS, StringRef OutsecName,
                   OutputSection *OS = nullptr);

private:
  llvm::SmallDenseMap<SectionKey, OutputSection *> Map;
};

uint64_t getHeaderSize();
void reportDiscarded(InputSectionBase *IS);
void sortByOrder(llvm::MutableArrayRef<InputSection *> In,
                 std::function<int(InputSectionBase *S)> Order);

extern std::vector<OutputSection *> OutputSections;
} // namespace elf
} // namespace lld

#endif
