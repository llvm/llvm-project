//===- OutputSections.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_OUTPUT_SECTIONS_H
#define LLD_ELF_OUTPUT_SECTIONS_H

#include "InputSection.h"
#include "LinkerScript.h"
#include "lld/Common/LLVM.h"
#include "llvm/ADT/SmallVectorWithFlags.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Parallel.h"

#include <array>

namespace lld::elf {

struct PhdrEntry;

struct CompressedData {
  std::unique_ptr<SmallVector<uint8_t, 0>[]> shards;
  uint32_t type = 0;
  uint32_t numShards = 0;
  uint32_t checksum = 0;
  uint64_t uncompressedSize;
};

// This represents a section in an output file.
// It is composed of multiple InputSections.
// The writer creates multiple OutputSections and assign them unique,
// non-overlapping file offsets and VAs.
class OutputSection final : public SectionBase {
public:
  OutputSection(Ctx &, StringRef name, uint32_t type, uint64_t flags);

  static bool classof(const SectionBase *s) {
    return s->kind() == SectionBase::Output;
  }

  uint64_t getLMA() const;
  template <typename ELFT> void writeHeaderTo(typename ELFT::Shdr *sHdr);

  Ctx &ctx;
  uint32_t sectionIndex = UINT32_MAX;
  unsigned sortRank;

  uint32_t getPhdrFlags() const;

  // Pointer to the PT_LOAD segment, which this section resides in. This field
  // is used to correctly compute file offset of a section. When two sections
  // share the same load segment, difference between their file offsets should
  // be equal to difference between their virtual addresses. To compute some
  // section offset we use the following formula: Off = Off_first + VA -
  // VA_first, where Off_first and VA_first is file offset and VA of first
  // section in PT_LOAD.
  PhdrEntry *ptLoad = nullptr;

  // Pointer to a relocation section for this section. Usually nullptr because
  // we consume relocations, but if --emit-relocs is specified (which is rare),
  // it may have a non-null value.
  OutputSection *relocationSection = nullptr;

  // Initially this field is the number of InputSections that have been added to
  // the OutputSection so far. Later on, after a call to assignAddresses, it
  // corresponds to the Elf_Shdr member.
  uint64_t size = 0;

  // The following fields correspond to Elf_Shdr members.
  uint64_t offset = 0;
  uint64_t addr = 0;
  uint32_t shName = 0;

  void recordSection(InputSectionBase *isec);
  void commitSection(InputSection *isec);
  void finalizeInputSections();

  // The following members are normally only used in linker scripts.
  MemoryRegion *memRegion = nullptr;
  MemoryRegion *lmaRegion = nullptr;
  Expr addrExpr;
  Expr alignExpr;
  Expr lmaExpr;
  Expr subalignExpr;

  // Used by non-alloc SHT_CREL to hold the header and content byte stream.
  uint64_t crelHeader = 0;
  SmallVector<char, 0> crelBody;

  llvm::SmallVectorWithFlags<SectionCommand *, 0, 4> commands;
  SmallVector<StringRef, 0> phdrs;
  std::optional<std::array<uint8_t, 4>> filler;
  ConstraintKind constraint = ConstraintKind::NoConstraint;
  std::string location;
  std::string memoryRegionName;
  std::string lmaRegionName;
  bool getNonAlloc() const { return storage.getFlag(0); }
  void setNonAlloc(bool Value) { storage.setFlag(0, Value); }
  bool getTypeIsSet() const { return storage.getFlag(1); }
  void setTypeIsSet(bool Value) { storage.setFlag(1, Value); }
  bool getExpressionsUseSymbols() const { return storage.getFlag(2); }
  bool setExpressionsUseSymbols(bool Value) {
    storage.setFlag(2, Value);
    return Value;
  }
  bool getUsedInExpression() const { return storage.getFlag(3); }
  bool setUsedInExpression(bool Value) {
    storage.setFlag(3, Value);
    return Value;
  }
  bool getInOverlay() const { return commands.getFlag(0); }
  void setInOverlay(bool Value) { commands.setFlag(0, Value); }
  bool getFirstInOverlay() const { return commands.getFlag(1); }
  bool setFirstInOverlay(bool Value) {
    commands.setFlag(1, Value);
    return Value;
  }

  // Tracks whether the section has ever had an input section added to it, even
  // if the section was later removed (e.g. because it is a synthetic section
  // that wasn't needed). This is needed for orphan placement.
  bool getHasInputSections() const { return commands.getFlag(2); }
  void setHasInputSections(bool Value) { commands.setFlag(2, Value); }

  // The output section description is specified between DATA_SEGMENT_ALIGN and
  // DATA_RELRO_END.
  bool getRelro() const { return commands.getFlag(3); }
  bool setRelro(bool Value) {
    commands.setFlag(3, Value);
    return Value;
  }

  template <bool is64> void finalizeNonAllocCrel(Ctx &);
  void finalize(Ctx &);
  template <class ELFT>
  void writeTo(Ctx &, uint8_t *buf, llvm::parallel::TaskGroup &tg);
  // Check that the addends for dynamic relocations were written correctly.
  void checkDynRelAddends(Ctx &);
  template <class ELFT> void maybeCompress(Ctx &);

  void sort(llvm::function_ref<int(InputSectionBase *s)> order);
  void sortInitFini();
  void sortCtorsDtors();

  std::array<uint8_t, 4> getFiller(Ctx &);

  // Used for implementation of --compress-debug-sections and
  // --compress-sections.
  CompressedData compressed;

private:
  llvm::SmallVectorWithFlags<InputSection *, 0, 4> storage;
};

struct OutputDesc final : SectionCommand {
  OutputSection osec;
  OutputDesc(Ctx &ctx, StringRef name, uint32_t type, uint64_t flags)
      : SectionCommand(OutputSectionKind), osec(ctx, name, type, flags) {}

  static bool classof(const SectionCommand *c) {
    return c->kind == OutputSectionKind;
  }
};

// This represents a CLASS(class_name) { ... } that can be referenced by output
// section descriptions. If referenced more than once, the sections can be
// spilled to the next reference like --enable-non-contiguous-regions.
struct SectionClass final : public SectionBase {
  llvm::SmallVectorWithFlags<InputSectionDescription *, 0> commands;
  bool getAssigned() const { return commands.getFlag(0); }
  void setAssigned(bool Value) { commands.setFlag(0, Value); }

  SectionClass(StringRef name)
      : SectionBase(Class, nullptr, name, 0, 0, 0, 0, 0, 0) {}
  static bool classof(const SectionBase *s) { return s->kind() == Class; }
};

struct SectionClassDesc : SectionCommand {
  SectionClass sc;

  SectionClassDesc(StringRef name) : SectionCommand(ClassKind), sc(name) {}

  static bool classof(const SectionCommand *c) { return c->kind == ClassKind; }
};

int getPriority(StringRef s);

InputSection *getFirstInputSection(const OutputSection *os);
template <typename StorageT>
ArrayRef<InputSection *> getInputSections(const OutputSection &os,
                                          StorageT &storage) {
  ArrayRef<InputSection *> ret;
  storage.clear();
  for (SectionCommand *cmd : os.commands) {
    auto *isd = dyn_cast<InputSectionDescription>(cmd);
    if (!isd)
      continue;
    if (ret.empty()) {
      ret = isd->sections;
    } else {
      if (storage.empty())
        storage.assign(ret.begin(), ret.end());
      storage.insert(storage.end(), isd->sections.begin(), isd->sections.end());
    }
  }
  return storage.empty() ? ret : ArrayRef(storage);
}

uint64_t getHeaderSize(Ctx &);
} // namespace lld::elf

#endif
