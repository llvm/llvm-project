//===- DWARF.h -----------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#ifndef LLD_ELF_DWARF_H
#define LLD_ELF_DWARF_H

#include "InputFiles.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf {

class InputSection;

struct LLDDWARFSection final : public llvm::DWARFSection {
  InputSectionBase *Sec = nullptr;
};

template <class ELFT> class LLDDwarfObj final : public llvm::DWARFObject {
public:
  explicit LLDDwarfObj(ObjFile<ELFT> *Obj);

  const llvm::DWARFSection &getInfoSection() const override {
    return InfoSection;
  }

  const llvm::DWARFSection &getRangeSection() const override {
    return RangeSection;
  }

  const llvm::DWARFSection &getLineSection() const override {
    return LineSection;
  }

  StringRef getFileName() const override { return ""; }
  StringRef getAbbrevSection() const override { return AbbrevSection; }
  StringRef getStringSection() const override { return StrSection; }

  StringRef getGnuPubNamesSection() const override {
    return GnuPubNamesSection;
  }

  StringRef getGnuPubTypesSection() const override {
    return GnuPubTypesSection;
  }

  bool isLittleEndian() const override {
    return ELFT::TargetEndianness == llvm::support::little;
  }

  llvm::Optional<llvm::RelocAddrEntry> find(const llvm::DWARFSection &Sec,
                                            uint64_t Pos) const override;

private:
  template <class RelTy>
  llvm::Optional<llvm::RelocAddrEntry> findAux(const InputSectionBase &Sec,
                                               uint64_t Pos,
                                               ArrayRef<RelTy> Rels) const;

  LLDDWARFSection InfoSection;
  LLDDWARFSection RangeSection;
  LLDDWARFSection LineSection;

  StringRef AbbrevSection;
  StringRef GnuPubNamesSection;
  StringRef GnuPubTypesSection;
  StringRef StrSection;
};

} // namespace elf
} // namespace lld

#endif
