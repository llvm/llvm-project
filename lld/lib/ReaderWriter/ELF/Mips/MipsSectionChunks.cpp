//===- lib/ReaderWriter/ELF/Mips/MipsSectionChunks.cpp --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsLinkingContext.h"
#include "MipsSectionChunks.h"
#include "MipsTargetLayout.h"

namespace lld {
namespace elf {

template <class ELFT>
MipsReginfoSection<ELFT>::MipsReginfoSection(
    const ELFLinkingContext &ctx, MipsTargetLayout<ELFT> &targetLayout,
    const Elf_Mips_RegInfo &reginfo)
    : Section<ELFT>(ctx, ".reginfo", "MipsReginfo"), _reginfo(reginfo),
      _targetLayout(targetLayout) {
  this->setOrder(MipsTargetLayout<ELFT>::ORDER_MIPS_REGINFO);
  this->_entSize = sizeof(Elf_Mips_RegInfo);
  this->_fsize = sizeof(Elf_Mips_RegInfo);
  this->_msize = sizeof(Elf_Mips_RegInfo);
  this->_alignment = 4;
  this->_type = SHT_MIPS_REGINFO;
  this->_flags = SHF_ALLOC;
}

template <class ELFT>
void MipsReginfoSection<ELFT>::write(ELFWriter *writer,
                                     TargetLayout<ELFT> &layout,
                                     llvm::FileOutputBuffer &buffer) {
  uint8_t *dest = buffer.getBufferStart() + this->fileOffset();
  std::memcpy(dest, &_reginfo, this->_fsize);
}

template <class ELFT> void MipsReginfoSection<ELFT>::finalize() {
  _reginfo.ri_gp_value = _targetLayout.getGPAddr();

  if (this->_outputSection)
    this->_outputSection->setType(this->_type);
}

template class MipsReginfoSection<ELF32BE>;
template class MipsReginfoSection<ELF32LE>;
template class MipsReginfoSection<ELF64BE>;
template class MipsReginfoSection<ELF64LE>;

template <class ELFT>
MipsOptionsSection<ELFT>::MipsOptionsSection(
    const ELFLinkingContext &ctx, MipsTargetLayout<ELFT> &targetLayout,
    const Elf_Mips_RegInfo &reginfo)
    : Section<ELFT>(ctx, ".MIPS.options", "MipsOptions"), _reginfo(reginfo),
      _targetLayout(targetLayout) {
  this->setOrder(MipsTargetLayout<ELFT>::ORDER_MIPS_OPTIONS);
  this->_entSize = 1;
  this->_alignment = 8;
  this->_fsize = llvm::alignTo(
      sizeof(Elf_Mips_Options) + sizeof(Elf_Mips_RegInfo), this->_alignment);
  this->_msize = this->_fsize;
  this->_type = SHT_MIPS_OPTIONS;
  this->_flags = SHF_ALLOC | SHF_MIPS_NOSTRIP;

  _header.kind = ODK_REGINFO;
  _header.size = this->_fsize;
  _header.section = 0;
  _header.info = 0;
}

template <class ELFT>
void MipsOptionsSection<ELFT>::write(ELFWriter *writer,
                                     TargetLayout<ELFT> &layout,
                                     llvm::FileOutputBuffer &buffer) {
  uint8_t *dest = buffer.getBufferStart() + this->fileOffset();
  std::memset(dest, 0, this->_fsize);
  std::memcpy(dest, &_header, sizeof(_header));
  std::memcpy(dest + sizeof(_header), &_reginfo, sizeof(_reginfo));
}

template <class ELFT> void MipsOptionsSection<ELFT>::finalize() {
  _reginfo.ri_gp_value = _targetLayout.getGPAddr();

  if (this->_outputSection)
    this->_outputSection->setType(this->_type);
}

template class MipsOptionsSection<ELF32BE>;
template class MipsOptionsSection<ELF32LE>;
template class MipsOptionsSection<ELF64BE>;
template class MipsOptionsSection<ELF64LE>;

template <class ELFT>
MipsAbiFlagsSection<ELFT>::MipsAbiFlagsSection(
    const ELFLinkingContext &ctx, MipsTargetLayout<ELFT> &targetLayout,
    const Elf_Mips_ABIFlags &abiFlags)
    : Section<ELFT>(ctx, ".MIPS.abiflags", "MipsAbiFlags"), _abiFlags(abiFlags),
      _targetLayout(targetLayout) {
  this->setOrder(MipsTargetLayout<ELFT>::ORDER_MIPS_ABI_FLAGS);
  this->_alignment = 8;
  this->_fsize = llvm::alignTo(sizeof(_abiFlags), this->_alignment);
  this->_msize = this->_fsize;
  this->_entSize = this->_fsize;
  this->_type = SHT_MIPS_ABIFLAGS;
  this->_flags = SHF_ALLOC;
}

template <class ELFT>
void MipsAbiFlagsSection<ELFT>::write(ELFWriter *writer,
                                      TargetLayout<ELFT> &layout,
                                      llvm::FileOutputBuffer &buffer) {
  uint8_t *dest = buffer.getBufferStart() + this->fileOffset();
  std::memcpy(dest, &_abiFlags, this->_fsize);
}

template <class ELFT> void MipsAbiFlagsSection<ELFT>::finalize() {
  if (this->_outputSection)
    this->_outputSection->setType(this->_type);
}

template class MipsAbiFlagsSection<ELF32BE>;
template class MipsAbiFlagsSection<ELF32LE>;
template class MipsAbiFlagsSection<ELF64BE>;
template class MipsAbiFlagsSection<ELF64LE>;

template <class ELFT>
MipsGOTSection<ELFT>::MipsGOTSection(const MipsLinkingContext &ctx)
    : AtomSection<ELFT>(ctx, ".got", DefinedAtom::typeGOT, DefinedAtom::permRW_,
                        MipsTargetLayout<ELFT>::ORDER_GOT),
      _hasNonLocal(false), _localCount(0) {
  this->_flags |= SHF_MIPS_GPREL;
  this->_alignment = 4;
}

template <class ELFT>
bool MipsGOTSection<ELFT>::compare(const Atom *a, const Atom *b) const {
  auto ia = _posMap.find(a);
  auto ib = _posMap.find(b);

  if (ia != _posMap.end() && ib != _posMap.end())
    return ia->second < ib->second;

  return ia == _posMap.end() && ib != _posMap.end();
}

template <class ELFT>
const AtomLayout *MipsGOTSection<ELFT>::appendAtom(const Atom *atom) {
  const DefinedAtom *da = dyn_cast<DefinedAtom>(atom);

  if (atom->name() == "_GLOBAL_OFFSET_TABLE_")
    return AtomSection<ELFT>::appendAtom(atom);

  for (const auto &r : *da) {
    if (r->kindNamespace() != Reference::KindNamespace::ELF)
      continue;
    assert(r->kindArch() == Reference::KindArch::Mips);
    switch (r->kindValue()) {
    case LLD_R_MIPS_GLOBAL_GOT:
      _hasNonLocal = true;
      _posMap[r->target()] = _posMap.size();
      return AtomSection<ELFT>::appendAtom(atom);
    case R_MIPS_TLS_TPREL32:
    case R_MIPS_TLS_DTPREL32:
    case R_MIPS_TLS_TPREL64:
    case R_MIPS_TLS_DTPREL64:
      _hasNonLocal = true;
      _tlsMap[r->target()] = _tlsMap.size();
      return AtomSection<ELFT>::appendAtom(atom);
    case R_MIPS_TLS_DTPMOD32:
    case R_MIPS_TLS_DTPMOD64:
      _hasNonLocal = true;
      break;
    }
  }

  if (!_hasNonLocal)
    ++_localCount;

  return AtomSection<ELFT>::appendAtom(atom);
}

template class MipsGOTSection<ELF32BE>;
template class MipsGOTSection<ELF32LE>;
template class MipsGOTSection<ELF64BE>;
template class MipsGOTSection<ELF64LE>;

template <class ELFT>
MipsPLTSection<ELFT>::MipsPLTSection(const MipsLinkingContext &ctx)
    : AtomSection<ELFT>(ctx, ".plt", DefinedAtom::typeGOT, DefinedAtom::permR_X,
                        MipsTargetLayout<ELFT>::ORDER_PLT) {}

template <class ELFT>
const AtomLayout *MipsPLTSection<ELFT>::findPLTLayout(const Atom *plt) const {
  auto it = _pltLayoutMap.find(plt);
  return it != _pltLayoutMap.end() ? it->second : nullptr;
}

template <class ELFT>
const AtomLayout *MipsPLTSection<ELFT>::appendAtom(const Atom *atom) {
  const auto *layout = AtomSection<ELFT>::appendAtom(atom);

  const DefinedAtom *da = cast<DefinedAtom>(atom);

  for (const auto &r : *da) {
    if (r->kindNamespace() != Reference::KindNamespace::ELF)
      continue;
    assert(r->kindArch() == Reference::KindArch::Mips);
    if (r->kindValue() == LLD_R_MIPS_STO_PLT) {
      _pltLayoutMap[r->target()] = layout;
      break;
    }
  }

  return layout;
}

template class MipsPLTSection<ELF32BE>;
template class MipsPLTSection<ELF32LE>;
template class MipsPLTSection<ELF64BE>;
template class MipsPLTSection<ELF64LE>;

template <class ELFT> static bool isMips64EL() {
  return ELFT::Is64Bits && ELFT::TargetEndianness == llvm::support::little;
}

template <class ELFT>
MipsRelocationTable<ELFT>::MipsRelocationTable(const ELFLinkingContext &ctx,
                                               StringRef str, int32_t order)
    : RelocationTable<ELFT>(ctx, str, order) {}

template <class ELFT>
void MipsRelocationTable<ELFT>::writeRela(ELFWriter *writer, Elf_Rela &r,
                                          const DefinedAtom &atom,
                                          const Reference &ref) {
  uint32_t rType = ref.kindValue() | (ref.tag() << 8);
  r.setSymbolAndType(this->getSymbolIndex(ref.target()), rType,
                     isMips64EL<ELFT>());
  r.r_offset = writer->addressOfAtom(&atom) + ref.offsetInAtom();
  // The addend is used only by relative relocations
  if (this->_ctx.isRelativeReloc(ref))
    r.r_addend = writer->addressOfAtom(ref.target()) + ref.addend();
  else
    r.r_addend = 0;
}

template <class ELFT>
void MipsRelocationTable<ELFT>::writeRel(ELFWriter *writer, Elf_Rel &r,
                                         const DefinedAtom &atom,
                                         const Reference &ref) {
  uint32_t rType = ref.kindValue() | (ref.tag() << 8);
  r.setSymbolAndType(this->getSymbolIndex(ref.target()), rType,
                     isMips64EL<ELFT>());
  r.r_offset = writer->addressOfAtom(&atom) + ref.offsetInAtom();
}

template class MipsRelocationTable<ELF32BE>;
template class MipsRelocationTable<ELF32LE>;
template class MipsRelocationTable<ELF64BE>;
template class MipsRelocationTable<ELF64LE>;

} // elf
} // lld
