//===--------- lib/ReaderWriter/ELF/ARM/ARMTargetHandler.h ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_ARM_ARM_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_ARM_ARM_TARGET_HANDLER_H

#include "ARMELFFile.h"
#include "ARMRelocationHandler.h"
#include "ELFReader.h"
#include "TargetLayout.h"

namespace lld {
class ELFLinkingContext;

namespace elf {

/// \brief ARM specific section (.ARM.exidx) with indexes to exception handlers
class ARMExidxSection : public AtomSection<ELF32LE> {
  typedef AtomSection<ELF32LE> Base;

public:
  ARMExidxSection(const ELFLinkingContext &ctx, StringRef sectionName,
                  int32_t permissions, int32_t order)
      : Base(ctx, sectionName, ARMELFDefinedAtom::typeARMExidx, permissions,
             order) {
    this->_type = SHT_ARM_EXIDX;
    this->_isLoadedInMemory = true;
  }

  bool hasOutputSegment() const override { return true; }

  const AtomLayout *appendAtom(const Atom *atom) override {
    const DefinedAtom *definedAtom = cast<DefinedAtom>(atom);
    assert((ARMELFDefinedAtom::ARMContentType)definedAtom->contentType() ==
               ARMELFDefinedAtom::typeARMExidx &&
           "atom content type for .ARM.exidx section has to be typeARMExidx");

    DefinedAtom::Alignment atomAlign = definedAtom->alignment();
    uint64_t fOffset = alignOffset(this->fileSize(), atomAlign);
    uint64_t mOffset = alignOffset(this->memSize(), atomAlign);

    _atoms.push_back(new (_alloc) AtomLayout(atom, fOffset, 0));
    this->_fsize = fOffset + definedAtom->size();
    this->_msize = mOffset + definedAtom->size();
    DEBUG_WITH_TYPE("Section", llvm::dbgs()
                                   << "[" << this->name() << " " << this << "] "
                                   << "Adding atom: " << atom->name() << "@"
                                   << fOffset << "\n");

    uint64_t alignment = atomAlign.value;
    if (this->_alignment < alignment)
      this->_alignment = alignment;

    return _atoms.back();
  }
};

class ARMTargetLayout : public TargetLayout<ELF32LE> {
public:
  enum ARMSectionOrder {
    ORDER_ARM_EXIDX = TargetLayout::ORDER_EH_FRAME + 1,
  };

  ARMTargetLayout(ELFLinkingContext &ctx) : TargetLayout(ctx) {}

  SectionOrder getSectionOrder(StringRef name, int32_t contentType,
                               int32_t contentPermissions) override {
    switch (contentType) {
    case ARMELFDefinedAtom::typeARMExidx:
      return ORDER_ARM_EXIDX;
    default:
      return TargetLayout::getSectionOrder(name, contentType,
                                           contentPermissions);
    }
  }

  StringRef getOutputSectionName(StringRef archivePath, StringRef memberPath,
                                 StringRef inputSectionName) const override {
    return llvm::StringSwitch<StringRef>(inputSectionName)
        .StartsWith(".ARM.exidx", ".ARM.exidx")
        .StartsWith(".ARM.extab", ".ARM.extab")
        .Default(TargetLayout::getOutputSectionName(archivePath, memberPath,
                                                    inputSectionName));
  }

  SegmentType getSegmentType(const Section<ELF32LE> *section) const override {
    switch (section->order()) {
    case ORDER_ARM_EXIDX:
      return llvm::ELF::PT_ARM_EXIDX;
    default:
      return TargetLayout::getSegmentType(section);
    }
  }

  AtomSection<ELF32LE> *
  createSection(StringRef name, int32_t contentType,
                DefinedAtom::ContentPermissions contentPermissions,
                SectionOrder sectionOrder) override {
    if ((ARMELFDefinedAtom::ARMContentType)contentType ==
        ARMELFDefinedAtom::typeARMExidx)
      return new ARMExidxSection(_ctx, name, contentPermissions, sectionOrder);

    return TargetLayout::createSection(name, contentType, contentPermissions,
                                       sectionOrder);
  }

  uint64_t getGOTSymAddr() {
    std::call_once(_gotSymOnce, [this]() {
      if (AtomLayout *gotAtom = findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_"))
        _gotSymAddr = gotAtom->_virtualAddr;
    });
    return _gotSymAddr;
  }

  uint64_t getTPOffset() {
    std::call_once(_tpOffOnce, [this]() {
      for (const auto &phdr : *_programHeader) {
        if (phdr->p_type == llvm::ELF::PT_TLS) {
          _tpOff = llvm::alignTo(TCB_SIZE, phdr->p_align);
          break;
        }
      }
      assert(_tpOff != 0 && "TLS segment not found");
    });
    return _tpOff;
  }

  bool target1Rel() const { return _ctx.armTarget1Rel(); }

private:
  // TCB block size of the TLS.
  enum { TCB_SIZE = 0x8 };

private:
  uint64_t _gotSymAddr = 0;
  uint64_t _tpOff = 0;
  std::once_flag _gotSymOnce;
  std::once_flag _tpOffOnce;
};

class ARMTargetHandler final : public TargetHandler {
public:
  ARMTargetHandler(ARMLinkingContext &ctx);

  const TargetRelocationHandler &getRelocationHandler() const override {
    return *_relocationHandler;
  }

  std::unique_ptr<Reader> getObjReader() override {
    return llvm::make_unique<ELFReader<ARMELFFile>>(_ctx);
  }

  std::unique_ptr<Reader> getDSOReader() override {
    return llvm::make_unique<ELFReader<DynamicFile<ELF32LE>>>(_ctx);
  }

  std::unique_ptr<Writer> getWriter() override;

private:
  ARMLinkingContext &_ctx;
  std::unique_ptr<ARMTargetLayout> _targetLayout;
  std::unique_ptr<ARMTargetRelocationHandler> _relocationHandler;
};

} // end namespace elf
} // end namespace lld

#endif // LLD_READER_WRITER_ELF_ARM_ARM_TARGET_HANDLER_H
