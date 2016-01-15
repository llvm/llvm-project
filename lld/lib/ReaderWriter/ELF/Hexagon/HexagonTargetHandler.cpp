//===- lib/ReaderWriter/ELF/Hexagon/HexagonTargetHandler.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HexagonExecutableWriter.h"
#include "HexagonDynamicLibraryWriter.h"
#include "HexagonLinkingContext.h"
#include "HexagonTargetHandler.h"

using namespace llvm::ELF;

using llvm::makeArrayRef;

namespace lld {
namespace elf {

HexagonTargetHandler::HexagonTargetHandler(HexagonLinkingContext &ctx)
    : _ctx(ctx), _targetLayout(new HexagonTargetLayout(ctx)),
      _relocationHandler(new HexagonTargetRelocationHandler(*_targetLayout)) {}

std::unique_ptr<Writer> HexagonTargetHandler::getWriter() {
  switch (_ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return llvm::make_unique<HexagonExecutableWriter>(_ctx, *_targetLayout);
  case llvm::ELF::ET_DYN:
    return llvm::make_unique<HexagonDynamicLibraryWriter>(_ctx, *_targetLayout);
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
}

using namespace llvm::ELF;

// .got atom
const uint8_t hexagonGotAtomContent[4] = { 0 };
// .got.plt atom (entry 0)
const uint8_t hexagonGotPlt0AtomContent[16] = { 0 };
// .got.plt atom (all other entries)
const uint8_t hexagonGotPltAtomContent[4] = { 0 };
// .plt (entry 0)
const uint8_t hexagonPlt0AtomContent[28] = {
  0x00, 0x40, 0x00, 0x00, // { immext (#0)
  0x1c, 0xc0, 0x49, 0x6a, //   r28 = add (pc, ##GOT0@PCREL) } # address of GOT0
  0x0e, 0x42, 0x9c, 0xe2, // { r14 -= add (r28, #16)  # offset of GOTn from GOTa
  0x4f, 0x40, 0x9c, 0x91, //   r15 = memw (r28 + #8)  # object ID at GOT2
  0x3c, 0xc0, 0x9c, 0x91, //   r28 = memw (r28 + #4) }# dynamic link at GOT1
  0x0e, 0x42, 0x0e, 0x8c, // { r14 = asr (r14, #2)    # index of PLTn
  0x00, 0xc0, 0x9c, 0x52, //   jumpr r28 }            # call dynamic linker
};

// .plt (other entries)
const uint8_t hexagonPltAtomContent[16] = {
  0x00, 0x40, 0x00, 0x00, // { immext (#0)
  0x0e, 0xc0, 0x49, 0x6a, //   r14 = add (pc, ##GOTn@PCREL) } # address of GOTn
  0x1c, 0xc0, 0x8e, 0x91, // r28 = memw (r14)                 # contents of GOTn
  0x00, 0xc0, 0x9c, 0x52, // jumpr r28                        # call it
};

class HexagonGOTAtom : public GOTAtom {
public:
  HexagonGOTAtom(const File &f) : GOTAtom(f, ".got") {}

  ArrayRef<uint8_t> rawContent() const override {
    return makeArrayRef(hexagonGotAtomContent);
  }

  Alignment alignment() const override { return 4; }
};

class HexagonGOTPLTAtom : public GOTAtom {
public:
  HexagonGOTPLTAtom(const File &f) : GOTAtom(f, ".got.plt") {}

  ArrayRef<uint8_t> rawContent() const override {
    return makeArrayRef(hexagonGotPltAtomContent);
  }

  Alignment alignment() const override { return 4; }
};

class HexagonGOTPLT0Atom : public GOTAtom {
public:
  HexagonGOTPLT0Atom(const File &f) : GOTAtom(f, ".got.plt") {}

  ArrayRef<uint8_t> rawContent() const override {
    return makeArrayRef(hexagonGotPlt0AtomContent);
  }

  Alignment alignment() const override { return 8; }
};

class HexagonPLT0Atom : public PLT0Atom {
public:
  HexagonPLT0Atom(const File &f) : PLT0Atom(f) {}

  ArrayRef<uint8_t> rawContent() const override {
    return makeArrayRef(hexagonPlt0AtomContent);
  }
};

class HexagonPLTAtom : public PLTAtom {

public:
  HexagonPLTAtom(const File &f, StringRef secName) : PLTAtom(f, secName) {}

  ArrayRef<uint8_t> rawContent() const override {
    return makeArrayRef(hexagonPltAtomContent);
  }
};

class ELFPassFile : public SimpleFile {
public:
  ELFPassFile(const ELFLinkingContext &eti)
    : SimpleFile("ELFPassFile", kindELFObject) {
    setOrdinal(eti.getNextOrdinalAndIncrement());
  }

  llvm::BumpPtrAllocator _alloc;
};

/// \brief Create GOT and PLT entries for relocations. Handles standard GOT/PLT
template <class Derived> class GOTPLTPass : public Pass {
  /// \brief Handle a specific reference.
  void handleReference(const DefinedAtom &atom, const Reference &ref) {
    if (ref.kindNamespace() != Reference::KindNamespace::ELF)
      return;
    assert(ref.kindArch() == Reference::KindArch::Hexagon);
    switch (ref.kindValue()) {
    case R_HEX_PLT_B22_PCREL:
    case R_HEX_B22_PCREL:
      static_cast<Derived *>(this)->handlePLT32(ref);
      break;
    case R_HEX_GOT_LO16:
    case R_HEX_GOT_HI16:
    case R_HEX_GOT_32_6_X:
    case R_HEX_GOT_16_X:
    case R_HEX_GOT_11_X:
      static_cast<Derived *>(this)->handleGOTREL(ref);
      break;
    }
  }

protected:
  /// \brief Create a GOT entry containing 0.
  const GOTAtom *getNullGOT() {
    if (!_null) {
      _null = new (_file._alloc) HexagonGOTPLTAtom(_file);
#ifndef NDEBUG
      _null->_name = "__got_null";
#endif
    }
    return _null;
  }

public:
  GOTPLTPass(const ELFLinkingContext &ctx) : _file(ctx) {}

  /// \brief Do the pass.
  ///
  /// The goal here is to first process each reference individually. Each call
  /// to handleReference may modify the reference itself and/or create new
  /// atoms which must be stored in one of the maps below.
  ///
  /// After all references are handled, the atoms created during that are all
  /// added to mf.
  std::error_code perform(SimpleFile &mf) override {
    // Process all references.
    for (const auto &atom : mf.defined())
      for (const auto &ref : *atom)
        handleReference(*atom, *ref);

    // Add all created atoms to the link.
    uint64_t ordinal = 0;
    if (_plt0) {
      _plt0->setOrdinal(ordinal++);
      mf.addAtom(*_plt0);
    }
    for (auto &plt : _pltVector) {
      plt->setOrdinal(ordinal++);
      mf.addAtom(*plt);
    }
    if (_null) {
      _null->setOrdinal(ordinal++);
      mf.addAtom(*_null);
    }
    if (_got0) {
      _got0->setOrdinal(ordinal++);
      mf.addAtom(*_got0);
    }
    for (auto &got : _gotVector) {
      got->setOrdinal(ordinal++);
      mf.addAtom(*got);
    }

    return std::error_code();
  }

protected:
  /// \brief Owner of all the Atoms created by this pass.
  ELFPassFile _file;

  /// \brief Map Atoms to their GOT entries.
  llvm::DenseMap<const Atom *, GOTAtom *> _gotMap;

  /// \brief Map Atoms to their PLT entries.
  llvm::DenseMap<const Atom *, PLTAtom *> _pltMap;

  /// \brief the list of GOT/PLT atoms
  std::vector<GOTAtom *> _gotVector;
  std::vector<PLTAtom *> _pltVector;

  /// \brief GOT entry that is always 0. Used for undefined weaks.
  GOTAtom *_null = nullptr;

  /// \brief The got and plt entries for .PLT0. This is used to call into the
  /// dynamic linker for symbol resolution.
  /// @{
  PLT0Atom *_plt0 = nullptr;
  GOTAtom *_got0 = nullptr;
  /// @}
};

class DynamicGOTPLTPass final : public GOTPLTPass<DynamicGOTPLTPass> {
public:
  DynamicGOTPLTPass(const HexagonLinkingContext &ctx) : GOTPLTPass(ctx) {
    _got0 = new (_file._alloc) HexagonGOTPLT0Atom(_file);
#ifndef NDEBUG
    _got0->_name = "__got0";
#endif
  }

  const PLT0Atom *getPLT0() {
    if (_plt0)
      return _plt0;
    _plt0 = new (_file._alloc) HexagonPLT0Atom(_file);
    _plt0->addReferenceELF_Hexagon(R_HEX_B32_PCREL_X, 0, _got0, 0);
    _plt0->addReferenceELF_Hexagon(R_HEX_6_PCREL_X, 4, _got0, 4);
    DEBUG_WITH_TYPE("PLT", llvm::dbgs() << "[ PLT0/GOT0 ] "
                                        << "Adding plt0/got0 \n");
    return _plt0;
  }

  const PLTAtom *getPLTEntry(const Atom *a) {
    auto plt = _pltMap.find(a);
    if (plt != _pltMap.end())
      return plt->second;
    auto ga = new (_file._alloc) HexagonGOTPLTAtom(_file);
    ga->addReferenceELF_Hexagon(R_HEX_JMP_SLOT, 0, a, 0);
    auto pa = new (_file._alloc) HexagonPLTAtom(_file, ".plt");
    pa->addReferenceELF_Hexagon(R_HEX_B32_PCREL_X, 0, ga, 0);
    pa->addReferenceELF_Hexagon(R_HEX_6_PCREL_X, 4, ga, 4);

    // Point the got entry to the PLT0 atom initially
    ga->addReferenceELF_Hexagon(R_HEX_32, 0, getPLT0(), 0);
#ifndef NDEBUG
    ga->_name = "__got_";
    ga->_name += a->name();
    pa->_name = "__plt_";
    pa->_name += a->name();
    DEBUG_WITH_TYPE("PLT", llvm::dbgs() << "[" << a->name() << "] "
                                        << "Adding plt/got: " << pa->_name
                                        << "/" << ga->_name << "\n");
#endif
    _gotMap[a] = ga;
    _pltMap[a] = pa;
    _gotVector.push_back(ga);
    _pltVector.push_back(pa);
    return pa;
  }

  const GOTAtom *getGOTEntry(const Atom *a) {
    auto got = _gotMap.find(a);
    if (got != _gotMap.end())
      return got->second;
    auto ga = new (_file._alloc) HexagonGOTAtom(_file);
    ga->addReferenceELF_Hexagon(R_HEX_GLOB_DAT, 0, a, 0);

#ifndef NDEBUG
    ga->_name = "__got_";
    ga->_name += a->name();
    DEBUG_WITH_TYPE("GOT", llvm::dbgs() << "[" << a->name() << "] "
                                        << "Adding got: " << ga->_name << "\n");
#endif
    _gotMap[a] = ga;
    _gotVector.push_back(ga);
    return ga;
  }

  std::error_code handleGOTREL(const Reference &ref) {
    // Turn this so that the target is set to the GOT entry
    const_cast<Reference &>(ref).setTarget(getGOTEntry(ref.target()));
    return std::error_code();
  }

  std::error_code handlePLT32(const Reference &ref) {
    // Turn this into a PC32 to the PLT entry.
    assert(ref.kindNamespace() == Reference::KindNamespace::ELF);
    assert(ref.kindArch() == Reference::KindArch::Hexagon);
    const_cast<Reference &>(ref).setKindValue(R_HEX_B22_PCREL);
    const_cast<Reference &>(ref).setTarget(getPLTEntry(ref.target()));
    return std::error_code();
  }
};

void HexagonLinkingContext::addPasses(PassManager &pm) {
  if (isDynamic())
    pm.add(llvm::make_unique<DynamicGOTPLTPass>(*this));
  ELFLinkingContext::addPasses(pm);
}

void SDataSection::doPreFlight() {
  // sort the atoms on the alignments they have been set
  std::stable_sort(_atoms.begin(), _atoms.end(), [](const AtomLayout *A,
                                                    const AtomLayout *B) {
    const DefinedAtom *definedAtomA = cast<DefinedAtom>(A->_atom);
    const DefinedAtom *definedAtomB = cast<DefinedAtom>(B->_atom);
    int64_t alignmentA = definedAtomA->alignment().value;
    int64_t alignmentB = definedAtomB->alignment().value;
    if (alignmentA == alignmentB) {
      if (definedAtomA->merge() == DefinedAtom::mergeAsTentative)
        return false;
      if (definedAtomB->merge() == DefinedAtom::mergeAsTentative)
        return true;
    }
    return alignmentA < alignmentB;
  });

  // Set the fileOffset, and the appropriate size of the section
  for (auto &ai : _atoms) {
    const DefinedAtom *definedAtom = cast<DefinedAtom>(ai->_atom);
    DefinedAtom::Alignment atomAlign = definedAtom->alignment();
    uint64_t fOffset = alignOffset(fileSize(), atomAlign);
    uint64_t mOffset = alignOffset(memSize(), atomAlign);
    ai->_fileOffset = fOffset;
    _fsize = fOffset + definedAtom->size();
    _msize = mOffset + definedAtom->size();
  }
} // finalize

SDataSection::SDataSection(const HexagonLinkingContext &ctx)
    : AtomSection(ctx, ".sdata", DefinedAtom::typeDataFast, 0,
                  HexagonTargetLayout::ORDER_SDATA) {
  _type = SHT_PROGBITS;
  _flags = SHF_ALLOC | SHF_WRITE;
  _alignment = 4096;
}

const AtomLayout *SDataSection::appendAtom(const Atom *atom) {
  const DefinedAtom *definedAtom = cast<DefinedAtom>(atom);
  DefinedAtom::Alignment atomAlign = definedAtom->alignment();
  uint64_t alignment = atomAlign.value;
  _atoms.push_back(new (_alloc) AtomLayout(atom, 0, 0));
  // Set the section alignment to the largest alignment
  // std::max doesn't support uint64_t
  if (_alignment < alignment)
    _alignment = alignment;
  return _atoms.back();
}

void finalizeHexagonRuntimeAtomValues(HexagonTargetLayout &layout) {
  AtomLayout *gotAtom = layout.findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
  OutputSection<ELF32LE> *gotpltSection = layout.findOutputSection(".got.plt");
  if (gotpltSection)
    gotAtom->_virtualAddr = gotpltSection->virtualAddr();
  else
    gotAtom->_virtualAddr = 0;
  AtomLayout *dynamicAtom = layout.findAbsoluteAtom("_DYNAMIC");
  OutputSection<ELF32LE> *dynamicSection = layout.findOutputSection(".dynamic");
  if (dynamicSection)
    dynamicAtom->_virtualAddr = dynamicSection->virtualAddr();
  else
    dynamicAtom->_virtualAddr = 0;
}

} // namespace elf
} // namespace lld
