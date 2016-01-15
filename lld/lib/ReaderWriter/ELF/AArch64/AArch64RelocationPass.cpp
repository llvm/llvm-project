//===- lib/ReaderWriter/ELF/AArch64/AArch64RelocationPass.cpp -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the relocation processing pass for AArch64. This includes
///   GOT and PLT entries, TLS, COPY, and ifunc.
///
/// This also includes additional behavior that gnu-ld and gold implement but
/// which is not specified anywhere.
///
//===----------------------------------------------------------------------===//

#include "AArch64RelocationPass.h"
#include "AArch64LinkingContext.h"
#include "Atoms.h"
#include "lld/Core/Simple.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

using namespace lld;
using namespace lld::elf;
using namespace llvm::ELF;

// .got values
static const uint8_t AArch64GotAtomContent[8] = {0};

// tls descriptor .got values, the layout is:
// struct tlsdesc {
//   ptrdiff_t (*entry) (struct tlsdesc *);
//   void *arg;
// };
static const uint8_t AArch64TlsdescGotAtomContent[16] = {0};

// .plt value (entry 0)
static const uint8_t AArch64Plt0AtomContent[32] = {
    0xf0, 0x7b, 0xbf, 0xa9, // stp	x16, x30, [sp,#-16]!
    0x10, 0x00, 0x00, 0x90, // adrp	x16, Page(eh_frame)
    0x11, 0x02, 0x40, 0xf9, // ldr	x17, [x16,#offset]
    0x10, 0x02, 0x00, 0x91, // add	x16, x16, #offset
    0x20, 0x02, 0x1f, 0xd6, // br	x17
    0x1f, 0x20, 0x03, 0xd5, // nop
    0x1f, 0x20, 0x03, 0xd5, // nop
    0x1f, 0x20, 0x03, 0xd5  // nop
};

// .plt values (other entries)
static const uint8_t AArch64PltAtomContent[16] = {
    0x10, 0x00, 0x00, 0x90, // adrp x16, PAGE(<GLOBAL_OFFSET_TABLE>)
    0x11, 0x02, 0x40, 0xf9, // ldr  x17, [x16,#offset]
    0x10, 0x02, 0x00, 0x91, // add  x16, x16, #offset
    0x20, 0x02, 0x1f, 0xd6  // br   x17
};

// .plt tlsdesc values
static const uint8_t AArch64PltTlsdescAtomContent[32] = {
    0xe2, 0x0f, 0xbf, 0xa9, // stp  x2, x3, [sp, #-16]
    0x02, 0x00, 0x00, 0x90, // adpr x2, 0
    0x03, 0x00, 0x00, 0x90, // adpr x3, 0
    0x42, 0x00, 0x40, 0xf9, // ldr  x2, [x2, #0]
    0x63, 0x00, 0x00, 0x91, // add  x3, x3, 0
    0x40, 0x00, 0x1f, 0xd6, // br   x2
    0x1f, 0x20, 0x03, 0xd5, // nop
    0x1f, 0x20, 0x03, 0xd5  // nop
};

namespace {

/// \brief Atoms that are used by AArch64 dynamic linking
class AArch64GOTAtom : public GOTAtom {
public:
  AArch64GOTAtom(const File &f) : GOTAtom(f, ".got") {}

  ArrayRef<uint8_t> rawContent() const override {
    return ArrayRef<uint8_t>(AArch64GotAtomContent, 8);
  }

protected:
  // Constructor for AArch64GOTAtom
  AArch64GOTAtom(const File &f, StringRef secName) : GOTAtom(f, secName) {}
};

class AArch64GOTPLTAtom : public AArch64GOTAtom {
public:
  AArch64GOTPLTAtom(const File &f) : AArch64GOTAtom(f, ".got.plt") {}
};

class AArch64TLSDESCGOTAtom : public AArch64GOTPLTAtom {
public:
  AArch64TLSDESCGOTAtom(const File &f) : AArch64GOTPLTAtom(f) {}

  ArrayRef<uint8_t> rawContent() const override {
    return ArrayRef<uint8_t>(AArch64TlsdescGotAtomContent, 16);
  }
};


class AArch64PLT0Atom : public PLT0Atom {
public:
  AArch64PLT0Atom(const File &f) : PLT0Atom(f) {}
  ArrayRef<uint8_t> rawContent() const override {
    return ArrayRef<uint8_t>(AArch64Plt0AtomContent, 32);
  }
};

class AArch64PLTAtom : public PLTAtom {
public:
  AArch64PLTAtom(const File &f) : PLTAtom(f, ".plt") {}

  ArrayRef<uint8_t> rawContent() const override {
    return ArrayRef<uint8_t>(AArch64PltAtomContent, 16);
  }
};

class AArch64PLTTLSDESCAtom : public PLTAtom {
public:
  AArch64PLTTLSDESCAtom(const File &f) : PLTAtom(f, ".plt") {}

  ArrayRef<uint8_t> rawContent() const override {
    return ArrayRef<uint8_t>(AArch64PltTlsdescAtomContent, 32);
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

/// \brief CRTP base for handling relocations.
template <class Derived> class AArch64RelocationPass : public Pass {
  /// \brief Handle a specific reference.
  void handleReference(const DefinedAtom &atom, const Reference &ref) {
    DEBUG_WITH_TYPE(
        "AArch64", llvm::dbgs()
                       << "\t" << LLVM_FUNCTION_NAME << "()"
                       << ":   Name of Defined Atom: " << atom.name().str();
        llvm::dbgs() << "   kindValue: " << ref.kindValue() << "\n");
    if (ref.kindNamespace() != Reference::KindNamespace::ELF)
      return;
    assert(ref.kindArch() == Reference::KindArch::AArch64);
    switch (ref.kindValue()) {
    case R_AARCH64_ABS32:
    case R_AARCH64_ABS16:
    case R_AARCH64_ABS64:
    case R_AARCH64_PREL16:
    case R_AARCH64_PREL32:
    case R_AARCH64_PREL64:
      static_cast<Derived *>(this)->handlePlain(ref);
      break;
    case R_AARCH64_GOTREL32:
    case R_AARCH64_GOTREL64:
      static_cast<Derived *>(this)->handleGOT(ref);
      break;
    case R_AARCH64_ADR_PREL_PG_HI21:
      static_cast<Derived *>(this)->handlePlain(ref);
      break;
    case R_AARCH64_LDST8_ABS_LO12_NC:
    case R_AARCH64_LDST16_ABS_LO12_NC:
    case R_AARCH64_LDST32_ABS_LO12_NC:
    case R_AARCH64_LDST64_ABS_LO12_NC:
    case R_AARCH64_LDST128_ABS_LO12_NC:
      static_cast<Derived *>(this)->handlePlain(ref);
      break;
    case R_AARCH64_ADD_ABS_LO12_NC:
      static_cast<Derived *>(this)->handlePlain(ref);
      break;
    case R_AARCH64_CALL26:
    case R_AARCH64_JUMP26:
    case R_AARCH64_CONDBR19:
      static_cast<Derived *>(this)->handlePlain(ref);
      break;
    case R_AARCH64_TLSLE_ADD_TPREL_HI12:
    case R_AARCH64_TLSLE_ADD_TPREL_LO12_NC:
      static_cast<Derived *>(this)->handlePlain(ref);
      break;
    case R_AARCH64_ADR_GOT_PAGE:
    case R_AARCH64_LD64_GOT_LO12_NC:
      static_cast<Derived *>(this)->handleGOT(ref);
      break;
    case R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
    case R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
      static_cast<Derived *>(this)->handleGOTTPREL(ref);
      break;
    case R_AARCH64_TLSDESC_ADR_PAGE21:
    case R_AARCH64_TLSDESC_LD64_LO12_NC:
    case R_AARCH64_TLSDESC_ADD_LO12_NC:
      static_cast<Derived *>(this)->handleTLSDESC(ref);
      break;
    }
  }

protected:
  /// \brief get the PLT entry for a given IFUNC Atom.
  ///
  /// If the entry does not exist. Both the GOT and PLT entry is created.
  const PLTAtom *getIFUNCPLTEntry(const DefinedAtom *da) {
    auto plt = _pltMap.find(da);
    if (plt != _pltMap.end())
      return plt->second;
    auto ga = new (_file._alloc) AArch64GOTPLTAtom(_file);
    ga->addReferenceELF_AArch64(R_AARCH64_IRELATIVE, 0, da, 0);
    auto pa = new (_file._alloc) AArch64PLTAtom(_file);
    pa->addReferenceELF_AArch64(R_AARCH64_PREL32, 2, ga, -4);
#ifndef NDEBUG
    ga->_name = "__got_ifunc_";
    ga->_name += da->name();
    pa->_name = "__plt_ifunc_";
    pa->_name += da->name();
#endif
    _gotMap[da] = ga;
    _pltMap[da] = pa;
    _gotVector.push_back(ga);
    _pltVector.push_back(pa);
    return pa;
  }

  /// \brief Redirect the call to the PLT stub for the target IFUNC.
  ///
  /// This create a PLT and GOT entry for the IFUNC if one does not exist. The
  /// GOT entry and a IRELATIVE relocation to the original target resolver.
  std::error_code handleIFUNC(const Reference &ref) {
    auto target = dyn_cast_or_null<const DefinedAtom>(ref.target());
    if (target && target->contentType() == DefinedAtom::typeResolver)
      const_cast<Reference &>(ref).setTarget(getIFUNCPLTEntry(target));
    return std::error_code();
  }

  /// \brief Create a GOT entry for the TP offset of a TLS atom.
  const GOTAtom *getGOTTPREL(const Atom *atom) {
    auto got = _gotMap.find(atom);
    if (got == _gotMap.end()) {
      auto g = new (_file._alloc) AArch64GOTAtom(_file);
      g->addReferenceELF_AArch64(R_AARCH64_TLS_TPREL64, 0, atom, 0);
#ifndef NDEBUG
      g->_name = "__got_tls_";
      g->_name += atom->name();
#endif
      _gotMap[atom] = g;
      _gotVector.push_back(g);
      return g;
    }
    return got->second;
  }

  /// \brief Create a GOT TPREL entry to local or external TLS variable.
  std::error_code handleGOTTPREL(const Reference &ref) {
    if (isa<DefinedAtom>(ref.target()) ||
        isa<SharedLibraryAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getGOTTPREL(ref.target()));
    return std::error_code();
  }

  /// \brief Generates a double GOT entry with R_AARCH64_TLSDESC dynamic
  /// relocation reference.  Since the dynamic relocation is resolved
  /// lazily so the GOT associated should be in .got.plt.
  const GOTAtom *getTLSDESCPLTEntry(const Atom *da) {
    auto got = _gotMap.find(da);
    if (got != _gotMap.end())
      return got->second;
    auto ga = new (_file._alloc) AArch64TLSDESCGOTAtom(_file);
    ga->addReferenceELF_AArch64(R_AARCH64_TLSDESC, 0, da, 0);
    auto pa = new (_file._alloc) AArch64PLTTLSDESCAtom(_file);
    pa->addReferenceELF_AArch64(R_AARCH64_ADR_PREL_PG_HI21, 4, ga, 0);
    pa->addReferenceELF_AArch64(R_AARCH64_ADR_PREL_PG_HI21, 8, ga, 0);
    pa->addReferenceELF_AArch64(R_AARCH64_LDST64_ABS_LO12_NC, 12, ga, 0);
    pa->addReferenceELF_AArch64(R_AARCH64_ADD_ABS_LO12_NC, 16, ga, 0);
#ifndef NDEBUG
    ga->_name = "__got_tlsdesc_";
    ga->_name += da->name();
    pa->_name = "__plt_tlsdesc_";
    pa->_name += da->name();
#endif
    _gotMap[da] = ga;
    _pltMap[da] = pa;
    _tlsdescVector.push_back(ga);
    _pltVector.push_back(pa);
    return ga;
  }

  std::error_code handleTLSDESC(const Reference &ref) {
    if (isa<DefinedAtom>(ref.target()) ||
        isa<SharedLibraryAtom>(ref.target())) {
      const_cast<Reference &>(ref).setTarget(getTLSDESCPLTEntry(ref.target()));
    }
    return std::error_code();
  }

  /// \brief Create a GOT entry containing 0.
  const GOTAtom *getNullGOT() {
    if (!_null) {
      _null = new (_file._alloc) AArch64GOTPLTAtom(_file);
#ifndef NDEBUG
      _null->_name = "__got_null";
#endif
    }
    return _null;
  }

  const GOTAtom *getGOT(const DefinedAtom *da) {
    auto got = _gotMap.find(da);
    if (got == _gotMap.end()) {
      auto g = new (_file._alloc) AArch64GOTAtom(_file);
      g->addReferenceELF_AArch64(R_AARCH64_ABS64, 0, da, 0);
#ifndef NDEBUG
      g->_name = "__got_";
      g->_name += da->name();
#endif
      _gotMap[da] = g;
      _gotVector.push_back(g);
      return g;
    }
    return got->second;
  }

public:
  AArch64RelocationPass(const ELFLinkingContext &ctx) : _file(ctx), _ctx(ctx) {}

  /// \brief Do the pass.
  ///
  /// The goal here is to first process each reference individually. Each call
  /// to handleReference may modify the reference itself and/or create new
  /// atoms which must be stored in one of the maps below.
  ///
  /// After all references are handled, the atoms created during that are all
  /// added to mf.
  std::error_code perform(SimpleFile &mf) override {
    ScopedTask task(getDefaultDomain(), "AArch64 GOT/PLT Pass");
    DEBUG_WITH_TYPE(
        "AArch64", llvm::dbgs() << "Undefined Atoms"
                                << "\n";
        for (const auto &atom
             : mf.undefined()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        } llvm::dbgs()
            << "Shared Library Atoms"
            << "\n";
        for (const auto &atom
             : mf.sharedLibrary()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        } llvm::dbgs()
            << "Absolute Atoms"
            << "\n";
        for (const auto &atom
             : mf.absolute()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        }
            // Process all references.
            llvm::dbgs()
            << "Defined Atoms"
            << "\n");
    for (const auto &atom : mf.defined()) {
      for (const auto &ref : *atom) {
        handleReference(*atom, *ref);
      }
    }

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
    if (_plt0) {
      _got0->setOrdinal(ordinal++);
      _got1->setOrdinal(ordinal++);
      mf.addAtom(*_got0);
      mf.addAtom(*_got1);
    }
    for (auto &got : _gotVector) {
      got->setOrdinal(ordinal++);
      mf.addAtom(*got);
    }
    // Add any tlsdesc GOT relocation after default PLT and iFUNC entries.
    for (auto &tlsdesc : _tlsdescVector) {
      tlsdesc->setOrdinal(ordinal++);
      mf.addAtom(*tlsdesc);
    }
    for (auto obj : _objectVector) {
      obj->setOrdinal(ordinal++);
      mf.addAtom(*obj);
    }

    return std::error_code();
  }

protected:
  /// \brief Owner of all the Atoms created by this pass.
  ELFPassFile _file;
  const ELFLinkingContext &_ctx;

  /// \brief Map Atoms to their GOT entries.
  llvm::DenseMap<const Atom *, GOTAtom *> _gotMap;

  /// \brief Map Atoms to their PLT entries.
  llvm::DenseMap<const Atom *, PLTAtom *> _pltMap;

  /// \brief Map Atoms to their Object entries.
  llvm::DenseMap<const Atom *, ObjectAtom *> _objectMap;

  /// \brief the list of GOT/PLT atoms
  std::vector<GOTAtom *> _gotVector;
  std::vector<GOTAtom *> _tlsdescVector;
  std::vector<PLTAtom *> _pltVector;
  std::vector<ObjectAtom *> _objectVector;

  /// \brief GOT entry that is always 0. Used for undefined weaks.
  GOTAtom *_null = nullptr;

  /// \brief The got and plt entries for .PLT0. This is used to call into the
  /// dynamic linker for symbol resolution.
  /// @{
  PLT0Atom *_plt0 = nullptr;
  GOTAtom *_got0 = nullptr;
  GOTAtom *_got1 = nullptr;
  /// @}
};

/// This implements the static relocation model. Meaning GOT and PLT entries are
/// not created for references that can be directly resolved. These are
/// converted to a direct relocation. For entries that do require a GOT or PLT
/// entry, that entry is statically bound.
///
/// TLS always assumes module 1 and attempts to remove indirection.
class AArch64StaticRelocationPass final
    : public AArch64RelocationPass<AArch64StaticRelocationPass> {
public:
  AArch64StaticRelocationPass(const elf::AArch64LinkingContext &ctx)
      : AArch64RelocationPass(ctx) {}

  std::error_code handlePlain(const Reference &ref) { return handleIFUNC(ref); }

  std::error_code handlePLT32(const Reference &ref) {
    // __tls_get_addr is handled elsewhere.
    if (ref.target() && ref.target()->name() == "__tls_get_addr") {
      const_cast<Reference &>(ref).setKindValue(R_AARCH64_NONE);
      return std::error_code();
    }
    // Static code doesn't need PLTs.
    const_cast<Reference &>(ref).setKindValue(R_AARCH64_PREL32);
    // Handle IFUNC.
    if (const DefinedAtom *da =
            dyn_cast_or_null<const DefinedAtom>(ref.target()))
      if (da->contentType() == DefinedAtom::typeResolver)
        return handleIFUNC(ref);
    return std::error_code();
  }

  std::error_code handleGOT(const Reference &ref) {
    if (isa<UndefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getNullGOT());
    else if (const DefinedAtom *da = dyn_cast<const DefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getGOT(da));
    return std::error_code();
  }
};

class AArch64DynamicRelocationPass final
    : public AArch64RelocationPass<AArch64DynamicRelocationPass> {
public:
  AArch64DynamicRelocationPass(const elf::AArch64LinkingContext &ctx)
      : AArch64RelocationPass(ctx) {}

  const PLT0Atom *getPLT0() {
    if (_plt0)
      return _plt0;
    // Fill in the null entry.
    getNullGOT();
    _plt0 = new (_file._alloc) AArch64PLT0Atom(_file);
    _got0 = new (_file._alloc) AArch64GOTPLTAtom(_file);
    _got1 = new (_file._alloc) AArch64GOTPLTAtom(_file);
    _plt0->addReferenceELF_AArch64(R_AARCH64_ADR_GOT_PAGE, 4, _got0, 0);
    _plt0->addReferenceELF_AArch64(R_AARCH64_LD64_GOT_LO12_NC, 8, _got1, 0);
    _plt0->addReferenceELF_AArch64(ADD_AARCH64_GOTRELINDEX, 12, _got1, 0);
#ifndef NDEBUG
    _plt0->_name = "__PLT0";
    _got0->_name = "__got0";
    _got1->_name = "__got1";
#endif
    return _plt0;
  }

  const PLTAtom *getPLTEntry(const Atom *a) {
    auto plt = _pltMap.find(a);
    if (plt != _pltMap.end())
      return plt->second;
    auto ga = new (_file._alloc) AArch64GOTPLTAtom(_file);
    ga->addReferenceELF_AArch64(R_AARCH64_JUMP_SLOT, 0, a, 0);
    auto pa = new (_file._alloc) AArch64PLTAtom(_file);
    pa->addReferenceELF_AArch64(R_AARCH64_ADR_GOT_PAGE, 0, ga, 0);
    pa->addReferenceELF_AArch64(R_AARCH64_LD64_GOT_LO12_NC, 4, ga, 0);
    pa->addReferenceELF_AArch64(ADD_AARCH64_GOTRELINDEX, 8, ga, 0);
    pa->addReferenceELF_AArch64(R_AARCH64_NONE, 12, getPLT0(), 0);
    // Set the starting address of the got entry to the first instruction in
    // the plt0 entry.
    ga->addReferenceELF_AArch64(R_AARCH64_ABS32, 0, getPLT0(), 0);
#ifndef NDEBUG
    ga->_name = "__got_";
    ga->_name += a->name();
    pa->_name = "__plt_";
    pa->_name += a->name();
#endif
    _gotMap[a] = ga;
    _pltMap[a] = pa;
    _gotVector.push_back(ga);
    _pltVector.push_back(pa);
    return pa;
  }

  const ObjectAtom *getObjectEntry(const SharedLibraryAtom *a) {
    auto obj = _objectMap.find(a);
    if (obj != _objectMap.end())
      return obj->second;

    auto oa = new (_file._alloc) ObjectAtom(_file);
    // This needs to point to the atom that we just created.
    oa->addReferenceELF_AArch64(R_AARCH64_COPY, 0, oa, 0);

    oa->_name = a->name();
    oa->_size = a->size();

    _objectMap[a] = oa;
    _objectVector.push_back(oa);
    return oa;
  }

  std::error_code handlePlain(const Reference &ref) {
    if (!ref.target())
      return std::error_code();
    if (auto sla = dyn_cast<SharedLibraryAtom>(ref.target())) {
      if (sla->type() == SharedLibraryAtom::Type::Data)
        const_cast<Reference &>(ref).setTarget(getObjectEntry(sla));
      else if (sla->type() == SharedLibraryAtom::Type::Code)
        const_cast<Reference &>(ref).setTarget(getPLTEntry(sla));
    } else
      return handleIFUNC(ref);
    return std::error_code();
  }

  std::error_code handlePLT32(const Reference &ref) {
    // Turn this into a PC32 to the PLT entry.
    const_cast<Reference &>(ref).setKindValue(R_AARCH64_PREL32);
    // Handle IFUNC.
    if (const DefinedAtom *da =
            dyn_cast_or_null<const DefinedAtom>(ref.target()))
      if (da->contentType() == DefinedAtom::typeResolver)
        return handleIFUNC(ref);
    if (isa<const SharedLibraryAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getPLTEntry(ref.target()));
    return std::error_code();
  }

  const GOTAtom *getSharedGOT(const SharedLibraryAtom *sla) {
    auto got = _gotMap.find(sla);
    if (got == _gotMap.end()) {
      auto g = new (_file._alloc) AArch64GOTAtom(_file);
      g->addReferenceELF_AArch64(R_AARCH64_GLOB_DAT, 0, sla, 0);
#ifndef NDEBUG
      g->_name = "__got_";
      g->_name += sla->name();
#endif
      _gotMap[sla] = g;
      _gotVector.push_back(g);
      return g;
    }
    return got->second;
  }

  std::error_code handleGOT(const Reference &ref) {
    if (isa<UndefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getNullGOT());
    else if (const DefinedAtom *da = dyn_cast<const DefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getGOT(da));
    else if (const auto sla = dyn_cast<const SharedLibraryAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getSharedGOT(sla));
    return std::error_code();
  }
};
} // end anon namespace

std::unique_ptr<Pass>
lld::elf::createAArch64RelocationPass(const AArch64LinkingContext &ctx) {
  switch (ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    if (ctx.isDynamic())
      return llvm::make_unique<AArch64DynamicRelocationPass>(ctx);
    return llvm::make_unique<AArch64StaticRelocationPass>(ctx);
  case llvm::ELF::ET_DYN:
    return llvm::make_unique<AArch64DynamicRelocationPass>(ctx);
  case llvm::ELF::ET_REL:
    return nullptr;
  default:
    llvm_unreachable("Unhandled output file type");
  }
}
