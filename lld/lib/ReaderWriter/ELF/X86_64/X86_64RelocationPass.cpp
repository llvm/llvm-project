//===- lib/ReaderWriter/ELF/X86_64/X86_64RelocationPass.cpp ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the relocation processing pass for x86-64. This includes
///   GOT and PLT entries, TLS, COPY, and ifunc.
///
/// This is based on section 4.4.1 of the AMD64 ABI (no stable URL as of Oct,
/// 2013).
///
/// This also includes aditional behaivor that gnu-ld and gold implement but
/// which is not specified anywhere.
///
//===----------------------------------------------------------------------===//

#include "X86_64RelocationPass.h"
#include "Atoms.h"
#include "X86_64LinkingContext.h"
#include "lld/Core/Simple.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace lld;
using namespace lld::elf;
using namespace llvm::ELF;

// .got values
static const uint8_t x86_64GotAtomContent[8] = {0};

// .plt value (entry 0)
static const uint8_t x86_64Plt0AtomContent[16] = {
    0xff, 0x35, 0x00, 0x00, 0x00, 0x00, // pushq GOT+8(%rip)
    0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmp *GOT+16(%rip)
    0x90, 0x90, 0x90, 0x90              // nopnopnop
};

// .plt values (other entries)
static const uint8_t x86_64PltAtomContent[16] = {
    0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmpq *gotatom(%rip)
    0x68, 0x00, 0x00, 0x00, 0x00,       // pushq reloc-index
    0xe9, 0x00, 0x00, 0x00, 0x00        // jmpq plt[-1]
};

// TLS GD Entry
static const uint8_t x86_64GotTlsGdAtomContent[] = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

namespace {
/// \brief Atoms that are used by X86_64 dynamic linking
class X86_64GOTAtom : public GOTAtom {
public:
  X86_64GOTAtom(const File &f, StringRef secName) : GOTAtom(f, secName) {}

  ArrayRef<uint8_t> rawContent() const override {
    return ArrayRef<uint8_t>(x86_64GotAtomContent, 8);
  }
};

/// \brief X86_64 GOT TLS GD entry.
class GOTTLSGdAtom : public X86_64GOTAtom {
public:
  GOTTLSGdAtom(const File &f, StringRef secName) : X86_64GOTAtom(f, secName) {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(x86_64GotTlsGdAtomContent);
  }
};

class X86_64PLT0Atom : public PLT0Atom {
public:
  X86_64PLT0Atom(const File &f) : PLT0Atom(f) {}
  ArrayRef<uint8_t> rawContent() const override {
    return ArrayRef<uint8_t>(x86_64Plt0AtomContent, 16);
  }
};

class X86_64PLTAtom : public PLTAtom {
public:
  X86_64PLTAtom(const File &f, StringRef secName) : PLTAtom(f, secName) {}

  ArrayRef<uint8_t> rawContent() const override {
    return ArrayRef<uint8_t>(x86_64PltAtomContent, 16);
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
template <class Derived> class RelocationPass : public Pass {
  /// \brief Handle a specific reference.
  void handleReference(const DefinedAtom &atom, const Reference &ref) {
    if (ref.kindNamespace() != Reference::KindNamespace::ELF)
      return;
    assert(ref.kindArch() == Reference::KindArch::x86_64);
    switch (ref.kindValue()) {
    case R_X86_64_16:
    case R_X86_64_32:
    case R_X86_64_32S:
    case R_X86_64_64:
    case R_X86_64_PC16:
    case R_X86_64_PC32:
    case R_X86_64_PC64:
      static_cast<Derived *>(this)->handlePlain(ref);
      break;
    case R_X86_64_PLT32:
      static_cast<Derived *>(this)->handlePLT32(ref);
      break;
    case R_X86_64_GOT32:
    case R_X86_64_GOTPC32:
    case R_X86_64_GOTPCREL:
    case R_X86_64_GOTOFF64:
      static_cast<Derived *>(this)->handleGOT(ref);
      break;
    case R_X86_64_GOTTPOFF: // GOT Thread Pointer Offset
      static_cast<Derived *>(this)->handleGOTTPOFF(ref);
      break;
    case R_X86_64_TLSGD:
      static_cast<Derived *>(this)->handleTLSGd(ref);
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
    auto ga = new (_file._alloc) X86_64GOTAtom(_file, ".got.plt");
    ga->addReferenceELF_x86_64(R_X86_64_IRELATIVE, 0, da, 0);
    auto pa = new (_file._alloc) X86_64PLTAtom(_file, ".plt");
    pa->addReferenceELF_x86_64(R_X86_64_PC32, 2, ga, -4);
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
  const GOTAtom *getGOTTPOFF(const Atom *atom) {
    auto got = _gotMap.find(atom);
    if (got == _gotMap.end()) {
      auto g = new (_file._alloc) X86_64GOTAtom(_file, ".got");
      g->addReferenceELF_x86_64(R_X86_64_TPOFF64, 0, atom, 0);
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

  /// \brief Create a TPOFF64 GOT entry.
  std::error_code handleGOTTPOFF(const Reference &ref) {
    if (isa<DefinedAtom>(ref.target())) {
      const_cast<Reference &>(ref).setTarget(getGOTTPOFF(ref.target()));
    }
    return std::error_code();
  }

  /// \brief Create a TLS GOT entry with DTPMOD64/DTPOFF64 dynamic relocations.
  void handleTLSGd(const Reference &ref) {
    const_cast<Reference &>(ref).setTarget(getTLSGdGOTEntry(ref.target()));
  }

  /// \brief Create a GOT entry containing 0.
  const GOTAtom *getNullGOT() {
    if (!_null) {
      _null = new (_file._alloc) X86_64GOTAtom(_file, ".got.plt");
#ifndef NDEBUG
      _null->_name = "__got_null";
#endif
    }
    return _null;
  }

  const GOTAtom *getGOT(const DefinedAtom *da) {
    auto got = _gotMap.find(da);
    if (got == _gotMap.end()) {
      auto g = new (_file._alloc) X86_64GOTAtom(_file, ".got");
      g->addReferenceELF_x86_64(R_X86_64_64, 0, da, 0);
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

  const GOTAtom *getTLSGdGOTEntry(const Atom *a) {
    auto got = _gotTLSGdMap.find(a);
    if (got != _gotTLSGdMap.end())
      return got->second;

    auto ga = new (_file._alloc) GOTTLSGdAtom(_file, ".got");
    _gotTLSGdMap[a] = ga;

    _tlsGotVector.push_back(ga);
    ga->addReferenceELF_x86_64(R_X86_64_DTPMOD64, 0, a, 0);
    ga->addReferenceELF_x86_64(R_X86_64_DTPOFF64, 8, a, 0);

    return ga;
  }

public:
  RelocationPass(const ELFLinkingContext &ctx) : _file(ctx), _ctx(ctx) {}

  /// \brief Do the pass.
  ///
  /// The goal here is to first process each reference individually. Each call
  /// to handleReference may modify the reference itself and/or create new
  /// atoms which must be stored in one of the maps below.
  ///
  /// After all references are handled, the atoms created during that are all
  /// added to mf.
  std::error_code perform(SimpleFile &mf) override {
    ScopedTask task(getDefaultDomain(), "X86-64 GOT/PLT Pass");
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
    for (auto &got : _tlsGotVector) {
      got->setOrdinal(ordinal++);
      mf.addAtom(*got);
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

  /// \brief Map Atoms to TLS GD GOT entries.
  llvm::DenseMap<const Atom *, GOTAtom *> _gotTLSGdMap;

  /// \brief Map Atoms to their Object entries.
  llvm::DenseMap<const Atom *, ObjectAtom *> _objectMap;

  /// \brief the list of GOT/PLT atoms
  std::vector<GOTAtom *> _gotVector;
  std::vector<PLTAtom *> _pltVector;
  std::vector<ObjectAtom *> _objectVector;

  /// \brief the list of TLS GOT atoms.
  std::vector<GOTAtom *> _tlsGotVector;

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
class StaticRelocationPass final
    : public RelocationPass<StaticRelocationPass> {
public:
  StaticRelocationPass(const elf::X86_64LinkingContext &ctx)
      : RelocationPass(ctx) {}

  std::error_code handlePlain(const Reference &ref) { return handleIFUNC(ref); }

  std::error_code handlePLT32(const Reference &ref) {
    // __tls_get_addr is handled elsewhere.
    if (ref.target() && ref.target()->name() == "__tls_get_addr") {
      const_cast<Reference &>(ref).setKindValue(R_X86_64_NONE);
      return std::error_code();
    }
    // Static code doesn't need PLTs.
    const_cast<Reference &>(ref).setKindValue(R_X86_64_PC32);
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

class DynamicRelocationPass final
    : public RelocationPass<DynamicRelocationPass> {
public:
  DynamicRelocationPass(const elf::X86_64LinkingContext &ctx)
      : RelocationPass(ctx) {}

  const PLT0Atom *getPLT0() {
    if (_plt0)
      return _plt0;
    // Fill in the null entry.
    getNullGOT();
    _plt0 = new (_file._alloc) X86_64PLT0Atom(_file);
    _got0 = new (_file._alloc) X86_64GOTAtom(_file, ".got.plt");
    _got1 = new (_file._alloc) X86_64GOTAtom(_file, ".got.plt");
    _plt0->addReferenceELF_x86_64(R_X86_64_PC32, 2, _got0, -4);
    _plt0->addReferenceELF_x86_64(R_X86_64_PC32, 8, _got1, -4);
#ifndef NDEBUG
    _got0->_name = "__got0";
    _got1->_name = "__got1";
#endif
    return _plt0;
  }

  const PLTAtom *getPLTEntry(const Atom *a) {
    auto plt = _pltMap.find(a);
    if (plt != _pltMap.end())
      return plt->second;
    auto ga = new (_file._alloc) X86_64GOTAtom(_file, ".got.plt");
    ga->addReferenceELF_x86_64(R_X86_64_JUMP_SLOT, 0, a, 0);
    auto pa = new (_file._alloc) X86_64PLTAtom(_file, ".plt");
    pa->addReferenceELF_x86_64(R_X86_64_PC32, 2, ga, -4);
    pa->addReferenceELF_x86_64(LLD_R_X86_64_GOTRELINDEX, 7, ga, 0);
    pa->addReferenceELF_x86_64(R_X86_64_PC32, 12, getPLT0(), -4);
    // Set the starting address of the got entry to the second instruction in
    // the plt entry.
    ga->addReferenceELF_x86_64(R_X86_64_64, 0, pa, 6);
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
    oa->addReferenceELF_x86_64(R_X86_64_COPY, 0, oa, 0);

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
    const_cast<Reference &>(ref).setKindValue(R_X86_64_PC32);
    // Handle IFUNC.
    if (const DefinedAtom *da =
            dyn_cast_or_null<const DefinedAtom>(ref.target()))
      if (da->contentType() == DefinedAtom::typeResolver)
        return handleIFUNC(ref);
    // If it is undefined at link time, push the work to the dynamic linker by
    // creating a PLT entry
    if (isa<SharedLibraryAtom>(ref.target()) ||
        isa<UndefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getPLTEntry(ref.target()));
    return std::error_code();
  }

  const GOTAtom *getSharedGOT(const Atom *a) {
    auto got = _gotMap.find(a);
    if (got == _gotMap.end()) {
      auto g = new (_file._alloc) X86_64GOTAtom(_file, ".got");
      g->addReferenceELF_x86_64(R_X86_64_GLOB_DAT, 0, a, 0);
#ifndef NDEBUG
      g->_name = "__got_";
      g->_name += a->name();
#endif
      _gotMap[a] = g;
      _gotVector.push_back(g);
      return g;
    }
    return got->second;
  }

  std::error_code handleGOT(const Reference &ref) {
    if (const DefinedAtom *da = dyn_cast<const DefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getGOT(da));
    // Handle undefined atoms in the same way as shared lib atoms: to be
    // resolved at run time.
    else if (isa<SharedLibraryAtom>(ref.target()) ||
             isa<UndefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getSharedGOT(ref.target()));
    return std::error_code();
  }
};
} // end anon namespace

std::unique_ptr<Pass>
lld::elf::createX86_64RelocationPass(const X86_64LinkingContext &ctx) {
  switch (ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    if (ctx.isDynamic())
      return llvm::make_unique<DynamicRelocationPass>(ctx);
    return llvm::make_unique<StaticRelocationPass>(ctx);
  case llvm::ELF::ET_DYN:
    return llvm::make_unique<DynamicRelocationPass>(ctx);
  case llvm::ELF::ET_REL:
    return nullptr;
  default:
    llvm_unreachable("Unhandled output file type");
  }
}
