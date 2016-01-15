//===--------- lib/ReaderWriter/ELF/ARM/ARMRelocationPass.cpp -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the relocation processing pass for ARM. This includes
///   GOT and PLT entries, TLS, COPY, and ifunc.
///
/// This also includes additional behavior that gnu-ld and gold implement but
/// which is not specified anywhere.
///
//===----------------------------------------------------------------------===//

#include "ARMRelocationPass.h"
#include "ARMLinkingContext.h"
#include "Atoms.h"
#include "lld/Core/Simple.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

using namespace lld;
using namespace lld::elf;
using namespace llvm::ELF;

namespace {
// ARM B/BL instructions of absolute relocation veneer.
// TODO: consider different instruction set for archs below ARMv5
// (one as for Thumb may be used though it's less optimal).
static const uint8_t Veneer_ARM_B_BL_Abs_a_AtomContent[4] = {
    0x04, 0xf0, 0x1f, 0xe5   // ldr pc, [pc, #-4]
};
static const uint8_t Veneer_ARM_B_BL_Abs_d_AtomContent[4] = {
    0x00, 0x00, 0x00, 0x00   // <target_symbol_address>
};

// Thumb B/BL instructions of absolute relocation veneer.
// TODO: consider different instruction set for archs above ARMv5
// (one as for ARM may be used since it's more optimal).
static const uint8_t Veneer_THM_B_BL_Abs_t_AtomContent[4] = {
    0x78, 0x47,              // bx pc
    0x00, 0x00               // nop
};
static const uint8_t Veneer_THM_B_BL_Abs_a_AtomContent[4] = {
    0xfe, 0xff, 0xff, 0xea   // b <target_symbol_address>
};

// .got values
static const uint8_t ARMGotAtomContent[4] = {0};

// .plt value (entry 0)
static const uint8_t ARMPlt0_a_AtomContent[16] = {
    0x04, 0xe0, 0x2d, 0xe5,  // push {lr}
    0x04, 0xe0, 0x9f, 0xe5,  // ldr lr, [pc, #4]
    0x0e, 0xe0, 0x8f, 0xe0,  // add lr, pc, lr
    0x00, 0xf0, 0xbe, 0xe5   // ldr pc, [lr, #0]!
};
static const uint8_t ARMPlt0_d_AtomContent[4] = {
    0x00, 0x00, 0x00, 0x00   // <got1_symbol_address>
};

// .plt values (other entries)
static const uint8_t ARMPltAtomContent[12] = {
    0x00, 0xc0, 0x8f, 0xe2,  // add ip, pc, #offset[G0]
    0x00, 0xc0, 0x8c, 0xe2,  // add ip, ip, #offset[G1]
    0x00, 0xf0, 0xbc, 0xe5   // ldr pc, [ip, #offset[G2]]!
};

// Veneer for switching from Thumb to ARM code for PLT entries.
static const uint8_t ARMPltVeneerAtomContent[4] = {
    0x78, 0x47,              // bx pc
    0x00, 0x00               // nop
};

// Determine proper names for mapping symbols.
static std::string getMappingAtomName(DefinedAtom::CodeModel model,
                                      const std::string &part) {
  switch (model) {
  case DefinedAtom::codeARM_a:
    return part.empty() ? "$a" : "$a." + part;
  case DefinedAtom::codeARM_d:
    return part.empty() ? "$d" : "$d." + part;
  case DefinedAtom::codeARM_t:
    return part.empty() ? "$t" : "$t." + part;
  default:
    llvm_unreachable("Wrong code model of mapping atom");
  }
}

/// \brief Atoms that hold veneer code.
class VeneerAtom : public SimpleELFDefinedAtom {
  StringRef _section;

public:
  VeneerAtom(const File &f, StringRef secName, const std::string &name = "")
      : SimpleELFDefinedAtom(f), _section(secName), _name(name) {}

  Scope scope() const override { return DefinedAtom::scopeTranslationUnit; }

  SectionChoice sectionChoice() const override {
    return DefinedAtom::sectionBasedOnContent;
  }

  StringRef customSectionName() const override { return _section; }

  ContentType contentType() const override { return DefinedAtom::typeCode; }

  uint64_t size() const override { return rawContent().size(); }

  ContentPermissions permissions() const override { return permR_X; }

  Alignment alignment() const override { return 4; }

  StringRef name() const override { return _name; }

private:
  std::string _name;
};

/// \brief Atoms that hold veneer for relocated ARM B/BL instructions
/// in absolute code.
class Veneer_ARM_B_BL_Abs_a_Atom : public VeneerAtom {
public:
  Veneer_ARM_B_BL_Abs_a_Atom(const File &f, StringRef secName,
                             const std::string &name)
      : VeneerAtom(f, secName, name) {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(Veneer_ARM_B_BL_Abs_a_AtomContent);
  }
};

class Veneer_ARM_B_BL_Abs_d_Atom : public VeneerAtom {
public:
  Veneer_ARM_B_BL_Abs_d_Atom(const File &f, StringRef secName)
      : VeneerAtom(f, secName) {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(Veneer_ARM_B_BL_Abs_d_AtomContent);
  }
};

/// \brief Atoms that hold veneer for relocated Thumb B/BL instructions
/// in absolute code.
class Veneer_THM_B_BL_Abs_t_Atom : public VeneerAtom {
public:
  Veneer_THM_B_BL_Abs_t_Atom(const File &f, StringRef secName,
                             const std::string &name)
      : VeneerAtom(f, secName, name) {}

  DefinedAtom::CodeModel codeModel() const override {
    return DefinedAtom::codeARMThumb;
  }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(Veneer_THM_B_BL_Abs_t_AtomContent);
  }
};

class Veneer_THM_B_BL_Abs_a_Atom : public VeneerAtom {
public:
  Veneer_THM_B_BL_Abs_a_Atom(const File &f, StringRef secName)
      : VeneerAtom(f, secName) {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(Veneer_THM_B_BL_Abs_a_AtomContent);
  }
};

template <DefinedAtom::CodeModel Model>
class ARMVeneerMappingAtom : public VeneerAtom {
public:
  ARMVeneerMappingAtom(const File &f, StringRef secName, StringRef name)
      : VeneerAtom(f, secName, getMappingAtomName(Model, name)) {
    static_assert((Model == DefinedAtom::codeARM_a ||
                   Model == DefinedAtom::codeARM_d ||
                   Model == DefinedAtom::codeARM_t),
                  "Only mapping atom types are allowed");
  }

  uint64_t size() const override { return 0; }

  ArrayRef<uint8_t> rawContent() const override { return ArrayRef<uint8_t>(); }

  DefinedAtom::CodeModel codeModel() const override { return Model; }
};

template <class BaseAtom, DefinedAtom::CodeModel Model>
class BaseMappingAtom : public BaseAtom {
public:
  BaseMappingAtom(const File &f, StringRef secName, StringRef name)
      : BaseAtom(f, secName) {
    static_assert((Model == DefinedAtom::codeARM_a ||
                   Model == DefinedAtom::codeARM_d ||
                   Model == DefinedAtom::codeARM_t),
                  "Only mapping atom types are allowed");
#ifndef NDEBUG
    _name = name;
#else
    _name = getMappingAtomName(Model, name);
#endif
  }

  DefinedAtom::CodeModel codeModel() const override {
#ifndef NDEBUG
    return isThumbCode(Model) ? DefinedAtom::codeARMThumb : DefinedAtom::codeNA;
#else
    return Model;
#endif
  }

  StringRef name() const override { return _name; }

private:
  std::string _name;
};

/// \brief Atoms that are used by ARM dynamic linking
class ARMGOTAtom : public GOTAtom {
public:
  ARMGOTAtom(const File &f) : GOTAtom(f, ".got") {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(ARMGotAtomContent);
  }

  Alignment alignment() const override { return 4; }

protected:
  // Constructor for PLTGOT atom.
  ARMGOTAtom(const File &f, StringRef secName) : GOTAtom(f, secName) {}
};

class ARMGOTPLTAtom : public ARMGOTAtom {
public:
  ARMGOTPLTAtom(const File &f) : ARMGOTAtom(f, ".got.plt") {}
};

/// \brief Proxy class to keep type compatibility with PLT0Atom.
class ARMPLT0Atom : public PLT0Atom {
public:
  ARMPLT0Atom(const File &f, StringRef) : PLT0Atom(f) {}
};

/// \brief PLT0 entry atom.
/// Serves as a mapping symbol in the release mode.
class ARMPLT0_a_Atom
    : public BaseMappingAtom<ARMPLT0Atom, DefinedAtom::codeARM_a> {
public:
  ARMPLT0_a_Atom(const File &f, const std::string &name)
      : BaseMappingAtom(f, ".plt", name) {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(ARMPlt0_a_AtomContent);
  }

  Alignment alignment() const override { return 4; }
};

class ARMPLT0_d_Atom
    : public BaseMappingAtom<ARMPLT0Atom, DefinedAtom::codeARM_d> {
public:
  ARMPLT0_d_Atom(const File &f, const std::string &name)
      : BaseMappingAtom(f, ".plt", name) {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(ARMPlt0_d_AtomContent);
  }

  Alignment alignment() const override { return 4; }
};

/// \brief PLT entry atom.
/// Serves as a mapping symbol in the release mode.
class ARMPLTAtom : public BaseMappingAtom<PLTAtom, DefinedAtom::codeARM_a> {
public:
  ARMPLTAtom(const File &f, const std::string &name)
      : BaseMappingAtom(f, ".plt", name) {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(ARMPltAtomContent);
  }

  Alignment alignment() const override { return 4; }
};

/// \brief Veneer atom for PLT entry.
/// Serves as a mapping symbol in the release mode.
class ARMPLTVeneerAtom
    : public BaseMappingAtom<PLTAtom, DefinedAtom::codeARM_t> {
public:
  ARMPLTVeneerAtom(const File &f, const std::string &name)
      : BaseMappingAtom(f, ".plt", name) {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(ARMPltVeneerAtomContent);
  }

  Alignment alignment() const override { return 4; }
};

/// \brief Atom which represents an object for which a COPY relocation will
/// be generated.
class ARMObjectAtom : public ObjectAtom {
public:
  ARMObjectAtom(const File &f) : ObjectAtom(f) {}
  Alignment alignment() const override { return 4; }
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
template <class Derived> class ARMRelocationPass : public Pass {
  /// \brief Handle a specific reference.
  void handleReference(const DefinedAtom &atom, const Reference &ref) {
    DEBUG_WITH_TYPE(
        "ARM", llvm::dbgs() << "\t" << LLVM_FUNCTION_NAME << "()"
            << ":   Name of Defined Atom: " << atom.name().str();
        llvm::dbgs() << "   kindValue: " << ref.kindValue() << "\n");
    if (ref.kindNamespace() != Reference::KindNamespace::ELF)
      return;
    assert(ref.kindArch() == Reference::KindArch::ARM);
    switch (ref.kindValue()) {
    case R_ARM_ABS32:
    case R_ARM_REL32:
    case R_ARM_TARGET1:
    case R_ARM_MOVW_ABS_NC:
    case R_ARM_MOVT_ABS:
    case R_ARM_THM_MOVW_ABS_NC:
    case R_ARM_THM_MOVT_ABS:
      static_cast<Derived *>(this)->handlePlain(isThumbCode(&atom), ref);
      break;
    case R_ARM_THM_CALL:
    case R_ARM_CALL:
    case R_ARM_JUMP24:
    case R_ARM_THM_JUMP24:
    case R_ARM_THM_JUMP11: {
      const auto actualModel = actualSourceCodeModel(atom, ref);
      const bool fromThumb = isThumbCode(actualModel);
      static_cast<Derived *>(this)->handlePlain(fromThumb, ref);
      static_cast<Derived *>(this)->handleVeneer(atom, fromThumb, ref);
    } break;
    case R_ARM_TLS_IE32:
      static_cast<Derived *>(this)->handleTLSIE32(ref);
      break;
    case R_ARM_GOT_BREL:
      static_cast<Derived *>(this)->handleGOT(ref);
      break;
    default:
      break;
    }
  }

protected:
  /// \brief Determine source atom's actual code model.
  ///
  /// Actual code model may differ from the existing one if fixup
  /// is possible on the later stages for given relocation type.
  DefinedAtom::CodeModel actualSourceCodeModel(const DefinedAtom &atom,
                                               const Reference &ref) {
    const auto kindValue = ref.kindValue();
    if (kindValue != R_ARM_CALL && kindValue != R_ARM_THM_CALL)
      return atom.codeModel();

    // TODO: For unconditional jump instructions (R_ARM_CALL and R_ARM_THM_CALL)
    // fixup isn't possible without veneer generation for archs below ARMv5.

    auto actualModel = atom.codeModel();
    if (const auto *da = dyn_cast<DefinedAtom>(ref.target())) {
      actualModel = da->codeModel();
    } else if (const auto *sla = dyn_cast<SharedLibraryAtom>(ref.target())) {
      if (sla->type() == SharedLibraryAtom::Type::Code) {
        // PLT entry will be generated here - assume we don't want a veneer
        // on top of it and prefer instruction fixup if needed.
        actualModel = DefinedAtom::codeNA;
      }
    }
    return actualModel;
  }

  std::error_code handleVeneer(const DefinedAtom &atom, bool fromThumb,
                               const Reference &ref) {
    // Actual instruction mode differs meaning that further fixup will be
    // applied.
    if (isThumbCode(&atom) != fromThumb)
      return std::error_code();

    const VeneerAtom *(Derived::*getVeneer)(const DefinedAtom *, StringRef) =
        nullptr;
    const auto kindValue = ref.kindValue();
    switch (kindValue) {
    case R_ARM_JUMP24:
      getVeneer = &Derived::getVeneer_ARM_B_BL;
      break;
    case R_ARM_THM_JUMP24:
      getVeneer = &Derived::getVeneer_THM_B_BL;
      break;
    default:
      return std::error_code();
    }

    // Target symbol and relocated place should have different
    // instruction sets in order a veneer to be generated in between.
    const auto *target = dyn_cast<DefinedAtom>(ref.target());
    if (!target || isThumbCode(target) == isThumbCode(&atom))
      return std::error_code();

    // Veneers may only be generated for STT_FUNC target symbols
    // or for symbols located in sections different to the place of relocation.
    StringRef secName = atom.customSectionName();
    if (DefinedAtom::typeCode != target->contentType() &&
        !target->customSectionName().equals(secName)) {
      StringRef kindValStr;
      if (!this->_ctx.registry().referenceKindToString(
              ref.kindNamespace(), ref.kindArch(), kindValue, kindValStr)) {
        kindValStr = "unknown";
      }

      std::string errStr =
          (Twine("Reference of type ") + Twine(kindValue) + " (" + kindValStr +
           ") from " + atom.name() + "+" + Twine(ref.offsetInAtom()) + " to " +
           ref.target()->name() + "+" + Twine(ref.addend()) +
           " cannot be effected without a veneer").str();

      llvm_unreachable(errStr.c_str());
    }

    assert(getVeneer && "The veneer handler is missing");
    const Atom *veneer =
        (static_cast<Derived *>(this)->*getVeneer)(target, secName);

    assert(veneer && "The veneer is not set");
    const_cast<Reference &>(ref).setTarget(veneer);
    return std::error_code();
  }

  /// \brief Get the veneer for ARM B/BL instructions
  /// in absolute code.
  const VeneerAtom *getVeneer_ARM_B_BL_Abs(const DefinedAtom *da,
                                           StringRef secName) {
    auto veneer = _veneerAtoms.lookup(da);
    if (!veneer.empty())
      return veneer._veneer;

    std::string name = "__";
    name += da->name();
    name += "_from_arm";
    // Create parts of veneer with mapping symbols.
    auto v_a =
        new (_file._alloc) Veneer_ARM_B_BL_Abs_a_Atom(_file, secName, name);
    addVeneerWithMapping<DefinedAtom::codeARM_a>(da, v_a, name);
    auto v_d = new (_file._alloc) Veneer_ARM_B_BL_Abs_d_Atom(_file, secName);
    addVeneerWithMapping<DefinedAtom::codeARM_d>(v_a, v_d, name);

    // Fake reference to show connection between parts of veneer.
    v_a->addReferenceELF_ARM(R_ARM_NONE, 0, v_d, 0);
    // Real reference to fixup.
    v_d->addReferenceELF_ARM(R_ARM_ABS32, 0, da, 0);
    return v_a;
  }

  /// \brief Get the veneer for Thumb B/BL instructions
  /// in absolute code.
  const VeneerAtom *getVeneer_THM_B_BL_Abs(const DefinedAtom *da,
                                           StringRef secName) {
    auto veneer = _veneerAtoms.lookup(da);
    if (!veneer.empty())
      return veneer._veneer;

    std::string name = "__";
    name += da->name();
    name += "_from_thumb";
    // Create parts of veneer with mapping symbols.
    auto v_t =
        new (_file._alloc) Veneer_THM_B_BL_Abs_t_Atom(_file, secName, name);
    addVeneerWithMapping<DefinedAtom::codeARM_t>(da, v_t, name);
    auto v_a = new (_file._alloc) Veneer_THM_B_BL_Abs_a_Atom(_file, secName);
    addVeneerWithMapping<DefinedAtom::codeARM_a>(v_t, v_a, name);

    // Fake reference to show connection between parts of veneer.
    v_t->addReferenceELF_ARM(R_ARM_NONE, 0, v_a, 0);
    // Real reference to fixup.
    v_a->addReferenceELF_ARM(R_ARM_JUMP24, 0, da, 0);
    return v_t;
  }

  std::error_code handleTLSIE32(const Reference &ref) {
    if (const auto *target = dyn_cast<DefinedAtom>(ref.target())) {
      const_cast<Reference &>(ref)
          .setTarget(static_cast<Derived *>(this)->getTLSTPOFF32(target));
      return std::error_code();
    }
    llvm_unreachable("R_ARM_TLS_IE32 reloc targets wrong atom type");
  }

  /// \brief Create a GOT entry for TLS with reloc type and addend specified.
  template <Reference::KindValue R_ARM_TLS, Reference::Addend A = 0>
  const GOTAtom *getGOTTLSEntry(const DefinedAtom *da) {
    StringRef source;
#ifndef NDEBUG
    source = "_tls_";
#endif
    return getGOT<R_ARM_TLS, A>(da, source);
  }

  /// \brief Add veneer with mapping symbol.
  template <DefinedAtom::CodeModel Model>
  void addVeneerWithMapping(const DefinedAtom *da, VeneerAtom *va,
                            const std::string &name) {
    assert(_veneerAtoms.lookup(da).empty() &&
           "Veneer or mapping already exists");
    auto *ma = new (_file._alloc)
        ARMVeneerMappingAtom<Model>(_file, va->customSectionName(), name);

    // Fake reference to show connection between the mapping symbol and veneer.
    va->addReferenceELF_ARM(R_ARM_NONE, 0, ma, 0);
    _veneerAtoms[da] = VeneerWithMapping(va, ma);
  }

  /// \brief get a veneer for a PLT entry.
  const PLTAtom *getPLTVeneer(const Atom *da, PLTAtom *pa, StringRef source) {
    std::string name = "__plt_from_thumb";
    name += source.empty() ? "_" : source;
    name += da->name();
    // Create veneer for PLT entry.
    auto va = new (_file._alloc) ARMPLTVeneerAtom(_file, name);
    // Fake reference to show connection between veneer and PLT entry.
    va->addReferenceELF_ARM(R_ARM_NONE, 0, pa, 0);

    _pltAtoms[da] = PLTWithVeneer(pa, va);
    return va;
  }

  typedef const GOTAtom *(Derived::*GOTFactory)(const Atom *);

  /// \brief get a PLT entry referencing PLTGOT entry.
  ///
  /// If the entry does not exist, both GOT and PLT entry are created.
  const PLTAtom *getPLT(const Atom *da, bool fromThumb, GOTFactory gotFactory,
                        StringRef source = "") {
    auto pltVeneer = _pltAtoms.lookup(da);
    if (!pltVeneer.empty()) {
      // Return clean PLT entry provided it is ARM code.
      if (!fromThumb)
        return pltVeneer._plt;

      // Check if veneer is present for Thumb to ARM transition.
      if (pltVeneer._veneer)
        return pltVeneer._veneer;

      // Create veneer for existing PLT entry.
      return getPLTVeneer(da, pltVeneer._plt, source);
    }

    // Create specific GOT entry.
    const auto *ga = (static_cast<Derived *>(this)->*gotFactory)(da);
    assert(_gotpltAtoms.lookup(da) == ga &&
           "GOT entry should be added to the PLTGOT map");
    assert(ga->customSectionName() == ".got.plt" &&
           "GOT entry should be in a special section");

    std::string name = "__plt";
    name += source.empty() ? "_" : source;
    name += da->name();
    // Create PLT entry for the GOT entry.
    auto pa = new (_file._alloc) ARMPLTAtom(_file, name);
    pa->addReferenceELF_ARM(R_ARM_ALU_PC_G0_NC, 0, ga, -8);
    pa->addReferenceELF_ARM(R_ARM_ALU_PC_G1_NC, 4, ga, -4);
    pa->addReferenceELF_ARM(R_ARM_LDR_PC_G2, 8, ga, 0);

    // Since all PLT entries are in ARM code, Thumb to ARM
    // switching should be added if the relocated place contais Thumb code.
    if (fromThumb)
      return getPLTVeneer(da, pa, source);

    // Otherwise just add PLT entry and return it to the caller.
    _pltAtoms[da] = PLTWithVeneer(pa);
    return pa;
  }

  /// \brief Create the GOT entry for a given IFUNC Atom.
  const GOTAtom *createIFUNCGOT(const Atom *da) {
    assert(!_gotpltAtoms.lookup(da) && "IFUNC GOT entry already exists");
    auto g = new (_file._alloc) ARMGOTPLTAtom(_file);
    g->addReferenceELF_ARM(R_ARM_ABS32, 0, da, 0);
    g->addReferenceELF_ARM(R_ARM_IRELATIVE, 0, da, 0);
#ifndef NDEBUG
    g->_name = "__got_ifunc_";
    g->_name += da->name();
#endif
    _gotpltAtoms[da] = g;
    return g;
  }

  /// \brief get the PLT entry for a given IFUNC Atom.
  const PLTAtom *getIFUNCPLTEntry(const DefinedAtom *da, bool fromThumb) {
    return getPLT(da, fromThumb, &Derived::createIFUNCGOT, "_ifunc_");
  }

  /// \brief Redirect the call to the PLT stub for the target IFUNC.
  ///
  /// This create a PLT and GOT entry for the IFUNC if one does not exist. The
  /// GOT entry and a IRELATIVE relocation to the original target resolver.
  std::error_code handleIFUNC(bool fromThumb, const Reference &ref) {
    auto target = dyn_cast<const DefinedAtom>(ref.target());
    if (target && target->contentType() == DefinedAtom::typeResolver) {
      const_cast<Reference &>(ref)
          .setTarget(getIFUNCPLTEntry(target, fromThumb));
    }
    return std::error_code();
  }

  /// \brief Create a GOT entry containing 0.
  const GOTAtom *getNullGOT() {
    if (!_null) {
      _null = new (_file._alloc) ARMGOTPLTAtom(_file);
#ifndef NDEBUG
      _null->_name = "__got_null";
#endif
    }
    return _null;
  }

  /// \brief Create regular GOT entry which cannot be used in PLTGOT operation.
  template <Reference::KindValue R_ARM_REL, Reference::Addend A = 0>
  const GOTAtom *getGOT(const Atom *da, StringRef source = "") {
    if (auto got = _gotAtoms.lookup(da))
      return got;
    auto g = new (_file._alloc) ARMGOTAtom(_file);
    g->addReferenceELF_ARM(R_ARM_REL, 0, da, A);
#ifndef NDEBUG
    g->_name = "__got";
    g->_name += source.empty() ? "_" : source;
    g->_name += da->name();
#endif
    _gotAtoms[da] = g;
    return g;
  }

  /// \brief get GOT entry for a regular defined atom.
  const GOTAtom *getGOTEntry(const DefinedAtom *da) {
    return getGOT<R_ARM_ABS32>(da);
  }

  std::error_code handleGOT(const Reference &ref) {
    if (isa<UndefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getNullGOT());
    else if (const auto *da = dyn_cast<DefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getGOTEntry(da));
    return std::error_code();
  }

public:
  ARMRelocationPass(const ELFLinkingContext &ctx) : _file(ctx), _ctx(ctx) {}

  /// \brief Do the pass.
  ///
  /// The goal here is to first process each reference individually. Each call
  /// to handleReference may modify the reference itself and/or create new
  /// atoms which must be stored in one of the maps below.
  ///
  /// After all references are handled, the atoms created during that are all
  /// added to mf.
  std::error_code perform(SimpleFile &mf) override {
    ScopedTask task(getDefaultDomain(), "ARM GOT/PLT Pass");
    DEBUG_WITH_TYPE(
        "ARM", llvm::dbgs() << "Undefined Atoms" << "\n";
        for (const auto &atom
             : mf.undefined()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        }

        llvm::dbgs() << "Shared Library Atoms" << "\n";
        for (const auto &atom
             : mf.sharedLibrary()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        }

        llvm::dbgs() << "Absolute Atoms" << "\n";
        for (const auto &atom
             : mf.absolute()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        }

        llvm::dbgs() << "Defined Atoms" << "\n";
        for (const auto &atom
             : mf.defined()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        });

    // Process all references.
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
      _plt0_d->setOrdinal(ordinal++);
      mf.addAtom(*_plt0_d);
    }
    for (auto &pltKV : _pltAtoms) {
      auto &plt = pltKV.second;
      if (auto *v = plt._veneer) {
        v->setOrdinal(ordinal++);
        mf.addAtom(*v);
      }
      auto *p = plt._plt;
      p->setOrdinal(ordinal++);
      mf.addAtom(*p);
    }
    if (_null) {
      _null->setOrdinal(ordinal++);
      mf.addAtom(*_null);
    }
    if (_plt0) {
      _got0->setOrdinal(ordinal++);
      mf.addAtom(*_got0);
      _got1->setOrdinal(ordinal++);
      mf.addAtom(*_got1);
    }
    for (auto &gotKV : _gotAtoms) {
      auto &got = gotKV.second;
      got->setOrdinal(ordinal++);
      mf.addAtom(*got);
    }
    for (auto &gotKV : _gotpltAtoms) {
      auto &got = gotKV.second;
      got->setOrdinal(ordinal++);
      mf.addAtom(*got);
    }
    for (auto &objectKV : _objectAtoms) {
      auto &obj = objectKV.second;
      obj->setOrdinal(ordinal++);
      mf.addAtom(*obj);
    }
    for (auto &veneerKV : _veneerAtoms) {
      auto &veneer = veneerKV.second;
      auto *m = veneer._mapping;
      m->setOrdinal(ordinal++);
      mf.addAtom(*m);
      auto *v = veneer._veneer;
      v->setOrdinal(ordinal++);
      mf.addAtom(*v);
    }

    return std::error_code();
  }

protected:
  /// \brief Owner of all the Atoms created by this pass.
  ELFPassFile _file;
  const ELFLinkingContext &_ctx;

  /// \brief Map Atoms to their GOT entries.
  llvm::MapVector<const Atom *, GOTAtom *> _gotAtoms;

  /// \brief Map Atoms to their PLTGOT entries.
  llvm::MapVector<const Atom *, GOTAtom *> _gotpltAtoms;

  /// \brief Map Atoms to their Object entries.
  llvm::MapVector<const Atom *, ObjectAtom *> _objectAtoms;

  /// \brief Map Atoms to their PLT entries depending on the code model.
  struct PLTWithVeneer {
    PLTWithVeneer(PLTAtom *p = nullptr, PLTAtom *v = nullptr)
        : _plt(p), _veneer(v) {}

    bool empty() const {
      assert((_plt || !_veneer) && "Veneer appears without PLT entry");
      return !_plt && !_veneer;
    }

    PLTAtom *_plt;
    PLTAtom *_veneer;
  };
  llvm::MapVector<const Atom *, PLTWithVeneer> _pltAtoms;

  /// \brief Map Atoms to their veneers.
  struct VeneerWithMapping {
    VeneerWithMapping(VeneerAtom *v = nullptr, VeneerAtom *m = nullptr)
        : _veneer(v), _mapping(m) {}

    bool empty() const {
      assert(((bool)_veneer == (bool)_mapping) &&
             "Mapping symbol should always be paired with veneer");
      return !_veneer && !_mapping;
    }

    VeneerAtom *_veneer;
    VeneerAtom *_mapping;
  };
  llvm::MapVector<const Atom *, VeneerWithMapping> _veneerAtoms;

  /// \brief GOT entry that is always 0. Used for undefined weaks.
  GOTAtom *_null = nullptr;

  /// \brief The got and plt entries for .PLT0. This is used to call into the
  /// dynamic linker for symbol resolution.
  /// @{
  PLT0Atom *_plt0 = nullptr;
  PLT0Atom *_plt0_d = nullptr;
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
class ARMStaticRelocationPass final
    : public ARMRelocationPass<ARMStaticRelocationPass> {
public:
  ARMStaticRelocationPass(const elf::ARMLinkingContext &ctx)
      : ARMRelocationPass(ctx) {}

  /// \brief Handle ordinary relocation references.
  std::error_code handlePlain(bool fromThumb, const Reference &ref) {
    return handleIFUNC(fromThumb, ref);
  }

  /// \brief Get the veneer for ARM B/BL instructions.
  const VeneerAtom *getVeneer_ARM_B_BL(const DefinedAtom *da,
                                       StringRef secName) {
    return getVeneer_ARM_B_BL_Abs(da, secName);
  }

  /// \brief Get the veneer for Thumb B/BL instructions.
  const VeneerAtom *getVeneer_THM_B_BL(const DefinedAtom *da,
                                       StringRef secName) {
    return getVeneer_THM_B_BL_Abs(da, secName);
  }

  /// \brief Create a GOT entry for R_ARM_TLS_TPOFF32 reloc.
  const GOTAtom *getTLSTPOFF32(const DefinedAtom *da) {
    return getGOTTLSEntry<R_ARM_TLS_LE32>(da);
  }
};

/// This implements the dynamic relocation model. GOT and PLT entries are
/// created for references that cannot be directly resolved.
class ARMDynamicRelocationPass final
    : public ARMRelocationPass<ARMDynamicRelocationPass> {
public:
  ARMDynamicRelocationPass(const elf::ARMLinkingContext &ctx)
      : ARMRelocationPass(ctx) {}

  /// \brief get the PLT entry for a given atom.
  const PLTAtom *getPLTEntry(const SharedLibraryAtom *sla, bool fromThumb) {
    return getPLT(sla, fromThumb, &ARMDynamicRelocationPass::createPLTGOT);
  }

  /// \brief Create the GOT entry for a given atom.
  const GOTAtom *createPLTGOT(const Atom *da) {
    assert(!_gotpltAtoms.lookup(da) && "PLTGOT entry already exists");
    auto g = new (_file._alloc) ARMGOTPLTAtom(_file);
    g->addReferenceELF_ARM(R_ARM_ABS32, 0, getPLT0(), 0);
    g->addReferenceELF_ARM(R_ARM_JUMP_SLOT, 0, da, 0);
#ifndef NDEBUG
    g->_name = "__got_plt0_";
    g->_name += da->name();
#endif
    _gotpltAtoms[da] = g;
    return g;
  }

  const ObjectAtom *getObjectEntry(const SharedLibraryAtom *a) {
    if (auto obj = _objectAtoms.lookup(a))
      return obj;

    auto oa = new (_file._alloc) ARMObjectAtom(_file);
    oa->addReferenceELF_ARM(R_ARM_COPY, 0, oa, 0);

    oa->_name = a->name();
    oa->_size = a->size();

    _objectAtoms[a] = oa;
    return oa;
  }

  /// \brief Handle ordinary relocation references.
  std::error_code handlePlain(bool fromThumb, const Reference &ref) {
    if (auto sla = dyn_cast<SharedLibraryAtom>(ref.target())) {
      if (sla->type() == SharedLibraryAtom::Type::Data &&
          _ctx.getOutputELFType() == llvm::ELF::ET_EXEC) {
        const_cast<Reference &>(ref).setTarget(getObjectEntry(sla));
      } else if (sla->type() == SharedLibraryAtom::Type::Code) {
        const_cast<Reference &>(ref).setTarget(getPLTEntry(sla, fromThumb));
      }
      return std::error_code();
    }
    return handleIFUNC(fromThumb, ref);
  }

  /// \brief Get the veneer for ARM B/BL instructions.
  const VeneerAtom *getVeneer_ARM_B_BL(const DefinedAtom *da,
                                       StringRef secName) {
    if (_ctx.getOutputELFType() == llvm::ELF::ET_EXEC) {
      return getVeneer_ARM_B_BL_Abs(da, secName);
    }
    llvm_unreachable("Handle ARM veneer for DSOs");
  }

  /// \brief Get the veneer for Thumb B/BL instructions.
  const VeneerAtom *getVeneer_THM_B_BL(const DefinedAtom *da,
                                       StringRef secName) {
    if (_ctx.getOutputELFType() == llvm::ELF::ET_EXEC) {
      return getVeneer_THM_B_BL_Abs(da, secName);
    }
    llvm_unreachable("Handle Thumb veneer for DSOs");
  }

  /// \brief Create a GOT entry for R_ARM_TLS_TPOFF32 reloc.
  const GOTAtom *getTLSTPOFF32(const DefinedAtom *da) {
    return getGOTTLSEntry<R_ARM_TLS_TPOFF32>(da);
  }

  const PLT0Atom *getPLT0() {
    if (_plt0)
      return _plt0;
    // Fill in the null entry.
    getNullGOT();
    _plt0 = new (_file._alloc) ARMPLT0_a_Atom(_file, "__PLT0");
    _plt0_d = new (_file._alloc) ARMPLT0_d_Atom(_file, "__PLT0_d");
    _got0 = new (_file._alloc) ARMGOTPLTAtom(_file);
    _got1 = new (_file._alloc) ARMGOTPLTAtom(_file);
    _plt0_d->addReferenceELF_ARM(R_ARM_REL32, 0, _got1, 0);
    // Fake reference to show connection between the GOT and PLT entries.
    _plt0->addReferenceELF_ARM(R_ARM_NONE, 0, _got0, 0);
    // Fake reference to show connection between parts of PLT entry.
    _plt0->addReferenceELF_ARM(R_ARM_NONE, 0, _plt0_d, 0);
#ifndef NDEBUG
    _got0->_name = "__got0";
    _got1->_name = "__got1";
#endif
    return _plt0;
  }

  const GOTAtom *getSharedGOTEntry(const SharedLibraryAtom *sla) {
    return getGOT<R_ARM_GLOB_DAT>(sla);
  }

  std::error_code handleGOT(const Reference &ref) {
    if (const auto sla = dyn_cast<const SharedLibraryAtom>(ref.target())) {
      const_cast<Reference &>(ref).setTarget(getSharedGOTEntry(sla));
      return std::error_code();
    }
    return ARMRelocationPass::handleGOT(ref);
  }
};

} // end of anon namespace

std::unique_ptr<Pass>
lld::elf::createARMRelocationPass(const ARMLinkingContext &ctx) {
  switch (ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    if (ctx.isDynamic())
      return llvm::make_unique<ARMDynamicRelocationPass>(ctx);
    return llvm::make_unique<ARMStaticRelocationPass>(ctx);
  case llvm::ELF::ET_DYN:
    return llvm::make_unique<ARMDynamicRelocationPass>(ctx);
  default:
    llvm_unreachable("Unhandled output file type");
  }
}
