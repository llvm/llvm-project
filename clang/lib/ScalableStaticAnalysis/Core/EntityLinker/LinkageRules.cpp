//===- LinkageRules.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Each rule below cites the probe that establishes it. The probes, including
// the full programs and linker output, are in
// docs/ssaf-linker-{elf,macho,coff}-behavior.md.
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LinkageRules.h"
#include "clang/ScalableStaticAnalysis/Core/Support/ErrorBuilder.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang::ssaf {

namespace {

constexpr const char *UnsupportedObjectFormat =
    "unsupported object format for target triple '{0}': the entity linker "
    "emulates ELF, Mach-O and COFF";

} // namespace

EntityLinkageType LinkageRules::linkageOf(const EntityLinkage &L) {
  return L.Linkage;
}

EntityBinding LinkageRules::bindingOf(const EntityLinkage &L) {
  return L.Binding;
}

EntityCoalescing LinkageRules::coalescingOf(const EntityLinkage &L) {
  return L.Coalescing;
}

EntityVisibility LinkageRules::visibilityOf(const EntityLinkage &L) {
  return L.Visibility;
}

bool LinkageRules::defines(const EntityLinkage &Linkage) {
  return Linkage.DefinitionKind == EntityDefinitionKind::Definition;
}

EntityLinkage LinkageRules::withVisibility(const EntityLinkage &Linkage,
                                           EntityVisibility V) {
  return EntityLinkage(Linkage.Linkage, Linkage.Binding, Linkage.Coalescing, V,
                       Linkage.DefinitionKind);
}

EntityVisibility
LinkageRules::mergeVisibility(const EntityLinkage &Current,
                              const EntityLinkage &Incoming) const {
  return visibilityRank(visibilityOf(Incoming)) >
                 visibilityRank(visibilityOf(Current))
             ? visibilityOf(Incoming)
             : visibilityOf(Current);
}

bool LinkageRules::isOrderDependentMerge(const EntityLinkage &,
                                         const EntityLinkage &) const {
  return false;
}

EntityLinkage LinkageRules::normalize(const EntityLinkage &Linkage) const {
  return Linkage;
}

bool LinkageRules::isRepresentable(const EntityLinkage &) const { return true; }

//===----------------------------------------------------------------------===//
// ELF
//===----------------------------------------------------------------------===//

namespace {

class ELFLinkageRules : public LinkageRules {
public:
  llvm::StringRef getName() const override { return "ELF"; }

  /// ELF has no general coalescing mechanism, so clang lowers an ODR
  /// definition to a weak symbol plus a COMDAT group:
  ///
  ///   inline int inl(void) { return 42; }
  ///   $ llvm-nm inl_a.o  ->  W _Z3inlv
  ///   $ llvm-readelf --section-groups inl_a.o
  ///     COMDAT group section [4] `.group' [_Z3inlv]
  ///
  /// The weak binding is what lets two copies coexist, and what makes an
  /// ordinary definition of the same symbol win rather than conflict.
  /// (ELF doc §3, §8.)
  EntityBinding effectiveBinding(const EntityLinkage &Linkage) const override {
    if (bindingOf(Linkage) == EntityBinding::Strong &&
        coalescingOf(Linkage) == EntityCoalescing::ODR) {
      return EntityBinding::Weak;
    }
    return bindingOf(Linkage);
  }

  /// Undefined < Weak < Common < Strong.
  ///
  ///   int g;                             // common, 4 bytes
  ///   __attribute__((weak)) int g = 9;   // weak definition
  ///   $ ld.lld -o out main.o weakdef_g.o common_big.o
  ///     g -> the common (size 32 survives)   => Common > Weak
  ///
  ///   int g = 7;                         // strong definition
  ///   $ ld.lld -o out main.o common_big.o def_g.o
  ///     g -> def_g.o (size 4)                => Strong > Common
  ///
  /// (ELF doc §2, probes C2b and C3.)
  unsigned strengthRank(EntityBinding B) const override {
    switch (B) {
    case EntityBinding::Undefined:
      return 0;
    case EntityBinding::Weak:
      return 1;
    case EntityBinding::Common:
      return 2;
    case EntityBinding::Strong:
      return 3;
    }
    llvm_unreachable("Unhandled EntityBinding variant");
  }

  /// Most restrictive wins: Default < Protected < Hidden.
  ///
  ///   // tu1.c
  ///   int v(void) { return 1; }                                 // default
  ///   // tu2.c
  ///   __attribute__((visibility("hidden"))) int v(void);        // hidden decl
  ///   $ ld.lld -shared -o out.so tu1.o tu2.o
  ///     v: FUNC LOCAL HIDDEN, absent from .dynsym
  ///
  /// Note a hidden *declaration* demotes a default-visibility definition, and
  /// that hidden beats protected in either order. (ELF doc §4 V1, §6.2.)
  unsigned visibilityRank(EntityVisibility V) const override {
    switch (V) {
    case EntityVisibility::Default:
      return 0;
    case EntityVisibility::Protected:
      return 1;
    case EntityVisibility::Hidden:
      return 2;
    }
    llvm_unreachable("Unhandled EntityVisibility variant");
  }

  /// Two non-weak definitions collide; anything involving a weak one does not.
  ///
  ///   int f(void) { return 1; }   // strong_a.c
  ///   int f(void) { return 2; }   // strong_b.c
  ///   $ ld.lld -o out main.o strong_a.o strong_b.o
  ///     ld.lld: error: duplicate symbol: f
  ///
  /// Replacing either with __attribute__((weak)) links cleanly, as does an
  /// inline definition, which lowers to weak. Commons never conflict.
  /// (ELF doc §1, probes P1-P4; §2 C1.)
  bool isConflictingDefinition(const EntityLinkage &Current,
                               const EntityLinkage &Incoming) const override {
    return effectiveBinding(Current) != EntityBinding::Weak &&
           effectiveBinding(Incoming) != EntityBinding::Weak &&
           effectiveBinding(Current) != EntityBinding::Common &&
           effectiveBinding(Incoming) != EntityBinding::Common;
  }
};

//===----------------------------------------------------------------------===//
// Mach-O
//===----------------------------------------------------------------------===//

class MachOLinkageRules : public LinkageRules {
public:
  llvm::StringRef getName() const override { return "Mach-O"; }

  /// Like ELF, Mach-O expresses an ODR definition as a weak one — the symbol
  /// table entry is byte-identical to __attribute__((weak)):
  ///
  ///   inline int q(void) { return 42; }
  ///   $ llvm-nm -m inl.o    -> (__TEXT,__text) weak external __Z1qv
  ///   $ llvm-nm -m weak.o   -> (__TEXT,__text) weak external __Z1qv
  ///
  /// and it resolves the same way: first-wins against another weak definition,
  /// loses to a strong one. (Mach-O doc §4, §7.1.)
  EntityBinding effectiveBinding(const EntityLinkage &Linkage) const override {
    if (bindingOf(Linkage) == EntityBinding::Strong &&
        coalescingOf(Linkage) == EntityCoalescing::ODR) {
      return EntityBinding::Weak;
    }
    return bindingOf(Linkage);
  }

  /// Undefined < Common < Weak < Strong.
  ///
  /// Mach-O inverts Weak and Common relative to ELF and COFF: a weak
  /// definition displaces a common one, and ld-prime says so out loud.
  ///
  ///   int g;                             // common, 32 bytes
  ///   __attribute__((weak)) int g = 9;   // weak definition, 4 bytes
  ///   $ ld -o out main.o weakdef_g.o common_big.o
  ///     ld: warning: tentative definition of '_g' with size 32 ... is being
  ///         replaced by real definition of smaller size 4
  ///     _g -> (__DATA,__data) weak external
  ///
  /// (Mach-O doc §3, probe MC3.)
  unsigned strengthRank(EntityBinding B) const override {
    switch (B) {
    case EntityBinding::Undefined:
      return 0;
    case EntityBinding::Common:
      return 1;
    case EntityBinding::Weak:
      return 2;
    case EntityBinding::Strong:
      return 3;
    }
    llvm_unreachable("Unhandled EntityBinding variant");
  }

  /// LEAST restrictive wins — the opposite of ELF.
  ///
  ///   // tu1.c
  ///   __attribute__((weak, visibility("hidden"))) int w(void) { return 1; }
  ///   // tu2.c
  ///   __attribute__((weak)) int w(void) { return 2; }
  ///   $ ld -o out main.o tu1.o tu2.o     (either order)
  ///     _w -> weak external               i.e. exported, not private
  ///
  /// Only when every copy is hidden does the merged symbol stay private, which
  /// is `privateExtern &= isPrivateExtern` in lld/MachO/SymbolTable.cpp:118.
  /// Mach-O has no protected visibility at all: clang warns and downgrades it
  /// to default, so it is ranked alongside default here.
  /// (Mach-O doc §2, probes W1-W3.)
  unsigned visibilityRank(EntityVisibility V) const override {
    switch (V) {
    case EntityVisibility::Hidden:
      return 0;
    case EntityVisibility::Default:
    case EntityVisibility::Protected:
      return 1;
    }
    llvm_unreachable("Unhandled EntityVisibility variant");
  }

  /// Same predicate as ELF: two non-weak definitions collide.
  ///
  ///   int f(void) { return 1; }   // strong_a.c
  ///   int f(void) { return 2; }   // strong_b.c
  ///   $ ld -o out main.o strong_a.o strong_b.o
  ///     duplicate symbol '_f' in: strong_b.o, strong_a.o
  ///
  /// (Mach-O doc §1, probes M1-M4.)
  bool isConflictingDefinition(const EntityLinkage &Current,
                               const EntityLinkage &Incoming) const override {
    return effectiveBinding(Current) != EntityBinding::Weak &&
           effectiveBinding(Incoming) != EntityBinding::Weak &&
           effectiveBinding(Current) != EntityBinding::Common &&
           effectiveBinding(Incoming) != EntityBinding::Common;
  }

  /// Common symbols do not merge their visibility: ld-prime keeps whichever
  /// common it saw first, so the result depends on link order.
  ///
  ///   // tu1.c
  ///   __attribute__((visibility("hidden"))) int g;
  ///   // tu2.c
  ///   int g;
  ///   $ ld -o out main.o tu1.o tu2.o   -> _g non-external (private)
  ///   $ ld -o out main.o tu2.o tu1.o   -> _g external
  ///
  /// addCommon() carries the isPrivateExtern of the common it keeps, with no
  /// merge step; only addDefined()'s weak-def path performs the `&=`. We
  /// reproduce that faithfully and warn, rather than approximating it
  /// commutatively. (Mach-O doc §7.2.)
  EntityVisibility
  mergeVisibility(const EntityLinkage &Current,
                  const EntityLinkage &Incoming) const override {
    if (isCommonPair(Current, Incoming)) {
      return visibilityOf(Current);
    }
    return LinkageRules::mergeVisibility(Current, Incoming);
  }

  bool isOrderDependentMerge(const EntityLinkage &Current,
                             const EntityLinkage &Incoming) const override {
    return isCommonPair(Current, Incoming) &&
           visibilityOf(Current) != visibilityOf(Incoming);
  }

  /// Protected is not representable, but clang never emits it for a Mach-O
  /// target: it warns and downgrades to default at compile time.
  ///
  ///   $ clang --target=arm64-apple-macosx -c v.c
  ///     warning: target does not support 'protected' visibility;
  ///              using 'default' [-Wunsupported-visibility]
  ///
  /// A summary that carries it therefore did not come from the compiler, which
  /// is worth failing over rather than silently coercing. (Mach-O doc §2.)
  bool isRepresentable(const EntityLinkage &Linkage) const override {
    return visibilityOf(Linkage) != EntityVisibility::Protected;
  }

private:
  static bool isCommonPair(const EntityLinkage &Current,
                           const EntityLinkage &Incoming) {
    return defines(Current) && defines(Incoming) &&
           bindingOf(Current) == EntityBinding::Common &&
           bindingOf(Incoming) == EntityBinding::Common;
  }
};

//===----------------------------------------------------------------------===//
// COFF
//===----------------------------------------------------------------------===//

class COFFLinkageRules : public LinkageRules {
public:
  llvm::StringRef getName() const override { return "COFF"; }

  /// COFF is the one format that expresses ODR directly: the binding stays
  /// strong and a COMDAT licenses the duplicates.
  ///
  ///   inline int inl(void) { return 42; }
  ///   $ llvm-nm inl_a.obj      -> T ?inl@@YAHXZ        (strong, not weak)
  ///   $ llvm-readobj --sections inl_a.obj
  ///       IMAGE_SCN_LNK_COMDAT (0x1000)
  ///       Selection: Any (0x2)
  ///
  /// So no lowering is needed. (COFF doc §2.)
  EntityBinding effectiveBinding(const EntityLinkage &Linkage) const override {
    return bindingOf(Linkage);
  }

  /// Undefined < Weak < Common < Strong, as on ELF.
  ///
  ///   $ lld-link ... /map:k2.map read.obj common_a.obj def_g.obj
  ///       g -> def_g.obj          => Strong > Common
  ///   $ lld-link ... /map:k3.map read.obj weakdef_g.obj common_big.obj
  ///       g -> <common>           => Common > Weak
  ///
  /// (COFF doc §6.1.)
  unsigned strengthRank(EntityBinding B) const override {
    switch (B) {
    case EntityBinding::Undefined:
      return 0;
    case EntityBinding::Weak:
      return 1;
    case EntityBinding::Common:
      return 2;
    case EntityBinding::Strong:
      return 3;
    }
    llvm_unreachable("Unhandled EntityBinding variant");
  }

  /// COFF has no symbol visibility. Export is controlled per-DLL by
  /// __declspec(dllexport) and .def files, which is not a per-symbol attribute
  /// merged during resolution, so every value ranks equally here and
  /// normalize() coerces them all to Default. (COFF doc §5, §7.)
  unsigned visibilityRank(EntityVisibility) const override { return 0; }

  /// Always Default, matching normalize(). Ranking every value equally would
  /// otherwise make the merge keep whichever occurrence was linked first,
  /// which is an order dependence on a field the platform does not have.
  EntityVisibility mergeVisibility(const EntityLinkage &,
                                   const EntityLinkage &) const override {
    return EntityVisibility::Default;
  }

  /// COMDAT, not weakness, is what licenses duplicate definitions.
  ///
  ///   inline int inl(void) { return 42; }   // COMDAT, in two TUs
  ///   $ lld-link ... main.obj inl_a.obj inl_b.obj        -> links
  ///
  ///   int inl(void) { return 99; }          // regular definition
  ///   $ lld-link ... main.obj inl_comdat.obj inl_regular.obj
  ///       lld-link: error: duplicate symbol: int __cdecl inl(void)
  ///
  ///   __attribute__((weak)) int f(void);    // in two TUs
  ///   $ lld-link ... main.obj weak_a.obj weak_b.obj
  ///       lld-link: error: duplicate symbol: .weak.f.default
  ///
  /// That last case is the one ELF and Mach-O accept: COFF emulates weak
  /// symbols with an alias, so two of them collide on the alias while one weak
  /// against one strong does not. (COFF doc §1 C1/C4, §2 CI1/K4, §6.)
  bool isConflictingDefinition(const EntityLinkage &Current,
                               const EntityLinkage &Incoming) const override {
    // Commons merge rather than conflict.
    if (bindingOf(Current) == EntityBinding::Common ||
        bindingOf(Incoming) == EntityBinding::Common) {
      return false;
    }
    // Two COMDATs coalesce.
    if (coalescingOf(Current) == EntityCoalescing::ODR &&
        coalescingOf(Incoming) == EntityCoalescing::ODR) {
      return false;
    }
    // Exactly one weak definition yields to the other; two collide on the
    // alias COFF uses to emulate them.
    const bool CurrentWeak = bindingOf(Current) == EntityBinding::Weak;
    const bool IncomingWeak = bindingOf(Incoming) == EntityBinding::Weak;
    if (CurrentWeak != IncomingWeak) {
      return false;
    }
    return true;
  }

  /// Visibility is dropped at emission, so a summary carrying Hidden or
  /// Protected is coerced rather than rejected: clang accepts both silently on
  /// Windows, and portable code applies them unconditionally.
  ///
  ///   __attribute__((visibility("hidden"))) int g;
  ///   $ clang --target=x86_64-pc-windows-msvc -c g.c    (no diagnostic)
  ///   $ llvm-nm g.obj  ->  00000004 C g                 (indistinguishable)
  ///
  /// (COFF doc §7.)
  EntityLinkage normalize(const EntityLinkage &Linkage) const override {
    if (visibilityOf(Linkage) == EntityVisibility::Default) {
      return Linkage;
    }
    return withVisibility(Linkage, EntityVisibility::Default);
  }
};

} // namespace

const LinkageRules &LinkageRules::forTarget(const llvm::Triple &TargetTriple) {
  static const ELFLinkageRules ELF;
  static const MachOLinkageRules MachO;
  static const COFFLinkageRules COFF;

  switch (TargetTriple.getObjectFormat()) {
  case llvm::Triple::ELF:
    return ELF;
  case llvm::Triple::MachO:
    return MachO;
  case llvm::Triple::COFF:
    return COFF;
  default:
    break;
  }

  ErrorBuilder::fatal(UnsupportedObjectFormat, TargetTriple.str());
}

} // namespace clang::ssaf
