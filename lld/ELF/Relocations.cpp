//===- Relocations.cpp ----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains platform-independent functions to process relocations.
// I'll describe the overview of this file here.
//
// Simple relocations are easy to handle for the linker. For example,
// for R_X86_64_PC64 relocs, the linker just has to fix up locations
// with the relative offsets to the target symbols. It would just be
// reading records from relocation sections and applying them to output.
//
// But not all relocations are that easy to handle. For example, for
// R_386_GOTOFF relocs, the linker has to create new GOT entries for
// symbols if they don't exist, and fix up locations with GOT entry
// offsets from the beginning of GOT section. So there is more than
// fixing addresses in relocation processing.
//
// ELF defines a large number of complex relocations.
//
// The functions in this file analyze relocations and do whatever needs
// to be done. It includes, but not limited to, the following.
//
//  - create GOT/PLT entries
//  - create new relocations in .dynsym to let the dynamic linker resolve
//    them at runtime (since ELF supports dynamic linking, not all
//    relocations can be resolved at link-time)
//  - create COPY relocs and reserve space in .bss
//  - replace expensive relocs (in terms of runtime cost) with cheap ones
//  - error out infeasible combinations such as PIC and non-relative relocs
//
// Note that the functions in this file don't actually apply relocations
// because it doesn't know about the output file nor the output file buffer.
// It instead stores Relocation objects to InputSection's Relocations
// vector to let it apply later in InputSection::writeTo.
//
//===----------------------------------------------------------------------===//

#include "Relocations.h"
#include "Config.h"
#include "Memory.h"
#include "OutputSections.h"
#include "Strings.h"
#include "SymbolTable.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "Thunks.h"

#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support::endian;

namespace lld {
namespace elf {

static bool refersToGotEntry(RelExpr Expr) {
  return isRelExprOneOf<R_GOT, R_GOT_OFF, R_MIPS_GOT_LOCAL_PAGE, R_MIPS_GOT_OFF,
                        R_MIPS_GOT_OFF32, R_MIPS_TLSGD, R_MIPS_TLSLD,
                        R_GOT_PAGE_PC, R_GOT_PC, R_GOT_FROM_END, R_TLSGD,
                        R_TLSGD_PC, R_TLSDESC, R_TLSDESC_PAGE>(Expr);
}

static bool isPreemptible(const SymbolBody &Body, uint32_t Type) {
  // In case of MIPS GP-relative relocations always resolve to a definition
  // in a regular input file, ignoring the one-definition rule. So we,
  // for example, should not attempt to create a dynamic relocation even
  // if the target symbol is preemptible. There are two two MIPS GP-relative
  // relocations R_MIPS_GPREL16 and R_MIPS_GPREL32. But only R_MIPS_GPREL16
  // can be against a preemptible symbol.
  // To get MIPS relocation type we apply 0xff mask. In case of O32 ABI all
  // relocation types occupy eight bit. In case of N64 ABI we extract first
  // relocation from 3-in-1 packet because only the first relocation can
  // be against a real symbol.
  if (Config->EMachine == EM_MIPS && (Type & 0xff) == R_MIPS_GPREL16)
    return false;
  return Body.isPreemptible();
}

// This function is similar to the `handleTlsRelocation`. ARM and MIPS do not
// support any relaxations for TLS relocations so by factoring out ARM and MIPS
// handling in to the separate function we can simplify the code and do not
// pollute `handleTlsRelocation` by ARM and MIPS `ifs` statements.
template <class ELFT, class GOT>
static unsigned handleNoRelaxTlsRelocation(
    GOT *Got, uint32_t Type, SymbolBody &Body, InputSectionBase<ELFT> &C,
    typename ELFT::uint Offset, typename ELFT::uint Addend, RelExpr Expr) {
  typedef typename ELFT::uint uintX_t;
  auto addModuleReloc = [](SymbolBody &Body, GOT *Got, uintX_t Off, bool LD) {
    // The Dynamic TLS Module Index Relocation can be statically resolved to 1
    // if we know that we are linking an executable. For ARM we resolve the
    // relocation when writing the Got. MIPS has a custom Got implementation
    // that writes the Module index in directly.
    if (!Body.isPreemptible() && !Config->pic() && Config->EMachine == EM_ARM)
      Got->Relocations.push_back(
          {R_ABS, Target->TlsModuleIndexRel, Off, 0, &Body});
    else {
      SymbolBody *Dest = LD ? nullptr : &Body;
      In<ELFT>::RelaDyn->addReloc(
          {Target->TlsModuleIndexRel, Got, Off, false, Dest, 0});
    }
  };
  if (Expr == R_MIPS_TLSLD || Expr == R_TLSLD_PC) {
    if (Got->addTlsIndex() && (Config->pic() || Config->EMachine == EM_ARM))
      addModuleReloc(Body, Got, Got->getTlsIndexOff(), true);
    C.Relocations.push_back({Expr, Type, Offset, Addend, &Body});
    return 1;
  }
  if (Target->isTlsGlobalDynamicRel(Type)) {
    if (Got->addDynTlsEntry(Body) &&
        (Body.isPreemptible() || Config->EMachine == EM_ARM)) {
      uintX_t Off = Got->getGlobalDynOffset(Body);
      addModuleReloc(Body, Got, Off, false);
      if (Body.isPreemptible())
        In<ELFT>::RelaDyn->addReloc({Target->TlsOffsetRel, Got,
                                     Off + (uintX_t)sizeof(uintX_t), false,
                                     &Body, 0});
    }
    C.Relocations.push_back({Expr, Type, Offset, Addend, &Body});
    return 1;
  }
  return 0;
}

// Returns the number of relocations processed.
template <class ELFT>
static unsigned handleTlsRelocation(uint32_t Type, SymbolBody &Body,
                                    InputSectionBase<ELFT> &C,
                                    typename ELFT::uint Offset,
                                    typename ELFT::uint Addend, RelExpr Expr) {
  if (!(C.Flags & SHF_ALLOC))
    return 0;

  if (!Body.isTls())
    return 0;

  typedef typename ELFT::uint uintX_t;

  if (Config->EMachine == EM_ARM)
    return handleNoRelaxTlsRelocation<ELFT>(In<ELFT>::Got, Type, Body, C,
                                            Offset, Addend, Expr);
  if (Config->EMachine == EM_MIPS)
    return handleNoRelaxTlsRelocation<ELFT>(In<ELFT>::MipsGot, Type, Body, C,
                                            Offset, Addend, Expr);

  bool IsPreemptible = isPreemptible(Body, Type);
  if ((Expr == R_TLSDESC || Expr == R_TLSDESC_PAGE || Expr == R_TLSDESC_CALL) &&
      Config->Shared) {
    if (In<ELFT>::Got->addDynTlsEntry(Body)) {
      uintX_t Off = In<ELFT>::Got->getGlobalDynOffset(Body);
      In<ELFT>::RelaDyn->addReloc({Target->TlsDescRel, In<ELFT>::Got, Off,
                                   !IsPreemptible, &Body, 0});
    }
    if (Expr != R_TLSDESC_CALL)
      C.Relocations.push_back({Expr, Type, Offset, Addend, &Body});
    return 1;
  }

  if (Expr == R_TLSLD_PC || Expr == R_TLSLD) {
    // Local-Dynamic relocs can be relaxed to Local-Exec.
    if (!Config->Shared) {
      C.Relocations.push_back(
          {R_RELAX_TLS_LD_TO_LE, Type, Offset, Addend, &Body});
      return 2;
    }
    if (In<ELFT>::Got->addTlsIndex())
      In<ELFT>::RelaDyn->addReloc({Target->TlsModuleIndexRel, In<ELFT>::Got,
                                   In<ELFT>::Got->getTlsIndexOff(), false,
                                   nullptr, 0});
    C.Relocations.push_back({Expr, Type, Offset, Addend, &Body});
    return 1;
  }

  // Local-Dynamic relocs can be relaxed to Local-Exec.
  if (Target->isTlsLocalDynamicRel(Type) && !Config->Shared) {
    C.Relocations.push_back(
        {R_RELAX_TLS_LD_TO_LE, Type, Offset, Addend, &Body});
    return 1;
  }

  if (Expr == R_TLSDESC_PAGE || Expr == R_TLSDESC || Expr == R_TLSDESC_CALL ||
      Target->isTlsGlobalDynamicRel(Type)) {
    if (Config->Shared) {
      if (In<ELFT>::Got->addDynTlsEntry(Body)) {
        uintX_t Off = In<ELFT>::Got->getGlobalDynOffset(Body);
        In<ELFT>::RelaDyn->addReloc(
            {Target->TlsModuleIndexRel, In<ELFT>::Got, Off, false, &Body, 0});

        // If the symbol is preemptible we need the dynamic linker to write
        // the offset too.
        uintX_t OffsetOff = Off + (uintX_t)sizeof(uintX_t);
        if (IsPreemptible)
          In<ELFT>::RelaDyn->addReloc({Target->TlsOffsetRel, In<ELFT>::Got,
                                       OffsetOff, false, &Body, 0});
        else
          In<ELFT>::Got->Relocations.push_back(
              {R_ABS, Target->TlsOffsetRel, OffsetOff, 0, &Body});
      }
      C.Relocations.push_back({Expr, Type, Offset, Addend, &Body});
      return 1;
    }

    // Global-Dynamic relocs can be relaxed to Initial-Exec or Local-Exec
    // depending on the symbol being locally defined or not.
    if (IsPreemptible) {
      C.Relocations.push_back(
          {Target->adjustRelaxExpr(Type, nullptr, R_RELAX_TLS_GD_TO_IE), Type,
           Offset, Addend, &Body});
      if (!Body.isInGot()) {
        In<ELFT>::Got->addEntry(Body);
        In<ELFT>::RelaDyn->addReloc({Target->TlsGotRel, In<ELFT>::Got,
                                     Body.getGotOffset<ELFT>(), false, &Body,
                                     0});
      }
      return Target->TlsGdRelaxSkip;
    }
    C.Relocations.push_back(
        {Target->adjustRelaxExpr(Type, nullptr, R_RELAX_TLS_GD_TO_LE), Type,
         Offset, Addend, &Body});
    return Target->TlsGdRelaxSkip;
  }

  // Initial-Exec relocs can be relaxed to Local-Exec if the symbol is locally
  // defined.
  if (Target->isTlsInitialExecRel(Type) && !Config->Shared && !IsPreemptible) {
    C.Relocations.push_back(
        {R_RELAX_TLS_IE_TO_LE, Type, Offset, Addend, &Body});
    return 1;
  }
  return 0;
}

template <endianness E> static int16_t readSignedLo16(const uint8_t *Loc) {
  return read32<E>(Loc) & 0xffff;
}

template <class RelTy>
static uint32_t getMipsPairType(const RelTy *Rel, const SymbolBody &Sym) {
  switch (Rel->getType(Config->Mips64EL)) {
  case R_MIPS_HI16:
    return R_MIPS_LO16;
  case R_MIPS_GOT16:
    return Sym.isLocal() ? R_MIPS_LO16 : R_MIPS_NONE;
  case R_MIPS_PCHI16:
    return R_MIPS_PCLO16;
  case R_MICROMIPS_HI16:
    return R_MICROMIPS_LO16;
  default:
    return R_MIPS_NONE;
  }
}

template <class ELFT, class RelTy>
static int32_t findMipsPairedAddend(const uint8_t *Buf, const uint8_t *BufLoc,
                                    SymbolBody &Sym, const RelTy *Rel,
                                    const RelTy *End) {
  uint32_t SymIndex = Rel->getSymbol(Config->Mips64EL);
  uint32_t Type = getMipsPairType(Rel, Sym);

  // Some MIPS relocations use addend calculated from addend of the relocation
  // itself and addend of paired relocation. ABI requires to compute such
  // combined addend in case of REL relocation record format only.
  // See p. 4-17 at ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
  if (RelTy::IsRela || Type == R_MIPS_NONE)
    return 0;

  for (const RelTy *RI = Rel; RI != End; ++RI) {
    if (RI->getType(Config->Mips64EL) != Type)
      continue;
    if (RI->getSymbol(Config->Mips64EL) != SymIndex)
      continue;
    const endianness E = ELFT::TargetEndianness;
    return ((read32<E>(BufLoc) & 0xffff) << 16) +
           readSignedLo16<E>(Buf + RI->r_offset);
  }
  warn("can't find matching " + toString(Type) + " relocation for " +
       toString(Rel->getType(Config->Mips64EL)));
  return 0;
}

// True if non-preemptable symbol always has the same value regardless of where
// the DSO is loaded.
template <class ELFT> static bool isAbsolute(const SymbolBody &Body) {
  if (Body.isUndefined())
    return !Body.isLocal() && Body.symbol()->isWeak();
  if (const auto *DR = dyn_cast<DefinedRegular<ELFT>>(&Body))
    return DR->Section == nullptr; // Absolute symbol.
  return false;
}

template <class ELFT> static bool isAbsoluteValue(const SymbolBody &Body) {
  return isAbsolute<ELFT>(Body) || Body.isTls();
}

static bool needsPlt(RelExpr Expr) {
  return isRelExprOneOf<R_PLT_PC, R_PPC_PLT_OPD, R_PLT, R_PLT_PAGE_PC>(Expr);
}

// True if this expression is of the form Sym - X, where X is a position in the
// file (PC, or GOT for example).
static bool isRelExpr(RelExpr Expr) {
  return isRelExprOneOf<R_PC, R_GOTREL, R_GOTREL_FROM_END, R_MIPS_GOTREL,
                        R_PAGE_PC, R_RELAX_GOT_PC>(Expr);
}

template <class ELFT>
static bool isStaticLinkTimeConstant(RelExpr E, uint32_t Type,
                                     const SymbolBody &Body,
                                     InputSectionBase<ELFT> &S,
                                     typename ELFT::uint RelOff) {
  // These expressions always compute a constant
  if (isRelExprOneOf<R_SIZE, R_GOT_FROM_END, R_GOT_OFF, R_MIPS_GOT_LOCAL_PAGE,
                     R_MIPS_GOT_OFF, R_MIPS_GOT_OFF32, R_MIPS_TLSGD,
                     R_GOT_PAGE_PC, R_GOT_PC, R_PLT_PC, R_TLSGD_PC, R_TLSGD,
                     R_PPC_PLT_OPD, R_TLSDESC_CALL, R_TLSDESC_PAGE, R_HINT>(E))
    return true;

  // These never do, except if the entire file is position dependent or if
  // only the low bits are used.
  if (E == R_GOT || E == R_PLT || E == R_TLSDESC)
    return Target->usesOnlyLowPageBits(Type) || !Config->pic();

  if (isPreemptible(Body, Type))
    return false;

  if (!Config->pic())
    return true;

  bool AbsVal = isAbsoluteValue<ELFT>(Body);
  bool RelE = isRelExpr(E);
  if (AbsVal && !RelE)
    return true;
  if (!AbsVal && RelE)
    return true;

  // Relative relocation to an absolute value. This is normally unrepresentable,
  // but if the relocation refers to a weak undefined symbol, we allow it to
  // resolve to the image base. This is a little strange, but it allows us to
  // link function calls to such symbols. Normally such a call will be guarded
  // with a comparison, which will load a zero from the GOT.
  // Another special case is MIPS _gp_disp symbol which represents offset
  // between start of a function and '_gp' value and defined as absolute just
  // to simplify the code.
  if (AbsVal && RelE) {
    if (Body.isUndefined() && !Body.isLocal() && Body.symbol()->isWeak())
      return true;
    if (&Body == ElfSym<ELFT>::MipsGpDisp)
      return true;
    error(S.getLocation(RelOff) + ": relocation " + toString(Type) +
          " cannot refer to absolute symbol '" + toString(Body) +
          "' defined in " + toString(Body.File));
    return true;
  }

  return Target->usesOnlyLowPageBits(Type);
}

static RelExpr toPlt(RelExpr Expr) {
  if (Expr == R_PPC_OPD)
    return R_PPC_PLT_OPD;
  if (Expr == R_PC)
    return R_PLT_PC;
  if (Expr == R_PAGE_PC)
    return R_PLT_PAGE_PC;
  if (Expr == R_ABS)
    return R_PLT;
  return Expr;
}

static RelExpr fromPlt(RelExpr Expr) {
  // We decided not to use a plt. Optimize a reference to the plt to a
  // reference to the symbol itself.
  if (Expr == R_PLT_PC)
    return R_PC;
  if (Expr == R_PPC_PLT_OPD)
    return R_PPC_OPD;
  if (Expr == R_PLT)
    return R_ABS;
  return Expr;
}

template <class ELFT> static uint32_t getAlignment(SharedSymbol<ELFT> *SS) {
  typedef typename ELFT::uint uintX_t;

  uintX_t SecAlign = SS->file()->getSection(SS->Sym)->sh_addralign;
  uintX_t SymValue = SS->Sym.st_value;
  int TrailingZeros =
      std::min(countTrailingZeros(SecAlign), countTrailingZeros(SymValue));
  return 1 << TrailingZeros;
}

template <class ELFT> static bool isReadOnly(SharedSymbol<ELFT> *SS) {
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Phdr Elf_Phdr;

  // Determine if the symbol is read-only by scanning the DSO's program headers.
  uintX_t Value = SS->Sym.st_value;
  for (const Elf_Phdr &Phdr : check(SS->file()->getObj().program_headers()))
    if ((Phdr.p_type == ELF::PT_LOAD || Phdr.p_type == ELF::PT_GNU_RELRO) &&
        !(Phdr.p_flags & ELF::PF_W) && Value >= Phdr.p_vaddr &&
        Value < Phdr.p_vaddr + Phdr.p_memsz)
      return true;
  return false;
}

// Reserve space in .bss or .bss.rel.ro for copy relocation.
template <class ELFT> static void addCopyRelSymbol(SharedSymbol<ELFT> *SS) {
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Sym Elf_Sym;

  // Copy relocation against zero-sized symbol doesn't make sense.
  uintX_t SymSize = SS->template getSize<ELFT>();
  if (SymSize == 0)
    fatal("cannot create a copy relocation for symbol " + toString(*SS));

  // See if this symbol is in a read-only segment. If so, preserve the symbol's
  // memory protection by reserving space in the .bss.rel.ro section.
  bool IsReadOnly = isReadOnly(SS);
  OutputSection<ELFT> *CopySec =
      IsReadOnly ? Out<ELFT>::BssRelRo : Out<ELFT>::Bss;

  uintX_t Alignment = getAlignment(SS);
  uintX_t Off = alignTo(CopySec->Size, Alignment);
  CopySec->Size = Off + SymSize;
  CopySec->updateAlignment(Alignment);
  uintX_t Shndx = SS->Sym.st_shndx;
  uintX_t Value = SS->Sym.st_value;

  // Create a SyntheticSection in CopySec to hold the .bss and the Copy Reloc
  auto *CopyISec = make<CopyRelSection<ELFT>>(IsReadOnly, Alignment, SymSize);
  CopyISec->OutSecOff = Off;
  CopyISec->OutSec = CopySec;
  CopySec->Sections.push_back(CopyISec);

  // Look through the DSO's dynamic symbol table for aliases and create a
  // dynamic symbol for each one. This causes the copy relocation to correctly
  // interpose any aliases.
  for (const Elf_Sym &S : SS->file()->getGlobalSymbols()) {
    if (S.st_shndx != Shndx || S.st_value != Value)
      continue;
    auto *Alias = dyn_cast_or_null<SharedSymbol<ELFT>>(
        Symtab<ELFT>::X->find(check(S.getName(SS->file()->getStringTable()))));
    if (!Alias)
      continue;
    Alias->CopySection = CopyISec;
    Alias->NeedsCopyOrPltAddr = true;
    Alias->symbol()->IsUsedInRegularObj = true;
  }
  In<ELFT>::RelaDyn->addReloc({Target->CopyRel, CopyISec, 0, false, SS, 0});
}

template <class ELFT>
static RelExpr adjustExpr(const elf::ObjectFile<ELFT> &File, SymbolBody &Body,
                          bool IsWrite, RelExpr Expr, uint32_t Type,
                          const uint8_t *Data, InputSectionBase<ELFT> &S,
                          typename ELFT::uint RelOff) {
  bool Preemptible = isPreemptible(Body, Type);
  if (Body.isGnuIFunc()) {
    Expr = toPlt(Expr);
  } else if (!Preemptible) {
    if (needsPlt(Expr))
      Expr = fromPlt(Expr);
    if (Expr == R_GOT_PC && !isAbsoluteValue<ELFT>(Body))
      Expr = Target->adjustRelaxExpr(Type, Data, Expr);
  }

  if (IsWrite || isStaticLinkTimeConstant<ELFT>(Expr, Type, Body, S, RelOff))
    return Expr;

  // This relocation would require the dynamic linker to write a value to read
  // only memory. We can hack around it if we are producing an executable and
  // the refered symbol can be preemepted to refer to the executable.
  if (Config->Shared || (Config->pic() && !isRelExpr(Expr))) {
    error(S.getLocation(RelOff) + ": can't create dynamic relocation " +
          toString(Type) + " against " +
          (Body.getName().empty() ? "local symbol in readonly segment"
                                  : "symbol '" + toString(Body) + "'") +
          " defined in " + toString(Body.File));
    return Expr;
  }
  if (Body.getVisibility() != STV_DEFAULT) {
    error(S.getLocation(RelOff) + ": cannot preempt symbol '" + toString(Body) +
          "' defined in " + toString(Body.File));
    return Expr;
  }
  if (Body.isObject()) {
    // Produce a copy relocation.
    auto *B = cast<SharedSymbol<ELFT>>(&Body);
    if (!B->needsCopy())
      addCopyRelSymbol(B);
    return Expr;
  }
  if (Body.isFunc()) {
    // This handles a non PIC program call to function in a shared library. In
    // an ideal world, we could just report an error saying the relocation can
    // overflow at runtime. In the real world with glibc, crt1.o has a
    // R_X86_64_PC32 pointing to libc.so.
    //
    // The general idea on how to handle such cases is to create a PLT entry and
    // use that as the function value.
    //
    // For the static linking part, we just return a plt expr and everything
    // else will use the the PLT entry as the address.
    //
    // The remaining problem is making sure pointer equality still works. We
    // need the help of the dynamic linker for that. We let it know that we have
    // a direct reference to a so symbol by creating an undefined symbol with a
    // non zero st_value. Seeing that, the dynamic linker resolves the symbol to
    // the value of the symbol we created. This is true even for got entries, so
    // pointer equality is maintained. To avoid an infinite loop, the only entry
    // that points to the real function is a dedicated got entry used by the
    // plt. That is identified by special relocation types (R_X86_64_JUMP_SLOT,
    // R_386_JMP_SLOT, etc).
    Body.NeedsCopyOrPltAddr = true;
    return toPlt(Expr);
  }
  error("symbol '" + toString(Body) + "' defined in " + toString(Body.File) +
        " is missing type");

  return Expr;
}

template <class ELFT, class RelTy>
static typename ELFT::uint computeAddend(const elf::ObjectFile<ELFT> &File,
                                         const uint8_t *SectionData,
                                         const RelTy *End, const RelTy &RI,
                                         RelExpr Expr, SymbolBody &Body) {
  typedef typename ELFT::uint uintX_t;

  uint32_t Type = RI.getType(Config->Mips64EL);
  uintX_t Addend = getAddend<ELFT>(RI);
  const uint8_t *BufLoc = SectionData + RI.r_offset;
  if (!RelTy::IsRela)
    Addend += Target->getImplicitAddend(BufLoc, Type);
  if (Config->EMachine == EM_MIPS) {
    Addend += findMipsPairedAddend<ELFT>(SectionData, BufLoc, Body, &RI, End);
    if (Type == R_MIPS_LO16 && Expr == R_PC)
      // R_MIPS_LO16 expression has R_PC type iif the target is _gp_disp
      // symbol. In that case we should use the following formula for
      // calculation "AHL + GP - P + 4". Let's add 4 right here.
      // For details see p. 4-19 at
      // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
      Addend += 4;
    if (Expr == R_MIPS_GOTREL && Body.isLocal())
      Addend += File.MipsGp0;
  }
  if (Config->pic() && Config->EMachine == EM_PPC64 && Type == R_PPC64_TOC)
    Addend += getPPC64TocBase();
  return Addend;
}

template <class ELFT>
static void reportUndefined(SymbolBody &Sym, InputSectionBase<ELFT> &S,
                            typename ELFT::uint Offset) {
  bool CanBeExternal = Sym.symbol()->computeBinding() != STB_LOCAL &&
                       Sym.getVisibility() == STV_DEFAULT;
  if (Config->UnresolvedSymbols == UnresolvedPolicy::IgnoreAll ||
      (Config->UnresolvedSymbols == UnresolvedPolicy::Ignore && CanBeExternal))
    return;

  std::string Msg =
      S.getLocation(Offset) + ": undefined symbol '" + toString(Sym) + "'";

  if (Config->UnresolvedSymbols == UnresolvedPolicy::WarnAll ||
      (Config->UnresolvedSymbols == UnresolvedPolicy::Warn && CanBeExternal))
    warn(Msg);
  else
    error(Msg);
}

template <class RelTy>
static std::pair<uint32_t, uint32_t>
mergeMipsN32RelTypes(uint32_t Type, uint32_t Offset, RelTy *I, RelTy *E) {
  // MIPS N32 ABI treats series of successive relocations with the same offset
  // as a single relocation. The similar approach used by N64 ABI, but this ABI
  // packs all relocations into the single relocation record. Here we emulate
  // this for the N32 ABI. Iterate over relocation with the same offset and put
  // theirs types into the single bit-set.
  uint32_t Processed = 0;
  for (; I != E && Offset == I->r_offset; ++I) {
    ++Processed;
    Type |= I->getType(Config->Mips64EL) << (8 * Processed);
  }
  return std::make_pair(Type, Processed);
}

// The reason we have to do this early scan is as follows
// * To mmap the output file, we need to know the size
// * For that, we need to know how many dynamic relocs we will have.
// It might be possible to avoid this by outputting the file with write:
// * Write the allocated output sections, computing addresses.
// * Apply relocations, recording which ones require a dynamic reloc.
// * Write the dynamic relocations.
// * Write the rest of the file.
// This would have some drawbacks. For example, we would only know if .rela.dyn
// is needed after applying relocations. If it is, it will go after rw and rx
// sections. Given that it is ro, we will need an extra PT_LOAD. This
// complicates things for the dynamic linker and means we would have to reserve
// space for the extra PT_LOAD even if we end up not using it.
template <class ELFT, class RelTy>
static void scanRelocs(InputSectionBase<ELFT> &C, ArrayRef<RelTy> Rels) {
  typedef typename ELFT::uint uintX_t;

  bool IsWrite = C.Flags & SHF_WRITE;

  auto AddDyn = [=](const DynamicReloc<ELFT> &Reloc) {
    In<ELFT>::RelaDyn->addReloc(Reloc);
  };

  const elf::ObjectFile<ELFT> *File = C.getFile();
  ArrayRef<uint8_t> SectionData = C.Data;
  const uint8_t *Buf = SectionData.begin();

  ArrayRef<EhSectionPiece> Pieces;
  if (auto *Eh = dyn_cast<EhInputSection<ELFT>>(&C))
    Pieces = Eh->Pieces;

  ArrayRef<EhSectionPiece>::iterator PieceI = Pieces.begin();
  ArrayRef<EhSectionPiece>::iterator PieceE = Pieces.end();

  for (auto I = Rels.begin(), E = Rels.end(); I != E; ++I) {
    const RelTy &RI = *I;
    SymbolBody &Body = File->getRelocTargetSym(RI);
    uint32_t Type = RI.getType(Config->Mips64EL);

    if (Config->MipsN32Abi) {
      uint32_t Processed;
      std::tie(Type, Processed) =
          mergeMipsN32RelTypes(Type, RI.r_offset, I + 1, E);
      I += Processed;
    }

    // We only report undefined symbols if they are referenced somewhere in the
    // code.
    if (!Body.isLocal() && Body.isUndefined() && !Body.symbol()->isWeak())
      reportUndefined(Body, C, RI.r_offset);

    RelExpr Expr = Target->getRelExpr(Type, Body);
    bool Preemptible = isPreemptible(Body, Type);
    Expr = adjustExpr(*File, Body, IsWrite, Expr, Type, Buf + RI.r_offset, C,
                      RI.r_offset);
    if (ErrorCount)
      continue;

    // Skip a relocation that points to a dead piece
    // in a eh_frame section.
    while (PieceI != PieceE &&
           (PieceI->InputOff + PieceI->size() <= RI.r_offset))
      ++PieceI;

    // Compute the offset of this section in the output section. We do it here
    // to try to compute it only once.
    uintX_t Offset;
    if (PieceI != PieceE) {
      assert(PieceI->InputOff <= RI.r_offset && "Relocation not in any piece");
      if (PieceI->OutputOff == -1)
        continue;
      Offset = PieceI->OutputOff + RI.r_offset - PieceI->InputOff;
    } else {
      Offset = RI.r_offset;
    }

    // This relocation does not require got entry, but it is relative to got and
    // needs it to be created. Here we request for that.
    if (Expr == R_GOTONLY_PC || Expr == R_GOTONLY_PC_FROM_END ||
        Expr == R_GOTREL || Expr == R_GOTREL_FROM_END || Expr == R_PPC_TOC)
      In<ELFT>::Got->HasGotOffRel = true;

    uintX_t Addend = computeAddend(*File, Buf, E, RI, Expr, Body);

    if (unsigned Processed =
            handleTlsRelocation<ELFT>(Type, Body, C, Offset, Addend, Expr)) {
      I += (Processed - 1);
      continue;
    }

    // Ignore "hint" and TLS Descriptor call relocation because they are
    // only markers for relaxation.
    if (isRelExprOneOf<R_HINT, R_TLSDESC_CALL>(Expr))
      continue;

    if (needsPlt(Expr) ||
        refersToGotEntry(Expr) || !isPreemptible(Body, Type)) {
      // If the relocation points to something in the file, we can process it.
      bool Constant =
          isStaticLinkTimeConstant<ELFT>(Expr, Type, Body, C, RI.r_offset);

      // If the output being produced is position independent, the final value
      // is still not known. In that case we still need some help from the
      // dynamic linker. We can however do better than just copying the incoming
      // relocation. We can process some of it and and just ask the dynamic
      // linker to add the load address.
      if (!Constant)
        AddDyn({Target->RelativeRel, &C, Offset, true, &Body, Addend});

      // If the produced value is a constant, we just remember to write it
      // when outputting this section. We also have to do it if the format
      // uses Elf_Rel, since in that case the written value is the addend.
      if (Constant || !RelTy::IsRela)
        C.Relocations.push_back({Expr, Type, Offset, Addend, &Body});
    } else {
      // We don't know anything about the finaly symbol. Just ask the dynamic
      // linker to handle the relocation for us.
      if (!Target->isPicRel(Type))
        error(C.getLocation(Offset) + ": relocation " + toString(Type) +
              " cannot be used against shared object; recompile with -fPIC.");
      AddDyn({Target->getDynRel(Type), &C, Offset, false, &Body, Addend});

      // MIPS ABI turns using of GOT and dynamic relocations inside out.
      // While regular ABI uses dynamic relocations to fill up GOT entries
      // MIPS ABI requires dynamic linker to fills up GOT entries using
      // specially sorted dynamic symbol table. This affects even dynamic
      // relocations against symbols which do not require GOT entries
      // creation explicitly, i.e. do not have any GOT-relocations. So if
      // a preemptible symbol has a dynamic relocation we anyway have
      // to create a GOT entry for it.
      // If a non-preemptible symbol has a dynamic relocation against it,
      // dynamic linker takes it st_value, adds offset and writes down
      // result of the dynamic relocation. In case of preemptible symbol
      // dynamic linker performs symbol resolution, writes the symbol value
      // to the GOT entry and reads the GOT entry when it needs to perform
      // a dynamic relocation.
      // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf p.4-19
      if (Config->EMachine == EM_MIPS)
        In<ELFT>::MipsGot->addEntry(Body, Addend, Expr);
      continue;
    }

    // At this point we are done with the relocated position. Some relocations
    // also require us to create a got or plt entry.

    // If a relocation needs PLT, we create a PLT and a GOT slot for the symbol.
    if (needsPlt(Expr)) {
      if (Body.isInPlt())
        continue;

      if (Body.isGnuIFunc() && !Preemptible) {
        In<ELFT>::Iplt->addEntry(Body);
        In<ELFT>::IgotPlt->addEntry(Body);
        In<ELFT>::RelaIplt->addReloc({Target->IRelativeRel, In<ELFT>::IgotPlt,
                                      Body.getGotPltOffset<ELFT>(),
                                      !Preemptible, &Body, 0});
      } else {
        In<ELFT>::Plt->addEntry(Body);
        In<ELFT>::GotPlt->addEntry(Body);
        In<ELFT>::RelaPlt->addReloc({Target->PltRel, In<ELFT>::GotPlt,
                                     Body.getGotPltOffset<ELFT>(), !Preemptible,
                                     &Body, 0});
      }
      continue;
    }

    if (refersToGotEntry(Expr)) {
      if (Config->EMachine == EM_MIPS) {
        // MIPS ABI has special rules to process GOT entries and doesn't
        // require relocation entries for them. A special case is TLS
        // relocations. In that case dynamic loader applies dynamic
        // relocations to initialize TLS GOT entries.
        // See "Global Offset Table" in Chapter 5 in the following document
        // for detailed description:
        // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
        In<ELFT>::MipsGot->addEntry(Body, Addend, Expr);
        if (Body.isTls() && Body.isPreemptible())
          AddDyn({Target->TlsGotRel, In<ELFT>::MipsGot,
                  Body.getGotOffset<ELFT>(), false, &Body, 0});
        continue;
      }

      if (Body.isInGot())
        continue;

      In<ELFT>::Got->addEntry(Body);
      uintX_t Off = Body.getGotOffset<ELFT>();
      uint32_t DynType;
      RelExpr GotRE = R_ABS;
      if (Body.isTls()) {
        DynType = Target->TlsGotRel;
        GotRE = R_TLS;
      } else if (!Preemptible && Config->pic() && !isAbsolute<ELFT>(Body))
        DynType = Target->RelativeRel;
      else
        DynType = Target->GotRel;

      // FIXME: this logic is almost duplicated above.
      bool Constant =
          !Preemptible && !(Config->pic() && !isAbsolute<ELFT>(Body));
      if (!Constant)
        AddDyn({DynType, In<ELFT>::Got, Off, !Preemptible, &Body, 0});
      if (Constant || (!RelTy::IsRela && !Preemptible))
        In<ELFT>::Got->Relocations.push_back({GotRE, DynType, Off, 0, &Body});
      continue;
    }
  }
}

template <class ELFT> void scanRelocations(InputSectionBase<ELFT> &S) {
  if (S.AreRelocsRela)
    scanRelocs(S, S.relas());
  else
    scanRelocs(S, S.rels());
}

// Insert the Thunks for OutputSection OS into their designated place
// in the Sections vector, and recalculate the InputSection output section
// offsets.
// This may invalidate any output section offsets stored outside of InputSection
template <class ELFT>
static void mergeThunks(OutputSection<ELFT> *OS,
                        std::vector<ThunkSection<ELFT> *> &Thunks) {
  // Order Thunks in ascending OutSecOff
  auto ThunkCmp = [](const ThunkSection<ELFT> *A, const ThunkSection<ELFT> *B) {
    return A->OutSecOff < B->OutSecOff;
  };
  std::stable_sort(Thunks.begin(), Thunks.end(), ThunkCmp);

  // Merge sorted vectors of Thunks and InputSections by OutSecOff
  std::vector<InputSection<ELFT> *> Tmp;
  Tmp.reserve(OS->Sections.size() + Thunks.size());
  auto MergeCmp = [](const InputSection<ELFT> *A, const InputSection<ELFT> *B) {
    // std::merge requires a strict weak ordering.
    if (A->OutSecOff < B->OutSecOff)
      return true;
    if (A->OutSecOff == B->OutSecOff)
      // Check if Thunk is immediately before any specific Target InputSection
      // for example Mips LA25 Thunks.
      if (auto *TA = dyn_cast<ThunkSection<ELFT>>(A))
        if (TA && TA->getTargetInputSection() == B)
          return true;
    return false;
  };
  std::merge(OS->Sections.begin(), OS->Sections.end(), Thunks.begin(),
             Thunks.end(), std::back_inserter(Tmp), MergeCmp);
  OS->Sections = std::move(Tmp);
  OS->Size = 0;
  OS->assignOffsets();
}

// Process all relocations from the InputSections that have been assigned
// to OutputSections and redirect through Thunks if needed.
//
// createThunks must be called after scanRelocs has created the Relocations for
// each InputSection. It must be called before the static symbol table is
// finalized. If any Thunks are added to an OutputSection the output section
// offsets of the InputSections will change.
//
// FIXME: All Thunks are assumed to be in range of the relocation. Range
// extension Thunks are not yet supported.
template <class ELFT>
void createThunks(ArrayRef<OutputSectionBase *> OutputSections) {
  // Track Symbols that already have a Thunk
  DenseMap<SymbolBody *, Thunk<ELFT> *> ThunkedSymbols;
  // Track InputSections that have a ThunkSection placed in front
  DenseMap<InputSection<ELFT> *, ThunkSection<ELFT> *> ThunkedSections;
  // Track the ThunksSections that need to be inserted into an OutputSection
  std::map<OutputSection<ELFT> *, std::vector<ThunkSection<ELFT> *>>
      ThunkSections;

  // Find or create a Thunk for Body for relocation Type
  auto GetThunk = [&](SymbolBody &Body, uint32_t Type) {
    auto res = ThunkedSymbols.insert({&Body, nullptr});
    if (res.second == true)
      res.first->second = addThunk<ELFT>(Type, Body);
    return std::make_pair(res.first->second, res.second);
  };

  // Find or create a ThunkSection to be placed immediately before IS
  auto GetISThunkSec = [&](InputSection<ELFT> *IS, OutputSection<ELFT> *OS) {
    ThunkSection<ELFT> *TS = ThunkedSections.lookup(IS);
    if (TS)
      return TS;
    auto *TOS = cast<OutputSection<ELFT>>(IS->OutSec);
    TS = make<ThunkSection<ELFT>>(TOS, IS->OutSecOff);
    ThunkSections[OS].push_back(TS);
    ThunkedSections[IS] = TS;
    return TS;
  };
  // Find or create a ThunkSection to be placed as last executable section in
  // OS.
  auto GetOSThunkSec = [&](ThunkSection<ELFT> *&TS, OutputSection<ELFT> *OS) {
    if (TS == nullptr) {
      uint32_t Off = 0;
      for (auto *IS : OS->Sections) {
        Off = IS->OutSecOff + IS->getSize();
        if ((IS->Flags & SHF_EXECINSTR) == 0)
          break;
      }
      TS = make<ThunkSection<ELFT>>(OS, Off);
      ThunkSections[OS].push_back(TS);
    }
    return TS;
  };
  // Create all the Thunks and insert them into synthetic ThunkSections. The
  // ThunkSections are later inserted back into the OutputSection.

  // We separate the creation of ThunkSections from the insertion of the
  // ThunkSections back into the OutputSection as ThunkSections are not always
  // inserted into the same OutputSection as the caller.
  for (OutputSectionBase *Base : OutputSections) {
    auto *OS = dyn_cast<OutputSection<ELFT>>(Base);
    if (OS == nullptr)
      continue;

    ThunkSection<ELFT> *OSTS = nullptr;
    for (InputSection<ELFT> *IS : OS->Sections) {
      for (Relocation &Rel : IS->Relocations) {
        SymbolBody &Body = *Rel.Sym;
        if (Target->needsThunk(Rel.Expr, Rel.Type, IS->getFile(), Body)) {
          Thunk<ELFT> *T;
          bool IsNew;
          std::tie(T, IsNew) = GetThunk(Body, Rel.Type);
          if (IsNew) {
            // Find or create a ThunkSection for the new Thunk
            ThunkSection<ELFT> *TS;
            if (auto *TIS = T->getTargetInputSection())
              TS = GetISThunkSec(TIS, OS);
            else
              TS = GetOSThunkSec(OSTS, OS);
            TS->addThunk(T);
          }
          // Redirect relocation to Thunk, we never go via the PLT to a Thunk
          Rel.Sym = T->ThunkSym;
          Rel.Expr = fromPlt(Rel.Expr);
        }
      }
    }
  }

  // Merge all created synthetic ThunkSections back into OutputSection
  for (auto &KV : ThunkSections)
    mergeThunks<ELFT>(KV.first, KV.second);
}

template void scanRelocations<ELF32LE>(InputSectionBase<ELF32LE> &);
template void scanRelocations<ELF32BE>(InputSectionBase<ELF32BE> &);
template void scanRelocations<ELF64LE>(InputSectionBase<ELF64LE> &);
template void scanRelocations<ELF64BE>(InputSectionBase<ELF64BE> &);

template void createThunks<ELF32LE>(ArrayRef<OutputSectionBase *>);
template void createThunks<ELF32BE>(ArrayRef<OutputSectionBase *>);
template void createThunks<ELF64LE>(ArrayRef<OutputSectionBase *>);
template void createThunks<ELF64BE>(ArrayRef<OutputSectionBase *>);
}
}
