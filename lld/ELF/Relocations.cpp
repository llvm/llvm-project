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
#include "OutputSections.h"
#include "SymbolTable.h"
#include "Target.h"
#include "Thunks.h"

#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support::endian;

namespace lld {
namespace elf {

static bool refersToGotEntry(RelExpr Expr) {
  return Expr == R_GOT || Expr == R_GOT_OFF || Expr == R_MIPS_GOT_LOCAL_PAGE ||
         Expr == R_MIPS_GOT_OFF || Expr == R_MIPS_TLSGD ||
         Expr == R_MIPS_TLSLD || Expr == R_GOT_PAGE_PC || Expr == R_GOT_PC ||
         Expr == R_GOT_FROM_END || Expr == R_TLSGD || Expr == R_TLSGD_PC ||
         Expr == R_TLSDESC || Expr == R_TLSDESC_PAGE;
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
// FIXME: The ARM implementation always adds the module index dynamic
// relocation even for non-preemptible symbols in applications. For static
// linking support we must either resolve the module index relocation at static
// link time, or hard code the module index (1) for the application in the GOT.
template <class ELFT>
static unsigned handleNoRelaxTlsRelocation(uint32_t Type, SymbolBody &Body,
                                           InputSectionBase<ELFT> &C,
                                           typename ELFT::uint Offset,
                                           typename ELFT::uint Addend,
                                           RelExpr Expr) {
  if (Expr == R_MIPS_TLSLD || Expr == R_TLSLD_PC) {
    if (Out<ELFT>::Got->addTlsIndex() &&
        (Config->Pic || Config->EMachine == EM_ARM))
      Out<ELFT>::RelaDyn->addReloc({Target->TlsModuleIndexRel, Out<ELFT>::Got,
                                    Out<ELFT>::Got->getTlsIndexOff(), false,
                                    nullptr, 0});
    C.Relocations.push_back({Expr, Type, Offset, Addend, &Body});
    return 1;
  }
  typedef typename ELFT::uint uintX_t;
  if (Target->isTlsGlobalDynamicRel(Type)) {
    if (Out<ELFT>::Got->addDynTlsEntry(Body) &&
        (Body.isPreemptible() || Config->EMachine == EM_ARM)) {
      uintX_t Off = Out<ELFT>::Got->getGlobalDynOffset(Body);
      Out<ELFT>::RelaDyn->addReloc(
          {Target->TlsModuleIndexRel, Out<ELFT>::Got, Off, false, &Body, 0});
      if (Body.isPreemptible())
        Out<ELFT>::RelaDyn->addReloc({Target->TlsOffsetRel, Out<ELFT>::Got,
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
  if (!(C.getSectionHdr()->sh_flags & SHF_ALLOC))
    return 0;

  if (!Body.isTls())
    return 0;

  typedef typename ELFT::uint uintX_t;

  if (Config->EMachine == EM_MIPS || Config->EMachine == EM_ARM)
    return handleNoRelaxTlsRelocation<ELFT>(Type, Body, C, Offset, Addend,
                                            Expr);

  if ((Expr == R_TLSDESC || Expr == R_TLSDESC_PAGE || Expr == R_HINT) &&
      Config->Shared) {
    if (Out<ELFT>::Got->addDynTlsEntry(Body)) {
      uintX_t Off = Out<ELFT>::Got->getGlobalDynOffset(Body);
      Out<ELFT>::RelaDyn->addReloc(
          {Target->TlsDescRel, Out<ELFT>::Got, Off, false, &Body, 0});
    }
    if (Expr != R_HINT)
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
    if (Out<ELFT>::Got->addTlsIndex())
      Out<ELFT>::RelaDyn->addReloc({Target->TlsModuleIndexRel, Out<ELFT>::Got,
                                    Out<ELFT>::Got->getTlsIndexOff(), false,
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

  if (Expr == R_TLSDESC_PAGE || Expr == R_TLSDESC || Expr == R_HINT ||
      Target->isTlsGlobalDynamicRel(Type)) {
    if (Config->Shared) {
      if (Out<ELFT>::Got->addDynTlsEntry(Body)) {
        uintX_t Off = Out<ELFT>::Got->getGlobalDynOffset(Body);
        Out<ELFT>::RelaDyn->addReloc(
            {Target->TlsModuleIndexRel, Out<ELFT>::Got, Off, false, &Body, 0});

        // If the symbol is preemptible we need the dynamic linker to write
        // the offset too.
        if (isPreemptible(Body, Type))
          Out<ELFT>::RelaDyn->addReloc({Target->TlsOffsetRel, Out<ELFT>::Got,
                                        Off + (uintX_t)sizeof(uintX_t), false,
                                        &Body, 0});
      }
      C.Relocations.push_back({Expr, Type, Offset, Addend, &Body});
      return 1;
    }

    // Global-Dynamic relocs can be relaxed to Initial-Exec or Local-Exec
    // depending on the symbol being locally defined or not.
    if (isPreemptible(Body, Type)) {
      C.Relocations.push_back(
          {Target->adjustRelaxExpr(Type, nullptr, R_RELAX_TLS_GD_TO_IE), Type,
           Offset, Addend, &Body});
      if (!Body.isInGot()) {
        Out<ELFT>::Got->addEntry(Body);
        Out<ELFT>::RelaDyn->addReloc({Target->TlsGotRel, Out<ELFT>::Got,
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
  if (Target->isTlsInitialExecRel(Type) && !Config->Shared &&
      !isPreemptible(Body, Type)) {
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
  warning("can't find matching " + getRelName(Type) + " relocation for " +
          getRelName(Rel->getType(Config->Mips64EL)));
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

static bool needsPlt(RelExpr Expr) {
  return Expr == R_PLT_PC || Expr == R_PPC_PLT_OPD || Expr == R_PLT ||
         Expr == R_PLT_PAGE_PC || Expr == R_THUNK_PLT_PC;
}

// True if this expression is of the form Sym - X, where X is a position in the
// file (PC, or GOT for example).
static bool isRelExpr(RelExpr Expr) {
  return Expr == R_PC || Expr == R_GOTREL || Expr == R_GOTREL_FROM_END ||
         Expr == R_PAGE_PC || Expr == R_RELAX_GOT_PC || Expr == R_THUNK_PC ||
         Expr == R_THUNK_PLT_PC;
}

template <class ELFT>
static bool isStaticLinkTimeConstant(RelExpr E, uint32_t Type,
                                     const SymbolBody &Body) {
  // These expressions always compute a constant
  if (E == R_SIZE || E == R_GOT_FROM_END || E == R_GOT_OFF ||
      E == R_MIPS_GOT_LOCAL_PAGE || E == R_MIPS_GOT_OFF || E == R_MIPS_TLSGD ||
      E == R_GOT_PAGE_PC || E == R_GOT_PC || E == R_PLT_PC || E == R_TLSGD_PC ||
      E == R_TLSGD || E == R_PPC_PLT_OPD || E == R_TLSDESC_PAGE ||
      E == R_HINT || E == R_THUNK_PC || E == R_THUNK_PLT_PC)
    return true;

  // These never do, except if the entire file is position dependent or if
  // only the low bits are used.
  if (E == R_GOT || E == R_PLT || E == R_TLSDESC)
    return Target->usesOnlyLowPageBits(Type) || !Config->Pic;

  if (isPreemptible(Body, Type))
    return false;

  if (!Config->Pic)
    return true;

  bool AbsVal = isAbsolute<ELFT>(Body) || Body.isTls();
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
  if (AbsVal && RelE) {
    if (Body.isUndefined() && !Body.isLocal() && Body.symbol()->isWeak())
      return true;
    error("relocation " + getRelName(Type) +
          " cannot refer to absolute symbol " + Body.getName());
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

// Reserve space in .bss for copy relocation.
template <class ELFT> static void addCopyRelSymbol(SharedSymbol<ELFT> *SS) {
  typedef typename ELFT::uint uintX_t;
  typedef typename ELFT::Sym Elf_Sym;

  // Copy relocation against zero-sized symbol doesn't make sense.
  uintX_t SymSize = SS->template getSize<ELFT>();
  if (SymSize == 0)
    fatal("cannot create a copy relocation for symbol " + SS->getName());

  uintX_t Alignment = getAlignment(SS);
  uintX_t Off = alignTo(Out<ELFT>::Bss->getSize(), Alignment);
  Out<ELFT>::Bss->setSize(Off + SymSize);
  Out<ELFT>::Bss->updateAlignment(Alignment);
  uintX_t Shndx = SS->Sym.st_shndx;
  uintX_t Value = SS->Sym.st_value;
  // Look through the DSO's dynamic symbol table for aliases and create a
  // dynamic symbol for each one. This causes the copy relocation to correctly
  // interpose any aliases.
  for (const Elf_Sym &S : SS->file()->getElfSymbols(true)) {
    if (S.st_shndx != Shndx || S.st_value != Value)
      continue;
    auto *Alias = dyn_cast_or_null<SharedSymbol<ELFT>>(
        Symtab<ELFT>::X->find(check(S.getName(SS->file()->getStringTable()))));
    if (!Alias)
      continue;
    Alias->OffsetInBss = Off;
    Alias->NeedsCopyOrPltAddr = true;
    Alias->symbol()->IsUsedInRegularObj = true;
  }
  Out<ELFT>::RelaDyn->addReloc(
      {Target->CopyRel, Out<ELFT>::Bss, SS->OffsetInBss, false, SS, 0});
}

template <class ELFT>
static StringRef getSymbolName(const elf::ObjectFile<ELFT> &File,
                               SymbolBody &Body) {
  if (Body.isLocal() && Body.getNameOffset())
    return File.getStringTable().data() + Body.getNameOffset();
  if (!Body.isLocal())
    return Body.getName();
  return "";
}

template <class ELFT>
static RelExpr adjustExpr(const elf::ObjectFile<ELFT> &File, SymbolBody &Body,
                          bool IsWrite, RelExpr Expr, uint32_t Type,
                          const uint8_t *Data) {
  bool Preemptible = isPreemptible(Body, Type);
  if (Body.isGnuIFunc()) {
    Expr = toPlt(Expr);
  } else if (!Preemptible) {
    if (needsPlt(Expr))
      Expr = fromPlt(Expr);
    if (Expr == R_GOT_PC)
      Expr = Target->adjustRelaxExpr(Type, Data, Expr);
  }
  Expr = Target->getThunkExpr(Expr, Type, File, Body);

  if (IsWrite || isStaticLinkTimeConstant<ELFT>(Expr, Type, Body))
    return Expr;

  // This relocation would require the dynamic linker to write a value to read
  // only memory. We can hack around it if we are producing an executable and
  // the refered symbol can be preemepted to refer to the executable.
  if (Config->Shared || (Config->Pic && !isRelExpr(Expr))) {
    StringRef Name = getSymbolName(File, Body);
    error("can't create dynamic relocation " + getRelName(Type) +
          " against " + (Name.empty() ? "readonly segment" : "symbol " + Name));
    return Expr;
  }
  if (Body.getVisibility() != STV_DEFAULT) {
    error("cannot preempt symbol " + Body.getName());
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
  error("symbol " + Body.getName() + " is missing type");

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
    if (Expr == R_GOTREL) {
      Addend -= MipsGPOffset;
      if (Body.isLocal())
        Addend += File.getMipsGp0();
    }
  }
  if (Config->Pic && Config->EMachine == EM_PPC64 && Type == R_PPC64_TOC)
    Addend += getPPC64TocBase();
  return Addend;
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

  bool IsWrite = C.getSectionHdr()->sh_flags & SHF_WRITE;

  auto AddDyn = [=](const DynamicReloc<ELFT> &Reloc) {
    Out<ELFT>::RelaDyn->addReloc(Reloc);
  };

  const elf::ObjectFile<ELFT> &File = *C.getFile();
  ArrayRef<uint8_t> SectionData = C.Data;
  const uint8_t *Buf = SectionData.begin();

  ArrayRef<EhSectionPiece> Pieces;
  if (auto *Eh = dyn_cast<EhInputSection<ELFT>>(&C))
    Pieces = Eh->Pieces;

  ArrayRef<EhSectionPiece>::iterator PieceI = Pieces.begin();
  ArrayRef<EhSectionPiece>::iterator PieceE = Pieces.end();

  for (auto I = Rels.begin(), E = Rels.end(); I != E; ++I) {
    const RelTy &RI = *I;
    SymbolBody &Body = File.getRelocTargetSym(RI);
    uint32_t Type = RI.getType(Config->Mips64EL);

    RelExpr Expr = Target->getRelExpr(Type, Body);
    bool Preemptible = isPreemptible(Body, Type);
    Expr = adjustExpr(File, Body, IsWrite, Expr, Type, Buf + RI.r_offset);
    if (HasError)
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
      if (PieceI->OutputOff == (size_t)-1)
        continue;
      Offset = PieceI->OutputOff + RI.r_offset - PieceI->InputOff;
    } else {
      Offset = RI.r_offset;
    }

    // This relocation does not require got entry, but it is relative to got and
    // needs it to be created. Here we request for that.
    if (Expr == R_GOTONLY_PC || Expr == R_GOTONLY_PC_FROM_END ||
        Expr == R_GOTREL || Expr == R_GOTREL_FROM_END || Expr == R_PPC_TOC)
      Out<ELFT>::Got->HasGotOffRel = true;

    uintX_t Addend = computeAddend(File, Buf, E, RI, Expr, Body);

    if (unsigned Processed =
            handleTlsRelocation<ELFT>(Type, Body, C, Offset, Addend, Expr)) {
      I += (Processed - 1);
      continue;
    }

    // Ignore "hint" relocation because it is for optional code optimization.
    if (Expr == R_HINT)
      continue;

    if (needsPlt(Expr) || Expr == R_THUNK_ABS || Expr == R_THUNK_PC ||
        Expr == R_THUNK_PLT_PC || refersToGotEntry(Expr) ||
        !isPreemptible(Body, Type)) {
      // If the relocation points to something in the file, we can process it.
      bool Constant = isStaticLinkTimeConstant<ELFT>(Expr, Type, Body);

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
        Out<ELFT>::Got->addMipsEntry(Body, Addend, Expr);
      continue;
    }

    // At this point we are done with the relocated position. Some relocations
    // also require us to create a got or plt entry.

    // If a relocation needs PLT, we create a PLT and a GOT slot for the symbol.
    if (needsPlt(Expr)) {
      if (Body.isInPlt())
        continue;
      Out<ELFT>::Plt->addEntry(Body);

      uint32_t Rel;
      if (Body.isGnuIFunc() && !Preemptible)
        Rel = Target->IRelativeRel;
      else
        Rel = Target->PltRel;

      Out<ELFT>::GotPlt->addEntry(Body);
      Out<ELFT>::RelaPlt->addReloc({Rel, Out<ELFT>::GotPlt,
                                    Body.getGotPltOffset<ELFT>(), !Preemptible,
                                    &Body, 0});
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
        Out<ELFT>::Got->addMipsEntry(Body, Addend, Expr);
        if (Body.isTls() && Body.isPreemptible())
          AddDyn({Target->TlsGotRel, Out<ELFT>::Got, Body.getGotOffset<ELFT>(),
                  false, &Body, 0});
        continue;
      }

      if (Body.isInGot())
        continue;

      Out<ELFT>::Got->addEntry(Body);
      if (Preemptible || (Config->Pic && !isAbsolute<ELFT>(Body))) {
        uint32_t DynType;
        if (Body.isTls())
          DynType = Target->TlsGotRel;
        else if (Preemptible)
          DynType = Target->GotRel;
        else
          DynType = Target->RelativeRel;
        AddDyn({DynType, Out<ELFT>::Got, Body.getGotOffset<ELFT>(),
                !Preemptible, &Body, 0});
      }
      continue;
    }
  }
}

template <class ELFT>
void scanRelocations(InputSectionBase<ELFT> &S,
                     const typename ELFT::Shdr &RelSec) {
  ELFFile<ELFT> &EObj = S.getFile()->getObj();
  if (RelSec.sh_type == SHT_RELA)
    scanRelocs(S, EObj.relas(&RelSec));
  else
    scanRelocs(S, EObj.rels(&RelSec));
}

template <class ELFT, class RelTy>
static void createThunks(InputSectionBase<ELFT> &C, ArrayRef<RelTy> Rels) {
  const elf::ObjectFile<ELFT> &File = *C.getFile();
  for (const RelTy &Rel : Rels) {
    SymbolBody &Body = File.getRelocTargetSym(Rel);
    uint32_t Type = Rel.getType(Config->Mips64EL);
    RelExpr Expr = Target->getRelExpr(Type, Body);
    if (!isPreemptible(Body, Type) && needsPlt(Expr))
      Expr = fromPlt(Expr);
    Expr = Target->getThunkExpr(Expr, Type, File, Body);
    // Some targets might require creation of thunks for relocations.
    // Now we support only MIPS which requires LA25 thunk to call PIC
    // code from non-PIC one, and ARM which requires interworking.
    if (Expr == R_THUNK_ABS || Expr == R_THUNK_PC || Expr == R_THUNK_PLT_PC) {
      auto *Sec = cast<InputSection<ELFT>>(&C);
      addThunk<ELFT>(Type, Body, *Sec);
    }
  }
}

template <class ELFT>
void createThunks(InputSectionBase<ELFT> &S,
                  const typename ELFT::Shdr &RelSec) {
  ELFFile<ELFT> &EObj = S.getFile()->getObj();
  if (RelSec.sh_type == SHT_RELA)
    createThunks(S, EObj.relas(&RelSec));
  else
    createThunks(S, EObj.rels(&RelSec));
}

template void scanRelocations<ELF32LE>(InputSectionBase<ELF32LE> &,
                                       const ELF32LE::Shdr &);
template void scanRelocations<ELF32BE>(InputSectionBase<ELF32BE> &,
                                       const ELF32BE::Shdr &);
template void scanRelocations<ELF64LE>(InputSectionBase<ELF64LE> &,
                                       const ELF64LE::Shdr &);
template void scanRelocations<ELF64BE>(InputSectionBase<ELF64BE> &,
                                       const ELF64BE::Shdr &);

template void createThunks<ELF32LE>(InputSectionBase<ELF32LE> &,
                                    const ELF32LE::Shdr &);
template void createThunks<ELF32BE>(InputSectionBase<ELF32BE> &,
                                    const ELF32BE::Shdr &);
template void createThunks<ELF64LE>(InputSectionBase<ELF64LE> &,
                                    const ELF64LE::Shdr &);
template void createThunks<ELF64BE>(InputSectionBase<ELF64BE> &,
                                    const ELF64BE::Shdr &);
}
}
