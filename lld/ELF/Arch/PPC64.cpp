//===- PPC64.cpp ----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

static uint64_t PPC64TocOffset = 0x8000;

uint64_t elf::getPPC64TocBase() {
  // The TOC consists of sections .got, .toc, .tocbss, .plt in that order. The
  // TOC starts where the first of these sections starts. We always create a
  // .got when we see a relocation that uses it, so for us the start is always
  // the .got.
  uint64_t TocVA = InX::Got->getVA();

  // Per the ppc64-elf-linux ABI, The TOC base is TOC value plus 0x8000
  // thus permitting a full 64 Kbytes segment. Note that the glibc startup
  // code (crt1.o) assumes that you can get from the TOC base to the
  // start of the .toc section with only a single (signed) 16-bit relocation.
  return TocVA + PPC64TocOffset;
}

namespace {
class PPC64 final : public TargetInfo {
public:
  PPC64();
  uint32_t calcEFlags() const override;
  RelExpr getRelExpr(RelType Type, const Symbol &S,
                     const uint8_t *Loc) const override;
  void writePlt(uint8_t *Buf, uint64_t GotPltEntryAddr, uint64_t PltEntryAddr,
                int32_t Index, unsigned RelOff) const override;
  void relocateOne(uint8_t *Loc, RelType Type, uint64_t Val) const override;
  void writeGotHeader(uint8_t *Buf) const override;
};
} // namespace

// Relocation masks following the #lo(value), #hi(value), #ha(value),
// #higher(value), #highera(value), #highest(value), and #highesta(value)
// macros defined in section 4.5.1. Relocation Types of the PPC-elf64abi
// document.
static uint16_t applyPPCLo(uint64_t V) { return V; }
static uint16_t applyPPCHi(uint64_t V) { return V >> 16; }
static uint16_t applyPPCHa(uint64_t V) { return (V + 0x8000) >> 16; }
static uint16_t applyPPCHigher(uint64_t V) { return V >> 32; }
static uint16_t applyPPCHighera(uint64_t V) { return (V + 0x8000) >> 32; }
static uint16_t applyPPCHighest(uint64_t V) { return V >> 48; }
static uint16_t applyPPCHighesta(uint64_t V) { return (V + 0x8000) >> 48; }

PPC64::PPC64() {
  GotRel = R_PPC64_GLOB_DAT;
  RelativeRel = R_PPC64_RELATIVE;
  GotEntrySize = 8;
  GotPltEntrySize = 8;
  PltEntrySize = 32;
  PltHeaderSize = 0;
  GotBaseSymInGotPlt = false;
  GotBaseSymOff = 0x8000;

  if (Config->EKind == ELF64LEKind) {
    GotHeaderEntriesNum = 1;
    GotPltHeaderEntriesNum = 2;
    PltRel = R_PPC64_JMP_SLOT;
  } else {
    PltRel = R_PPC64_GLOB_DAT;
  }

  // We need 64K pages (at least under glibc/Linux, the loader won't
  // set different permissions on a finer granularity than that).
  DefaultMaxPageSize = 65536;

  // The PPC64 ELF ABI v1 spec, says:
  //
  //   It is normally desirable to put segments with different characteristics
  //   in separate 256 Mbyte portions of the address space, to give the
  //   operating system full paging flexibility in the 64-bit address space.
  //
  // And because the lowest non-zero 256M boundary is 0x10000000, PPC64 linkers
  // use 0x10000000 as the starting address.
  DefaultImageBase = 0x10000000;

  TrapInstr =
      (Config->IsLE == sys::IsLittleEndianHost) ? 0x7fe00008 : 0x0800e07f;
}

static uint32_t getEFlags(InputFile *File) {
  // Get the e_flag from the input file and if it is unspecified, then set it to
  // the e_flag appropriate for the ABI.

  // We are currently handling both ELF64LE and ELF64BE but eventually will
  // remove BE support once v2 ABI support is complete.
  switch (Config->EKind) {
  case ELF64BEKind:
    if (uint32_t EFlags =
        cast<ObjFile<ELF64BE>>(File)->getObj().getHeader()->e_flags)
      return EFlags;
    return 1;
  case ELF64LEKind:
    if (uint32_t EFlags =
        cast<ObjFile<ELF64LE>>(File)->getObj().getHeader()->e_flags)
      return EFlags;
    return 2;
  default:
    llvm_unreachable("unknown Config->EKind");
  }
}

uint32_t PPC64::calcEFlags() const {
  assert(!ObjectFiles.empty());
  uint32_t Ret = getEFlags(ObjectFiles[0]);

  // Verify that all input files have the same e_flags.
  for (InputFile *F : makeArrayRef(ObjectFiles).slice(1)) {
    if (Ret == getEFlags(F))
      continue;
    error("incompatible e_flags: " + toString(F));
    return 0;
  }
  return Ret;
}

RelExpr PPC64::getRelExpr(RelType Type, const Symbol &S,
                          const uint8_t *Loc) const {
  switch (Type) {
  case R_PPC64_TOC16:
  case R_PPC64_TOC16_DS:
  case R_PPC64_TOC16_HA:
  case R_PPC64_TOC16_HI:
  case R_PPC64_TOC16_LO:
  case R_PPC64_TOC16_LO_DS:
    return R_GOTREL;
  case R_PPC64_TOC:
    return R_PPC_TOC;
  case R_PPC64_REL24:
    return R_PPC_PLT_OPD;
  case R_PPC64_REL16_LO:
  case R_PPC64_REL16_HA:
    return R_PC;
  default:
    return R_ABS;
  }
}

void PPC64::writeGotHeader(uint8_t *Buf) const {
  if (Config->EKind == ELF64LEKind)
    write64(Buf, getPPC64TocBase());
}

void PPC64::writePlt(uint8_t *Buf, uint64_t GotPltEntryAddr,
                     uint64_t PltEntryAddr, int32_t Index,
                     unsigned RelOff) const {
  uint64_t Off = GotPltEntryAddr - getPPC64TocBase();

  if (Config->EKind == ELF64LEKind) {
    // The most-common form of the plt stub. This assumes that the toc-pointer
    // register is properly initalized, and that the stub must save the toc
    // pointer value to the stack-save slot reserved for it (sp + 24).
    // There are 2 other variants but we don't have to emit those until we add
    // support for R_PPC64_REL24_NOTOC and R_PPC64_TOCSAVE relocations.
    // We are missing a super simple optimization, where if the upper 16 bits of
    // the offset are zero, then we can omit the addis instruction, and load
    // r2 + lo-offset directly into r12. I decided to leave this out in the
    // spirit of keeping it simple until we can link actual non-trivial
    // programs.
    write32(Buf +  0, 0xf8410018);                    // std     r2,24(r1)
    write32(Buf +  4, 0x3d820000 | applyPPCHa(Off));  // addis   r12,r2, X@plt@to@ha
    write32(Buf +  8, 0xe98c0000 | applyPPCLo(Off));  // ld      r12,X@plt@toc@l(r12)
    write32(Buf + 12, 0x7d8903a6);                    // mtctr    r12
    write32(Buf + 16, 0x4e800420);                    // bctr
  } else {
    // FIXME: What we should do, in theory, is get the offset of the function
    // descriptor in the .opd section, and use that as the offset from %r2 (the
    // TOC-base pointer). Instead, we have the GOT-entry offset, and that will
    // be a pointer to the function descriptor in the .opd section. Using
    // this scheme is simpler, but requires an extra indirection per PLT dispatch.
    write32(Buf, 0xf8410028);                       // std %r2, 40(%r1)
    write32(Buf + 4, 0x3d620000 | applyPPCHa(Off)); // addis %r11, %r2, X@ha
    write32(Buf + 8, 0xe98b0000 | applyPPCLo(Off)); // ld %r12, X@l(%r11)
    write32(Buf + 12, 0xe96c0000);                  // ld %r11,0(%r12)
    write32(Buf + 16, 0x7d6903a6);                  // mtctr %r11
    write32(Buf + 20, 0xe84c0008);                  // ld %r2,8(%r12)
    write32(Buf + 24, 0xe96c0010);                  // ld %r11,16(%r12)
    write32(Buf + 28, 0x4e800420);                  // bctr
  }
}

static std::pair<RelType, uint64_t> toAddr16Rel(RelType Type, uint64_t Val) {
  uint64_t V = Val - PPC64TocOffset;
  switch (Type) {
  case R_PPC64_TOC16:
    return {R_PPC64_ADDR16, V};
  case R_PPC64_TOC16_DS:
    return {R_PPC64_ADDR16_DS, V};
  case R_PPC64_TOC16_HA:
    return {R_PPC64_ADDR16_HA, V};
  case R_PPC64_TOC16_HI:
    return {R_PPC64_ADDR16_HI, V};
  case R_PPC64_TOC16_LO:
    return {R_PPC64_ADDR16_LO, V};
  case R_PPC64_TOC16_LO_DS:
    return {R_PPC64_ADDR16_LO_DS, V};
  default:
    return {Type, Val};
  }
}

void PPC64::relocateOne(uint8_t *Loc, RelType Type, uint64_t Val) const {
  // For a TOC-relative relocation, proceed in terms of the corresponding
  // ADDR16 relocation type.
  std::tie(Type, Val) = toAddr16Rel(Type, Val);

  switch (Type) {
  case R_PPC64_ADDR14: {
    checkAlignment(Loc, Val, 4, Type);
    // Preserve the AA/LK bits in the branch instruction
    uint8_t AALK = Loc[3];
    write16(Loc + 2, (AALK & 3) | (Val & 0xfffc));
    break;
  }
  case R_PPC64_ADDR16:
    checkInt(Loc, Val, 16, Type);
    write16(Loc, Val);
    break;
  case R_PPC64_ADDR16_DS:
    checkInt(Loc, Val, 16, Type);
    write16(Loc, (read16(Loc) & 3) | (Val & ~3));
    break;
  case R_PPC64_ADDR16_HA:
  case R_PPC64_REL16_HA:
    write16(Loc, applyPPCHa(Val));
    break;
  case R_PPC64_ADDR16_HI:
  case R_PPC64_REL16_HI:
    write16(Loc, applyPPCHi(Val));
    break;
  case R_PPC64_ADDR16_HIGHER:
    write16(Loc, applyPPCHigher(Val));
    break;
  case R_PPC64_ADDR16_HIGHERA:
    write16(Loc, applyPPCHighera(Val));
    break;
  case R_PPC64_ADDR16_HIGHEST:
    write16(Loc, applyPPCHighest(Val));
    break;
  case R_PPC64_ADDR16_HIGHESTA:
    write16(Loc, applyPPCHighesta(Val));
    break;
  case R_PPC64_ADDR16_LO:
  case R_PPC64_REL16_LO:
    write16(Loc, applyPPCLo(Val));
    break;
  case R_PPC64_ADDR16_LO_DS:
    write16(Loc, (read16(Loc) & 3) | (applyPPCLo(Val) & ~3));
    break;
  case R_PPC64_ADDR32:
  case R_PPC64_REL32:
    checkInt(Loc, Val, 32, Type);
    write32(Loc, Val);
    break;
  case R_PPC64_ADDR64:
  case R_PPC64_REL64:
  case R_PPC64_TOC:
    write64(Loc, Val);
    break;
  case R_PPC64_REL24: {
    uint32_t Mask = 0x03FFFFFC;
    checkInt(Loc, Val, 24, Type);
    write32(Loc, (read32(Loc) & ~Mask) | (Val & Mask));
    break;
  }
  default:
    error(getErrorLocation(Loc) + "unrecognized reloc " + Twine(Type));
  }
}

TargetInfo *elf::getPPC64TargetInfo() {
  static PPC64 Target;
  return &Target;
}
