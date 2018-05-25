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
  void writePltHeader(uint8_t *Buf) const override;
  void writePlt(uint8_t *Buf, uint64_t GotPltEntryAddr, uint64_t PltEntryAddr,
                int32_t Index, unsigned RelOff) const override;
  void relocateOne(uint8_t *Loc, RelType Type, uint64_t Val) const override;
  void writeGotHeader(uint8_t *Buf) const override;
  bool needsThunk(RelExpr Expr, RelType Type, const InputFile *File,
                  uint64_t BranchAddr, const Symbol &S) const override;
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
  PltRel = R_PPC64_JMP_SLOT;
  RelativeRel = R_PPC64_RELATIVE;
  IRelativeRel = R_PPC64_IRELATIVE;
  GotEntrySize = 8;
  PltEntrySize = 4;
  GotPltEntrySize = 8;
  GotBaseSymInGotPlt = false;
  GotBaseSymOff = 0x8000;
  GotHeaderEntriesNum = 1;
  GotPltHeaderEntriesNum = 2;
  PltHeaderSize = 60;
  NeedsThunks = true;

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
  // Get the e_flag from the input file and issue an error if incompatible
  // e_flag encountered.
  uint32_t EFlags;
  switch (Config->EKind) {
  case ELF64BEKind:
    EFlags = cast<ObjFile<ELF64BE>>(File)->getObj().getHeader()->e_flags;
    break;
  case ELF64LEKind:
    EFlags = cast<ObjFile<ELF64LE>>(File)->getObj().getHeader()->e_flags;
    break;
  default:
    llvm_unreachable("unknown Config->EKind");
  }
  if (EFlags > 2) {
    error("incompatible e_flags: " +  toString(File));
    return 0;
  }
  return EFlags;
}

uint32_t PPC64::calcEFlags() const {
  assert(!ObjectFiles.empty());

  uint32_t NonZeroFlag;
  for (InputFile *F : makeArrayRef(ObjectFiles)) {
    NonZeroFlag = getEFlags(F);
    if (NonZeroFlag)
      break;
  }

  // Verify that all input files have either the same e_flags, or zero.
  for (InputFile *F : makeArrayRef(ObjectFiles)) {
    uint32_t Flag = getEFlags(F);
    if (Flag == 0 || Flag == NonZeroFlag)
      continue;
    error(toString(F) + ": ABI version " + Twine(Flag) +
          " is not compatible with ABI version " + Twine(NonZeroFlag) +
          " output");
    return 0;
  }

  if (NonZeroFlag == 1) {
    error("PPC64 V1 ABI not supported");
    return 0;
  }

  return 2;
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
    return R_PPC_CALL_PLT;
  case R_PPC64_REL16_LO:
  case R_PPC64_REL16_HA:
  case R_PPC64_REL32:
  case R_PPC64_REL64:
    return R_PC;
  default:
    return R_ABS;
  }
}

void PPC64::writeGotHeader(uint8_t *Buf) const {
  write64(Buf, getPPC64TocBase());
}

void PPC64::writePltHeader(uint8_t *Buf) const {
  // The generic resolver stub goes first.
  write32(Buf +  0, 0x7c0802a6); // mflr r0
  write32(Buf +  4, 0x429f0005); // bcl  20,4*cr7+so,8 <_glink+0x8>
  write32(Buf +  8, 0x7d6802a6); // mflr r11
  write32(Buf + 12, 0x7c0803a6); // mtlr r0
  write32(Buf + 16, 0x7d8b6050); // subf r12, r11, r12
  write32(Buf + 20, 0x380cffcc); // subi r0,r12,52
  write32(Buf + 24, 0x7800f082); // srdi r0,r0,62,2
  write32(Buf + 28, 0xe98b002c); // ld   r12,44(r11)
  write32(Buf + 32, 0x7d6c5a14); // add  r11,r12,r11
  write32(Buf + 36, 0xe98b0000); // ld   r12,0(r11)
  write32(Buf + 40, 0xe96b0008); // ld   r11,8(r11)
  write32(Buf + 44, 0x7d8903a6); // mtctr   r12
  write32(Buf + 48, 0x4e800420); // bctr

  // The 'bcl' instruction will set the link register to the address of the
  // following instruction ('mflr r11'). Here we store the offset from that
  // instruction  to the first entry in the GotPlt section.
  int64_t GotPltOffset = InX::GotPlt->getVA() - (InX::Plt->getVA() + 8);
  write64(Buf + 52, GotPltOffset);
}

void PPC64::writePlt(uint8_t *Buf, uint64_t GotPltEntryAddr,
                     uint64_t PltEntryAddr, int32_t Index,
                     unsigned RelOff) const {
 int32_t Offset = PltHeaderSize + Index * PltEntrySize;
 // bl __glink_PLTresolve
 write32(Buf, 0x48000000 | ((-Offset) & 0x03FFFFFc));
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

bool PPC64::needsThunk(RelExpr Expr, RelType Type, const InputFile *File,
                       uint64_t BranchAddr, const Symbol &S) const {
  // If a function is in the plt it needs to be called through
  // a call stub.
  return Type == R_PPC64_REL24 && S.isInPlt();
}

TargetInfo *elf::getPPC64TargetInfo() {
  static PPC64 Target;
  return &Target;
}
