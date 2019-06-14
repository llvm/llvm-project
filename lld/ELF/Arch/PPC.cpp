//===- PPC.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OutputSections.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {
class PPC final : public TargetInfo {
public:
  PPC();
  void writeGotHeader(uint8_t *Buf) const override;
  void writePltHeader(uint8_t *Buf) const override {
    llvm_unreachable("should call writePPC32GlinkSection() instead");
  }
  void writePlt(uint8_t *Buf, uint64_t GotPltEntryAddr, uint64_t PltEntryAddr,
    int32_t Index, unsigned RelOff) const override {
    llvm_unreachable("should call writePPC32GlinkSection() instead");
  }
  void writeGotPlt(uint8_t *Buf, const Symbol &S) const override;
  bool needsThunk(RelExpr Expr, RelType RelocType, const InputFile *File,
                  uint64_t BranchAddr, const Symbol &S) const override;
  uint32_t getThunkSectionSpacing() const override;
  bool inBranchRange(RelType Type, uint64_t Src, uint64_t Dst) const override;
  RelExpr getRelExpr(RelType Type, const Symbol &S,
                     const uint8_t *Loc) const override;
  void relocateOne(uint8_t *Loc, RelType Type, uint64_t Val) const override;
  RelExpr adjustRelaxExpr(RelType Type, const uint8_t *Data,
                          RelExpr Expr) const override;
  int getTlsGdRelaxSkip(RelType Type) const override;
  void relaxTlsGdToIe(uint8_t *Loc, RelType Type, uint64_t Val) const override;
  void relaxTlsGdToLe(uint8_t *Loc, RelType Type, uint64_t Val) const override;
  void relaxTlsLdToLe(uint8_t *Loc, RelType Type, uint64_t Val) const override;
  void relaxTlsIeToLe(uint8_t *Loc, RelType Type, uint64_t Val) const override;
};
} // namespace

static uint16_t lo(uint32_t V) { return V; }
static uint16_t ha(uint32_t V) { return (V + 0x8000) >> 16; }

static uint32_t readFromHalf16(const uint8_t *Loc) {
  return read32(Config->IsLE ? Loc : Loc - 2);
}

static void writeFromHalf16(uint8_t *Loc, uint32_t Insn) {
  write32(Config->IsLE ? Loc : Loc - 2, Insn);
}

void elf::writePPC32GlinkSection(uint8_t *Buf, size_t NumEntries) {
  // On PPC Secure PLT ABI, bl foo@plt jumps to a call stub, which loads an
  // absolute address from a specific .plt slot (usually called .got.plt on
  // other targets) and jumps there.
  //
  // a) With immediate binding (BIND_NOW), the .plt entry is resolved at load
  // time. The .glink section is not used.
  // b) With lazy binding, the .plt entry points to a `b PLTresolve`
  // instruction in .glink, filled in by PPC::writeGotPlt().

  // Write N `b PLTresolve` first.
  for (size_t I = 0; I != NumEntries; ++I)
    write32(Buf + 4 * I, 0x48000000 | 4 * (NumEntries - I));
  Buf += 4 * NumEntries;

  // Then write PLTresolve(), which has two forms: PIC and non-PIC. PLTresolve()
  // computes the PLT index (by computing the distance from the landing b to
  // itself) and calls _dl_runtime_resolve() (in glibc).
  uint32_t GOT = In.Got->getVA();
  uint32_t Glink = In.Plt->getVA(); // VA of .glink
  const uint8_t *End = Buf + 64;
  if (Config->Pic) {
    uint32_t AfterBcl = In.Plt->getSize() - Target->PltHeaderSize + 12;
    uint32_t GotBcl = GOT + 4 - (Glink + AfterBcl);
    write32(Buf + 0, 0x3d6b0000 | ha(AfterBcl));  // addis r11,r11,1f-glink@ha
    write32(Buf + 4, 0x7c0802a6);                 // mflr r0
    write32(Buf + 8, 0x429f0005);                 // bcl 20,30,.+4
    write32(Buf + 12, 0x396b0000 | lo(AfterBcl)); // 1: addi r11,r11,1b-.glink@l
    write32(Buf + 16, 0x7d8802a6);                // mflr r12
    write32(Buf + 20, 0x7c0803a6);                // mtlr r0
    write32(Buf + 24, 0x7d6c5850);                // sub r11,r11,r12
    write32(Buf + 28, 0x3d8c0000 | ha(GotBcl));   // addis 12,12,GOT+4-1b@ha
    if (ha(GotBcl) == ha(GotBcl + 4)) {
      write32(Buf + 32, 0x800c0000 | lo(GotBcl)); // lwz r0,r12,GOT+4-1b@l(r12)
      write32(Buf + 36,
              0x818c0000 | lo(GotBcl + 4));       // lwz r12,r12,GOT+8-1b@l(r12)
    } else {
      write32(Buf + 32, 0x840c0000 | lo(GotBcl)); // lwzu r0,r12,GOT+4-1b@l(r12)
      write32(Buf + 36, 0x818c0000 | 4);          // lwz r12,r12,4(r12)
    }
    write32(Buf + 40, 0x7c0903a6);                // mtctr 0
    write32(Buf + 44, 0x7c0b5a14);                // add r0,11,11
    write32(Buf + 48, 0x7d605a14);                // add r11,0,11
    write32(Buf + 52, 0x4e800420);                // bctr
    Buf += 56;
  } else {
    write32(Buf + 0, 0x3d800000 | ha(GOT + 4));   // lis     r12,GOT+4@ha
    write32(Buf + 4, 0x3d6b0000 | ha(-Glink));    // addis   r11,r11,-Glink@ha
    if (ha(GOT + 4) == ha(GOT + 8))
      write32(Buf + 8, 0x800c0000 | lo(GOT + 4)); // lwz r0,GOT+4@l(r12)
    else
      write32(Buf + 8, 0x840c0000 | lo(GOT + 4)); // lwzu r0,GOT+4@l(r12)
    write32(Buf + 12, 0x396b0000 | lo(-Glink));   // addi    r11,r11,-Glink@l
    write32(Buf + 16, 0x7c0903a6);                // mtctr   r0
    write32(Buf + 20, 0x7c0b5a14);                // add     r0,r11,r11
    if (ha(GOT + 4) == ha(GOT + 8))
      write32(Buf + 24, 0x818c0000 | lo(GOT + 8)); // lwz r12,GOT+8@ha(r12)
    else
      write32(Buf + 24, 0x818c0000 | 4);          // lwz r12,4(r12)
    write32(Buf + 28, 0x7d605a14);                // add     r11,r0,r11
    write32(Buf + 32, 0x4e800420);                // bctr
    Buf += 36;
  }

  // Pad with nop. They should not be executed.
  for (; Buf < End; Buf += 4)
    write32(Buf, 0x60000000);
}

PPC::PPC() {
  GotRel = R_PPC_GLOB_DAT;
  NoneRel = R_PPC_NONE;
  PltRel = R_PPC_JMP_SLOT;
  RelativeRel = R_PPC_RELATIVE;
  IRelativeRel = R_PPC_IRELATIVE;
  SymbolicRel = R_PPC_ADDR32;
  GotBaseSymInGotPlt = false;
  GotHeaderEntriesNum = 3;
  GotPltHeaderEntriesNum = 0;
  PltHeaderSize = 64; // size of PLTresolve in .glink
  PltEntrySize = 4;

  NeedsThunks = true;

  TlsModuleIndexRel = R_PPC_DTPMOD32;
  TlsOffsetRel = R_PPC_DTPREL32;
  TlsGotRel = R_PPC_TPREL32;

  DefaultMaxPageSize = 65536;
  DefaultImageBase = 0x10000000;

  write32(TrapInstr.data(), 0x7fe00008);
}

void PPC::writeGotHeader(uint8_t *Buf) const {
  // _GLOBAL_OFFSET_TABLE_[0] = _DYNAMIC
  // glibc stores _dl_runtime_resolve in _GLOBAL_OFFSET_TABLE_[1],
  // link_map in _GLOBAL_OFFSET_TABLE_[2].
  write32(Buf, Main->Dynamic->getVA());
}

void PPC::writeGotPlt(uint8_t *Buf, const Symbol &S) const {
  // Address of the symbol resolver stub in .glink .
  write32(Buf, In.Plt->getVA() + 4 * S.PltIndex);
}

bool PPC::needsThunk(RelExpr Expr, RelType Type, const InputFile *File,
                     uint64_t BranchAddr, const Symbol &S) const {
  if (Type != R_PPC_REL24 && Type != R_PPC_PLTREL24)
    return false;
  if (S.isInPlt())
    return true;
  if (S.isUndefWeak())
    return false;
  return !(Expr == R_PC && PPC::inBranchRange(Type, BranchAddr, S.getVA()));
}

uint32_t PPC::getThunkSectionSpacing() const { return 0x2000000; }

bool PPC::inBranchRange(RelType Type, uint64_t Src, uint64_t Dst) const {
  uint64_t Offset = Dst - Src;
  if (Type == R_PPC_REL24 || Type == R_PPC_PLTREL24)
    return isInt<26>(Offset);
  llvm_unreachable("unsupported relocation type used in branch");
}

RelExpr PPC::getRelExpr(RelType Type, const Symbol &S,
                        const uint8_t *Loc) const {
  switch (Type) {
  case R_PPC_DTPREL16:
  case R_PPC_DTPREL16_HA:
  case R_PPC_DTPREL16_HI:
  case R_PPC_DTPREL16_LO:
  case R_PPC_DTPREL32:
    return R_DTPREL;
  case R_PPC_REL14:
  case R_PPC_REL32:
  case R_PPC_LOCAL24PC:
  case R_PPC_REL16_LO:
  case R_PPC_REL16_HI:
  case R_PPC_REL16_HA:
    return R_PC;
  case R_PPC_GOT16:
    return R_GOT_OFF;
  case R_PPC_REL24:
    return R_PLT_PC;
  case R_PPC_PLTREL24:
    return R_PPC32_PLTREL;
  case R_PPC_GOT_TLSGD16:
    return R_TLSGD_GOT;
  case R_PPC_GOT_TLSLD16:
    return R_TLSLD_GOT;
  case R_PPC_GOT_TPREL16:
    return R_GOT_OFF;
  case R_PPC_TLS:
    return R_TLSIE_HINT;
  case R_PPC_TLSGD:
    return R_TLSDESC_CALL;
  case R_PPC_TLSLD:
    return R_TLSLD_HINT;
  case R_PPC_TPREL16:
  case R_PPC_TPREL16_HA:
  case R_PPC_TPREL16_LO:
  case R_PPC_TPREL16_HI:
    return R_TLS;
  default:
    return R_ABS;
  }
}

static std::pair<RelType, uint64_t> fromDTPREL(RelType Type, uint64_t Val) {
  uint64_t DTPBiasedVal = Val - 0x8000;
  switch (Type) {
  case R_PPC_DTPREL16:
    return {R_PPC64_ADDR16, DTPBiasedVal};
  case R_PPC_DTPREL16_HA:
    return {R_PPC_ADDR16_HA, DTPBiasedVal};
  case R_PPC_DTPREL16_HI:
    return {R_PPC_ADDR16_HI, DTPBiasedVal};
  case R_PPC_DTPREL16_LO:
    return {R_PPC_ADDR16_LO, DTPBiasedVal};
  case R_PPC_DTPREL32:
    return {R_PPC_ADDR32, DTPBiasedVal};
  default:
    return {Type, Val};
  }
}

void PPC::relocateOne(uint8_t *Loc, RelType Type, uint64_t Val) const {
  RelType NewType;
  std::tie(NewType, Val) = fromDTPREL(Type, Val);
  switch (NewType) {
  case R_PPC_ADDR16:
  case R_PPC_GOT16:
  case R_PPC_GOT_TLSGD16:
  case R_PPC_GOT_TLSLD16:
  case R_PPC_GOT_TPREL16:
  case R_PPC_TPREL16:
    checkInt(Loc, Val, 16, Type);
    write16(Loc, Val);
    break;
  case R_PPC_ADDR16_HA:
  case R_PPC_DTPREL16_HA:
  case R_PPC_GOT_TLSGD16_HA:
  case R_PPC_GOT_TLSLD16_HA:
  case R_PPC_GOT_TPREL16_HA:
  case R_PPC_REL16_HA:
  case R_PPC_TPREL16_HA:
    write16(Loc, ha(Val));
    break;
  case R_PPC_ADDR16_HI:
  case R_PPC_DTPREL16_HI:
  case R_PPC_GOT_TLSGD16_HI:
  case R_PPC_GOT_TLSLD16_HI:
  case R_PPC_GOT_TPREL16_HI:
  case R_PPC_REL16_HI:
  case R_PPC_TPREL16_HI:
    write16(Loc, Val >> 16);
    break;
  case R_PPC_ADDR16_LO:
  case R_PPC_DTPREL16_LO:
  case R_PPC_GOT_TLSGD16_LO:
  case R_PPC_GOT_TLSLD16_LO:
  case R_PPC_GOT_TPREL16_LO:
  case R_PPC_REL16_LO:
  case R_PPC_TPREL16_LO:
    write16(Loc, Val);
    break;
  case R_PPC_ADDR32:
  case R_PPC_REL32:
    write32(Loc, Val);
    break;
  case R_PPC_REL14: {
    uint32_t Mask = 0x0000FFFC;
    checkInt(Loc, Val, 16, Type);
    checkAlignment(Loc, Val, 4, Type);
    write32(Loc, (read32(Loc) & ~Mask) | (Val & Mask));
    break;
  }
  case R_PPC_REL24:
  case R_PPC_LOCAL24PC:
  case R_PPC_PLTREL24: {
    uint32_t Mask = 0x03FFFFFC;
    checkInt(Loc, Val, 26, Type);
    checkAlignment(Loc, Val, 4, Type);
    write32(Loc, (read32(Loc) & ~Mask) | (Val & Mask));
    break;
  }
  default:
    error(getErrorLocation(Loc) + "unrecognized relocation " + toString(Type));
  }
}

RelExpr PPC::adjustRelaxExpr(RelType Type, const uint8_t *Data,
                             RelExpr Expr) const {
  if (Expr == R_RELAX_TLS_GD_TO_IE)
    return R_RELAX_TLS_GD_TO_IE_GOT_OFF;
  if (Expr == R_RELAX_TLS_LD_TO_LE)
    return R_RELAX_TLS_LD_TO_LE_ABS;
  return Expr;
}

int PPC::getTlsGdRelaxSkip(RelType Type) const {
  // A __tls_get_addr call instruction is marked with 2 relocations:
  //
  //   R_PPC_TLSGD / R_PPC_TLSLD: marker relocation
  //   R_PPC_REL24: __tls_get_addr
  //
  // After the relaxation we no longer call __tls_get_addr and should skip both
  // relocations to not create a false dependence on __tls_get_addr being
  // defined.
  if (Type == R_PPC_TLSGD || Type == R_PPC_TLSLD)
    return 2;
  return 1;
}

void PPC::relaxTlsGdToIe(uint8_t *Loc, RelType Type, uint64_t Val) const {
  switch (Type) {
  case R_PPC_GOT_TLSGD16: {
    // addi rT, rA, x@got@tlsgd --> lwz rT, x@got@tprel(rA)
    uint32_t Insn = readFromHalf16(Loc);
    writeFromHalf16(Loc, 0x80000000 | (Insn & 0x03ff0000));
    relocateOne(Loc, R_PPC_GOT_TPREL16, Val);
    break;
  }
  case R_PPC_TLSGD:
    // bl __tls_get_addr(x@tldgd) --> add r3, r3, r2
    write32(Loc, 0x7c631214);
    break;
  default:
    llvm_unreachable("unsupported relocation for TLS GD to IE relaxation");
  }
}

void PPC::relaxTlsGdToLe(uint8_t *Loc, RelType Type, uint64_t Val) const {
  switch (Type) {
  case R_PPC_GOT_TLSGD16:
    // addi r3, r31, x@got@tlsgd --> addis r3, r2, x@tprel@ha
    writeFromHalf16(Loc, 0x3c620000 | ha(Val));
    break;
  case R_PPC_TLSGD:
    // bl __tls_get_addr(x@tldgd) --> add r3, r3, x@tprel@l
    write32(Loc, 0x38630000 | lo(Val));
    break;
  default:
    llvm_unreachable("unsupported relocation for TLS GD to LE relaxation");
  }
}

void PPC::relaxTlsLdToLe(uint8_t *Loc, RelType Type, uint64_t Val) const {
  switch (Type) {
  case R_PPC_GOT_TLSLD16:
    // addi r3, rA, x@got@tlsgd --> addis r3, r2, 0
    writeFromHalf16(Loc, 0x3c620000);
    break;
  case R_PPC_TLSLD:
    // r3+x@dtprel computes r3+x-0x8000, while we want it to compute r3+x@tprel
    // = r3+x-0x7000, so add 4096 to r3.
    // bl __tls_get_addr(x@tlsld) --> addi r3, r3, 4096
    write32(Loc, 0x38631000);
    break;
  case R_PPC_DTPREL16:
  case R_PPC_DTPREL16_HA:
  case R_PPC_DTPREL16_HI:
  case R_PPC_DTPREL16_LO:
    relocateOne(Loc, Type, Val);
    break;
  default:
    llvm_unreachable("unsupported relocation for TLS LD to LE relaxation");
  }
}

void PPC::relaxTlsIeToLe(uint8_t *Loc, RelType Type, uint64_t Val) const {
  switch (Type) {
  case R_PPC_GOT_TPREL16: {
    // lwz rT, x@got@tprel(rA) --> addis rT, r2, x@tprel@ha
    uint32_t RT = readFromHalf16(Loc) & 0x03e00000;
    writeFromHalf16(Loc, 0x3c020000 | RT | ha(Val));
    break;
  }
  case R_PPC_TLS: {
    uint32_t Insn = read32(Loc);
    if (Insn >> 26 != 31)
      error("unrecognized instruction for IE to LE R_PPC_TLS");
    // addi rT, rT, x@tls --> addi rT, rT, x@tprel@l
    uint32_t DFormOp = getPPCDFormOp((read32(Loc) & 0x000007fe) >> 1);
    if (DFormOp == 0)
      error("unrecognized instruction for IE to LE R_PPC_TLS");
    write32(Loc, (DFormOp << 26) | (Insn & 0x03ff0000) | lo(Val));
    break;
  }
  default:
    llvm_unreachable("unsupported relocation for TLS IE to LE relaxation");
  }
}

TargetInfo *elf::getPPCTargetInfo() {
  static PPC Target;
  return &Target;
}
