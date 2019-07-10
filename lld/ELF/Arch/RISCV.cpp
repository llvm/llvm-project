//===- RISCV.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "SyntheticSections.h"
#include "Target.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {

class RISCV final : public TargetInfo {
public:
  RISCV();
  uint32_t calcEFlags() const override;
  void writeGotHeader(uint8_t *Buf) const override;
  void writeGotPlt(uint8_t *Buf, const Symbol &S) const override;
  void writePltHeader(uint8_t *Buf) const override;
  void writePlt(uint8_t *Buf, uint64_t GotPltEntryAddr, uint64_t PltEntryAddr,
                int32_t Index, unsigned RelOff) const override;
  RelType getDynRel(RelType Type) const override;
  RelExpr getRelExpr(RelType Type, const Symbol &S,
                     const uint8_t *Loc) const override;
  void relocateOne(uint8_t *Loc, RelType Type, uint64_t Val) const override;
};

} // end anonymous namespace

const uint64_t DTPOffset = 0x800;

enum Op {
  ADDI = 0x13,
  AUIPC = 0x17,
  JALR = 0x67,
  LD = 0x3003,
  LW = 0x2003,
  SRLI = 0x5013,
  SUB = 0x40000033,
};

enum Reg {
  X_RA = 1,
  X_T0 = 5,
  X_T1 = 6,
  X_T2 = 7,
  X_T3 = 28,
};

static uint32_t hi20(uint32_t Val) { return (Val + 0x800) >> 12; }
static uint32_t lo12(uint32_t Val) { return Val & 4095; }

static uint32_t itype(uint32_t Op, uint32_t Rd, uint32_t Rs1, uint32_t Imm) {
  return Op | (Rd << 7) | (Rs1 << 15) | (Imm << 20);
}
static uint32_t rtype(uint32_t Op, uint32_t Rd, uint32_t Rs1, uint32_t Rs2) {
  return Op | (Rd << 7) | (Rs1 << 15) | (Rs2 << 20);
}
static uint32_t utype(uint32_t Op, uint32_t Rd, uint32_t Imm) {
  return Op | (Rd << 7) | (Imm << 12);
}

RISCV::RISCV() {
  CopyRel = R_RISCV_COPY;
  NoneRel = R_RISCV_NONE;
  PltRel = R_RISCV_JUMP_SLOT;
  RelativeRel = R_RISCV_RELATIVE;
  if (Config->Is64) {
    SymbolicRel = R_RISCV_64;
    TlsModuleIndexRel = R_RISCV_TLS_DTPMOD64;
    TlsOffsetRel = R_RISCV_TLS_DTPREL64;
    TlsGotRel = R_RISCV_TLS_TPREL64;
  } else {
    SymbolicRel = R_RISCV_32;
    TlsModuleIndexRel = R_RISCV_TLS_DTPMOD32;
    TlsOffsetRel = R_RISCV_TLS_DTPREL32;
    TlsGotRel = R_RISCV_TLS_TPREL32;
  }
  GotRel = SymbolicRel;

  // .got[0] = _DYNAMIC
  GotBaseSymInGotPlt = false;
  GotHeaderEntriesNum = 1;

  // .got.plt[0] = _dl_runtime_resolve, .got.plt[1] = link_map
  GotPltHeaderEntriesNum = 2;

  PltEntrySize = 16;
  PltHeaderSize = 32;
}

static uint32_t getEFlags(InputFile *F) {
  if (Config->Is64)
    return cast<ObjFile<ELF64LE>>(F)->getObj().getHeader()->e_flags;
  return cast<ObjFile<ELF32LE>>(F)->getObj().getHeader()->e_flags;
}

uint32_t RISCV::calcEFlags() const {
  assert(!ObjectFiles.empty());

  uint32_t Target = getEFlags(ObjectFiles.front());

  for (InputFile *F : ObjectFiles) {
    uint32_t EFlags = getEFlags(F);
    if (EFlags & EF_RISCV_RVC)
      Target |= EF_RISCV_RVC;

    if ((EFlags & EF_RISCV_FLOAT_ABI) != (Target & EF_RISCV_FLOAT_ABI))
      error(toString(F) +
            ": cannot link object files with different floating-point ABI");

    if ((EFlags & EF_RISCV_RVE) != (Target & EF_RISCV_RVE))
      error(toString(F) +
            ": cannot link object files with different EF_RISCV_RVE");
  }

  return Target;
}

void RISCV::writeGotHeader(uint8_t *Buf) const {
  if (Config->Is64)
    write64le(Buf, Main->Dynamic->getVA());
  else
    write32le(Buf, Main->Dynamic->getVA());
}

void RISCV::writeGotPlt(uint8_t *Buf, const Symbol &S) const {
  if (Config->Is64)
    write64le(Buf, In.Plt->getVA());
  else
    write32le(Buf, In.Plt->getVA());
}

void RISCV::writePltHeader(uint8_t *Buf) const {
  // 1: auipc t2, %pcrel_hi(.got.plt)
  // sub t1, t1, t3
  // l[wd] t3, %pcrel_lo(1b)(t2); t3 = _dl_runtime_resolve
  // addi t1, t1, -PltHeaderSize-12; t1 = &.plt[i] - &.plt[0]
  // addi t0, t2, %pcrel_lo(1b)
  // srli t1, t1, (rv64?1:2); t1 = &.got.plt[i] - &.got.plt[0]
  // l[wd] t0, Wordsize(t0); t0 = link_map
  // jr t3
  uint32_t Offset = In.GotPlt->getVA() - In.Plt->getVA();
  uint32_t Load = Config->Is64 ? LD : LW;
  write32le(Buf + 0, utype(AUIPC, X_T2, hi20(Offset)));
  write32le(Buf + 4, rtype(SUB, X_T1, X_T1, X_T3));
  write32le(Buf + 8, itype(Load, X_T3, X_T2, lo12(Offset)));
  write32le(Buf + 12, itype(ADDI, X_T1, X_T1, -Target->PltHeaderSize - 12));
  write32le(Buf + 16, itype(ADDI, X_T0, X_T2, lo12(Offset)));
  write32le(Buf + 20, itype(SRLI, X_T1, X_T1, Config->Is64 ? 1 : 2));
  write32le(Buf + 24, itype(Load, X_T0, X_T0, Config->Wordsize));
  write32le(Buf + 28, itype(JALR, 0, X_T3, 0));
}

void RISCV::writePlt(uint8_t *Buf, uint64_t GotPltEntryAddr,
                     uint64_t PltEntryAddr, int32_t Index,
                     unsigned RelOff) const {
  // 1: auipc t3, %pcrel_hi(f@.got.plt)
  // l[wd] t3, %pcrel_lo(1b)(t3)
  // jalr t1, t3
  // nop
  uint32_t Offset = GotPltEntryAddr - PltEntryAddr;
  write32le(Buf + 0, utype(AUIPC, X_T3, hi20(Offset)));
  write32le(Buf + 4, itype(Config->Is64 ? LD : LW, X_T3, X_T3, lo12(Offset)));
  write32le(Buf + 8, itype(JALR, X_T1, X_T3, 0));
  write32le(Buf + 12, itype(ADDI, 0, 0, 0));
}

RelType RISCV::getDynRel(RelType Type) const {
  return Type == Target->SymbolicRel ? Type
                                     : static_cast<RelType>(R_RISCV_NONE);
}

RelExpr RISCV::getRelExpr(const RelType Type, const Symbol &S,
                          const uint8_t *Loc) const {
  switch (Type) {
  case R_RISCV_ADD8:
  case R_RISCV_ADD16:
  case R_RISCV_ADD32:
  case R_RISCV_ADD64:
  case R_RISCV_SET6:
  case R_RISCV_SET8:
  case R_RISCV_SET16:
  case R_RISCV_SET32:
  case R_RISCV_SUB6:
  case R_RISCV_SUB8:
  case R_RISCV_SUB16:
  case R_RISCV_SUB32:
  case R_RISCV_SUB64:
    return R_RISCV_ADD;
  case R_RISCV_JAL:
  case R_RISCV_BRANCH:
  case R_RISCV_PCREL_HI20:
  case R_RISCV_RVC_BRANCH:
  case R_RISCV_RVC_JUMP:
  case R_RISCV_32_PCREL:
    return R_PC;
  case R_RISCV_CALL:
  case R_RISCV_CALL_PLT:
    return R_PLT_PC;
  case R_RISCV_GOT_HI20:
    return R_GOT_PC;
  case R_RISCV_PCREL_LO12_I:
  case R_RISCV_PCREL_LO12_S:
    return R_RISCV_PC_INDIRECT;
  case R_RISCV_TLS_GD_HI20:
    return R_TLSGD_PC;
  case R_RISCV_TLS_GOT_HI20:
    Config->HasStaticTlsModel = true;
    return R_GOT_PC;
  case R_RISCV_TPREL_HI20:
  case R_RISCV_TPREL_LO12_I:
  case R_RISCV_TPREL_LO12_S:
    return R_TLS;
  case R_RISCV_RELAX:
  case R_RISCV_ALIGN:
  case R_RISCV_TPREL_ADD:
    return R_HINT;
  default:
    return R_ABS;
  }
}

// Extract bits V[Begin:End], where range is inclusive, and Begin must be < 63.
static uint32_t extractBits(uint64_t V, uint32_t Begin, uint32_t End) {
  return (V & ((1ULL << (Begin + 1)) - 1)) >> End;
}

void RISCV::relocateOne(uint8_t *Loc, const RelType Type,
                        const uint64_t Val) const {
  const unsigned Bits = Config->Wordsize * 8;

  switch (Type) {
  case R_RISCV_32:
    write32le(Loc, Val);
    return;
  case R_RISCV_64:
    write64le(Loc, Val);
    return;

  case R_RISCV_RVC_BRANCH: {
    checkInt(Loc, static_cast<int64_t>(Val) >> 1, 8, Type);
    checkAlignment(Loc, Val, 2, Type);
    uint16_t Insn = read16le(Loc) & 0xE383;
    uint16_t Imm8 = extractBits(Val, 8, 8) << 12;
    uint16_t Imm4_3 = extractBits(Val, 4, 3) << 10;
    uint16_t Imm7_6 = extractBits(Val, 7, 6) << 5;
    uint16_t Imm2_1 = extractBits(Val, 2, 1) << 3;
    uint16_t Imm5 = extractBits(Val, 5, 5) << 2;
    Insn |= Imm8 | Imm4_3 | Imm7_6 | Imm2_1 | Imm5;

    write16le(Loc, Insn);
    return;
  }

  case R_RISCV_RVC_JUMP: {
    checkInt(Loc, static_cast<int64_t>(Val) >> 1, 11, Type);
    checkAlignment(Loc, Val, 2, Type);
    uint16_t Insn = read16le(Loc) & 0xE003;
    uint16_t Imm11 = extractBits(Val, 11, 11) << 12;
    uint16_t Imm4 = extractBits(Val, 4, 4) << 11;
    uint16_t Imm9_8 = extractBits(Val, 9, 8) << 9;
    uint16_t Imm10 = extractBits(Val, 10, 10) << 8;
    uint16_t Imm6 = extractBits(Val, 6, 6) << 7;
    uint16_t Imm7 = extractBits(Val, 7, 7) << 6;
    uint16_t Imm3_1 = extractBits(Val, 3, 1) << 3;
    uint16_t Imm5 = extractBits(Val, 5, 5) << 2;
    Insn |= Imm11 | Imm4 | Imm9_8 | Imm10 | Imm6 | Imm7 | Imm3_1 | Imm5;

    write16le(Loc, Insn);
    return;
  }

  case R_RISCV_RVC_LUI: {
    int64_t Imm = SignExtend64(Val + 0x800, Bits) >> 12;
    checkInt(Loc, Imm, 6, Type);
    if (Imm == 0) { // `c.lui rd, 0` is illegal, convert to `c.li rd, 0`
      write16le(Loc, (read16le(Loc) & 0x0F83) | 0x4000);
    } else {
      uint16_t Imm17 = extractBits(Val + 0x800, 17, 17) << 12;
      uint16_t Imm16_12 = extractBits(Val + 0x800, 16, 12) << 2;
      write16le(Loc, (read16le(Loc) & 0xEF83) | Imm17 | Imm16_12);
    }
    return;
  }

  case R_RISCV_JAL: {
    checkInt(Loc, static_cast<int64_t>(Val) >> 1, 20, Type);
    checkAlignment(Loc, Val, 2, Type);

    uint32_t Insn = read32le(Loc) & 0xFFF;
    uint32_t Imm20 = extractBits(Val, 20, 20) << 31;
    uint32_t Imm10_1 = extractBits(Val, 10, 1) << 21;
    uint32_t Imm11 = extractBits(Val, 11, 11) << 20;
    uint32_t Imm19_12 = extractBits(Val, 19, 12) << 12;
    Insn |= Imm20 | Imm10_1 | Imm11 | Imm19_12;

    write32le(Loc, Insn);
    return;
  }

  case R_RISCV_BRANCH: {
    checkInt(Loc, static_cast<int64_t>(Val) >> 1, 12, Type);
    checkAlignment(Loc, Val, 2, Type);

    uint32_t Insn = read32le(Loc) & 0x1FFF07F;
    uint32_t Imm12 = extractBits(Val, 12, 12) << 31;
    uint32_t Imm10_5 = extractBits(Val, 10, 5) << 25;
    uint32_t Imm4_1 = extractBits(Val, 4, 1) << 8;
    uint32_t Imm11 = extractBits(Val, 11, 11) << 7;
    Insn |= Imm12 | Imm10_5 | Imm4_1 | Imm11;

    write32le(Loc, Insn);
    return;
  }

  // auipc + jalr pair
  case R_RISCV_CALL:
  case R_RISCV_CALL_PLT: {
    int64_t Hi = SignExtend64(Val + 0x800, Bits) >> 12;
    checkInt(Loc, Hi, 20, Type);
    if (isInt<20>(Hi)) {
      relocateOne(Loc, R_RISCV_PCREL_HI20, Val);
      relocateOne(Loc + 4, R_RISCV_PCREL_LO12_I, Val);
    }
    return;
  }

  case R_RISCV_GOT_HI20:
  case R_RISCV_PCREL_HI20:
  case R_RISCV_TLS_GD_HI20:
  case R_RISCV_TLS_GOT_HI20:
  case R_RISCV_TPREL_HI20:
  case R_RISCV_HI20: {
    uint64_t Hi = Val + 0x800;
    checkInt(Loc, SignExtend64(Hi, Bits) >> 12, 20, Type);
    write32le(Loc, (read32le(Loc) & 0xFFF) | (Hi & 0xFFFFF000));
    return;
  }

  case R_RISCV_PCREL_LO12_I:
  case R_RISCV_TPREL_LO12_I:
  case R_RISCV_LO12_I: {
    uint64_t Hi = (Val + 0x800) >> 12;
    uint64_t Lo = Val - (Hi << 12);
    write32le(Loc, (read32le(Loc) & 0xFFFFF) | ((Lo & 0xFFF) << 20));
    return;
  }

  case R_RISCV_PCREL_LO12_S:
  case R_RISCV_TPREL_LO12_S:
  case R_RISCV_LO12_S: {
    uint64_t Hi = (Val + 0x800) >> 12;
    uint64_t Lo = Val - (Hi << 12);
    uint32_t Imm11_5 = extractBits(Lo, 11, 5) << 25;
    uint32_t Imm4_0 = extractBits(Lo, 4, 0) << 7;
    write32le(Loc, (read32le(Loc) & 0x1FFF07F) | Imm11_5 | Imm4_0);
    return;
  }

  case R_RISCV_ADD8:
    *Loc += Val;
    return;
  case R_RISCV_ADD16:
    write16le(Loc, read16le(Loc) + Val);
    return;
  case R_RISCV_ADD32:
    write32le(Loc, read32le(Loc) + Val);
    return;
  case R_RISCV_ADD64:
    write64le(Loc, read64le(Loc) + Val);
    return;
  case R_RISCV_SUB6:
    *Loc = (*Loc & 0xc0) | (((*Loc & 0x3f) - Val) & 0x3f);
    return;
  case R_RISCV_SUB8:
    *Loc -= Val;
    return;
  case R_RISCV_SUB16:
    write16le(Loc, read16le(Loc) - Val);
    return;
  case R_RISCV_SUB32:
    write32le(Loc, read32le(Loc) - Val);
    return;
  case R_RISCV_SUB64:
    write64le(Loc, read64le(Loc) - Val);
    return;
  case R_RISCV_SET6:
    *Loc = (*Loc & 0xc0) | (Val & 0x3f);
    return;
  case R_RISCV_SET8:
    *Loc = Val;
    return;
  case R_RISCV_SET16:
    write16le(Loc, Val);
    return;
  case R_RISCV_SET32:
  case R_RISCV_32_PCREL:
    write32le(Loc, Val);
    return;

  case R_RISCV_TLS_DTPREL32:
    write32le(Loc, Val - DTPOffset);
    break;
  case R_RISCV_TLS_DTPREL64:
    write64le(Loc, Val - DTPOffset);
    break;

  case R_RISCV_ALIGN:
  case R_RISCV_RELAX:
    return; // Ignored (for now)
  case R_RISCV_NONE:
    return; // Do nothing

  // These are handled by the dynamic linker
  case R_RISCV_RELATIVE:
  case R_RISCV_COPY:
  case R_RISCV_JUMP_SLOT:
  // GP-relative relocations are only produced after relaxation, which
  // we don't support for now
  case R_RISCV_GPREL_I:
  case R_RISCV_GPREL_S:
  default:
    error(getErrorLocation(Loc) +
          "unimplemented relocation: " + toString(Type));
    return;
  }
}

TargetInfo *elf::getRISCVTargetInfo() {
  static RISCV Target;
  return &Target;
}
