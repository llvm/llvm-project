//===- RISCV.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "OutputSections.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "llvm/Support/TimeProfiler.h"

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
  int64_t getImplicitAddend(const uint8_t *buf, RelType type) const override;
  void writeGotHeader(uint8_t *buf) const override;
  void writeGotPlt(uint8_t *buf, const Symbol &s) const override;
  void writeIgotPlt(uint8_t *buf, const Symbol &s) const override;
  void writePltHeader(uint8_t *buf) const override;
  void writePlt(uint8_t *buf, const Symbol &sym,
                uint64_t pltEntryAddr) const override;
  RelType getDynRel(RelType type) const override;
  RelExpr getRelExpr(RelType type, const Symbol &s,
                     const uint8_t *loc) const override;
  void relocate(uint8_t *loc, const Relocation &rel,
                uint64_t val) const override;
  bool relaxOnce(int pass) const override;
};

} // end anonymous namespace

const uint64_t dtpOffset = 0x800;

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

static uint32_t hi20(uint32_t val) { return (val + 0x800) >> 12; }
static uint32_t lo12(uint32_t val) { return val & 4095; }

static uint32_t itype(uint32_t op, uint32_t rd, uint32_t rs1, uint32_t imm) {
  return op | (rd << 7) | (rs1 << 15) | (imm << 20);
}
static uint32_t rtype(uint32_t op, uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return op | (rd << 7) | (rs1 << 15) | (rs2 << 20);
}
static uint32_t utype(uint32_t op, uint32_t rd, uint32_t imm) {
  return op | (rd << 7) | (imm << 12);
}

RISCV::RISCV() {
  copyRel = R_RISCV_COPY;
  pltRel = R_RISCV_JUMP_SLOT;
  relativeRel = R_RISCV_RELATIVE;
  iRelativeRel = R_RISCV_IRELATIVE;
  if (config->is64) {
    symbolicRel = R_RISCV_64;
    tlsModuleIndexRel = R_RISCV_TLS_DTPMOD64;
    tlsOffsetRel = R_RISCV_TLS_DTPREL64;
    tlsGotRel = R_RISCV_TLS_TPREL64;
  } else {
    symbolicRel = R_RISCV_32;
    tlsModuleIndexRel = R_RISCV_TLS_DTPMOD32;
    tlsOffsetRel = R_RISCV_TLS_DTPREL32;
    tlsGotRel = R_RISCV_TLS_TPREL32;
  }
  gotRel = symbolicRel;

  // .got[0] = _DYNAMIC
  gotHeaderEntriesNum = 1;

  // .got.plt[0] = _dl_runtime_resolve, .got.plt[1] = link_map
  gotPltHeaderEntriesNum = 2;

  pltHeaderSize = 32;
  pltEntrySize = 16;
  ipltEntrySize = 16;
}

static uint32_t getEFlags(InputFile *f) {
  if (config->is64)
    return cast<ObjFile<ELF64LE>>(f)->getObj().getHeader().e_flags;
  return cast<ObjFile<ELF32LE>>(f)->getObj().getHeader().e_flags;
}

uint32_t RISCV::calcEFlags() const {
  // If there are only binary input files (from -b binary), use a
  // value of 0 for the ELF header flags.
  if (ctx->objectFiles.empty())
    return 0;

  uint32_t target = getEFlags(ctx->objectFiles.front());

  for (InputFile *f : ctx->objectFiles) {
    uint32_t eflags = getEFlags(f);
    if (eflags & EF_RISCV_RVC)
      target |= EF_RISCV_RVC;

    if ((eflags & EF_RISCV_FLOAT_ABI) != (target & EF_RISCV_FLOAT_ABI))
      error(toString(f) +
            ": cannot link object files with different floating-point ABI");

    if ((eflags & EF_RISCV_RVE) != (target & EF_RISCV_RVE))
      error(toString(f) +
            ": cannot link object files with different EF_RISCV_RVE");
  }

  return target;
}

int64_t RISCV::getImplicitAddend(const uint8_t *buf, RelType type) const {
  switch (type) {
  default:
    internalLinkerError(getErrorLocation(buf),
                        "cannot read addend for relocation " + toString(type));
    return 0;
  case R_RISCV_32:
  case R_RISCV_TLS_DTPMOD32:
  case R_RISCV_TLS_DTPREL32:
    return SignExtend64<32>(read32le(buf));
  case R_RISCV_64:
    return read64le(buf);
  case R_RISCV_RELATIVE:
  case R_RISCV_IRELATIVE:
    return config->is64 ? read64le(buf) : read32le(buf);
  case R_RISCV_NONE:
  case R_RISCV_JUMP_SLOT:
    // These relocations are defined as not having an implicit addend.
    return 0;
  }
}

void RISCV::writeGotHeader(uint8_t *buf) const {
  if (config->is64)
    write64le(buf, mainPart->dynamic->getVA());
  else
    write32le(buf, mainPart->dynamic->getVA());
}

void RISCV::writeGotPlt(uint8_t *buf, const Symbol &s) const {
  if (config->is64)
    write64le(buf, in.plt->getVA());
  else
    write32le(buf, in.plt->getVA());
}

void RISCV::writeIgotPlt(uint8_t *buf, const Symbol &s) const {
  if (config->writeAddends) {
    if (config->is64)
      write64le(buf, s.getVA());
    else
      write32le(buf, s.getVA());
  }
}

void RISCV::writePltHeader(uint8_t *buf) const {
  // 1: auipc t2, %pcrel_hi(.got.plt)
  // sub t1, t1, t3
  // l[wd] t3, %pcrel_lo(1b)(t2); t3 = _dl_runtime_resolve
  // addi t1, t1, -pltHeaderSize-12; t1 = &.plt[i] - &.plt[0]
  // addi t0, t2, %pcrel_lo(1b)
  // srli t1, t1, (rv64?1:2); t1 = &.got.plt[i] - &.got.plt[0]
  // l[wd] t0, Wordsize(t0); t0 = link_map
  // jr t3
  uint32_t offset = in.gotPlt->getVA() - in.plt->getVA();
  uint32_t load = config->is64 ? LD : LW;
  write32le(buf + 0, utype(AUIPC, X_T2, hi20(offset)));
  write32le(buf + 4, rtype(SUB, X_T1, X_T1, X_T3));
  write32le(buf + 8, itype(load, X_T3, X_T2, lo12(offset)));
  write32le(buf + 12, itype(ADDI, X_T1, X_T1, -target->pltHeaderSize - 12));
  write32le(buf + 16, itype(ADDI, X_T0, X_T2, lo12(offset)));
  write32le(buf + 20, itype(SRLI, X_T1, X_T1, config->is64 ? 1 : 2));
  write32le(buf + 24, itype(load, X_T0, X_T0, config->wordsize));
  write32le(buf + 28, itype(JALR, 0, X_T3, 0));
}

void RISCV::writePlt(uint8_t *buf, const Symbol &sym,
                     uint64_t pltEntryAddr) const {
  // 1: auipc t3, %pcrel_hi(f@.got.plt)
  // l[wd] t3, %pcrel_lo(1b)(t3)
  // jalr t1, t3
  // nop
  uint32_t offset = sym.getGotPltVA() - pltEntryAddr;
  write32le(buf + 0, utype(AUIPC, X_T3, hi20(offset)));
  write32le(buf + 4, itype(config->is64 ? LD : LW, X_T3, X_T3, lo12(offset)));
  write32le(buf + 8, itype(JALR, X_T1, X_T3, 0));
  write32le(buf + 12, itype(ADDI, 0, 0, 0));
}

RelType RISCV::getDynRel(RelType type) const {
  return type == target->symbolicRel ? type
                                     : static_cast<RelType>(R_RISCV_NONE);
}

RelExpr RISCV::getRelExpr(const RelType type, const Symbol &s,
                          const uint8_t *loc) const {
  switch (type) {
  case R_RISCV_NONE:
    return R_NONE;
  case R_RISCV_32:
  case R_RISCV_64:
  case R_RISCV_HI20:
  case R_RISCV_LO12_I:
  case R_RISCV_LO12_S:
  case R_RISCV_RVC_LUI:
    return R_ABS;
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
    config->hasTlsIe = true;
    return R_GOT_PC;
  case R_RISCV_TPREL_HI20:
  case R_RISCV_TPREL_LO12_I:
  case R_RISCV_TPREL_LO12_S:
    return R_TPREL;
  case R_RISCV_RELAX:
  case R_RISCV_TPREL_ADD:
    return R_NONE;
  case R_RISCV_ALIGN:
    return R_RELAX_HINT;
  default:
    error(getErrorLocation(loc) + "unknown relocation (" + Twine(type) +
          ") against symbol " + toString(s));
    return R_NONE;
  }
}

// Extract bits V[Begin:End], where range is inclusive, and Begin must be < 63.
static uint32_t extractBits(uint64_t v, uint32_t begin, uint32_t end) {
  return (v & ((1ULL << (begin + 1)) - 1)) >> end;
}

void RISCV::relocate(uint8_t *loc, const Relocation &rel, uint64_t val) const {
  const unsigned bits = config->wordsize * 8;

  switch (rel.type) {
  case R_RISCV_32:
    write32le(loc, val);
    return;
  case R_RISCV_64:
    write64le(loc, val);
    return;

  case R_RISCV_RVC_BRANCH: {
    checkInt(loc, static_cast<int64_t>(val) >> 1, 8, rel);
    checkAlignment(loc, val, 2, rel);
    uint16_t insn = read16le(loc) & 0xE383;
    uint16_t imm8 = extractBits(val, 8, 8) << 12;
    uint16_t imm4_3 = extractBits(val, 4, 3) << 10;
    uint16_t imm7_6 = extractBits(val, 7, 6) << 5;
    uint16_t imm2_1 = extractBits(val, 2, 1) << 3;
    uint16_t imm5 = extractBits(val, 5, 5) << 2;
    insn |= imm8 | imm4_3 | imm7_6 | imm2_1 | imm5;

    write16le(loc, insn);
    return;
  }

  case R_RISCV_RVC_JUMP: {
    checkInt(loc, static_cast<int64_t>(val) >> 1, 11, rel);
    checkAlignment(loc, val, 2, rel);
    uint16_t insn = read16le(loc) & 0xE003;
    uint16_t imm11 = extractBits(val, 11, 11) << 12;
    uint16_t imm4 = extractBits(val, 4, 4) << 11;
    uint16_t imm9_8 = extractBits(val, 9, 8) << 9;
    uint16_t imm10 = extractBits(val, 10, 10) << 8;
    uint16_t imm6 = extractBits(val, 6, 6) << 7;
    uint16_t imm7 = extractBits(val, 7, 7) << 6;
    uint16_t imm3_1 = extractBits(val, 3, 1) << 3;
    uint16_t imm5 = extractBits(val, 5, 5) << 2;
    insn |= imm11 | imm4 | imm9_8 | imm10 | imm6 | imm7 | imm3_1 | imm5;

    write16le(loc, insn);
    return;
  }

  case R_RISCV_RVC_LUI: {
    int64_t imm = SignExtend64(val + 0x800, bits) >> 12;
    checkInt(loc, imm, 6, rel);
    if (imm == 0) { // `c.lui rd, 0` is illegal, convert to `c.li rd, 0`
      write16le(loc, (read16le(loc) & 0x0F83) | 0x4000);
    } else {
      uint16_t imm17 = extractBits(val + 0x800, 17, 17) << 12;
      uint16_t imm16_12 = extractBits(val + 0x800, 16, 12) << 2;
      write16le(loc, (read16le(loc) & 0xEF83) | imm17 | imm16_12);
    }
    return;
  }

  case R_RISCV_JAL: {
    checkInt(loc, static_cast<int64_t>(val) >> 1, 20, rel);
    checkAlignment(loc, val, 2, rel);

    uint32_t insn = read32le(loc) & 0xFFF;
    uint32_t imm20 = extractBits(val, 20, 20) << 31;
    uint32_t imm10_1 = extractBits(val, 10, 1) << 21;
    uint32_t imm11 = extractBits(val, 11, 11) << 20;
    uint32_t imm19_12 = extractBits(val, 19, 12) << 12;
    insn |= imm20 | imm10_1 | imm11 | imm19_12;

    write32le(loc, insn);
    return;
  }

  case R_RISCV_BRANCH: {
    checkInt(loc, static_cast<int64_t>(val) >> 1, 12, rel);
    checkAlignment(loc, val, 2, rel);

    uint32_t insn = read32le(loc) & 0x1FFF07F;
    uint32_t imm12 = extractBits(val, 12, 12) << 31;
    uint32_t imm10_5 = extractBits(val, 10, 5) << 25;
    uint32_t imm4_1 = extractBits(val, 4, 1) << 8;
    uint32_t imm11 = extractBits(val, 11, 11) << 7;
    insn |= imm12 | imm10_5 | imm4_1 | imm11;

    write32le(loc, insn);
    return;
  }

  // auipc + jalr pair
  case R_RISCV_CALL:
  case R_RISCV_CALL_PLT: {
    int64_t hi = SignExtend64(val + 0x800, bits) >> 12;
    checkInt(loc, hi, 20, rel);
    if (isInt<20>(hi)) {
      relocateNoSym(loc, R_RISCV_PCREL_HI20, val);
      relocateNoSym(loc + 4, R_RISCV_PCREL_LO12_I, val);
    }
    return;
  }

  case R_RISCV_GOT_HI20:
  case R_RISCV_PCREL_HI20:
  case R_RISCV_TLS_GD_HI20:
  case R_RISCV_TLS_GOT_HI20:
  case R_RISCV_TPREL_HI20:
  case R_RISCV_HI20: {
    uint64_t hi = val + 0x800;
    checkInt(loc, SignExtend64(hi, bits) >> 12, 20, rel);
    write32le(loc, (read32le(loc) & 0xFFF) | (hi & 0xFFFFF000));
    return;
  }

  case R_RISCV_PCREL_LO12_I:
  case R_RISCV_TPREL_LO12_I:
  case R_RISCV_LO12_I: {
    uint64_t hi = (val + 0x800) >> 12;
    uint64_t lo = val - (hi << 12);
    write32le(loc, (read32le(loc) & 0xFFFFF) | ((lo & 0xFFF) << 20));
    return;
  }

  case R_RISCV_PCREL_LO12_S:
  case R_RISCV_TPREL_LO12_S:
  case R_RISCV_LO12_S: {
    uint64_t hi = (val + 0x800) >> 12;
    uint64_t lo = val - (hi << 12);
    uint32_t imm11_5 = extractBits(lo, 11, 5) << 25;
    uint32_t imm4_0 = extractBits(lo, 4, 0) << 7;
    write32le(loc, (read32le(loc) & 0x1FFF07F) | imm11_5 | imm4_0);
    return;
  }

  case R_RISCV_ADD8:
    *loc += val;
    return;
  case R_RISCV_ADD16:
    write16le(loc, read16le(loc) + val);
    return;
  case R_RISCV_ADD32:
    write32le(loc, read32le(loc) + val);
    return;
  case R_RISCV_ADD64:
    write64le(loc, read64le(loc) + val);
    return;
  case R_RISCV_SUB6:
    *loc = (*loc & 0xc0) | (((*loc & 0x3f) - val) & 0x3f);
    return;
  case R_RISCV_SUB8:
    *loc -= val;
    return;
  case R_RISCV_SUB16:
    write16le(loc, read16le(loc) - val);
    return;
  case R_RISCV_SUB32:
    write32le(loc, read32le(loc) - val);
    return;
  case R_RISCV_SUB64:
    write64le(loc, read64le(loc) - val);
    return;
  case R_RISCV_SET6:
    *loc = (*loc & 0xc0) | (val & 0x3f);
    return;
  case R_RISCV_SET8:
    *loc = val;
    return;
  case R_RISCV_SET16:
    write16le(loc, val);
    return;
  case R_RISCV_SET32:
  case R_RISCV_32_PCREL:
    write32le(loc, val);
    return;

  case R_RISCV_TLS_DTPREL32:
    write32le(loc, val - dtpOffset);
    break;
  case R_RISCV_TLS_DTPREL64:
    write64le(loc, val - dtpOffset);
    break;

  case R_RISCV_RELAX:
    return; // Ignored (for now)

  default:
    llvm_unreachable("unknown relocation");
  }
}

namespace {
struct SymbolAnchor {
  uint64_t offset;
  Defined *d;
  bool end; // true for the anchor of st_value+st_size
};
} // namespace

struct elf::RISCVRelaxAux {
  // This records symbol start and end offsets which will be adjusted according
  // to the nearest relocDeltas element.
  SmallVector<SymbolAnchor, 0> anchors;
  // For relocations[i], the actual offset is r_offset - (i ? relocDeltas[i-1] :
  // 0).
  std::unique_ptr<uint32_t[]> relocDeltas;
};

static void initSymbolAnchors() {
  SmallVector<InputSection *, 0> storage;
  for (OutputSection *osec : outputSections) {
    if (!(osec->flags & SHF_EXECINSTR))
      continue;
    for (InputSection *sec : getInputSections(*osec, storage)) {
      sec->relaxAux = make<RISCVRelaxAux>();
      if (sec->relocations.size())
        sec->relaxAux->relocDeltas =
            std::make_unique<uint32_t[]>(sec->relocations.size());
    }
  }
  // Store anchors (st_value and st_value+st_size) for symbols relative to text
  // sections.
  for (InputFile *file : ctx->objectFiles)
    for (Symbol *sym : file->getSymbols()) {
      auto *d = dyn_cast<Defined>(sym);
      if (!d || d->file != file)
        continue;
      if (auto *sec = dyn_cast_or_null<InputSection>(d->section))
        if (sec->flags & SHF_EXECINSTR && sec->relaxAux) {
          // If sec is discarded, relaxAux will be nullptr.
          sec->relaxAux->anchors.push_back({d->value, d, false});
          sec->relaxAux->anchors.push_back({d->value + d->size, d, true});
        }
    }
  // Sort anchors by offset so that we can find the closest relocation
  // efficiently. For a zero size symbol, ensure that its start anchor precedes
  // its end anchor. For two symbols with anchors at the same offset, their
  // order does not matter.
  for (OutputSection *osec : outputSections) {
    if (!(osec->flags & SHF_EXECINSTR))
      continue;
    for (InputSection *sec : getInputSections(*osec, storage)) {
      llvm::sort(sec->relaxAux->anchors, [](auto &a, auto &b) {
        return std::make_pair(a.offset, a.end) <
               std::make_pair(b.offset, b.end);
      });
    }
  }
}

static bool relax(InputSection &sec) {
  const uint64_t secAddr = sec.getVA();
  auto &aux = *sec.relaxAux;
  bool changed = false;

  // Restore original st_value for symbols relative to this section.
  ArrayRef<SymbolAnchor> sa = makeArrayRef(aux.anchors);
  uint32_t delta = 0;
  for (auto it : llvm::enumerate(sec.relocations)) {
    for (; sa.size() && sa[0].offset <= it.value().offset; sa = sa.slice(1))
      if (!sa[0].end)
        sa[0].d->value += delta;
    delta = aux.relocDeltas[it.index()];
  }
  for (const SymbolAnchor &sa : sa)
    if (!sa.end)
      sa.d->value += delta;
  sa = makeArrayRef(aux.anchors);
  delta = 0;

  for (auto it : llvm::enumerate(sec.relocations)) {
    Relocation &r = it.value();
    const size_t i = it.index();
    const uint64_t loc = secAddr + r.offset - delta;
    uint32_t &cur = aux.relocDeltas[i], remove = 0;
    switch (r.type) {
    case R_RISCV_ALIGN: {
      const uint64_t nextLoc = loc + r.addend;
      const uint64_t align = PowerOf2Ceil(r.addend + 2);
      // All bytes beyond the alignment boundary should be removed.
      remove = nextLoc - ((loc + align - 1) & -align);
      assert(static_cast<int32_t>(remove) >= 0 &&
             "R_RISCV_ALIGN needs expanding the content");
      break;
    }
    }

    // For all anchors whose offsets are <= r.offset, they are preceded by
    // the previous relocation whose `relocDeltas` value equals `delta`.
    // Decrease their st_value and update their st_size.
    if (remove) {
      for (; sa.size() && sa[0].offset <= r.offset; sa = sa.slice(1)) {
        if (sa[0].end)
          sa[0].d->size = sa[0].offset - delta - sa[0].d->value;
        else
          sa[0].d->value -= delta;
      }
    }
    delta += remove;
    if (delta != cur) {
      cur = delta;
      changed = true;
    }
  }

  for (const SymbolAnchor &a : sa) {
    if (a.end)
      a.d->size = a.offset - delta - a.d->value;
    else
      a.d->value -= delta;
  }
  // Inform assignAddresses that the size has changed.
  if (!isUInt<16>(delta))
    fatal("section size decrease is too large");
  sec.bytesDropped = delta;
  return changed;
}

// When relaxing just R_RISCV_ALIGN, relocDeltas is usually changed only once in
// the absence of a linker script. For call and load/store R_RISCV_RELAX, code
// shrinkage may reduce displacement and make more relocations eligible for
// relaxation. Code shrinkage may increase displacement to a call/load/store
// target at a higher fixed address, invalidating an earlier relaxation. Any
// change in section sizes can have cascading effect and require another
// relaxation pass.
bool RISCV::relaxOnce(int pass) const {
  llvm::TimeTraceScope timeScope("RISC-V relaxOnce");
  if (config->relocatable)
    return false;

  if (pass == 0)
    initSymbolAnchors();

  SmallVector<InputSection *, 0> storage;
  bool changed = false;
  for (OutputSection *osec : outputSections) {
    if (!(osec->flags & SHF_EXECINSTR))
      continue;
    for (InputSection *sec : getInputSections(*osec, storage))
      changed |= relax(*sec);
  }
  return changed;
}

void elf::riscvFinalizeRelax(int passes) {
  llvm::TimeTraceScope timeScope("Finalize RISC-V relaxation");
  log("relaxation passes: " + Twine(passes));
  SmallVector<InputSection *, 0> storage;
  for (OutputSection *osec : outputSections) {
    if (!(osec->flags & SHF_EXECINSTR))
      continue;
    for (InputSection *sec : getInputSections(*osec, storage)) {
      RISCVRelaxAux &aux = *sec->relaxAux;
      if (!aux.relocDeltas)
        continue;

      auto &rels = sec->relocations;
      ArrayRef<uint8_t> old = sec->rawData;
      size_t newSize =
          old.size() - aux.relocDeltas[sec->relocations.size() - 1];
      uint8_t *p = context().bAlloc.Allocate<uint8_t>(newSize);
      uint64_t offset = 0;
      int64_t delta = 0;
      sec->rawData = makeArrayRef(p, newSize);
      sec->bytesDropped = 0;

      // Update section content: remove NOPs for R_RISCV_ALIGN and rewrite
      // instructions for relaxed relocations.
      for (size_t i = 0, e = rels.size(); i != e; ++i) {
        uint32_t remove = aux.relocDeltas[i] - delta;
        delta = aux.relocDeltas[i];
        if (remove == 0)
          continue;

        // Copy from last location to the current relocated location.
        const Relocation &r = rels[i];
        uint64_t size = r.offset - offset;
        memcpy(p, old.data() + offset, size);
        p += size;

        // For R_RISCV_ALIGN, we will place `offset` in a location (among NOPs)
        // to satisfy the alignment requirement. If `remove` is a multiple of 4,
        // it is as if we have skipped some NOPs. Otherwise we are in the middle
        // of a 4-byte NOP, and we need to rewrite the NOP sequence.
        int64_t skip = 0;
        if (r.type == R_RISCV_ALIGN) {
          if (remove % 4 != 0) {
            skip = r.addend - remove;
            int64_t j = 0;
            for (; j + 4 <= skip; j += 4)
              write32le(p + j, 0x00000013); // nop
            if (j != skip) {
              assert(j + 2 == skip);
              write16le(p + j, 0x0001); // c.nop
            }
          }
        }

        p += skip;
        offset = r.offset + skip + remove;
      }
      memcpy(p, old.data() + offset, old.size() - offset);

      // Substract the previous relocDeltas value from the relocation offset.
      // For a pair of R_RISCV_CALL/R_RISCV_RELAX with the same offset, decrease
      // their r_offset by the same delta.
      delta = 0;
      for (size_t i = 0, e = rels.size(); i != e;) {
        uint64_t cur = rels[i].offset;
        do {
          rels[i].offset -= delta;
        } while (++i != e && rels[i].offset == cur);
        delta = aux.relocDeltas[i - 1];
      }
    }
  }
}

TargetInfo *elf::getRISCVTargetInfo() {
  static RISCV target;
  return &target;
}
