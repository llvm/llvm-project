//===- SystemZ.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OutputSections.h"
#include "RelocScan.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {
class SystemZ : public TargetInfo {
public:
  SystemZ(Ctx &);
  RelExpr getRelExpr(RelType type, const Symbol &s,
                     const uint8_t *loc) const override;
  RelType getDynRel(RelType type) const override;
  void writeGotHeader(uint8_t *buf) const override;
  void writeGotPlt(uint8_t *buf, const Symbol &s) const override;
  void writeIgotPlt(uint8_t *buf, const Symbol &s) const override;
  void writePltHeader(uint8_t *buf) const override;
  void addPltHeaderSymbols(InputSection &isd) const override;
  void writePlt(uint8_t *buf, const Symbol &sym,
                uint64_t pltEntryAddr) const override;
  template <class ELFT, class RelTy>
  void scanSectionImpl(InputSectionBase &sec, Relocs<RelTy> rels);
  void scanSection(InputSectionBase &sec) override;
  RelExpr adjustGotPcExpr(RelType type, int64_t addend,
                          const uint8_t *loc) const override;
  bool relaxOnce(int pass) const override;
  void relocate(uint8_t *loc, const Relocation &rel,
                uint64_t val) const override;
  int64_t getImplicitAddend(const uint8_t *buf, RelType type) const override;

private:
  void relaxGot(uint8_t *loc, const Relocation &rel, uint64_t val) const;
  void relaxTlsGdCall(uint8_t *loc, const Relocation &rel) const;
};
} // namespace

SystemZ::SystemZ(Ctx &ctx) : TargetInfo(ctx) {
  copyRel = R_390_COPY;
  gotRel = R_390_GLOB_DAT;
  pltRel = R_390_JMP_SLOT;
  relativeRel = R_390_RELATIVE;
  iRelativeRel = R_390_IRELATIVE;
  symbolicRel = R_390_64;
  tlsGotRel = R_390_TLS_TPOFF;
  tlsModuleIndexRel = R_390_TLS_DTPMOD;
  tlsOffsetRel = R_390_TLS_DTPOFF;
  gotHeaderEntriesNum = 3;
  gotPltHeaderEntriesNum = 0;
  gotEntrySize = 8;
  pltHeaderSize = 32;
  pltEntrySize = 32;
  ipltEntrySize = 32;

  // This "trap instruction" is used to fill gaps between sections.
  // On SystemZ, the behavior of the GNU ld is to fill those gaps
  // with nop instructions instead - and unfortunately the default
  // glibc crt object files (used to) rely on that behavior since
  // they use an alignment on the .init section fragments that causes
  // gaps which must be filled with nops as they are being executed.
  // Therefore, we provide a nop instruction as "trapInstr" here.
  trapInstr = {0x07, 0x07, 0x07, 0x07};

  defaultImageBase = 0x1000000;
}

// Only handles relocations used by relocateNonAlloc and preprocessRelocs.
RelExpr SystemZ::getRelExpr(RelType type, const Symbol &s,
                            const uint8_t *loc) const {
  switch (type) {
  case R_390_NONE:
    return R_NONE;
  case R_390_32:
  case R_390_64:
    return R_ABS;
  case R_390_TLS_LDO32:
  case R_390_TLS_LDO64:
    return R_DTPREL;
  case R_390_PC32:
  case R_390_PC64:
    return R_PC;
  default:
    Err(ctx) << getErrorLoc(ctx, loc) << "unknown relocation (" << type.v
             << ") against symbol " << &s;
    return R_NONE;
  }
}

void SystemZ::writeGotHeader(uint8_t *buf) const {
  // _GLOBAL_OFFSET_TABLE_[0] holds the value of _DYNAMIC.
  // _GLOBAL_OFFSET_TABLE_[1] and [2] are reserved.
  write64be(buf, ctx.mainPart->dynamic->getVA());
}

void SystemZ::writeGotPlt(uint8_t *buf, const Symbol &s) const {
  write64be(buf, s.getPltVA(ctx) + 14);
}

void SystemZ::writeIgotPlt(uint8_t *buf, const Symbol &s) const {
  if (ctx.arg.writeAddends)
    write64be(buf, s.getVA(ctx));
}

void SystemZ::writePltHeader(uint8_t *buf) const {
  const uint8_t pltData[] = {
      0xe3, 0x10, 0xf0, 0x38, 0x00, 0x24, // stg     %r1,56(%r15)
      0xc0, 0x10, 0x00, 0x00, 0x00, 0x00, // larl    %r1,_GLOBAL_OFFSET_TABLE_
      0xd2, 0x07, 0xf0, 0x30, 0x10, 0x08, // mvc     48(8,%r15),8(%r1)
      0xe3, 0x10, 0x10, 0x10, 0x00, 0x04, // lg      %r1,16(%r1)
      0x07, 0xf1,                         // br      %r1
      0x07, 0x00,                         // nopr
      0x07, 0x00,                         // nopr
      0x07, 0x00,                         // nopr
  };
  memcpy(buf, pltData, sizeof(pltData));
  uint64_t got = ctx.in.got->getVA();
  uint64_t plt = ctx.in.plt->getVA();
  write32be(buf + 8, (got - plt - 6) >> 1);
}

void SystemZ::addPltHeaderSymbols(InputSection &isec) const {
  // The PLT header needs a reference to _GLOBAL_OFFSET_TABLE_, so we
  // must ensure the .got section is created even if otherwise unused.
  ctx.in.got->hasGotOffRel.store(true, std::memory_order_relaxed);
}

void SystemZ::writePlt(uint8_t *buf, const Symbol &sym,
                       uint64_t pltEntryAddr) const {
  const uint8_t inst[] = {
      0xc0, 0x10, 0x00, 0x00, 0x00, 0x00, // larl    %r1,<.got.plt slot>
      0xe3, 0x10, 0x10, 0x00, 0x00, 0x04, // lg      %r1,0(%r1)
      0x07, 0xf1,                         // br      %r1
      0x0d, 0x10,                         // basr    %r1,%r0
      0xe3, 0x10, 0x10, 0x0c, 0x00, 0x14, // lgf     %r1,12(%r1)
      0xc0, 0xf4, 0x00, 0x00, 0x00, 0x00, // jg      <plt header>
      0x00, 0x00, 0x00, 0x00,             // <relocation offset>
  };
  memcpy(buf, inst, sizeof(inst));

  write32be(buf + 2, (sym.getGotPltVA(ctx) - pltEntryAddr) >> 1);
  write32be(buf + 24, (ctx.in.plt->getVA() - pltEntryAddr - 22) >> 1);
  write32be(buf + 28, ctx.in.relaPlt->entsize * sym.getPltIdx(ctx));
}

int64_t SystemZ::getImplicitAddend(const uint8_t *buf, RelType type) const {
  switch (type) {
  case R_390_8:
    return SignExtend64<8>(*buf);
  case R_390_16:
  case R_390_PC16:
    return SignExtend64<16>(read16be(buf));
  case R_390_PC16DBL:
    return SignExtend64<16>(read16be(buf)) << 1;
  case R_390_32:
  case R_390_PC32:
    return SignExtend64<32>(read32be(buf));
  case R_390_PC32DBL:
    return SignExtend64<32>(read32be(buf)) << 1;
  case R_390_64:
  case R_390_PC64:
  case R_390_TLS_DTPMOD:
  case R_390_TLS_DTPOFF:
  case R_390_TLS_TPOFF:
  case R_390_GLOB_DAT:
  case R_390_RELATIVE:
  case R_390_IRELATIVE:
    return read64be(buf);
  case R_390_COPY:
  case R_390_JMP_SLOT:
  case R_390_NONE:
    // These relocations are defined as not having an implicit addend.
    return 0;
  default:
    InternalErr(ctx, buf) << "cannot read addend for relocation " << type;
    return 0;
  }
}

template <class ELFT, class RelTy>
void SystemZ::scanSectionImpl(InputSectionBase &sec, Relocs<RelTy> rels) {
  RelocScan rs(ctx, &sec);
  sec.relocations.reserve(rels.size());

  for (auto it = rels.begin(); it != rels.end(); ++it) {
    RelType type = it->getType(false);

    // The assembler emits R_390_PLT32DBL (at the displacement field) before
    // R_390_TLS_GDCALL/LDCALL (at the instruction start) for the same brasl.
    // When optimizing TLS, skip PLT32DBL before maybeReportUndefined would
    // flag __tls_get_offset as undefined.
    if (type == R_390_PLT32DBL && !ctx.arg.shared &&
        std::next(it) != rels.end()) {
      RelType nextType = std::next(it)->getType(false);
      if (nextType == R_390_TLS_GDCALL || nextType == R_390_TLS_LDCALL)
        continue;
    }

    uint32_t symIdx = it->getSymbol(false);
    Symbol &sym = sec.getFile<ELFT>()->getSymbol(symIdx);
    uint64_t offset = it->r_offset;
    if (sym.isUndefined() && symIdx != 0 &&
        rs.maybeReportUndefined(cast<Undefined>(sym), offset))
      continue;
    int64_t addend = rs.getAddend<ELFT>(*it, type);
    RelExpr expr;
    // Relocation types that only need a RelExpr set `expr` and break out of
    // the switch to reach rs.process(). Types that need special handling
    // (fast-path helpers, TLS) call a handler and use `continue`.
    switch (type) {
    case R_390_NONE:
    case R_390_TLS_LOAD:
      continue;

    // Absolute relocations:
    case R_390_8:
    case R_390_12:
    case R_390_16:
    case R_390_20:
    case R_390_32:
    case R_390_64:
      expr = R_ABS;
      break;

    // PC-relative relocations:
    case R_390_PC16:
    case R_390_PC32:
    case R_390_PC64:
    case R_390_PC12DBL:
    case R_390_PC16DBL:
    case R_390_PC24DBL:
    case R_390_PC32DBL:
      rs.processR_PC(type, offset, addend, sym);
      continue;

    // PLT-generating relocations:
    case R_390_PLT32:
    case R_390_PLT64:
    case R_390_PLT12DBL:
    case R_390_PLT16DBL:
    case R_390_PLT24DBL:
    case R_390_PLT32DBL:
      rs.processR_PLT_PC(type, offset, addend, sym);
      continue;
    case R_390_PLTOFF16:
    case R_390_PLTOFF32:
    case R_390_PLTOFF64:
      expr = R_PLT_GOTREL;
      break;

    // GOT-generating relocations:
    case R_390_GOTOFF16:
    case R_390_GOTOFF: // a.k.a. R_390_GOTOFF32
    case R_390_GOTOFF64:
      ctx.in.got->hasGotOffRel.store(true, std::memory_order_relaxed);
      expr = R_GOTREL;
      break;
    case R_390_GOTENT:
      expr = R_GOT_PC;
      break;
    case R_390_GOT12:
    case R_390_GOT16:
    case R_390_GOT20:
    case R_390_GOT32:
    case R_390_GOT64:
      expr = R_GOT_OFF;
      break;

    case R_390_GOTPLTENT:
      expr = R_GOTPLT_PC;
      break;
    case R_390_GOTPLT12:
    case R_390_GOTPLT16:
    case R_390_GOTPLT20:
    case R_390_GOTPLT32:
    case R_390_GOTPLT64:
      expr = R_GOTPLT_GOTREL;
      break;
    case R_390_GOTPC:
    case R_390_GOTPCDBL:
      ctx.in.got->hasGotOffRel.store(true, std::memory_order_relaxed);
      expr = R_GOTONLY_PC;
      break;

    // TLS relocations:
    case R_390_TLS_LE32:
    case R_390_TLS_LE64:
      if (rs.checkTlsLe(offset, sym, type))
        continue;
      expr = R_TPREL;
      break;
    case R_390_TLS_IE32:
    case R_390_TLS_IE64:
      // There is no IE to LE optimization.
      rs.handleTlsIe<false>(R_GOT, type, offset, addend, sym);
      continue;
    case R_390_TLS_GOTIE12:
    case R_390_TLS_GOTIE20:
    case R_390_TLS_GOTIE32:
    case R_390_TLS_GOTIE64:
      sym.setFlags(NEEDS_TLSIE);
      sec.addReloc({R_GOT_OFF, type, offset, addend, &sym});
      continue;
    case R_390_TLS_IEENT:
      sym.setFlags(NEEDS_TLSIE);
      sec.addReloc({R_GOT_PC, type, offset, addend, &sym});
      continue;
    case R_390_TLS_GDCALL:
      // Use dummy R_ABS for `sharedExpr` (no optimization), which is a no-op in
      // relocate().
      rs.handleTlsGd(R_ABS, R_GOT_OFF, R_TPREL, type, offset, addend, sym);
      continue;
    case R_390_TLS_GD32:
    case R_390_TLS_GD64:
      rs.handleTlsGd(R_TLSGD_GOT, R_GOT_OFF, R_TPREL, type, offset, addend,
                     sym);
      continue;

    case R_390_TLS_LDCALL:
      // Use dummy R_ABS for `sharedExpr` (no optimization), which is a no-op in
      // relocate().
      rs.handleTlsLd(R_ABS, type, offset, addend, sym);
      continue;
    // TLS LD GOT relocations:
    case R_390_TLS_LDM32:
    case R_390_TLS_LDM64:
      rs.handleTlsLd(R_TLSLD_GOT, type, offset, addend, sym);
      continue;
    // TLS DTPREL relocations:
    case R_390_TLS_LDO32:
    case R_390_TLS_LDO64:
      if (ctx.arg.shared)
        sec.addReloc({R_DTPREL, type, offset, addend, &sym});
      else
        sec.addReloc({R_TPREL, type, offset, addend, &sym});
      continue;

    default:
      Err(ctx) << getErrorLoc(ctx, sec.content().data() + offset)
               << "unknown relocation (" << type.v << ") against symbol "
               << &sym;
      continue;
    }
    rs.process(expr, type, offset, sym, addend);
  }
}

void SystemZ::scanSection(InputSectionBase &sec) {
  elf::scanSection1<SystemZ, ELF64BE>(*this, sec);
}

RelType SystemZ::getDynRel(RelType type) const {
  if (type == R_390_64 || type == R_390_PC64)
    return type;
  return R_390_NONE;
}

// Rewrite the brasl instruction at loc for TLS GD/LD optimization.
//
// The general-dynamic code sequence for a global `x`:
//
// Instruction                      Relocation       Symbol
// ear %rX,%a0
// sllg %rX,%rX,32
// ear %rX,%a1
// larl %r12,_GLOBAL_OFFSET_TABLE_  R_390_GOTPCDBL   _GLOBAL_OFFSET_TABLE_
// lgrl %r2,.LC0                    R_390_PC32DBL    .LC0
// brasl %r14,__tls_get_offset@plt  R_390_TLS_GDCALL x
//            :tls_gdcall:x         R_390_PLT32DBL   __tls_get_offset
// la %r2,0(%r2,%rX)
//
// .LC0:
// .quad   x@TLSGD                  R_390_TLS_GD64   x
//
// GD -> IE: replacing the call by a GOT load and LC0 by R_390_TLS_GOTIE64.
// GD -> LE: replacing the call by a nop and LC0 by R_390_TLS_LE64.
//
// The local-dynamic code sequence for a global `x`:
//
// Instruction                      Relocation       Symbol
// ear %rX,%a0
// sllg %rX,%rX,32
// ear %rX,%a1
// larl %r12,_GLOBAL_OFFSET_TABLE_  R_390_GOTPCDBL   _GLOBAL_OFFSET_TABLE_
// lgrl %r2,.LC0                    R_390_PC32DBL    .LC0
// brasl %r14,__tls_get_offset@plt  R_390_TLS_LDCALL <sym>
//            :tls_ldcall:<sym>     R_390_PLT32DBL   __tls_get_offset
// la %r2,0(%r2,%rX)
// lgrl %rY,.LC1                    R_390_PC32DBL    .LC1
// la %r2,0(%r2,%rY)
//
// .LC0:
// .quad   <sym>@tlsldm             R_390_TLS_LDM64  <sym>
// .LC1:
// .quad   x@dtpoff                 R_390_TLS_LDO64  x
//
// LD -> LE: replacing the call by a nop, LC0 by 0, LC1 by R_390_TLS_LE64.
void SystemZ::relaxTlsGdCall(uint8_t *loc, const Relocation &rel) const {
  if (rel.expr == R_GOT_OFF) {
    // brasl %r14,__tls_get_offset@plt -> lg %r2,0(%r2,%r12)
    write16be(loc, 0xe322);
    write32be(loc + 2, 0xc0000004);
  } else {
    // brasl %r14,__tls_get_offset@plt -> brcl 0,.
    write16be(loc, 0xc004);
    write32be(loc + 2, 0x00000000);
  }
}

RelExpr SystemZ::adjustGotPcExpr(RelType type, int64_t addend,
                                 const uint8_t *loc) const {
  // Only R_390_GOTENT with addend 2 can be relaxed.
  if (!ctx.arg.relax || addend != 2 || type != R_390_GOTENT)
    return R_GOT_PC;
  const uint16_t op = read16be(loc - 2);

  // lgrl rx,sym@GOTENT -> larl rx, sym
  // This relaxation is legal if "sym" binds locally (which was already
  // verified by our caller) and is in-range and properly aligned for a
  // LARL instruction.  We cannot verify the latter constraint here, so
  // we assume it is true and revert the decision later on in relaxOnce
  // if necessary.
  if ((op & 0xff0f) == 0xc408)
    return R_RELAX_GOT_PC;

  return R_GOT_PC;
}

bool SystemZ::relaxOnce(int pass) const {
  // If we decided in adjustGotPcExpr to relax a R_390_GOTENT,
  // we need to validate the target symbol is in-range and aligned.
  SmallVector<InputSection *, 0> storage;
  bool changed = false;
  for (OutputSection *osec : ctx.outputSections) {
    if (!(osec->flags & SHF_EXECINSTR))
      continue;
    for (InputSection *sec : getInputSections(*osec, storage)) {
      for (Relocation &rel : sec->relocs()) {
        if (rel.expr != R_RELAX_GOT_PC)
          continue;

        uint64_t v = sec->getRelocTargetVA(
            ctx, rel, sec->getOutputSection()->addr + rel.offset);
        if (isInt<33>(v) && !(v & 1))
          continue;
        if (rel.sym->auxIdx == 0) {
          rel.sym->allocateAux(ctx);
          addGotEntry(ctx, *rel.sym);
          changed = true;
        }
        rel.expr = R_GOT_PC;
      }
    }
  }
  return changed;
}

void SystemZ::relaxGot(uint8_t *loc, const Relocation &rel,
                       uint64_t val) const {
  assert(isInt<33>(val) &&
         "R_390_GOTENT should not have been relaxed if it overflows");
  assert(!(val & 1) &&
         "R_390_GOTENT should not have been relaxed if it is misaligned");
  const uint16_t op = read16be(loc - 2);

  // lgrl rx,sym@GOTENT -> larl rx, sym
  if ((op & 0xff0f) == 0xc408) {
    write16be(loc - 2, 0xc000 | (op & 0x00f0));
    write32be(loc, val >> 1);
  }
}

void SystemZ::relocate(uint8_t *loc, const Relocation &rel,
                       uint64_t val) const {
  if (rel.expr == R_RELAX_GOT_PC)
    return relaxGot(loc, rel, val);

  // Handle TLS optimizations. GDCALL/LDCALL: rewrite the brasl instruction
  // and return. LDM slots are zeroed when relaxed to LE. Other TLS data slot
  // types (GD32/GD64, LDO) fall through to the normal type-based switch below.
  switch (rel.type) {
  case R_390_TLS_GDCALL:
  case R_390_TLS_LDCALL:
    if (rel.expr == R_ABS) // Shared: no optimization.
      return;
    relaxTlsGdCall(loc, rel);
    return;
  case R_390_TLS_LDM32:
  case R_390_TLS_LDM64:
    if (rel.expr == R_TPREL)
      return; // LD -> LE: slot stays 0.
    break;
  default:
    break;
  }

  switch (rel.type) {
  case R_390_8:
    checkIntUInt(ctx, loc, val, 8, rel);
    *loc = val;
    break;
  case R_390_12:
  case R_390_GOT12:
  case R_390_GOTPLT12:
  case R_390_TLS_GOTIE12:
    checkUInt(ctx, loc, val, 12, rel);
    write16be(loc, (read16be(loc) & 0xF000) | val);
    break;
  case R_390_PC12DBL:
  case R_390_PLT12DBL:
    checkInt(ctx, loc, val, 13, rel);
    checkAlignment(ctx, loc, val, 2, rel);
    write16be(loc, (read16be(loc) & 0xF000) | ((val >> 1) & 0x0FFF));
    break;
  case R_390_16:
  case R_390_GOT16:
  case R_390_GOTPLT16:
  case R_390_GOTOFF16:
  case R_390_PLTOFF16:
    checkIntUInt(ctx, loc, val, 16, rel);
    write16be(loc, val);
    break;
  case R_390_PC16:
    checkInt(ctx, loc, val, 16, rel);
    write16be(loc, val);
    break;
  case R_390_PC16DBL:
  case R_390_PLT16DBL:
    checkInt(ctx, loc, val, 17, rel);
    checkAlignment(ctx, loc, val, 2, rel);
    write16be(loc, val >> 1);
    break;
  case R_390_20:
  case R_390_GOT20:
  case R_390_GOTPLT20:
  case R_390_TLS_GOTIE20:
    checkInt(ctx, loc, val, 20, rel);
    write32be(loc, (read32be(loc) & 0xF00000FF) | ((val & 0xFFF) << 16) |
                       ((val & 0xFF000) >> 4));
    break;
  case R_390_PC24DBL:
  case R_390_PLT24DBL:
    checkInt(ctx, loc, val, 25, rel);
    checkAlignment(ctx, loc, val, 2, rel);
    loc[0] = val >> 17;
    loc[1] = val >> 9;
    loc[2] = val >> 1;
    break;
  case R_390_32:
  case R_390_GOT32:
  case R_390_GOTPLT32:
  case R_390_GOTOFF:
  case R_390_PLTOFF32:
  case R_390_TLS_IE32:
  case R_390_TLS_GOTIE32:
  case R_390_TLS_GD32:
  case R_390_TLS_LDM32:
  case R_390_TLS_LDO32:
  case R_390_TLS_LE32:
    checkIntUInt(ctx, loc, val, 32, rel);
    write32be(loc, val);
    break;
  case R_390_PC32:
  case R_390_PLT32:
    checkInt(ctx, loc, val, 32, rel);
    write32be(loc, val);
    break;
  case R_390_PC32DBL:
  case R_390_PLT32DBL:
  case R_390_GOTPCDBL:
  case R_390_GOTENT:
  case R_390_GOTPLTENT:
  case R_390_TLS_IEENT:
    checkInt(ctx, loc, val, 33, rel);
    checkAlignment(ctx, loc, val, 2, rel);
    write32be(loc, val >> 1);
    break;
  case R_390_64:
  case R_390_PC64:
  case R_390_PLT64:
  case R_390_GOT64:
  case R_390_GOTPLT64:
  case R_390_GOTOFF64:
  case R_390_PLTOFF64:
  case R_390_GOTPC:
  case R_390_TLS_IE64:
  case R_390_TLS_GOTIE64:
  case R_390_TLS_GD64:
  case R_390_TLS_LDM64:
  case R_390_TLS_LDO64:
  case R_390_TLS_LE64:
  case R_390_TLS_DTPMOD:
  case R_390_TLS_DTPOFF:
  case R_390_TLS_TPOFF:
    write64be(loc, val);
    break;
  case R_390_TLS_LOAD:
    break;
  default:
    llvm_unreachable("unknown relocation");
  }
}

void elf::setSystemZTargetInfo(Ctx &ctx) { ctx.target.reset(new SystemZ(ctx)); }
