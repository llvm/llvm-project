//===- SPARCV9.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "RelocScan.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {
class SPARCV9 final : public TargetInfo {
public:
  SPARCV9(Ctx &);
  RelExpr getRelExpr(RelType type, const Symbol &s,
                     const uint8_t *loc) const override;
  RelType getDynRel(RelType type) const override;
  int64_t getImplicitAddend(const uint8_t *buf, RelType type) const override;
  void writeGotHeader(uint8_t *buf) const override;
  void writePlt(uint8_t *buf, const Symbol &sym,
                uint64_t pltEntryAddr) const override;
  void finalizeRelocScan() override;
  template <class ELFT, class RelTy>
  void scanSectionImpl(InputSectionBase &sec, Relocs<RelTy> rels,
                       unsigned shard);
  void scanSection(InputSectionBase &sec, unsigned shard) override {
    elf::scanSection1<SPARCV9, ELF64BE>(*this, sec, shard);
  }
  void relocate(uint8_t *loc, const Relocation &rel,
                uint64_t val) const override;
};
} // namespace

SPARCV9::SPARCV9(Ctx &ctx) : TargetInfo(ctx) {
  copyRel = R_SPARC_COPY;
  gotRel = R_SPARC_GLOB_DAT;
  pltRel = R_SPARC_JMP_SLOT;
  relativeRel = R_SPARC_RELATIVE;
  symbolicRel = R_SPARC_64;
  tlsGotRel = R_SPARC_TLS_TPOFF64;
  tlsModuleIndexRel = R_SPARC_TLS_DTPMOD64;
  tlsOffsetRel = R_SPARC_TLS_DTPOFF64;
  gotHeaderEntriesNum = 1;
  pltEntrySize = 32;
  pltHeaderSize = 4 * pltEntrySize;
  usesGotPlt = false;

  defaultCommonPageSize = 8192;
  defaultMaxPageSize = 0x100000;
  defaultImageBase = 0x100000;
}

// Only needed to support relocations used by relocateNonAlloc and
// preprocessRelocs.
RelExpr SPARCV9::getRelExpr(RelType type, const Symbol &s,
                            const uint8_t *loc) const {
  switch (type) {
  case R_SPARC_8:
  case R_SPARC_16:
  case R_SPARC_UA16:
  case R_SPARC_32:
  case R_SPARC_UA32:
  case R_SPARC_64:
  case R_SPARC_UA64:
    return R_ABS;
  case R_SPARC_DISP32:
    return R_PC;
  case R_SPARC_NONE:
    return R_NONE;
  default:
    Err(ctx) << getErrorLoc(ctx, loc) << "unknown relocation (" << type.v
             << ") against symbol " << &s;
    return R_NONE;
  }
}

RelType SPARCV9::getDynRel(RelType type) const {
  if (type == R_SPARC_64)
    return type;
  return R_SPARC_NONE;
}

int64_t SPARCV9::getImplicitAddend(const uint8_t *buf, RelType type) const {
  switch (type) {
  case R_SPARC_64:
  case R_SPARC_GLOB_DAT:
    return read64be(buf);
  default:
    InternalErr(ctx, buf) << "cannot read addend for relocation " << type;
    return 0;
  }
}

template <class ELFT, class RelTy>
void SPARCV9::scanSectionImpl(InputSectionBase &sec, Relocs<RelTy> rels,
                              unsigned shard) {
  RelocScan rs(ctx, &sec, shard);
  sec.relocations.reserve(rels.size());
  for (auto it = rels.begin(); it != rels.end(); ++it) {
    const RelTy &rel = *it;
    uint32_t symIdx = rel.getSymbol(false);
    Symbol &sym = sec.getFile<ELFT>()->getSymbol(symIdx);
    uint64_t offset = rel.r_offset;
    RelType type = rel.getType(false);
    if (sym.isUndefined() && symIdx != 0 &&
        rs.maybeReportUndefined(cast<Undefined>(sym), offset))
      continue;
    int64_t addend = rs.getAddend<ELFT>(rel, type);
    RelExpr expr;
    switch (type) {
    case R_SPARC_NONE:
    case R_SPARC_TLS_IE_ADD:
    case R_SPARC_TLS_LDO_ADD:
      continue;

    // Absolute relocations:
    case R_SPARC_8:
    case R_SPARC_16:
    case R_SPARC_UA16:
    case R_SPARC_32:
    case R_SPARC_UA32:
    case R_SPARC_64:
    case R_SPARC_UA64:
    case R_SPARC_H44:
    case R_SPARC_M44:
    case R_SPARC_L44:
    case R_SPARC_HH22:
    case R_SPARC_HM10:
    case R_SPARC_LM22:
    case R_SPARC_HI22:
    case R_SPARC_13:
    case R_SPARC_LO10:
    case R_SPARC_HIX22:
    case R_SPARC_LOX10:
      expr = R_ABS;
      break;

    // PLT-generating relocations:
    case R_SPARC_WPLT30:
      rs.processR_PLT_PC(type, offset, addend, sym);
      continue;

    // PC-relative relocations:
    case R_SPARC_DISP8:
    case R_SPARC_DISP16:
    case R_SPARC_DISP32:
    case R_SPARC_DISP64:
    case R_SPARC_PC10:
    case R_SPARC_PC22:
    case R_SPARC_WDISP16:
    case R_SPARC_WDISP19:
    case R_SPARC_WDISP22:
    case R_SPARC_WDISP30:
      rs.processR_PC(type, offset, addend, sym);
      continue;

    // GOT relocations:
    case R_SPARC_GOT10:
    case R_SPARC_GOT13:
    case R_SPARC_GOT22:
      expr = R_GOT_OFF;
      break;

    // Optimize the GOT load to an add of the symbol's GOT-relative address if
    // applicable. Exclude absolute symbol, which can be arbitrarily far. %l7
    // points at .got, which must be retained.
    case R_SPARC_GOTDATA_OP_HIX22:
    case R_SPARC_GOTDATA_OP_LOX10:
    case R_SPARC_GOTDATA_OP:
      if (sym.isPreemptible || isAbsolute(sym)) {
        expr = R_GOT_OFF;
      } else {
        ctx.in.got->hasGotOffRel.store(true, std::memory_order_relaxed);
        expr = R_GOTREL;
      }
      break;

    // TLS GD relocations. In an executable the sequence is optimized to
    // Initial Exec for a preemptible symbol and to Local Exec otherwise.
    case R_SPARC_TLS_GD_HI22:
    case R_SPARC_TLS_GD_LO10:
      rs.handleTlsGd(R_TLSGD_GOT, R_GOT_OFF, R_TPREL, type, offset, addend,
                     sym);
      continue;
    case R_SPARC_TLS_GD_ADD:
      // A marker on the add. R_ABS is a dummy for the unoptimized sequence and
      // writes nothing; an optimized one rewrites the instruction.
      rs.handleTlsGd(R_ABS, R_GOT_OFF, R_TPREL, type, offset, addend, sym);
      continue;
    case R_SPARC_TLS_GD_CALL:
      // The call names the TLS symbol rather than __tls_get_addr, so an
      // unoptimized call is rebound by finalizeRelocScan().
      rs.handleTlsGd(R_PLT_PC, R_GOT_OFF, R_TPREL, type, offset, addend, sym);
      continue;

    // TLS LD relocations. In an executable the sequence is optimized to
    // Local Exec.
    case R_SPARC_TLS_LDM_HI22:
    case R_SPARC_TLS_LDM_LO10:
      rs.handleTlsLd(R_TLSLD_GOT, type, offset, addend, sym);
      continue;
    case R_SPARC_TLS_LDM_ADD:
      rs.handleTlsLd(R_ABS, type, offset, addend, sym);
      continue;
    case R_SPARC_TLS_LDM_CALL:
      rs.handleTlsLd(R_PLT_PC, type, offset, addend, sym);
      continue;
    case R_SPARC_TLS_LDO_HIX22:
    case R_SPARC_TLS_LDO_LOX10:
      // @dtpoff is a non-negative offset into the module's TLS block, so it
      // uses the plain high/low split despite the HIX/LOX names. Local Exec
      // makes it the negative @tpoff, which needs the complement encoding.
      expr = ctx.arg.shared ? R_DTPREL : R_TPREL;
      break;

    // TLS LE relocations:
    case R_SPARC_TLS_LE_HIX22:
    case R_SPARC_TLS_LE_LOX10:
      if (rs.checkTlsLe(offset, sym, type))
        continue;
      expr = R_TPREL;
      break;

    // TLS IE relocations. In an executable, a non-preemptible symbol is
    // optimized to Local Exec: the add becomes an xor and the load becomes a
    // register move.
    case R_SPARC_TLS_IE_HI22:
    case R_SPARC_TLS_IE_LO10:
      rs.handleTlsIe(R_GOT_OFF, type, offset, addend, sym);
      continue;
    case R_SPARC_TLS_IE_LD:
    case R_SPARC_TLS_IE_LDX:
      if (!ctx.arg.shared && !sym.isPreemptible)
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

void SPARCV9::relocate(uint8_t *loc, const Relocation &rel,
                       uint64_t val) const {
  switch (rel.type) {
  case R_SPARC_8:
    // V-byte8
    checkIntUInt(ctx, loc, val, 8, rel);
    *loc = val;
    break;
  case R_SPARC_16:
  case R_SPARC_UA16:
    // V-half16
    checkIntUInt(ctx, loc, val, 16, rel);
    write16be(loc, val);
    break;
  case R_SPARC_32:
  case R_SPARC_UA32:
    // V-word32
    checkUInt(ctx, loc, val, 32, rel);
    write32be(loc, val);
    break;
  case R_SPARC_DISP8:
    // V-byte8
    checkInt(ctx, loc, val, 8, rel);
    *loc = val;
    break;
  case R_SPARC_DISP16:
    // V-half16
    checkInt(ctx, loc, val, 16, rel);
    write16be(loc, val);
    break;
  case R_SPARC_DISP32:
    // V-disp32
    checkInt(ctx, loc, val, 32, rel);
    write32be(loc, val);
    break;
  case R_SPARC_WDISP30:
  case R_SPARC_WPLT30:
    // V-disp30
    checkInt(ctx, loc, val, 32, rel);
    write32be(loc, (read32be(loc) & ~0x3fffffff) | ((val >> 2) & 0x3fffffff));
    break;
  case R_SPARC_22:
    // V-imm22
    checkUInt(ctx, loc, val, 22, rel);
    write32be(loc, (read32be(loc) & ~0x003fffff) | (val & 0x003fffff));
    break;
  case R_SPARC_13:
    // V-simm13
    checkIntUInt(ctx, loc, val, 13, rel);
    write32be(loc, (read32be(loc) & ~0x00001fff) | (val & 0x00001fff));
    break;
  case R_SPARC_GOT13:
    // V-simm13
    checkInt(ctx, loc, val, 13, rel);
    write32be(loc, (read32be(loc) & ~0x00001fff) | (val & 0x00001fff));
    break;
  case R_SPARC_GOT22:
  case R_SPARC_LM22:
    // T-imm22
    write32be(loc, (read32be(loc) & ~0x003fffff) | ((val >> 10) & 0x003fffff));
    break;
  case R_SPARC_PC22:
    // V-disp22
    checkIntUInt(ctx, loc, val, 32, rel);
    write32be(loc, (read32be(loc) & ~0x003fffff) | ((val >> 10) & 0x003fffff));
    break;
  case R_SPARC_HI22:
    // V-imm22
    checkUInt(ctx, loc, val, 32, rel);
    write32be(loc, (read32be(loc) & ~0x003fffff) | ((val >> 10) & 0x003fffff));
    break;
  case R_SPARC_WDISP22:
    // V-disp22
    checkInt(ctx, loc, val, 24, rel);
    write32be(loc, (read32be(loc) & ~0x003fffff) | ((val >> 2) & 0x003fffff));
    break;
  case R_SPARC_WDISP19:
    // V-disp19
    checkInt(ctx, loc, val, 21, rel);
    write32be(loc, (read32be(loc) & ~0x0007ffff) | ((val >> 2) & 0x0007ffff));
    break;
  case R_SPARC_WDISP16:
    // V-d2/disp14
    checkInt(ctx, loc, val, 18, rel);
    write32be(loc, (read32be(loc) & ~0x00303fff) |
                       (((val >> 2) & 0x0000c000) << 6) |
                       ((val >> 2) & 0x00003fff));
    break;
  case R_SPARC_GOT10:
  case R_SPARC_PC10:
    // T-simm10
    write32be(loc, (read32be(loc) & ~0x000003ff) | (val & 0x000003ff));
    break;
  case R_SPARC_LO10:
    // T-simm13
    write32be(loc, (read32be(loc) & ~0x000003ff) | (val & 0x000003ff));
    break;
  case R_SPARC_64:
  case R_SPARC_DISP64:
  case R_SPARC_UA64:
    // V-xword64
    write64be(loc, val);
    break;
  case R_SPARC_HH22:
    // V-imm22
    write32be(loc, (read32be(loc) & ~0x003fffff) | ((val >> 42) & 0x003fffff));
    break;
  case R_SPARC_HM10:
    // T-simm13
    write32be(loc, (read32be(loc) & ~0x000003ff) | ((val >> 32) & 0x000003ff));
    break;
  case R_SPARC_H44:
    // V-imm22
    checkUInt(ctx, loc, val, 44, rel);
    write32be(loc, (read32be(loc) & ~0x003fffff) | ((val >> 22) & 0x003fffff));
    break;
  case R_SPARC_M44:
    // T-imm10
    write32be(loc, (read32be(loc) & ~0x000003ff) | ((val >> 12) & 0x000003ff));
    break;
  case R_SPARC_L44:
    // T-imm13
    write32be(loc, (read32be(loc) & ~0x00000fff) | (val & 0x00000fff));
    break;
  case R_SPARC_HIX22:
    // V-imm22
    checkUInt(ctx, loc, ~val, 32, rel);
    write32be(loc, (read32be(loc) & ~0x003fffff) | ((~val >> 10) & 0x003fffff));
    break;
  case R_SPARC_TLS_LE_HIX22:
    // T-imm22
    write32be(loc, (read32be(loc) & ~0x003fffff) | ((~val >> 10) & 0x003fffff));
    break;
  case R_SPARC_LOX10:
  case R_SPARC_TLS_LE_LOX10:
    // T-simm13
    write32be(loc, (read32be(loc) & ~0x00001fff) | (val & 0x000003ff) | 0x1C00);
    break;
  case R_SPARC_GOTDATA_OP_HIX22: {
    // V-imm22. sethi encodes the complement of a negative value, which the
    // paired xor undoes, so the encodable range is 33 bits.
    checkInt(ctx, loc, val, 33, rel);
    uint64_t v = int64_t(val) < 0 ? ~val : val;
    write32be(loc, (read32be(loc) & ~0x003fffff) | ((v >> 10) & 0x003fffff));
    break;
  }
  case R_SPARC_GOTDATA_OP_LOX10:
    // T-simm13. Only a negative value needs the sign extension bits.
    write32be(loc, (read32be(loc) & ~0x00001fff) | (val & 0x000003ff) |
                       (int64_t(val) < 0 ? 0x1c00 : 0));
    break;
  case R_SPARC_GOTDATA_OP:
    // ldx [%rs1 + %rs2], %rd -> add %rs1, %rs2, %rd
    if (rel.expr == R_GOTREL)
      write32be(loc, (read32be(loc) & 0x3e07c01f) | 0x80000000);
    break;
  case R_SPARC_TLS_IE_HI22: {
    // T-imm22. Local Exec encodes the complement, as R_SPARC_TLS_LE_HIX22 does.
    uint64_t v = rel.expr == R_TPREL ? ~val : val;
    write32be(loc, (read32be(loc) & ~0x003fffff) | ((v >> 10) & 0x003fffff));
    break;
  }
  case R_SPARC_TLS_IE_LO10:
    if (rel.expr == R_TPREL)
      // add %rs1, imm, %rd -> xor %rs1, imm, %rd, T-simm13.
      write32be(loc, (read32be(loc) & ~0x00001fff) | 0x80182000 |
                         (val & 0x000003ff) | 0x1c00);
    else
      // T-simm10
      write32be(loc, (read32be(loc) & ~0x000003ff) | (val & 0x000003ff));
    break;
  case R_SPARC_TLS_IE_LD:
  case R_SPARC_TLS_IE_LDX: {
    // ld/ldx [%rs1 + %rs2], %rd -> mov %rs2, %rd, or nop if the move is
    // redundant. Only reached when the sequence is optimized to Local Exec.
    uint32_t insn = read32be(loc);
    write32be(loc, ((insn >> 25) & 0x1f) == (insn & 0x1f)
                       ? 0x01000000
                       : 0x80100000 | (insn & 0x3e00001f));
    break;
  }
  case R_SPARC_TLS_DTPMOD64:
  case R_SPARC_TLS_DTPOFF64:
  case R_SPARC_TLS_TPOFF64:
    // V-xword64. A GOT slot the link resolves: the module index of the output
    // module, or an offset within a module whose TLS block is known.
    write64be(loc, val);
    break;
  case R_SPARC_TLS_GD_HI22: {
    // T-imm22. Local Exec encodes the complement, as R_SPARC_TLS_LE_HIX22 does.
    uint64_t v = rel.expr == R_TPREL ? ~val : val;
    write32be(loc, (read32be(loc) & ~0x003fffff) | ((v >> 10) & 0x003fffff));
    break;
  }
  case R_SPARC_TLS_GD_LO10:
    if (rel.expr == R_TPREL)
      // add %rs1, imm, %rd -> xor %rs1, imm, %rd, T-simm13.
      write32be(loc, (read32be(loc) & ~0x00001fff) | 0x80182000 |
                         (val & 0x000003ff) | 0x1c00);
    else
      // T-simm10
      write32be(loc, (read32be(loc) & ~0x000003ff) | (val & 0x000003ff));
    break;
  case R_SPARC_TLS_GD_ADD:
    if (rel.expr == R_GOT_OFF)
      // Initial Exec: add %rs1, %rs2, %rd -> ldx [%rs1 + %rs2], %rd.
      write32be(loc, (read32be(loc) & 0x3e07c01f) | 0xc0000000 | (0x0b << 19));
    else if (rel.expr == R_TPREL)
      // Local Exec: the GOT pointer becomes the thread pointer, %rs1 -> %g7.
      write32be(loc, (read32be(loc) & ~0x0007c000) | (7 << 14));
    break;
  case R_SPARC_TLS_GD_CALL:
    if (rel.expr == R_GOT_OFF)
      write32be(loc, 0x9001c008); // add %g7, %o0, %o0
    else if (rel.expr == R_TPREL)
      write32be(loc, 0x01000000); // nop
    else
      // V-disp30, the call to __tls_get_addr.
      write32be(loc, (read32be(loc) & ~0x3fffffff) | ((val >> 2) & 0x3fffffff));
    break;
  case R_SPARC_TLS_LDM_HI22:
    if (rel.expr == R_TPREL)
      write32be(loc, 0x01000000); // nop
    else
      // T-imm22
      write32be(loc,
                (read32be(loc) & ~0x003fffff) | ((val >> 10) & 0x003fffff));
    break;
  case R_SPARC_TLS_LDM_LO10:
    if (rel.expr == R_TPREL)
      write32be(loc, 0x01000000); // nop
    else
      // T-simm10
      write32be(loc, (read32be(loc) & ~0x000003ff) | (val & 0x000003ff));
    break;
  case R_SPARC_TLS_LDM_ADD:
    if (rel.expr == R_TPREL)
      write32be(loc, 0x01000000); // nop
    break;
  case R_SPARC_TLS_LDM_CALL:
    if (rel.expr == R_TPREL)
      // Local Exec: the paired LDO add takes the thread pointer from %o0.
      write32be(loc, 0x90100007); // mov %g7, %o0
    else
      // V-disp30, the call to __tls_get_addr.
      write32be(loc, (read32be(loc) & ~0x3fffffff) | ((val >> 2) & 0x3fffffff));
    break;
  case R_SPARC_TLS_LDO_HIX22: {
    // T-imm22
    uint64_t v = rel.expr == R_TPREL ? ~val : val;
    write32be(loc, (read32be(loc) & ~0x003fffff) | ((v >> 10) & 0x003fffff));
    break;
  }
  case R_SPARC_TLS_LDO_LOX10:
    // T-simm13. Only the negative @tpoff needs the sign extension bits.
    write32be(loc, (read32be(loc) & ~0x00001fff) | (val & 0x000003ff) |
                       (rel.expr == R_TPREL ? 0x1c00 : 0));
    break;
  default:
    llvm_unreachable("unknown relocation");
  }
}

void SPARCV9::finalizeRelocScan() {
  Symbol *tga = nullptr;

  // R_SPARC_TLS_GD_CALL/LDM_CALL name the TLS symbol, not the callee. Rebind
  // the calls that survived optimization (recorded as R_PLT_PC by
  // scanSectionImpl) to __tls_get_addr. The symbol table cannot be reached
  // from scanSectionImpl, which runs in parallel.
  for (ELFFileBase *f : ctx.objectFiles) {
    for (InputSectionBase *s : f->getSections()) {
      auto *isec = dyn_cast_or_null<InputSection>(s);
      if (!isec || !isec->isLive())
        continue;
      for (Relocation &rel : isec->relocs()) {
        if (rel.expr != R_PLT_PC || (rel.type != R_SPARC_TLS_GD_CALL &&
                                     rel.type != R_SPARC_TLS_LDM_CALL))
          continue;
        if (!tga) {
          tga = ctx.symtab->addSymbol(Undefined{ctx.internalFile,
                                                "__tls_get_addr", STB_GLOBAL,
                                                STV_DEFAULT, STT_FUNC});
          tga->isUsedInRegularObj = true;
          tga->isPreemptible = true;
          tga->setFlags(NEEDS_PLT | USED);
        }
        rel.sym = tga;
      }
    }
  }
}

void SPARCV9::writeGotHeader(uint8_t *buf) const {
  // _GLOBAL_OFFSET_TABLE_[0] = _DYNAMIC
  write64be(buf, ctx.in.dynamic->getVA());
}

void SPARCV9::writePlt(uint8_t *buf, const Symbol & /*sym*/,
                       uint64_t pltEntryAddr) const {
  const uint8_t pltData[] = {
      0x03, 0x00, 0x00, 0x00, // sethi   (. - .PLT0), %g1
      0x30, 0x68, 0x00, 0x00, // ba,a    %xcc, .PLT1
      0x01, 0x00, 0x00, 0x00, // nop
      0x01, 0x00, 0x00, 0x00, // nop
      0x01, 0x00, 0x00, 0x00, // nop
      0x01, 0x00, 0x00, 0x00, // nop
      0x01, 0x00, 0x00, 0x00, // nop
      0x01, 0x00, 0x00, 0x00  // nop
  };
  memcpy(buf, pltData, sizeof(pltData));

  uint64_t off = pltEntryAddr - ctx.in.plt->getVA();
  relocateNoSym(buf, R_SPARC_22, off);
  relocateNoSym(buf + 4, R_SPARC_WDISP19, -(off + 4 - pltEntrySize));
}

void elf::setSPARCV9TargetInfo(Ctx &ctx) { ctx.target.reset(new SPARCV9(ctx)); }
