//===- SuperH.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OutputSections.h"
#include "RelocScan.h"
#include "Relocations.h"
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
class SuperH : public TargetInfo {
public:
  SuperH(Ctx &);
  RelExpr getRelExpr(RelType type, const Symbol &s,
                             const uint8_t *loc) const override;
  int64_t getImplicitAddend(const uint8_t *buf, RelType type) const override;
  void relocate(uint8_t *loc, const Relocation &rel,
                        uint64_t val) const override;
  template <class ELFT, class RelTy>
  void scanSectionImpl(InputSectionBase &sec, Relocs<RelTy> rels,
                       unsigned shard);
  void scanSection(InputSectionBase &sec, unsigned shard) override {
    if (ctx.arg.ekind == ELF32BEKind)
      elf::scanSection1<SuperH, ELF32BE>(*this, sec, shard);
    else
      elf::scanSection1<SuperH, ELF32LE>(*this, sec, shard);
  }
};
} // namespace

SuperH::SuperH(Ctx &ctx) : TargetInfo(ctx) {
  copyRel = R_SH_COPY;
  symbolicRel = R_SH_DIR32;
  relativeRel = R_SH_RELATIVE;
  iRelativeRel = R_SH_RELATIVE;
  gotRel = R_SH_GLOB_DAT;
  pltRel = R_SH_JMP_SLOT;

  // GNU generally will insert NOPs instead of a trap instruction.
  // so we recreate that behaviour here.
  if (ctx.arg.ekind == ELF32BEKind)
    trapInstr = { 0x09, 0x00, 0x09, 0x00 };
  else
    trapInstr = { 0x00, 0x09, 0x00, 0x09 };
  
  defaultImageBase = 0x00010000;
}

RelExpr SuperH::getRelExpr(RelType type, const Symbol &s,
                            const uint8_t *loc) const {
  switch (type) {
  case R_SH_NONE:
    return R_NONE;
  case R_SH_DIR32:
    return R_ABS;
  case R_SH_REL32:
    return R_PC;
  default:
    Err(ctx) << getErrorLoc(ctx, loc) << "unknown relocation (" << type.v
             << ") against symbol " << &s;
    return R_NONE;
  }
}

int64_t SuperH::getImplicitAddend(const uint8_t *buf, RelType type) const {
  switch (type) {
  default:
    InternalErr(ctx, buf) << "cannot read addend for relocation " << type;
    return 0;
  case R_SH_DIR32:
  case R_SH_REL32:
  case R_SH_GOT32:
  case R_SH_PLT32:
  case R_SH_GOTOFF:
  case R_SH_GOTPC:
  case R_SH_GLOB_DAT:
  case R_SH_JMP_SLOT:
  case R_SH_RELATIVE:
    return SignExtend64<32>(read32(ctx, buf));
  case R_SH_NONE:
  case R_SH_COPY:
    return 0;
  }
}

template <class ELFT, class RelTy>
void SuperH::scanSectionImpl(InputSectionBase &sec, Relocs<RelTy> rels,
                              unsigned shard) {
  RelocScan rs(ctx, &sec, shard);
  sec.relocations.reserve(rels.size());
  for (auto it = rels.begin(); it != rels.end(); ++it) {
    const RelTy &rel = *it;
    RelType type = rel.getType(false);
    if (type == R_SH_NONE)
      continue;

    uint32_t symIdx = rel.getSymbol(false);
    uint64_t offset = rel.r_offset;
    Symbol &sym = sec.getFile<ELFT>()->getSymbol(symIdx);
    if (sym.isUndefined() && symIdx != 0 &&
        rs.maybeReportUndefined(cast<Undefined>(sym), offset))
      continue;
    int64_t addend = rs.getAddend<ELFT>(rel, type);

    RelExpr expr;
    switch (type) {
    case R_SH_DIR32:
      expr = R_ABS;
      break;

    case R_SH_REL32:
      rs.processR_PC(type, offset, addend, sym);
      continue;

    case R_SH_PLT32:
      rs.processR_PLT_PC(type, offset, addend, sym);
      continue;

    case R_SH_GOT32:
      expr = R_GOT_OFF;
      break;

    case R_SH_GOTOFF:
      ctx.in.got->hasGotOffRel.store(true, std::memory_order_relaxed);
      expr = R_GOTREL;
      break;

    case R_SH_GOTPC:
      expr = R_GOT_PC;
      break;

    default:
      Err(ctx) << getErrorLoc(ctx, sec.content().data() + offset)
               << "unknown relocation (" << type.v << ") against symbol "
               << &sym;
      continue;
    }
    rs.process(expr, type, offset, sym, addend);
  }

}

void SuperH::relocate(uint8_t *loc, const Relocation &rel,
                      uint64_t val) const {
  switch (rel.type) {
  case R_SH_DIR32:
  case R_SH_REL32:
  case R_SH_GOT32:
  case R_SH_PLT32:
  case R_SH_GOTOFF:
  case R_SH_GOTPC:
    write32(ctx, loc, val);
    break;
  default:
    llvm_unreachable("unknown relocation");
  }
}

void elf::setSuperHTargetInfo(Ctx &ctx) { ctx.target.reset(new SuperH(ctx)); }