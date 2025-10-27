//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_RELOCSCAN_H
#define LLD_ELF_RELOCSCAN_H

#include "Config.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "Relocations.h"
#include "SyntheticSections.h"
#include "Target.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

namespace lld::elf {

// Build a bitmask with one bit set for each 64 subset of RelExpr.
inline constexpr uint64_t buildMask() { return 0; }

template <typename... Tails>
inline constexpr uint64_t buildMask(int head, Tails... tails) {
  return (0 <= head && head < 64 ? uint64_t(1) << head : 0) |
         buildMask(tails...);
}

// Return true if `Expr` is one of `Exprs`.
// There are more than 64 but less than 128 RelExprs, so we divide the set of
// exprs into [0, 64) and [64, 128) and represent each range as a constant
// 64-bit mask. Then we decide which mask to test depending on the value of
// expr and use a simple shift and bitwise-and to test for membership.
template <RelExpr... Exprs> bool oneof(RelExpr expr) {
  assert(0 <= expr && (int)expr < 128 &&
         "RelExpr is too large for 128-bit mask!");

  if (expr >= 64)
    return (uint64_t(1) << (expr - 64)) & buildMask((Exprs - 64)...);
  return (uint64_t(1) << expr) & buildMask(Exprs...);
}

// This class encapsulates states needed to scan relocations for one
// InputSectionBase.
class RelocScan {
public:
  Ctx &ctx;
  InputSectionBase *sec;

  RelocScan(Ctx &ctx, InputSectionBase *sec = nullptr) : ctx(ctx), sec(sec) {}
  template <class ELFT, class RelTy>
  void scan(typename Relocs<RelTy>::const_iterator &i, RelType type,
            int64_t addend);
  void scanEhSection(EhInputSection &s);

  template <class ELFT, class RelTy>
  int64_t getAddend(const RelTy &r, RelType type);
  bool maybeReportUndefined(Undefined &sym, uint64_t offset);
  bool checkTlsLe(uint64_t offset, Symbol &sym, RelType type);
  bool isStaticLinkTimeConstant(RelExpr e, RelType type, const Symbol &sym,
                                uint64_t relOff) const;
  void process(RelExpr expr, RelType type, uint64_t offset, Symbol &sym,
               int64_t addend) const;
  unsigned handleTlsRelocation(RelExpr expr, RelType type, uint64_t offset,
                               Symbol &sym, int64_t addend);
};

template <class ELFT, class RelTy>
int64_t RelocScan::getAddend(const RelTy &r, RelType type) {
  return RelTy::HasAddend ? elf::getAddend<ELFT>(r)
                          : ctx.target->getImplicitAddend(
                                sec->content().data() + r.r_offset, type);
}

template <class ELFT, class RelTy>
void RelocScan::scan(typename Relocs<RelTy>::const_iterator &it, RelType type,
                     int64_t addend) {
  const RelTy &rel = *it;
  uint32_t symIdx = rel.getSymbol(false);
  Symbol &sym = sec->getFile<ELFT>()->getSymbol(symIdx);
  uint64_t offset = rel.r_offset;
  RelExpr expr =
      ctx.target->getRelExpr(type, sym, sec->content().data() + offset);

  // Ignore R_*_NONE and other marker relocations.
  if (expr == R_NONE)
    return;

  // Error if the target symbol is undefined. Symbol index 0 may be used by
  // marker relocations, e.g. R_*_NONE and R_ARM_V4BX. Don't error on them.
  if (sym.isUndefined() && symIdx != 0 &&
      maybeReportUndefined(cast<Undefined>(sym), offset))
    return;

  // Ensure GOT or GOTPLT is created for relocations that reference their base
  // addresses without directly creating entries.
  if (oneof<R_GOTPLTONLY_PC, R_GOTPLTREL, R_GOTPLT, R_PLT_GOTPLT,
            R_TLSDESC_GOTPLT, R_TLSGD_GOTPLT>(expr)) {
    ctx.in.gotPlt->hasGotPltOffRel.store(true, std::memory_order_relaxed);
  } else if (oneof<R_GOTONLY_PC, R_GOTREL, RE_PPC32_PLTREL>(expr)) {
    ctx.in.got->hasGotOffRel.store(true, std::memory_order_relaxed);
  }

  // Process TLS relocations, including TLS optimizations. Note that
  // R_TPREL and R_TPREL_NEG relocations are resolved in processAux.
  //
  // Some RISCV TLSDESC relocations reference a local NOTYPE symbol,
  // but we need to process them in handleTlsRelocation.
  if (sym.isTls() || oneof<R_TLSDESC_PC, R_TLSDESC_CALL>(expr)) {
    if (unsigned processed =
            handleTlsRelocation(expr, type, offset, sym, addend)) {
      it += processed - 1;
      return;
    }
  }

  process(expr, type, offset, sym, addend);
}
} // namespace lld::elf

#endif
