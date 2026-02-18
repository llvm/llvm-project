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
  // Process relocation after needsGot/needsPlt flags are already handled.
  void processAux(RelExpr expr, RelType type, uint64_t offset, Symbol &sym,
                  int64_t addend) const;
  unsigned handleTlsRelocation(RelExpr expr, RelType type, uint64_t offset,
                               Symbol &sym, int64_t addend);

  // Process R_PC relocations. These are the most common relocation type, so we
  // inline the isStaticLinkTimeConstant check.
  void processR_PC(RelType type, uint64_t offset, int64_t addend, Symbol &sym) {
    if (LLVM_UNLIKELY(sym.isGnuIFunc()))
      sym.setFlags(HAS_DIRECT_RELOC);
    if (sym.isPreemptible || (isAbsolute(sym) && ctx.arg.isPic))
      processAux(R_PC, type, offset, sym, addend);
    else
      sec->addReloc({R_PC, type, offset, addend, &sym});
  }

  // Process R_PLT_PC relocations. These are very common (calls), so we inline
  // the isStaticLinkTimeConstant check. Non-preemptible symbols are optimized
  // to R_PC (direct call).
  void processR_PLT_PC(RelType type, uint64_t offset, int64_t addend,
                       Symbol &sym) {
    if (LLVM_UNLIKELY(sym.isGnuIFunc())) {
      process(R_PLT_PC, type, offset, sym, addend);
      return;
    }
    if (sym.isPreemptible) {
      sym.setFlags(NEEDS_PLT);
      sec->addReloc({R_PLT_PC, type, offset, addend, &sym});
    } else if (!(isAbsolute(sym) && ctx.arg.isPic)) {
      sec->addReloc({R_PC, type, offset, addend, &sym});
    } else {
      processAux(R_PC, type, offset, sym, addend);
    }
  }

  // Handle TLS Initial-Exec relocation.
  template <bool enableIeToLe = true>
  void handleTlsIe(RelExpr ieExpr, RelType type, uint64_t offset,
                   int64_t addend, Symbol &sym) {
    if (enableIeToLe && !ctx.arg.shared && !sym.isPreemptible) {
      // Optimize to Local Exec.
      sec->addReloc({R_TPREL, type, offset, addend, &sym});
    } else {
      sym.setFlags(NEEDS_TLSIE);
      // R_GOT (absolute GOT address) needs a RELATIVE dynamic relocation in
      // PIC when the relocation uses the full address (not just low page bits).
      if (ieExpr == R_GOT && ctx.arg.isPic &&
          !ctx.target->usesOnlyLowPageBits(type))
        sec->getPartition(ctx).relaDyn->addRelativeReloc(
            ctx.target->relativeRel, *sec, offset, sym, addend, type, ieExpr);
      else
        sec->addReloc({ieExpr, type, offset, addend, &sym});
    }
  }

  // Handle TLS Local-Dynamic relocation. Returns true if the __tls_get_addr
  // call should be skipped (i.e., caller should ++it).
  bool handleTlsLd(RelExpr sharedExpr, RelType type, uint64_t offset,
                   int64_t addend, Symbol &sym) {
    if (ctx.arg.shared) {
      ctx.needsTlsLd.store(true, std::memory_order_relaxed);
      sec->addReloc({sharedExpr, type, offset, addend, &sym});
      return false;
    }
    // Optimize to Local Exec.
    sec->addReloc({R_TPREL, type, offset, addend, &sym});
    return true;
  }

  // Handle TLS General-Dynamic relocation. Returns true if the __tls_get_addr
  // call should be skipped (i.e., caller should ++it).
  bool handleTlsGd(RelExpr sharedExpr, RelExpr ieExpr, RelExpr leExpr,
                   RelType type, uint64_t offset, int64_t addend, Symbol &sym) {
    if (ctx.arg.shared) {
      sym.setFlags(NEEDS_TLSGD);
      sec->addReloc({sharedExpr, type, offset, addend, &sym});
      return false;
    }
    if (sym.isPreemptible) {
      // Optimize to Initial Exec.
      sym.setFlags(NEEDS_TLSIE);
      sec->addReloc({ieExpr, type, offset, addend, &sym});
    } else {
      // Optimize to Local Exec.
      sec->addReloc({leExpr, type, offset, addend, &sym});
    }
    return true;
  }

  // Handle TLSDESC relocation.
  void handleTlsDesc(RelExpr sharedExpr, RelExpr ieExpr, RelType type,
                     uint64_t offset, int64_t addend, Symbol &sym) {
    if (ctx.arg.shared) {
      // NEEDS_TLSDESC_NONAUTH is a no-op for non-AArch64 targets and detects
      // incompatibility with NEEDS_TLSDESC_AUTH.
      sym.setFlags(NEEDS_TLSDESC | NEEDS_TLSDESC_NONAUTH);
      sec->addReloc({sharedExpr, type, offset, addend, &sym});
    } else if (sym.isPreemptible) {
      // Optimize to Initial Exec.
      sym.setFlags(NEEDS_TLSIE);
      sec->addReloc({ieExpr, type, offset, addend, &sym});
    } else {
      // Optimize to Local Exec.
      sec->addReloc({R_TPREL, type, offset, addend, &sym});
    }
  }
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
  if (oneof<R_GOTPLTREL, R_GOTPLT, R_TLSGD_GOTPLT>(expr)) {
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

// Dispatch to target-specific scanSectionImpl based on relocation format.
template <class Target, class ELFT>
void scanSection1(Target &target, InputSectionBase &sec) {
  const RelsOrRelas<ELFT> rels = sec.template relsOrRelas<ELFT>();
  if (rels.areRelocsCrel())
    target.template scanSectionImpl<ELFT>(sec, rels.crels);
  else if (rels.areRelocsRel())
    target.template scanSectionImpl<ELFT>(sec, rels.rels);
  else
    target.template scanSectionImpl<ELFT>(sec, rels.relas);
}

} // namespace lld::elf

#endif
