//===- Relocations.h -------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_RELOCATIONS_H
#define LLD_ELF_RELOCATIONS_H

#include "lld/Core/LLVM.h"

namespace lld {
namespace elf {
class SymbolBody;
class InputSectionData;
template <class ELFT> class InputSection;
template <class ELFT> class InputSectionBase;
class OutputSectionBase;

// List of target-independent relocation types. Relocations read
// from files are converted to these types so that the main code
// doesn't have to know about architecture-specific details.
enum RelExpr {
  R_ABS,
  R_GOT,
  R_GOTONLY_PC,
  R_GOTONLY_PC_FROM_END,
  R_GOTREL,
  R_GOTREL_FROM_END,
  R_GOT_FROM_END,
  R_GOT_OFF,
  R_GOT_PAGE_PC,
  R_GOT_PC,
  R_HINT,
  R_MIPS_GOT_LOCAL_PAGE,
  R_MIPS_GOT_OFF,
  R_MIPS_GOT_OFF32,
  R_MIPS_GOTREL,
  R_MIPS_TLSGD,
  R_MIPS_TLSLD,
  R_NEG_TLS,
  R_PAGE_PC,
  R_PC,
  R_PLT,
  R_PLT_PC,
  R_PLT_PAGE_PC,
  R_PPC_OPD,
  R_PPC_PLT_OPD,
  R_PPC_TOC,
  R_RELAX_GOT_PC,
  R_RELAX_GOT_PC_NOPIC,
  R_RELAX_TLS_GD_TO_IE,
  R_RELAX_TLS_GD_TO_IE_END,
  R_RELAX_TLS_GD_TO_IE_ABS,
  R_RELAX_TLS_GD_TO_IE_PAGE_PC,
  R_RELAX_TLS_GD_TO_LE,
  R_RELAX_TLS_GD_TO_LE_NEG,
  R_RELAX_TLS_IE_TO_LE,
  R_RELAX_TLS_LD_TO_LE,
  R_SIZE,
  R_TLS,
  R_TLSDESC,
  R_TLSDESC_PAGE,
  R_TLSDESC_CALL,
  R_TLSGD,
  R_TLSGD_PC,
  R_TLSLD,
  R_TLSLD_PC,
};

// Build a bitmask with one bit set for each RelExpr.
//
// Constexpr function arguments can't be used in static asserts, so we
// use template arguments to build the mask.
// But function template partial specializations don't exist (needed
// for base case of the recursion), so we need a dummy struct.
template <RelExpr... Exprs> struct RelExprMaskBuilder {
  static inline uint64_t build() { return 0; }
};

// Specialization for recursive case.
template <RelExpr Head, RelExpr... Tail>
struct RelExprMaskBuilder<Head, Tail...> {
  static inline uint64_t build() {
    static_assert(0 <= Head && Head < 64,
                  "RelExpr is too large for 64-bit mask!");
    return (uint64_t(1) << Head) | RelExprMaskBuilder<Tail...>::build();
  }
};

// Return true if `Expr` is one of `Exprs`.
// There are fewer than 64 RelExpr's, so we can represent any set of
// RelExpr's as a constant bit mask and test for membership with a
// couple cheap bitwise operations.
template <RelExpr... Exprs> bool isRelExprOneOf(RelExpr Expr) {
  assert(0 <= Expr && (int)Expr < 64 && "RelExpr is too large for 64-bit mask!");
  return (uint64_t(1) << Expr) & RelExprMaskBuilder<Exprs...>::build();
}

// Architecture-neutral representation of relocation.
struct Relocation {
  RelExpr Expr;
  uint32_t Type;
  uint64_t Offset;
  uint64_t Addend;
  SymbolBody *Sym;
};

template <class ELFT> void scanRelocations(InputSectionBase<ELFT> &);

template <class ELFT>
void createThunks(ArrayRef<OutputSectionBase *> OutputSections);

template <class ELFT>
static inline typename ELFT::uint getAddend(const typename ELFT::Rel &Rel) {
  return 0;
}

template <class ELFT>
static inline typename ELFT::uint getAddend(const typename ELFT::Rela &Rel) {
  return Rel.r_addend;
}
}
}

#endif
