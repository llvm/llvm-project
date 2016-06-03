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
template <class ELFT> class InputSection;
template <class ELFT> class InputSectionBase;

enum RelExpr {
  R_ABS,
  R_GOT,
  R_GOTONLY_PC,
  R_GOTREL,
  R_GOT_FROM_END,
  R_GOT_OFF,
  R_GOT_PAGE_PC,
  R_GOT_PC,
  R_HINT,
  R_MIPS_GOT_LOCAL,
  R_MIPS_GOT_LOCAL_PAGE,
  R_NEG_TLS,
  R_PAGE_PC,
  R_PC,
  R_PLT,
  R_PLT_PC,
  R_PPC_OPD,
  R_PPC_PLT_OPD,
  R_PPC_TOC,
  R_RELAX_GOT_PC,
  R_RELAX_GOT_PC_NOPIC,
  R_RELAX_TLS_GD_TO_IE,
  R_RELAX_TLS_GD_TO_LE,
  R_RELAX_TLS_IE_TO_LE,
  R_RELAX_TLS_LD_TO_LE,
  R_SIZE,
  R_THUNK,
  R_TLS,
  R_TLSDESC,
  R_TLSDESC_PAGE,
  R_TLSGD,
  R_TLSGD_PC,
  R_TLSLD,
  R_TLSLD_PC
};

struct Relocation {
  RelExpr Expr;
  uint32_t Type;
  uint64_t Offset;
  uint64_t Addend;
  SymbolBody *Sym;
};

template <class ELFT> void scanRelocations(InputSection<ELFT> &);

template <class ELFT>
void scanRelocations(InputSectionBase<ELFT> &, const typename ELFT::Shdr &);
}
}

#endif
