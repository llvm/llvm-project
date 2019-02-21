//===- Relocations.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_RELOCATIONS_H
#define LLD_ELF_RELOCATIONS_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include <map>
#include <vector>

namespace lld {
namespace elf {
class Symbol;
class InputSection;
class InputSectionBase;
class OutputSection;
class SectionBase;

// Represents a relocation type, such as R_X86_64_PC32 or R_ARM_THM_CALL.
typedef uint32_t RelType;

// List of target-independent relocation types. Relocations read
// from files are converted to these types so that the main code
// doesn't have to know about architecture-specific details.
enum RelExpr {
  R_ABS,
  R_ADDEND,
  R_GOT,
  R_GOTONLY_PC,
  R_GOTONLY_PC_FROM_END,
  R_GOTREL,
  R_GOTREL_FROM_END,
  R_GOT_FROM_END,
  R_GOT_OFF,
  R_GOT_PC,
  R_HINT,
  R_NEG_TLS,
  R_NONE,
  R_PC,
  R_PLT,
  R_PLT_PC,
  R_RELAX_GOT_PC,
  R_RELAX_GOT_PC_NOPIC,
  R_RELAX_TLS_GD_TO_IE,
  R_RELAX_TLS_GD_TO_IE_ABS,
  R_RELAX_TLS_GD_TO_IE_END,
  R_RELAX_TLS_GD_TO_IE_GOT_OFF,
  R_RELAX_TLS_GD_TO_LE,
  R_RELAX_TLS_GD_TO_LE_NEG,
  R_RELAX_TLS_IE_TO_LE,
  R_RELAX_TLS_LD_TO_LE,
  R_RELAX_TLS_LD_TO_LE_ABS,
  R_SIZE,
  R_TLS,
  R_TLSDESC,
  R_TLSDESC_CALL,
  R_TLSGD_GOT,
  R_TLSGD_GOT_FROM_END,
  R_TLSGD_PC,
  R_TLSIE_HINT,
  R_TLSLD_GOT,
  R_TLSLD_GOT_FROM_END,
  R_TLSLD_GOT_OFF,
  R_TLSLD_HINT,
  R_TLSLD_PC,

  // The following is abstract relocation types used for only one target.
  //
  // Even though RelExpr is intended to be a target-neutral representation
  // of a relocation type, there are some relocations whose semantics are
  // unique to a target. Such relocation are marked with R_<TARGET_NAME>.
  R_AARCH64_GOT_PAGE_PC,
  R_AARCH64_PAGE_PC,
  R_AARCH64_RELAX_TLS_GD_TO_IE_PAGE_PC,
  R_AARCH64_TLSDESC_PAGE,
  R_ARM_SBREL,
  R_HEXAGON_GOT,
  R_MIPS_GOTREL,
  R_MIPS_GOT_GP,
  R_MIPS_GOT_GP_PC,
  R_MIPS_GOT_LOCAL_PAGE,
  R_MIPS_GOT_OFF,
  R_MIPS_GOT_OFF32,
  R_MIPS_TLSGD,
  R_MIPS_TLSLD,
  R_PPC_CALL,
  R_PPC_CALL_PLT,
  R_PPC_TOC,
  R_RISCV_PC_INDIRECT,
};

// Architecture-neutral representation of relocation.
struct Relocation {
  RelExpr Expr;
  RelType Type;
  uint64_t Offset;
  int64_t Addend;
  Symbol *Sym;
};

template <class ELFT> void scanRelocations(InputSectionBase &);

void addIRelativeRelocs();

class ThunkSection;
class Thunk;
struct InputSectionDescription;

class ThunkCreator {
public:
  // Return true if Thunks have been added to OutputSections
  bool createThunks(ArrayRef<OutputSection *> OutputSections);

  // The number of completed passes of createThunks this permits us
  // to do one time initialization on Pass 0 and put a limit on the
  // number of times it can be called to prevent infinite loops.
  uint32_t Pass = 0;

private:
  void mergeThunks(ArrayRef<OutputSection *> OutputSections);

  ThunkSection *getISDThunkSec(OutputSection *OS, InputSection *IS,
                               InputSectionDescription *ISD, uint32_t Type,
                               uint64_t Src);

  ThunkSection *getISThunkSec(InputSection *IS);

  void createInitialThunkSections(ArrayRef<OutputSection *> OutputSections);

  std::pair<Thunk *, bool> getThunk(Symbol &Sym, RelType Type, uint64_t Src);

  ThunkSection *addThunkSection(OutputSection *OS, InputSectionDescription *,
                                uint64_t Off);

  bool normalizeExistingThunk(Relocation &Rel, uint64_t Src);

  // Record all the available Thunks for a Symbol
  llvm::DenseMap<std::pair<SectionBase *, uint64_t>, std::vector<Thunk *>>
      ThunkedSymbolsBySection;
  llvm::DenseMap<Symbol *, std::vector<Thunk *>> ThunkedSymbols;

  // Find a Thunk from the Thunks symbol definition, we can use this to find
  // the Thunk from a relocation to the Thunks symbol definition.
  llvm::DenseMap<Symbol *, Thunk *> Thunks;

  // Track InputSections that have an inline ThunkSection placed in front
  // an inline ThunkSection may have control fall through to the section below
  // so we need to make sure that there is only one of them.
  // The Mips LA25 Thunk is an example of an inline ThunkSection.
  llvm::DenseMap<InputSection *, ThunkSection *> ThunkedSections;
};

// Return a int64_t to make sure we get the sign extension out of the way as
// early as possible.
template <class ELFT>
static inline int64_t getAddend(const typename ELFT::Rel &Rel) {
  return 0;
}
template <class ELFT>
static inline int64_t getAddend(const typename ELFT::Rela &Rel) {
  return Rel.r_addend;
}
} // namespace elf
} // namespace lld

#endif
