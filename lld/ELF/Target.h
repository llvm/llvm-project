//===- Target.h -------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_TARGET_H
#define LLD_ELF_TARGET_H

#include "InputSection.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELF.h"

#include <memory>

namespace lld {
namespace elf {
class InputFile;
class SymbolBody;

class TargetInfo {
public:
  virtual bool isTlsInitialExecRel(uint32_t Type) const;
  virtual bool isTlsLocalDynamicRel(uint32_t Type) const;
  virtual bool isTlsGlobalDynamicRel(uint32_t Type) const;
  virtual uint32_t getDynRel(uint32_t Type) const { return Type; }
  virtual void writeGotPltHeader(uint8_t *Buf) const {}
  virtual void writeGotPlt(uint8_t *Buf, const SymbolBody &S) const {};
  virtual uint64_t getImplicitAddend(const uint8_t *Buf, uint32_t Type) const;

  // If lazy binding is supported, the first entry of the PLT has code
  // to call the dynamic linker to resolve PLT entries the first time
  // they are called. This function writes that code.
  virtual void writePltHeader(uint8_t *Buf) const {}

  virtual void writePlt(uint8_t *Buf, uint64_t GotEntryAddr,
                        uint64_t PltEntryAddr, int32_t Index,
                        unsigned RelOff) const {}

  // Returns true if a relocation only uses the low bits of a value such that
  // all those bits are in in the same page. For example, if the relocation
  // only uses the low 12 bits in a system with 4k pages. If this is true, the
  // bits will always have the same value at runtime and we don't have to emit
  // a dynamic relocation.
  virtual bool usesOnlyLowPageBits(uint32_t Type) const;

  // Decide whether a Thunk is needed for the relocation from File
  // targeting S. Returns one of:
  // Expr if there is no Thunk required
  // R_THUNK_ABS if thunk is required and expression is absolute
  // R_THUNK_PC if thunk is required and expression is pc rel
  // R_THUNK_PLT_PC if thunk is required to PLT entry and expression is pc rel
  virtual RelExpr getThunkExpr(RelExpr Expr, uint32_t RelocType,
                               const InputFile &File,
                               const SymbolBody &S) const;
  virtual RelExpr getRelExpr(uint32_t Type, const SymbolBody &S) const = 0;
  virtual void relocateOne(uint8_t *Loc, uint32_t Type, uint64_t Val) const = 0;
  virtual ~TargetInfo();

  unsigned TlsGdRelaxSkip = 1;
  unsigned PageSize = 4096;

  // On freebsd x86_64 the first page cannot be mmaped.
  // On linux that is controled by vm.mmap_min_addr. At least on some x86_64
  // installs that is 65536, so the first 15 pages cannot be used.
  // Given that, the smallest value that can be used in here is 0x10000.
  // If using 2MB pages, the smallest page aligned address that works is
  // 0x200000, but it looks like every OS uses 4k pages for executables.
  uint64_t DefaultImageBase = 0x10000;

  uint32_t CopyRel;
  uint32_t GotRel;
  uint32_t PltRel;
  uint32_t RelativeRel;
  uint32_t IRelativeRel;
  uint32_t TlsDescRel;
  uint32_t TlsGotRel;
  uint32_t TlsModuleIndexRel;
  uint32_t TlsOffsetRel;
  unsigned GotEntrySize;
  unsigned GotPltEntrySize;
  unsigned PltEntrySize;
  unsigned PltHeaderSize;

  // At least on x86_64 positions 1 and 2 are used by the first plt entry
  // to support lazy loading.
  unsigned GotPltHeaderEntriesNum = 3;

  // Set to 0 for variant 2
  unsigned TcbSize = 0;

  virtual RelExpr adjustRelaxExpr(uint32_t Type, const uint8_t *Data,
                                  RelExpr Expr) const;
  virtual void relaxGot(uint8_t *Loc, uint64_t Val) const;
  virtual void relaxTlsGdToIe(uint8_t *Loc, uint32_t Type, uint64_t Val) const;
  virtual void relaxTlsGdToLe(uint8_t *Loc, uint32_t Type, uint64_t Val) const;
  virtual void relaxTlsIeToLe(uint8_t *Loc, uint32_t Type, uint64_t Val) const;
  virtual void relaxTlsLdToLe(uint8_t *Loc, uint32_t Type, uint64_t Val) const;
};

StringRef getRelName(uint32_t Type);
uint64_t getPPC64TocBase();

const unsigned MipsGPOffset = 0x7ff0;

extern TargetInfo *Target;
TargetInfo *createTarget();
}
}

#endif
