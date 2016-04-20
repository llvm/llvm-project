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
  uint64_t getVAStart() const;
  virtual bool isTlsInitialExecRel(uint32_t Type) const;
  virtual bool isTlsLocalDynamicRel(uint32_t Type) const;
  virtual bool isTlsGlobalDynamicRel(uint32_t Type) const;
  virtual uint32_t getDynRel(uint32_t Type) const { return Type; }
  virtual uint32_t getTlsGotRel(uint32_t Type) const { return TlsGotRel; }
  virtual void writeGotHeader(uint8_t *Buf) const {}
  virtual void writeGotPltHeader(uint8_t *Buf) const {}
  virtual void writeGotPlt(uint8_t *Buf, uint64_t Plt) const {};
  virtual uint64_t getImplicitAddend(const uint8_t *Buf, uint32_t Type) const;

  // If lazy binding is supported, the first entry of the PLT has code
  // to call the dynamic linker to resolve PLT entries the first time
  // they are called. This function writes that code.
  virtual void writePltZero(uint8_t *Buf) const {}

  virtual void writePlt(uint8_t *Buf, uint64_t GotEntryAddr,
                        uint64_t PltEntryAddr, int32_t Index,
                        unsigned RelOff) const {}

  // Returns true if a relocation is just a hint for linker to make for example
  // some code optimization. Such relocations should not be handled as a regular
  // ones and lead to dynamic relocation creation etc.
  virtual bool isHintRel(uint32_t Type) const;

  // Returns true if a relocation is relative to the place being relocated,
  // such as relocations used for PC-relative instructions. Such relocations
  // need not be fixed up if an image is loaded to a different address than
  // the link-time address. So we don't have to emit a relocation for the
  // dynamic linker if isRelRelative returns true.
  virtual bool isRelRelative(uint32_t Type) const;

  enum PltNeed { Plt_No, Plt_Explicit, Plt_Implicit };
  PltNeed needsPlt(uint32_t Type, const SymbolBody &S) const;

  virtual bool needsThunk(uint32_t Type, const InputFile &File,
                          const SymbolBody &S) const;

  virtual void writeThunk(uint8_t *Buf, uint64_t S) const {}

  virtual RelExpr getRelExpr(uint32_t Type, const SymbolBody &S) const = 0;
  virtual void relocateOne(uint8_t *Loc, uint32_t Type, uint64_t Val) const = 0;
  template <class ELFT>
  bool needsCopyRel(uint32_t Type, const SymbolBody &S) const;
  virtual ~TargetInfo();

  unsigned TlsGdToLeSkip = 1;
  unsigned PageSize = 4096;

  // On freebsd x86_64 the first page cannot be mmaped.
  // On linux that is controled by vm.mmap_min_addr. At least on some x86_64
  // installs that is 65536, so the first 15 pages cannot be used.
  // Given that, the smallest value that can be used in here is 0x10000.
  // If using 2MB pages, the smallest page aligned address that works is
  // 0x200000, but it looks like every OS uses 4k pages for executables.
  uint64_t VAStart = 0x10000;

  uint32_t CopyRel;
  uint32_t GotRel;
  uint32_t PltRel;
  uint32_t RelativeRel;
  uint32_t IRelativeRel;
  uint32_t TlsGotRel = 0;
  uint32_t TlsModuleIndexRel;
  uint32_t TlsOffsetRel;
  unsigned PltEntrySize = 8;
  unsigned PltZeroSize = 0;
  unsigned GotHeaderEntriesNum = 0;
  unsigned GotPltHeaderEntriesNum = 3;
  uint32_t ThunkSize = 0;
  bool UseLazyBinding = false;

  virtual void relaxTlsGdToIe(uint8_t *Loc, uint32_t Type, uint64_t Val) const;
  virtual void relaxTlsGdToLe(uint8_t *Loc, uint32_t Type, uint64_t Val) const;
  virtual void relaxTlsIeToLe(uint8_t *Loc, uint32_t Type, uint64_t Val) const;
  virtual void relaxTlsLdToLe(uint8_t *Loc, uint32_t Type, uint64_t Val) const;

private:
  virtual bool needsCopyRelImpl(uint32_t Type) const;
  virtual bool needsPltImpl(uint32_t Type) const;
};

uint64_t getPPC64TocBase();

const unsigned MipsGPOffset = 0x7ff0;
template <class ELFT> typename ELFT::uint getMipsGpAddr();

extern TargetInfo *Target;
TargetInfo *createTarget();
}
}

#endif
