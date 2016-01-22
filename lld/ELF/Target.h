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

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELF.h"

#include <memory>

namespace lld {
namespace elf2 {
class SymbolBody;

class TargetInfo {
public:
  unsigned getPageSize() const { return PageSize; }
  uint64_t getVAStart() const;
  unsigned getCopyReloc() const { return CopyReloc; }
  unsigned getGotReloc() const { return GotReloc; }
  unsigned getPltReloc() const { return PltReloc; }
  unsigned getRelativeReloc() const { return RelativeReloc; }
  unsigned getIRelativeReloc() const { return IRelativeReloc; }
  bool isTlsLocalDynamicReloc(unsigned Type) const {
    return Type == TlsLocalDynamicReloc;
  }
  bool isTlsGlobalDynamicReloc(unsigned Type) const {
    return Type == TlsGlobalDynamicReloc;
  }
  unsigned getTlsModuleIndexReloc() const { return TlsModuleIndexReloc; }
  unsigned getTlsOffsetReloc() const { return TlsOffsetReloc; }
  unsigned getPltZeroEntrySize() const { return PltZeroEntrySize; }
  unsigned getPltEntrySize() const { return PltEntrySize; }
  bool supportsLazyRelocations() const { return LazyRelocations; }
  unsigned getGotHeaderEntriesNum() const { return GotHeaderEntriesNum; }
  unsigned getGotPltHeaderEntriesNum() const { return GotPltHeaderEntriesNum; }
  virtual unsigned getDynReloc(unsigned Type) const { return Type; }
  virtual bool isTlsDynReloc(unsigned Type, const SymbolBody &S) const {
    return false;
  }
  virtual unsigned getTlsGotReloc(unsigned Type = -1) const {
    return TlsGotReloc;
  }
  virtual void writeGotHeaderEntries(uint8_t *Buf) const;
  virtual void writeGotPltHeaderEntries(uint8_t *Buf) const;
  virtual void writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const = 0;
  virtual void writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                 uint64_t PltEntryAddr) const = 0;
  virtual void writePltEntry(uint8_t *Buf, uint64_t GotAddr,
                             uint64_t GotEntryAddr, uint64_t PltEntryAddr,
                             int32_t Index, unsigned RelOff) const = 0;

  // Returns true if a relocation is just a hint for linker to make for example
  // some code optimization. Such relocations should not be handled as a regular
  // ones and lead to dynamic relocation creation etc.
  virtual bool isHintReloc(uint32_t Type) const;

  // Returns true if a relocation is relative to the place being relocated,
  // such as relocations used for PC-relative instructions. Such relocations
  // need not be fixed up if an image is loaded to a different address than
  // the link-time address. So we don't have to emit a relocation for the
  // dynamic linker if isRelRelative returns true.
  virtual bool isRelRelative(uint32_t Type) const;

  virtual bool isSizeReloc(uint32_t Type) const;
  virtual bool relocNeedsDynRelative(unsigned Type) const { return false; }
  virtual bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const = 0;
  virtual bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const = 0;
  virtual void relocateOne(uint8_t *Loc, uint8_t *BufEnd, uint32_t Type,
                           uint64_t P, uint64_t SA, uint64_t ZA = 0,
                           uint8_t *PairedLoc = nullptr) const = 0;
  virtual bool isGotRelative(uint32_t Type) const;
  virtual bool isTlsOptimized(unsigned Type, const SymbolBody *S) const;
  virtual bool needsCopyRel(uint32_t Type, const SymbolBody &S) const;
  virtual unsigned relocateTlsOptimize(uint8_t *Loc, uint8_t *BufEnd,
                                       uint32_t Type, uint64_t P, uint64_t SA,
                                       const SymbolBody &S) const;
  virtual ~TargetInfo();

protected:
  unsigned PageSize = 4096;

  // On freebsd x86_64 the first page cannot be mmaped.
  // On linux that is controled by vm.mmap_min_addr. At least on some x86_64
  // installs that is 65536, so the first 15 pages cannot be used.
  // Given that, the smallest value that can be used in here is 0x10000.
  // If using 2MB pages, the smallest page aligned address that works is
  // 0x200000, but it looks like every OS uses 4k pages for executables.
  uint64_t VAStart = 0x10000;

  unsigned CopyReloc;
  unsigned PCRelReloc;
  unsigned GotReloc;
  unsigned PltReloc;
  unsigned RelativeReloc;
  unsigned IRelativeReloc;
  unsigned TlsGotReloc = 0;
  unsigned TlsLocalDynamicReloc = 0;
  unsigned TlsGlobalDynamicReloc = 0;
  unsigned TlsModuleIndexReloc;
  unsigned TlsOffsetReloc;
  unsigned PltEntrySize = 8;
  unsigned PltZeroEntrySize = 0;
  unsigned GotHeaderEntriesNum = 0;
  unsigned GotPltHeaderEntriesNum = 3;
  bool LazyRelocations = false;
};

uint64_t getPPC64TocBase();

template <class ELFT>
typename llvm::object::ELFFile<ELFT>::uintX_t getMipsGpAddr();

// Returns true if the relocation requires entry in the local part of GOT.
bool needsMipsLocalGot(uint32_t Type, SymbolBody *Body);

template <class ELFT> bool isGnuIFunc(const SymbolBody &S);

extern std::unique_ptr<TargetInfo> Target;
TargetInfo *createTarget();
}
}

#endif
