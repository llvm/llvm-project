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
  uint64_t getVAStart() const;
  bool isTlsLocalDynamicRel(unsigned Type) const;
  bool isTlsGlobalDynamicRel(unsigned Type) const;
  virtual unsigned getDynRel(unsigned Type) const { return Type; }
  virtual bool isTlsDynRel(unsigned Type, const SymbolBody &S) const;
  virtual unsigned getTlsGotRel(unsigned Type = -1) const { return TlsGotRel; }
  virtual void writeGotHeader(uint8_t *Buf) const {}
  virtual void writeGotPltHeader(uint8_t *Buf) const {}
  virtual void writeGotPlt(uint8_t *Buf, uint64_t Plt) const {};

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

  virtual bool isSizeRel(uint32_t Type) const;
  virtual bool needsDynRelative(unsigned Type) const { return false; }
  virtual bool needsGot(uint32_t Type, const SymbolBody &S) const;
  virtual bool needsPlt(uint32_t Type, const SymbolBody &S) const;
  virtual void relocateOne(uint8_t *Loc, uint8_t *BufEnd, uint32_t Type,
                           uint64_t P, uint64_t SA, uint64_t ZA = 0,
                           uint8_t *PairedLoc = nullptr) const = 0;
  virtual bool isGotRelative(uint32_t Type) const;
  virtual bool canRelaxTls(unsigned Type, const SymbolBody *S) const;
  virtual bool needsCopyRel(uint32_t Type, const SymbolBody &S) const;
  virtual unsigned relaxTls(uint8_t *Loc, uint8_t *BufEnd, uint32_t Type,
                            uint64_t P, uint64_t SA, const SymbolBody *S) const;
  virtual ~TargetInfo();

  unsigned PageSize = 4096;

  // On freebsd x86_64 the first page cannot be mmaped.
  // On linux that is controled by vm.mmap_min_addr. At least on some x86_64
  // installs that is 65536, so the first 15 pages cannot be used.
  // Given that, the smallest value that can be used in here is 0x10000.
  // If using 2MB pages, the smallest page aligned address that works is
  // 0x200000, but it looks like every OS uses 4k pages for executables.
  uint64_t VAStart = 0x10000;

  unsigned CopyRel;
  unsigned GotRel;
  unsigned PltRel;
  unsigned RelativeRel;
  unsigned IRelativeRel;
  unsigned TlsGotRel = 0;
  unsigned TlsLocalDynamicRel = 0;
  unsigned TlsGlobalDynamicRel = 0;
  unsigned TlsModuleIndexRel;
  unsigned TlsOffsetRel;
  unsigned PltEntrySize = 8;
  unsigned PltZeroSize = 0;
  unsigned GotHeaderEntriesNum = 0;
  unsigned GotPltHeaderEntriesNum = 3;
  bool UseLazyBinding = false;
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
