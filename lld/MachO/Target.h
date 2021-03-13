//===- Target.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_TARGET_H
#define LLD_MACHO_TARGET_H

#include "Relocations.h"

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstddef>
#include <cstdint>

namespace lld {
namespace macho {
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

class Symbol;
class DylibSymbol;
class InputSection;

enum : uint64_t {
  // We are currently only supporting 64-bit targets since macOS and iOS are
  // deprecating 32-bit apps.
  WordSize = 8,
  PageZeroSize = 1ull << 32, // XXX should be 4096 for 32-bit targets
  MaxAlignmentPowerOf2 = 32,
};

class TargetInfo {
public:
  virtual ~TargetInfo() = default;

  // Validate the relocation structure and get its addend.
  virtual int64_t
  getEmbeddedAddend(llvm::MemoryBufferRef, const llvm::MachO::section_64 &,
                    const llvm::MachO::relocation_info) const = 0;
  virtual void relocateOne(uint8_t *loc, const Reloc &, uint64_t va,
                           uint64_t relocVA) const = 0;

  // Write code for lazy binding. See the comments on StubsSection for more
  // details.
  virtual void writeStub(uint8_t *buf, const Symbol &) const = 0;
  virtual void writeStubHelperHeader(uint8_t *buf) const = 0;
  virtual void writeStubHelperEntry(uint8_t *buf, const DylibSymbol &,
                                    uint64_t entryAddr) const = 0;

  // Symbols may be referenced via either the GOT or the stubs section,
  // depending on the relocation type. prepareSymbolRelocation() will set up the
  // GOT/stubs entries, and resolveSymbolVA() will return the addresses of those
  // entries. resolveSymbolVA() may also relax the target instructions to save
  // on a level of address indirection.
  virtual void relaxGotLoad(uint8_t *loc, uint8_t type) const = 0;

  virtual const RelocAttrs &getRelocAttrs(uint8_t type) const = 0;

  virtual uint64_t getPageSize() const = 0;

  bool hasAttr(uint8_t type, RelocAttrBits bit) const {
    return getRelocAttrs(type).hasAttr(bit);
  }

  bool validateRelocationInfo(llvm::MemoryBufferRef,
                              const llvm::MachO::section_64 &sec,
                              llvm::MachO::relocation_info);
  void prepareSymbolRelocation(Symbol *, const InputSection *, const Reloc &);

  uint32_t cpuType;
  uint32_t cpuSubtype;

  size_t stubSize;
  size_t stubHelperHeaderSize;
  size_t stubHelperEntrySize;
};

TargetInfo *createX86_64TargetInfo();
TargetInfo *createARM64TargetInfo();

extern TargetInfo *target;

} // namespace macho
} // namespace lld

#endif
