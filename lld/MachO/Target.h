//===- Target.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_TARGET_H
#define LLD_MACHO_TARGET_H

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
struct Reloc;

enum : uint64_t {
  // We are currently only supporting 64-bit targets since macOS and iOS are
  // deprecating 32-bit apps.
  WordSize = 8,
  PageZeroSize = 1ull << 32, // XXX should be 4096 for 32-bit targets
  MaxAlignmentPowerOf2 = 32,
};

enum class RelocAttrBits {
  _0 = 0,              // invalid
  PCREL = 1 << 0,      // Value is PC-relative offset
  ABSOLUTE = 1 << 1,   // Value is an absolute address or fixed offset
  BYTE4 = 1 << 2,      // 4 byte datum
  BYTE8 = 1 << 3,      // 8 byte datum
  EXTERN = 1 << 4,     // Can have an external symbol
  LOCAL = 1 << 5,      // Can have a local symbol
  ADDEND = 1 << 6,     // *_ADDEND paired prefix reloc
  SUBTRAHEND = 1 << 7, // *_SUBTRACTOR paired prefix reloc
  BRANCH = 1 << 8,     // Value is branch target
  GOT = 1 << 9,        // References a symbol in the Global Offset Table
  TLV = 1 << 10,       // References a thread-local symbol
  DYSYM8 = 1 << 11,    // Requires DySym width to be 8 bytes
  LOAD = 1 << 12,      // Relaxable indirect load
  POINTER = 1 << 13,   // Non-relaxable indirect load (pointer is taken)
  UNSIGNED = 1 << 14,  // *_UNSIGNED relocs
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue*/ (1 << 15) - 1),
};

class TargetInfo {
public:
  struct RelocAttrs {
    llvm::StringRef name;
    RelocAttrBits bits;
    bool hasAttr(RelocAttrBits b) const { return (bits & b) == b; }
  };
  static const RelocAttrs invalidRelocAttrs;

  virtual ~TargetInfo() = default;

  // Validate the relocation structure and get its addend.
  virtual uint64_t
  getEmbeddedAddend(llvm::MemoryBufferRef, const llvm::MachO::section_64 &,
                    const llvm::MachO::relocation_info) const = 0;
  virtual void relocateOne(uint8_t *loc, const Reloc &, uint64_t va,
                           uint64_t pc) const = 0;

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
  bool validateSymbolRelocation(const Symbol *, const InputSection *isec,
                                const Reloc &);
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
