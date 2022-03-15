//===- ARM64Common.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_ARCH_ARM64COMMON_H
#define LLD_MACHO_ARCH_ARM64COMMON_H

#include "InputFiles.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "llvm/BinaryFormat/MachO.h"

namespace lld {
namespace macho {

struct ARM64Common : TargetInfo {
  template <class LP> ARM64Common(LP lp) : TargetInfo(lp) {}

  int64_t getEmbeddedAddend(MemoryBufferRef, uint64_t offset,
                            const llvm::MachO::relocation_info) const override;
  void relocateOne(uint8_t *loc, const Reloc &, uint64_t va,
                   uint64_t pc) const override;

  void relaxGotLoad(uint8_t *loc, uint8_t type) const override;
  uint64_t getPageSize() const override { return 16 * 1024; }
};

inline uint64_t bitField(uint64_t value, int right, int width, int left) {
  return ((value >> right) & ((1 << width) - 1)) << left;
}

//              25                                                0
// +-----------+---------------------------------------------------+
// |           |                       imm26                       |
// +-----------+---------------------------------------------------+

inline void encodeBranch26(uint32_t *loc, const Reloc &r, uint32_t base,
                           uint64_t va) {
  checkInt(loc, r, va, 28);
  // Since branch destinations are 4-byte aligned, the 2 least-
  // significant bits are 0. They are right shifted off the end.
  llvm::support::endian::write32le(loc, base | bitField(va, 2, 26, 0));
}

inline void encodeBranch26(uint32_t *loc, SymbolDiagnostic d, uint32_t base,
                           uint64_t va) {
  checkInt(loc, d, va, 28);
  llvm::support::endian::write32le(loc, base | bitField(va, 2, 26, 0));
}

//   30 29          23                                  5
// +-+---+---------+-------------------------------------+---------+
// | |ilo|         |                immhi                |         |
// +-+---+---------+-------------------------------------+---------+

inline void encodePage21(uint32_t *loc, const Reloc &r, uint32_t base,
                         uint64_t va) {
  checkInt(loc, r, va, 35);
  llvm::support::endian::write32le(loc, base | bitField(va, 12, 2, 29) |
                                            bitField(va, 14, 19, 5));
}

inline void encodePage21(uint32_t *loc, SymbolDiagnostic d, uint32_t base,
                         uint64_t va) {
  checkInt(loc, d, va, 35);
  llvm::support::endian::write32le(loc, base | bitField(va, 12, 2, 29) |
                                            bitField(va, 14, 19, 5));
}

//                      21                   10
// +-------------------+-----------------------+-------------------+
// |                   |         imm12         |                   |
// +-------------------+-----------------------+-------------------+

inline void encodePageOff12(uint32_t *loc, uint32_t base, uint64_t va) {
  int scale = 0;
  if ((base & 0x3b00'0000) == 0x3900'0000) { // load/store
    scale = base >> 30;
    if (scale == 0 && (base & 0x0480'0000) == 0x0480'0000) // 128-bit variant
      scale = 4;
  }

  // TODO(gkm): extract embedded addend and warn if != 0
  // uint64_t addend = ((base & 0x003FFC00) >> 10);
  llvm::support::endian::write32le(loc,
                                   base | bitField(va, scale, 12 - scale, 10));
}

inline uint64_t pageBits(uint64_t address) {
  const uint64_t pageMask = ~0xfffull;
  return address & pageMask;
}

template <class LP>
inline void writeStub(uint8_t *buf8, const uint32_t stubCode[3],
                      const macho::Symbol &sym) {
  auto *buf32 = reinterpret_cast<uint32_t *>(buf8);
  constexpr size_t stubCodeSize = 3 * sizeof(uint32_t);
  uint64_t pcPageBits =
      pageBits(in.stubs->addr + sym.stubsIndex * stubCodeSize);
  uint64_t lazyPointerVA =
      in.lazyPointers->addr + sym.stubsIndex * LP::wordSize;
  encodePage21(&buf32[0], {&sym, "stub"}, stubCode[0],
               pageBits(lazyPointerVA) - pcPageBits);
  encodePageOff12(&buf32[1], stubCode[1], lazyPointerVA);
  buf32[2] = stubCode[2];
}

template <class LP>
inline void writeStubHelperHeader(uint8_t *buf8,
                                  const uint32_t stubHelperHeaderCode[6]) {
  auto *buf32 = reinterpret_cast<uint32_t *>(buf8);
  auto pcPageBits = [](int i) {
    return pageBits(in.stubHelper->addr + i * sizeof(uint32_t));
  };
  uint64_t loaderVA = in.imageLoaderCache->getVA();
  SymbolDiagnostic d = {nullptr, "stub header helper"};
  encodePage21(&buf32[0], d, stubHelperHeaderCode[0],
               pageBits(loaderVA) - pcPageBits(0));
  encodePageOff12(&buf32[1], stubHelperHeaderCode[1], loaderVA);
  buf32[2] = stubHelperHeaderCode[2];
  uint64_t binderVA =
      in.got->addr + in.stubHelper->stubBinder->gotIndex * LP::wordSize;
  encodePage21(&buf32[3], d, stubHelperHeaderCode[3],
               pageBits(binderVA) - pcPageBits(3));
  encodePageOff12(&buf32[4], stubHelperHeaderCode[4], binderVA);
  buf32[5] = stubHelperHeaderCode[5];
}

inline void writeStubHelperEntry(uint8_t *buf8,
                                 const uint32_t stubHelperEntryCode[3],
                                 const Symbol &sym, uint64_t entryVA) {
  auto *buf32 = reinterpret_cast<uint32_t *>(buf8);
  auto pcVA = [entryVA](int i) { return entryVA + i * sizeof(uint32_t); };
  uint64_t stubHelperHeaderVA = in.stubHelper->addr;
  buf32[0] = stubHelperEntryCode[0];
  encodeBranch26(&buf32[1], {&sym, "stub helper"}, stubHelperEntryCode[1],
                 stubHelperHeaderVA - pcVA(1));
  buf32[2] = sym.lazyBindOffset;
}

} // namespace macho
} // namespace lld

#endif
