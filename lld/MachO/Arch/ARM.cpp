//===- ARM.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/Bitfields.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::macho;

namespace {

struct ARM : TargetInfo {
  ARM(uint32_t cpuSubtype);

  int64_t getEmbeddedAddend(MemoryBufferRef, uint64_t offset,
                            const relocation_info) const override;
  void relocateOne(uint8_t *loc, const Reloc &, uint64_t va,
                   uint64_t pc) const override;

  void writeStub(uint8_t *buf, const Symbol &) const override;
  void writeStubHelperHeader(uint8_t *buf) const override;
  void writeStubHelperEntry(uint8_t *buf, const DylibSymbol &,
                            uint64_t entryAddr) const override;

  void relaxGotLoad(uint8_t *loc, uint8_t type) const override;
  const RelocAttrs &getRelocAttrs(uint8_t type) const override;
  uint64_t getPageSize() const override { return 4 * 1024; }
};

} // namespace

const RelocAttrs &ARM::getRelocAttrs(uint8_t type) const {
  static const std::array<RelocAttrs, 10> relocAttrsArray{{
#define B(x) RelocAttrBits::x
      {"VANILLA", /* FIXME populate this */ B(_0)},
      {"PAIR", /* FIXME populate this */ B(_0)},
      {"SECTDIFF", /* FIXME populate this */ B(_0)},
      {"LOCAL_SECTDIFF", /* FIXME populate this */ B(_0)},
      {"PB_LA_PTR", /* FIXME populate this */ B(_0)},
      {"BR24", B(PCREL) | B(LOCAL) | B(EXTERN) | B(BRANCH) | B(BYTE4)},
      {"BR22", B(PCREL) | B(LOCAL) | B(EXTERN) | B(BRANCH) | B(BYTE4)},
      {"32BIT_BRANCH", /* FIXME populate this */ B(_0)},
      {"HALF", /* FIXME populate this */ B(_0)},
      {"HALF_SECTDIFF", /* FIXME populate this */ B(_0)},
#undef B
  }};
  assert(type < relocAttrsArray.size() && "invalid relocation type");
  if (type >= relocAttrsArray.size())
    return invalidRelocAttrs;
  return relocAttrsArray[type];
}

int64_t ARM::getEmbeddedAddend(MemoryBufferRef mb, uint64_t offset,
                               relocation_info rel) const {
  // FIXME: implement this
  return 0;
}

template <int N> using BitfieldFlag = Bitfield::Element<bool, N, 1>;

// ARM BL encoding:
//
// 30       28        24                                              0
// +---------+---------+----------------------------------------------+
// |  cond   | 1 0 1 1 |                  imm24                       |
// +---------+---------+----------------------------------------------+
//
// `cond` here varies depending on whether we have bleq, blne, etc.
// `imm24` encodes a 26-bit pcrel offset -- last 2 bits are zero as ARM
// functions are 4-byte-aligned.
//
// ARM BLX encoding:
//
// 30       28        24                                              0
// +---------+---------+----------------------------------------------+
// | 1 1 1 1 | 1 0 1 H |                  imm24                       |
// +---------+---------+----------------------------------------------+
//
// Since Thumb functions are 2-byte-aligned, we need one extra bit to encode
// the offset -- that is the H bit.
//
// BLX is always unconditional, so while we can convert directly from BLX to BL,
// we need to insert a shim if a BL's target is a Thumb function.
//
// Helper aliases for decoding BL / BLX:
using Cond = Bitfield::Element<uint32_t, 28, 4>;
using Imm24 = Bitfield::Element<int32_t, 0, 24>;

void ARM::relocateOne(uint8_t *loc, const Reloc &r, uint64_t value,
                      uint64_t pc) const {
  switch (r.type) {
  case ARM_RELOC_BR24: {
    uint32_t base = read32le(loc);
    bool isBlx = Bitfield::get<Cond>(base) == 0xf;
    const Symbol *sym = r.referent.get<Symbol *>();
    int32_t offset = value - (pc + 8);

    if (auto *defined = dyn_cast<Defined>(sym)) {
      if (!isBlx && defined->thumb) {
        error("TODO: implement interworking shim");
        return;
      } else if (isBlx && !defined->thumb) {
        Bitfield::set<Cond>(base, 0xe); // unconditional BL
        Bitfield::set<BitfieldFlag<24>>(base, true);
        isBlx = false;
      }
    } else {
      error("TODO: Implement ARM_RELOC_BR24 for dylib symbols");
      return;
    }

    if (isBlx) {
      assert((0x1 & value) == 0);
      Bitfield::set<Imm24>(base, offset >> 2);
      Bitfield::set<BitfieldFlag<24>>(base, (offset >> 1) & 1); // H bit
    } else {
      assert((0x3 & value) == 0);
      Bitfield::set<Imm24>(base, offset >> 2);
    }
    write32le(loc, base);
    break;
  }
  default:
    fatal("unhandled relocation type");
  }
}

void ARM::writeStub(uint8_t *buf, const Symbol &sym) const {
  fatal("TODO: implement this");
}

void ARM::writeStubHelperHeader(uint8_t *buf) const {
  fatal("TODO: implement this");
}

void ARM::writeStubHelperEntry(uint8_t *buf, const DylibSymbol &sym,
                               uint64_t entryAddr) const {
  fatal("TODO: implement this");
}

void ARM::relaxGotLoad(uint8_t *loc, uint8_t type) const {
  fatal("TODO: implement this");
}

ARM::ARM(uint32_t cpuSubtype) : TargetInfo(ILP32()) {
  cpuType = CPU_TYPE_ARM;
  this->cpuSubtype = cpuSubtype;

  stubSize = 0 /* FIXME */;
  stubHelperHeaderSize = 0 /* FIXME */;
  stubHelperEntrySize = 0 /* FIXME */;
}

TargetInfo *macho::createARMTargetInfo(uint32_t cpuSubtype) {
  static ARM t(cpuSubtype);
  return &t;
}
