//===- ARM64.cpp ----------------------------------------------------------===//
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm::MachO;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::macho;

namespace {

struct ARM64 : TargetInfo {
  ARM64();

  uint64_t getEmbeddedAddend(MemoryBufferRef, const section_64 &,
                             const relocation_info) const override;
  void relocateOne(uint8_t *loc, const Reloc &, uint64_t va,
                   uint64_t pc) const override;

  void writeStub(uint8_t *buf, const macho::Symbol &) const override;
  void writeStubHelperHeader(uint8_t *buf) const override;
  void writeStubHelperEntry(uint8_t *buf, const DylibSymbol &,
                            uint64_t entryAddr) const override;

  void relaxGotLoad(uint8_t *loc, uint8_t type) const override;
  const RelocAttrs &getRelocAttrs(uint8_t type) const override;
  uint64_t getPageSize() const override { return 16 * 1024; }
};

} // namespace

// Random notes on reloc types:
// ADDEND always pairs with BRANCH26, PAGE21, or PAGEOFF12
// POINTER_TO_GOT: ld64 supports a 4-byte pc-relative form as well as an 8-byte
// absolute version of this relocation. The semantics of the absolute relocation
// are weird -- it results in the value of the GOT slot being written, instead
// of the address. Let's not support it unless we find a real-world use case.

const RelocAttrs &ARM64::getRelocAttrs(uint8_t type) const {
  static const std::array<RelocAttrs, 11> relocAttrsArray{{
#define B(x) RelocAttrBits::x
      {"UNSIGNED", B(UNSIGNED) | B(ABSOLUTE) | B(EXTERN) | B(LOCAL) |
                       B(DYSYM8) | B(BYTE4) | B(BYTE8)},
      {"SUBTRACTOR", B(SUBTRAHEND) | B(BYTE4) | B(BYTE8)},
      {"BRANCH26", B(PCREL) | B(EXTERN) | B(BRANCH) | B(BYTE4)},
      {"PAGE21", B(PCREL) | B(EXTERN) | B(BYTE4)},
      {"PAGEOFF12", B(ABSOLUTE) | B(EXTERN) | B(BYTE4)},
      {"GOT_LOAD_PAGE21", B(PCREL) | B(EXTERN) | B(GOT) | B(BYTE4)},
      {"GOT_LOAD_PAGEOFF12",
       B(ABSOLUTE) | B(EXTERN) | B(GOT) | B(LOAD) | B(BYTE4)},
      {"POINTER_TO_GOT", B(PCREL) | B(EXTERN) | B(GOT) | B(POINTER) | B(BYTE4)},
      {"TLVP_LOAD_PAGE21", B(PCREL) | B(EXTERN) | B(TLV) | B(BYTE4)},
      {"TLVP_LOAD_PAGEOFF12",
       B(ABSOLUTE) | B(EXTERN) | B(TLV) | B(LOAD) | B(BYTE4)},
      {"ADDEND", B(ADDEND)},
#undef B
  }};
  assert(type < relocAttrsArray.size() && "invalid relocation type");
  if (type >= relocAttrsArray.size())
    return invalidRelocAttrs;
  return relocAttrsArray[type];
}

uint64_t ARM64::getEmbeddedAddend(MemoryBufferRef mb, const section_64 &sec,
                                  const relocation_info rel) const {
  if (rel.r_type != ARM64_RELOC_UNSIGNED) {
    // All other reloc types should use the ADDEND relocation to store their
    // addends.
    // TODO(gkm): extract embedded addend just so we can assert that it is 0
    return 0;
  }

  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  const uint8_t *loc = buf + sec.offset + rel.r_address;
  switch (rel.r_length) {
  case 2:
    return read32le(loc);
  case 3:
    return read64le(loc);
  default:
    llvm_unreachable("invalid r_length");
  }
}

inline uint64_t bitField(uint64_t value, int right, int width, int left) {
  return ((value >> right) & ((1 << width) - 1)) << left;
}

//              25                                                0
// +-----------+---------------------------------------------------+
// |           |                       imm26                       |
// +-----------+---------------------------------------------------+

inline uint64_t encodeBranch26(uint64_t base, uint64_t va) {
  // Since branch destinations are 4-byte aligned, the 2 least-
  // significant bits are 0. They are right shifted off the end.
  return (base | bitField(va, 2, 26, 0));
}

//   30 29          23                                  5
// +-+---+---------+-------------------------------------+---------+
// | |ilo|         |                immhi                |         |
// +-+---+---------+-------------------------------------+---------+

inline uint64_t encodePage21(uint64_t base, uint64_t va) {
  return (base | bitField(va, 12, 2, 29) | bitField(va, 14, 19, 5));
}

//                      21                   10
// +-------------------+-----------------------+-------------------+
// |                   |         imm12         |                   |
// +-------------------+-----------------------+-------------------+

inline uint64_t encodePageOff12(uint32_t base, uint64_t va) {
  int scale = 0;
  if ((base & 0x3b00'0000) == 0x3900'0000) { // load/store
    scale = base >> 30;
    if (scale == 0 && (base & 0x0480'0000) == 0x0480'0000) // 128-bit variant
      scale = 4;
  }

  // TODO(gkm): extract embedded addend and warn if != 0
  // uint64_t addend = ((base & 0x003FFC00) >> 10);
  return (base | bitField(va, scale, 12 - scale, 10));
}

inline uint64_t pageBits(uint64_t address) {
  const uint64_t pageMask = ~0xfffull;
  return address & pageMask;
}

// For instruction relocations (load, store, add), the base
// instruction is pre-populated in the text section. A pre-populated
// instruction has opcode & register-operand bits set, with immediate
// operands zeroed. We read it from text, OR-in the immediate
// operands, then write-back the completed instruction.

void ARM64::relocateOne(uint8_t *loc, const Reloc &r, uint64_t value,
                        uint64_t pc) const {
  uint32_t base = ((r.length == 2) ? read32le(loc) : 0);
  value += r.addend;
  switch (r.type) {
  case ARM64_RELOC_BRANCH26:
    value = encodeBranch26(base, value - pc);
    break;
  case ARM64_RELOC_UNSIGNED:
    break;
  case ARM64_RELOC_POINTER_TO_GOT:
    if (r.pcrel)
      value -= pc;
    break;
  case ARM64_RELOC_PAGE21:
  case ARM64_RELOC_GOT_LOAD_PAGE21:
  case ARM64_RELOC_TLVP_LOAD_PAGE21:
    assert(r.pcrel);
    value = encodePage21(base, pageBits(value) - pageBits(pc));
    break;
  case ARM64_RELOC_PAGEOFF12:
  case ARM64_RELOC_GOT_LOAD_PAGEOFF12:
  case ARM64_RELOC_TLVP_LOAD_PAGEOFF12:
    assert(!r.pcrel);
    value = encodePageOff12(base, value);
    break;
  default:
    llvm_unreachable("unexpected relocation type");
  }

  switch (r.length) {
  case 2:
    write32le(loc, value);
    break;
  case 3:
    write64le(loc, value);
    break;
  default:
    llvm_unreachable("invalid r_length");
  }
}

static constexpr uint32_t stubCode[] = {
    0x90000010, // 00: adrp  x16, __la_symbol_ptr@page
    0xf9400210, // 04: ldr   x16, [x16, __la_symbol_ptr@pageoff]
    0xd61f0200, // 08: br    x16
};

void ARM64::writeStub(uint8_t *buf8, const macho::Symbol &sym) const {
  auto *buf32 = reinterpret_cast<uint32_t *>(buf8);
  uint64_t pcPageBits =
      pageBits(in.stubs->addr + sym.stubsIndex * sizeof(stubCode));
  uint64_t lazyPointerVA = in.lazyPointers->addr + sym.stubsIndex * WordSize;
  buf32[0] = encodePage21(stubCode[0], pageBits(lazyPointerVA) - pcPageBits);
  buf32[1] = encodePageOff12(stubCode[1], lazyPointerVA);
  buf32[2] = stubCode[2];
}

static constexpr uint32_t stubHelperHeaderCode[] = {
    0x90000011, // 00: adrp  x17, _dyld_private@page
    0x91000231, // 04: add   x17, x17, _dyld_private@pageoff
    0xa9bf47f0, // 08: stp   x16/x17, [sp, #-16]!
    0x90000010, // 0c: adrp  x16, dyld_stub_binder@page
    0xf9400210, // 10: ldr   x16, [x16, dyld_stub_binder@pageoff]
    0xd61f0200, // 14: br    x16
};

void ARM64::writeStubHelperHeader(uint8_t *buf8) const {
  auto *buf32 = reinterpret_cast<uint32_t *>(buf8);
  auto pcPageBits = [](int i) {
    return pageBits(in.stubHelper->addr + i * sizeof(uint32_t));
  };
  uint64_t loaderVA = in.imageLoaderCache->getVA();
  buf32[0] =
      encodePage21(stubHelperHeaderCode[0], pageBits(loaderVA) - pcPageBits(0));
  buf32[1] = encodePageOff12(stubHelperHeaderCode[1], loaderVA);
  buf32[2] = stubHelperHeaderCode[2];
  uint64_t binderVA =
      in.got->addr + in.stubHelper->stubBinder->gotIndex * WordSize;
  buf32[3] =
      encodePage21(stubHelperHeaderCode[3], pageBits(binderVA) - pcPageBits(3));
  buf32[4] = encodePageOff12(stubHelperHeaderCode[4], binderVA);
  buf32[5] = stubHelperHeaderCode[5];
}

static constexpr uint32_t stubHelperEntryCode[] = {
    0x18000050, // 00: ldr  w16, l0
    0x14000000, // 04: b    stubHelperHeader
    0x00000000, // 08: l0: .long 0
};

void ARM64::writeStubHelperEntry(uint8_t *buf8, const DylibSymbol &sym,
                                 uint64_t entryVA) const {
  auto *buf32 = reinterpret_cast<uint32_t *>(buf8);
  auto pcVA = [entryVA](int i) { return entryVA + i * sizeof(uint32_t); };
  uint64_t stubHelperHeaderVA = in.stubHelper->addr;
  buf32[0] = stubHelperEntryCode[0];
  buf32[1] =
      encodeBranch26(stubHelperEntryCode[1], stubHelperHeaderVA - pcVA(1));
  buf32[2] = sym.lazyBindOffset;
}

void ARM64::relaxGotLoad(uint8_t *loc, uint8_t type) const {
  // The instruction format comments below are quoted from
  // Arm® Architecture Reference Manual
  // Armv8, for Armv8-A architecture profile
  // ARM DDI 0487G.a (ID011921)
  uint32_t instruction = read32le(loc);
  // C6.2.132 LDR (immediate)
  // LDR <Xt>, [<Xn|SP>{, #<pimm>}]
  if ((instruction & 0xffc00000) != 0xf9400000)
    error(getRelocAttrs(type).name + " reloc requires LDR instruction");
  assert(((instruction >> 10) & 0xfff) == 0 &&
         "non-zero embedded LDR immediate");
  // C6.2.4 ADD (immediate)
  // ADD <Xd|SP>, <Xn|SP>, #<imm>{, <shift>}
  instruction = ((instruction & 0x001fffff) | 0x91000000);
  write32le(loc, instruction);
}

ARM64::ARM64() {
  cpuType = CPU_TYPE_ARM64;
  cpuSubtype = CPU_SUBTYPE_ARM64_ALL;

  stubSize = sizeof(stubCode);
  stubHelperHeaderSize = sizeof(stubHelperHeaderCode);
  stubHelperEntrySize = sizeof(stubHelperEntryCode);
}

TargetInfo *macho::createARM64TargetInfo() {
  static ARM64 t;
  return &t;
}
