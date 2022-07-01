//===- ARM64.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Arch/ARM64Common.h"
#include "InputFiles.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "lld/Common/ErrorHandler.h"
#include "mach-o/compact_unwind_encoding.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::macho;

namespace {

struct ARM64 : ARM64Common {
  ARM64();
  void writeStub(uint8_t *buf, const Symbol &) const override;
  void writeStubHelperHeader(uint8_t *buf) const override;
  void writeStubHelperEntry(uint8_t *buf, const Symbol &,
                            uint64_t entryAddr) const override;
  const RelocAttrs &getRelocAttrs(uint8_t type) const override;
  void populateThunk(InputSection *thunk, Symbol *funcSym) override;
  void applyOptimizationHints(uint8_t *, const ConcatInputSection *,
                              ArrayRef<uint64_t>) const override;
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
      {"UNSIGNED",
       B(UNSIGNED) | B(ABSOLUTE) | B(EXTERN) | B(LOCAL) | B(BYTE4) | B(BYTE8)},
      {"SUBTRACTOR", B(SUBTRAHEND) | B(EXTERN) | B(BYTE4) | B(BYTE8)},
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

static constexpr uint32_t stubCode[] = {
    0x90000010, // 00: adrp  x16, __la_symbol_ptr@page
    0xf9400210, // 04: ldr   x16, [x16, __la_symbol_ptr@pageoff]
    0xd61f0200, // 08: br    x16
};

void ARM64::writeStub(uint8_t *buf8, const Symbol &sym) const {
  ::writeStub<LP64>(buf8, stubCode, sym);
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
  ::writeStubHelperHeader<LP64>(buf8, stubHelperHeaderCode);
}

static constexpr uint32_t stubHelperEntryCode[] = {
    0x18000050, // 00: ldr  w16, l0
    0x14000000, // 04: b    stubHelperHeader
    0x00000000, // 08: l0: .long 0
};

void ARM64::writeStubHelperEntry(uint8_t *buf8, const Symbol &sym,
                                 uint64_t entryVA) const {
  ::writeStubHelperEntry(buf8, stubHelperEntryCode, sym, entryVA);
}

// A thunk is the relaxed variation of stubCode. We don't need the
// extra indirection through a lazy pointer because the target address
// is known at link time.
static constexpr uint32_t thunkCode[] = {
    0x90000010, // 00: adrp  x16, <thunk.ptr>@page
    0x91000210, // 04: add   x16, [x16,<thunk.ptr>@pageoff]
    0xd61f0200, // 08: br    x16
};

void ARM64::populateThunk(InputSection *thunk, Symbol *funcSym) {
  thunk->align = 4;
  thunk->data = {reinterpret_cast<const uint8_t *>(thunkCode),
                 sizeof(thunkCode)};
  thunk->relocs.push_back({/*type=*/ARM64_RELOC_PAGEOFF12,
                           /*pcrel=*/false, /*length=*/2,
                           /*offset=*/4, /*addend=*/0,
                           /*referent=*/funcSym});
  thunk->relocs.push_back({/*type=*/ARM64_RELOC_PAGE21,
                           /*pcrel=*/true, /*length=*/2,
                           /*offset=*/0, /*addend=*/0,
                           /*referent=*/funcSym});
}

ARM64::ARM64() : ARM64Common(LP64()) {
  cpuType = CPU_TYPE_ARM64;
  cpuSubtype = CPU_SUBTYPE_ARM64_ALL;

  stubSize = sizeof(stubCode);
  thunkSize = sizeof(thunkCode);

  // Branch immediate is two's complement 26 bits, which is implicitly
  // multiplied by 4 (since all functions are 4-aligned: The branch range
  // is -4*(2**(26-1))..4*(2**(26-1) - 1).
  backwardBranchRange = 128 * 1024 * 1024;
  forwardBranchRange = backwardBranchRange - 4;

  modeDwarfEncoding = UNWIND_ARM64_MODE_DWARF;
  subtractorRelocType = ARM64_RELOC_SUBTRACTOR;
  unsignedRelocType = ARM64_RELOC_UNSIGNED;

  stubHelperHeaderSize = sizeof(stubHelperHeaderCode);
  stubHelperEntrySize = sizeof(stubHelperEntryCode);
}

namespace {
struct Adrp {
  uint32_t destRegister;
};

struct Add {
  uint8_t destRegister;
  uint8_t srcRegister;
  uint32_t addend;
};

enum ExtendType { ZeroExtend = 1, Sign64 = 2, Sign32 = 3 };

struct Ldr {
  uint8_t destRegister;
  uint8_t baseRegister;
  uint8_t size;
  bool isFloat;
  ExtendType extendType;
  uint64_t offset;
};

struct PerformedReloc {
  const Reloc &rel;
  uint64_t referentVA;
};

class OptimizationHintContext {
public:
  OptimizationHintContext(uint8_t *buf, const ConcatInputSection *isec,
                          ArrayRef<uint64_t> relocTargets)
      : buf(buf), isec(isec), relocTargets(relocTargets),
        relocIt(isec->relocs.rbegin()) {}

  void applyAdrpAdd(const OptimizationHint &);
  void applyAdrpAdrp(const OptimizationHint &);
  void applyAdrpLdr(const OptimizationHint &);

private:
  uint8_t *buf;
  const ConcatInputSection *isec;
  ArrayRef<uint64_t> relocTargets;
  std::vector<Reloc>::const_reverse_iterator relocIt;

  uint64_t getRelocTarget(const Reloc &);

  Optional<PerformedReloc> findPrimaryReloc(uint64_t offset);
  Optional<PerformedReloc> findReloc(uint64_t offset);
};
} // namespace

static bool parseAdrp(uint32_t insn, Adrp &adrp) {
  if ((insn & 0x9f000000) != 0x90000000)
    return false;
  adrp.destRegister = insn & 0x1f;
  return true;
}

static bool parseAdd(uint32_t insn, Add &add) {
  if ((insn & 0xffc00000) != 0x91000000)
    return false;
  add.destRegister = insn & 0x1f;
  add.srcRegister = (insn >> 5) & 0x1f;
  add.addend = (insn >> 10) & 0xfff;
  return true;
}

static bool parseLdr(uint32_t insn, Ldr &ldr) {
  ldr.destRegister = insn & 0x1f;
  ldr.baseRegister = (insn >> 5) & 0x1f;
  uint8_t size = insn >> 30;
  uint8_t opc = (insn >> 22) & 3;

  if ((insn & 0x3fc00000) == 0x39400000) {
    // LDR (immediate), LDRB (immediate), LDRH (immediate)
    ldr.size = 1 << size;
    ldr.extendType = ZeroExtend;
    ldr.isFloat = false;
  } else if ((insn & 0x3f800000) == 0x39800000) {
    // LDRSB (immediate), LDRSH (immediate), LDRSW (immediate)
    ldr.size = 1 << size;
    ldr.extendType = static_cast<ExtendType>(opc);
    ldr.isFloat = false;
  } else if ((insn & 0x3f400000) == 0x3d400000) {
    // LDR (immediate, SIMD&FP)
    ldr.extendType = ZeroExtend;
    ldr.isFloat = true;
    if (size == 2 && opc == 1)
      ldr.size = 4;
    else if (size == 3 && opc == 1)
      ldr.size = 8;
    else if (size == 0 && opc == 3)
      ldr.size = 16;
    else
      return false;
  } else {
    return false;
  }
  ldr.offset = ((insn >> 10) & 0xfff) * ldr.size;
  return true;
}

static void writeAdr(void *loc, uint32_t dest, int32_t delta) {
  uint32_t opcode = 0x10000000;
  uint32_t immHi = (delta & 0x001ffffc) << 3;
  uint32_t immLo = (delta & 0x00000003) << 29;
  write32le(loc, opcode | immHi | immLo | dest);
}

static void writeNop(void *loc) { write32le(loc, 0xd503201f); }

static void writeLiteralLdr(void *loc, Ldr original, int32_t delta) {
  uint32_t imm19 = (delta & 0x001ffffc) << 3;
  uint32_t opcode = 0;
  switch (original.size) {
  case 4:
    if (original.isFloat)
      opcode = 0x1c000000;
    else
      opcode = original.extendType == Sign64 ? 0x98000000 : 0x18000000;
    break;
  case 8:
    opcode = original.isFloat ? 0x5c000000 : 0x58000000;
    break;
  case 16:
    opcode = 0x9c000000;
    break;
  default:
    assert(false && "Invalid size for literal ldr");
  }
  write32le(loc, opcode | imm19 | original.destRegister);
}

uint64_t OptimizationHintContext::getRelocTarget(const Reloc &reloc) {
  size_t relocIdx = &reloc - isec->relocs.data();
  return relocTargets[relocIdx];
}

// Optimization hints are sorted in a monotonically increasing order by their
// first address as are relocations (albeit in decreasing order), so if we keep
// a pointer around to the last found relocation, we don't have to do a full
// binary search every time.
Optional<PerformedReloc>
OptimizationHintContext::findPrimaryReloc(uint64_t offset) {
  const auto end = isec->relocs.rend();
  while (relocIt != end && relocIt->offset < offset)
    ++relocIt;
  if (relocIt == end || relocIt->offset != offset)
    return None;
  return PerformedReloc{*relocIt, getRelocTarget(*relocIt)};
}

// The second and third addresses of optimization hints have no such
// monotonicity as the first, so we search the entire range of relocations.
Optional<PerformedReloc> OptimizationHintContext::findReloc(uint64_t offset) {
  // Optimization hints often apply to successive relocations, so we check for
  // that first before doing a full binary search.
  auto end = isec->relocs.rend();
  if (relocIt < end - 1 && (relocIt + 1)->offset == offset)
    return PerformedReloc{*(relocIt + 1), getRelocTarget(*(relocIt + 1))};

  auto reloc = lower_bound(isec->relocs, offset,
                           [](const Reloc &reloc, uint64_t offset) {
                             return offset < reloc.offset;
                           });

  if (reloc == isec->relocs.end() || reloc->offset != offset)
    return None;
  return PerformedReloc{*reloc, getRelocTarget(*reloc)};
}

// Transforms a pair of adrp+add instructions into an adr instruction if the
// target is within the +/- 1 MiB range allowed by the adr's 21 bit signed
// immediate offset.
//
//   adrp xN, _foo@PAGE
//   add  xM, xN, _foo@PAGEOFF
// ->
//   adr  xM, _foo
//   nop
void OptimizationHintContext::applyAdrpAdd(const OptimizationHint &hint) {
  uint32_t ins1 = read32le(buf + hint.offset0);
  uint32_t ins2 = read32le(buf + hint.offset0 + hint.delta[0]);
  Adrp adrp;
  if (!parseAdrp(ins1, adrp))
    return;
  Add add;
  if (!parseAdd(ins2, add))
    return;
  if (adrp.destRegister != add.srcRegister)
    return;

  Optional<PerformedReloc> rel1 = findPrimaryReloc(hint.offset0);
  Optional<PerformedReloc> rel2 = findReloc(hint.offset0 + hint.delta[0]);
  if (!rel1 || !rel2)
    return;
  if (rel1->referentVA != rel2->referentVA)
    return;
  int64_t delta = rel1->referentVA - rel1->rel.offset - isec->getVA();
  if (delta >= (1 << 20) || delta < -(1 << 20))
    return;

  writeAdr(buf + hint.offset0, add.destRegister, delta);
  writeNop(buf + hint.offset0 + hint.delta[0]);
}

// Transforms two adrp instructions into a single adrp if their referent
// addresses are located on the same 4096 byte page.
//
//   adrp xN, _foo@PAGE
//   adrp xN, _bar@PAGE
// ->
//   adrp xN, _foo@PAGE
//   nop
void OptimizationHintContext::applyAdrpAdrp(const OptimizationHint &hint) {
  uint32_t ins1 = read32le(buf + hint.offset0);
  uint32_t ins2 = read32le(buf + hint.offset0 + hint.delta[0]);
  Adrp adrp1, adrp2;
  if (!parseAdrp(ins1, adrp1) || !parseAdrp(ins2, adrp2))
    return;
  if (adrp1.destRegister != adrp2.destRegister)
    return;

  Optional<PerformedReloc> rel1 = findPrimaryReloc(hint.offset0);
  Optional<PerformedReloc> rel2 = findReloc(hint.offset0 + hint.delta[0]);
  if (!rel1 || !rel2)
    return;
  if ((rel1->referentVA & ~0xfffULL) != (rel2->referentVA & ~0xfffULL))
    return;

  writeNop(buf + hint.offset0 + hint.delta[0]);
}

// Transforms a pair of adrp+ldr (immediate) instructions into an ldr (literal)
// load from a PC-relative address if it is 4-byte aligned and within +/- 1 MiB,
// as ldr can encode a signed 19-bit offset that gets multiplied by 4.
//
//   adrp xN, _foo@PAGE
//   ldr  xM, [xN, _foo@PAGEOFF]
// ->
//   nop
//   ldr  xM, _foo
void OptimizationHintContext::applyAdrpLdr(const OptimizationHint &hint) {
  uint32_t ins1 = read32le(buf + hint.offset0);
  uint32_t ins2 = read32le(buf + hint.offset0 + hint.delta[0]);
  Adrp adrp;
  if (!parseAdrp(ins1, adrp))
    return;
  Ldr ldr;
  if (!parseLdr(ins2, ldr))
    return;
  if (adrp.destRegister != ldr.baseRegister)
    return;

  Optional<PerformedReloc> rel1 = findPrimaryReloc(hint.offset0);
  Optional<PerformedReloc> rel2 = findReloc(hint.offset0 + hint.delta[0]);
  if (!rel1 || !rel2)
    return;
  if (ldr.offset != (rel1->referentVA & 0xfff))
    return;
  if ((rel1->referentVA & 3) != 0)
    return;
  if (ldr.size == 1 || ldr.size == 2)
    return;
  int64_t delta = rel1->referentVA - rel2->rel.offset - isec->getVA();
  if (delta >= (1 << 20) || delta < -(1 << 20))
    return;

  writeNop(buf + hint.offset0);
  writeLiteralLdr(buf + hint.offset0 + hint.delta[0], ldr, delta);
}

void ARM64::applyOptimizationHints(uint8_t *buf, const ConcatInputSection *isec,
                                   ArrayRef<uint64_t> relocTargets) const {
  assert(isec);
  assert(relocTargets.size() == isec->relocs.size());

  // Note: Some of these optimizations might not be valid when shared regions
  // are in use. Will need to revisit this if splitSegInfo is added.

  OptimizationHintContext ctx1(buf, isec, relocTargets);
  for (const OptimizationHint &hint : isec->optimizationHints) {
    switch (hint.type) {
    case LOH_ARM64_ADRP_ADRP:
      // This is done in another pass because the other optimization hints
      // might cause its targets to be turned into NOPs.
      break;
    case LOH_ARM64_ADRP_LDR:
      ctx1.applyAdrpLdr(hint);
      break;
    case LOH_ARM64_ADRP_ADD_LDR:
    case LOH_ARM64_ADRP_LDR_GOT_LDR:
    case LOH_ARM64_ADRP_ADD_STR:
    case LOH_ARM64_ADRP_LDR_GOT_STR:
      // TODO: Implement these
      break;
    case LOH_ARM64_ADRP_ADD:
      ctx1.applyAdrpAdd(hint);
      break;
    case LOH_ARM64_ADRP_LDR_GOT:
      // TODO: Implement this as well
      break;
    }
  }

  OptimizationHintContext ctx2(buf, isec, relocTargets);
  for (const OptimizationHint &hint : isec->optimizationHints)
    if (hint.type == LOH_ARM64_ADRP_ADRP)
      ctx2.applyAdrpAdrp(hint);
}

TargetInfo *macho::createARM64TargetInfo() {
  static ARM64 t;
  return &t;
}
