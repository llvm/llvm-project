//===- ARM64.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Arch/ARM64Common.h"
#include "Driver.h"
#include "InputFiles.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "lld/Common/ErrorHandler.h"
#include "mach-o/compact_unwind_encoding.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Option/ArgList.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;
using namespace lld::macho;

namespace {

struct ARM64 : ARM64Common {
  ARM64();
  void configureBranchExtenders(const llvm::opt::ArgList &args) override;
  void writeStub(uint8_t *buf, const Symbol &, uint64_t) const override;
  void writeStubHelperHeader(uint8_t *buf) const override;
  void writeStubHelperEntry(uint8_t *buf, const Symbol &,
                            uint64_t entryAddr) const override;

  void writeObjCMsgSendStub(uint8_t *buf, Symbol *sym, uint64_t stubsAddr,
                            uint64_t &stubOffset, uint64_t selrefVA,
                            Symbol *objcMsgSend) const override;

  void initICFSafeThunkBody(InputSection *thunk,
                            Symbol *targetSym) const override;
  Symbol *getThunkBranchTarget(InputSection *thunk) const override;
  uint32_t getICFSafeThunkSize() const override;
};

} // namespace

// Random notes on reloc types:
// ADDEND always pairs with BRANCH26, PAGE21, or PAGEOFF12
// POINTER_TO_GOT: ld64 supports a 4-byte pc-relative form as well as an 8-byte
// absolute version of this relocation. The semantics of the absolute relocation
// are weird -- it results in the value of the GOT slot being written, instead
// of the address. Let's not support it unless we find a real-world use case.
static constexpr std::array<RelocAttrs, 11> relocAttrsArray{{
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

static constexpr uint32_t stubCode[] = {
    0x90000010, // 00: adrp  x16, __la_symbol_ptr@page
    0xf9400210, // 04: ldr   x16, [x16, __la_symbol_ptr@pageoff]
    0xd61f0200, // 08: br    x16
};

void ARM64::writeStub(uint8_t *buf8, const Symbol &sym,
                      uint64_t pointerVA) const {
  ::writeStub(buf8, stubCode, sym, pointerVA);
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

static constexpr uint32_t objcStubsFastCode[] = {
    0x90000001, // adrp  x1, __objc_selrefs@page
    0xf9400021, // ldr   x1, [x1, @selector("foo")@pageoff]
    0x90000010, // adrp  x16, _got@page
    0xf9400210, // ldr   x16, [x16, _objc_msgSend@pageoff]
    0xd61f0200, // br    x16
    0xd4200020, // brk   #0x1
    0xd4200020, // brk   #0x1
    0xd4200020, // brk   #0x1
};

static constexpr uint32_t objcStubsSmallCode[] = {
    0x90000001, // adrp  x1, __objc_selrefs@page
    0xf9400021, // ldr   x1, [x1, @selector("foo")@pageoff]
    0x14000000, // b     _objc_msgSend
};

void ARM64::writeObjCMsgSendStub(uint8_t *buf, Symbol *sym, uint64_t stubsAddr,
                                 uint64_t &stubOffset, uint64_t selrefVA,
                                 Symbol *objcMsgSend) const {
  uint64_t objcMsgSendAddr;
  uint64_t objcStubSize;
  uint64_t objcMsgSendIndex;

  if (config->objcStubsMode == ObjCStubsMode::fast) {
    objcStubSize = target->objcStubsFastSize;
    objcMsgSendAddr = in.got->addr;
    objcMsgSendIndex = objcMsgSend->gotIndex;
    ::writeObjCMsgSendFastStub<LP64>(buf, objcStubsFastCode, sym, stubsAddr,
                                     stubOffset, selrefVA, objcMsgSendAddr,
                                     objcMsgSendIndex);
  } else {
    assert(config->objcStubsMode == ObjCStubsMode::small);
    objcStubSize = target->objcStubsSmallSize;
    if (auto *d = dyn_cast<Defined>(objcMsgSend)) {
      objcMsgSendAddr = d->getVA();
      objcMsgSendIndex = 0;
    } else {
      objcMsgSendAddr = in.stubs->addr;
      objcMsgSendIndex = objcMsgSend->stubsIndex;
    }
    ::writeObjCMsgSendSmallStub<LP64>(buf, objcStubsSmallCode, sym, stubsAddr,
                                      stubOffset, selrefVA, objcMsgSendAddr,
                                      objcMsgSendIndex);
  }
  stubOffset += objcStubSize;
}

// A thunk is the relaxed variation of stubCode. We don't need the
// extra indirection through a lazy pointer because the target address
// is known at link time.
static constexpr uint32_t thunkCode[] = {
    0x90000010, // 00: adrp  x16, <thunk.ptr>@page
    0x91000210, // 04: add   x16, [x16,<thunk.ptr>@pageoff]
    0xd61f0200, // 08: br    x16
};

// A branch island preserves all registers and may be chained to other islands.
static constexpr uint32_t islandCode[] = {
    0x14000000, // b <target>
};

namespace {
class Island final : public BranchExtender {
public:
  Island() : BranchExtender(/*maxReach=*/uint64_t{1} << 27) {}
  std::pair<uint64_t, uint64_t> getSourceRange(uint64_t to) const override {
    assert(to % 4 == 0 && "unaligned branch destination");
    constexpr uint64_t range = uint64_t{1} << 27;
    return {to - std::min(to, range - 4), SaturatingAdd(to, range)};
  }
  size_t getSize() const override { return sizeof(islandCode); }
  uint32_t getAlign() const override { return 4; }
  StringRef getSuffix() const override { return ".island."; }
  void populate(InputSection *isec, Symbol *sym,
                int64_t addend) const override {
    isec->align = getAlign();
    isec->data = {reinterpret_cast<const uint8_t *>(islandCode), getSize()};
    isec->relocs.emplace_back(/*type=*/ARM64_RELOC_BRANCH26,
                              /*pcrel=*/true, /*length=*/2,
                              /*offset=*/0, /*addend=*/addend,
                              /*referent=*/sym);
  }
};

class Thunk final : public BranchExtender {
public:
  Thunk() : BranchExtender(/*maxReach=*/(uint64_t{1} << 32) + 4095) {}
  std::pair<uint64_t, uint64_t> getSourceRange(uint64_t to) const override {
    // ADRP encodes a signed 21-bit displacement between 4 KiB pages.
    constexpr uint64_t range = uint64_t{1} << 32;
    to &= ~uint64_t{4095};
    return {to - std::min(to, range - 4096), SaturatingAdd(to, range + 4095)};
  }
  size_t getSize() const override { return sizeof(thunkCode); }
  uint32_t getAlign() const override { return 4; }
  StringRef getSuffix() const override { return ".thunk."; }
  void populate(InputSection *isec, Symbol *sym,
                int64_t addend) const override {
    isec->align = getAlign();
    isec->data = {reinterpret_cast<const uint8_t *>(thunkCode), getSize()};
    isec->relocs.emplace_back(/*type=*/ARM64_RELOC_PAGEOFF12,
                              /*pcrel=*/false, /*length=*/2,
                              /*offset=*/4, /*addend=*/addend,
                              /*referent=*/sym);
    isec->relocs.emplace_back(/*type=*/ARM64_RELOC_PAGE21,
                              /*pcrel=*/true, /*length=*/2,
                              /*offset=*/0, /*addend=*/addend,
                              /*referent=*/sym);
  }
};
} // namespace

// Just a single direct branch to the target function.
static constexpr uint32_t icfSafeThunkCode[] = {
    0x14000000, // 08: b    target
};

void ARM64::initICFSafeThunkBody(InputSection *thunk, Symbol *targetSym) const {
  // The base data here will not be itself modified, we'll just be adding a
  // reloc below. So we can directly use the constexpr above as the data.
  thunk->data = {reinterpret_cast<const uint8_t *>(icfSafeThunkCode),
                 sizeof(icfSafeThunkCode)};

  thunk->relocs.emplace_back(/*type=*/ARM64_RELOC_BRANCH26,
                             /*pcrel=*/true, /*length=*/2,
                             /*offset=*/0, /*addend=*/0,
                             /*referent=*/targetSym);
}

Symbol *ARM64::getThunkBranchTarget(InputSection *thunk) const {
  assert(thunk->relocs.size() == 1 &&
         "expected a single reloc on ARM64 ICF thunk");
  auto &reloc = thunk->relocs[0];
  assert(isa<Symbol *>(reloc.referent) &&
         "ARM64 thunk reloc is expected to point to a Symbol");

  return cast<Symbol *>(reloc.referent);
}

uint32_t ARM64::getICFSafeThunkSize() const { return sizeof(icfSafeThunkCode); }

void ARM64::configureBranchExtenders(const llvm::opt::ArgList &args) {
  auto getInteger = [&](unsigned option, uint32_t defaultValue) {
    const llvm::opt::Arg *arg = args.getLastArg(option);
    if (!arg)
      return defaultValue;
    uint32_t value;
    if (llvm::to_integer(arg->getValue(), value))
      return value;
    error(arg->getSpelling() +
          ": expected an integer in the range [0, 4294967295], but got '" +
          arg->getValue() + "'");
    return defaultValue;
  };
  static Island island;
  static Thunk thunk;
  static const BranchExtender *const extenders[] = {&island, &thunk};
  island.maxHops = getInteger(OPT_arm64_islands_max_hops, 2);
  thunk.maxHops = getInteger(OPT_arm64_thunks_max_hops, 1);
  coarseBoundarySpacing = getInteger(
      OPT_arm64_branch_extension_coarse_boundary_spacing, 1024 * 1024);
  branchExtenders = extenders;
  for ([[maybe_unused]] const BranchExtender *extender : branchExtenders)
    assert(extender->maxReach > 0 && extender->maxReach >= forwardBranchRange &&
           extender->maxReach >= backwardBranchRange);
}

ARM64::ARM64() : ARM64Common(LP64()) {
  cpuType = CPU_TYPE_ARM64;
  cpuSubtype = CPU_SUBTYPE_ARM64_ALL;

  stubSize = sizeof(stubCode);

  objcStubsFastSize = sizeof(objcStubsFastCode);
  objcStubsFastAlignment = 32;
  objcStubsSmallSize = sizeof(objcStubsSmallCode);
  objcStubsSmallAlignment = 4;

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

  relocAttrs = {relocAttrsArray.data(), relocAttrsArray.size()};
}

TargetInfo *macho::createARM64TargetInfo() {
  static ARM64 t;
  return &t;
}
