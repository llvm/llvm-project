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
#include "llvm/BinaryFormat/MachO.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;
using namespace lld::macho;

namespace {

struct ARM64 : ARM64Common {
  ARM64();
  void writeStub(uint8_t *buf, const Symbol &, uint64_t) const override;
  void writeStubHelperHeader(uint8_t *buf) const override;
  void writeStubHelperEntry(uint8_t *buf, const Symbol &,
                            uint64_t entryAddr) const override;

  void writeObjCMsgSendStub(uint8_t *buf, Symbol *sym, uint64_t stubsAddr,
                            uint64_t &stubOffset, uint64_t selrefVA,
                            Symbol *objcMsgSend) const override;
  void populateThunk(InputSection *thunk, Symbol *funcSym) override;

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

void ARM64::populateThunk(InputSection *thunk, Symbol *funcSym) {
  thunk->align = 4;
  thunk->data = {reinterpret_cast<const uint8_t *>(thunkCode),
                 sizeof(thunkCode)};
  thunk->relocs.emplace_back(/*type=*/ARM64_RELOC_PAGEOFF12,
                             /*pcrel=*/false, /*length=*/2,
                             /*offset=*/4, /*addend=*/0,
                             /*referent=*/funcSym);
  thunk->relocs.emplace_back(/*type=*/ARM64_RELOC_PAGE21,
                             /*pcrel=*/true, /*length=*/2,
                             /*offset=*/0, /*addend=*/0,
                             /*referent=*/funcSym);
}
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

ARM64::ARM64() : ARM64Common(LP64()) {
  cpuType = CPU_TYPE_ARM64;
  cpuSubtype = CPU_SUBTYPE_ARM64_ALL;

  stubSize = sizeof(stubCode);
  thunkSize = sizeof(thunkCode);

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
