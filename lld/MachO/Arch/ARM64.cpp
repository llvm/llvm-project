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

  void writeObjCMsgSendStub(uint8_t *buf, Symbol *sym, uint64_t stubsAddr,
                            uint64_t &stubOffset, uint64_t selrefVA,
                            Symbol *objcMsgSend) const override;
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

ARM64::ARM64() : ARM64Common(LP64()) {
  cpuType = CPU_TYPE_ARM64;
  cpuSubtype = CPU_SUBTYPE_ARM64_ALL;

  stubSize = sizeof(stubCode);
  thunkSize = sizeof(arm64ThunkCode);

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

  stubHelperHeaderSize = sizeof(arm64StubHelperHeaderCode);
  stubHelperEntrySize = sizeof(arm64StubHelperEntryCode);

  relocAttrs = {relocAttrsArray.data(), relocAttrsArray.size()};
}

TargetInfo *macho::createARM64TargetInfo() {
  static ARM64 t;
  return &t;
}
