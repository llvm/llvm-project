//===- ARM64e.cpp ---------------------------------------------------------===//
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

struct ARM64e : ARM64Common {
  ARM64e();
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
static constexpr std::array<RelocAttrs, 12> relocAttrsArray{{
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
    // ARM64e-specific: AUTHENTICATED_POINTER (64-bit absolute, external or
    // local)
    {"AUTHENTICATED_POINTER",
     B(ABSOLUTE) | B(UNSIGNED) | B(EXTERN) | B(LOCAL) | B(BYTE8) | B(AUTH)},
#undef B
}};

// ARM64e uses authenticated stubs with braa instruction.
// These are 16 bytes (4 instructions) instead of the regular 12 bytes.
// The stub computes the GOT address in x17 for use as authentication context.
static constexpr uint32_t stubCode[] = {
    0x90000011, // 00: adrp  x17, __auth_got@page
    0x91000231, // 04: add   x17, x17, __auth_got@pageoff
    0xf9400230, // 08: ldr   x16, [x17]
    0xd71f0a11, // 0c: braa  x16, x17  ; authenticate with IA key, context=x17
};

void ARM64e::writeStub(uint8_t *buf8, const Symbol &sym,
                       uint64_t pointerVA) const {
  auto *buf32 = reinterpret_cast<uint32_t *>(buf8);
  constexpr size_t stubCodeSize = sizeof(stubCode);
  SymbolDiagnostic d = {&sym, "stub"};
  uint64_t stubAddr = in.stubs->addr + sym.stubsIndex * stubCodeSize;
  uint64_t pcPageBits = pageBits(stubAddr);
  uint64_t targetPageBits = pageBits(pointerVA);
  int64_t pageDiff = static_cast<int64_t>(targetPageBits - pcPageBits);
  // adrp x17, __auth_got@page
  encodePage21(&buf32[0], d, stubCode[0], pageDiff);
  // add x17, x17, __auth_got@pageoff
  encodePageOff12(&buf32[1], d, stubCode[1], pointerVA);
  // ldr x16, [x17]
  buf32[2] = stubCode[2];
  // braa x16, x17
  buf32[3] = stubCode[3];
}

// ARM64e uses authenticated ObjC stubs with braa instruction.
// Uses x17 as both the address register and authentication context,
// matching the pattern used in ARM64e auth stubs.
static constexpr uint32_t objcStubsFastCode[] = {
    0x90000001, // adrp  x1, __objc_selrefs@page
    0xf9400021, // ldr   x1, [x1, @selector("foo")@pageoff]
    0x90000011, // adrp  x17, __auth_got@page
    0x91000231, // add   x17, x17, __auth_got@pageoff
    0xf9400230, // ldr   x16, [x17]
    0xd71f0a11, // braa  x16, x17  ; authenticate with IA key
    0xd4200020, // brk   #0x1
    0xd4200020, // brk   #0x1
};

static constexpr uint32_t objcStubsSmallCode[] = {
    0x90000001, // adrp  x1, __objc_selrefs@page
    0xf9400021, // ldr   x1, [x1, @selector("foo")@pageoff]
    0x14000000, // b     _objc_msgSend
};

void ARM64e::writeObjCMsgSendStub(uint8_t *buf, Symbol *sym, uint64_t stubsAddr,
                                  uint64_t &stubOffset, uint64_t selrefVA,
                                  Symbol *objcMsgSend) const {
  uint64_t objcMsgSendAddr;
  uint64_t objcStubSize;
  uint64_t objcMsgSendIndex;

  if (config->objcStubsMode == ObjCStubsMode::fast) {
    objcStubSize = target->objcStubsFastSize;
    // ARM64e uses authgot for objc_msgSend.
    objcMsgSendAddr = in.authgot->addr;
    objcMsgSendIndex = objcMsgSend->authGotIndex;
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

ARM64e::ARM64e() : ARM64Common(LP64()) {
  cpuType = CPU_TYPE_ARM64;
  // ARM64e-specific: Use ARM64E subtype with pointer authentication ABI version
  // 0
  cpuSubtype = CPU_SUBTYPE_ARM64E_WITH_PTRAUTH_VERSION(/*version*/ 0,
                                                       /*kernel*/ false);

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

TargetInfo *macho::createARM64eTargetInfo() {
  static ARM64e t;
  return &t;
}
