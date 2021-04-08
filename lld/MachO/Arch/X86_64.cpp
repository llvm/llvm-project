//===- X86_64.cpp ---------------------------------------------------------===//
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
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/Endian.h"

using namespace llvm::MachO;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::macho;

namespace {

struct X86_64 : TargetInfo {
  X86_64();

  int64_t getEmbeddedAddend(MemoryBufferRef, uint64_t offset,
                            const relocation_info) const override;
  void relocateOne(uint8_t *loc, const Reloc &, uint64_t va,
                   uint64_t relocVA) const override;

  void writeStub(uint8_t *buf, const Symbol &) const override;
  void writeStubHelperHeader(uint8_t *buf) const override;
  void writeStubHelperEntry(uint8_t *buf, const DylibSymbol &,
                            uint64_t entryAddr) const override;

  void relaxGotLoad(uint8_t *loc, uint8_t type) const override;
  const RelocAttrs &getRelocAttrs(uint8_t type) const override;
  uint64_t getPageSize() const override { return 4 * 1024; }
};

} // namespace

const RelocAttrs &X86_64::getRelocAttrs(uint8_t type) const {
  static const std::array<RelocAttrs, 10> relocAttrsArray{{
#define B(x) RelocAttrBits::x
      {"UNSIGNED", B(UNSIGNED) | B(ABSOLUTE) | B(EXTERN) | B(LOCAL) |
                       B(DYSYM8) | B(BYTE4) | B(BYTE8)},
      {"SIGNED", B(PCREL) | B(EXTERN) | B(LOCAL) | B(BYTE4)},
      {"BRANCH", B(PCREL) | B(EXTERN) | B(BRANCH) | B(BYTE4)},
      {"GOT_LOAD", B(PCREL) | B(EXTERN) | B(GOT) | B(LOAD) | B(BYTE4)},
      {"GOT", B(PCREL) | B(EXTERN) | B(GOT) | B(POINTER) | B(BYTE4)},
      {"SUBTRACTOR", B(SUBTRAHEND) | B(BYTE4) | B(BYTE8)},
      {"SIGNED_1", B(PCREL) | B(EXTERN) | B(LOCAL) | B(BYTE4)},
      {"SIGNED_2", B(PCREL) | B(EXTERN) | B(LOCAL) | B(BYTE4)},
      {"SIGNED_4", B(PCREL) | B(EXTERN) | B(LOCAL) | B(BYTE4)},
      {"TLV", B(PCREL) | B(EXTERN) | B(TLV) | B(LOAD) | B(BYTE4)},
#undef B
  }};
  assert(type < relocAttrsArray.size() && "invalid relocation type");
  if (type >= relocAttrsArray.size())
    return invalidRelocAttrs;
  return relocAttrsArray[type];
}

static int pcrelOffset(uint8_t type) {
  switch (type) {
  case X86_64_RELOC_SIGNED_1:
    return 1;
  case X86_64_RELOC_SIGNED_2:
    return 2;
  case X86_64_RELOC_SIGNED_4:
    return 4;
  default:
    return 0;
  }
}

int64_t X86_64::getEmbeddedAddend(MemoryBufferRef mb, uint64_t offset,
                                  relocation_info rel) const {
  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  const uint8_t *loc = buf + offset + rel.r_address;

  switch (rel.r_length) {
  case 2:
    return static_cast<int32_t>(read32le(loc)) + pcrelOffset(rel.r_type);
  case 3:
    return read64le(loc) + pcrelOffset(rel.r_type);
  default:
    llvm_unreachable("invalid r_length");
  }
}

void X86_64::relocateOne(uint8_t *loc, const Reloc &r, uint64_t value,
                         uint64_t relocVA) const {
  value += r.addend;
  if (r.pcrel) {
    uint64_t pc = relocVA + 4 + pcrelOffset(r.type);
    value -= pc;
  }

  switch (r.length) {
  case 2:
    if (r.type == X86_64_RELOC_UNSIGNED)
      checkUInt(r, value, 32);
    else
      checkInt(r, value, 32);
    write32le(loc, value);
    break;
  case 3:
    write64le(loc, value);
    break;
  default:
    llvm_unreachable("invalid r_length");
  }
}

// The following methods emit a number of assembly sequences with RIP-relative
// addressing. Note that RIP-relative addressing on X86-64 has the RIP pointing
// to the next instruction, not the current instruction, so we always have to
// account for the current instruction's size when calculating offsets.
// writeRipRelative helps with that.
//
// bufAddr:  The virtual address corresponding to buf[0].
// bufOff:   The offset within buf of the next instruction.
// destAddr: The destination address that the current instruction references.
static void writeRipRelative(SymbolDiagnostic d, uint8_t *buf, uint64_t bufAddr,
                             uint64_t bufOff, uint64_t destAddr) {
  uint64_t rip = bufAddr + bufOff;
  checkInt(d, destAddr - rip, 32);
  // For the instructions we care about, the RIP-relative address is always
  // stored in the last 4 bytes of the instruction.
  write32le(buf + bufOff - 4, destAddr - rip);
}

static constexpr uint8_t stub[] = {
    0xff, 0x25, 0, 0, 0, 0, // jmpq *__la_symbol_ptr(%rip)
};

void X86_64::writeStub(uint8_t *buf, const Symbol &sym) const {
  memcpy(buf, stub, 2); // just copy the two nonzero bytes
  uint64_t stubAddr = in.stubs->addr + sym.stubsIndex * sizeof(stub);
  writeRipRelative({&sym, "stub"}, buf, stubAddr, sizeof(stub),
                   in.lazyPointers->addr + sym.stubsIndex * LP64::wordSize);
}

static constexpr uint8_t stubHelperHeader[] = {
    0x4c, 0x8d, 0x1d, 0, 0, 0, 0, // 0x0: leaq ImageLoaderCache(%rip), %r11
    0x41, 0x53,                   // 0x7: pushq %r11
    0xff, 0x25, 0,    0, 0, 0,    // 0x9: jmpq *dyld_stub_binder@GOT(%rip)
    0x90,                         // 0xf: nop
};

void X86_64::writeStubHelperHeader(uint8_t *buf) const {
  memcpy(buf, stubHelperHeader, sizeof(stubHelperHeader));
  SymbolDiagnostic d = {nullptr, "stub helper header"};
  writeRipRelative(d, buf, in.stubHelper->addr, 7,
                   in.imageLoaderCache->getVA());
  writeRipRelative(d, buf, in.stubHelper->addr, 0xf,
                   in.got->addr +
                       in.stubHelper->stubBinder->gotIndex * LP64::wordSize);
}

static constexpr uint8_t stubHelperEntry[] = {
    0x68, 0, 0, 0, 0, // 0x0: pushq <bind offset>
    0xe9, 0, 0, 0, 0, // 0x5: jmp <__stub_helper>
};

void X86_64::writeStubHelperEntry(uint8_t *buf, const DylibSymbol &sym,
                                  uint64_t entryAddr) const {
  memcpy(buf, stubHelperEntry, sizeof(stubHelperEntry));
  write32le(buf + 1, sym.lazyBindOffset);
  writeRipRelative({&sym, "stub helper"}, buf, entryAddr,
                   sizeof(stubHelperEntry), in.stubHelper->addr);
}

void X86_64::relaxGotLoad(uint8_t *loc, uint8_t type) const {
  // Convert MOVQ to LEAQ
  if (loc[-2] != 0x8b)
    error(getRelocAttrs(type).name + " reloc requires MOVQ instruction");
  loc[-2] = 0x8d;
}

X86_64::X86_64() : TargetInfo(LP64()) {
  cpuType = CPU_TYPE_X86_64;
  cpuSubtype = CPU_SUBTYPE_X86_64_ALL;

  stubSize = sizeof(stub);
  stubHelperHeaderSize = sizeof(stubHelperHeader);
  stubHelperEntrySize = sizeof(stubHelperEntry);
}

TargetInfo *macho::createX86_64TargetInfo() {
  static X86_64 t;
  return &t;
}
