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

  uint64_t getImplicitAddend(MemoryBufferRef, const section_64 &,
                             const relocation_info &) const override;
  void relocateOne(uint8_t *loc, const Reloc &, uint64_t val) const override;

  void writeStub(uint8_t *buf, const DylibSymbol &) const override;
  void writeStubHelperHeader(uint8_t *buf) const override;
  void writeStubHelperEntry(uint8_t *buf, const DylibSymbol &,
                            uint64_t entryAddr) const override;

  void prepareDylibSymbolRelocation(DylibSymbol &, uint8_t type) override;
  uint64_t getDylibSymbolVA(const DylibSymbol &, uint8_t type) const override;
};

} // namespace

static std::string getErrorLocation(MemoryBufferRef mb, const section_64 &sec,
                                    const relocation_info &rel) {
  return ("invalid relocation at offset " + std::to_string(rel.r_address) +
          " of " + sec.segname + "," + sec.sectname + " in " +
          mb.getBufferIdentifier())
      .str();
}

static void validateLength(MemoryBufferRef mb, const section_64 &sec,
                           const relocation_info &rel,
                           const std::vector<uint8_t> &validLengths) {
  if (std::find(validLengths.begin(), validLengths.end(), rel.r_length) !=
      validLengths.end())
    return;

  std::string msg = getErrorLocation(mb, sec, rel) + ": relocations of type " +
                    std::to_string(rel.r_type) + " must have r_length of ";
  bool first = true;
  for (uint8_t length : validLengths) {
    if (!first)
      msg += " or ";
    first = false;
    msg += std::to_string(length);
  }
  fatal(msg);
}

uint64_t X86_64::getImplicitAddend(MemoryBufferRef mb, const section_64 &sec,
                                   const relocation_info &rel) const {
  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  const uint8_t *loc = buf + sec.offset + rel.r_address;
  switch (rel.r_type) {
  case X86_64_RELOC_BRANCH:
    // XXX: ld64 also supports r_length = 0 here but I'm not sure when such a
    // relocation will actually be generated.
    validateLength(mb, sec, rel, {2});
    break;
  case X86_64_RELOC_SIGNED:
  case X86_64_RELOC_SIGNED_1:
  case X86_64_RELOC_SIGNED_2:
  case X86_64_RELOC_SIGNED_4:
  case X86_64_RELOC_GOT_LOAD:
    if (!rel.r_pcrel)
      fatal(getErrorLocation(mb, sec, rel) + ": relocations of type " +
            std::to_string(rel.r_type) + " must be pcrel");
    validateLength(mb, sec, rel, {2});
    break;
  case X86_64_RELOC_UNSIGNED:
    if (rel.r_pcrel)
      fatal(getErrorLocation(mb, sec, rel) + ": relocations of type " +
            std::to_string(rel.r_type) + " must not be pcrel");
    validateLength(mb, sec, rel, {2, 3});
    break;
  default:
    error("TODO: Unhandled relocation type " + std::to_string(rel.r_type));
    return 0;
  }

  switch (rel.r_length) {
  case 0:
    return *loc;
  case 1:
    return read16le(loc);
  case 2:
    return read32le(loc);
  case 3:
    return read64le(loc);
  default:
    llvm_unreachable("invalid r_length");
  }
}

void X86_64::relocateOne(uint8_t *loc, const Reloc &r, uint64_t val) const {
  switch (r.type) {
  case X86_64_RELOC_BRANCH:
  case X86_64_RELOC_SIGNED:
  case X86_64_RELOC_SIGNED_1:
  case X86_64_RELOC_SIGNED_2:
  case X86_64_RELOC_SIGNED_4:
  case X86_64_RELOC_GOT_LOAD:
    // These types are only used for pc-relative relocations, so offset by 4
    // since the RIP has advanced by 4 at this point. This is only valid when
    // r_length = 2, which is enforced by validateLength().
    val -= 4;
    break;
  case X86_64_RELOC_UNSIGNED:
    break;
  default:
    llvm_unreachable(
        "getImplicitAddend should have flagged all unhandled relocation types");
  }

  switch (r.length) {
  case 0:
    *loc = val;
    break;
  case 1:
    write16le(loc, val);
    break;
  case 2:
    write32le(loc, val);
    break;
  case 3:
    write64le(loc, val);
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
static void writeRipRelative(uint8_t *buf, uint64_t bufAddr, uint64_t bufOff,
                             uint64_t destAddr) {
  uint64_t rip = bufAddr + bufOff;
  // For the instructions we care about, the RIP-relative address is always
  // stored in the last 4 bytes of the instruction.
  write32le(buf + bufOff - 4, destAddr - rip);
}

static constexpr uint8_t stub[] = {
    0xff, 0x25, 0, 0, 0, 0, // jmpq *__la_symbol_ptr(%rip)
};

void X86_64::writeStub(uint8_t *buf, const DylibSymbol &sym) const {
  memcpy(buf, stub, 2); // just copy the two nonzero bytes
  uint64_t stubAddr = in.stubs->addr + sym.stubsIndex * sizeof(stub);
  writeRipRelative(buf, stubAddr, sizeof(stub),
                   in.lazyPointers->addr + sym.stubsIndex * WordSize);
}

static constexpr uint8_t stubHelperHeader[] = {
    0x4c, 0x8d, 0x1d, 0, 0, 0, 0, // 0x0: leaq ImageLoaderCache(%rip), %r11
    0x41, 0x53,                   // 0x7: pushq %r11
    0xff, 0x25, 0,    0, 0, 0,    // 0x9: jmpq *dyld_stub_binder@GOT(%rip)
    0x90,                         // 0xf: nop
};

static constexpr uint8_t stubHelperEntry[] = {
    0x68, 0, 0, 0, 0, // 0x0: pushq <bind offset>
    0xe9, 0, 0, 0, 0, // 0x5: jmp <__stub_helper>
};

void X86_64::writeStubHelperHeader(uint8_t *buf) const {
  memcpy(buf, stubHelperHeader, sizeof(stubHelperHeader));
  writeRipRelative(buf, in.stubHelper->addr, 7, in.imageLoaderCache->getVA());
  writeRipRelative(buf, in.stubHelper->addr, 0xf,
                   in.got->addr +
                       in.stubHelper->stubBinder->gotIndex * WordSize);
}

void X86_64::writeStubHelperEntry(uint8_t *buf, const DylibSymbol &sym,
                                  uint64_t entryAddr) const {
  memcpy(buf, stubHelperEntry, sizeof(stubHelperEntry));
  write32le(buf + 1, sym.lazyBindOffset);
  writeRipRelative(buf, entryAddr, sizeof(stubHelperEntry),
                   in.stubHelper->addr);
}

void X86_64::prepareDylibSymbolRelocation(DylibSymbol &sym, uint8_t type) {
  switch (type) {
  case X86_64_RELOC_GOT_LOAD:
    in.got->addEntry(sym);
    break;
  case X86_64_RELOC_BRANCH:
    in.stubs->addEntry(sym);
    break;
  case X86_64_RELOC_GOT:
    fatal("TODO: Unhandled dylib symbol relocation X86_64_RELOC_GOT");
  default:
    llvm_unreachable("Unexpected dylib relocation type");
  }
}

uint64_t X86_64::getDylibSymbolVA(const DylibSymbol &sym, uint8_t type) const {
  switch (type) {
  case X86_64_RELOC_GOT_LOAD:
    return in.got->addr + sym.gotIndex * WordSize;
  case X86_64_RELOC_BRANCH:
    return in.stubs->addr + sym.stubsIndex * sizeof(stub);
  case X86_64_RELOC_GOT:
    fatal("TODO: Unhandled dylib symbol relocation X86_64_RELOC_GOT");
  default:
    llvm_unreachable("Unexpected dylib relocation type");
  }
}

X86_64::X86_64() {
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
