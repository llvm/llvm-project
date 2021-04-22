//===- ARM64Common.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Arch/ARM64Common.h"

#include "lld/Common/ErrorHandler.h"
#include "llvm/Support/Endian.h"

using namespace llvm::MachO;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::macho;

int64_t ARM64Common::getEmbeddedAddend(MemoryBufferRef mb, uint64_t offset,
                                       const relocation_info rel) const {
  if (rel.r_type != ARM64_RELOC_UNSIGNED &&
      rel.r_type != ARM64_RELOC_SUBTRACTOR) {
    // All other reloc types should use the ADDEND relocation to store their
    // addends.
    // TODO(gkm): extract embedded addend just so we can assert that it is 0
    return 0;
  }

  const auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  const uint8_t *loc = buf + offset + rel.r_address;
  switch (rel.r_length) {
  case 2:
    return static_cast<int32_t>(read32le(loc));
  case 3:
    return read64le(loc);
  default:
    llvm_unreachable("invalid r_length");
  }
}

// For instruction relocations (load, store, add), the base
// instruction is pre-populated in the text section. A pre-populated
// instruction has opcode & register-operand bits set, with immediate
// operands zeroed. We read it from text, OR-in the immediate
// operands, then write-back the completed instruction.

void ARM64Common::relocateOne(uint8_t *loc, const Reloc &r, uint64_t value,
                              uint64_t pc) const {
  uint32_t base = ((r.length == 2) ? read32le(loc) : 0);
  switch (r.type) {
  case ARM64_RELOC_BRANCH26:
    value = encodeBranch26(r, base, value - pc);
    break;
  case ARM64_RELOC_SUBTRACTOR:
  case ARM64_RELOC_UNSIGNED:
    if (r.length == 2)
      checkInt(r, value, 32);
    break;
  case ARM64_RELOC_POINTER_TO_GOT:
    if (r.pcrel)
      value -= pc;
    checkInt(r, value, 32);
    break;
  case ARM64_RELOC_PAGE21:
  case ARM64_RELOC_GOT_LOAD_PAGE21:
  case ARM64_RELOC_TLVP_LOAD_PAGE21: {
    assert(r.pcrel);
    value = encodePage21(r, base, pageBits(value) - pageBits(pc));
    break;
  }
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

void ARM64Common::relaxGotLoad(uint8_t *loc, uint8_t type) const {
  // The instruction format comments below are quoted from
  // Arm® Architecture Reference Manual
  // Armv8, for Armv8-A architecture profile
  // ARM DDI 0487G.a (ID011921)
  uint32_t instruction = read32le(loc);
  // C6.2.132 LDR (immediate)
  // This matches both the 64- and 32-bit variants:
  // LDR <(X|W)t>, [<Xn|SP>{, #<pimm>}]
  if ((instruction & 0xbfc00000) != 0xb9400000)
    error(getRelocAttrs(type).name + " reloc requires LDR instruction");
  assert(((instruction >> 10) & 0xfff) == 0 &&
         "non-zero embedded LDR immediate");
  // C6.2.4 ADD (immediate)
  // ADD <Xd|SP>, <Xn|SP>, #<imm>{, <shift>}
  instruction = ((instruction & 0x001fffff) | 0x91000000);
  write32le(loc, instruction);
}
