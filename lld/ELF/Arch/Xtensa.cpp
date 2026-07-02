//===- Xtensa.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Symbols.h"
#include "Target.h"
#include <bitset>
#include <iostream>
#include <string>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {

class Xtensa final : public TargetInfo {
public:
  Xtensa(Ctx &ctx);
  RelExpr getRelExpr(RelType type, const Symbol &s,
                     const uint8_t *loc) const override;
  void relocate(uint8_t *loc, const Relocation &rel,
                uint64_t val) const override;
};

} // namespace

Xtensa::Xtensa(Ctx &ctx) : TargetInfo(ctx) {}

RelExpr Xtensa::getRelExpr(RelType type, const Symbol &s,
                           const uint8_t *loc) const {
  switch (type) {
  case R_XTENSA_32:
    return R_ABS;
  case R_XTENSA_SLOT0_OP:
    // This relocation is used for various instructions, with varying ways to
    // calculate the relocation value. This is unlike most ELF architectures,
    // and is arguably bad design (see the comment on R_386_GOT32 in X86.cpp).
    // But that's what compilers emit, so it needs to be supported.
    //
    // We work around this by returning R_PC here and calculating the PC address
    // in Xtensa::relocate based on the relative value. That's ugly. A better
    // solution would be to look at the instruction here and emit various
    // Xtensa-specific RelTypes, but that has another problem: the RelExpr enum
    // is at its maximum size of 64. This will need to be fixed eventually, but
    // for now hack around it and return R_PC.
    return R_PC;
  case R_XTENSA_ASM_EXPAND:
    // This relocation appears to be emitted by the GNU Xtensa compiler as a
    // linker relaxation hint. For example, for the following code:
    //
    //   .section .foo
    //   .align  4
    //   foo:
    //       nop
    //       nop
    //       call0 bar
    //   .align  4
    //       bar:
    //
    // The call0 instruction is compiled to a l32r and callx0 instruction.
    // The LLVM Xtensa backend does not emit this relocation.
    // Because it's a relaxation hint, this relocation can be ignored for now
    // until linker relaxations are implemented.
    return R_NONE;
  case R_XTENSA_DIFF8:
  case R_XTENSA_DIFF16:
  case R_XTENSA_DIFF32:
  case R_XTENSA_PDIFF8:
  case R_XTENSA_PDIFF16:
  case R_XTENSA_PDIFF32:
  case R_XTENSA_NDIFF8:
  case R_XTENSA_NDIFF16:
  case R_XTENSA_NDIFF32:
    // > Xtensa relocations to mark the difference of two local symbols.
    // > These are only needed to support linker relaxation and can be ignored
    // > when not relaxing.
    // Source:
    // https://github.com/espressif/binutils-gdb/commit/30ce8e47fad9b057b6d7af9e1d43061126d34d20:
    // Because we don't do linker relaxation, we can ignore these relocations.
    return R_NONE;
  default:
    Err(ctx) << getErrorLoc(ctx, loc) << "unknown relocation (" << type.v
             << ") against symbol " << &s;
    return R_NONE;
  }
}

static inline bool isRRI8Branch(uint8_t *loc) {
  // instructions: ball, bany, bbc, bbci, bbs, bbsi, beq, bge, bgeu, blt,
  // bltu, bnall, bne, bnone
  if ((loc[0] & 0x0f) == 0b0111)
    return true;
  // instructions: beqi, bgei, bnei, blti
  if ((loc[0] & 0b11'1111) == 0b10'0110)
    return true;
  // instructions: bgeui, bltui
  if ((loc[0] & 0b1011'1111) == 0b1011'0110)
    return true;
  if ((loc[0] & 0b0111'1111) == 0b0111'0110) {
    // instruction: bf
    if ((loc[1] & 0b1111'0000) == 0b0000'0000)
      return true;
    // instruction: bt
    if ((loc[1] & 0b1111'0000) == 0b0001'0000)
      return true;
  }
  // some other instruction
  return false;
}

static inline bool isLoop(uint8_t *loc) {
  // instructions: loop, loopgtz, loopnez
  if ((loc[0] & 0b1111'1111) == 0b0111'0110) {
    // instruction: loop
    if ((loc[1] & 0b1111'0000) == 0b1000'0000)
      return true;
    // instruction: loopgtz
    if ((loc[1] & 0b1111'0000) == 0b1010'0000)
      return true;
    // instruction: loopnez
    if ((loc[1] & 0b1111'0000) == 0b1001'0000)
      return true;
  }
  // some other instruction
  return false;
}

void Xtensa::relocate(uint8_t *loc, const Relocation &rel, uint64_t val) const {
  switch (rel.type) {
  case R_XTENSA_32:
    // R_XTENSA_32 is a partial_inplace relocation (GNU bfd howto:
    // src_mask = dst_mask = 0xffffffff). The GNU Xtensa assembler stores the
    // addend in the relocated word itself and leaves the RELA r_addend as 0,
    // so the result must be S + r_addend + <existing contents>. Objects from
    // the LLVM Xtensa backend keep the addend in r_addend and zero the field,
    // so reading the contents back adds 0 and is harmless. Plain
    // write32le(loc, val) drops the in-place addend, which miscomputes every
    // R_XTENSA_32 with a nonzero addend -- e.g. the switch jump tables in the
    // precompiled Espressif Wi-Fi blobs, which then point at a section base
    // instead of the intended in-section offset.
    write32le(loc, val + read32le(loc));
    break;
  case R_XTENSA_SLOT0_OP: {
    // HACK: calculate the instruction location based on the PC-relative
    // relocation value.
    uint64_t dest = rel.sym->getVA(ctx, rel.addend);
    uint64_t p = dest - val;

    // This relocation is used for various instructions.
    // Look at the instruction to determine how to do the relocation.
    uint8_t opcode = loc[0] & 0x0f;
    if (opcode == 0b0001) { // RI16 format: l32r
      int64_t val = dest - ((p + 3) & (uint64_t)0xfffffffc);
      if ((val < -262144 || val > -4))
        reportRangeError(ctx, loc, rel, Twine(static_cast<int64_t>(val)),
                         -262141, -4);
      checkAlignment(ctx, loc, static_cast<uint64_t>(val), 4, rel);
      write16le(loc + 1, val >> 2);
    } else if (opcode == 0b0101) { // call0, call4, call8, call12 (CALL format)
      uint64_t val = dest - ((p + 4) & (uint64_t)0xfffffffc);
      checkInt(ctx, loc, static_cast<int64_t>(val) >> 2, 18, rel);
      checkAlignment(ctx, loc, val, 4, rel);
      const int64_t target = static_cast<int64_t>(val) >> 2;
      loc[0] = (loc[0] & 0b0011'1111) | ((target & 0b0000'0011) << 6);
      loc[1] = target >> 2;
      loc[2] = target >> 10;
    } else if ((loc[0] & 0x3f) == 0b00'0110) { // j (CALL format)
      uint64_t valJ = val - 4;
      checkInt(ctx, loc, static_cast<int64_t>(valJ), 18, rel);
      loc[0] = (loc[0] & 0b0011'1111) | ((valJ & 0b0000'0011) << 6);
      loc[1] = valJ >> 2;
      loc[2] = valJ >> 10;
    } else if (isRRI8Branch(loc)) { // RRI8 format (various branch instructions)
      uint64_t v = val - 4;
      checkInt(ctx, loc, static_cast<int64_t>(v), 8, rel);
      loc[2] = v & 0xff;
    } else if (isLoop(loc)) { // loop instructions
      uint64_t v = val - 4;
      checkUInt(ctx, loc, v, 8, rel);
      loc[2] = v & 0xff;
    } else if ((loc[0] & 0b1000'1111) ==
               0b1000'1100) { // RI16 format: beqz.n, bnez.n
      uint64_t v = val - 4;
      checkUInt(ctx, loc, v, 6, rel);
      loc[0] = (loc[0] & 0xcf) | (v & 0x30);
      loc[1] = (loc[1] & 0x0f) | ((v & 0x0f) << 4);
    } else if ((loc[0] & 0b0011'1111) ==
               0b0001'0110) { // BRI12 format: beqz, bgez, bltz, bnez
      uint64_t v = val - 4;
      checkInt(ctx, loc, static_cast<int64_t>(v), 12, rel);
      loc[1] = ((loc[1] & 0x0f)) | ((v & 0x0f) << 4);
      loc[2] = (v >> 4) & 0xff;
    } else {
      Err(ctx) << getErrorLoc(ctx, loc)
               << "unknown opcode for relocation: " << loc[0];
    }
    break;
  }
  default:
    llvm_unreachable("unknown relocation");
  }
}

void elf::setXtensaTargetInfo(Ctx &ctx) { ctx.target.reset(new Xtensa(ctx)); }
