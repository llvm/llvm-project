//===- Next32.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Symbols.h"
#include "Target.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {
class Next32 final : public TargetInfo {
public:
  Next32();
  void relocate(uint8_t *loc, const Relocation &rel,
                uint64_t val) const override;
  RelExpr getRelExpr(RelType type, const Symbol &s,
                     const uint8_t *loc) const override;
  RelType getDynRel(RelType type) const override;
  bool forceEmitSectionRelocs(const InputSectionBase *target) const override;
};
} // namespace

Next32::Next32() {
  defaultImageBase = 0x0;
  relativeRel = R_NEXT32_SYM_RELATIVE;
  config->copyRelocs = true;
  config->zText = false;
}

// All our relocations resolve to absolute addresses at runtime, and so
// they are never link-time constants (see isStaticLinkTimeConstant) and
// are always dynamic

RelExpr Next32::getRelExpr(RelType type, const Symbol &s,
                           const uint8_t *loc) const {
  return R_ABS;
}

RelType Next32::getDynRel(RelType type) const { return type; }

void Next32::relocate(uint8_t *loc, const Relocation &rel, uint64_t val) const {
  checkUInt(loc, val, 32, rel);
  switch (rel.type) {
  /* Must contain all symbol types from Next32.def! */
  case R_NEXT32_SYM_MEM_64HI:
  case R_NEXT32_SYM_FUNC_64HI:
    val >>= 32;
    LLVM_FALLTHROUGH;
  case R_NEXT32_SYM_BB_IMM:
  case R_NEXT32_SYM_FUNCTION:
  case R_NEXT32_SYM_FUNC_64LO:
  case R_NEXT32_SYM_MEM_64LO:
    /* Next32 instruction set encodes immediates as Big-Endian */
    write32be(loc, val);
    break;
  case R_NEXT32_ABS32:
    /* DWARF debug relocations have target-endianness */
    write32(loc, val, config->endianness);
    break;
  case R_NEXT32_ABS64:
    write64(loc, val, config->endianness);
    break;
  default:
    error(getErrorLocation(loc) + "unrecognized relocation " +
          toString(rel.type));
  }
}

bool Next32::forceEmitSectionRelocs(const InputSectionBase *target) const {
  return !!(target->flags & SHF_ALLOC);
}

TargetInfo *elf::getNext32TargetInfo() {
  static Next32 Target;
  return &Target;
}
