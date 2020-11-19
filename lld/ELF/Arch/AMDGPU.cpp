//===- AMDGPU.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Symbols.h"
#include "Target.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {
class AMDGPU final : public TargetInfo {
private:
  uint32_t calcEFlagsV3() const;
  uint32_t calcEFlagsV4() const;

public:
  AMDGPU();
  uint32_t calcEFlags() const override;
  void relocate(uint8_t *loc, const Relocation &rel,
                uint64_t val) const override;
  RelExpr getRelExpr(RelType type, const Symbol &s,
                     const uint8_t *loc) const override;
  RelType getDynRel(RelType type) const override;
};
} // namespace

AMDGPU::AMDGPU() {
  relativeRel = R_AMDGPU_RELATIVE64;
  gotRel = R_AMDGPU_ABS64;
  noneRel = R_AMDGPU_NONE;
  symbolicRel = R_AMDGPU_ABS64;
}

static uint8_t getAbiVersion(InputFile *file) {
  return cast<ObjFile<ELF64LE>>(file)->getObj().getHeader().e_ident[EI_ABIVERSION];
}

static uint32_t getEFlags(InputFile *file) {
  return cast<ObjFile<ELF64LE>>(file)->getObj().getHeader().e_flags;
}

static uint32_t getMach(InputFile *file) {
  return getEFlags(file) & EF_AMDGPU_MACH;
}

static uint32_t getXnackV4(InputFile *file) {
  return getEFlags(file) & EF_AMDGPU_FEATURE_XNACK_V4;
}

static uint32_t getSramEccV4(InputFile *file) {
  return getEFlags(file) & EF_AMDGPU_FEATURE_SRAMECC_V4;
}

uint32_t AMDGPU::calcEFlagsV3() const {
  uint32_t ret = getEFlags(objectFiles[0]);

  // Verify that all input files have the same e_flags.
  for (InputFile *f : makeArrayRef(objectFiles).slice(1)) {
    if (ret == getEFlags(f))
      continue;
    error("incompatible e_flags: " + toString(f));
    return 0;
  }
  return ret;
}

uint32_t AMDGPU::calcEFlagsV4() const {
  uint32_t retMach = getMach(objectFiles[0]);
  uint32_t retXnack = getXnackV4(objectFiles[0]);
  uint32_t retSramEcc = getSramEccV4(objectFiles[0]);

  // Verify that all input files have compatible e_flags (same mach, all
  // features in the same category are either ANY, ANY and ON, or ANY and OFF).
  for (InputFile *f : makeArrayRef(objectFiles).slice(1)) {
    if (retMach != getMach(f)) {
      error("incompatible mach: " + toString(f));
      return 0;
    }

    if ((retXnack == EF_AMDGPU_FEATURE_XNACK_UNSUPPORTED_V4) ||
        (retXnack != EF_AMDGPU_FEATURE_XNACK_ANY_V4 &&
            getXnackV4(f) != EF_AMDGPU_FEATURE_XNACK_ANY_V4)) {
      if (retXnack != getXnackV4(f)) {
        error("incompatible xnack: " + toString(f));
        return 0;
      }
    } else {
      if (retXnack == EF_AMDGPU_FEATURE_XNACK_ANY_V4) {
        retXnack = getXnackV4(f);
      }
    }

    if ((retSramEcc == EF_AMDGPU_FEATURE_SRAMECC_UNSUPPORTED_V4) ||
        (retSramEcc != EF_AMDGPU_FEATURE_SRAMECC_ANY_V4 &&
            getSramEccV4(f) != EF_AMDGPU_FEATURE_SRAMECC_ANY_V4)) {
      if (retSramEcc != getSramEccV4(f)) {
        error("incompatible sramecc: " + toString(f));
        return 0;
      }
    } else {
      if (retSramEcc == EF_AMDGPU_FEATURE_SRAMECC_ANY_V4) {
        retSramEcc = getSramEccV4(f);
      }
    }
  }

  return retMach | retXnack | retSramEcc;
}

uint32_t AMDGPU::calcEFlags() const {
  assert(!objectFiles.empty());
  switch (getAbiVersion(objectFiles[0])) {
  case ELFABIVERSION_AMDGPU_HSA_V2:
  case ELFABIVERSION_AMDGPU_HSA_V3:
    return calcEFlagsV3();
  case ELFABIVERSION_AMDGPU_HSA_V4:
    return calcEFlagsV4();
  default:
    llvm_unreachable("Unknown ABI Version");
  }
}

void AMDGPU::relocate(uint8_t *loc, const Relocation &rel, uint64_t val) const {
  switch (rel.type) {
  case R_AMDGPU_ABS32:
  case R_AMDGPU_GOTPCREL:
  case R_AMDGPU_GOTPCREL32_LO:
  case R_AMDGPU_REL32:
  case R_AMDGPU_REL32_LO:
    write32le(loc, val);
    break;
  case R_AMDGPU_ABS64:
  case R_AMDGPU_REL64:
    write64le(loc, val);
    break;
  case R_AMDGPU_GOTPCREL32_HI:
  case R_AMDGPU_REL32_HI:
    write32le(loc, val >> 32);
    break;
  default:
    llvm_unreachable("unknown relocation");
  }
}

RelExpr AMDGPU::getRelExpr(RelType type, const Symbol &s,
                           const uint8_t *loc) const {
  switch (type) {
  case R_AMDGPU_ABS32:
  case R_AMDGPU_ABS64:
    return R_ABS;
  case R_AMDGPU_REL32:
  case R_AMDGPU_REL32_LO:
  case R_AMDGPU_REL32_HI:
  case R_AMDGPU_REL64:
    return R_PC;
  case R_AMDGPU_GOTPCREL:
  case R_AMDGPU_GOTPCREL32_LO:
  case R_AMDGPU_GOTPCREL32_HI:
    return R_GOT_PC;
  default:
    error(getErrorLocation(loc) + "unknown relocation (" + Twine(type) +
          ") against symbol " + toString(s));
    return R_NONE;
  }
}

RelType AMDGPU::getDynRel(RelType type) const {
  if (type == R_AMDGPU_ABS64)
    return type;
  return R_AMDGPU_NONE;
}

TargetInfo *elf::getAMDGPUTargetInfo() {
  static AMDGPU target;
  return &target;
}
