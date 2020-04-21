//===- X86_64.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
  uint64_t getImplicitAddend(const uint8_t *loc, uint8_t type) const override;
  void relocateOne(uint8_t *loc, uint8_t type, uint64_t val) const override;
};

X86_64::X86_64() {
  cpuType = CPU_TYPE_X86_64;
  cpuSubtype = CPU_SUBTYPE_X86_64_ALL;
}

uint64_t X86_64::getImplicitAddend(const uint8_t *loc, uint8_t type) const {
  switch (type) {
  case X86_64_RELOC_SIGNED:
  case X86_64_RELOC_GOT_LOAD:
    return read32le(loc);
  default:
    error("TODO: Unhandled relocation type " + std::to_string(type));
    return 0;
  }
}

void X86_64::relocateOne(uint8_t *loc, uint8_t type, uint64_t val) const {
  switch (type) {
  case X86_64_RELOC_SIGNED:
  case X86_64_RELOC_GOT_LOAD:
    // These types are only used for pc-relative relocations, so offset by 4
    // since the RIP has advanced by 4 at this point.
    write32le(loc, val - 4);
    break;
  default:
    llvm_unreachable(
        "getImplicitAddend should have flagged all unhandled relocation types");
  }
}

} // namespace

TargetInfo *macho::createX86_64TargetInfo() {
  static X86_64 t;
  return &t;
}
