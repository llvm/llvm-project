//===- ARM.cpp ------------------------------------------------------------===//
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

struct ARM : TargetInfo {
  ARM(uint32_t cpuSubtype);

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

const RelocAttrs &ARM::getRelocAttrs(uint8_t type) const {
  static const std::array<RelocAttrs, 10> relocAttrsArray{{
#define B(x) RelocAttrBits::x
      {"VANILLA", /* FIXME populate this */ B(_0)},
      {"PAIR", /* FIXME populate this */ B(_0)},
      {"SECTDIFF", /* FIXME populate this */ B(_0)},
      {"LOCAL_SECTDIFF", /* FIXME populate this */ B(_0)},
      {"PB_LA_PTR", /* FIXME populate this */ B(_0)},
      {"BR24", /* FIXME populate this */ B(_0)},
      {"BR22", /* FIXME populate this */ B(_0)},
      {"32BIT_BRANCH", /* FIXME populate this */ B(_0)},
      {"HALF", /* FIXME populate this */ B(_0)},
      {"HALF_SECTDIFF", /* FIXME populate this */ B(_0)},
#undef B
  }};
  assert(type < relocAttrsArray.size() && "invalid relocation type");
  if (type >= relocAttrsArray.size())
    return invalidRelocAttrs;
  return relocAttrsArray[type];
}

int64_t ARM::getEmbeddedAddend(MemoryBufferRef mb, uint64_t offset,
                               relocation_info rel) const {
  fatal("TODO: implement this");
}

void ARM::relocateOne(uint8_t *loc, const Reloc &r, uint64_t value,
                      uint64_t relocVA) const {
  fatal("TODO: implement this");
}

void ARM::writeStub(uint8_t *buf, const Symbol &sym) const {
  fatal("TODO: implement this");
}

void ARM::writeStubHelperHeader(uint8_t *buf) const {
  fatal("TODO: implement this");
}

void ARM::writeStubHelperEntry(uint8_t *buf, const DylibSymbol &sym,
                               uint64_t entryAddr) const {
  fatal("TODO: implement this");
}

void ARM::relaxGotLoad(uint8_t *loc, uint8_t type) const {
  fatal("TODO: implement this");
}

ARM::ARM(uint32_t cpuSubtype) : TargetInfo(ILP32()) {
  cpuType = CPU_TYPE_ARM;
  this->cpuSubtype = cpuSubtype;

  stubSize = 0 /* FIXME */;
  stubHelperHeaderSize = 0 /* FIXME */;
  stubHelperEntrySize = 0 /* FIXME */;
}

TargetInfo *macho::createARMTargetInfo(uint32_t cpuSubtype) {
  static ARM t(cpuSubtype);
  return &t;
}
