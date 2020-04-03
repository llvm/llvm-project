//===- Target.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_TARGET_H
#define LLD_MACHO_TARGET_H

#include <cstdint>

namespace lld {
namespace macho {

enum {
  PageSize = 4096,
  ImageBase = 4096,
  MaxAlignmentPowerOf2 = 32,
};

class TargetInfo {
public:
  virtual ~TargetInfo() = default;
  virtual uint64_t getImplicitAddend(const uint8_t *loc,
                                     uint8_t type) const = 0;
  virtual void relocateOne(uint8_t *loc, uint8_t type, uint64_t val) const = 0;

  uint32_t cpuType;
  uint32_t cpuSubtype;
};

TargetInfo *createX86_64TargetInfo();

extern TargetInfo *target;

} // namespace macho
} // namespace lld

#endif
