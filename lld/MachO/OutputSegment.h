//===- OutputSegment.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_OUTPUT_SEGMENT_H
#define LLD_MACHO_OUTPUT_SEGMENT_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/MapVector.h"

namespace lld {
namespace macho {

class InputSection;

class OutputSegment {
public:
  StringRef name;
  uint32_t perms;
  llvm::MapVector<StringRef, std::vector<InputSection *>> sections;
};

extern std::vector<OutputSegment *> outputSegments;

OutputSegment *getOrCreateOutputSegment(StringRef name, uint32_t perms);

} // namespace macho
} // namespace lld

#endif
