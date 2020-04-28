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

namespace segment_names {

constexpr const char *text = "__TEXT";
constexpr const char *pageZero = "__PAGEZERO";
constexpr const char *linkEdit = "__LINKEDIT";

} // namespace segment_names

class InputSection;

class OutputSegment {
public:
  InputSection *firstSection() const { return sections.front().second.at(0); }

  InputSection *lastSection() const { return sections.back().second.back(); }

  bool isNeeded() const {
    return !sections.empty() || name == segment_names::linkEdit;
  }

  void addSection(InputSection *);

  const llvm::MapVector<StringRef, std::vector<InputSection *>> &
  getSections() const {
    return sections;
  }

  uint64_t fileOff = 0;
  StringRef name;
  uint32_t numNonHiddenSections = 0;
  uint32_t maxProt = 0;
  uint32_t initProt = 0;
  uint8_t index;

private:
  llvm::MapVector<StringRef, std::vector<InputSection *>> sections;
};

extern std::vector<OutputSegment *> outputSegments;

OutputSegment *getOutputSegment(StringRef name);
OutputSegment *getOrCreateOutputSegment(StringRef name);

} // namespace macho
} // namespace lld

#endif
