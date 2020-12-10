//===- OutputSegment.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_OUTPUT_SEGMENT_H
#define LLD_MACHO_OUTPUT_SEGMENT_H

#include "OutputSection.h"
#include "lld/Common/LLVM.h"

namespace lld {
namespace macho {

namespace segment_names {

constexpr const char pageZero[] = "__PAGEZERO";
constexpr const char text[] = "__TEXT";
constexpr const char data[] = "__DATA";
constexpr const char linkEdit[] = "__LINKEDIT";
constexpr const char dataConst[] = "__DATA_CONST";
constexpr const char ld[] = "__LD"; // output only with -r
constexpr const char dwarf[] = "__DWARF";

} // namespace segment_names

class OutputSection;
class InputSection;

class OutputSegment {
public:
  const OutputSection *firstSection() const { return sections.front(); }
  const OutputSection *lastSection() const { return sections.back(); }

  void addOutputSection(OutputSection *os);
  void sortOutputSections(
      llvm::function_ref<bool(OutputSection *, OutputSection *)> comparator) {
    llvm::stable_sort(sections, comparator);
  }

  const std::vector<OutputSection *> &getSections() const { return sections; }
  size_t numNonHiddenSections() const;

  uint64_t fileOff = 0;
  StringRef name;
  uint32_t maxProt = 0;
  uint32_t initProt = 0;
  uint8_t index;

private:
  std::vector<OutputSection *> sections;
};

extern std::vector<OutputSegment *> outputSegments;

OutputSegment *getOrCreateOutputSegment(StringRef name);

} // namespace macho
} // namespace lld

#endif
