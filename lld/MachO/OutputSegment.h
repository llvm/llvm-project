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
#include "llvm/ADT/MapVector.h"

namespace lld {
namespace macho {

namespace segment_names {

constexpr const char *pageZero = "__PAGEZERO";
constexpr const char *text = "__TEXT";
constexpr const char *data = "__DATA";
constexpr const char *linkEdit = "__LINKEDIT";
constexpr const char *dataConst = "__DATA_CONST";

} // namespace segment_names

class OutputSection;
class OutputSegmentComparator;
class InputSection;

class OutputSegment {
public:
  using SectionMap = typename llvm::MapVector<StringRef, OutputSection *>;
  using SectionMapEntry = typename std::pair<StringRef, OutputSection *>;

  const OutputSection *firstSection() const { return sections.front().second; }
  const OutputSection *lastSection() const { return sections.back().second; }

  bool isNeeded() const {
    if (name == segment_names::linkEdit)
      return true;
    for (const SectionMapEntry &i : sections) {
      OutputSection *os = i.second;
      if (os->isNeeded())
        return true;
    }
    return false;
  }

  OutputSection *getOrCreateOutputSection(StringRef name);
  void addOutputSection(OutputSection *os);
  void sortOutputSections(OutputSegmentComparator *comparator);
  void removeUnneededSections();

  const SectionMap &getSections() const { return sections; }
  size_t numNonHiddenSections() const;

  uint64_t fileOff = 0;
  StringRef name;
  uint32_t maxProt = 0;
  uint32_t initProt = 0;
  uint8_t index;

private:
  SectionMap sections;
};

class OutputSegmentComparator {
public:
  OutputSegmentComparator();

  OutputSectionComparator *sectionComparator(const OutputSegment *os) {
    auto it = orderMap.find(os->name);
    if (it == orderMap.end()) {
      return defaultPositionComparator;
    }
    return &it->second;
  }

  bool operator()(const OutputSegment *a, const OutputSegment *b) {
    return *sectionComparator(a) < *sectionComparator(b);
  }

private:
  const StringRef defaultPosition = StringRef();
  llvm::DenseMap<StringRef, OutputSectionComparator> orderMap;
  OutputSectionComparator *defaultPositionComparator;
};

extern std::vector<OutputSegment *> outputSegments;

OutputSegment *getOutputSegment(StringRef name);
OutputSegment *getOrCreateOutputSegment(StringRef name);

} // namespace macho
} // namespace lld

#endif
