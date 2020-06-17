//===- OutputSection.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_OUTPUT_SECTION_H
#define LLD_MACHO_OUTPUT_SECTION_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/DenseMap.h"

namespace lld {
namespace macho {

class InputSection;
class OutputSegment;

// Output sections represent the finalized sections present within the final
// linked executable. They can represent special sections (like the symbol
// table), or represent coalesced sections from the various inputs given to the
// linker with the same segment / section name.
class OutputSection {
public:
  enum Kind {
    MergedKind,
    SyntheticKind,
  };

  OutputSection(Kind kind, StringRef name) : name(name), sectionKind(kind) {}
  virtual ~OutputSection() = default;
  Kind kind() const { return sectionKind; }

  // These accessors will only be valid after finalizing the section.
  uint64_t getSegmentOffset() const;

  // How much space the section occupies in the address space.
  virtual uint64_t getSize() const = 0;
  // How much space the section occupies in the file. Most sections are copied
  // as-is so their file size is the same as their address space size.
  virtual uint64_t getFileSize() const { return getSize(); }

  // Hidden sections omit header content, but body content may still be present.
  virtual bool isHidden() const { return false; }
  // Unneeded sections are omitted entirely (header and body).
  virtual bool isNeeded() const { return true; }

  // Some sections may allow coalescing other raw input sections.
  virtual void mergeInput(InputSection *input);

  // Specifically finalizes addresses and section size, not content.
  virtual void finalize() {
    // TODO investigate refactoring synthetic section finalization logic into
    // overrides of this function.
  }

  virtual void writeTo(uint8_t *buf) const = 0;

  StringRef name;
  OutputSegment *parent = nullptr;

  uint32_t index = 0;
  uint64_t addr = 0;
  uint64_t fileOff = 0;
  uint32_t align = 1;
  uint32_t flags = 0;

private:
  Kind sectionKind;
};

class OutputSectionComparator {
public:
  OutputSectionComparator(uint32_t segmentOrder,
                          const std::vector<StringRef> &sectOrdering)
      : segmentOrder(segmentOrder) {
    for (uint32_t j = 0, m = sectOrdering.size(); j < m; ++j)
      sectionOrdering[sectOrdering[j]] = j;
  }

  uint32_t sectionOrder(StringRef secname) {
    auto sectIt = sectionOrdering.find(secname);
    if (sectIt != sectionOrdering.end())
      return sectIt->second;
    return sectionOrdering.size();
  }

  // Sort sections within a common segment, which stores them in
  // a MapVector of section name -> section
  bool operator()(const std::pair<StringRef, OutputSection *> &a,
                  const std::pair<StringRef, OutputSection *> &b) {
    return sectionOrder(a.first) < sectionOrder(b.first);
  }

  bool operator<(const OutputSectionComparator &b) {
    return segmentOrder < b.segmentOrder;
  }

private:
  uint32_t segmentOrder;
  llvm::DenseMap<StringRef, uint32_t> sectionOrdering;
};

} // namespace macho
} // namespace lld

#endif
