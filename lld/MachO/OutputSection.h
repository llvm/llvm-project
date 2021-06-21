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

#include <limits>

namespace lld {
namespace macho {

class InputSection;
class OutputSegment;

// The default order value for OutputSections that are not constructed from
// InputSections (i.e. SyntheticSections). We make it less than INT_MAX in order
// not to conflict with the ordering of zerofill sections, which must always be
// placed at the end of their segment.
constexpr int UnspecifiedInputOrder = std::numeric_limits<int>::max() - 1024;

// Output sections represent the finalized sections present within the final
// linked executable. They can represent special sections (like the symbol
// table), or represent coalesced sections from the various inputs given to the
// linker with the same segment / section name.
class OutputSection {
public:
  enum Kind {
    ConcatKind,
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

  virtual void finalize() {
    // TODO investigate refactoring synthetic section finalization logic into
    // overrides of this function.
  }

  virtual void writeTo(uint8_t *buf) const = 0;

  StringRef name;
  OutputSegment *parent = nullptr;
  // For output sections that don't have explicit ordering requirements, their
  // output order should be based on the order of the input sections they
  // contain.
  int inputOrder = UnspecifiedInputOrder;

  uint32_t index = 0;
  uint64_t addr = 0;
  uint64_t fileOff = 0;
  uint32_t align = 1;
  uint32_t flags = 0;
  uint32_t reserved1 = 0;
  uint32_t reserved2 = 0;

private:
  Kind sectionKind;
};

} // namespace macho
} // namespace lld

#endif
