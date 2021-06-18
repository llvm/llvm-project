//===- UnwindInfoSection.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_UNWIND_INFO_H
#define LLD_MACHO_UNWIND_INFO_H

#include "ConcatOutputSection.h"
#include "SyntheticSections.h"

#include "mach-o/compact_unwind_encoding.h"

namespace lld {
namespace macho {

template <class Ptr> struct CompactUnwindEntry {
  Ptr functionAddress;
  uint32_t functionLength;
  compact_unwind_encoding_t encoding;
  Ptr personality;
  Ptr lsda;
};

class UnwindInfoSection : public SyntheticSection {
public:
  bool isNeeded() const override { return compactUnwindSection != nullptr; }
  uint64_t getSize() const override { return unwindInfoSize; }
  virtual void prepareRelocations(ConcatInputSection *) = 0;

  void setCompactUnwindSection(ConcatOutputSection *cuSection) {
    compactUnwindSection = cuSection;
  }

protected:
  UnwindInfoSection()
      : SyntheticSection(segment_names::text, section_names::unwindInfo) {
    align = 4;
  }

  ConcatOutputSection *compactUnwindSection = nullptr;
  uint64_t unwindInfoSize = 0;
};

UnwindInfoSection *makeUnwindInfoSection();

} // namespace macho
} // namespace lld

#endif
