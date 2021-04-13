//===- UnwindInfoSection.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_UNWIND_INFO_H
#define LLD_MACHO_UNWIND_INFO_H

#include "MergedOutputSection.h"
#include "SyntheticSections.h"

#include "mach-o/compact_unwind_encoding.h"
#include "llvm/ADT/DenseMap.h"

#include <vector>

namespace lld {
namespace macho {

class UnwindInfoSection : public SyntheticSection {
public:
  bool isNeeded() const override { return compactUnwindSection != nullptr; }
  uint64_t getSize() const override { return unwindInfoSize; }
  virtual void prepareRelocations(InputSection *) = 0;

  void setCompactUnwindSection(MergedOutputSection *cuSection) {
    compactUnwindSection = cuSection;
  }

protected:
  UnwindInfoSection()
      : SyntheticSection(segment_names::text, section_names::unwindInfo) {
    align = 4;
  }

  MergedOutputSection *compactUnwindSection = nullptr;
  uint64_t unwindInfoSize = 0;
};

UnwindInfoSection *makeUnwindInfoSection();
void prepareCompactUnwind(InputSection *isec);

} // namespace macho
} // namespace lld

#endif
