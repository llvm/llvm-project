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
  bool isNeeded() const override {
    return !compactUnwindSection->inputs.empty() && !allEntriesAreOmitted;
  }
  uint64_t getSize() const override { return unwindInfoSize; }
  virtual void addInput(ConcatInputSection *) = 0;
  std::vector<ConcatInputSection *> getInputs() {
    return compactUnwindSection->inputs;
  }
  void prepareRelocations();

protected:
  UnwindInfoSection();
  virtual void prepareRelocations(ConcatInputSection *) = 0;

  ConcatOutputSection *compactUnwindSection;
  uint64_t unwindInfoSize = 0;
  bool allEntriesAreOmitted = true;
};

UnwindInfoSection *makeUnwindInfoSection();

} // namespace macho
} // namespace lld

#endif
