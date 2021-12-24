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
#include "llvm/ADT/MapVector.h"

#include "mach-o/compact_unwind_encoding.h"

namespace lld {
namespace macho {

class UnwindInfoSection : public SyntheticSection {
public:
  // If all functions are free of unwind info, we can omit the unwind info
  // section entirely.
  bool isNeeded() const override { return !allEntriesAreOmitted; }
  uint64_t getSize() const override { return unwindInfoSize; }
  void addSymbol(const Defined *);
  void prepareRelocations();

protected:
  UnwindInfoSection();
  virtual void prepareRelocations(ConcatInputSection *) = 0;

  llvm::MapVector<std::pair<const InputSection *, uint64_t /*Defined::value*/>,
                  const Defined *>
      symbols;
  std::vector<decltype(symbols)::value_type> symbolsVec;
  uint64_t unwindInfoSize = 0;
  bool allEntriesAreOmitted = true;
};

UnwindInfoSection *makeUnwindInfoSection();

} // namespace macho
} // namespace lld

#endif
