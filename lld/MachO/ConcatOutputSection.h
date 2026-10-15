//===- ConcatOutputSection.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_CONCAT_OUTPUT_SECTION_H
#define LLD_MACHO_CONCAT_OUTPUT_SECTION_H

#include "InputSection.h"
#include "OutputSection.h"
#include "lld/Common/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

namespace lld::macho {
class Defined;
class InputSectionBoundary;

// Linking multiple files will inevitably mean resolving sections in different
// files that are labeled with the same segment and section name. This class
// contains all such sections and writes the data from each section sequentially
// in the final binary.
class ConcatOutputSection : public OutputSection {
public:
  explicit ConcatOutputSection(StringRef name,
                               OutputSection::Kind kind = ConcatKind)
      : OutputSection(kind, name) {}

  const ConcatInputSection *firstSection() const { return inputs.front(); }
  const ConcatInputSection *lastSection() const { return inputs.back(); }
  bool isNeeded() const override { return !inputs.empty(); }

  // These accessors will only be valid after finalizing the section
  uint64_t getSize() const override { return size; }
  uint64_t getFileSize() const override { return fileSize; }

  // Assign input offsets and finalize planned branch extenders.
  void finalizeContents();

  // The planner keeps this range alive until finalizeContents() consumes it.
  void setBoundaries(ArrayRef<InputSectionBoundary> boundaries,
                     uint64_t expectedSize);
  ArrayRef<ConcatInputSection *> getExtenders() const { return extenders; }

  void addInput(ConcatInputSection *input);
  void writeTo(uint8_t *buf) const override;

  static bool classof(const OutputSection *sec) {
    return sec->kind() == ConcatKind;
  }

  static ConcatOutputSection *getOrCreateForInput(const InputSection *);

  std::vector<ConcatInputSection *> inputs;

private:
  size_t size = 0;
  uint64_t fileSize = 0;
  ArrayRef<InputSectionBoundary> boundaries;
  std::optional<uint64_t> plannedSize;
  std::vector<ConcatInputSection *> extenders;
  void finalizeOne(ConcatInputSection *);
  void finalizeFlags(InputSection *input);
};

NamePair maybeRenameSection(NamePair key);

// Output sections are added to output segments in iteration order
// of ConcatOutputSection, so must have deterministic iteration order.
extern llvm::MapVector<NamePair, ConcatOutputSection *> concatOutputSections;

} // namespace lld::macho

#endif
