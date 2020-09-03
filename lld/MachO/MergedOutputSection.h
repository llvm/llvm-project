//===- OutputSection.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_MERGED_OUTPUT_SECTION_H
#define LLD_MACHO_MERGED_OUTPUT_SECTION_H

#include "InputSection.h"
#include "OutputSection.h"
#include "lld/Common/LLVM.h"
#include "llvm/ADT/MapVector.h"

namespace lld {
namespace macho {

// Linking multiple files will inevitably mean resolving sections in different
// files that are labeled with the same segment and section name. This class
// contains all such sections and writes the data from each section sequentially
// in the final binary.
class MergedOutputSection : public OutputSection {
public:
  MergedOutputSection(StringRef name) : OutputSection(MergedKind, name) {}

  const InputSection *firstSection() const { return inputs.front(); }
  const InputSection *lastSection() const { return inputs.back(); }

  // These accessors will only be valid after finalizing the section
  uint64_t getSize() const override { return size; }
  uint64_t getFileSize() const override { return fileSize; }

  void mergeInput(InputSection *input);
  void finalize() override;

  void writeTo(uint8_t *buf) const override;

  std::vector<InputSection *> inputs;

  static bool classof(const OutputSection *sec) {
    return sec->kind() == MergedKind;
  }

private:
  void mergeFlags(uint32_t inputFlags);

  size_t size = 0;
  uint64_t fileSize = 0;
};

} // namespace macho
} // namespace lld

#endif
