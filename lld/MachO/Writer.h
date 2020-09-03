//===- Writer.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_WRITER_H
#define LLD_MACHO_WRITER_H

#include <cstdint>

namespace lld {
namespace macho {

class LoadCommand {
public:
  virtual ~LoadCommand() = default;
  virtual uint32_t getSize() const = 0;
  virtual void writeTo(uint8_t *buf) const = 0;
};

void writeResult();

void createSyntheticSections();

} // namespace macho
} // namespace lld

#endif
