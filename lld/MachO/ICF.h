//===- ICF.h ----------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_ICF_H
#define LLD_MACHO_ICF_H

#include "lld/Common/LLVM.h"
#include <vector>

namespace lld {
namespace macho {

class ConcatInputSection;

class ICF {
public:
  ICF(std::vector<ConcatInputSection *> &inputs);

  void run();
  void segregate(size_t begin, size_t end,
                 std::function<bool(const ConcatInputSection *,
                                    const ConcatInputSection *)>
                     equals);
  size_t findBoundary(size_t begin, size_t end);
  void forEachClassRange(size_t begin, size_t end,
                         std::function<void(size_t, size_t)> func);
  void forEachClass(std::function<void(size_t, size_t)> func);

  // ICF needs a copy of the inputs vector because its equivalence-class
  // segregation algorithm destroys the proper sequence.
  std::vector<ConcatInputSection *> icfInputs;
};

} // namespace macho
} // namespace lld

#endif
