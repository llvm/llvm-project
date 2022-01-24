//===- SectionPriorities.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_SECTION_PRIORITIES_H
#define LLD_MACHO_SECTION_PRIORITIES_H

#include "InputSection.h"
#include "llvm/ADT/DenseMap.h"

namespace lld {
namespace macho {

llvm::DenseMap<const InputSection *, size_t> computeCallGraphProfileOrder();
} // namespace macho
} // namespace lld

#endif
