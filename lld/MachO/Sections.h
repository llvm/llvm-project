//===- Sections.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_SECTIONS_H
#define LLD_MACHO_SECTIONS_H

#include "lld/Common/LLVM.h"

namespace lld::macho::sections {
bool isCodeSection(StringRef name, StringRef segName, uint32_t flags);
} // namespace lld::macho::sections

#endif // #ifndef LLD_MACHO_SECTIONS_H
